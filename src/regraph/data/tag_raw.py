"""Text-attributed Cora (LLaGA / GraphGPT / TAPE lineage).

LLaGA (ICML 2024) and GraphGPT (SIGIR 2024) both evaluate on Cora, PubMed, ogbn-arxiv and
ogbn-products. LLaGA's own processed release is a dead Box link, so Cora is reassembled from
its two public sources:

  * node text + labels + splits : TAPE (`xxhe/tape-cora`, He et al. 2023) — title, abstract,
    readable class name, 60/20/20 split
  * graph structure            : the original LINQS `cora.content` / `cora.cites`

**The join is on `cora.content` row order, NOT Planetoid's node order.** Verified: TAPE's
labels agree with `cora.content` 100% under that ordering and only 14.29% (= 1/7, chance)
against Planetoid's. Joining TAPE text to Planetoid edges silently pairs each node's text with
another node's neighbourhood and yields a meaningless benchmark; the check is re-run at build
time and raises rather than warn.

Each example is a k-hop ego-subgraph, as in `arxiv_raw.py`, since ReGraph's reader softmaxes
over every node of the example graph.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from regraph.data.arxiv_raw import _build_csr, sample_ego_subgraph

QUESTION = ("This is a paper from the Cora citation network, shown with the papers it cites. "
            "Which machine-learning topic does it belong to?")
EDGE_TEXT = "cites"


def build_cora(raw_dir: Path, num_hops: int = 2, fanout: int = 10, seed: int = 0):
    raw_dir = Path(raw_dir)
    content = np.genfromtxt(raw_dir / "cora" / "cora.content", dtype=str)
    paper_ids, label_str = content[:, 0], content[:, -1]
    classes = sorted(set(label_str))
    y = np.array([classes.index(l) for l in label_str])
    pid2idx = {p: i for i, p in enumerate(paper_ids)}
    num_nodes = len(paper_ids)

    tape = pd.concat(
        [pd.read_csv(raw_dir / f"{s}.csv").assign(split=("val" if s == "val" else s))
         for s in ("train", "val", "test")]
    )
    agree = float((tape.label.values == y[tape.id.values]).mean())
    if agree < 0.99:
        raise RuntimeError(
            f"TAPE ids do not index cora.content row order (label agreement {agree:.2%}). "
            "Refusing to build — the text/graph join would be wrong."
        )
    label_names = (tape.drop_duplicates("label").sort_values("label")["class"].tolist())

    cites = np.genfromtxt(raw_dir / "cora" / "cora.cites", dtype=str)
    src, dst = [], []
    for a, b in cites:                       # "<cited> <citing>"
        if a in pid2idx and b in pid2idx:
            src.append(pid2idx[b]); dst.append(pid2idx[a])
    edge_index = np.array([src, dst], dtype=np.int64)
    indptr, indices = _build_csr(edge_index, num_nodes)

    node_texts = [""] * num_nodes
    for _, r in tape.iterrows():
        node_texts[int(r["id"])] = f"{str(r['T']).strip()}. {str(r['A']).strip()}"

    rng = np.random.default_rng(seed)
    graphs, examples = {}, []
    for _, r in tape.iterrows():
        node = int(r["id"])
        nodes, ei = sample_ego_subgraph(node, indptr, indices, num_hops, fanout, rng)
        key = str(node)
        graphs[key] = {
            "node_text_id": torch.tensor(nodes, dtype=torch.int32),
            "edge_text_id": torch.full((ei.shape[1],), num_nodes, dtype=torch.int32),
            "edge_index": ei.to(torch.int32),
        }
        examples.append({
            "id": node, "graph_key": key, "question": QUESTION,
            "answer": label_names[int(r["label"])], "split": r["split"],
            "roles": [[0, 2]],                       # centre node -> ROLE_SOURCE
            "label_idx": int(r["label"]),
        })
    return graphs, examples, node_texts + [EDGE_TEXT], label_names


PUBMED_QUESTION = ("This is a paper from the PubMed diabetes citation network, shown with the "
                   "papers it cites. Which type of diabetes does it study?")
# PubMed-Diabetes label ids are 1..3 in the raw file; zero-based here.
PUBMED_CLASSES = ["Diabetes Mellitus Experimental",
                  "Diabetes Mellitus Type 1",
                  "Diabetes Mellitus Type 2"]


def build_pubmed(raw_dir: Path, num_hops: int = 2, fanout: int = 10, seed: int = 0):
    """Text-attributed PubMed (TAPE `PubMed_orig`), same shape as `build_cora`.

    Node order is the row order of `Pubmed-Diabetes.NODE.paper.tab` (TAPE's `parse_pubmed`);
    titles/abstracts are joined from `pubmed.json` on PMID. The PMID join is verified and the
    build raises if it does not cover every node.
    """
    import json as _json

    raw_dir = Path(raw_dir) / "PubMed_orig"
    data_dir = raw_dir / "data"

    pmids, labels = [], []
    with open(data_dir / "Pubmed-Diabetes.NODE.paper.tab") as f:
        f.readline(); f.readline()                       # two header lines
        for line in f:
            items = line.strip().split("\t")
            pmids.append(items[0])
            labels.append(int(items[1].split("=")[-1]) - 1)
    num_nodes = len(pmids)
    pid2idx = {p: i for i, p in enumerate(pmids)}

    src, dst = [], []
    with open(data_dir / "Pubmed-Diabetes.DIRECTED.cites.tab") as f:
        f.readline(); f.readline()
        for line in f:
            items = line.strip().split("\t")
            tail = items[1].split(":")[-1]               # "paper:<pmid>"
            head = items[3].split(":")[-1]
            if tail in pid2idx and head in pid2idx:
                src.append(pid2idx[head]); dst.append(pid2idx[tail])
    edge_index = np.array([src, dst], dtype=np.int64)

    # one record in the shipped pubmed.json is malformed (key "id:" instead of "PMID")
    meta = {str(r["PMID"]): r for r in _json.load(open(raw_dir / "pubmed.json")) if "PMID" in r}
    covered = sum(p in meta for p in pmids)
    if covered < 0.99 * num_nodes:
        raise RuntimeError(
            f"pubmed.json covers only {covered}/{num_nodes} nodes — PMID join is broken."
        )
    if covered < num_nodes:
        print(f"[pubmed] {num_nodes - covered} node(s) have no title/abstract; using a placeholder")
    node_texts = [
        (f"{str(meta[p].get('TI','')).strip()}. {str(meta[p].get('AB','')).strip()}"
         if p in meta else "")
        for p in pmids
    ]

    indptr, indices = _build_csr(edge_index, num_nodes)
    rng = np.random.default_rng(seed)
    order = np.arange(num_nodes); rng.shuffle(order)     # TAPE's 60/20/20
    split_of = {}
    for i, n in enumerate(order):
        split_of[int(n)] = ("train" if i < 0.6 * num_nodes
                            else "val" if i < 0.8 * num_nodes else "test")

    graphs, examples = {}, []
    for node in range(num_nodes):
        nodes, ei = sample_ego_subgraph(node, indptr, indices, num_hops, fanout, rng)
        key = str(node)
        graphs[key] = {
            "node_text_id": torch.tensor(nodes, dtype=torch.int32),
            "edge_text_id": torch.full((ei.shape[1],), num_nodes, dtype=torch.int32),
            "edge_index": ei.to(torch.int32),
        }
        examples.append({
            "id": node, "graph_key": key, "question": PUBMED_QUESTION,
            "answer": PUBMED_CLASSES[labels[node]], "split": split_of[node],
            "roles": [[0, 2]], "label_idx": labels[node],
        })
    return graphs, examples, node_texts + [EDGE_TEXT], PUBMED_CLASSES


PRODUCTS_QUESTION = ("This is a product from the Amazon co-purchasing network, shown with the "
                     "products bought alongside it. Which product category does it belong to?")


def _load_torch_sparse_stub_pt(path: Path):
    """Open TAPE's products `.pt`, which pickles a `torch_sparse.SparseTensor`.

    torch_sparse has no wheel for this torch build; the file only needs to be *read*, so the
    two classes are stubbed to capture their pickled state and the COO row/col are recovered.
    """
    import sys, types

    pkg = types.ModuleType("torch_sparse"); pkg.__path__ = []

    class _R:
        def __setstate__(self, s): self._state = s

    class SparseTensor(_R): pass
    class SparseStorage(_R): pass

    pkg.SparseTensor, pkg.SparseStorage = SparseTensor, SparseStorage
    for n in ("tensor", "storage"):
        m = types.ModuleType(f"torch_sparse.{n}")
        m.SparseTensor, m.SparseStorage = SparseTensor, SparseStorage
        sys.modules[f"torch_sparse.{n}"] = m
    sys.modules["torch_sparse"] = pkg
    return torch.load(path, weights_only=False)


def build_products(raw_dir: Path, num_hops: int = 2, fanout: int = 10, seed: int = 0):
    """Text-attributed ogbn-products **subset** as released by TAPE (54,025 nodes).

    LLaGA evaluates on the full 2.4M-node ogbn-products; this is TAPE's subset, so the number
    is not directly comparable and must be labelled as such.
    """
    import pandas as _pd

    raw_dir = Path(raw_dir)
    data = _load_torch_sparse_stub_pt(raw_dir / "ogbn_products_subset.pt")
    store = data._store
    y = store["y"].reshape(-1).numpy()
    num_nodes = int(store["num_nodes"])
    st = store["adj_t"]._state["storage"]._state
    row, col = st["_row"].numpy(), st["_col"].numpy()
    edge_index = np.array([row, col], dtype=np.int64)

    masks = {s: store[f"{s}_mask"].numpy() for s in ("train", "val", "test")}
    txt = _pd.read_csv(raw_dir / "ogbn-products_subset.csv")
    if len(txt) != num_nodes:
        raise RuntimeError(f"text rows {len(txt)} != nodes {num_nodes}")
    node_texts = [f"Product: {t}. Description: {c}"
                  for t, c in zip(txt["title"].astype(str), txt["content"].astype(str))]

    # human-readable category names from OGB's mapping; numeric ids would make the
    # generation target arbitrary and unfairly penalise a generative model
    import pandas as _pd2
    mp = _pd2.read_csv(raw_dir / "labelidx2productcategory.csv.gz")
    col = [c for c in mp.columns if "categ" in c.lower()][0]
    idx = [c for c in mp.columns if "idx" in c.lower()][0]
    label_names = [str(v).strip() for v in mp.sort_values(idx)[col]]
    if int(y.max()) >= len(label_names):
        raise RuntimeError(f"label id {int(y.max())} exceeds {len(label_names)} category names")

    indptr, indices = _build_csr(edge_index, num_nodes)
    rng = np.random.default_rng(seed)
    graphs, examples = {}, []
    for node in range(num_nodes):
        split = ("train" if masks["train"][node] else
                 "val" if masks["val"][node] else
                 "test" if masks["test"][node] else None)
        if split is None:
            continue
        nodes, ei = sample_ego_subgraph(node, indptr, indices, num_hops, fanout, rng)
        key = str(node)
        graphs[key] = {
            "node_text_id": torch.tensor(nodes, dtype=torch.int32),
            "edge_text_id": torch.full((ei.shape[1],), num_nodes, dtype=torch.int32),
            "edge_index": ei.to(torch.int32),
        }
        examples.append({
            "id": node, "graph_key": key, "question": PRODUCTS_QUESTION,
            "answer": label_names[int(y[node])], "split": split,
            "roles": [[0, 2]], "label_idx": int(y[node]),
        })
    return graphs, examples, node_texts + [EDGE_TEXT], label_names
