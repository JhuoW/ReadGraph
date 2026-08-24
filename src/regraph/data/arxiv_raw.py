"""ogbn-arxiv (the objective half of the GraphTranslator benchmark).

GraphTranslator (Zhang et al., WWW 2024) evaluates on ArXiv = ogbn-arxiv: 169,343 nodes,
1,166,243 edges, 40 CS categories, with node text = paper title + abstract. It keeps the
LLM **frozen** and does **not** serialize the graph into the prompt — the same two
constraints ReGraph operates under, which is why this benchmark is a like-for-like
comparison where GraphQA was not.

Two things differ from ReGraph's GraphQA datasets and are deliberate:

1. **Ego-subgraphs.** ogbn-arxiv is one graph of 169k nodes; ReGraph's reader softmaxes over
   every node of the example graph and its localization was measured to collapse past
   n ~ 1371. Each example is therefore the target node's k-hop sampled neighbourhood
   (default 2 hops, 10 samples per hop, <= 111 nodes), matching GraphTranslator's and
   LLaGA's sampling. This must be stated in any reported result.
2. **Untyped edges.** Citations carry no relation text, so every edge gets one shared
   "cites" attribute. The relation-aware encoder therefore buys nothing here.

Node text ids are the global ogbn-arxiv node indices, so the attribute store is built once
over all 169,343 papers and every ego-subgraph indexes into it.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# arXiv CS category code -> human-readable name. Reference data (the official arXiv
# taxonomy), used as the generation target so the task stays open-ended text generation
# with no classification head (`ReGraph.md` §1).
ARXIV_CS_CATEGORIES: dict[str, str] = {
    "AI": "Artificial Intelligence", "AR": "Hardware Architecture",
    "CC": "Computational Complexity", "CE": "Computational Engineering",
    "CG": "Computational Geometry", "CL": "Computation and Language",
    "CR": "Cryptography and Security", "CV": "Computer Vision",
    "CY": "Computers and Society", "DB": "Databases",
    "DC": "Distributed and Cluster Computing", "DL": "Digital Libraries",
    "DM": "Discrete Mathematics", "DS": "Data Structures and Algorithms",
    "ET": "Emerging Technologies", "FL": "Formal Languages and Automata Theory",
    "GL": "General Literature", "GR": "Graphics", "GT": "Computer Science and Game Theory",
    "HC": "Human-Computer Interaction", "IR": "Information Retrieval",
    "IT": "Information Theory", "LG": "Machine Learning", "LO": "Logic in Computer Science",
    "MA": "Multiagent Systems", "MM": "Multimedia", "MS": "Mathematical Software",
    "NA": "Numerical Analysis", "NE": "Neural and Evolutionary Computing",
    "NI": "Networking and Internet Architecture", "OH": "Other Computer Science",
    "OS": "Operating Systems", "PF": "Performance", "PL": "Programming Languages",
    "RO": "Robotics", "SC": "Symbolic Computation", "SD": "Sound",
    "SE": "Software Engineering", "SI": "Social and Information Networks",
    "SY": "Systems and Control",
}

QUESTION = (
    "This is a paper from the arXiv computer science collection, shown together with the "
    "papers it cites. Which arXiv CS subject area does it belong to?"
)
EDGE_TEXT = "cites"


def _load_label_names(ogb_root: Path) -> list[str]:
    """label idx -> readable category name, in OGB's label order."""
    df = pd.read_csv(ogb_root / "ogbn_arxiv" / "mapping" / "labelidx2arxivcategeory.csv.gz")
    names = []
    for _, row in df.sort_values("label idx").iterrows():
        code = str(row["arxiv category"]).replace("arxiv cs ", "").upper()
        if code not in ARXIV_CS_CATEGORIES:
            raise KeyError(f"unknown arXiv CS category code {code!r}")
        names.append(ARXIV_CS_CATEGORIES[code])
    assert len(names) == 40
    return names


def _load_node_texts(raw_dir: Path, ogb_root: Path, num_nodes: int) -> list[str]:
    """Node text = 'title. abstract', indexed by ogbn-arxiv node index."""
    with gzip.open(ogb_root / "ogbn_arxiv" / "mapping" / "nodeidx2paperid.csv.gz", "rt") as f:
        mapping = pd.read_csv(f)
    paper2node = dict(zip(mapping["paper id"].astype(np.int64), mapping["node idx"].astype(np.int64)))

    texts: list[str | None] = [None] * num_nodes
    with open(raw_dir / "titleabs.tsv", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue  # the file carries a stray header-ish first row
            node = paper2node.get(pid)
            if node is None:
                continue
            title, abstract = parts[1].strip(), parts[2].strip()
            texts[node] = f"{title}. {abstract}"

    missing = sum(t is None for t in texts)
    if missing:
        print(f"[arxiv] {missing} nodes have no title/abstract; using an empty placeholder")
    return [t if t is not None else "" for t in texts]


def _build_csr(edge_index: np.ndarray, num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Undirected adjacency in CSR form for fast neighbour sampling."""
    src = np.concatenate([edge_index[0], edge_index[1]])
    dst = np.concatenate([edge_index[1], edge_index[0]])
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.add.at(indptr, src + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, dst


def sample_ego_subgraph(
    center: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    num_hops: int,
    fanout: int,
    rng: np.random.Generator,
) -> tuple[list[int], torch.Tensor]:
    """Sampled k-hop neighbourhood around `center`.

    Returns (global node ids with the center at local index 0, local edge_index).
    Edges are emitted parent -> child in sampling order; `build_transition_edges`
    symmetrizes for diffusion, and the encoder sees them as given.
    """
    nodes = [int(center)]
    local = {int(center): 0}
    frontier = [int(center)]
    edges: list[tuple[int, int]] = []
    for _ in range(num_hops):
        next_frontier = []
        for u in frontier:
            lo, hi = indptr[u], indptr[u + 1]
            neigh = indices[lo:hi]
            if neigh.size == 0:
                continue
            if neigh.size > fanout:
                neigh = rng.choice(neigh, size=fanout, replace=False)
            for v in neigh:
                v = int(v)
                if v not in local:
                    local[v] = len(nodes)
                    nodes.append(v)
                    next_frontier.append(v)
                edges.append((local[u], local[v]))
        frontier = next_frontier
        if not frontier:
            break
    if edges:
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        ei = torch.zeros(2, 0, dtype=torch.long)
    return nodes, ei


def build_arxiv(
    raw_dir: Path,
    num_hops: int = 2,
    fanout: int = 10,
    max_train: int | None = None,
    max_val: int | None = None,
    seed: int = 0,
    graphtranslator_test_subset: Path | None = None,
) -> tuple[dict, list[dict], list[str]]:
    """Returns (graphs, examples, node_texts). Node text ids are global node indices."""
    from ogb.nodeproppred import NodePropPredDataset

    ogb_root = raw_dir / "ogb"
    # ogb 1.3.6 calls torch.load without weights_only, which PyTorch >= 2.6 defaults to True
    # and then refuses its cached .pt. The file is one we produced locally from the official
    # OGB download, so loading it fully is safe; scope the override to this one call.
    _torch_load = torch.load

    def _load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _torch_load(*args, **kwargs)

    torch.load = _load_compat
    try:
        ds = NodePropPredDataset(name="ogbn-arxiv", root=str(ogb_root))
    finally:
        torch.load = _torch_load
    graph, labels = ds[0]
    split = ds.get_idx_split()
    num_nodes = int(graph["num_nodes"])
    labels = labels.reshape(-1)

    label_names = _load_label_names(ogb_root)
    node_texts = _load_node_texts(raw_dir, ogb_root, num_nodes)
    indptr, indices = _build_csr(graph["edge_index"], num_nodes)

    test_idx = split["test"]
    if graphtranslator_test_subset is not None and Path(graphtranslator_test_subset).exists():
        sub = pd.read_csv(graphtranslator_test_subset)
        test_idx = sub["index"].to_numpy()
        print(f"[arxiv] using GraphTranslator's {len(test_idx)}-node test subset")

    rng = np.random.default_rng(seed)
    wanted: list[tuple[str, np.ndarray]] = []
    for name, idx, cap in (
        ("train", split["train"], max_train),
        ("val", split["valid"], max_val),
        ("test", test_idx, None),
    ):
        idx = np.asarray(idx).reshape(-1)
        if cap is not None and len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        wanted.append((name, idx))

    graphs: dict[str, dict] = {}
    examples: list[dict] = []
    for name, idx in wanted:
        for node in idx:
            node = int(node)
            key = str(node)
            nodes, ei = sample_ego_subgraph(node, indptr, indices, num_hops, fanout, rng)
            graphs[key] = {
                "node_text_id": torch.tensor(nodes, dtype=torch.int32),
                "edge_text_id": torch.zeros(ei.shape[1], dtype=torch.int32),  # filled later
                "edge_index": ei.to(torch.int32),
            }
            examples.append(
                {
                    "id": node,
                    "graph_key": key,
                    "question": QUESTION,
                    "answer": label_names[int(labels[node])],
                    "split": name,
                    "roles": [[0, 2]],  # center node -> ROLE_SOURCE (ReGraph.md 2.1)
                    "label_idx": int(labels[node]),
                }
            )
    return graphs, examples, node_texts
