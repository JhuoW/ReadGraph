"""Build PCST-retrieved variants of an existing preprocessed dataset.

GRAFF (Findings of EACL 2026) and G-Retriever both evaluate on the *retrieved* subgraph, not
the full graph: GRAFF's Table 2 reports WebQSP at 8.39 average nodes against the full graph's
~1,371. ReGraph as specified reads the full graph (`ReGraph.md` §3.2 — no retrieval step), so
its numbers are not comparable to theirs without this stage.

Running it is also a direct test of a measured failure: ReGraph's reader localizes the WebQSP
gold node at median rank 301/1371, and retrieval removes that problem by construction.

The selection logic is a port of G-Retriever `src/dataset/utils/retrieval.py::retrieval_via_pcst`
(same prizes, same costs, same virtual-node trick), changed only to return *indices* so they can
be remapped into this repo's store layout. The attribute store is reused unchanged — retrieval
selects a subset of existing nodes/edges, it does not create new text.

Usage:
  python -m regraph.data.pcst_retrieve --config configs/webqsp.yaml --topk 3 --topk-e 5
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from regraph.data.attr_encoder import AttrStore, encoder_slug
from regraph.data.preprocess import store_dir_for
from regraph.utils.config import load_config


def pcst_select(x, edge_index, edge_attr, q_emb, topk=3, topk_e=3, cost_e=0.5):
    """Port of retrieval_via_pcst, returning (selected_nodes, selected_edges) index arrays."""
    from pcst_fast import pcst_fast

    c = 0.01
    num_nodes = x.shape[0]
    if num_nodes == 0 or edge_index.shape[1] == 0:
        return np.arange(num_nodes), np.arange(edge_index.shape[1])

    if topk > 0:
        n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, x)
        k = min(topk, num_nodes)
        _, topk_n_indices = torch.topk(n_prizes, k, largest=True)
        n_prizes = torch.zeros_like(n_prizes)
        n_prizes[topk_n_indices] = torch.arange(k, 0, -1).float()
    else:
        n_prizes = torch.zeros(num_nodes)

    if topk_e > 0:
        e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, edge_attr)
        topk_e = min(topk_e, e_prizes.unique().size(0))
        topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
        e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        last = topk_e
        for k in range(topk_e):
            indices = e_prizes == topk_e_values[k]
            value = min((topk_e - k) / max(int(indices.sum()), 1), last)
            e_prizes[indices] = value
            last = value * (1 - c)
        cost_e = min(cost_e, e_prizes.max().item() * (1 - c / 2))
    else:
        e_prizes = torch.zeros(edge_index.shape[1])

    costs, edges, virt_prizes, virt_edges, virt_costs = [], [], [], [], []
    mapping_n, mapping_e = {}, {}
    for i, (src, dst) in enumerate(edge_index.T.numpy()):
        prize_e = e_prizes[i]
        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            vid = num_nodes + len(virt_prizes)
            mapping_n[vid] = i
            virt_edges += [(src, vid), (vid, dst)]
            virt_costs += [0, 0]
            virt_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([n_prizes.numpy(), np.array(virt_prizes)])
    num_edges = len(edges)
    all_costs = np.array(costs + virt_costs)
    all_edges = np.array(edges + virt_edges)
    if all_edges.size == 0:
        return np.arange(num_nodes), np.arange(edge_index.shape[1])

    vertices, sel = pcst_fast(all_edges, prizes, all_costs, -1, 1, "gw", 0)
    selected_nodes = vertices[vertices < num_nodes]
    selected_edges = [mapping_e[e] for e in sel if e < num_edges]
    virtual_vertices = vertices[vertices >= num_nodes]
    if len(virtual_vertices) > 0:
        selected_edges = np.array(selected_edges + [mapping_n[i] for i in virtual_vertices])
    selected_edges = np.asarray(selected_edges, dtype=np.int64).reshape(-1)

    if selected_edges.size:
        ei = edge_index[:, selected_edges]
        selected_nodes = np.unique(
            np.concatenate([selected_nodes, ei[0].numpy(), ei[1].numpy()])
        )
    return np.asarray(selected_nodes, dtype=np.int64), selected_edges


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--topk-e", type=int, default=5)
    ap.add_argument("--cost-e", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    src_dir = store_dir_for(cfg)
    dst_dir = src_dir.parent.parent / f"{cfg['dataset']['name']}_pcst" / encoder_slug(
        cfg["data"]["attr_encoder"]
    )
    dst_dir.mkdir(parents=True, exist_ok=True)

    store = AttrStore(src_dir)
    graphs = torch.load(src_dir / "graphs.pt", weights_only=True)
    with open(src_dir / "examples.json") as f:
        examples = json.load(f)

    # question embeddings drive the PCST prizes, exactly as in G-Retriever
    from regraph.data.attr_encoder import encode_texts
    questions = [e["question"] for e in examples]
    print(f"[pcst] encoding {len(questions):,} questions")
    q_embs = encode_texts(questions, cfg["data"]["attr_encoder"], device=args.device,
                          batch_size=256, d_attr=cfg["data"]["d_attr"])

    new_graphs, sizes = {}, []
    for i, ex in enumerate(tqdm(examples, desc="pcst")):
        g = graphs[ex["graph_key"]]
        nid, eid = g["node_text_id"], g["edge_text_id"]
        x = store.embed(nid)
        edge_attr = store.embed(eid)
        ei = g["edge_index"].long()

        sel_n, sel_e = pcst_select(x, ei, edge_attr, q_embs[i],
                                   topk=args.topk, topk_e=args.topk_e, cost_e=args.cost_e)
        remap = {int(o): k for k, o in enumerate(sel_n.tolist())}
        sub_ei = ei[:, sel_e] if sel_e.size else torch.zeros(2, 0, dtype=torch.long)
        sub_ei = torch.tensor(
            [[remap[int(v)] for v in sub_ei[0].tolist()],
             [remap[int(v)] for v in sub_ei[1].tolist()]], dtype=torch.int32
        ) if sub_ei.shape[1] else torch.zeros(2, 0, dtype=torch.int32)

        key = ex["graph_key"] + "-pcst"
        new_graphs[key] = {
            "node_text_id": nid[torch.from_numpy(sel_n)],
            "edge_text_id": eid[torch.from_numpy(sel_e)] if sel_e.size
                            else torch.zeros(0, dtype=torch.int32),
            "edge_index": sub_ei,
        }
        ex["graph_key"] = key
        ex["roles"] = [[remap[i0], r] for i0, r in ex.get("roles", []) if i0 in remap]
        sizes.append(len(sel_n))

    for f in ("emb.f16.npy", "texts.json", "meta.json"):
        if not (dst_dir / f).exists():
            shutil.copy(src_dir / f, dst_dir / f)
    torch.save(new_graphs, dst_dir / "graphs.pt")
    with open(dst_dir / "examples.json", "w") as f:
        json.dump(examples, f)

    from regraph.data.preprocess import compute_stats
    stats = compute_stats(new_graphs, examples)
    stats["pcst"] = {"topk": args.topk, "topk_e": args.topk_e, "cost_e": args.cost_e}
    with open(dst_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[pcst] {dst_dir}")
    print(f"[pcst] nodes/graph: mean {np.mean(sizes):.2f} (GRAFF Table 2 reports 8.39 for WebQSP)")
    print(f"[pcst] stats: {json.dumps(stats)}")


if __name__ == "__main__":
    main()
