"""Export NeighborhoodQA to a portable, self-contained benchmark file.

The preprocessed store is ReGraph-specific (torch tensors + an sbert embedding memmap) and
~600 MB. This writes the benchmark itself: for each example, the question, the gold area set,
the split, and the sampled ego-subgraph as ogbn-arxiv node indices plus an edge list. Node text
is *referenced* rather than inlined (it is recoverable from the public `titleabs.tsv`), which
keeps the file small and encoder-agnostic.

Usage: python benchmarks/neighborhoodqa/export.py
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import torch

from regraph.data.preprocess import store_dir_for
from regraph.utils.config import load_config

OUT = Path(__file__).parent


def main() -> None:
    cfg = load_config("configs/arxiv_nbrqa.yaml")
    src = store_dir_for(cfg)
    graphs = torch.load(src / "graphs.pt", weights_only=True)
    examples = json.loads((src / "examples.json").read_text())

    n_written = 0
    with gzip.open(OUT / "neighborhoodqa.jsonl.gz", "wt") as f:
        for ex in examples:
            g = graphs[ex["graph_key"]]
            nodes = [int(v) for v in g["node_text_id"].tolist()]   # ogbn-arxiv node indices
            ei = g["edge_index"].long().tolist()
            f.write(json.dumps({
                "id": ex["id"],
                "split": ex["split"],
                "question": ex["question"],
                "answer": ex["answer"],                 # gold area set, comma-separated
                "n_areas": ex["n_areas"],
                "n_neighbours": ex["n_neighbours"],
                "center_node": nodes[0],                # local index 0 is always the centre
                "nodes": nodes,                         # local index -> ogbn-arxiv node index
                "edges": [[int(a), int(b)] for a, b in zip(ei[0], ei[1])],  # local indices
            }) + "\n")
            n_written += 1

    stats = json.loads((src / "stats.json").read_text())
    (OUT / "manifest.json").write_text(json.dumps({
        "name": "NeighborhoodQA",
        "version": "1.0",
        "built_from": "ogbn-arxiv (OGB) + titleabs.tsv",
        "task": "Given a paper's sampled citation ego-subgraph, name the SET of arXiv CS "
                "subject areas represented among its 1-hop neighbours.",
        "output": "free text; a variable-length list of area names",
        "metric": "set-F1 (per-example, averaged); exact-set-match reported alongside",
        "examples": n_written,
        "splits": stats["split_sizes"],
        "mean_areas_per_answer": stats["mean_areas"],
        "sampling": {"num_hops": 2, "fanout": 10,
                     "note": "ground truth is computed on the SAMPLED subgraph, not the full "
                             "neighbourhood, so the task is answerable from what is shown"},
        "reference_points": {
            "no_graph_at_all (num_rounds=0)": 6.98,
            "answer_centre_paper_own_area (analytic shortcut)": 63.81,
            "centre_node_only (0-hop control)": 72.65,
            "ReGraph full 2-hop": 83.87,
        },
        "node_text": "node indices refer to ogbn-arxiv; recover title+abstract from "
                     "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz "
                     "via mapping/nodeidx2paperid.csv.gz",
        "scorer": "benchmarks/neighborhoodqa/score.py",
    }, indent=2))
    print(f"wrote {n_written:,} examples")


if __name__ == "__main__":
    main()
