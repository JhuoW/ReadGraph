"""Preprocessing CLI for ogbn-arxiv (GraphTranslator benchmark).

Produces the same on-disk layout as `regraph.data.preprocess`, so `GraphQADataset`,
`make_collate_fn` and the model consume it unchanged.

Usage:
  python -m regraph.data.preprocess_arxiv --config configs/arxiv.yaml [--device cuda]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from regraph.data.arxiv_raw import EDGE_TEXT, build_arxiv
from regraph.data.attr_encoder import AttrStore
from regraph.data.preprocess import compute_stats, store_dir_for
from regraph.utils.config import load_config


def preprocess_arxiv(cfg: dict, device: str = "cuda", batch_size: int = 128) -> Path:
    ds_cfg = cfg["dataset"]
    raw_dir = Path(cfg["data"].get("arxiv_raw_dir", "/mnt/ssd1/zhuowei/regraph-cache/arxiv_raw"))
    out_dir = store_dir_for(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[arxiv] building ego-subgraphs from {raw_dir}")
    graphs, examples, node_texts = build_arxiv(
        raw_dir,
        num_hops=ds_cfg["num_hops"],
        fanout=ds_cfg["fanout"],
        max_train=ds_cfg.get("max_train"),
        max_val=ds_cfg.get("max_val"),
        seed=cfg["seed"],
        graphtranslator_test_subset=ds_cfg.get("graphtranslator_test_subset"),
    )

    # one shared "cites" attribute for every (untyped) citation edge, appended after the
    # node texts so node_text_id stays equal to the global ogbn-arxiv node index
    edge_text_id = len(node_texts)
    all_texts = node_texts + [EDGE_TEXT]
    for g in graphs.values():
        g["edge_text_id"] = torch.full(
            (g["edge_index"].shape[1],), edge_text_id, dtype=torch.int32
        )

    sizes = [len(g["node_text_id"]) for g in graphs.values()]
    print(f"[arxiv] {len(examples):,} examples | {len(all_texts):,} unique texts "
          f"| subgraph nodes: mean {sum(sizes)/len(sizes):.1f}, max {max(sizes)}")

    AttrStore.build(out_dir, all_texts, cfg["data"]["attr_encoder"],
                    device=device, batch_size=batch_size, d_attr=cfg["data"]["d_attr"])
    torch.save(graphs, out_dir / "graphs.pt")
    with open(out_dir / "examples.json", "w") as f:
        json.dump(examples, f)

    stats = compute_stats(graphs, examples)
    stats["subgraph_nodes_mean"] = round(sum(sizes) / len(sizes), 2)
    stats["num_hops"] = ds_cfg["num_hops"]
    stats["fanout"] = ds_cfg["fanout"]
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[arxiv] stats: {json.dumps(stats)}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/arxiv.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    preprocess_arxiv(cfg, device=args.device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
