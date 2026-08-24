"""Preprocessing CLI for NLGraph tasks (Wang et al., NeurIPS 2023).

Usage: python -m regraph.data.preprocess_nlgraph --config configs/nlgraph_connectivity.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from regraph.data.attr_encoder import AttrStore
from regraph.data.nlgraph_raw import build_nlgraph
from regraph.data.preprocess import compute_stats, store_dir_for
from regraph.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    task = cfg["dataset"]["task"]
    raw_dir = Path(cfg["data"]["nlgraph_raw_dir"])
    out_dir = store_dir_for(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    graphs, examples, texts = build_nlgraph(
        raw_dir, task, seed=cfg["seed"],
        name_anchors=bool(cfg["dataset"].get("name_anchors", False)))
    AttrStore.build(out_dir, texts, cfg["data"]["attr_encoder"],
                    device=args.device, batch_size=8, d_attr=cfg["data"]["d_attr"])
    torch.save(graphs, out_dir / "graphs.pt")
    with open(out_dir / "examples.json", "w") as f:
        json.dump(examples, f)

    stats = compute_stats(graphs, examples)
    import collections
    stats["difficulty"] = dict(collections.Counter(e["difficulty"] for e in examples))
    stats["answers"] = dict(collections.Counter(e["answer"] for e in examples))
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[nlgraph:{task}] {json.dumps(stats)}")


if __name__ == "__main__":
    main()
