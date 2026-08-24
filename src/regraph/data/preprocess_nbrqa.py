"""Preprocessing CLI for NeighborhoodQA (open-ended graph reasoning)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from regraph.data.attr_encoder import AttrStore
from regraph.data.nbrqa_raw import build_nbrqa
from regraph.data.preprocess import compute_stats, store_dir_for
from regraph.utils.config import load_config

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/arxiv_nbrqa.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    ds = cfg["dataset"]; out = store_dir_for(cfg); out.mkdir(parents=True, exist_ok=True)
    graphs, examples, texts, names, extra = build_nbrqa(
        Path(cfg["data"]["arxiv_raw_dir"]), num_hops=ds["num_hops"], fanout=ds["fanout"],
        max_train=ds.get("max_train"), max_val=ds.get("max_val"),
        max_test=ds.get("max_test"), seed=cfg["seed"],
        target_hop=ds.get("target_hop", 1))
    AttrStore.build(out, texts, cfg["data"]["attr_encoder"], device=a.device,
                    batch_size=a.batch_size, d_attr=cfg["data"]["d_attr"])
    torch.save(graphs, out / "graphs.pt")
    (out / "examples.json").write_text(json.dumps(examples))
    stats = compute_stats(graphs, examples); stats.update(extra)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[nbrqa] {json.dumps(stats)}")

if __name__ == "__main__":
    main()
