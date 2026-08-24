"""Preprocessing CLI for StructuralAnomaly, the synthetic anchor-free benchmark."""
from __future__ import annotations
import argparse, json
import torch
from regraph.data.attr_encoder import AttrStore
from regraph.data.synth_raw import build_synth
from regraph.data.preprocess import compute_stats, store_dir_for
from regraph.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synth_anomaly.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    ds = cfg["dataset"]; out = store_dir_for(cfg); out.mkdir(parents=True, exist_ok=True)
    graphs, examples, texts, names, extra = build_synth(
        num_communities=ds["num_communities"], community_size=ds["community_size"],
        p_in=ds["p_in"], density_ratio=ds["density_ratio"], p_out=ds["p_out"],
        n_train=ds["n_train"], n_val=ds["n_val"], n_test=ds["n_test"],
        seed=cfg["seed"], anomaly=ds.get("anomaly", True))
    AttrStore.build(out, texts, cfg["data"]["attr_encoder"], device=a.device,
                    batch_size=a.batch_size, d_attr=cfg["data"]["d_attr"])
    torch.save(graphs, out / "graphs.pt")
    (out / "examples.json").write_text(json.dumps(examples))
    (out / "classes.json").write_text(json.dumps(names))     # theme vocabulary for scoring
    stats = compute_stats(graphs, examples); stats.update(extra)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[synth] {json.dumps(stats)}")


if __name__ == "__main__":
    main()
