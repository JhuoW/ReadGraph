"""Preprocessing CLI for text-attributed Cora (LLaGA / GraphGPT benchmark)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from regraph.data.attr_encoder import AttrStore
from regraph.data.preprocess import compute_stats, store_dir_for
from regraph.data.tag_raw import build_cora, build_products, build_pubmed
from regraph.utils.config import load_config

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    ds = cfg["dataset"]
    out = store_dir_for(cfg); out.mkdir(parents=True, exist_ok=True)

    builder = {"cora": build_cora, "pubmed": build_pubmed,
               "products": build_products}[ds["name"]]
    graphs, examples, texts, names = builder(
        Path(cfg["data"]["tag_raw_dir"]), num_hops=ds["num_hops"],
        fanout=ds["fanout"], seed=cfg["seed"])
    AttrStore.build(out, texts, cfg["data"]["attr_encoder"],
                    device=args.device, batch_size=args.batch_size, d_attr=cfg["data"]["d_attr"])
    torch.save(graphs, out / "graphs.pt")
    with open(out / "examples.json", "w") as f: json.dump(examples, f)
    stats = compute_stats(graphs, examples); stats["classes"] = names
    (out / "classes.json").write_text(json.dumps(names))
    with open(out / "stats.json", "w") as f: json.dump(stats, f, indent=2)
    print(f"[{ds['name']}] {json.dumps({k:v for k,v in stats.items() if k!='classes'})}")

if __name__ == "__main__":
    main()
