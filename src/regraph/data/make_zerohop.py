"""Derive a 0-hop variant of a preprocessed dataset: centre node only, no neighbours.

The isolation control for NeighborhoodQA. The `num_rounds=0` control removes *all* graph
information — including which paper is being asked about, since the prompt is a fixed question —
so it cannot separate "read the neighbourhood" from "read the centre node". This variant keeps
the centre node's text flowing through the graph channel and deletes only its neighbours, while
reusing the **identical** examples.json (same ground truth, same splits). Any drop versus the
full run is therefore attributable to the neighbourhood alone.

Usage: python -m regraph.data.make_zerohop --config configs/arxiv_nbrqa.yaml --suffix zerohop
"""

from __future__ import annotations

import argparse
import shutil

import torch

from regraph.data.preprocess import store_dir_for
from regraph.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--keep-hops", type=int, default=0,
                    help="0 = centre only; 1 = centre + direct neighbours")
    ap.add_argument("--suffix", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    src = store_dir_for(cfg)
    suffix = args.suffix or f"{args.keep_hops}hop"
    dst = src.parent.parent / f"{cfg['dataset']['name']}_{suffix}" / src.name
    dst.mkdir(parents=True, exist_ok=True)

    graphs = torch.load(src / "graphs.pt", weights_only=True)
    out = {}
    for k, g in graphs.items():
        # sample_ego_subgraph always places the centre at local index 0
        ei = g["edge_index"].long()
        if args.keep_hops == 0:
            keep = [0]
            new_ei = torch.zeros(2, 0, dtype=torch.int32)
            keep_e = []
        else:
            src_l, dst_l = ei[0].tolist(), ei[1].tolist()
            hop1 = {int(d) for s_, d in zip(src_l, dst_l) if s_ == 0}
            keep = [0] + sorted(hop1)
            remap = {o: i for i, o in enumerate(keep)}
            keep_e = [i for i, (a, b) in enumerate(zip(src_l, dst_l))
                      if a in remap and b in remap]
            new_ei = torch.tensor(
                [[remap[src_l[i]] for i in keep_e], [remap[dst_l[i]] for i in keep_e]],
                dtype=torch.int32) if keep_e else torch.zeros(2, 0, dtype=torch.int32)
        out[k] = {
            "node_text_id": g["node_text_id"][torch.tensor(keep)].clone(),
            "edge_text_id": (g["edge_text_id"][torch.tensor(keep_e)].clone()
                             if keep_e else torch.zeros(0, dtype=torch.int32)),
            "edge_index": new_ei,
        }
    torch.save(out, dst / "graphs.pt")
    for f in ("emb.f16.npy", "texts.json", "meta.json", "examples.json", "stats.json"):
        if (src / f).exists() and not (dst / f).exists():
            shutil.copy(src / f, dst / f)
    import numpy as _np
    sizes = [len(v["node_text_id"]) for v in out.values()]
    print(f"[truncate] {len(out):,} graphs -> keep_hops={args.keep_hops}, "
          f"mean {_np.mean(sizes):.1f} nodes (was {_np.mean([len(v['node_text_id']) for v in graphs.values()]):.1f}) -> {dst}")


if __name__ == "__main__":
    main()
