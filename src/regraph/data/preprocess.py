"""GraphQA preprocessing CLI.

Builds, per dataset (docs/components/01-data.md):
  {cache_dir}/{dataset}/{encoder_slug}/
    meta.json / texts.json / emb.f16.npy   deduplicated attribute embeddings
    graphs.pt      {key: {node_text_id, edge_text_id, edge_index}}
    examples.json  [{id, graph_key, question, answer, split, roles: [[node, role], ...]}]
    stats.json     split sizes, avg nodes/edges, non-none role rate

Usage: python -m regraph.data.preprocess --config configs/expla_graphs.yaml [--device cuda]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from regraph.data.attr_encoder import AttrStore, TextInterner, encoder_slug
from regraph.data.gretriever_raw import BUILDERS
from regraph.data.roles import assign_roles
from regraph.utils.config import load_config


def store_dir_for(cfg: dict) -> Path:
    return (
        Path(cfg["data"]["cache_dir"])
        / cfg["dataset"]["name"]
        / encoder_slug(cfg["data"]["attr_encoder"])
    )


def compute_stats(graphs: dict, examples: list[dict]) -> dict:
    n_nodes = 0
    n_edges = 0
    split_sizes: dict[str, int] = {}
    role_examples = 0
    for ex in examples:
        g = graphs[ex["graph_key"]]
        n_nodes += len(g["node_text_id"]) if "node_text_id" in g else len(g["node_texts"])
        n_edges += g["edge_index"].shape[1]
        split_sizes[ex["split"]] = split_sizes.get(ex["split"], 0) + 1
        if ex.get("roles"):
            role_examples += 1
    n = len(examples)
    return {
        "num_examples": n,
        "split_sizes": split_sizes,
        "avg_nodes": round(n_nodes / n, 2),
        "avg_edges": round(n_edges / n, 2),
        "frac_examples_with_roles": round(role_examples / n, 4),
    }


def verify_stats(stats: dict, dataset_cfg: dict) -> list[str]:
    """Compare against docs/experimental-protocol.md expectations. A mismatch means
    wrong split files — stop and report (docs/components/01-data.md)."""
    problems = []
    if "expected_splits" in dataset_cfg:
        for k, v in dataset_cfg["expected_splits"].items():
            got = stats["split_sizes"].get(k, 0)
            if got != v:
                problems.append(f"split {k}: expected {v}, got {got}")
    if "expected_total" in dataset_cfg and stats["num_examples"] != dataset_cfg["expected_total"]:
        problems.append(
            f"total: expected {dataset_cfg['expected_total']}, got {stats['num_examples']}"
        )
    for key, stat in (("expected_avg_nodes", "avg_nodes"), ("expected_avg_edges", "avg_edges")):
        if key in dataset_cfg and abs(stats[stat] - dataset_cfg[key]) > 0.011:
            problems.append(f"{stat}: expected {dataset_cfg[key]}, got {stats[stat]}")
    return problems


def preprocess(cfg: dict, device: str = "cuda", batch_size: int = 256) -> Path:
    name = cfg["dataset"]["name"]
    raw_dir = Path(cfg["data"].get("raw_dir", "data/raw")) / name
    out_dir = store_dir_for(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[preprocess] building {name} from {raw_dir}")
    graphs_raw, examples = BUILDERS[name](raw_dir)

    # ---- roles (needs raw node texts + question) --------------------------------
    print("[preprocess] assigning query roles")
    for ex in tqdm(examples, unit="ex"):
        roles = assign_roles(
            ex["question"],
            graphs_raw[ex["graph_key"]]["node_texts"],
            source_entities=ex.get("q_entity"),
        )
        ex["roles"] = [[i, r] for i, r in enumerate(roles) if r != 0]

    # ---- dedup + encode ---------------------------------------------------------
    print("[preprocess] interning texts")
    interner = TextInterner()
    graphs = {}
    for key, g in tqdm(graphs_raw.items(), unit="graph"):
        graphs[key] = {
            "node_text_id": interner.intern(g["node_texts"]),
            "edge_text_id": interner.intern(g["edge_texts"]),
            "edge_index": g["edge_index"].to(torch.int32),
        }
    uniq = interner.texts
    print(f"[preprocess] {len(uniq):,} unique texts "
          f"(from {sum(len(g['node_texts']) + len(g['edge_texts']) for g in graphs_raw.values()):,})")
    del graphs_raw

    AttrStore.build(
        out_dir, uniq, cfg["data"]["attr_encoder"],
        device=device, batch_size=batch_size, d_attr=cfg["data"]["d_attr"],
    )

    torch.save(graphs, out_dir / "graphs.pt")
    with open(out_dir / "examples.json", "w") as f:
        json.dump(examples, f)

    stats = compute_stats(graphs, examples)
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[preprocess] stats: {json.dumps(stats)}")

    problems = verify_stats(stats, cfg["dataset"])
    if problems:
        raise RuntimeError(
            "preprocessed stats do not match docs/experimental-protocol.md — wrong "
            "split files? " + "; ".join(problems)
        )
    print("[preprocess] stats match the experimental protocol ✓")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("overrides", nargs="*", help="dotted config overrides key=value")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    preprocess(cfg, device=args.device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
