"""Export StructuralAnomaly to a portable, self-contained benchmark file.

Both arms are written: the main task and the falsifiable control. The control is not optional
extra material -- it is what makes the main number interpretable, so it ships with the data.

Node text is stored as indices into a small shared vocabulary (61 strings) carried in
manifest.json, rather than inlined per node: the graphs are synthetic, so there is no external
source to recover text from, but the vocabulary is tiny and repeating it 660,000 times would
bloat the file for no benefit.

Usage: python benchmarks/structuralanomaly/export.py
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import torch

from regraph.data.preprocess import store_dir_for
from regraph.data.synth_raw import THEMES, EDGE_TEXT
from regraph.utils.config import load_config

OUT = Path(__file__).parent
ARMS = {"main": "configs/synth_anomaly.yaml", "control": "configs/synth_anomaly_control.yaml"}


def dump(cfg_path: str, out_name: str) -> tuple[int, dict]:
    cfg = load_config(cfg_path)
    src = store_dir_for(cfg)
    graphs = torch.load(src / "graphs.pt", weights_only=True)
    examples = json.loads((src / "examples.json").read_text())
    n = 0
    with gzip.open(OUT / out_name, "wt") as f:
        for ex in examples:
            g = graphs[ex["graph_key"]]
            ei = g["edge_index"].long().tolist()
            f.write(json.dumps({
                "id": ex["id"],
                "split": ex["split"],
                "question": ex["question"],
                "answer": ex["answer"],                       # the theme of the dense community
                "n_nodes": ex["n_nodes"],
                "hot_community": ex["hot_community"],         # index into the per-graph ordering
                "node_text_ids": [int(v) for v in g["node_text_id"].tolist()],
                "edges": [[int(a), int(b)] for a, b in zip(ei[0], ei[1])],
            }) + "\n")
            n += 1
    return n, json.loads((src / "stats.json").read_text())


def main() -> None:
    counts, stats = {}, {}
    for arm, cfg_path in ARMS.items():
        counts[arm], stats[arm] = dump(cfg_path, f"structuralanomaly_{arm}.jsonl.gz")

    vocab: list[str] = []
    for t in sorted(THEMES):
        vocab.extend(THEMES[t])
    vocab.append(EDGE_TEXT)

    (OUT / "manifest.json").write_text(json.dumps({
        "name": "StructuralAnomaly",
        "version": "1.0",
        "built_from": "synthetic (planted-partition generator, src/regraph/data/synth_raw.py)",
        "task": "A graph holds k equally sized thematic communities; exactly one is wired far "
                "more densely than the rest. No node is named in the question. Say what the "
                "dense group is about.",
        "anchor_free": True,
        "why_anchor_free_matters": "ReGraph.md §2.1 motivates the method with anchor-free "
                                   "queries where every node carries the `none` marker and the "
                                   "model must locate the region itself. Every other benchmark "
                                   "in this repo marks a question entity or a centre node.",
        "output": "free text; the theme name",
        "metric": "accuracy (first-match over the theme vocabulary); legality rate alongside",
        "chance": round(100.0 / 5, 2),
        "chance_is_analytic": "All k themes appear in every graph, each is equally likely to be "
                              "the dense one, and all communities have the same size, so no "
                              "policy that ignores topology can exceed 1/k. Not an estimate.",
        "examples": counts,
        "splits": stats["main"]["split_sizes"],
        "generator": {
            "num_communities": 5, "community_size": 12,
            "p_in": stats["main"]["p_in"], "p_hot": stats["main"]["p_hot"],
            "p_out": stats["main"]["p_out"],
            "mean_degree_gap_hot_vs_rest": {
                "main": stats["main"]["mean_degree_gap_hot_vs_rest"],
                "control": stats["control"]["mean_degree_gap_hot_vs_rest"],
            },
        },
        "control": "structuralanomaly_control.jsonl.gz removes the density contrast and draws "
                   "the label uniformly at random, so the task is unanswerable by construction. "
                   "Any score above chance there means the generator leaks and the main number "
                   "is void. ReGraph scores 19.80 on it against an analytic 20.00.",
        "reference_points": {
            "chance (analytic)": 20.00,
            "ReGraph 8B, control arm": 19.80,
            "ReGraph 8B, main arm": 99.05,
            "ReGraph 3B, control arm": 20.05,
            "ReGraph 3B, main arm": 99.15,
        },
        "known_limitation": "The structural signal is DEGREE (the dense community's nodes carry "
                            "~4.4 more edges), a first-order local quantity that one round of "
                            "message passing can compute. This benchmark therefore demonstrates "
                            "anchor-free localization by a local structural cue, not path-level "
                            "reasoning -- ReGraph remains at chance on NLGraph connectivity and "
                            "cycle. At density_ratio=5.0 the main arm is also saturated (99.05), "
                            "so the informative quantity is the accuracy-vs-density_ratio curve, "
                            "not this single point.",
        "node_vocabulary": vocab,
        "themes": {t: THEMES[t] for t in sorted(THEMES)},
        "scorer": "benchmarks/structuralanomaly/score.py",
    }, indent=2))
    print(f"wrote {counts} examples; vocabulary {len(vocab)} strings")


if __name__ == "__main__":
    main()
