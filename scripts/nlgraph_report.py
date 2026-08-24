"""Per-difficulty accuracy for NLGraph runs, laid out like the paper's Table 2.

Joins predictions_test.jsonl with examples.json (which carries the difficulty label) and
prints ReGraph next to the published text-davinci-003 baselines.

Usage: python scripts/nlgraph_report.py [connectivity cycle ...]
"""
import json
import sys
from pathlib import Path

from regraph.data.preprocess import store_dir_for
from regraph.utils.config import load_config

# Wang et al., NeurIPS 2023, Table 2 (text-davinci-003, standard set, prompting).
BASELINES = {
    "connectivity": {
        "RANDOM": (50.00, 50.00, 50.00, 50.00),
        "ZERO-SHOT": (83.81, 72.75, 63.38, 71.31),
        "FEW-SHOT": (93.75, 83.83, 76.61, 84.73),
        "CoT": (94.32, 82.17, 77.21, 84.57),
        "0-CoT": (79.55, 65.83, 68.53, 71.30),
        "CoT+SC": (93.18, 84.50, 82.79, 86.82),
    },
    "cycle": {
        "RANDOM": (50.00, 50.00, 50.00, 50.00),
        "ZERO-SHOT": (50.00, 50.00, 50.00, 50.00),
        "FEW-SHOT": (80.00, 70.00, 61.00, 70.33),
        "CoT": (84.67, 63.33, 53.25, 66.75),
        "0-CoT": (55.33, 57.67, 49.00, 54.00),
        "CoT+SC": (82.00, 63.67, 53.50, 66.39),
    },
}
LEVELS = ("easy", "medium", "hard")


def report(task: str, run: str = "seed0") -> None:
    cfg = load_config(f"configs/nlgraph_{task}.yaml")
    with open(store_dir_for(cfg) / "examples.json") as f:
        diff = {e["id"]: e["difficulty"] for e in json.load(f) if e["split"] == "test"}

    path = Path(f"runs/nlgraph_{task}/{run}/predictions_test.jsonl")
    if not path.exists():
        print(f"[{task}] no predictions at {path}")
        return
    recs = [json.loads(l) for l in open(path)]

    per = {lv: [0, 0] for lv in LEVELS}
    for r in recs:
        lv = diff.get(r["id"])
        if lv is None:
            continue
        per[lv][0] += int(r["correct"])
        per[lv][1] += 1
    overall = sum(r["correct"] for r in recs) / len(recs) * 100
    # NLGraph's own "Avg" column is the UNWEIGHTED mean of the three difficulty subsets
    # (verified against their Table 2: 4 of 5 rows reproduce exactly). Our overall accuracy is
    # size-weighted, which is a different quantity — report the matching one for comparison.
    sub = [per[lv][0] / per[lv][1] * 100 for lv in LEVELS if per[lv][1]]
    unweighted = sum(sub) / len(sub) if sub else float("nan")

    print(f"\n=== NLGraph / {task} — test split, n={len(recs)} ===")
    print(f"{'Method':<26}{'Easy':>8}{'Medium':>9}{'Hard':>8}{'Avg':>9}   protocol")
    for name, (e, m, h, a) in BASELINES[task].items():
        tag = "—" if name == "RANDOM" else "prompting (text-davinci-003)"
        print(f"{name:<26}{e:>8.2f}{m:>9.2f}{h:>8.2f}{a:>9.2f}   {tag}")
    cells = []
    for lv in LEVELS:
        c, n = per[lv]
        cells.append(f"{c/n*100:>8.2f}" if n else f"{'n/a':>8}")
    print(f"{'ReGraph (this repo)':<26}{''.join(cells)}{unweighted:>9.2f}   SUPERVISED — not like-for-like")
    print(f"{'  (size-weighted overall)':<26}{'':>25}{overall:>9.2f}   different convention, for reference")
    print("  per-level test counts: " + ", ".join(f"{lv}={per[lv][1]}" for lv in LEVELS))


if __name__ == "__main__":
    for t in (sys.argv[1:] or ["connectivity", "cycle"]):
        report(t)
