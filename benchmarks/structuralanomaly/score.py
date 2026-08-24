"""Standalone scorer for StructuralAnomaly — no ReGraph dependency.

Usage: python score.py predictions.jsonl [--arm main|control]
       predictions.jsonl: one {"id": ..., "pred": "..."} per test example.
       Gold is read from structuralanomaly_<arm>.jsonl.gz next to this file.

Scoring is first-match-wins over the theme vocabulary, the same convention the reference
implementation uses for Cora/PubMed and that G-Retriever uses for ExplaGraphs. `legality` is the
fraction of predictions naming any theme at all; report it, because a free-text model that emits
nothing valid would otherwise be indistinguishable from one that guesses.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import string
from pathlib import Path

THEMES = ["cryptography", "jazz music", "marine biology", "textile manufacturing", "volcanology"]
CHANCE = 100.0 / len(THEMES)


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def first_of(pred: str) -> str | None:
    """Earliest-mentioned theme in the prediction, or None."""
    p = normalize(pred)
    best, at_best = None, len(p) + 1
    for t in THEMES:
        at = p.find(normalize(t))
        if at != -1 and at < at_best:
            best, at_best = t, at
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--arm", default="main", choices=["main", "control"])
    a = ap.parse_args()

    here = Path(__file__).parent
    gold = {}
    with gzip.open(here / f"structuralanomaly_{a.arm}.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == "test":
                gold[r["id"]] = r["answer"]

    preds = {}
    with open(a.predictions) as f:
        for line in f:
            r = json.loads(line)
            preds[r["id"]] = first_of(r.get("pred", ""))

    n = len(gold)
    correct = sum(preds.get(i) == g for i, g in gold.items())
    legal = sum(preds.get(i) is not None for i in gold)
    print(f"arm={a.arm}  n={n}  accuracy={correct/n*100:.2f}  legality={legal/n*100:.2f}")
    print(f"chance={CHANCE:.2f} (analytic)")
    print("reference: 19.80 ReGraph on control | 99.05 ReGraph on main")
    if a.arm == "control" and correct / n * 100 > CHANCE + 3:
        print("WARNING: control is above chance -- the generator leaks and main is void.")


if __name__ == "__main__":
    main()
