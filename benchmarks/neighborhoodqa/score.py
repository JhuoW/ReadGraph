"""Standalone scorer for NeighborhoodQA — no ReGraph dependency.

Usage: python score.py predictions.jsonl        # lines: {"id": ..., "pred": "..."}
       (gold is read from neighborhoodqa.jsonl.gz next to this file)
"""
from __future__ import annotations

import gzip
import json
import re
import string
import sys
from pathlib import Path

AREAS = ["Artificial Intelligence","Hardware Architecture","Computational Complexity",
 "Computational Engineering","Computational Geometry","Computation and Language",
 "Cryptography and Security","Computer Vision","Computers and Society","Databases",
 "Distributed and Cluster Computing","Digital Libraries","Discrete Mathematics",
 "Data Structures and Algorithms","Emerging Technologies","Formal Languages and Automata Theory",
 "General Literature","Graphics","Computer Science and Game Theory","Human-Computer Interaction",
 "Information Retrieval","Information Theory","Machine Learning","Logic in Computer Science",
 "Multiagent Systems","Multimedia","Mathematical Software","Numerical Analysis",
 "Neural and Evolutionary Computing","Networking and Internet Architecture",
 "Other Computer Science","Operating Systems","Performance","Programming Languages","Robotics",
 "Symbolic Computation","Sound","Software Engineering","Social and Information Networks",
 "Systems and Control"]


def norm(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def parse(text: str) -> set[str]:
    t = norm(text)
    return {a for a in AREAS if norm(a) in t}


def main() -> None:
    here = Path(__file__).parent
    gold = {}
    with gzip.open(here / "neighborhoodqa.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == "test":
                gold[r["id"]] = parse(r["answer"])

    preds = {}
    with open(sys.argv[1]) as f:
        for line in f:
            r = json.loads(line)
            preds[r["id"]] = parse(r.get("pred", ""))

    f1s, em = [], 0
    for i, g in gold.items():
        p = preds.get(i, set())
        inter = len(p & g)
        pr = inter / len(p) if p else 0.0
        rc = inter / len(g) if g else 0.0
        f1s.append(0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
        em += int(p == g)
    n = len(gold)
    print(f"n={n}  set_f1={sum(f1s)/n*100:.2f}  exact_set_match={em/n*100:.2f}")
    print("reference: 63.81 shortcut | 72.65 centre-only | 83.87 ReGraph")


if __name__ == "__main__":
    main()
