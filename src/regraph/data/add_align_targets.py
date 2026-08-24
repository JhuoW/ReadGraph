"""Add the alignment-stage target (the centre paper's title) to preprocessed arXiv examples.

Node text is stored as "title. abstract", so the title is recoverable from the attribute
store without re-encoding anything. Run once after `preprocess_arxiv`.

Usage: python -m regraph.data.add_align_targets --config configs/arxiv.yaml
"""

from __future__ import annotations

import argparse
import json

import torch

from regraph.data.attr_encoder import AttrStore
from regraph.data.preprocess import store_dir_for
from regraph.utils.config import load_config

MAX_TITLE_CHARS = 300


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/arxiv.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    store_dir = store_dir_for(cfg)
    store = AttrStore(store_dir)
    graphs = torch.load(store_dir / "graphs.pt", weights_only=True)
    with open(store_dir / "examples.json") as f:
        examples = json.load(f)

    texts = store.texts
    n_empty = 0
    for ex in examples:
        centre = int(graphs[ex["graph_key"]]["node_text_id"][0])
        title = texts[centre].split(". ", 1)[0].strip()[:MAX_TITLE_CHARS]
        if not title:
            n_empty += 1
        ex["align_target"] = title

    with open(store_dir / "examples.json", "w") as f:
        json.dump(examples, f)
    lens = [len(ex["align_target"].split()) for ex in examples]
    print(f"[align] wrote align_target for {len(examples):,} examples "
          f"(mean {sum(lens)/len(lens):.1f} words, {n_empty} empty)")
    print(f"[align] sample: {examples[0]['align_target']!r}")


if __name__ == "__main__":
    main()
