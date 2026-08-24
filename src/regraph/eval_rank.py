"""Top-k accuracy for the arXiv benchmark by likelihood ranking.

GraphTranslator reports Top-1/3/5 on ArXiv. ReGraph generates free text, so Top-k is obtained
by scoring each of the 40 category names as the continuation and ranking them by the model's
answer-token log-likelihood — the standard way to get a ranking out of a generative model
without a classification head (`ReGraph.md` §1 forbids one).

Both the summed and the length-normalized log-likelihood are reported: category names differ
in token length, and sum-log-prob mildly favours short names.

Usage:
  python -m regraph.eval_rank --config configs/arxiv.yaml --ckpt runs/arxiv/control/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from regraph.data.arxiv_raw import ARXIV_CS_CATEGORIES
from regraph.data.collate import make_collate_fn
from regraph.data.datasets import GraphQADataset
from regraph.model import load_regraph
from regraph.utils.config import load_config
from regraph.utils.seeding import seed_everything


@torch.no_grad()
def rank_eval(cfg: dict, ckpt: str, split: str = "test", device: str = "cuda",
              limit: int | None = None) -> dict:
    from regraph.train import load_trainable_state

    seed_everything(cfg["seed"])
    model = load_regraph(cfg, device=device)
    load_trainable_state(model, torch.load(ckpt, map_location=device, weights_only=False)["model"])
    model.eval()

    categories = list(ARXIV_CS_CATEGORIES.values())
    cand_ids = [
        model.tokenizer(f" {c}", add_special_tokens=False)["input_ids"] + [model.tokenizer.eos_token_id]
        for c in categories
    ]

    ds = GraphQADataset(cfg, split, model.tokenizer, mode="eval")
    if limit:
        ds = torch.utils.data.Subset(ds, range(min(limit, len(ds))))
    collate = make_collate_fn(
        model.tokenizer.pad_token_id, model.placeholder_id, cfg["model"]["num_query_tokens"],
        cfg["graph"]["symmetrize_for_diffusion"], cfg["graph"]["add_self_loops"],
    )

    hits = {1: 0, 3: 0, 5: 0}
    hits_norm = {1: 0, 3: 0, 5: 0}
    n = 0
    for item in tqdm(ds, desc="rank-eval"):
        # one batch = the same example paired with all 40 candidate answers
        variants = []
        for ids in cand_ids:
            v = dict(item)
            v["answer_ids"] = torch.tensor(ids, dtype=torch.long)
            variants.append(v)
        batch = collate(variants)
        out = model(batch)
        logits = out.logits[:, :-1].float()
        labels = batch["labels"][:, 1:].to(logits.device)
        mask = labels != -100
        lp = torch.gather(
            F.log_softmax(logits, -1), 2, labels.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)
        total = (lp * mask).sum(-1)                     # summed log-likelihood
        mean = total / mask.sum(-1).clamp(min=1)        # length-normalized

        gold = categories.index(item["answer"])
        for scores, acc in ((total, hits), (mean, hits_norm)):
            order = scores.argsort(descending=True).tolist()
            for k in (1, 3, 5):
                acc[k] += int(gold in order[:k])
        n += 1

    res = {f"top{k}": hits[k] / n for k in (1, 3, 5)}
    res.update({f"top{k}_lennorm": hits_norm[k] / n for k in (1, 3, 5)})
    res["num_examples"] = n
    out_path = Path(ckpt).parent / f"metrics_rank_{split}.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[rank-eval] {json.dumps(res)}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/arxiv.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    rank_eval(load_config(args.config, args.overrides), args.ckpt,
              split=args.split, device=args.device, limit=args.limit)


if __name__ == "__main__":
    main()
