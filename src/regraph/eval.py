"""Evaluation (docs/components/09-evaluation.md).

Metric logic is **ported from G-Retriever** `src/utils/evaluate.py` (He et al., 2024,
https://github.com/XiaoxinHe/G-Retriever) — same normalization, same matching — so
numbers are comparable with the published protocol. Ported functions are named after
their source. The only change: they operate on in-memory records instead of a CSV/JSON
path, and get_accuracy_webqsp returns Hit@1 plus the auxiliary scores.

Usage:
  python -m regraph.eval --config configs/expla_graphs.yaml --ckpt runs/.../best.pt \
      [--split test] [--dump-readings 5]
"""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from regraph.data.collate import make_collate_fn
from regraph.data.datasets import GraphQADataset
from regraph.model import ReGraph, load_regraph
from regraph.utils.config import load_config
from regraph.utils.seeding import seed_everything

# --------------------------------------------------------------------------- #
# G-Retriever src/utils/evaluate.py ports                                     #
# --------------------------------------------------------------------------- #


def get_accuracy_expla_graphs(records: list[dict]) -> float:
    """Port of G-Retriever evaluate.py::get_accuracy_expla_graphs."""
    correct = 0
    for r in records:
        matches = re.findall(r"support|Support|Counter|counter", r["pred"].strip())
        if len(matches) > 0 and matches[0].lower() == r["label"]:
            correct += 1
    return correct / len(records)


def get_accuracy_gqa(records: list[dict]) -> float:
    """Port of G-Retriever evaluate.py::get_accuracy_gqa (SceneGraphs accuracy)."""
    correct = 0
    for r in records:
        if r["label"] in r["pred"]:
            correct += 1
    return correct / len(records)


def normalize(s: str) -> str:
    """Port of G-Retriever evaluate.py::normalize."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    """Port of G-Retriever evaluate.py::match."""
    return normalize(s2) in normalize(s1)


def eval_f1(prediction: list[str], answer: list[str]) -> tuple[float, float, float]:
    """Port of G-Retriever evaluate.py::eval_f1."""
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = " ".join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    return 2 * precision * recall / (precision + recall), precision, recall


def eval_acc(prediction: str, answer: list[str]) -> float:
    """Port of G-Retriever evaluate.py::eval_acc."""
    matched = 0.0
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)


def eval_hit(prediction: str, answer: list[str]) -> int:
    """Port of G-Retriever evaluate.py::eval_hit."""
    for a in answer:
        if match(prediction, a):
            return 1
    return 0


def get_accuracy_webqsp(records: list[dict]) -> dict:
    """Port of G-Retriever evaluate.py::get_accuracy_webqsp. Primary metric: Hit@1."""
    acc_list, hit_list, f1_list, prec_list, rec_list = [], [], [], [], []
    for r in records:
        prediction = r["pred"].replace("|", "\n").split("\n")
        answer = r["label"].split("|")
        f1, prec, rec = eval_f1(prediction, answer)
        f1_list.append(f1)
        prec_list.append(prec)
        rec_list.append(rec)
        prediction_str = " ".join(prediction)
        acc_list.append(eval_acc(prediction_str, answer))
        hit_list.append(eval_hit(prediction_str, answer))
    n = len(records)
    return {
        "hit@1": sum(hit_list) / n,
        "accuracy": sum(acc_list) / n,
        "precision": sum(prec_list) / n,
        "recall": sum(rec_list) / n,
        "f1": sum(f1_list) / n,
    }


def _first_arxiv_category(pred: str) -> str | None:
    """Earliest-mentioned arXiv CS category in the prediction, or None if it names none.

    Mirrors GraphTranslator's Legality Rate (did the model emit a valid category at all)
    and G-Retriever's "first match wins" convention for ExplaGraphs.
    """
    from regraph.data.arxiv_raw import ARXIV_CS_CATEGORIES

    p = normalize(pred)
    best, best_at = None, len(p) + 1
    for name in ARXIV_CS_CATEGORIES.values():
        at = p.find(normalize(name))
        if at != -1 and at < best_at:
            best, best_at = name, at
    return best


def get_accuracy_arxiv(records: list[dict]) -> dict:
    """Top-1 accuracy over the 40 arXiv CS categories, plus legality rate."""
    correct = sum(_first_arxiv_category(r["pred"]) == r["label"] for r in records)
    legal = sum(_first_arxiv_category(r["pred"]) is not None for r in records)
    return {"accuracy": correct / len(records), "legality_rate": legal / len(records)}


CORA_CLASSES = ["Case Based", "Genetic Algorithms", "Neural Networks", "Probabilistic Methods",
                "Reinforcement Learning", "Rule Learning", "Theory"]


def _first_of(pred: str, options: list[str]) -> str | None:
    """Earliest-mentioned option in the prediction (first-match-wins), or None."""
    p = normalize(pred)
    best, at_best = None, len(p) + 1
    for o in options:
        at = p.find(normalize(o))
        if at != -1 and at < at_best:
            best, at_best = o, at
    return best


PUBMED_CLASSES = ["Diabetes Mellitus Experimental", "Diabetes Mellitus Type 1",
                  "Diabetes Mellitus Type 2"]
_CLASS_SETS = {"cora": CORA_CLASSES, "pubmed": PUBMED_CLASSES}


def _load_classes(dataset_name: str, cfg: dict | None = None) -> list[str] | None:
    """Class vocabulary for datasets whose label set is written at preprocessing time."""
    if dataset_name in _CLASS_SETS:
        return _CLASS_SETS[dataset_name]
    if cfg is not None:
        from regraph.data.preprocess import store_dir_for
        f = store_dir_for(cfg) / "classes.json"
        if f.exists():
            return json.loads(f.read_text())
    return None


def get_accuracy_classes(records: list[dict], options: list[str]) -> dict:
    """First-match-wins accuracy over a fixed label vocabulary, plus legality rate."""
    correct = sum(_first_of(r["pred"], options) == r["label"] for r in records)
    legal = sum(_first_of(r["pred"], options) is not None for r in records)
    return {"accuracy": correct / len(records), "legality_rate": legal / len(records)}


def get_setf1_arxiv_areas(records: list[dict]) -> dict:
    """Set-F1 for NeighborhoodQA: the answer is a *set* of arXiv areas, not one label.

    Prediction and gold are parsed into sets of category names (substring match over the 40
    names, so ordering and separators do not matter); per-example F1 is averaged.
    """
    from regraph.data.arxiv_raw import ARXIV_CS_CATEGORIES
    names = list(ARXIV_CS_CATEGORIES.values())

    def parse(text: str) -> set[str]:
        t = normalize(text)
        return {n for n in names if normalize(n) in t}

    f1s, ems, empty = [], 0, 0
    for r in records:
        pred, gold = parse(r["pred"]), parse(r["label"])
        if not pred:
            empty += 1
        inter = len(pred & gold)
        p = inter / len(pred) if pred else 0.0
        rc = inter / len(gold) if gold else 0.0
        f1s.append(0.0 if p + rc == 0 else 2 * p * rc / (p + rc))
        ems += int(pred == gold)
    n = len(records)
    return {"set_f1": sum(f1s) / n, "exact_set_match": ems / n,
            "legality_rate": 1 - empty / n}


def _first_yes_no(pred: str) -> str | None:
    """Earliest of yes/no named in the prediction, mirroring NLGraph's true/false scoring."""
    p = normalize(pred)
    hits = [(p.find(w), w) for w in ("yes", "no") if f" {w} " in f" {p} "]
    return min(hits)[1] if hits else None


def get_accuracy_yes_no(records: list[dict]) -> dict:
    """Binary accuracy for NLGraph connectivity / cycle (random baseline = 50)."""
    correct = sum(_first_yes_no(r["pred"]) == r["label"] for r in records)
    legal = sum(_first_yes_no(r["pred"]) is not None for r in records)
    return {"accuracy": correct / len(records), "legality_rate": legal / len(records)}


def per_example_correct(dataset_name: str, pred: str, label: str,
                        classes: list[str] | None = None) -> bool:
    """Row-level correctness flag for the JSONL dump, matching the ported metrics."""
    if dataset_name == "expla_graphs":
        matches = re.findall(r"support|Support|Counter|counter", pred.strip())
        return len(matches) > 0 and matches[0].lower() == label
    if dataset_name == "scene_graphs":
        return label in pred
    if dataset_name == "webqsp":
        prediction = pred.replace("|", "\n").split("\n")
        return bool(eval_hit(" ".join(prediction), label.split("|")))
    if dataset_name == "arxiv":
        return _first_arxiv_category(pred) == label
    if dataset_name.startswith("nlgraph_"):
        return _first_yes_no(pred) == label
    if dataset_name == "arxiv_nbrqa":
        return get_setf1_arxiv_areas([{"pred": pred, "label": label}])["exact_set_match"] == 1.0
    options = classes or _CLASS_SETS.get(dataset_name)
    if options:
        return _first_of(pred, options) == label
    raise KeyError(dataset_name)


def compute_metric(dataset_name: str, records: list[dict],
                   classes: list[str] | None = None) -> dict:
    if dataset_name == "expla_graphs":
        return {"accuracy": get_accuracy_expla_graphs(records)}
    if dataset_name == "scene_graphs":
        return {"accuracy": get_accuracy_gqa(records)}
    if dataset_name == "webqsp":
        return get_accuracy_webqsp(records)
    if dataset_name == "arxiv":
        return get_accuracy_arxiv(records)
    if dataset_name.startswith("nlgraph_"):
        return get_accuracy_yes_no(records)
    if dataset_name == "arxiv_nbrqa":
        return get_setf1_arxiv_areas(records)
    options = classes or _CLASS_SETS.get(dataset_name)
    if options:
        return get_accuracy_classes(records, options)
    raise KeyError(dataset_name)


# --------------------------------------------------------------------------- #
# Runner                                                                      #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def run_eval(
    cfg: dict,
    ckpt_path: str | Path,
    split: str = "test",
    device: str = "cuda",
    dump_readings: int = 0,
    limit: int | None = None,
    out_dir: Path | None = None,
) -> dict:
    from regraph.train import load_trainable_state  # avoid cycle at import time

    seed_everything(cfg["seed"])
    model = load_regraph(cfg, device=device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    load_trainable_state(model, ck["model"])
    model.eval()
    print(f"[eval] loaded {ckpt_path} (epoch {ck.get('epoch')}, val {ck.get('val_loss')})")

    # `eval_as` lets a re-featurized variant reuse its parent dataset's metric
    name = cfg["dataset"].get("eval_as", cfg["dataset"]["name"])
    classes = _load_classes(name, cfg)
    ds = GraphQADataset(cfg, split, model.tokenizer, mode="eval")
    n_expected = len(ds)
    if limit is not None:
        ds = torch.utils.data.Subset(ds, range(min(limit, len(ds))))
    collate = make_collate_fn(
        pad_token_id=model.tokenizer.pad_token_id,
        placeholder_id=model.placeholder_id,
        num_query_tokens=cfg["model"]["num_query_tokens"],
        symmetrize_for_diffusion=cfg["graph"]["symmetrize_for_diffusion"],
        add_self_loops=cfg["graph"]["add_self_loops"],
    )
    loader = DataLoader(
        ds, batch_size=cfg["eval"]["batch_size"], shuffle=False, collate_fn=collate,
        num_workers=2, pin_memory=True,
    )

    max_new = cfg["eval"]["max_new_tokens"]
    assert not cfg["eval"].get("do_sample", False), "protocol is greedy decoding"

    records = []
    for batch in tqdm(loader, desc=f"eval:{name}:{split}"):
        out = model.generate(batch, max_new_tokens=max_new, collect_diagnostics=True)
        diag = out.get("diagnostics", {})
        gates = [g.float().mean(dim=(1, 2)).cpu() for g in diag.get("gates", [])]
        alphas = [a.float().mean(dim=(1, 2)).cpu() for a in diag.get("alphas", [])]
        for i in range(len(out["texts"])):
            records.append(
                {
                    "id": batch["id"][i],
                    "question": batch["question"][i],
                    "pred": out["texts"][i],
                    "label": batch["answer"][i],
                    "correct": per_example_correct(name, out["texts"][i], batch["answer"][i], classes),
                    "gate_per_round": [round(g[i].item(), 4) for g in gates],
                    "hop_dist_per_round": [
                        [round(v, 4) for v in a[i].tolist()] for a in alphas
                    ],
                }
            )

    metrics = compute_metric(name, records, classes)
    metrics["num_examples"] = len(records)
    metrics["expected_split_size"] = n_expected
    metrics["nonempty_frac"] = sum(bool(r["pred"]) for r in records) / len(records)
    metrics["split"] = split

    out_dir = Path(out_dir) if out_dir else Path(ckpt_path).parent
    with open(out_dir / f"predictions_{split}.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / f"metrics_{split}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] {name} {split}: {json.dumps(metrics)}")

    if dump_readings > 0:
        dump_reading_distributions(cfg, model, split, dump_readings, out_dir)
    return metrics


@torch.no_grad()
def dump_reading_distributions(
    cfg: dict, model: ReGraph, split: str, n_examples: int, out_dir: Path
) -> None:
    """Qualitative dump (09-evaluation.md §9.4): top-5 nodes per round with mass,
    plus hop weights — shows whether iterative reading moves attention across rounds."""
    ds = GraphQADataset(cfg, split, model.tokenizer, mode="eval", keep_node_texts=True)
    collate = make_collate_fn(
        pad_token_id=model.tokenizer.pad_token_id,
        placeholder_id=model.placeholder_id,
        num_query_tokens=cfg["model"]["num_query_tokens"],
        symmetrize_for_diffusion=cfg["graph"]["symmetrize_for_diffusion"],
        add_self_loops=cfg["graph"]["add_self_loops"],
    )
    dumps = []
    for idx in range(min(n_examples, len(ds))):
        batch = collate([ds[idx]])
        out = model.generate(
            batch, max_new_tokens=cfg["eval"]["max_new_tokens"],
            collect_diagnostics=True, return_reading=True,
        )
        node_texts = batch["node_texts"][0]
        rounds = []
        for t, s in enumerate(out["diagnostics"]["s_tilde"]):
            mass = s[0].float().mean(dim=(0, 1))            # mean over heads+tokens
            top = torch.topk(mass, k=min(5, mass.shape[0]))
            rounds.append(
                {
                    "round": t,
                    "top_nodes": [
                        {"node": node_texts[j] if j < len(node_texts) else "<pad>",
                         "mass": round(v, 4)}
                        for v, j in zip(top.values.tolist(), top.indices.tolist())
                    ],
                    "hop_dist": [
                        round(v, 4)
                        for v in out["diagnostics"]["alphas"][t][0].float()
                        .mean(dim=(0, 1)).tolist()
                    ],
                    "gate": round(out["diagnostics"]["gates"][t][0].float().mean().item(), 4),
                }
            )
        dumps.append(
            {
                "id": batch["id"][0],
                "question": batch["question"][0],
                "gold": batch["answer"][0],
                "pred": out["texts"][0],
                "rounds": rounds,
            }
        )
    with open(out_dir / f"readings_{split}.json", "w") as f:
        json.dump(dumps, f, indent=2)
    print(f"[eval] wrote reading distributions for {len(dumps)} examples")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dump-readings", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("overrides", nargs="*", help="dotted config overrides key=value")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    run_eval(
        cfg, args.ckpt, split=args.split, device=args.device,
        dump_readings=args.dump_readings, limit=args.limit,
    )


if __name__ == "__main__":
    main()
