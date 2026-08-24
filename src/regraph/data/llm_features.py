"""Re-featurize a preprocessed dataset with LLM-embedding node features (+ optional numerics).

Motivation. `ReGraph.md` §2.1 fixes only that "every node can be mapped to a vector"; §3.2
picks `all-roberta-large-v1`. Our measurements say that choice is load-bearing in a bad way:
the output projection `W_O` must learn an sbert -> Llama map from answer likelihood alone, it
never leaves std 5e-4..2e-3, and an explicit alignment stage did not fix it
(docs/OPEN-QUESTIONS.md Q17/Q21). GRAFF (Findings EACL 2026) avoids the whole problem by
reusing *the LLM's own embeddings* for node features, "eliminating embedding space
misalignment", and reports 90.2 on SceneGraphs.

`--mode llm` sets c_v = mean over the frozen Llama input embeddings of the node's text tokens,
exactly GRAFF's Eq. 4. Only an embedding lookup is needed, no forward pass.

`--add-coords` additionally appends normalized (x, y, w, h, area, cx, cy) parsed out of the
SceneGraphs node text. sbert mean-pooling does not preserve numeric magnitude, so left/right
comparisons are unrecoverable from the text embedding — and spatial questions are the largest
SceneGraphs category (6,916 / 20,025) at 50.4%. §2.1 explicitly allows numerical attributes.

Usage:
  python -m regraph.data.llm_features --config configs/scene_graphs.yaml --mode llm --add-coords
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from regraph.data.attr_encoder import AttrStore, encoder_slug
from regraph.data.preprocess import store_dir_for
from regraph.utils.config import load_config

COORD_RE = re.compile(r"\(x,y,w,h\):\s*\((-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\)")
N_COORD = 7


def coord_features(text: str) -> np.ndarray:
    """Normalized (x, y, w, h, area, cx, cy); zeros when the text carries no box."""
    m = COORD_RE.search(text)
    if not m:
        return np.zeros(N_COORD, dtype=np.float32)
    x, y, w, h = (float(v) for v in m.groups())
    # GQA images are ~640x480; scale to roughly [0,1] without needing per-image metadata
    sx, sy = 640.0, 480.0
    return np.array(
        [x / sx, y / sy, w / sx, h / sy, (w * h) / (sx * sy),
         (x + w / 2) / sx, (y + h / 2) / sy], dtype=np.float32
    )


@torch.no_grad()
def llm_mean_embeddings(texts: list[str], llm_name: str, device: str, batch: int = 512):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(llm_name)
    if tok.pad_token_id is None:      # Llama-3.1 ships no pad token; padding is masked out anyway
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(llm_name, dtype=torch.bfloat16)
    emb = model.get_input_embeddings().weight.to(device)          # [V, d]
    d = emb.shape[1]
    out = np.empty((len(texts), d), dtype=np.float16)
    for lo in tqdm(range(0, len(texts), batch), desc="llm-emb", unit="batch"):
        chunk = texts[lo : lo + batch]
        enc = tok(chunk, add_special_tokens=False, truncation=True, max_length=256,
                  padding=True, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device).unsqueeze(-1).to(emb.dtype)
        vecs = (emb[ids] * mask).sum(1) / mask.sum(1).clamp(min=1)   # GRAFF Eq. 4
        out[lo : lo + len(chunk)] = vecs.float().cpu().numpy().astype(np.float16)
    del model, emb
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["llm", "keep"], default="llm")
    ap.add_argument("--add-coords", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--suffix", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    src = store_dir_for(cfg)
    suffix = args.suffix or ("llmemb_coords" if args.add_coords else "llmemb")
    dst = src.parent.parent / f"{cfg['dataset']['name']}_{suffix}" / encoder_slug(
        cfg["data"]["attr_encoder"]
    )
    dst.mkdir(parents=True, exist_ok=True)

    with open(src / "texts.json") as f:
        texts = json.load(f)
    print(f"[llm-feat] re-featurizing {len(texts):,} texts from {src}")

    if args.mode == "keep":
        # keep the existing (sbert) embeddings and only append the numeric features
        emb = np.load(src / "emb.f16.npy", mmap_mode="r")[:].astype(np.float16)
        print(f"[llm-feat] reusing existing {emb.shape[1]}-d embeddings")
    else:
        emb = llm_mean_embeddings(texts, cfg["llm"]["name"], args.device)
    if args.add_coords:
        coords = np.stack([coord_features(t) for t in texts]).astype(np.float16)
        n_box = int((np.abs(coords).sum(1) > 0).sum())
        print(f"[llm-feat] parsed a box for {n_box:,}/{len(texts):,} texts")
        emb = np.concatenate([emb, coords], axis=1)

    np.save(dst / "emb.f16.npy", emb)
    if (dst / "emb.f16.npy.npy").exists():
        (dst / "emb.f16.npy.npy").rename(dst / "emb.f16.npy")
    shutil.copy(src / "texts.json", dst / "texts.json")
    for f in ("graphs.pt", "examples.json"):
        if not (dst / f).exists():
            shutil.copy(src / f, dst / f)
    with open(dst / "meta.json", "w") as f:
        base = (cfg["data"]["attr_encoder"] if args.mode == "keep"
                else f"{cfg['llm']['name']}::mean-input-embedding")
        json.dump({"encoder": base + ("+coords" if args.add_coords else ""),
                   "num_texts": len(texts), "d_attr": int(emb.shape[1])}, f)
    print(f"[llm-feat] wrote {dst}  d_attr={emb.shape[1]}")


if __name__ == "__main__":
    main()
