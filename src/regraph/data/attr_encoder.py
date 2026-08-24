"""Attribute encoding: c_v = f_attr(x_v) (`ReGraph.md` §2.1).

Encoder: `sentence-transformers/all-roberta-large-v1`, frozen, pre-computed once and
cached to disk (docs/components/01-data.md). The mean-pooling + L2-normalization is a
port of G-Retriever src/utils/lm_modeling.py::{Sentence_Transformer, sber_text2embedding}
so node/edge features match theirs bit-for-bit in expectation.

Texts are deduplicated before encoding and stored as a single fp16 memmap
(`emb.f16.npy`, one row per unique text) plus per-graph int32 id arrays — WebQSP's
full graphs would otherwise cost ~100 GB. Recorded in docs/OPEN-QUESTIONS.md (Q13).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def encoder_slug(name: str) -> str:
    """Cache key includes the encoder name so a change invalidates the cache."""
    return name.replace("/", "__")


@torch.no_grad()
def encode_texts(
    texts: list[str],
    encoder_name: str,
    device: str = "cuda",
    batch_size: int = 256,
    d_attr: int = 1024,
) -> torch.Tensor:
    """Mean-pooled, L2-normalized sentence embeddings, fp32 on CPU. [len(texts), d_attr]."""
    if len(texts) == 0:
        return torch.zeros((0, d_attr))
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model = AutoModel.from_pretrained(encoder_name).to(device).eval()

    # Sort by length so batches pad tightly; restore original order at the end.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = torch.empty((len(texts), d_attr), dtype=torch.float32)
    for lo in tqdm(range(0, len(texts), batch_size), desc="sbert", unit="batch"):
        idx = order[lo : lo + batch_size]
        enc = tokenizer(
            [texts[i] for i in idx], padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        token_emb = model(**enc)[0]  # [b, L, d]
        mask = enc["attention_mask"].unsqueeze(-1).to(token_emb.dtype)
        emb = (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = F.normalize(emb, p=2, dim=1)
        out[torch.tensor(idx)] = emb.float().cpu()
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


class AttrStore:
    """Deduplicated attribute-embedding store.

    Layout under `store_dir`:
      meta.json     {"encoder": ..., "num_texts": U, "d_attr": 1024}
      texts.json    list[str], row-aligned with the embedding matrix
      emb.f16.npy   float16 [U, d_attr], memory-mapped at load
    """

    def __init__(self, store_dir: Path):
        self.dir = Path(store_dir)
        with open(self.dir / "meta.json") as f:
            self.meta = json.load(f)
        self.emb = np.load(self.dir / "emb.f16.npy", mmap_mode="r")
        assert self.emb.shape[0] == self.meta["num_texts"]
        self._texts: list[str] | None = None

    @property
    def texts(self) -> list[str]:
        if self._texts is None:
            with open(self.dir / "texts.json") as f:
                self._texts = json.load(f)
        return self._texts

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """fp32 embeddings for int id tensor of any shape -> [..., d_attr]."""
        d_attr = self.meta["d_attr"]
        if ids.numel() == 0:  # e.g. scene graphs with zero edges
            return torch.zeros(*ids.shape, d_attr, dtype=torch.float32)
        arr = self.emb[ids.reshape(-1).numpy()]
        return torch.from_numpy(arr.astype(np.float32)).reshape(*ids.shape, d_attr)

    @staticmethod
    def build(
        store_dir: Path,
        texts: list[str],
        encoder_name: str,
        device: str = "cuda",
        batch_size: int = 256,
        d_attr: int = 1024,
    ) -> "AttrStore":
        store_dir = Path(store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)
        emb = encode_texts(texts, encoder_name, device=device, batch_size=batch_size, d_attr=d_attr)
        np.save(store_dir / "emb.f16.npy", emb.numpy().astype(np.float16))
        # np.save appends .npy if missing; normalize the filename
        if (store_dir / "emb.f16.npy.npy").exists():
            (store_dir / "emb.f16.npy.npy").rename(store_dir / "emb.f16.npy")
        with open(store_dir / "texts.json", "w") as f:
            json.dump(texts, f)
        with open(store_dir / "meta.json", "w") as f:
            json.dump({"encoder": encoder_name, "num_texts": len(texts), "d_attr": d_attr}, f)
        return AttrStore(store_dir)


class TextInterner:
    """Assigns stable int ids to unique texts."""

    def __init__(self) -> None:
        self.text2id: dict[str, int] = {}

    def intern(self, texts: list[str]) -> torch.Tensor:
        ids = []
        for t in texts:
            t = "" if t is None else str(t)
            if t not in self.text2id:
                self.text2id[t] = len(self.text2id)
            ids.append(self.text2id[t])
        return torch.tensor(ids, dtype=torch.int32)

    @property
    def texts(self) -> list[str]:
        return list(self.text2id.keys())
