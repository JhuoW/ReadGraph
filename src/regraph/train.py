"""Training (`ReGraph.md` §3.2-3.3, docs/components/07-training.md).

Answer-only next-token NLL; AdamW lr 1e-5, wd 0.05 (no decay on biases, norms,
b_base, role embedding); warm-up (5% of steps, OPEN-QUESTIONS Q5) + cosine decay;
<= 10 epochs, batch 4, grad clip 1.0, early stopping patience 2 on validation loss;
checkpoint = lowest validation loss. One model per dataset. No auxiliary losses, no
supervision on S / alpha / g.

Usage: python -m regraph.train --config configs/expla_graphs.yaml [run_name=...] [--resume]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from regraph.data.collate import make_collate_fn
from regraph.data.datasets import GraphQADataset
from regraph.model import ReGraph, load_regraph
from regraph.utils.config import load_config
from regraph.utils.runlog import RunDir
from regraph.utils.seeding import seed_everything

NO_DECAY_SUBSTRINGS = ("bias", "norm", "b_base", "role_emb")


def build_optimizer(model: ReGraph, cfg: dict) -> AdamW:
    """AdamW with the protocol's decay/no-decay split, plus optional per-module learning
    rate multipliers (`train.lr_mult`, all 1.0 by default so protocol runs are unchanged).

    The multipliers exist because every trainable module here is randomly initialized
    rather than fine-tuned, which `ReGraph.md`'s single lr=1e-5 does not account for —
    see docs/OPEN-QUESTIONS.md Q10/Q19.
    """
    base_lr = cfg["train"]["lr"]
    mults: dict = cfg["train"].get("lr_mult") or {}
    groups: dict[tuple[float, bool], list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        no_decay = any(tag in lname for tag in NO_DECAY_SUBSTRINGS) or p.ndim <= 1
        mult = 1.0
        for prefix, m in mults.items():
            if name == prefix or name.startswith(prefix + "."):
                mult = float(m)
                break
        groups.setdefault((mult, no_decay), []).append(p)

    param_groups = [
        {
            "params": params,
            "lr": base_lr * mult,
            "weight_decay": 0.0 if no_decay else cfg["train"]["weight_decay"],
        }
        for (mult, no_decay), params in sorted(groups.items(), key=lambda kv: kv[0])
    ]
    if mults:
        print(f"[train] lr {base_lr:g} with per-module multipliers {dict(mults)} -> "
              f"{len(param_groups)} param groups: "
              + ", ".join(f"{g['lr']:g}({len(g['params'])}t,wd={g['weight_decay']})"
                          for g in param_groups))
    return AdamW(param_groups, lr=base_lr)


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float) -> LambdaLR:
    warmup = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, lr_lambda)


def trainable_state_dict(model: ReGraph) -> dict:
    """Everything except the frozen LLM (reloaded from HF at eval time)."""
    return {k: v for k, v in model.state_dict().items() if not k.startswith("llm.")}


def load_trainable_state(model: ReGraph, state: dict) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected checkpoint keys: {unexpected[:5]}"
    assert all(k.startswith("llm.") for k in missing), (
        f"missing non-LLM keys: {[k for k in missing if not k.startswith('llm.')][:5]}"
    )


class DiagnosticsMeter:
    """Accumulates per-round gate / hop / entropy statistics over an epoch
    (07-training.md §7.4 — the fusion gate is the primary health signal)."""

    def __init__(self) -> None:
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}
        self.gate_lo: dict[int, int] = {}
        self.gate_hi: dict[int, int] = {}
        self.gate_n: dict[int, int] = {}

    def update(self, diag: dict) -> None:
        for t, gate in enumerate(diag.get("gates", [])):
            g = gate.float().reshape(-1)
            self._add(f"gate/round{t}", g.mean())
            self.gate_lo[t] = self.gate_lo.get(t, 0) + int((g < 0.1).sum())
            self.gate_hi[t] = self.gate_hi.get(t, 0) + int((g > 0.9).sum())
            self.gate_n[t] = self.gate_n.get(t, 0) + g.numel()
        for t, alpha in enumerate(diag.get("alphas", [])):
            # [B, heads, N_B, K+1] -> mean hop distribution
            self._add(f"alpha/round{t}", alpha.float().mean(dim=(0, 1, 2)))
        for key in ("s0_entropy", "s_tilde_entropy"):
            for t, val in enumerate(diag.get(key, [])):
                self._add(f"{key}/round{t}", val.float())

    def _add(self, key: str, val: torch.Tensor) -> None:
        val = val.detach().cpu()
        if key not in self.sums:
            self.sums[key] = torch.zeros_like(val)
            self.counts[key] = 0
        self.sums[key] += val
        self.counts[key] += 1

    def summary(self) -> dict:
        out = {}
        for key, s in self.sums.items():
            mean = (s / self.counts[key])
            out[key] = mean.item() if mean.ndim == 0 else [round(v, 4) for v in mean.tolist()]
        for t in self.gate_n:
            out[f"gate_frac_lt0.1/round{t}"] = round(self.gate_lo[t] / self.gate_n[t], 4)
            out[f"gate_frac_gt0.9/round{t}"] = round(self.gate_hi[t] / self.gate_n[t], 4)
        return out


@torch.no_grad()
def evaluate_loss(model: ReGraph, loader: DataLoader) -> tuple[float, dict]:
    model.eval()
    total, n = 0.0, 0
    meter = DiagnosticsMeter()
    for batch in loader:
        out = model(batch)
        total += out.loss.item()
        n += 1
        meter.update(out.diagnostics)
    return total / max(n, 1), meter.summary()


def make_loaders(cfg: dict, model: ReGraph) -> tuple[DataLoader, DataLoader]:
    collate = make_collate_fn(
        pad_token_id=model.tokenizer.pad_token_id,
        placeholder_id=model.placeholder_id,
        num_query_tokens=cfg["model"]["num_query_tokens"],
        symmetrize_for_diffusion=cfg["graph"]["symmetrize_for_diffusion"],
        add_self_loops=cfg["graph"]["add_self_loops"],
    )
    train_ds = GraphQADataset(cfg, "train", model.tokenizer, mode="train")
    val_ds = GraphQADataset(cfg, "val", model.tokenizer, mode="train")
    gen = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, generator=gen,
        collate_fn=collate, num_workers=cfg["train"].get("num_workers", 2),
        drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        collate_fn=collate, num_workers=cfg["train"].get("num_workers", 2),
        pin_memory=True,
    )
    return train_loader, val_loader


def train(
    cfg: dict, device: str = "cuda", resume: bool = False, init_from: str | None = None
) -> Path:
    seed_everything(cfg["seed"])
    run = RunDir(cfg["run_dir"], cfg, resume=resume)
    model = load_regraph(cfg, device=device)

    if init_from:
        # warm-start the graph-side modules from a previous stage (e.g. the alignment
        # pretraining that teaches W_O a decodable graph->LLM map). The frozen LLM is
        # untouched; only trainable weights are loaded.
        ck = torch.load(init_from, map_location=device, weights_only=False)
        load_trainable_state(model, ck["model"])
        print(f"[train] initialized graph-side weights from {init_from} "
              f"(epoch {ck.get('epoch')}, val {ck.get('val_loss')})")

    n_train = model.num_trainable_parameters()
    print(f"[train] group_bounds={model.group_bounds}")
    print(f"[train] trainable params: {n_train / 1e6:.2f}M (expect ≈ 28M, 06-model.md §6.3)")
    assert 5e6 < n_train < 100e6, "trainable parameter count far outside expected range"

    train_loader, val_loader = make_loaders(cfg, model)
    tcfg = cfg["train"]
    accum = tcfg.get("grad_accum", 1)
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = steps_per_epoch * tcfg["max_epochs"]
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, total_steps, tcfg["warmup_ratio"])
    print(
        f"[train] {len(train_loader)} micro-batches/epoch, accum={accum}, "
        f"total_steps={total_steps}, micro-batch={tcfg['batch_size']}, "
        f"effective batch={tcfg['batch_size'] * accum}"
    )

    start_epoch, best_val, best_epoch, global_step = 0, float("inf"), -1, 0
    last_path, best_path = run.file("last.pt"), run.file("best.pt")
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        load_trainable_state(model, ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_val, best_epoch, global_step = ck["best_val"], ck["best_epoch"], ck["step"]
        torch.set_rng_state(ck["rng"]["cpu"])
        if torch.cuda.is_available() and ck["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        print(f"[train] resumed at epoch {start_epoch} (best_val={best_val:.4f})")

    for epoch in range(start_epoch, tcfg["max_epochs"]):
        model.train()
        meter = DiagnosticsMeter()
        epoch_loss, t0 = 0.0, time.time()
        optimizer.zero_grad(set_to_none=True)
        for micro_step, batch in enumerate(train_loader):
            out = model(batch)
            (out.loss / accum).backward()
            epoch_loss += out.loss.item()
            meter.update(out.diagnostics)
            if (micro_step + 1) % accum == 0 or micro_step == len(train_loader) - 1:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.trainable_parameters(), tcfg["grad_clip"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % tcfg.get("log_every", 10) == 0:
                    run.metrics.log(
                        {
                            "kind": "step", "step": global_step, "epoch": epoch,
                            "loss": out.loss.item(),
                            "lr": scheduler.get_last_lr()[0],
                            "grad_norm": float(grad_norm),
                        }
                    )

        train_diag = meter.summary()
        val_loss, val_diag = evaluate_loss(model, val_loader)
        epoch_rec = {
            "kind": "epoch", "epoch": epoch,
            "train_loss": epoch_loss / max(len(train_loader), 1),
            "val_loss": val_loss,
            "seconds": round(time.time() - t0, 1),
            "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if torch.cuda.is_available() else 0.0,
            "train_diag": train_diag, "val_diag": val_diag,
        }
        run.metrics.log(epoch_rec)
        print(
            f"[epoch {epoch}] train {epoch_rec['train_loss']:.4f} | val {val_loss:.4f} | "
            f"gates " + " ".join(
                f"r{t}={train_diag.get(f'gate/round{t}', float('nan')):.3f}"
                for t in range(model.num_rounds)
            )
        )

        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            torch.save(
                {"model": trainable_state_dict(model), "epoch": epoch,
                 "val_loss": val_loss, "config": cfg},
                best_path,
            )
            print(f"[epoch {epoch}] new best (val {val_loss:.4f}) -> {best_path}")
        torch.save(
            {
                "model": trainable_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch, "step": global_step,
                "best_val": best_val, "best_epoch": best_epoch,
                "rng": {
                    "cpu": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None,
                },
            },
            last_path,
        )

        if epoch - best_epoch >= tcfg["early_stopping_patience"]:
            print(f"[train] early stop at epoch {epoch} (best epoch {best_epoch})")
            break

    with open(run.file("train_summary.json"), "w") as f:
        json.dump({"best_val": best_val, "best_epoch": best_epoch, "steps": global_step}, f)
    print(f"[train] done. best val loss {best_val:.4f} at epoch {best_epoch}")
    return best_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", default=None,
                    help="checkpoint whose graph-side weights warm-start this run")
    ap.add_argument("overrides", nargs="*", help="dotted config overrides key=value")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    train(cfg, device=args.device, resume=args.resume, init_from=args.init_from)


if __name__ == "__main__":
    main()
