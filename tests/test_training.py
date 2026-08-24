"""07-training.md acceptance tests: masking, determinism, optimizer grouping, overfit."""

import torch

from conftest import make_batch, make_model
from regraph.train import build_optimizer, build_scheduler, load_trainable_state, trainable_state_dict


def test_label_masking_all_ignored_gives_zero_loss(model, batch):
    b = dict(batch)
    b["labels"] = torch.full_like(batch["labels"], -100)
    out = model(b)
    assert out.loss.item() == 0.0
    out.loss.backward()  # must stay differentiable


def test_determinism_same_seed_same_losses():
    losses = []
    for _ in range(2):
        torch.manual_seed(123)
        model = make_model(seed=7)
        model.train()
        batch = make_batch(seed=5)
        opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
        run = []
        for _ in range(5):
            opt.zero_grad()
            loss = model(batch).loss
            loss.backward()
            opt.step()
            run.append(loss.item())
        losses.append(run)
    assert losses[0] == losses[1]


def test_optimizer_no_decay_groups():
    model = make_model()
    cfg = model.cfg
    opt = build_optimizer(model, cfg)
    assert opt.param_groups[0]["weight_decay"] == cfg["train"]["weight_decay"]
    assert opt.param_groups[1]["weight_decay"] == 0.0
    no_decay_params = {id(p) for p in opt.param_groups[1]["params"]}
    named = dict(model.named_parameters())
    assert id(named["b_base"]) in no_decay_params
    assert id(named["role_emb.emb.weight"]) in no_decay_params
    for n, p in named.items():
        if p.requires_grad and ("bias" in n or "ln" in n or "norm" in n.lower()):
            assert id(p) in no_decay_params, n
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_opt == model.num_trainable_parameters()


def test_scheduler_warmup_then_cosine():
    model = make_model()
    opt = build_optimizer(model, model.cfg)
    sched = build_scheduler(opt, total_steps=100, warmup_ratio=0.1)
    lrs = []
    for _ in range(100):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    assert lrs[0] == 0.0
    assert lrs[10] == max(lrs)                     # warmup peak at 10% of steps
    assert lrs[-1] < lrs[10] * 0.01                # cosine decayed to ~0


def test_checkpoint_round_trip_excludes_llm(model, batch):
    state = trainable_state_dict(model)
    assert all(not k.startswith("llm.") for k in state)
    # same frozen LLM (same seed), perturbed trainables — the real checkpoint story
    model2 = make_model(seed=0)
    with torch.no_grad():
        model2.b_base.add_(1.0)
        model2.reader.w_o.weight.add_(0.5)
    load_trainable_state(model2, state)
    with torch.no_grad():
        l1 = model(batch).loss
        l2 = model2(batch).loss
    assert torch.allclose(l1, l2, atol=1e-6)


def test_resume_matches_uninterrupted(tmp_path):
    """Save/load mid-training -> next-step loss matches the continuous run."""

    def steps(model, opt, batch, n):
        out = []
        for _ in range(n):
            opt.zero_grad()
            loss = model(batch).loss
            loss.backward()
            opt.step()
            out.append(loss.item())
        return out

    batch = make_batch(seed=5)

    torch.manual_seed(0)
    model_a = make_model(seed=7)
    model_a.train()
    opt_a = torch.optim.AdamW(model_a.trainable_parameters(), lr=1e-3)
    losses_a = steps(model_a, opt_a, batch, 4)

    torch.manual_seed(0)
    model_b = make_model(seed=7)
    model_b.train()
    opt_b = torch.optim.AdamW(model_b.trainable_parameters(), lr=1e-3)
    steps(model_b, opt_b, batch, 2)
    ck = tmp_path / "mid.pt"
    torch.save({"model": trainable_state_dict(model_b), "opt": opt_b.state_dict()}, ck)

    torch.manual_seed(0)
    model_c = make_model(seed=7)
    model_c.train()
    opt_c = torch.optim.AdamW(model_c.trainable_parameters(), lr=1e-3)
    loaded = torch.load(ck, weights_only=False)
    load_trainable_state(model_c, loaded["model"])
    opt_c.load_state_dict(loaded["opt"])
    losses_c = steps(model_c, opt_c, batch, 2)
    assert abs(losses_c[0] - losses_a[2]) < 1e-6
    assert abs(losses_c[1] - losses_a[3]) < 1e-6


def test_tiny_overfit_loss_decreases():
    """End-to-end learning signal: loss drops sharply on one repeated batch.
    (The protocol-level 16-example overfit run happens on the real model, phase 7.)"""
    torch.manual_seed(0)
    model = make_model()
    model.train()
    batch = make_batch(seed=5)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)
    first = None
    for i in range(150):
        opt.zero_grad()
        loss = model(batch).loss
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    # a frozen *random* tiny LLM steered only through 4 soft tokens: expect a clear,
    # sustained decrease, not literal memorization (that is the phase-7 real-model test)
    assert loss.item() < first - 0.5, (first, loss.item())


def test_evidence_dropout_applied_exactly_once():
    """`ReGraph.md` §2.4 applies one Dropout(R), inside Fuse. Model-level dropout on R
    would double-drop the graph signal (docs/OPEN-QUESTIONS.md Q18)."""
    model = make_model(**{"fuse.dropout": 0.5})
    assert not hasattr(model, "reader_dropout")
    torch.manual_seed(0)
    model.train()
    b_pre = torch.zeros(1, 4, model.d_llm)
    r = torch.ones(1, 4, model.d_llm)
    kept = [(model.fuse(b_pre, r)[0] != 0).float().mean().item() for _ in range(200)]
    # single dropout at p=0.5 keeps ~50% of units; double dropout would keep ~25%
    assert 0.42 < sum(kept) / len(kept) < 0.58


def test_lr_mult_builds_scaled_param_groups():
    model = make_model()
    cfg = model.cfg
    cfg["train"]["lr_mult"] = {"reader": 10.0, "b_base": 0.5}
    opt = build_optimizer(model, cfg)
    by_lr = {}
    for g in opt.param_groups:
        by_lr.setdefault(round(g["lr"] / cfg["train"]["lr"], 3), 0)
        by_lr[round(g["lr"] / cfg["train"]["lr"], 3)] += len(g["params"])
    assert set(by_lr) == {0.5, 1.0, 10.0}
    n_opt = sum(len(g["params"]) for g in opt.param_groups)
    assert n_opt == sum(1 for p in model.parameters() if p.requires_grad)
    # b_base keeps weight_decay 0 even in its own multiplier group
    for g in opt.param_groups:
        if abs(g["lr"] / cfg["train"]["lr"] - 0.5) < 1e-9:
            assert g["weight_decay"] == 0.0


def test_lr_mult_zero_freezes_a_module():
    """`lr_mult: {fuse: 0.0}` pins the fusion gate while the rest still trains.

    Used by the arXiv Stage-1 alignment (docs/OPEN-QUESTIONS.md Q21): the gate is a single
    scalar head that learns "this evidence is noise" and shuts the channel before the
    projector can make it useful, and since the gradient reaching W_O is scaled by the gate,
    shutting it re-severs the path."""
    torch.manual_seed(0)
    model = make_model()
    model.cfg["train"]["lr_mult"] = {"fuse": 0.0}
    opt = build_optimizer(model, model.cfg)
    w_g_before = model.fuse.w_g.weight.clone()
    ln_before = model.fuse.ln_b.weight.clone()
    batch = make_batch()
    model.train()
    for _ in range(3):
        opt.zero_grad()
        model(batch).loss.backward()
        opt.step()
    assert torch.equal(w_g_before, model.fuse.w_g.weight), "gate should be frozen"
    assert torch.equal(ln_before, model.fuse.ln_b.weight), "whole fuse module frozen"
    assert model.reader.w_o.weight.abs().sum() > 0, "reader must still train"
