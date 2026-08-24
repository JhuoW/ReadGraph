"""06-model.md acceptance tests: full assembly, gradients, equivariance."""

import torch

from conftest import VOCAB, make_batch, make_items, make_model
from regraph.data.collate import make_collate_fn
from conftest import PAD_ID, PLACEHOLDER_ID, N_B


def test_forward_finite_loss_and_shapes(model, batch):
    out = model(batch)
    assert out.loss is not None and torch.isfinite(out.loss)
    assert out.logits.shape == (*batch["input_ids"].shape, VOCAB)
    assert len(out.diagnostics["gates"]) == model.num_rounds
    assert len(out.diagnostics["alphas"]) == model.num_rounds


def test_gradients_reach_all_trainables_and_no_llm(model, batch):
    """06-model.md test 2: grad not-None for every trainable, None for the LLM.

    At step 0 only b_base and W_O can have *nonzero* grads (zero-init W_O blocks
    everything upstream of R^t by design); after one optimizer step, nonzero
    gradients must reach every module family."""
    out = model(batch)
    out.loss.backward()
    named = dict(model.named_parameters())
    for n, p in named.items():
        if n.startswith("llm."):
            assert p.grad is None, n
        else:
            assert p.grad is not None, f"trainable {n} disconnected from the loss"
    assert named["b_base"].grad.abs().sum() > 0
    assert named["reader.w_o.weight"].grad.abs().sum() > 0

    model.train()
    opt = torch.optim.SGD(model.trainable_parameters(), lr=1e-1)
    opt.step()  # W_O becomes nonzero -> the graph path opens
    opt.zero_grad()
    model(batch).loss.backward()
    for prefix in ("b_base", "reader.", "fuse.", "graph_encoder.", "role_emb."):
        got = [
            n for n, p in named.items()
            if (n == prefix or n.startswith(prefix)) and p.grad is not None
            and p.grad.abs().sum() > 0
        ]
        assert got, f"no nonzero gradient reached {prefix} after step 1"


def test_node_permutation_equivariance():
    """Permuting node order (edges consistent) leaves the loss unchanged."""
    items = make_items(n_qs=(6,), n_nodes=(7,), n_edges=(12,), seed=3)
    collate = make_collate_fn(
        pad_token_id=PAD_ID, placeholder_id=PLACEHOLDER_ID, num_query_tokens=N_B,
    )
    model = make_model()
    model.eval()
    with torch.no_grad():
        loss1 = model(collate(items)).loss

        perm = torch.randperm(7)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(7)
        it = dict(items[0])
        it["x"] = it["x"][perm]
        it["roles"] = it["roles"][perm]
        it["edge_index"] = inv[it["edge_index"]]
        loss2 = model(collate([it])).loss
    assert torch.allclose(loss1, loss2, atol=1e-3)


def test_zero_h_no_effect_at_step0_effect_after_training(batch):
    model = make_model()
    model.eval()

    def zero_h(b):
        ctx = type(model).encode_graph(model, b)
        ctx["h"] = torch.zeros_like(ctx["h"])
        if "k_h" in ctx:
            reader = model.reader if model.share_reader else model.reader[0]
            ctx["k_h"], ctx["v_h"] = reader.precompute(ctx["h"])
        return ctx

    with torch.no_grad():
        loss_plain = model(batch).loss
        model.encode_graph = zero_h
        loss_zero = model(batch).loss
        model.encode_graph = type(model).encode_graph.__get__(model)
    assert torch.allclose(loss_plain, loss_zero, atol=1e-6)  # W_O = 0 at step 0

    # one optimizer step -> the graph path matters
    model.train()
    opt = torch.optim.SGD(model.trainable_parameters(), lr=1e-1)
    model(batch).loss.backward()
    opt.step()
    model.eval()
    with torch.no_grad():
        loss_plain = model(batch).loss
        model.encode_graph = zero_h
        loss_zero = model(batch).loss
    assert not torch.allclose(loss_plain, loss_zero, atol=1e-6)


def test_trainable_param_count_tiny():
    model = make_model()
    n = model.num_trainable_parameters()
    total = sum(p.numel() for p in model.parameters())
    frozen = sum(p.numel() for p in model.llm.parameters())
    assert n == total - frozen
    assert n > 0


def test_diagnostics_gate_half_at_init(model, batch):
    with torch.no_grad():
        out = model(batch)
    for g in out.diagnostics["gates"]:
        assert torch.allclose(g.float(), torch.full_like(g.float(), 0.5), atol=1e-6)


def test_per_round_readers_when_not_shared(batch):
    model = make_model(**{"reader.share_across_rounds": False, "fuse.share_across_rounds": False})
    assert len(model.reader) == model.num_rounds
    assert len(model.fuse) == model.num_rounds
    out = model(batch)
    assert torch.isfinite(out.loss)
