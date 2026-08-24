"""08-inference.md acceptance tests: KV-cache equivalence, B-token invariance,
stopping, batch invariance."""

import torch

from conftest import EOS_ID, VOCAB, make_batch, make_items, make_model
from conftest import N_B, PAD_ID, PLACEHOLDER_ID
from regraph.data.collate import make_collate_fn


def test_cached_equals_naive():
    """The critical test: greedy outputs token-identical between the KV-cached path
    and full recompute, on a ragged batch."""
    model = make_model()
    batch = make_batch(with_answer=False)
    fast = model.generate(batch, max_new_tokens=8)
    slow = model.generate_naive(batch, max_new_tokens=8)
    assert torch.equal(fast["ids"], slow["ids"]), (fast["ids"], slow["ids"])
    assert fast["texts"] == slow["texts"]


def test_cached_equals_naive_zero_rounds():
    model = make_model(num_rounds=0)
    batch = make_batch(with_answer=False)
    fast = model.generate(batch, max_new_tokens=6)
    slow = model.generate_naive(batch, max_new_tokens=6)
    assert torch.equal(fast["ids"], slow["ids"])


def test_b_token_invariance_across_decode_steps():
    """b_pre at every round is identical whether 0 or 5 generated tokens follow
    (`ReGraph.md` §2.5: B_pre^{t,(s)} = B_pre^{t,(1)})."""
    model = make_model()
    batch = make_batch(with_answer=False)
    recorded: dict[int, list[torch.Tensor]] = {}

    orig = type(model)._graph_round

    def recording_round(self, t, hidden, b_positions, ctx, diag, rr=False):
        b_idx = torch.arange(hidden.shape[0]).unsqueeze(1)
        recorded.setdefault(t, []).append(hidden[b_idx, b_positions].clone())
        return orig(self, t, hidden, b_positions, ctx, diag, rr)

    model._graph_round = recording_round.__get__(model)
    model.generate_naive(batch, max_new_tokens=6)
    for t, states in recorded.items():
        for s in states[1:]:
            assert torch.allclose(states[0], s, atol=1e-5), f"round {t} drifted"


def test_generation_stops_at_eos_and_length_cap():
    model = make_model()
    batch = make_batch(with_answer=False)

    out = model.generate(batch, max_new_tokens=32)
    assert out["ids"].shape[1] <= 32

    # force EOS as the argmax at every step -> stops after 1 token
    class ConstantEOS(torch.nn.Module):
        def forward(self, h):
            logits = torch.zeros(*h.shape[:-1], VOCAB, dtype=h.dtype)
            logits[..., EOS_ID] = 10.0
            return logits

    model.llm.lm_head = ConstantEOS()
    out = model.generate(batch, max_new_tokens=32)
    assert out["ids"].shape[1] == 1
    assert (out["ids"][:, 0] == EOS_ID).all()
    assert out["texts"] == ["", "", ""]


def test_batch_invariance_single_vs_padded():
    """Generating an example alone and inside a ragged padded batch of 4 gives the
    same tokens (position ids and cache offsets from attention_mask)."""
    model = make_model()
    items = make_items(
        n_qs=(5, 12, 7, 9), n_nodes=(6, 3, 10, 4), n_edges=(9, 2, 14, 5),
        with_answer=False,
    )
    collate = make_collate_fn(
        pad_token_id=PAD_ID, placeholder_id=PLACEHOLDER_ID, num_query_tokens=N_B,
    )
    batched = model.generate(collate(items), max_new_tokens=6)
    for i, it in enumerate(items):
        alone = model.generate(collate([it]), max_new_tokens=6)
        n = alone["ids"].shape[1]
        assert alone["texts"][0] == batched["texts"][i], (
            i, alone["texts"][0], batched["texts"][i]
        )


def test_prefill_diagnostics_returned():
    model = make_model()
    batch = make_batch(with_answer=False)
    out = model.generate(batch, max_new_tokens=2, collect_diagnostics=True)
    assert len(out["diagnostics"]["gates"]) == model.num_rounds
    assert len(out["diagnostics"]["alphas"]) == model.num_rounds
