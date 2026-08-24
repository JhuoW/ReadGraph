"""03-query-tokens.md acceptance tests: sequence assembly and the manual layer loop."""

import torch

from conftest import N_B, PLACEHOLDER_ID, make_batch, make_model


def test_noop_equivalence_manual_loop_vs_hf(model, batch):
    """The critical test: with Γ_t as identity, the manual loop reproduces
    llm(inputs_embeds=...).logits (fp32 atol 1e-5)."""
    model.eval()
    model._graph_round = lambda t, hidden, b_positions, ctx, diag, rr=False: hidden
    with torch.no_grad():
        out = model(batch)
        embeds = model.build_inputs_embeds(batch["input_ids"], batch["b_positions"])
        ref = model.llm(
            inputs_embeds=embeds, attention_mask=batch["attention_mask"]
        ).logits
    real = batch["attention_mask"].bool()
    assert torch.allclose(out.logits[real], ref[real], atol=1e-5)


def test_noop_equivalence_zero_rounds(batch):
    """num_rounds=0 runs the plain frozen LLM with soft prompt tokens."""
    model = make_model(num_rounds=0)
    model.eval()
    assert model.group_bounds == [(0, 8)]
    with torch.no_grad():
        out = model(batch)
        embeds = model.build_inputs_embeds(batch["input_ids"], batch["b_positions"])
        ref = model.llm(
            inputs_embeds=embeds, attention_mask=batch["attention_mask"]
        ).logits
    real = batch["attention_mask"].bool()
    assert torch.allclose(out.logits[real], ref[real], atol=1e-5)


def test_gather_after_scatter_is_b_base(model, batch):
    """b_pre gathered right after scattering (zero layers run) equals b_base."""
    embeds = model.build_inputs_embeds(batch["input_ids"], batch["b_positions"])
    b_idx = torch.arange(embeds.shape[0]).unsqueeze(1)
    b_pre = embeds[b_idx, batch["b_positions"]]
    expected = model.b_base.to(embeds.dtype).unsqueeze(0).expand(embeds.shape[0], -1, -1)
    assert torch.allclose(b_pre, expected)


def test_replace_touches_only_b_positions(model, batch):
    """After a round with b_post = b_pre + 1, exactly B * N_B entries changed."""
    embeds = model.build_inputs_embeds(batch["input_ids"], batch["b_positions"])

    def bump_round(t, hidden, b_positions, ctx, diag, rr=False):
        b_idx = torch.arange(hidden.shape[0]).unsqueeze(1)
        return hidden.index_put((b_idx, b_positions), hidden[b_idx, b_positions] + 1.0)

    b_idx = torch.arange(embeds.shape[0]).unsqueeze(1)
    bumped = bump_round(0, embeds, batch["b_positions"], None, None)
    changed = (bumped != embeds).any(-1)
    assert int(changed.sum()) == embeds.shape[0] * N_B
    assert changed.gather(1, batch["b_positions"]).all()
    # and the original tensor was not modified in place
    assert torch.allclose(embeds[b_idx, batch["b_positions"]] + 1.0,
                          bumped[b_idx, batch["b_positions"]])


def test_ragged_batch_placeholder_alignment(batch):
    at_slots = batch["input_ids"].gather(1, batch["b_positions"])
    assert (at_slots == PLACEHOLDER_ID).all()
    # rows genuinely differ in offset
    assert len(set(batch["b_positions"][:, 0].tolist())) > 1


def test_llm_frozen_and_grads_flow(model, batch):
    out = model(batch)
    out.loss.backward()
    for name, p in model.llm.named_parameters():
        assert not p.requires_grad and p.grad is None, name
    assert model.b_base.grad is not None
