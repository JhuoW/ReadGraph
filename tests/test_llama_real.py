"""Real-backbone tests (meta-llama/Llama-3.1-8B-Instruct, bf16, GPU).

Run with: REGRAPH_LLAMA_TESTS=1 pytest tests/test_llama_real.py -x
These are the phase-3/8 exit criteria on the real model; everything else is covered
by the tiny-model tests."""

import os

import pytest
import torch

pytestmark = [
    pytest.mark.llama,
    pytest.mark.skipif(
        os.environ.get("REGRAPH_LLAMA_TESTS") != "1",
        reason="set REGRAPH_LLAMA_TESTS=1 to run real-backbone tests",
    ),
]


@pytest.fixture(scope="module")
def real_model():
    from regraph.model import load_regraph
    from regraph.utils.config import load_config

    cfg = load_config("configs/expla_graphs.yaml")
    model = load_regraph(cfg, device="cuda")
    return model


def _toy_batch(model, questions, with_graph_nodes=(4, 7)):
    import torch

    from regraph.data.collate import make_collate_fn

    tok = model.tokenizer
    items = []
    for i, q in enumerate(questions):
        n = with_graph_nodes[i % len(with_graph_nodes)]
        e = 2 * n
        g = torch.Generator().manual_seed(i)
        items.append(
            {
                "id": i,
                "question": q,
                "answer": "yes",
                "q_ids": torch.tensor(tok(q)["input_ids"]),
                "boundary_ids": torch.tensor(tok("\nAnswer:", add_special_tokens=False)["input_ids"]),
                "answer_ids": torch.zeros(0, dtype=torch.long),
                "x": torch.randn(n, 1024, generator=g),
                "edge_index": torch.randint(0, n, (2, e), generator=g),
                "edge_attr": torch.randn(e, 1024, generator=g),
                "roles": torch.zeros(n, dtype=torch.long),
            }
        )
    collate = make_collate_fn(
        pad_token_id=tok.pad_token_id, placeholder_id=model.placeholder_id,
        num_query_tokens=model.num_query_tokens,
    )
    return collate(items)


def test_noop_equivalence_bf16(real_model):
    """03-query-tokens.md test 1 on the real model: manual loop vs HF forward,
    atol 2e-2 in bf16 (graph rounds are identity at init because W_O = 0)."""
    model = real_model
    batch = _toy_batch(model, ["Question: Is the sky blue?\n", "Question: What color is grass, in one word?\n"])
    batch = model._to_device(batch)
    with torch.no_grad():
        out = model(batch)
        embeds = model.build_inputs_embeds(batch["input_ids"], batch["b_positions"])
        ref = model.llm(inputs_embeds=embeds, attention_mask=batch["attention_mask"]).logits
    real = batch["attention_mask"].bool()
    diff = (out.logits[real].float() - ref[real].float()).abs().max().item()
    assert diff < 2e-2, f"max |Δlogit| = {diff}"


def test_cached_equals_naive_real(real_model):
    model = real_model
    batch = _toy_batch(
        model,
        ["Question: What is 2+2?\n", "Question: Name one primary color.\n",
         "Question: Is water wet? Answer yes or no.\n"],
    )
    fast = model.generate(batch, max_new_tokens=16)
    slow = model.generate_naive(batch, max_new_tokens=16)
    assert fast["texts"] == slow["texts"], (fast["texts"], slow["texts"])


def test_trainable_count_expected_range(real_model):
    n = real_model.num_trainable_parameters()
    assert 20e6 < n < 40e6, f"{n/1e6:.1f}M outside the ≈28M expectation (06-model.md §6.3)"
