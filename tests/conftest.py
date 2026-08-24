"""Shared fixtures: a tiny random fp32 Llama, a fake tokenizer, tiny configs, and
ragged synthetic batches (00-conventions.md: uniform batches hide indexing bugs)."""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from regraph.data.collate import make_collate_fn
from regraph.model import ReGraph

VOCAB = 256
PLACEHOLDER_ID = 250
PAD_ID = 0
EOS_ID = 2
D_LLM = 64
D_ATTR = 16
D_GRAPH = 32
N_B = 4


class FakeTokenizer:
    """Just enough tokenizer surface for ReGraph in unit tests."""

    eos_token_id = EOS_ID
    pad_token_id = PAD_ID

    def convert_tokens_to_ids(self, token: str) -> int:
        return PLACEHOLDER_ID

    def decode(self, ids, skip_special_tokens=True) -> str:
        return " ".join(f"t{i}" for i in ids)


def tiny_cfg(num_rounds: int = 3, **overrides) -> dict:
    cfg = {
        "seed": 0,
        "llm": {
            "name": "tiny", "dtype": "float32", "freeze": True,
            "gradient_checkpointing": False, "d_llm": D_LLM, "num_layers": 8,
        },
        "model": {
            "num_query_tokens": N_B, "num_rounds": num_rounds,
            "placeholder_token": "<ph>", "pad_token": "<pad>",
        },
        "graph_encoder": {
            "name": "graph_transformer", "num_layers": 2, "d_graph": D_GRAPH,
            "heads": 4, "edge_dim": D_ATTR, "dropout": 0.0,
        },
        "graph": {"symmetrize_for_diffusion": True, "add_self_loops": True},
        "reader": {
            "d_reader": D_GRAPH, "heads": 4, "max_hops": 2,
            "share_across_rounds": True, "shared_hop_weights": False, "dropout": 0.0,
            "w_o_init": "zeros", "w_o_init_std": 1.0e-3,
        },
        "fuse": {"dropout": 0.0, "share_across_rounds": True},
        "data": {"d_attr": D_ATTR},
        "train": {
            "lr": 1e-2, "weight_decay": 0.05, "warmup_ratio": 0.05,
            "max_epochs": 2, "batch_size": 2, "grad_accum": 1, "grad_clip": 1.0,
            "early_stopping_patience": 2, "num_workers": 0,
        },
        "eval": {"max_new_tokens": 8, "do_sample": False, "batch_size": 2},
    }
    for key, val in overrides.items():
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = val
    return cfg


def make_tiny_llm(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=VOCAB, hidden_size=D_LLM, intermediate_size=128,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config).eval()


def make_model(num_rounds: int = 3, seed: int = 0, **cfg_overrides) -> ReGraph:
    cfg = tiny_cfg(num_rounds=num_rounds, **cfg_overrides)
    torch.manual_seed(seed)
    model = ReGraph(cfg, make_tiny_llm(seed), FakeTokenizer(), placeholder_id=PLACEHOLDER_ID)
    return model


def random_graph(n: int, e: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, D_ATTR, generator=g)
    if e > 0:
        edge_index = torch.randint(0, n, (2, e), generator=g)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    edge_attr = torch.randn(e, D_ATTR, generator=g)
    return x, edge_index, edge_attr


def make_items(
    n_qs=(5, 9, 7), n_nodes=(6, 3, 10), n_edges=(9, 2, 14),
    with_answer: bool = True, seed: int = 0,
) -> list[dict]:
    """Ragged synthetic items shaped exactly like GraphQADataset.__getitem__ output."""
    g = torch.Generator().manual_seed(seed)
    items = []
    for i, (n_q, n, e) in enumerate(zip(n_qs, n_nodes, n_edges)):
        x, edge_index, edge_attr = random_graph(n, e, seed + 17 * i)
        answer = (
            torch.cat([torch.randint(3, VOCAB - 6, (3,), generator=g),
                       torch.tensor([EOS_ID])])
            if with_answer else torch.zeros(0, dtype=torch.long)
        )
        roles = torch.zeros(n, dtype=torch.long)
        roles[0] = 1  # one "mentioned" node
        items.append(
            {
                "id": i,
                "question": f"q{i}",
                "answer": "gold",
                "q_ids": torch.randint(3, VOCAB - 6, (n_q,), generator=g),
                "boundary_ids": torch.tensor([11, 12]),
                "answer_ids": answer,
                "x": x,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "roles": roles,
            }
        )
    return items


def make_batch(**kwargs) -> dict:
    collate = make_collate_fn(
        pad_token_id=PAD_ID, placeholder_id=PLACEHOLDER_ID, num_query_tokens=N_B,
        symmetrize_for_diffusion=True, add_self_loops=True,
    )
    return collate(make_items(**kwargs))


@pytest.fixture
def batch() -> dict:
    return make_batch()


@pytest.fixture
def model() -> ReGraph:
    return make_model()


def dense_p(src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor, n: int) -> torch.Tensor:
    """Dense P from a local-layout edge list (reference implementation for tests)."""
    p = torch.zeros(n, n)
    p[src, dst] = w
    return p
