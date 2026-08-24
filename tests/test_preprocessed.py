"""01-data.md acceptance test 1: split sizes and averages match the protocol table.

Each dataset's test runs only once its preprocessing cache exists (the preprocess CLI
itself also enforces these checks at build time)."""

import json
from pathlib import Path

import pytest

from regraph.utils.config import load_config

EXPECTED = {
    "expla_graphs": {
        "splits": {"train": 1659, "val": 553, "test": 554},
        "avg_nodes": 5.17, "avg_edges": 4.25,
    },
    "scene_graphs": {
        "total": 100000, "avg_nodes": 19.13, "avg_edges": 68.44,
    },
    # G-Retriever's released splits: 2826/245/1628 usable examples; the paper's
    # 4,737 is the original WebQSP question count and its averages are over 4,700
    # (incl. the dropped empty graph) — see docs/OPEN-QUESTIONS.md Q16
    "webqsp": {
        "splits": {"train": 2826, "val": 245, "test": 1628},
        "total": 4699, "avg_nodes": 1371.18, "avg_edges": 4253.27,
    },
}


def stats_path(name: str) -> Path:
    cfg = load_config(f"configs/{name}.yaml")
    from regraph.data.attr_encoder import encoder_slug

    return (
        Path(cfg["data"]["cache_dir"]) / name
        / encoder_slug(cfg["data"]["attr_encoder"]) / "stats.json"
    )


@pytest.mark.parametrize("name", list(EXPECTED))
def test_protocol_stats(name):
    path = stats_path(name)
    if not path.exists():
        pytest.skip(f"{name} not preprocessed yet ({path})")
    with open(path) as f:
        stats = json.load(f)
    exp = EXPECTED[name]
    if "splits" in exp:
        assert stats["split_sizes"] == exp["splits"]
    if "total" in exp:
        assert stats["num_examples"] == exp["total"]
    assert abs(stats["avg_nodes"] - exp["avg_nodes"]) <= 0.011
    assert abs(stats["avg_edges"] - exp["avg_edges"]) <= 0.011
