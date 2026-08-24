"""01-data.md acceptance tests: collate layout, labels, roles, config machinery.

Split-size/statistics checks against the protocol table run in
tests/test_preprocessed.py once the caches exist (they need the raw data)."""

import torch

from conftest import EOS_ID, N_B, PAD_ID, PLACEHOLDER_ID, make_batch, make_items
from regraph.data.collate import make_collate_fn
from regraph.data.roles import ROLE_MENTIONED, ROLE_NONE, ROLE_SOURCE, assign_roles
from regraph.utils.config import deep_merge, load_config


def test_collate_ragged_layout():
    items = make_items(n_qs=(5, 9, 7), n_nodes=(6, 3, 10))
    batch = make_batch()
    # b_positions rows point at the placeholder token id
    at_slots = batch["input_ids"].gather(1, batch["b_positions"])
    assert (at_slots == PLACEHOLDER_ID).all()
    # labels -100 at all non-answer positions
    for i, it in enumerate(items):
        n_q, n_bd, n_a = it["q_ids"].shape[0], 2, it["answer_ids"].shape[0]
        ans_lo = n_q + N_B + n_bd
        row = batch["labels"][i]
        assert (row[:ans_lo] == -100).all()
        assert (row[ans_lo : ans_lo + n_a] == it["answer_ids"]).all()
        assert (row[ans_lo + n_a :] == -100).all()
    # node_mask sums equal true node counts
    assert batch["node_mask"].sum(-1).tolist() == [6, 3, 10]
    # attention over real length only
    for i, it in enumerate(items):
        real = it["q_ids"].shape[0] + N_B + 2 + it["answer_ids"].shape[0]
        assert int(batch["attention_mask"][i].sum()) == real


def test_collate_round_trip_ids():
    """Removing the B slots reproduces prompt + answer ids (01-data.md test 3)."""
    items = make_items()
    batch = make_batch()
    for i, it in enumerate(items):
        row = batch["input_ids"][i]
        real = row[batch["attention_mask"][i].bool()]
        keep = real[real != PLACEHOLDER_ID]
        expected = torch.cat([it["q_ids"], it["boundary_ids"], it["answer_ids"]])
        assert torch.equal(keep, expected)


def test_collate_eval_mode_has_no_labels_targets():
    batch = make_batch(with_answer=False)
    assert (batch["labels"] == -100).all()
    assert batch["input_ids"].shape[1] == max(5, 9, 7) + N_B + 2


def test_collate_graph_flattening():
    items = make_items(n_nodes=(6, 3, 10), n_edges=(9, 2, 14))
    batch = make_batch()
    assert batch["x"].shape[0] == 19
    assert batch["edge_index"].shape[1] == 25
    # per-example edges land in the right global slot range
    lo = 0
    off = 0
    for i, it in enumerate(items):
        e = it["edge_index"].shape[1]
        seg = batch["edge_index"][:, lo : lo + e]
        assert torch.equal(seg, it["edge_index"] + off)
        lo += e
        off += it["x"].shape[0]
    assert batch["node_batch"].tolist() == [0] * 6 + [1] * 3 + [2] * 10


def test_roles_hand_written():
    """01-data.md test 4: exact expected role vector; ambiguity -> none."""
    q = "Does the Global Warming increase sea levels?"
    nodes = ["global warming", "sea", "ice caps"]
    # "sea" is itself a span of the question ("increase sea levels"), so it is mentioned too
    assert assign_roles(q, nodes) == [ROLE_MENTIONED, ROLE_MENTIONED, ROLE_NONE]
    assert assign_roles("the sea is rising", nodes) == [ROLE_NONE, ROLE_MENTIONED, ROLE_NONE]
    # partial-word overlap is NOT a span match
    assert assign_roles("we discussed seasonal change", ["sea"]) == [ROLE_NONE]
    # duplicate node text -> ambiguous -> none
    assert assign_roles(q, ["global warming", "global warming"]) == [ROLE_NONE, ROLE_NONE]
    # anchor-free question: all none, first-class case
    assert assign_roles("Which region is unstable?", nodes) == [ROLE_NONE] * 3
    # WebQSP-style source entity, exact lowercased match
    assert assign_roles("who founded acme?", ["acme corp", "acme"], ["Acme"]) == [
        ROLE_NONE, ROLE_SOURCE,
    ]


def test_roles_normalization():
    # punctuation/articles stripped on both sides
    assert assign_roles("Is the U.S. big?", ["u.s."]) == [ROLE_MENTIONED]
    assert assign_roles("was the empire state building tall", ["empire state building"]) == [
        ROLE_MENTIONED
    ]


def test_config_merge_and_interp(tmp_path):
    base = tmp_path / "default.yaml"
    base.write_text("a: 1\nb: {c: 2, d: 3}\nrun_dir: runs/${b.c}/${run_name}\nrun_name: x\n")
    child = tmp_path / "child.yaml"
    child.write_text("defaults: default.yaml\nb: {c: 9}\n")
    cfg = load_config(child, overrides=["b.d=7", "run_name=hello"])
    assert cfg["a"] == 1 and cfg["b"]["c"] == 9 and cfg["b"]["d"] == 7
    assert cfg["run_dir"] == "runs/9/hello"
    assert deep_merge({"x": {"y": 1}}, {"x": {"z": 2}}) == {"x": {"y": 1, "z": 2}}


def test_attr_store_empty_ids(tmp_path):
    """Scene graphs can have zero edges; AttrStore.embed must handle empty id lists."""
    import json

    import numpy as np
    import torch

    from regraph.data.attr_encoder import AttrStore

    d = tmp_path / "store"
    d.mkdir()
    np.save(d / "emb.f16.npy", np.random.randn(3, 8).astype(np.float16))
    if (d / "emb.f16.npy.npy").exists():
        (d / "emb.f16.npy.npy").rename(d / "emb.f16.npy")
    (d / "texts.json").write_text(json.dumps(["a", "b", "c"]))
    (d / "meta.json").write_text(json.dumps({"encoder": "x", "num_texts": 3, "d_attr": 8}))
    store = AttrStore(d)
    out = store.embed(torch.zeros(0, dtype=torch.int32))
    assert out.shape == (0, 8) and out.dtype == torch.float32
    full = store.embed(torch.tensor([2, 0], dtype=torch.int32))
    assert full.shape == (2, 8)


def test_config_override_scientific_notation(tmp_path):
    """PyYAML reads `1e-5` as a string; overrides must still yield numbers."""
    base = tmp_path / "c.yaml"
    base.write_text("train: {lr: 1.0e-5, epochs: 10}\n")
    cfg = load_config(base, overrides=["train.lr=1e-5"])
    assert isinstance(cfg["train"]["lr"], float) and cfg["train"]["lr"] == 1e-5
    cfg = load_config(base, overrides=["train.lr=1e-05", "train.epochs=3"])
    assert cfg["train"]["lr"] == 1e-5
    assert isinstance(cfg["train"]["epochs"], int) and cfg["train"]["epochs"] == 3
    # non-numeric strings and structured values still parse as before
    cfg = load_config(base, overrides=["train.name=cosine", "train.mult={a: 2.0}"])
    assert cfg["train"]["name"] == "cosine" and cfg["train"]["mult"] == {"a": 2.0}
