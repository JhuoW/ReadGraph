"""09-evaluation.md acceptance test 1: ported metrics reproduce hand-computed scores."""

import pytest

from regraph.eval import (
    compute_metric,
    get_accuracy_expla_graphs,
    get_accuracy_gqa,
    get_accuracy_webqsp,
    match,
    normalize,
    per_example_correct,
)


def test_expla_graphs_metric_hand_computed():
    records = [
        {"pred": "support", "label": "support"},            # correct
        {"pred": " Counter.", "label": "counter"},          # first match, case-folded
        {"pred": "I think support", "label": "counter"},    # wrong
        {"pred": "counter maybe support", "label": "counter"},  # first match wins
        {"pred": "no stance", "label": "support"},          # no match
        {"pred": "supportive", "label": "support"},         # regex finds "support"
        {"pred": "Counter and support", "label": "support"},  # first is counter
        {"pred": "SUPPORT", "label": "support"},            # uppercase not in regex
        {"pred": "the answer is counter", "label": "counter"},
        {"pred": "support", "label": "counter"},
    ]
    assert get_accuracy_expla_graphs(records) == pytest.approx(5 / 10)


def test_gqa_metric_hand_computed():
    records = [
        {"pred": "the chair is red", "label": "red"},   # substring -> correct
        {"pred": "Red", "label": "red"},                # case-sensitive -> wrong
        {"pred": "blue", "label": "blue"},
        {"pred": "light blue", "label": "blue"},        # substring -> correct
        {"pred": "", "label": "dog"},
    ]
    assert get_accuracy_gqa(records) == pytest.approx(3 / 5)


def test_webqsp_hit_hand_computed():
    records = [
        {"pred": "Barack Obama", "label": "barack obama|michelle obama"},   # hit
        {"pred": "the United States!", "label": "united states"},           # normalize
        {"pred": "france|germany", "label": "spain|portugal"},              # miss
        {"pred": "unknown", "label": "known"},  # "known" not word-boundary... substring
    ]
    out = get_accuracy_webqsp(records)
    # row 4: normalize("unknown") contains "known" as substring -> hit (matches original)
    assert out["hit@1"] == pytest.approx(3 / 4)
    assert 0 <= out["f1"] <= 1


def test_normalize_and_match_port():
    assert normalize("The U.S.!") == "us"
    assert normalize("A dog; the cat.") == "dog cat"
    assert match("I saw The Dog.", "dog")
    assert not match("cat", "dog")


def test_per_example_correct_agrees_with_metrics():
    r = {"pred": "counter maybe", "label": "counter"}
    assert per_example_correct("expla_graphs", r["pred"], r["label"])
    assert per_example_correct("scene_graphs", "deep blue", "blue")
    assert not per_example_correct("scene_graphs", "Blue", "blue")
    assert per_example_correct("webqsp", "obama|george bush", "george bush")
    assert not per_example_correct("webqsp", "obama|bush", "george bush")
    assert compute_metric("expla_graphs", [r]) == {"accuracy": 1.0}
