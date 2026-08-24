"""NLGraph (Wang et al., NeurIPS 2023) — structural graph reasoning in natural language.

This is the benchmark family `ReGraph.md` §2.1 draws its own example from ("Find a path from
node 7 to node 19"), and the one that actually tests the paper's opening claim that GraphRAG
"can not handle non-retrieval query". Two tasks are ported, chosen because they exercise the
two role cases the spec defines and because their answers are 1 bit (well inside what the
graph→LLM channel was measured to carry):

  connectivity — "Is there a path between node i and node j?"  -> ROLE_SOURCE / ROLE_TARGET.
                 This is the first use of ROLE_TARGET anywhere in the project.
  cycle        — "Is there a cycle in this graph?"             -> anchor-free, all ROLE_NONE,
                 the case §2.1 calls "a first-class case, not a failure".

**The graph is parsed out of the prompt, not left in it.** NLGraph serializes the edge list
into the question text for text-only LLMs; ReGraph strips that and feeds the topology through
the encoder, which is the whole point of the method. The remaining question text is the task
sentence only.

**Nodes carry no attributes** — they are bare integers. `ReGraph.md` §2.1 requires only that
"every node can be mapped to a vector", so every node gets the same placeholder text and is
therefore distinguishable only by topology and by its role marker. That is the intended
setting for a structural task.

Splits: NLGraph ships train.json / test.json which partition main.json by question content
(verified: zero overlap, union == main). A validation slice is carved out of train because
ReGraph selects checkpoints on validation loss.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from regraph.data.roles import ROLE_NONE, ROLE_SOURCE, ROLE_TARGET

EDGE_RE = re.compile(r"\((\d+),\s*(\d+)\)")
PAIR_RE = re.compile(r"between node (\d+) and node (\d+)")
NUMBERED_RE = re.compile(r"numbered from (\d+) to (\d+)")
NODE_TEXT = "node"          # identical for every node: topology is the only signal

TASKS = {
    "connectivity": {
        "question": "Is there a path between the two marked nodes in this graph?",
        "anchored": True,
    },
    "cycle": {
        "question": "Is there a cycle in this graph?",
        "anchored": False,
    },
}


def _answer_label(raw: str) -> str:
    low = raw.lower()
    if "there is no cycle" in low or "answer is no" in low:
        return "no"
    if "there is a cycle" in low or "answer is yes" in low:
        return "yes"
    raise ValueError(f"cannot parse NLGraph answer: {raw!r}")


def build_nlgraph(raw_dir: Path, task: str, val_fraction: float = 0.1, seed: int = 0,
                  name_anchors: bool = False):
    """Returns (graphs, examples, node_texts) in the preprocessed-store layout.

    `name_anchors=True` writes the queried node ids into the question ("between node 0 and
    node 2") instead of "the two marked nodes". Needed whenever a **token channel** is used:
    the anchors would otherwise exist only in the role markers, i.e. only in the vector
    channel, and a text-only control could not possibly answer.
    """
    if task not in TASKS:
        raise KeyError(f"unknown NLGraph task {task!r}; known: {sorted(TASKS)}")
    spec = TASKS[task]
    root = Path(raw_dir) / task

    main = json.load(open(root / "main.json"))
    train_q = {v["question"] for v in json.load(open(root / "train.json")).values()}
    test_q = {v["question"] for v in json.load(open(root / "test.json")).values()}

    # deterministic val slice carved from train, stratified by nothing but order-shuffled
    import random
    rng = random.Random(seed)
    train_list = sorted(train_q)
    rng.shuffle(train_list)
    n_val = int(len(train_list) * val_fraction)
    val_q = set(train_list[:n_val])

    graphs, examples = {}, []
    for key, item in main.items():
        q = item["question"]
        edges = [(int(a), int(b)) for a, b in EDGE_RE.findall(q)]
        if not edges:
            continue
        n_nodes = max(max(a, b) for a, b in edges) + 1
        m = NUMBERED_RE.search(q)
        if m:
            n_nodes = max(n_nodes, int(m.group(2)) + 1)

        roles = [ROLE_NONE] * n_nodes
        question = spec["question"]
        if spec["anchored"]:
            pair = PAIR_RE.search(q)
            if pair is None:
                continue
            s, t = int(pair.group(1)), int(pair.group(2))
            n_nodes = max(n_nodes, s + 1, t + 1)
            roles = [ROLE_NONE] * n_nodes
            roles[s], roles[t] = ROLE_SOURCE, ROLE_TARGET
            if name_anchors:
                question = f"Is there a path between node {s} and node {t} in this graph?"

        # undirected: give the encoder both directions (diffusion symmetrizes separately)
        src = [a for a, b in edges] + [b for a, b in edges]
        dst = [b for a, b in edges] + [a for a, b in edges]
        ei = torch.tensor([src, dst], dtype=torch.int32)

        split = "test" if q in test_q else ("val" if q in val_q else "train")
        gk = f"{task}-{key}"
        graphs[gk] = {
            "node_text_id": torch.zeros(n_nodes, dtype=torch.int32),   # all the same text
            "edge_text_id": torch.zeros(ei.shape[1], dtype=torch.int32),
            "edge_index": ei,
        }
        examples.append(
            {
                "id": int(key),
                "graph_key": gk,
                "question": question,
                "answer": _answer_label(item["answer"]),
                "split": split,
                "roles": [[i, r] for i, r in enumerate(roles) if r != ROLE_NONE],
                "difficulty": item["difficulty"],
                "num_nodes": n_nodes,
            }
        )
    return graphs, examples, [NODE_TEXT]
