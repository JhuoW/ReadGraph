"""Raw GraphQA construction, ported from G-Retriever (He et al., 2024).

Source: https://github.com/XiaoxinHe/G-Retriever
  - src/dataset/preprocess/expla_graphs.py  (textualize_graph)
  - src/dataset/preprocess/scene_graphs.py  (textualize_graph, generate_split)
  - src/dataset/preprocess/webqsp.py        (step_one, generate_split)
  - src/dataset/preprocess/generate_split.py

The port keeps the exact node/edge textualization and the exact split RNG
(`sklearn.model_selection.train_test_split(..., random_state=42)`) so the split
indices match G-Retriever's released ones (`ReGraph.md` §3.1: "We use the exact
split indices released by G-Retriever"). ReGraph consumes the *full* graphs —
G-Retriever's PCST retrieval step is deliberately not ported (`ReGraph.md` §3.2:
"the graph is not serialized into the LLM context"; docs/components/01-data.md:
"There is no retrieval step, no subgraph selection").

Each builder returns:
  graphs:   dict key -> {"node_texts": list[str], "edge_texts": list[str],
                         "edge_index": LongTensor [2, e]}
  examples: list of dicts {"id", "graph_key", "question", "answer", "split", ...}
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# G-Retriever src/dataset/expla_graphs.py::ExplaGraphsDataset.prompt (the trailing
# "\n\nAnswer:" of the original prompt is supplied by ReGraph's answer boundary).
EXPLA_INSTRUCTION = (
    "Question: Do argument 1 and argument 2 support or counter each other? "
    "Answer in one word in the form of 'support' or 'counter'."
)


def _generate_split_indices(num_examples: int) -> dict[str, list[int]]:
    """Port of G-Retriever src/dataset/preprocess/generate_split.py::generate_split."""
    indices = np.arange(num_examples)
    train_idx, temp = train_test_split(indices, test_size=0.4, random_state=42)
    val_idx, test_idx = train_test_split(temp, test_size=0.5, random_state=42)
    return {
        "train": [int(i) for i in train_idx],
        "val": [int(i) for i in val_idx],
        "test": [int(i) for i in test_idx],
    }


def _split_of(idx_split: dict[str, list[int]], n: int) -> list[str]:
    split = [""] * n
    for name, idxs in idx_split.items():
        for i in idxs:
            split[i] = name
    assert all(split), "every example must belong to a split"
    return split


# ---------------------------------------------------------------------------
# ExplaGraphs
# ---------------------------------------------------------------------------

def _textualize_expla_graph(graph: str) -> tuple[list[str], list[str], torch.Tensor]:
    """Port of preprocess/expla_graphs.py::textualize_graph."""
    triplets = re.findall(r"\((.*?)\)", graph)
    nodes: dict[str, int] = {}
    src_ids, dst_ids, edge_texts = [], [], []
    for tri in triplets:
        src, edge_attr, dst = tri.split(";")
        src = src.lower().strip()
        dst = dst.lower().strip()
        if src not in nodes:
            nodes[src] = len(nodes)
        if dst not in nodes:
            nodes[dst] = len(nodes)
        src_ids.append(nodes[src])
        dst_ids.append(nodes[dst])
        edge_texts.append(edge_attr.lower().strip())
    node_texts = list(nodes.keys())
    edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)
    return node_texts, edge_texts, edge_index


def build_expla_graphs(raw_dir: Path) -> tuple[dict, list[dict]]:
    df = pd.read_csv(raw_dir / "train_dev.tsv", sep="\t")
    idx_split = _generate_split_indices(len(df))
    split = _split_of(idx_split, len(df))

    graphs, examples = {}, []
    for i, row in df.iterrows():
        node_texts, edge_texts, edge_index = _textualize_expla_graph(row["graph"])
        graphs[str(i)] = {
            "node_texts": node_texts,
            "edge_texts": edge_texts,
            "edge_index": edge_index,
        }
        question = (
            f"Argument 1: {row['arg1']}\nArgument 2: {row['arg2']}\n{EXPLA_INSTRUCTION}"
        )
        examples.append(
            {
                "id": int(i),
                "graph_key": str(i),
                "question": question,
                "answer": row["label"],
                "split": split[i],
            }
        )
    return graphs, examples


# ---------------------------------------------------------------------------
# SceneGraphs
# ---------------------------------------------------------------------------

def _textualize_scene_graph(data: dict) -> tuple[list[str], list[str], torch.Tensor]:
    """Port of preprocess/scene_graphs.py::textualize_graph."""
    objectid2nodeid = {object_id: idx for idx, object_id in enumerate(data["objects"].keys())}
    node_texts: list[str] = []
    src_ids, dst_ids, edge_texts = [], [], []
    for objectid, obj in data["objects"].items():
        node_attr = f"name: {obj['name']}"
        x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
        if len(obj["attributes"]) > 0:
            node_attr = node_attr + "; attribute: " + ", ".join(obj["attributes"])
        node_attr += "; (x,y,w,h): " + str((x, y, w, h))
        node_texts.append(node_attr)
        for rel in obj["relations"]:
            src_ids.append(objectid2nodeid[objectid])
            dst_ids.append(objectid2nodeid[rel["object"]])
            edge_texts.append(rel["name"])
    edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)
    if edge_index.numel() == 0:
        edge_index = edge_index.reshape(2, 0)
    return node_texts, edge_texts, edge_index


def _scene_graphs_split(questions: pd.DataFrame) -> list[str]:
    """Port of preprocess/scene_graphs.py::generate_split (split over image ids)."""
    unique_image_ids = questions["image_id"].unique()
    np.random.seed(42)  # matches the original; feeds the same shuffled id order in
    shuffled = np.random.permutation(unique_image_ids)
    train_ids, temp_ids = train_test_split(shuffled, test_size=0.4, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
    id_to_set = {image_id: "train" for image_id in train_ids}
    id_to_set.update({image_id: "val" for image_id in val_ids})
    id_to_set.update({image_id: "test" for image_id in test_ids})
    return [id_to_set[i] for i in questions["image_id"]]


def build_scene_graphs(raw_dir: Path) -> tuple[dict, list[dict]]:
    questions = pd.read_csv(raw_dir / "questions.csv")

    scene_json = raw_dir / "train_sceneGraphs.json"
    if not scene_json.exists():
        with zipfile.ZipFile(raw_dir / "sceneGraphs.zip") as zf:
            zf.extract("train_sceneGraphs.json", raw_dir)
    with open(scene_json) as f:
        scenes = json.load(f)

    split = _scene_graphs_split(questions)

    graphs = {}
    needed = [str(i) for i in questions["image_id"].unique()]
    empty_images = []
    for image_id in needed:
        node_texts, edge_texts, edge_index = _textualize_scene_graph(scenes[image_id])
        if len(node_texts) == 0:  # original prints "Empty graph, skipping image id"
            empty_images.append(image_id)
            continue
        graphs[image_id] = {
            "node_texts": node_texts,
            "edge_texts": edge_texts,
            "edge_index": edge_index,
        }
    if empty_images:
        print(f"[scene_graphs] {len(empty_images)} empty scene graphs skipped: {empty_images[:5]}...")

    examples = []
    for i, row in questions.iterrows():
        key = str(row["image_id"])
        if key not in graphs:
            continue  # question over an empty graph (none expected; guarded anyway)
        examples.append(
            {
                "id": int(i),
                "graph_key": key,
                "question": row["question"],
                "answer": row["answer"],
                "full_answer": row.get("full_answer", ""),
                "split": split[i],
            }
        )
    return graphs, examples


# ---------------------------------------------------------------------------
# WebQSP
# ---------------------------------------------------------------------------

def build_webqsp(raw_dir: Path) -> tuple[dict, list[dict]]:  # noqa: ARG001 (loads from HF)
    """Port of preprocess/webqsp.py::step_one + generate_split.

    Data comes from HF `rmanluo/RoG-webqsp`, train/validation/test concatenated in
    that order; the empty validation graph at concatenated index 2937 is dropped
    (the original's "Fix bug: remove the indices of the empty graphs").
    """
    from datasets import concatenate_datasets, load_dataset

    ds = load_dataset("rmanluo/RoG-webqsp")
    sizes = {k: len(ds[k]) for k in ("train", "validation", "test")}
    full = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])

    val_lo = sizes["train"]
    test_lo = sizes["train"] + sizes["validation"]
    dropped = {2937}

    graphs, examples = {}, []
    for i in range(len(full)):
        row = full[i]
        nodes: dict[str, int] = {}
        src_ids, dst_ids, edge_texts = [], [], []
        for h, r, t in row["graph"]:
            h = (h or "").lower()
            t = (t or "").lower()
            if h not in nodes:
                nodes[h] = len(nodes)
            if t not in nodes:
                nodes[t] = len(nodes)
            src_ids.append(nodes[h])
            dst_ids.append(nodes[t])
            edge_texts.append(r)
        if len(nodes) == 0:
            if i not in dropped:
                print(f"[webqsp] unexpected empty graph at index {i} (split kept per original)")
        if i in dropped:
            continue
        graphs[str(i)] = {
            "node_texts": list(nodes.keys()),
            "edge_texts": edge_texts,
            "edge_index": torch.tensor([src_ids, dst_ids], dtype=torch.long).reshape(2, -1),
        }
        split = "train" if i < val_lo else ("val" if i < test_lo else "test")
        examples.append(
            {
                "id": int(i),
                "graph_key": str(i),
                "question": row["question"],
                # original: label = ('|').join(data['answer']).lower()
                "answer": "|".join(row["answer"]).lower(),
                "q_entity": [str(e) for e in (row.get("q_entity") or [])],
                "split": split,
            }
        )
    return graphs, examples


BUILDERS = {
    "expla_graphs": build_expla_graphs,
    "scene_graphs": build_scene_graphs,
    "webqsp": build_webqsp,
}
