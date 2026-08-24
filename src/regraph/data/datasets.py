"""GraphQA datasets over the preprocessed store (docs/components/01-data.md).

Each item carries the tokenized prompt pieces and the full attributed graph.
The graph is never rendered into text — no retrieval, no subgraph selection,
no textual description of nodes or edges in the prompt.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from regraph.data.attr_encoder import AttrStore, encoder_slug


class GraphQADataset(Dataset):
    """One split of a preprocessed GraphQA dataset.

    `mode="train"` items carry answer tokens (teacher forcing); `mode="eval"` items
    carry the prompt only — at test time ReGraph receives only (G_i, q_i)
    (`ReGraph.md` §3.2).
    """

    def __init__(
        self,
        cfg: dict,
        split: str,
        tokenizer,
        mode: str = "train",
        keep_node_texts: bool = False,
        target_field: str | None = None,
    ):
        assert split in ("train", "val", "test")
        assert mode in ("train", "eval")
        self.cfg = cfg
        self.split = split
        self.mode = mode
        self.tokenizer = tokenizer
        self.keep_node_texts = keep_node_texts

        data_cfg = cfg["data"]
        store_dir = (
            Path(data_cfg["cache_dir"])
            / cfg["dataset"]["name"]
            / encoder_slug(data_cfg["attr_encoder"])
        )
        if not (store_dir / "graphs.pt").exists():
            raise FileNotFoundError(
                f"no preprocessed data at {store_dir} — run "
                f"`python -m regraph.data.preprocess --config configs/{cfg['dataset']['name']}.yaml`"
            )
        self.store = AttrStore(store_dir)
        self.graphs = torch.load(store_dir / "graphs.pt", weights_only=True)
        with open(store_dir / "examples.json") as f:
            all_examples = json.load(f)
        self.examples = [ex for ex in all_examples if ex["split"] == split]

        # `target_field` selects what the model is trained to generate. "answer" is the task
        # target; the alignment stage points it at a per-node text (e.g. the paper title) so
        # the graph->LLM projection learns a decodable map before task training.
        self.target_field: str = target_field or data_cfg.get("target_field", "answer")
        self.prompt_template: str = data_cfg["prompt_template"]
        self.boundary_ids: list[int] = tokenizer(
            data_cfg["answer_boundary"], add_special_tokens=False
        )["input_ids"]
        self.max_question_tokens: int = int(data_cfg.get("max_question_tokens", 512))
        # Optional TOKEN CHANNEL: additionally render the graph as text into the prompt, the
        # way G-Retriever and GRAFF do. This *augments* rather than replaces the Read-Fuse-
        # Replace rounds — both channels are active. `ReGraph.md` §3.2 keeps the graph out of
        # the context, so this is a deviation (docs/OPEN-QUESTIONS.md Q23) and exists to test
        # whether iterative graph reading adds anything on top of serialization.
        self.serialize_graph: bool = bool(data_cfg.get("serialize_graph", False))
        self.max_answer_tokens: int = int(data_cfg.get("max_answer_tokens", 32))
        self.eos_id: int = tokenizer.eos_token_id

    def _serialize(self, g: dict) -> str:
        """CSV rendering of nodes and edges, mirroring G-Retriever's `desc` format."""
        texts = self.store.texts
        nid, eid, ei = g["node_text_id"], g["edge_text_id"], g["edge_index"].long()
        lines = ["node_id,node_attr"]
        lines += [f"{i},{texts[int(t)]}" for i, t in enumerate(nid)]
        lines += ["", "src,edge_attr,dst"]
        lines += [f"{int(a)},{texts[int(t)]},{int(b)}"
                  for a, b, t in zip(ei[0].tolist(), ei[1].tolist(), eid.tolist())]
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        ex = self.examples[index]
        g = self.graphs[ex["graph_key"]]

        prompt = self.prompt_template.format(question=ex["question"])
        if self.serialize_graph:
            # Budget the GRAPH text, never the question. Plain truncation of
            # "graph + question" silently drops the question on long graphs (SceneGraphs
            # p90 is ~1700 tokens), which would leave the model answering blind.
            tail = self.tokenizer("\n" + prompt, add_special_tokens=False)["input_ids"]
            budget = max(self.max_question_tokens - len(tail) - 1, 0)
            head = self.tokenizer(
                self._serialize(g), add_special_tokens=False,
                truncation=True, max_length=budget,
            )["input_ids"]
            bos = self.tokenizer.bos_token_id
            q_ids = ([bos] if bos is not None else []) + head + tail
        else:
            q_ids = self.tokenizer(
                prompt, add_special_tokens=True, truncation=True,
                max_length=self.max_question_tokens,
            )["input_ids"]

        target = ex[self.target_field] if self.target_field in ex else ex["answer"]
        answer_ids: list[int] = []
        if self.mode == "train":
            # answer-only target " {target}" + EOS; truncation to max_answer_tokens
            # mirrors G-Retriever's label handling (docs/OPEN-QUESTIONS.md Q12)
            answer_ids = self.tokenizer(
                f" {target}", add_special_tokens=False, truncation=True,
                max_length=self.max_answer_tokens,
            )["input_ids"] + [self.eos_id]

        node_ids = g["node_text_id"]
        n = int(node_ids.shape[0])
        roles = torch.zeros(n, dtype=torch.long)
        for i, r in ex.get("roles", []):
            roles[i] = r

        item = {
            "id": ex["id"],
            "question": ex["question"],
            "answer": target,
            "q_ids": torch.tensor(q_ids, dtype=torch.long),
            "boundary_ids": torch.tensor(self.boundary_ids, dtype=torch.long),
            "answer_ids": torch.tensor(answer_ids, dtype=torch.long),
            "x": self.store.embed(node_ids),                    # [n, d_attr] fp32
            "edge_index": g["edge_index"].long(),               # [2, e] directed, as given
            "edge_attr": self.store.embed(g["edge_text_id"]),   # [e, d_attr] fp32
            "roles": roles,                                     # [n]
        }
        if self.keep_node_texts:
            item["node_texts"] = [self.store.texts[i] for i in node_ids.tolist()]
        return item
