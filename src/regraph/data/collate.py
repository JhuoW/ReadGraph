"""Batch collation (docs/components/01-data.md).

Sequence layout per example: `[T_q ; B ; boundary ; answer]` (`ReGraph.md` §2.2, §2.5),
right-padded. The B slots carry a placeholder token id so ids and embeds stay aligned;
the model embeds real tokens with the frozen `embed_tokens` and scatters `b_base` into
`b_positions` (docs/components/03-query-tokens.md).

Labels are -100 everywhere except the answer tokens and the final EOS (`ReGraph.md`
§3.2: losses on the question, graph-query tokens, answer boundary, and padding are
masked).

Nodes are padded per batch to `n_max` with `node_mask`; the transition operator P is
returned as a sparse edge list in *padded layout* — indices into the flattened
`B * n_max` axis — including self-loops, with w = 1/d̃(src)
(docs/components/02-graph-encoder.md §2.3).
"""

from __future__ import annotations

from typing import Callable

import torch

from regraph.modules.transition import build_transition_edges


def make_collate_fn(
    pad_token_id: int,
    placeholder_id: int,
    num_query_tokens: int,
    symmetrize_for_diffusion: bool = True,
    add_self_loops: bool = True,
) -> Callable[[list[dict]], dict]:
    n_b = num_query_tokens

    def collate(items: list[dict]) -> dict:
        bsz = len(items)
        seqs, b_pos_rows, label_rows = [], [], []
        for it in items:
            q, bd, ans = it["q_ids"], it["boundary_ids"], it["answer_ids"]
            n_q = q.shape[0]
            placeholder = torch.full((n_b,), placeholder_id, dtype=torch.long)
            seq = torch.cat([q, placeholder, bd, ans])
            labels = torch.full((seq.shape[0],), -100, dtype=torch.long)
            if ans.shape[0] > 0:
                labels[n_q + n_b + bd.shape[0]:] = ans
            seqs.append(seq)
            label_rows.append(labels)
            b_pos_rows.append(torch.arange(n_q, n_q + n_b, dtype=torch.long))

        s_max = max(s.shape[0] for s in seqs)
        input_ids = torch.full((bsz, s_max), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, s_max), dtype=torch.long)
        labels = torch.full((bsz, s_max), -100, dtype=torch.long)
        for i, (seq, lab) in enumerate(zip(seqs, label_rows)):
            input_ids[i, : seq.shape[0]] = seq
            attention_mask[i, : seq.shape[0]] = 1
            labels[i, : seq.shape[0]] = lab
        b_positions = torch.stack(b_pos_rows)  # [B, N_B]

        # ---- graphs: flat PyG-style arrays for the encoder ----------------------
        num_nodes = torch.tensor([it["x"].shape[0] for it in items], dtype=torch.long)
        n_max = int(num_nodes.max())
        x = torch.cat([it["x"] for it in items])                       # [sum n, d_attr]
        edge_attr = torch.cat([it["edge_attr"] for it in items])       # [sum e, d_attr]
        offsets = torch.cumsum(num_nodes, 0) - num_nodes
        edge_index = torch.cat(
            [it["edge_index"] + offsets[i] for i, it in enumerate(items)], dim=1
        )
        node_batch = torch.repeat_interleave(
            torch.arange(bsz, dtype=torch.long), num_nodes
        )

        node_mask = torch.zeros((bsz, n_max), dtype=torch.bool)
        roles = torch.zeros((bsz, n_max), dtype=torch.long)
        for i, it in enumerate(items):
            n_i = int(num_nodes[i])
            node_mask[i, :n_i] = True
            roles[i, :n_i] = it["roles"]

        # ---- transition operator P in padded layout -----------------------------
        srcs, dsts, ws = [], [], []
        for i, it in enumerate(items):
            src, dst, w = build_transition_edges(
                it["edge_index"], int(num_nodes[i]),
                symmetrize=symmetrize_for_diffusion, add_self_loops=add_self_loops,
            )
            srcs.append(src + i * n_max)
            dsts.append(dst + i * n_max)
            ws.append(w)
        edge_src_pad = torch.cat(srcs)
        edge_dst_pad = torch.cat(dsts)
        edge_w = torch.cat(ws).float()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "b_positions": b_positions,
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "node_batch": node_batch,
            "num_nodes": num_nodes,
            "node_mask": node_mask,
            "roles": roles,
            "edge_src_pad": edge_src_pad,
            "edge_dst_pad": edge_dst_pad,
            "edge_w": edge_w,
            "id": [it["id"] for it in items],
            "question": [it["question"] for it in items],
            "answer": [it["answer"] for it in items],
            **(
                {"node_texts": [it["node_texts"] for it in items]}
                if "node_texts" in items[0]
                else {}
            ),
        }

    return collate
