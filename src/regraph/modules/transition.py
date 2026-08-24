"""Transition operator P = D̃^{-1}(A + I) in sparse padded layout (`ReGraph.md` §2.1).

P is never materialized densely, and P^k is never formed — the diffusion recursion
S^{(k)} = S^{(k-1)} P is applied edge-by-edge (docs/components/02-graph-encoder.md §2.3).

Directedness: the spec says only "let A denote the adjacency matrix". Default is to
symmetrize before adding self-loops (docs/OPEN-QUESTIONS.md Q2); this affects
diffusion only — the graph encoder consumes the original directed `edge_index`.
"""

from __future__ import annotations

import torch


def build_transition_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    symmetrize: bool = True,
    add_self_loops: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Edge list of P = D̃^{-1}(A + I) for one graph, in *local* node indices.

    Returns (src, dst, w) with w[e] = 1 / d̃(src[e]). Duplicate directed edges are
    coalesced so A is binary; self-loops guarantee d̃ >= 1, so P is well defined.
    """
    assert num_nodes >= 1, "graphs must have at least one node"
    ei = edge_index.long()
    if symmetrize and ei.numel() > 0:
        ei = torch.cat([ei, ei.flip(0)], dim=1)
    parts = [ei[0] * num_nodes + ei[1]] if ei.numel() > 0 else []
    if add_self_loops:
        loop = torch.arange(num_nodes, dtype=torch.long)
        parts.append(loop * num_nodes + loop)
    keys = torch.unique(torch.cat(parts))
    src = keys // num_nodes
    dst = keys % num_nodes
    deg = torch.zeros(num_nodes, dtype=torch.float32).index_add_(
        0, src, torch.ones_like(src, dtype=torch.float32)
    )
    w = 1.0 / deg[src]
    return src, dst, w


def diffuse_once(
    s_flat: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    """One right-multiplication by P: (SP)_{iv} = sum_u S_{iu} P_{uv}.

    §2.1: S^{(k+1)} = S^{(k)} P. Mass flows from `src` to `dst`, weighted by
    1/d̃(src). `s_flat` is [B * n_max, R] fp32 with reader rows flattened into R;
    padded rows have no incident edges and stay exactly zero.
    """
    assert s_flat.dtype == torch.float32, "diffusion must run in fp32 (00-conventions.md)"
    out = torch.zeros_like(s_flat)
    out.index_add_(0, dst, s_flat.index_select(0, src) * w.unsqueeze(-1))
    return out
