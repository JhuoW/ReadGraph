"""Topology-Diffused Graph Reader: R^t = Read(B_pre^t, H, P) (`ReGraph.md` §2.3).

Canonical definition (§2.3):

    S^{t,(0)} = softmax_V( LN(B_pre^t) W_Q (LN(H) W_K)^T / sqrt(d_r) )
    S^{t,(k)} = S^{t,(0)} P^k,   k = 1..K          (computed iteratively, never P^k)
    alpha^t   = softmax_hop( LN(B_pre^t) W_alpha + b_alpha )
    S~^t      = sum_k Diag(alpha^t_{:,k}) S^{t,(k)}
    R^t       = S~^t V_H W_O

Multi-head (default): independent Q/K/V per head, concatenate evidence, apply W_O;
hop weights predicted per head (docs/components/04-reader.md §4.2, OPEN-QUESTIONS Q4).

Load-bearing invariants: S^{t,(0)} is row-stochastic over nodes; P is row-stochastic,
so every S^{t,(k)} stays row-stochastic; sum_k alpha_{ik} = 1, so S~^t is
row-stochastic. Softmax over nodes and the diffusion recursion run in fp32
(00-conventions.md), cast back after S~.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from regraph.modules.transition import diffuse_once


class TopologyDiffusedReader(nn.Module):
    def __init__(
        self,
        d_llm: int = 4096,
        d_graph: int = 1024,
        d_reader: int = 1024,
        heads: int = 8,
        max_hops: int = 2,
        shared_hop_weights: bool = False,
        w_o_init: str = "zeros",
        w_o_init_std: float = 1.0e-3,
    ):
        super().__init__()
        assert d_reader % heads == 0
        self.d_llm, self.d_graph, self.d_reader = d_llm, d_graph, d_reader
        self.heads, self.head_dim = heads, d_reader // heads
        self.max_hops = max_hops
        self.shared_hop_weights = shared_hop_weights

        self.ln_b = nn.LayerNorm(d_llm)
        self.ln_h = nn.LayerNorm(d_graph)
        self.w_q = nn.Linear(d_llm, d_reader, bias=False)
        self.w_k = nn.Linear(d_graph, d_reader, bias=False)
        self.w_v = nn.Linear(d_graph, d_reader, bias=False)
        self.w_o = nn.Linear(d_reader, d_llm, bias=False)
        hop_out = (1 if shared_hop_weights else heads) * (max_hops + 1)
        self.w_alpha = nn.Linear(d_llm, hop_out, bias=True)

        # 04-reader.md §4.5: Xavier for Q/K/V; zeros for W_alpha (uniform hops at init).
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.zeros_(self.w_alpha.weight)
        nn.init.zeros_(self.w_alpha.bias)

        # W_O init. `zeros` is the 04-reader.md §4.5 default: R^t = 0 at step 0, so the
        # model starts exactly at frozen-LLM behavior. It has a cost the doc does not
        # anticipate: since R = S~ V_H W_O, we get dL/dS~ = (dL/dR)(V_H W_O)^T, so with
        # W_O = 0 the *node-selection* weights W_Q/W_K receive exactly zero gradient at
        # step 0 and only ||W_O||-scaled gradient after, throttling the reader's ability
        # to learn where to attend (measured: W_Q/W_K stay at 1.00x their init scale).
        # `normal` breaks that bootstrap with a small random map, trading the exact
        # identity-at-init property for a trainable attention path.
        # See docs/OPEN-QUESTIONS.md Q17.
        if w_o_init == "zeros":
            nn.init.zeros_(self.w_o.weight)
        elif w_o_init == "normal":
            nn.init.normal_(self.w_o.weight, mean=0.0, std=w_o_init_std)
        else:
            raise ValueError(f"unknown reader.w_o_init {w_o_init!r}; use zeros|normal")

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        """[B, S, d_reader] -> [B, heads, S, head_dim]."""
        bsz, s, _ = t.shape
        return t.view(bsz, s, self.heads, self.head_dim).permute(0, 2, 1, 3)

    def precompute(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """K_H, V_H from the node memory. Parameters are shared across rounds, so
        this runs once per forward pass (04-reader.md §4.4)."""
        h_ln = self.ln_h(h)
        return self._split_heads(self.w_k(h_ln)), self._split_heads(self.w_v(h_ln))

    def forward(
        self,
        b_pre: torch.Tensor,       # [B, N_B, d_llm]
        k_h: torch.Tensor,         # [B, heads, n_max, head_dim]
        v_h: torch.Tensor,         # [B, heads, n_max, head_dim]
        node_mask: torch.Tensor,   # [B, n_max] bool
        edge_src_pad: torch.Tensor,
        edge_dst_pad: torch.Tensor,
        edge_w: torch.Tensor,      # fp32, 1/d̃(src), padded layout
        return_reading: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        bsz, n_b, _ = b_pre.shape
        n_max = k_h.shape[2]
        assert node_mask.shape == (bsz, n_max)

        b_ln = self.ln_b(b_pre)
        u = self._split_heads(self.w_q(b_ln))                       # [B, h, N_B, hd]

        # E^t_{iv} = <U_i, K_{H,v}> / sqrt(d_r-per-head); softmax over nodes, fp32
        logits = u @ k_h.transpose(-1, -2) / math.sqrt(self.head_dim)
        logits = logits.masked_fill(
            ~node_mask[:, None, None, :], torch.finfo(logits.dtype).min
        )
        s0 = logits.float().softmax(dim=-1)                         # [B, h, N_B, n_max]

        # alpha^t = softmax_hop(LN(B) W_alpha + b_alpha), per head (or shared)
        alpha = self.w_alpha(b_ln).float()
        if self.shared_hop_weights:
            alpha = alpha.view(bsz, n_b, 1, self.max_hops + 1).expand(
                bsz, n_b, self.heads, self.max_hops + 1
            )
        else:
            alpha = alpha.view(bsz, n_b, self.heads, self.max_hops + 1)
        alpha = alpha.softmax(dim=-1).permute(0, 2, 1, 3)           # [B, h, N_B, K+1]

        # S~^t = sum_k Diag(alpha_{:,k}) S^{t,(k)}, S^{(k)} = S^{(k-1)} P iteratively
        s_tilde = alpha[..., 0].unsqueeze(-1) * s0
        if self.max_hops > 0:
            s_flat = (
                s0.permute(0, 3, 1, 2).reshape(bsz * n_max, self.heads * n_b)
            )                                                        # [B*n_max, R]
            for k in range(1, self.max_hops + 1):
                s_flat = diffuse_once(s_flat, edge_src_pad, edge_dst_pad, edge_w)
                s_k = (
                    s_flat.view(bsz, n_max, self.heads, n_b).permute(0, 2, 3, 1)
                )
                s_tilde = s_tilde + alpha[..., k].unsqueeze(-1) * s_k

        # R^t = S~^t V_H W_O — evidence aggregation back in module compute dtype
        r_hat = s_tilde.to(v_h.dtype) @ v_h                          # [B, h, N_B, hd]
        r_evidence = self.w_o(
            r_hat.permute(0, 2, 1, 3).reshape(bsz, n_b, self.d_reader)
        )
        assert r_evidence.shape == (bsz, n_b, self.d_llm)

        eps = 1e-12
        diagnostics = {
            "alpha": alpha.detach(),                                  # [B, h, N_B, K+1]
            "s0_entropy": (-(s0 * (s0 + eps).log()).sum(-1)).mean().detach(),
            "s_tilde_entropy": (-(s_tilde * (s_tilde + eps).log()).sum(-1)).mean().detach(),
        }
        if return_reading:
            diagnostics["s_tilde"] = s_tilde.detach()
        return r_evidence, diagnostics
