"""Gated Residual Evidence Fusion: B_post^t = Fuse(B_pre^t, R^t) (`ReGraph.md` §2.4).

    B̂^t  = LN_B(B_pre^t)
    R̂^t  = LN_R(R^t)
    g^t  = sigma([B̂^t || R̂^t] w_g + b_g)
    B_post^t = B_pre^t + Diag(g^t) Dropout(R^t)

The gate is computed from the *normalized* states, but the injected quantity is the
*raw* R^t. One scalar gate per graph-query token, broadcast across d. Separate
LayerNorms because B comes from the LLM and R from the graph reader
(docs/components/05-fuse.md). No extra MLP, no attention, no cross-token mixing.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Fuse(nn.Module):
    def __init__(self, d_llm: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.ln_b = nn.LayerNorm(d_llm)
        self.ln_r = nn.LayerNorm(d_llm)
        self.w_g = nn.Linear(2 * d_llm, 1)
        self.dropout = nn.Dropout(dropout)
        # 05-fuse.md §5.3: zeros -> g ≈ 0.5 at init; with the reader's zero W_O this
        # makes Fuse exactly the identity at step 0 while gradients still flow.
        nn.init.zeros_(self.w_g.weight)
        nn.init.zeros_(self.w_g.bias)

    def forward(
        self, b_pre: torch.Tensor, r: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """b_pre, r: [B, N_B, d_llm] -> (b_post [B, N_B, d_llm], gate [B, N_B, 1])."""
        assert b_pre.shape == r.shape
        gate_in = torch.cat([self.ln_b(b_pre), self.ln_r(r)], dim=-1)
        # gate in fp32 for stability (05-fuse.md §5.2), then cast for the update
        with torch.autocast(device_type=b_pre.device.type, enabled=False):
            gate = torch.sigmoid(self.w_g(gate_in.float()))          # [B, N_B, 1]
        b_post = b_pre + gate.to(b_pre.dtype) * self.dropout(r)
        return b_post, gate
