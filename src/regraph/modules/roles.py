"""Query-role embedding: h_v = h_v^base + e_role(r_v(q)) (`ReGraph.md` §2.1).

`nn.Embedding(4, d_graph)`, index order [none, mentioned, source, target],
zero-initialized so roles are a no-op at step 0 (docs/components/02-graph-encoder.md §2.2).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from regraph.data.roles import NUM_ROLES


class RoleEmbedding(nn.Module):
    def __init__(self, d_graph: int):
        super().__init__()
        self.emb = nn.Embedding(NUM_ROLES, d_graph)
        nn.init.zeros_(self.emb.weight)

    def forward(self, roles: torch.Tensor) -> torch.Tensor:
        """roles int64 [...] -> [..., d_graph]."""
        return self.emb(roles)
