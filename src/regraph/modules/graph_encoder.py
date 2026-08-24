"""Graph encoder E_phi: H_base = E_phi(G, C) (`ReGraph.md` §2.1, §3.2).

Default: relation-aware graph transformer, 4 layers, 1024-d node memory —
PyG `TransformerConv` with `edge_dim`, which is exactly the relation-aware form
(edge features enter both keys and values). Per layer:
conv -> residual add -> LayerNorm -> GELU -> dropout
(docs/components/02-graph-encoder.md, docs/OPEN-QUESTIONS.md Q6).

Encoders are registered by name in a factory; the graph->LLM interface does not
change when the encoder changes (`ReGraph.md` §2.1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv


class RelationAwareGraphTransformer(nn.Module):
    """H_base = E_phi(G, C): flat node memory [sum n_i, d_graph] from PyG-style inputs."""

    def __init__(
        self,
        d_attr: int = 1024,
        d_graph: int = 1024,
        num_layers: int = 4,
        heads: int = 4,
        edge_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_graph % heads == 0
        # identity-shaped when d_attr == d_graph, but kept for encoder swaps (02 §2.1)
        self.input_proj = nn.Linear(d_attr, d_graph)
        self.convs = nn.ModuleList(
            TransformerConv(
                in_channels=d_graph,
                out_channels=d_graph // heads,
                heads=heads,
                edge_dim=edge_dim,
                beta=False,
            )
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(d_graph) for _ in range(num_layers))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.d_graph = d_graph

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        assert x.dim() == 2 and edge_index.shape[0] == 2
        assert edge_attr.shape[0] == edge_index.shape[1]
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = self.dropout(self.act(norm(h + conv(h, edge_index, edge_attr))))
        assert h.shape == (x.shape[0], self.d_graph)
        return h


_REGISTRY = {"graph_transformer": RelationAwareGraphTransformer}


def build_graph_encoder(cfg: dict) -> nn.Module:
    """Factory keyed by `graph_encoder.name`; swappable per `ReGraph.md` §2.1."""
    enc_cfg = cfg["graph_encoder"]
    name = enc_cfg["name"]
    if name not in _REGISTRY:
        raise KeyError(f"unknown graph encoder {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](
        d_attr=cfg["data"]["d_attr"],
        d_graph=enc_cfg["d_graph"],
        num_layers=enc_cfg["num_layers"],
        heads=enc_cfg["heads"],
        edge_dim=enc_cfg["edge_dim"],
        dropout=enc_cfg["dropout"],
    )
