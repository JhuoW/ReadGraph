# 02 — Graph encoder, roles, and transition operator

Spec reference: `ReGraph.md` §2.1, §3.2.
Module: `src/regraph/modules/graph_encoder.py`, `src/regraph/modules/roles.py`

This stage turns a batch of attributed graphs into two objects the reader consumes:
the **node memory** $H$ and the **transition operator** $P$. Both are computed once per forward
pass and reused by all $T$ rounds.

## 2.1 Node memory

$$H^{\text{base}} = E_\phi(G, C) \in \mathbb{R}^{n \times d_g}, \qquad C = [c_{v_1};\dots;c_{v_n}]$$

Default $E_\phi$: **relation-aware graph transformer, 4 layers, output width 1024** (`ReGraph.md`
§3.2). Use PyG `TransformerConv` with `edge_dim=1024` — it is exactly the relation-aware form
(edge features enter both keys and values):

```python
TransformerConv(in_channels=d, out_channels=d // heads, heads=heads, edge_dim=1024, beta=False)
```

Per layer: `conv → residual add → LayerNorm → GELU → dropout(0.1)`. Config: `heads=4`,
`d_graph=1024`, `num_layers=4`. Input projection maps `d_attr → d_graph` (identity when both are
1024, but keep the module for encoder swaps).

$E_\phi$ is deliberately swappable (`ReGraph.md` §2.1 lists GIN, GraphSAGE, GAT, graph
transformer). Register encoders by name in a small factory; the graph→LLM interface must not
change when the encoder changes.

## 2.2 Query-role embedding

$$h_v = h_v^{\text{base}} + e_{\text{role}}(r_v(q))$$

`nn.Embedding(4, d_graph)`, index order `[none, mentioned, source, target]`, **zero-initialized**
so that at step 0 roles are a no-op. Because $H$ depends on $q$ through the roles, it cannot be
cached across questions on the same graph unless the role vectors are identical.

Output: `h [B, n_max, d_graph]`, zeroed at `~node_mask`.

## 2.3 Transition operator $P$

$$\tilde A = A + I, \qquad P = \tilde D^{-1}\tilde A$$

Never materialize $P$ or $P^k$ — WebQSP graphs average 1,371 nodes and the padded dense form is
wasteful and, for $P^k$, wrong to precompute. Build a **sparse edge list in padded layout** at
collate time:

1. Take the example's `edge_index`, append a self-loop for every node.
2. `deg[u] = ` number of outgoing edges of `u` in $\tilde A$; `edge_w = 1 / deg[src]`.
3. Offset local node indices into the flattened `B * n_max` axis: `idx = b * n_max + local_idx`.
4. Concatenate over the batch → `edge_src_pad`, `edge_dst_pad`, `edge_w`.

Because $\tilde A$ has self-loops, no row of $\tilde D$ is zero and $P$ is always well defined.

### Diffusion helper

$S^{(k)} = S^{(k-1)}P$ means $(SP)_{iv} = \sum_u S_{iu}P_{uv}$: mass flows **from `src` to `dst`**,
weighted by $1/\tilde d_{\text{src}}$.

```python
def diffuse_once(s_flat, src, dst, w):
    """s_flat: [B * n_max, R] float32.  One multiplication by P, right-multiplied."""
    out = torch.zeros_like(s_flat)
    out.index_add_(0, dst, s_flat.index_select(0, src) * w.unsqueeze(-1))
    return out
```

`R = heads * num_query_tokens` (reader rows flattened). Pure PyTorch — no `torch_scatter`
dependency. Run in fp32. Padded rows have no incident edges, so they stay exactly zero.

### Directedness

The spec says only "let $A$ denote the adjacency matrix". ExplaGraphs and SceneGraphs edges are
directed; WebQSP triples are directed. Default: **symmetrize before adding self-loops**
(`graph.symmetrize_for_diffusion: true`), so evidence can propagate against edge direction, which
matches the intent of "propagating semantic seed mass through $k$ graph transitions". The directed
variant is a config flag. This is a documented ambiguity — see `docs/OPEN-QUESTIONS.md`, and
report the setting used in results.

Note this affects **diffusion only**. The graph encoder $E_\phi$ consumes the original directed
`edge_index`.

## Acceptance tests

1. `h.shape == (B, n_max, d_graph)`; rows at `~node_mask` are exactly zero.
2. Row-stochasticity: build a random small graph, run `diffuse_once` on `S = I`, and check every
   row sums to 1 (`atol=1e-6`, fp32).
3. Reference match: on a 12-node graph, dense `S @ P` (built with `torch.zeros` + explicit
   assignment) equals `diffuse_once(S, ...)`.
4. One-hot check: `S = e_u` diffuses to exactly `P[u, :]`.
5. Padding independence: encoding a batch of 3 graphs gives the same `h` per graph (`atol=1e-4`)
   as encoding each alone.
6. Role embedding is zero at init → `h == h_base`.
