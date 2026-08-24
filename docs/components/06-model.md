# 06 — Full model assembly

Spec reference: `ReGraph.md` §2.2, §2.5.
Module: `src/regraph/model.py`

## 6.1 The interleaved forward pass

$$Z_{\text{out}}^{(s)} = F_T \circ \Gamma_{T-1} \circ F_{T-1} \circ \cdots \circ \Gamma_0 \circ F_0\left(Z_{\text{in}}^{(s)}\right)$$

where

$$\Gamma_t(Z, H, P) = \operatorname{Replace}\left(Z, \mathcal I_B, \operatorname{Fuse}\left(B, \operatorname{Read}(B, H, P)\right)\right),\quad B = Z[\mathcal I_B]$$

Per round $t = 0,\dots,T-1$:

```
b_pre  = gather(hidden, b_positions)          # B_pre^t
r      = Read(b_pre, h, edges)                # R^t
b_post = Fuse(b_pre, r)                       # B_post^t
hidden = Replace(hidden, b_positions, b_post) # Z_post^t
hidden = F[t+1](hidden)                       # Z_pre^{t+1}
```

Then `norm → lm_head`. Under Llama, $b_{\text{vocab}} = 0$.

Read the equations in `ReGraph.md` §2.2 alongside this — the intuition
("$B_{\mathrm{pre}}^0$: what graph information appears necessary after reading only the
instruction; $B_{\mathrm{pre}}^1$: after considering the first graph read; …") is what the loop
must literally realize: **read graph → reason in language space → form a new graph query → read
again.**

## 6.2 `ReGraph(nn.Module)`

```python
class ReGraph(nn.Module):
    llm: LlamaForCausalLM            # frozen, bf16
    graph_encoder: GraphEncoder      # 02
    role_emb: nn.Embedding           # 02
    b_base: nn.Parameter             # 03, [N_B, d_llm]
    reader: TopologyDiffusedReader   # 04, shared across rounds
    fuse: Fuse                       # 05, shared across rounds

    def forward(self, batch) -> ReGraphOutput:   # loss, logits, gates, alphas
```

Order of operations in `forward`:

1. `h_base = graph_encoder(x, edge_index, edge_attr, node_batch)` → dense `[B, n_max, d_graph]`
2. `h = h_base + role_emb(roles)`, zeroed at `~node_mask`
3. Precompute `k_h`, `v_h` once (reader params are shared — see `04-reader.md` §4.4)
4. Build `inputs_embeds` with `b_base` scattered at `b_positions`
5. Run the interleaved layer loop (`03-query-tokens.md` §3.3)
6. `logits`, shifted cross-entropy on `labels`
7. Return gates and hop weights per round for logging

## 6.3 Trainable parameter set

Trainable (`ReGraph.md` §3.2): graph encoder $E_\phi$, `b_base`, the Topology-Diffused Graph
Reader, `Fuse`, and the graph→LLM projections (`W_O` is the projection; the role embedding and the
reader LayerNorms are part of these modules).

Frozen: **all** original LLM parameters, including `embed_tokens`, `lm_head`, and every RMSNorm.

Sanity check at startup — log the count and assert it is in the expected range:

| Module | approx. params |
|---|---|
| Graph encoder (4 × TransformerConv, 1024) | ~17 M |
| Reader (`W_Q`, `W_K`, `W_V`, `W_O`, `W_α`, LNs) | ~10.6 M |
| Fuse (`w_g`, LNs) | ~0.02 M |
| `b_base` (8 × 4096) | 0.03 M |
| **Total trainable** | **≈ 28 M** |

If the count is off by an order of magnitude, something is unfrozen or mis-shaped.

## 6.4 Training/inference parity

During teacher forcing the whole sequence `[q ; B ; boundary ; answer]` runs in one pass, while at
inference the answer is generated incrementally. These agree because:

- The graph-query tokens sit **before** the answer, so under the causal mask their hidden states
  never depend on any $y_s$.
- $\Gamma_t$ modifies the residual stream only at `b_positions`, after group $F_t$. Answer tokens
  inside groups $F_0..F_t$ attend to the *pre-replacement* B-token states at those depths in both
  training and inference — that is the same computation, not a discrepancy.

This is precisely the property that makes the KV-cache scheme in `08-inference.md` exact. Do not
break it by moving the B tokens after the answer or by conditioning `Read` on answer positions.

## Acceptance tests

1. Forward on one example returns a finite loss; `logits.shape == (B, S, vocab)`.
2. After `loss.backward()`: `grad is not None` for `b_base`, reader, fuse, graph encoder;
   `grad is None` for every LLM parameter.
3. `num_rounds = 0` reproduces the frozen LLM with soft prompt tokens (no graph path).
4. Permuting the node order of a graph (with `edge_index` permuted consistently) leaves the loss
   unchanged to `atol=1e-3` — the model must be permutation-equivariant in the nodes.
5. Zeroing `h` changes the loss once `W_O` is trained but not at step 0.
6. Memory smoke test: batch of 4 WebQSP-sized graphs forward+backward within the GPU budget with
   gradient checkpointing on.
