# 04 — Topology-Diffused Graph Reader (`Read`)

Spec reference: `ReGraph.md` §2.3 — read it in full before implementing.
Module: `src/regraph/modules/reader.py`

$$R^t = \operatorname{Read}(B_{\mathrm{pre}}^t, H, P)$$

"Use the instruction-conditioned query tokens to ask the graph what information is relevant, and
retrieve it from node representations while respecting the graph topology."

## 4.1 Canonical single-head definition

$$
\begin{aligned}
S^{t,(0)} &= \operatorname{softmax}_V\!\left(\frac{\mathrm{LN}(B_{\mathrm{pre}}^t)W_Q\,(\mathrm{LN}(H)W_K)^\top}{\sqrt{d_r}}\right)\\
S^{t,(k)} &= S^{t,(0)}P^k,\quad k=1,\dots,K\\
\alpha^t &= \operatorname{softmax}_{\mathrm{hop}}\!\left(\mathrm{LN}(B_{\mathrm{pre}}^t)W_\alpha + b_\alpha\right)\\
\tilde S^t &= \sum_{k=0}^{K}\operatorname{Diag}(\alpha^t_{:,k})\,S^{t,(k)}\\
R^t &= \tilde S^t V_H W_O
\end{aligned}
$$

Shapes: $W_Q \in \mathbb{R}^{d\times d_r}$, $W_K, W_V \in \mathbb{R}^{d_g\times d_r}$,
$W_O \in \mathbb{R}^{d_r\times d}$, $\alpha^t \in \mathbb{R}^{N_B\times(K+1)}$,
$R^t \in \mathbb{R}^{N_B\times d}$.

Three properties are load-bearing and must be preserved:

- $S^{t,(0)}$ is row-stochastic over **nodes** (softmax over $V$, not over query tokens).
- $P$ is row-stochastic, so each $S^{t,(k)}$ stays row-stochastic.
- $\sum_k \alpha^t_{ik} = 1$, so $\tilde S^t$ is row-stochastic. **Assert this in a test.**

Interpretation: this is a *query-conditioned polynomial graph filter applied to semantic relevance*
rather than to node features. $\alpha^t_{ik}$ is how much query token $i$ relies on evidence that
travelled $k$ hops, predicted per token and per round — attribute questions concentrate on small
$k$, connectivity/community/influence questions on larger $k$.

## 4.2 Multi-head implementation (default)

`ReGraph.md` §2.3: independent Q/K/V projections per head, concatenate the evidence, apply $W_O$.
Config: `reader_heads = 8`, `d_reader = 1024` → `head_dim = 128`. Scale by `sqrt(head_dim)`.

The hop distribution is predicted **per head**: `W_alpha: d_llm → heads * (K+1)`, softmax over the
last axis, giving `alpha [B, N_B, heads, K+1]`. This is the natural lift of the single-head
formula and lets different heads read at different structural scales in the same round. A
`reader.shared_hop_weights: true` flag reduces it to one distribution shared across heads. Record
the choice in `docs/OPEN-QUESTIONS.md`.

## 4.3 Implementation sketch

```python
u  = self.ln_b(b_pre) @ self.W_Q            # [B, N_B, d_r] -> [B, h, N_B, hd]
kh = self.ln_h(h)     @ self.W_K            # [B, n_max, d_r] -> [B, h, n_max, hd]
vh = self.ln_h(h)     @ self.W_V

logits = (u @ kh.transpose(-1, -2)) / sqrt(head_dim)      # [B, h, N_B, n_max]
logits = logits.masked_fill(~node_mask[:, None, None, :], finfo.min)
s0 = logits.float().softmax(-1)                            # fp32, row-stochastic

# diffusion: reuse s_k, accumulate weighted by alpha
s_flat = s0.permute(0, 3, 1, 2).reshape(B * n_max, heads * N_B)   # [B*n_max, R]
acc    = alpha[..., 0] * s0
for k in range(1, K + 1):
    s_flat = diffuse_once(s_flat, edge_src_pad, edge_dst_pad, edge_w)
    acc   += alpha[..., k] * unflatten(s_flat)
r_hat  = acc.to(vh.dtype) @ vh                             # [B, h, N_B, hd]
r_evd  = r_hat.transpose(1, 2).reshape(B, N_B, d_r) @ self.W_O   # [B, N_B, d_llm]
```

- Compute $S^{(k)}$ **iteratively** ($S^{(k)} = S^{(k-1)}P$). Never form $P^k$.
- Keep the softmax and the recursion in fp32; cast to bf16 only at the `@ vh` step.
- Cost per round: $O(N_B \cdot n \cdot d_r)$ for attention plus $O(K \cdot |E| \cdot N_B \cdot h)$
  for diffusion — linear in edges, so WebQSP-scale graphs are fine.

## 4.4 Parameter sharing across rounds

Default (`ReGraph.md` §2.3, §3.3): **one reader shared by all $T$ rounds.** Round-to-round
differences come from the evolving input $B_{\mathrm{pre}}^t$, not from separate readers.

Consequence: $K_H$ and $V_H$ do not depend on $t$. Compute `ln_h(h) @ W_K` and `ln_h(h) @ W_V`
**once per forward pass** and pass them to every round. With `reader.share_across_rounds: false`
(per-round readers), recompute them per round.

## 4.5 Initialization

- `W_Q, W_K, W_V`: Xavier uniform.
- `W_alpha`: zeros, `b_alpha`: zeros → uniform hop distribution at init.
- `W_O`: **zeros**. Combined with `05-fuse.md`, this makes $R^t = 0$ at step 0, so the model starts
  exactly at the frozen-LLM behavior while gradients still reach `W_O`. Do not zero-init `W_V` as
  well, or the gradient to `W_O` vanishes too.

## Acceptance tests

1. **Row-stochasticity**: `s_tilde.sum(-1)` is 1 everywhere (`atol=1e-5`) for random masked
   batches, all `k` and all heads.
2. **`K = 0` reduction**: with `max_hops=0`, `Read` equals a plain masked multi-head cross-attention
   from `b_pre` to `h` (write that reference in the test).
3. **`P = I` reduction**: with a self-loop-only edge list, `s_k == s_0` for all `k`, so `s_tilde
   == s_0` regardless of `alpha`.
4. **Dense reference**: on a 10-node graph, compare `s_tilde` against explicitly built
   `s0 @ matrix_power(P_dense, k)` weighted by `alpha` (`atol=1e-5`, fp32).
5. **Masking**: `s_tilde` is exactly 0 at padded nodes; results are unchanged when `n_max` grows.
6. **Zero-init**: at step 0, `r_evidence` is exactly zero; after one optimizer step it is not.
7. **Hop selectivity (behavioral, not a hard assert)**: on a synthetic task where the answer sits
   2 hops away, learned `alpha` mass shifts toward `k=2`. Log it; don't gate CI on it.
