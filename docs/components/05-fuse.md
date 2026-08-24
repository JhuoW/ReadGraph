# 05 — Gated Residual Evidence Fusion (`Fuse`)

Spec reference: `ReGraph.md` §2.4.
Module: `src/regraph/modules/fuse.py`

The reader has already decided *what* evidence is relevant. Fuse decides only *how much* of it is
written into the graph-query state. Keep it that simple — no extra MLP, no attention, no
cross-token mixing.

## 5.1 Definition

$$
\begin{aligned}
\widehat B^t &= \mathrm{LN}_B(B_{\mathrm{pre}}^t)\\
\widehat R^t &= \mathrm{LN}_R(R^t)\\
g^t &= \sigma\!\left(\left[\widehat B^t \,\|\, \widehat R^t\right] w_g + b_g\right)\\
B_{\mathrm{post}}^t &= B_{\mathrm{pre}}^t + \operatorname{Diag}(g^t)\,\mathrm{Dropout}(R^t)
\end{aligned}
$$

with $w_g \in \mathbb{R}^{2d}$, $b_g \in \mathbb{R}$, $g^t \in (0,1)^{N_B}$.

Two details that are easy to get wrong:

1. **Separate LayerNorms.** $B_{\mathrm{pre}}^t$ comes from the LLM and $R^t$ from the graph reader;
   their feature distributions differ, so they are normalized independently before being
   concatenated.
2. **The gate is computed from the normalized states, but the *injected* quantity is the raw
   $R^t$**, not $\widehat R^t$. Implement exactly as written.

$g^t$ is **one scalar per graph-query token**, broadcast across all $d$ dimensions:
$b_{\mathrm{post},i}^t = b_{\mathrm{pre},i}^t + g_i^t r_i^t$.

- $g_i^t \approx 0$: the token keeps its current LLM state (graph read ignored).
- $g_i^t \approx 1$: the full graph evidence is injected.
- in between: partial integration.

The gate therefore controls the strength of graph intervention per token and per round — log its
mean and histogram per round during training; it is the single most informative diagnostic in the
model. A gate that saturates at 0 for all rounds means the graph path is dead.

## 5.2 Implementation notes

```python
class Fuse(nn.Module):
    def forward(self, b_pre, r):                     # both [B, N_B, d_llm]
        gate_in = torch.cat([self.ln_b(b_pre), self.ln_r(r)], dim=-1)   # [B, N_B, 2d]
        gate = torch.sigmoid(self.w_g(gate_in))                          # [B, N_B, 1]
        return b_pre + gate * self.dropout(r), gate
```

- `self.w_g = nn.Linear(2 * d_llm, 1)` — output width 1, giving the scalar gate.
- `dropout = 0.1` (`ReGraph.md` §3.3); active in training only.
- Shared across rounds by default, matching the reader (`ReGraph.md` §3.3: "Reader and Fuse
  parameters are shared across rounds").
- Compute the gate in fp32 for stability, then cast.
- Return the gate so the training loop can log it.

## 5.3 Initialization

`w_g` zeros and `b_g` zero → $g \approx 0.5$ at init. Combined with the zero-initialized `W_O` in
`04-reader.md`, $R^t = 0$ at step 0 so `Fuse` is exactly the identity, yet gradients flow to both
`W_O` and `w_g` immediately. Do **not** additionally initialize `b_g` to a large negative value —
that stacks two independent "start at zero" mechanisms and starves the gate of gradient.

## Acceptance tests

1. Shapes: `b_post.shape == b_pre.shape`; `gate.shape == (B, N_B, 1)`; `0 < gate < 1`.
2. Zero evidence: `Fuse(b_pre, torch.zeros_like(r)) == b_pre` exactly (eval mode).
3. Identity at init: with the reader's `W_O` zero-initialized, `b_post == b_pre` at step 0.
4. Gate extremes: forcing `w_g` large-positive/large-negative gives `b_post ≈ b_pre + r` and
   `b_post ≈ b_pre` respectively.
5. Per-token independence: changing `r[0, 3]` changes `b_post[0, 3]` and no other row.
6. Raw-vs-normalized: the injected term equals `gate * r`, **not** `gate * ln_r(r)` — assert
   numerically against a hand-computed value.
