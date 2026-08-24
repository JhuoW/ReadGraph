# 08 — Inference with KV caching

Spec reference: `ReGraph.md` §2.5 ("Efficient inference with KV caching").
Module: `src/regraph/model.py::generate`

## 8.1 Why the graph is read only once

The sequence is $[q_1,\dots,q_{N_q}, B_{\text{base}}, y_0, y_1, \dots, y_{s-1}]$. The graph-query
tokens precede every generated token, so under the causal mask they cannot attend to any $y_s$.
Their hidden states are therefore invariant during decoding:

$$B_{\mathrm{pre}}^{t,(s)} = B_{\mathrm{pre}}^{t,(1)},\qquad \forall s\ge 1,\ t=0,\dots,T-1$$

So all $T$ Read-Fuse-Replace operations run **once, during prefill**. Decoding is then a completely
standard KV-cached Llama loop. Newly generated tokens still access the graph evidence by attending
to the cached B-token keys and values.

## 8.2 Prefill

Run `[q ; B_base ; boundary]` through
`F_0 → Γ_0 → F_1 → … → Γ_{T-1} → F_T`, with `use_cache=True`, producing the distribution of $y_1$
and storing keys/values for every prefix position at every layer.

The subtlety worth internalizing: for each $t$, the cached B-position keys/values **inside group
$F_{t+1}$** are computed *after* $\Gamma_t$ has run, so they already encode the graph evidence
injected at round $t$. Cached B-position keys/values inside groups $F_0..F_t$ encode the state
before that injection — which is exactly what the full-recompute path would also produce, because
the injection happens later in depth. Nothing is stale; nothing needs invalidating.

## 8.3 Decoding

For $s \ge 2$, propagate only the newly appended token $y_{s-1}$ through $F_0,\dots,F_T$ while
attending to the stored cache. **Do not** recompute $\Gamma_t$: it writes only to B positions,
which are not in the incremental slice. Guard this with an explicit branch:

```python
if past_key_values is None:        # prefill
    hidden = interleaved_forward(...)   # runs all Γ_t
else:                              # decode
    hidden = plain_forward(...)         # no Γ_t at all
```

Decoding settings (`ReGraph.md` §3.2, matching G-Retriever): **greedy**, sampling disabled, stop at
`EOS` or **32 generated tokens**.

Implement `generate` manually rather than through `model.generate` — the custom prefill makes the
HF generation loop awkward to hook. A ~40-line greedy loop over the cache is clearer and easier to
verify. Batched generation is fine with right-padded prompts as long as `position_ids` and the
cache offsets are computed from `attention_mask`.

## 8.4 Test-time inputs

At test time ReGraph receives only $(G_i, q_i)$ — no gold answer, no graph text in the prompt, no
retrieval. The node memory $H$ and transition operator $P$ are built exactly as in training.

## Acceptance tests

1. **Cache equivalence (the critical test).** Implement a slow reference `generate_naive` that
   re-runs the *entire* interleaved forward on the full sequence at every step. On ≥20 examples
   from each dataset, greedy outputs must be **token-identical** to the cached path. If they
   diverge, the KV-cache argument is broken — fix it before reporting any numbers.
2. B-token invariance: capture `b_pre` at every round during prefill and after 5 decoding steps
   under the naive path; assert they are identical.
3. Stopping: generation halts at `EOS`, and never exceeds 32 new tokens.
4. Batch invariance: generating an example alone and inside a padded batch of 4 gives the same
   string.
5. Speed sanity: cached decoding is at least ~5× faster per token than the naive path at 32 tokens.
