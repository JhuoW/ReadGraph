# 03 — Graph-query tokens and the LLM layer partition

Spec reference: `ReGraph.md` §2.2, §2.5.
Module: `src/regraph/model.py` (sequence assembly + `run_layer_group`)

This is the highest-risk component: it performs surgery on the middle of a HuggingFace Llama
forward pass. Build it *without* any graph reading first and prove it is a no-op, then add `Γ_t`.

## 3.1 Graph-query tokens

$$B_{\text{base}} = [b_1,\dots,b_{N_B}] \in \mathbb{R}^{N_B \times d}$$

`nn.Parameter(torch.empty(N_B, d_llm))`, initialized from the **empirical distribution of the
frozen embedding matrix** (`normal_(mean=embed.mean(), std=embed.std())`) so the tokens start
in-distribution for the LLM. `N_B = 8` (`ReGraph.md` §3.3).

Placement is **after the instruction** (`ReGraph.md` §2.2). Under the causal mask every
graph-query token attends to all instruction tokens and to preceding graph-query tokens. Do not
move them, and do not give them a custom attention mask — the ordinary causal mask is the design.

## 3.2 Layer partition

$L = 32$ Llama layers → $T+1 = 4$ consecutive groups (`ReGraph.md` §3.3):

```
F_0 = layers[0:8]   Γ_0   F_1 = layers[8:16]   Γ_1
F_2 = layers[16:24] Γ_2   F_3 = layers[24:32]  → norm → lm_head
```

Compute boundaries from `config.num_rounds` and `model.config.num_hidden_layers`; distribute the
remainder to the earlier groups if they don't divide evenly. Store as
`group_bounds: list[tuple[int, int]]` and log it.

## 3.3 Manual forward loop

Do not use forward hooks — they make the KV-cache path (see `08-inference.md`) hard to reason
about. Write an explicit loop:

```python
hidden = inputs_embeds                                    # [B, S, d_llm]
position_ids = attention_mask.cumsum(-1) - 1              # right padding
causal_mask = self.llm.model._update_causal_mask(...)     # version-dependent
pos_emb = self.llm.model.rotary_emb(hidden, position_ids) # transformers >= 4.43
for t, (lo, hi) in enumerate(self.group_bounds):
    for layer in self.llm.model.layers[lo:hi]:
        hidden = layer(hidden, attention_mask=causal_mask,
                       position_ids=position_ids,
                       position_embeddings=pos_emb,
                       past_key_value=cache, use_cache=use_cache)[0]
    if t < self.num_rounds:                               # Γ_t after F_t, t = 0..T-1
        hidden = self.graph_round(t, hidden, h, edges, b_positions)
hidden = self.llm.model.norm(hidden)
logits = self.llm.lm_head(hidden)
```

**Verify the signatures against the installed `transformers`.** `_update_causal_mask`, the rotary
embedding location, and `past_key_value` vs `past_key_values` have all moved between releases. Pin
the version in `requirements.txt` and read `modeling_llama.py` before writing this loop.

## 3.4 Gather and Replace

```python
b_idx = torch.arange(B, device=...).unsqueeze(1)            # [B, 1]
b_pre  = hidden[b_idx, b_positions]                         # [B, N_B, d_llm]   (gather)
hidden = hidden.index_put((b_idx, b_positions), b_post)     # Replace, out-of-place
```

Use `index_put` (not in-place `hidden[...] = ...` on a tensor that autograd needs) so the graph
stays clean under gradient checkpointing. `Replace` overwrites the residual stream only at
`b_positions`; every other position is untouched.

## 3.5 Gradient checkpointing

Checkpoint at the granularity of individual decoder layers, not whole groups, and **do not**
checkpoint `Γ_t` (it is cheap and re-entrancy interacts badly with the index_put). Use
`use_reentrant=False`.

## Acceptance tests

1. **No-op equivalence (the critical test).** With `graph_round` replaced by the identity and
   `b_base` scattered into the sequence, `manual_forward(inputs_embeds).logits` matches
   `llm(inputs_embeds=...).logits` to `atol=2e-2` in bf16 / `1e-5` in fp32.
2. `b_pre` gathered immediately after scattering `b_base` into `inputs_embeds` and running zero
   layers equals `b_base` broadcast over the batch.
3. Replace touches only `b_positions`: after a round with `b_post = b_pre + 1`, exactly
   `B * N_B` entries of `hidden` changed.
4. Ragged batch: examples with different prompt lengths gather the correct slots (assert the
   placeholder token id sits at every `b_positions`).
5. Every Llama parameter has `requires_grad == False` and `grad is None` after `loss.backward()`.
