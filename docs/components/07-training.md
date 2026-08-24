# 07 — Training

Spec reference: `ReGraph.md` §3.2, §3.3.
Module: `src/regraph/train.py`

## 7.1 Objective

Answer-only next-token likelihood:

$$\mathcal L_{\mathrm{gen}} = -\sum_{i\in\mathcal D_{\mathrm{train}}}\sum_{s=1}^{|A_i|+1}\log p_\Theta\!\left(a_{i,s}\mid q_i, G_i, a_{i,<s}\right)$$

The $|A_i|+1$ upper limit includes the `EOS` token. Losses on the question, the graph-query tokens,
the answer boundary, and padding are masked (`labels = -100`).

**GraphQA provides no intermediate supervision** for the reading distributions $S$, hop weights
$\alpha$, fusion gates $g$, or reasoning states. Everything is learned end to end from the final
answer likelihood. Do not add auxiliary losses, entropy regularizers, or supervision on $\alpha$ —
that would break comparability with the reported protocol.

## 7.2 Optimization (do not change without flagging)

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-5 |
| Weight decay | 0.05 |
| Schedule | warm-up then cosine decay |
| Max epochs | 10 |
| Early stopping | patience 2 (on validation loss) |
| Batch size | 4 |
| Dropout | 0.1 |
| Checkpoint selection | lowest validation loss |
| Models | one per dataset, trained separately |

Warm-up length is not specified in the spec — default to 5% of total steps, put it in the config,
log it, and record it in `docs/OPEN-QUESTIONS.md`.

No weight decay on biases, LayerNorm/RMSNorm weights, `b_base`, or the role embedding.

`lr = 1e-5` is low for modules trained from scratch. If training visibly underfits (validation
loss plateaus high, gates stay near 0), **report this to the user with evidence** rather than
silently raising the learning rate. Any deviation invalidates comparability with the protocol.

## 7.3 Memory and precision

- Llama frozen in bf16 with gradient checkpointing; trainable modules fp32 under `autocast(bf16)`.
- Gradients must flow *through* the frozen LLM: `requires_grad_(False)` on parameters, **never**
  `no_grad` on the forward.
- Gradient clipping at 1.0 (global norm over trainable params).
- Gradient accumulation is allowed to reach an effective batch of 4 on smaller GPUs; log both the
  micro-batch and effective batch size.

## 7.4 What to log

Per step: loss, lr, grad norm.
Per epoch (train and val):

- mean and histogram of the fusion gate $g^t$ **per round** — the primary health signal;
- mean hop distribution $\alpha^t$ per round (and per head) — shows which structural scale each
  round uses;
- entropy of $S^{t,(0)}$ and of $\tilde S^t$ — collapsing to a single node or staying uniform are
  both failure modes;
- fraction of examples with non-`none` roles (constant per dataset; log once).

Also dump the resolved config, git SHA, seed, split sizes, and trainable parameter count into the
run directory.

## 7.5 Failure modes to watch

| Symptom | Likely cause |
|---|---|
| Gates → 0 in every round, loss ≈ frozen-LLM baseline | graph path contributes nothing: check `W_O` receives gradient, check `Replace` actually writes |
| `NaN` after a few hundred steps | node softmax or diffusion running in bf16 — force fp32 |
| $\tilde S$ rows not summing to 1 | masked nodes not excluded from the softmax, or `edge_w` not $1/\tilde d_u$ |
| Loss identical with and without the graph | `b_positions` off by one; assert the placeholder token id at those positions |
| Val loss much better than accuracy suggests | answer formatting mismatch — check the boundary string and leading space |

## Acceptance tests

1. **Overfit test**: 16 training examples, no dropout, no early stopping → training loss below 0.05
   within 200 steps. If this fails, nothing downstream is worth running.
2. Label masking: loss computed on a batch where the answer is replaced by `-100` everywhere is
   exactly 0.
3. Determinism: two runs with the same seed give identical loss for the first 10 steps.
4. Resume: save/load a checkpoint mid-epoch and confirm the next step's loss matches the
   uninterrupted run.
5. Early stopping triggers after 2 non-improving validation epochs and restores the best checkpoint.
