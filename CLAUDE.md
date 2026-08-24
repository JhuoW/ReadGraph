# CLAUDE.md — ReGraph

Implementation guide for **ReGraph: Iterative Graph Reading for Open-Ended Reasoning with LLMs**.

ReGraph interleaves `Read → Fuse → Replace` graph-reading rounds between groups of frozen LLM
layers. A small set of learnable *graph-query tokens* sits in the LLM sequence; at each round they
query a node memory produced by a graph encoder, the retrieved evidence is gated back into their
hidden states, and the LLM continues reasoning. The graph is **never serialized into the prompt**.

---

## Ground rules

1. **`./ReGraph.md` is the specification of record.** Whenever a formula, shape, or design decision
   is unclear, re-read the relevant section of `ReGraph.md` before writing code. If a component doc
   and `ReGraph.md` disagree, `ReGraph.md` wins — fix the doc and note it in
   `docs/OPEN-QUESTIONS.md`.
2. **Never silently invent** hyperparameters, architectural details, or data handling that the spec
   does not state. Append the question to `docs/OPEN-QUESTIONS.md`, pick the documented default,
   log the choice loudly at startup, and tell the user in your response.
3. **The LLM is frozen.** All Llama parameters get `requires_grad_(False)`. Never wrap the LLM
   forward in `torch.no_grad()` — gradients must flow *through* the LLM back to the reader.
4. **Notation is contractual.** Use the code names in `docs/components/00-conventions.md`. A reader
   holding both the paper and the code should be able to line them up symbol by symbol.
5. **One phase at a time.** Each phase below has exit criteria and unit tests. Do not start phase
   *n+1* until phase *n*'s tests pass. Report progress after each phase.
6. **Numbers must be reproducible.** Everything configurable lives in `configs/`; nothing is
   hard-coded in `src/`. Every run writes its resolved config, git SHA, and seed to the run dir.
7. **Scope: ReGraph itself, evaluated on the three GraphQA datasets.** No ablation variants, no
   baseline reimplementations, no extra experiments. If you think a variant is needed to debug
   something, say so and wait — do not build it unasked.

## Repo layout (target)

```
ReGraph.md              spec of record
CLAUDE.md               this file
configs/                default.yaml + one per dataset
docs/components/        per-component implementation contracts (read before coding)
docs/experimental-protocol.md
docs/OPEN-QUESTIONS.md  ambiguities + chosen defaults (append as you go)
src/regraph/
  data/     datasets.py  preprocess.py  collate.py
  modules/  graph_encoder.py  reader.py  fuse.py  roles.py
  model.py  train.py  eval.py  utils/
tests/                  pytest, one file per component
```

---

## Implementation procedure

### Phase 0 — Environment and conventions
Read `docs/components/00-conventions.md`.
Set up `pyproject.toml`/`requirements.txt`, seeding, run-directory logging. Verify the installed
`transformers` version and **read the installed `modeling_llama.py`** — the manual layer loop in
phase 3 depends on its exact signatures.
*Exit:* `python -c "import regraph"` works; `pytest tests/` collects; Llama3.1-8B loads in bf16 and
a plain forward pass runs.

### Phase 1 — Data
Read `docs/components/01-data.md` and `docs/experimental-protocol.md`.
GraphQA (ExplaGraphs, SceneGraphs, WebQSP) with G-Retriever's exact splits; attribute encoding with
`sentence-transformers/all-roberta-large-v1`; PyG `Data` objects; padded batching; prompt
construction and answer-only label masking.
*Exit:* split sizes and average node/edge counts match the table in `docs/experimental-protocol.md`;
collate round-trip tests pass.

### Phase 2 — Graph encoder, roles, and transition operator
Read `docs/components/02-graph-encoder.md`.
Relation-aware 4-layer graph transformer → node memory `H_base` (1024-d); query-role embeddings
added to give `H`; row-normalized `P = D̃⁻¹(A + I)` materialized as a padded edge list.
*Exit:* `H` shapes correct; `P` rows sum to 1; diffusion helper matches dense `matmul` on toy graphs.

### Phase 3 — Graph-query tokens and the LLM layer partition
Read `docs/components/03-query-tokens.md`.
Learnable `B_base`, sequence assembly `[T_q ; B_base ; boundary ; answer]`, per-example
`b_positions`, and the manual layer loop that splits Llama into `F_0 … F_T`.
*Exit:* with all `Γ_t` replaced by identity, the manual loop reproduces
`model(...).logits` to within bf16 tolerance.

### Phase 4 — Topology-Diffused Graph Reader (`Read`)
Read `docs/components/04-reader.md`.
*Exit:* every row of `S̃` sums to 1; `K=0` reduces to plain cross-attention; per-head diffusion
matches a dense reference implementation.

### Phase 5 — Gated Residual Evidence Fusion (`Fuse`)
Read `docs/components/05-fuse.md`.
*Exit:* gate ∈ (0,1); with `W_O` zero-initialized the module is exactly the identity at step 0.

### Phase 6 — Full model assembly
Read `docs/components/06-model.md`.
Wire `F_0 → Γ_0 → F_1 → … → Γ_{T-1} → F_T → LMHead`, where `Γ_t = Replace ∘ Fuse ∘ Read`.
*Exit:* single-example forward returns finite loss; gradients reach the graph encoder, `B_base`,
reader, and fuse, and are `None` for every LLM parameter.

### Phase 7 — Training
Read `docs/components/07-training.md`.
Answer-only next-token NLL, AdamW (lr 1e-5, wd 0.05), warm-up + cosine, ≤10 epochs, batch size 4,
early stopping patience 2, best-validation-loss checkpoint.
*Exit:* overfits a 16-example subset to near-zero loss; a full ExplaGraphs run completes.

### Phase 8 — Inference with KV caching
Read `docs/components/08-inference.md`.
Run all `Γ_t` once during prefill; decode greedily with the standard KV cache, ≤32 new tokens.
*Exit:* cached decoding is token-identical to the naive full-recompute path on ≥20 examples.

### Phase 9 — Evaluation
Read `docs/components/09-evaluation.md`.
Accuracy for ExplaGraphs/SceneGraphs, Hit@1 for WebQSP, using G-Retriever's evaluation logic.
Train and evaluate one model per dataset.
*Exit:* the results table in `docs/experimental-protocol.md` filled in for all three datasets,
with per-dataset diagnostics (gate means, hop distributions, reading entropy) recorded alongside.

---

## Working style

- Prefer small, tested modules over one large file. Every module gets type hints and a docstring
  quoting the equation it implements (e.g. `"§2.3: S^{t,(k)} = S^{t,(0)} P^k"`).
- Assert shapes at module boundaries; shape bugs here are silent and expensive.
- Do not add features the spec does not ask for (no retrieval, no serialized subgraph in the
  prompt, no auxiliary losses, no task-specific heads).
- When you finish a phase, state what you built, which tests pass, and anything you had to decide.
