# 01 — Data pipeline

Spec reference: `ReGraph.md` §1, §3.1, §3.2. Protocol details: `docs/experimental-protocol.md`.

Module: `src/regraph/data/{preprocess.py, datasets.py, collate.py}`

## What this stage produces

For every example $(G_i, q_i, A_i)$:

| Field | Type | Meaning |
|---|---|---|
| `x` | `float32 [n_i, 1024]` | node attribute embeddings $c_v = f_{\text{attr}}(x_v)$ |
| `edge_index` | `int64 [2, e_i]` | directed edges as given by the dataset |
| `edge_attr` | `float32 [e_i, 1024]` | relation-text embeddings |
| `roles` | `int64 [n_i]` | 0=none, 1=mentioned, 2=source, 3=target |
| `question` | `str` | raw instruction $q_i$ |
| `answer` | `str` | gold target $A_i$ |
| `split` | `str` | train / val / test |

**The graph is never rendered into text.** There is no retrieval step, no subgraph selection, and
no textual description of nodes or edges in the prompt — this is the whole point of ReGraph
(`ReGraph.md` §3.2: "the graph is not serialized into the LLM context").

## Attribute encoding

- Encoder: `sentence-transformers/all-roberta-large-v1` (`d_attr = 1024`), frozen, **pre-computed
  once** and cached to disk (`.pt` per split, or a single memory-mapped tensor). It is not part of
  the training graph.
- Encode node text and relation/edge text with the same encoder. Mean-pool + normalize per the
  sentence-transformers default; do not add a projection here.
- Cache key must include the encoder name so a change invalidates the cache.

## Datasets

Follow G-Retriever (He et al., 2024) exactly — same raw files, same split index files. Do not
re-derive splits.

| Dataset | Source | Split | Target | Metric |
|---|---|---|---|---|
| ExplaGraphs | G-Retriever `expla_graphs` | 1,659 / 553 / 554 | `support` \| `counter` | Accuracy |
| SceneGraphs | G-Retriever `scene_graphs` (GQA) | 60/20/20 by **image id** | short NL answer | Accuracy |
| WebQSP | G-Retriever `webqsp` (RoG partition) | official | ≥1 entity | Hit@1 |

Notes:
- SceneGraphs splits are over image identifiers so that questions about the same scene graph never
  cross subsets. Split on image id, not on question id.
- WebQSP: drop the empty validation graph removed by G-Retriever's preprocessing. Multiple gold
  answers are joined with `|` for teacher forcing (`ReGraph.md` §3.1).
- After preprocessing, print `#examples`, `avg nodes`, `avg edges` per dataset and compare to the
  table in `docs/experimental-protocol.md`. A mismatch means the wrong split files — stop and report.

## Query-role markers (`ReGraph.md` §2.1)

$r_v(q) \in \{\text{none}, \text{mentioned}, \text{source}, \text{target}\}$, assigned **only when
the reference resolves unambiguously**. Implement in `src/regraph/data/roles.py`:

- Normalize (lowercase, strip punctuation/articles) both the question and node text; mark a node
  `mentioned` on an exact normalized full-string match against a question span. Ambiguous or
  multi-node matches → `none`.
- `source` / `target`: only where the dataset supplies them explicitly (WebQSP topic entity
  `q_entity` → `source`). Never guess from word order.
- Anchor-free questions leave every node as `none`; this is a first-class case, not a failure.
- Log the fraction of examples with ≥1 non-`none` role per dataset and record it in the run dir.

## Prompt assembly

Fixed per dataset, defined in the config, logged at startup:

```
T_q      = tokenizer(f"Question: {question}\n")        # includes BOS; length n_q
B        = num_query_tokens placeholder positions      # embeddings come from b_base
boundary = tokenizer("\nAnswer:", add_special_tokens=False)   # the y_0 block, n_0 tokens
answer   = tokenizer(f" {answer}", add_special_tokens=False) + [EOS]   # training only
```

Sequence: `[T_q ; B ; boundary ; answer]`. The spec's $y_0$ is a single answer-start token; a
short multi-token boundary is the practical equivalent and shifts the answer offset to
$j_s = N_q + N_B + n_0 + (s-1)$ (one-based). Set `n_0 = 1` to match the spec exactly if you prefer;
either way record the choice in `docs/OPEN-QUESTIONS.md`.

Input embeddings are built with `inputs_embeds`, **not** `input_ids`: embed the real tokens with
the frozen `embed_tokens`, then scatter `b_base` into the `b_positions` slots. Use a placeholder
token id (e.g. `<|reserved_special_token_0|>`) for those slots so ids and embeds stay aligned.

## Labels

`labels[-100]` everywhere except the answer tokens and the final `EOS`
(`ReGraph.md` §3.2: "Losses on the question, graph-query tokens, answer boundary, and padding are
masked").

## Collate

`collate_fn` returns a dict with:

- `inputs_embeds [B, S, d_llm]`, `attention_mask [B, S]`, `labels [B, S]`, `b_positions [B, N_B]`
- `x [B, n_max, 1024]`, `node_mask [B, n_max]`, `roles [B, n_max]`
- PyG-style flat `edge_index`, `edge_attr`, `node_batch` for the encoder
- **padded-layout transition edges** for the diffusion step (see `02-graph-encoder.md`):
  `edge_src_pad`, `edge_dst_pad` (int64, indices into a flattened `B * n_max` axis),
  `edge_w` (float32, $1/\tilde d_u$), including self-loops

Sort/bucket by sequence length within an epoch to limit padding, but keep the shuffle seed logged.

## Acceptance tests

1. Split sizes and avg node/edge counts match the protocol table for all three datasets.
2. Collate with a deliberately ragged batch (different `n_q`, different `num_nodes`): every
   `b_positions` row points at the placeholder token id; `labels` is `-100` at all non-answer
   positions; `node_mask` sums equal the true node counts.
3. Round trip: decoding `input_ids` with the B slots removed reproduces the prompt+answer string.
4. Roles: a hand-written question/graph pair yields exactly the expected role vector; an ambiguous
   mention yields all-`none`.
