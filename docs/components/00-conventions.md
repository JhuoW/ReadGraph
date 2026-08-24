# 00 — Conventions

Read this before writing any code. Spec reference: `ReGraph.md` §1, §2.2.

## Symbol → code name

| Spec                    | Code                           | Default  | Shape / notes                                  |
| ----------------------- | ------------------------------ | -------- | ---------------------------------------------- |
| $n = \|V\|$           | `num_nodes`                  | —       | per graph;`n_max` after padding              |
| $d_0$                 | `d_attr`                     | 1024     | `all-roberta-large-v1` output                |
| $d_g$                 | `d_graph`                    | 1024     | node-memory width                              |
| $d$                   | `d_llm`                      | 4096     | meta-llama/Llama-3.1-8B-Instruct hidden size  |
| $d_r$                 | `d_reader`                   | 1024     | reader width (all heads)                       |
| $N_B$                 | `num_query_tokens`           | 8        | graph-query tokens                             |
| $T$                   | `num_rounds`                 | 3        | graph-reading rounds                           |
| $K$                   | `max_hops`                   | 2        | diffusion depth                                |
| $L$                   | —                             | 32       | meta-llama/Llama-3.1-8B-Instruct layers       |
| $H^{\text{base}}$     | `h_base`                     | —       | `[B, n_max, d_graph]`                        |
| $H$                   | `h`                          | —       | `h_base + role_emb`, `[B, n_max, d_graph]` |
| $P$                   | `edge_src, edge_dst, edge_w` | —       | sparse form, see §02                          |
| $B_{\text{base}}$     | `b_base`                     | —       | `[N_B, d_llm]` learnable                     |
| $B_{\mathrm{pre}}^t$  | `b_pre`                      | —       | `[B, N_B, d_llm]`                            |
| $B_{\mathrm{post}}^t$ | `b_post`                     | —       | `[B, N_B, d_llm]`                            |
| $R^t$                 | `r_evidence`                 | —       | `[B, N_B, d_llm]`                            |
| $S^{t,(k)}$           | `s_k`                        | —       | `[B, heads, N_B, n_max]`                     |
| $\tilde S^t$          | `s_tilde`                    | —       | same                                           |
| $\alpha^t$            | `alpha`                      | —       | `[B, N_B, heads, K+1]`                       |
| $g^t$                 | `gate`                       | —       | `[B, N_B, 1]`                                |
| $\mathcal I_B$        | `b_positions`                | —       | `[B, N_B]` int64, per-example                |
| $F_t$                 | `layer_groups[t]`            | 4 groups | `[0:8), [8:16), [16:24), [24:32)`            |
| $\Gamma_t$            | `graph_round(t)`             | —       | Replace ∘ Fuse ∘ Read                        |

## Spec inconsistencies — resolved

`ReGraph.md` contains a few notational collisions. Use these resolutions everywhere; do not
"fix" them differently on the fly.

1. $m$, $N_b$, $N_B$ all denote the number of graph-query tokens → **`num_query_tokens`**.
2. $Q_{\text{base}}$ (§2.2) is a typo for $B_{\text{base}}$; $\mathcal I_Q$ is a typo for
   $\mathcal I_B$.
3. §2.2 writes the chain as `F_0 → Read_1 → … → Read_T → F_T` (1-indexed rounds) while §2.5 writes
   `F_0 → Γ_0 → … → Γ_{T-1} → F_T` (0-indexed). **Use §2.5.** With `T = 3` there are 4 layer groups
   and 3 rounds, matching §3.3.
4. $L$ is used both for the number of LLM layers (§2.2) and as the instruction length in
   $T_q \in \mathbb R^{L\times d}$. Instruction length is **`n_q`** ($N_q$); `L` is layer count only.
5. §2.5 "Read=Fuse-Replace" is a typo for "Read-Fuse-Replace".
6. Llama's `lm_head` has no bias, so $b_{\text{vocab}} = 0$.

## Tensor conventions

- Batch-first everywhere: `[B, ...]`.
- Nodes are **padded per batch** to `n_max` with a boolean `node_mask [B, n_max]`. Every softmax
  over nodes must add `-inf` (use `torch.finfo(dtype).min`) at masked positions; every aggregation
  must zero them.
- Graph-query tokens are contiguous in the sequence, but their absolute offset differs per example
  because prompts differ in length. Always index them with `b_positions`, never with a constant.
- Sequences are **right-padded**; `attention_mask` marks real tokens.

## Precision and devices

- Frozen Llama: `torch_dtype=torch.bfloat16`, `requires_grad_(False)`, gradient checkpointing on.
- Trainable modules (graph encoder, `b_base`, reader, fuse): **fp32 parameters**, run under
  `torch.autocast(dtype=torch.bfloat16)`. Cast hidden states explicitly at the LLM↔reader boundary
  and assert dtype there.
- Softmax over nodes and the diffusion recursion run in **fp32** even under autocast; bf16
  row-stochasticity drifts badly at `n > 1000` (WebQSP). Cast back to bf16 after `s_tilde`.

## Configuration

- YAML under `configs/`; `default.yaml` holds everything and dataset files override it. Resolve
  with a simple deep-merge; dump the resolved config into the run dir.
- Any value read at runtime that is *not* in the config is a bug.

## Testing

- `pytest`, one file per component, mirroring `src/regraph/`.
- Prefer **reference-implementation tests**: write the slow, obviously-correct dense version in the
  test and assert the fast implementation matches it (`torch.allclose`, `atol=1e-5` in fp32).
- Every module gets a shape test with a non-square, non-uniform batch (different `n_q`, different
  `num_nodes`) — uniform batches hide indexing bugs.
