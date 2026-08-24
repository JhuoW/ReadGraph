# 09 — Evaluation

Spec reference: `ReGraph.md` §3.1, §3.2. Protocol: `docs/experimental-protocol.md`.
Module: `src/regraph/eval.py`

`ReGraph.md` §3 is explicit: "We follow the official data splits, answer targets, meta-llama/Llama-3.1-8B-Instruct
backbone, decoding procedure, and evaluation scripts." Comparability depends entirely on this
stage, so **port G-Retriever's evaluation logic rather than writing your own**, and keep the ported
code in one file with a comment naming the source function.

## 9.1 Metrics

| Dataset     | Metric   | Definition                                                                   |
| ----------- | -------- | ---------------------------------------------------------------------------- |
| ExplaGraphs | Accuracy | prediction matches the gold `support`/`counter` label                    |
| SceneGraphs | Accuracy | prediction matches the gold short answer                                     |
| WebQSP      | Hit@1    | the prediction contains at least one gold entity (targets joined with `\|`) |

Normalization (lowercase, strip whitespace/punctuation/articles) must match G-Retriever's. Small
differences here move numbers by a point or more, which is enough to make a comparison meaningless.

## 9.2 Procedure

1. Load the checkpoint with the lowest validation loss.
2. Greedy decode, ≤32 new tokens, stop at `EOS` (`08-inference.md`).
3. Write one JSONL row per test example: `id`, `question`, `prediction`, `gold`, `correct`, plus
   per-round mean gate and mean hop distribution.
4. Report the metric with the number of test examples, and confirm that number equals the split
   size in `docs/experimental-protocol.md`.

## 9.3 Reference points

Report ReGraph next to numbers already published for these exact splits (G-Retriever and the
baselines reported in that paper) so a reader can place it. **Copy those numbers from the papers
and cite them — do not reimplement or re-run any baseline.** This repo produces one number per
dataset: ReGraph's.

If ReGraph underperforms on a dataset, report it plainly along with the diagnostic logs (gate
means, hop distributions, reading entropy). Do not tune on test.

## 9.4 Qualitative analysis (optional but cheap)

The reading distribution $\tilde S^t_{i,:}$ is a distribution over nodes, so it is directly
inspectable. For a handful of examples per dataset, dump the top-5 nodes per round with their mass
and the hop weights. This shows whether iterative reading actually moves attention across rounds —
the central claim of the method — and costs nothing beyond a hook.

## Acceptance tests

1. The ported metric reproduces a hand-computed score on a 10-row synthetic prediction file.
2. Test set size equals the protocol table for each dataset.
3. Predictions are non-empty for ≥99% of examples (empty outputs usually mean a broken boundary
   string).
4. Evaluation is deterministic: two runs on the same checkpoint give identical predictions.
