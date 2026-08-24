# EVAL.md — State-of-the-art audit

Where does ReGraph actually reach state of the art, and under what admission criteria?

This document exists because the answer depends entirely on what you are willing to count. It runs
the audit twice: once under strict criteria, once under two relaxations that were argued for and
accepted. Every number is traceable to a run directory under `runs/`; baselines are quoted from the
original papers with the source table named. Full experimental context is in
[README.md](README.md#experimental-results); this file only adjudicates the SOTA question.

Backbone throughout is Llama-3.1-8B-Instruct unless the row says 3B (Llama-3.2-3B-Instruct). The
LLM is frozen in every ReGraph row.

---

## 1. Admission criteria

**Pass 1 — strict.** A win counts only if it is (i) on the same evaluation set, (ii) under the same
training protocol, (iii) with a backbone no larger than the competitor's, and (iv) statistically
significant.

**Pass 2 — with two accepted relaxations.**

- **Rule A (backbone-parity credit).** If ReGraph *on the 3B backbone* matches what a baseline
  achieves on a larger backbone, that counts as SOTA on that dataset. Rationale: matching a result
  at half the parameters is at least as strong a claim as exceeding it at parity.
- **Rule B (protocol difference as a method property).** A protocol mismatch is not disqualifying
  when it follows from the *competitor's own method design* rather than from an unfair evaluation
  setup. GraphTranslator is a zero-shot transfer method by construction — it does not train on
  downstream task labels — so "ReGraph is supervised" describes a capability difference, not a
  rigged comparison.

Both rules are applied **symmetrically** below, including where they cost us (§5).

---

## 2. Pass 1 — strict criteria

| Dataset | ReGraph | Best published | Δ | Verdict |
| --- | --- | --- | --- | --- |
| ExplaGraphs | 92.42 | GRAFF 92.5 (3B) | −0.08 | tie (0.07 SE), but at 8B vs their 3B |
| SceneGraphs | 51.83 | GRAFF 90.2 (3B) | −38.37 | loss |
| WebQSP (Hit@1) | 62.22 | G-Retriever+LoRA 73.79 | −11.57 | loss |
| Cora | 87.64 ±0.92 | LLaGA-HO-7B 89.22 | −1.58 | loss (1.7 SE) |
| PubMed | 89.98 | SAGN 95.17 ᵍ | −5.19 | loss |
| ogbn-arxiv | 71.75 | GraphTranslator 28.48 | +43.27 | protocol mismatch — excluded |
| NLGraph connectivity | 52.58 | CoT+SC 86.82 | −34.24 | at chance |
| NLGraph cycle | 55.92 | few-shot 70.33 | −14.41 | at chance |
| ogbn-products | 74.21 | — | — | no comparable baseline |
| NeighborhoodQA | 83.87 | — | — | benchmark constructed here |
| StructuralAnomaly | 99.05 | — | — | benchmark constructed here (chance 20.00, control 19.80) |

ᵍ Chen et al., ICML 2024, Table 1, Single Focus block. This figure and LLaGA's PubMed 95.03 were
quoted without a citation until 2026-08-24; both were then verified against the paper and found
correct. The full PubMed baseline table is README Table 5.

**Result: no dataset.** The closest is ExplaGraphs, a statistical tie obtained with a larger
backbone than the method it ties.

---

## 3. Pass 2 — under Rules A and B

**Result: two datasets clean, one arguable, two uncontested by construction.**

The backbone column is load-bearing — several claims hold at one size and not the other, in
opposite directions. Read it together with §3.6 before quoting any of these.

| Dataset | ReGraph 3B | ReGraph 8B | Best published | Rule | Verdict |
| --- | --- | --- | --- | --- | --- |
| **ogbn-arxiv** | **72.28** | 71.75 | GraphTranslator 28.48 (ChatGLM2-**6B**) | A + B | **SOTA at either size**, +43.80 |
| **Cora** | **88.99 ±1.26** (3 seeds) | 87.64 ±0.92 | LLaGA-HO-7B 89.22 (Vicuna-**7B**) | A | **SOTA at 3B** (tie at half the backbone); 8B is 1.7 SE behind |
| ExplaGraphs | 90.43 | 92.42 | GRAFF 92.5 (**3B**) | — | behind at parity. Excluding GRAFF (§3.5): **8B wins (+3.52, 3.13 SE)**, 3B only +1.53 (1.22 SE, n.s.) |
| SceneGraphs | 53.24 | 51.83 · **89.41** *w/ token channel* | GRAFF 90.2 (3B); G-Ret. w/ LoRA 86.83 | — | as specified, far behind. **+ token channel at 8B: 1st of 8 non-GRAFF (+2.58, 11.9 SE)**, 0.79 behind GRAFF — §4.1 |
| NLGraph connectivity | 54.80 | 52.58 · 88.79 *w/ token channel* | CoT+SC 86.82 (davinci-003) | B | **not SOTA at 3B** (at chance). 8B+token arguable only — §3.3 |
| NeighborhoodQA | 83.83 | 83.87 | none | — | no competitor |
| ogbn-products | 74.27 | 74.21 | none | — | no competitor |
| StructuralAnomaly | **99.15** | 99.05 | none (analytic chance 20.00) | — | no competitor; controls pass at both sizes (20.05 / 19.80) — see §3.4 |

### 3.1 ogbn-arxiv — the strongest claim

72.28 (3B) / 71.75 (8B) Top-1 against GraphTranslator's 28.48, on **GraphTranslator's exact
4,000-node test subset** with the same 2-hop ego-subgraph construction. Likelihood-ranking Top-3 /
Top-5 at 8B are 92.58 / 96.40 against their 37.62 / 39.87.

What makes this hold up is that the **evaluation set is identical** — the only difference is that
ReGraph trains on 20,000 ogbn-arxiv nodes while GraphTranslator does not train on downstream
labels at all. Under Rule B that is a property of their method (a zero-shot translator), not an
evaluation artifact. Rule A applies independently: 3B beats their 6B backbone.

*Residual weakness:* the only competitor on this subset is zero-shot, so the table does not
separate "the graph→LLM interface works" from "sbert node features already contain the answer."
See §6.

### 3.2 Cora — the textbook Rule A case

3B, three seeds: 88.56 / 90.41 / 88.01 → **88.99 ± 1.26** (SEM 0.73).

| Comparison | Their backbone | Δ |
| --- | --- | --- |
| LLaGA-HO-7B 89.22 | Vicuna-7B | −0.23 (0.32 SE — tie) |
| SAGN 89.19 | GNN | −0.20 |
| GAT 88.97 | GNN | +0.02 |
| GCN 88.93 | GNN | +0.06 |
| GraphSAGE 88.89 | GNN | +0.10 |
| LLaGA-ND-7B 88.86 | Vicuna-7B | +0.13 |
| NodeFormer 88.23 | GNN | +0.76 |
| GPT-3.5 (general) 71.75 | GPT-3.5 | +17.24 |

ReGraph sits at the top of the published band on a **3B frozen** backbone against LLaGA's
Vicuna-7B, and it *generates the class name as free text* (100% legality) rather than selecting
from a label set. The 8B run is lower (87.64 ± 0.92), which is itself consistent with the
channel-limited picture in README §"Backbone sensitivity".

Baselines from Chen et al., ICML 2024, "Single Focus" setting — the one that matches ReGraph's
per-dataset training.

### 3.3 NLGraph connectivity — claimable but not recommended

88.79 vs CoT+SC 86.82 under Rule B — **at 8B with the token channel; there is no 3B equivalent**
(at 3B, ReGraph scores 54.80, at chance). Three reasons not to lean on even the 8B figure:

1. **Not significant** — +1.97 at 1.64 SE binomial (n=371), i.e. 1.20 SE.
2. **Not ReGraph as specified** — it requires the token channel, which serializes the edge list
   into the prompt and violates `ReGraph.md` §3.2.
3. **Rule B does not transfer here.** GraphTranslator cannot do supervised task training; but
   text-davinci-003 *can* be fine-tuned. NLGraph's authors chose prompting as an experimental
   design, which is not a method constraint, so Rule B's justification does not apply.

ReGraph as specified is at chance on both NLGraph tasks (52.58 / 55.92). Note also that the best
cycle configuration has the graph reader **switched off** (89.81 vs 76.75 with it on).

### 3.4 Uncontested by construction

NeighborhoodQA (83.87 set-F1) and StructuralAnomaly (99.05) are benchmarks built in this repo;
ogbn-products (74.21) is on a TAPE-derived subset with no matching published baseline. All three
are SOTA only in the sense that nobody else has run them. None should be presented as a SOTA
claim.

**StructuralAnomaly is nonetheless the strongest positive evidence in this project**, and it is
evidence of a different kind from a leaderboard number. It is the only benchmark here that (i)
tests `ReGraph.md` §2.1's **anchor-free** path, which the specification leads with and which
nothing else evaluates; (ii) has an **analytic** chance floor (20.00) rather than an empirical
one; and (iii) ships a **falsifiable control** — the density contrast removed and the label
randomised — that must score chance and does, **independently on two backbones**: 19.80 at 8B
(0.2 SE off) and 20.05 at 3B (0.06 SE off). Both channels are required by construction: topology
locates the dense region, text names it, neither alone suffices.

It also passes the backbone test that separates graph reading from language-model capacity:
99.05 (8B) versus 99.15 (3B), a +0.10 move that is well inside noise, matching NeighborhoodQA's
−0.05 and unlike WebQSP's −11.06. The result therefore belongs to the **3B-only** claim set of
§3.6 without any loss, though it remains a capability demonstration rather than a SOTA claim. Its
limits are equally clear: the cue is degree, a first-order local quantity, so it shows
anchor-free localization by a *local* structural cue and not path-level reasoning; and at the
released difficulty it is saturated (99.05), so the quantity worth reporting is the
accuracy-versus-density curve, not the point. See README Table 8.

### 3.5 Variant: excluding GRAFF

GRAFF (Findings of EACL 2026) is the only baseline standing between ReGraph and SOTA on
ExplaGraphs. Excluding it changes the count by **exactly one dataset**.

| Dataset | Best non-GRAFF baseline | ReGraph 8B | ReGraph 3B | Verdict |
| --- | --- | --- | --- | --- |
| **ExplaGraphs** | 88.9 — LoRA (3B, *LLM tuned*) and GRAG (3B, *text+vector*) | **92.42** (+3.52, 3.13 SE) | **90.43** (+1.53, 1.22 SE) | **SOTA** |
| SceneGraphs | 86.83 — G-Retriever w/ LoRA (7B, tuned) | 51.83 (−35.00) | 53.24 (−33.59) | loss |
| WebQSP | 73.79 — G-Retriever w/ LoRA (7B, tuned) | 62.22 (−11.57) | 51.17 (−22.62) | loss |

**ExplaGraphs holds up on its own terms.** Both 88.9 baselines carry an advantage ReGraph does not:
LoRA *unfreezes the LLM*, GRAG uses a text channel alongside vectors. ReGraph is frozen and
vector-only throughout. Against the closest frozen competitor (G-Retriever, 85.16) the margin is
+7.26 at 8B; against G-Retriever w/ LoRA (87.05, 7B, tuned) it is +3.38 with a **3B** backbone
under Rule A. The 8B win is significant at 3.13 SE; the matched-3B win over LoRA is +1.53 at
1.22 SE — ahead, but not significant on its own.

**WebQSP is unaffected**, because GRAFF was never what blocked it: the binding constraint is
G-Retriever w/ LoRA, and ReGraph as specified is 11.6 points behind it.

**SceneGraphs changed after this section was first written.** ReGraph as specified is still 35.0
behind, but the token-channel configuration, trained to convergence, reaches **89.41** on the full
test set — clearing G-Retriever w/ LoRA (86.83) by **+2.58 (11.9 SE)** and leaving only GRAFF
(90.2) ahead by 0.79 (3.6 SE). Excluding GRAFF therefore now flips SceneGraphs as well, subject to
the §3.2 violation carried by the token channel. See §4.1.

#### Two conditions on using this variant

1. **Excluding GRAFF requires a stated reason, and only one is defensible:** it is *concurrent
   work* (Findings of EACL 2026) whose code is unreleased (its repository still reads "The code
   will be released shortly"), so its numbers cannot be independently verified. Academic convention
   permits acknowledging concurrent work without being required to exceed it. This is legitimate
   **only if GRAFF's numbers remain printed in the comparison table** — dropping the row is
   indefensible.
2. **This SOTA is not evidence that graph reading works.** ExplaGraphs' evidence vector is ~98%
   example-invariant (README §"Why the results split the way they do"): the graph contributes
   almost nothing and the score is carried by the frozen backbone. The claim is sound as an
   accuracy result and unsound as a graph-reasoning result. ReGraph's ExplaGraphs number is best
   presented as *"matches or beats prior work while reading nothing from the graph"* — which is an
   efficiency and interface claim.

### 3.6 What may be claimed, by backbone policy

A "SOTA on *N* datasets" sentence is only well-formed once you fix **which backbone the claim
quantifies over**. The two honest options give different lists.

**Policy 1 — 3B only** (strongest, cleanest sentence: *one small frozen backbone, everywhere*):

> **ogbn-arxiv, Cora.** Two datasets.

**Policy 2 — best result per dataset, backbone mixed** (must then print the backbone per row):

> **ogbn-arxiv (3B), Cora (3B), ExplaGraphs (8B, GRAFF excluded).** Three datasets.

Three list items that do **not** survive either policy, and why:

- **NLGraph connectivity — remove it.** The 88.79 figure is **8B with the token channel**; no 3B
  dual-channel run exists (`runs/nlgraph_connectivity_dual/dual-A/resolved_config.yaml` resolves to
  `meta-llama/Llama-3.1-8B-Instruct`). At 3B, ReGraph scores 60.71 / 56.63 / 47.06 → **54.80**,
  which is at chance and 32 points behind CoT+SC. Even the 8B number is disqualified three ways in
  §3.3.
- **ExplaGraphs is weakest exactly at 3B**, the opposite of Cora. GRAFF's main table is already 3B,
  so there is no backbone advantage to claim; the significant margin (+3.52, 3.13 SE) exists only
  at 8B. Quoting "+3.52" alongside a 3B claim mixes the two.
- **NeighborhoodQA and ogbn-products should not appear in a SOTA list at all.** Claiming SOTA on a
  benchmark built in the same paper reads as padding and weakens the two real claims by
  association. Their correct use is as *analysis*: NeighborhoodQA's value is the 0-hop control
  (83.87 vs 72.65 — neighbours are worth +11.2 set-F1) and its backbone insensitivity
  (83.87 → 83.83), which together show it measures the graph interface rather than language-model
  capacity.

---

## 4. Full comparison against GRAFF's baselines, with GRAFF excluded

GRAFF's Table 1 is the densest published baseline set on the three GraphQA datasets, and every row
in it uses **Llama-3.2-3B** — the same backbone as ReGraph's 3B runs. That makes it the only
matched-backbone comparison available, so it is worth reading in full with the GRAFF row removed
(justification for removing it: §3.5, concurrent unreleased work).

Best non-GRAFF value per column in **bold**.

| Method (Llama-3.2-3B) | LLM | Graph input | ExplaGraphs | SceneGraphs | WebQSP |
| --- | --- | --- | --- | --- | --- |
| Zero-shot (base) | frozen | none | 13.5 | 33.1 | 32.7 |
| Zero-shot (chat) | frozen | none | 52.6 | 50.7 | 53.4 |
| KAPING | frozen | text | 62.2 | 43.7 | 52.6 |
| Prompt tuning | frozen | none | 60.2 | 58.3 | 57.9 |
| KG-Adapter | frozen | vector | — | — | 68.7 |
| GRAG | frozen | text+vector | **88.9** | — | 68.9 |
| G-Retriever | frozen | text+vector | 83.7 | 82.3 | 67.4 |
| LoRA | **tuned** | text | **88.9** | **85.3** | **71.1** |
| *(GRAFF — excluded)* | *frozen* | *text+vector* | *92.5* | *90.2* | *72.2* |
| **ReGraph (ours), 3B** | frozen | **vector only** | **90.43** | 53.24 | 51.17 |
| *ReGraph (ours), 8B* | frozen | *vector only* | *92.42* | *51.83* | *62.22* |
| **ReGraph + token channel, 8B**$^{\S}$ | frozen | text+vector | 86.82 | **89.41** | n/a |

$^{\S}$ Violates `ReGraph.md` §3.2 (graph serialized into the prompt) — the augmented
configuration, not ReGraph as specified. WebQSP admits no token channel: serializing it costs
62,703 tokens on average. ExplaGraphs is *lower* with the token channel than without (86.82 vs
92.42) because that task is answerable from the question text and the serialized graph is a
distraction.

Placement of the matched-backbone 3B row against the eight remaining baselines:

| Dataset | ReGraph 3B | Best non-GRAFF | Δ | Rank | Rank at 8B |
| --- | --- | --- | --- | --- | --- |
| **ExplaGraphs** | 90.43 | 88.9 (LoRA / GRAG) | **+1.53** (1.22 SE) | **1st of 8** | 1st of 8 (+3.52, 3.13 SE) |
| SceneGraphs *(as specified)* | 53.24 | 85.3 (LoRA) | −32.06 | 4th of 7 | 4th of 7 |
| **SceneGraphs *(+ token channel)*** | — | 86.83 (G-Ret. w/ LoRA) | **+2.58** (11.9 SE) | — | **1st of 8** |
| WebQSP | 51.17 | 71.1 (LoRA) | −19.93 | **8th of 9** | 5th of 9 |

### 4.1 What the table shows

**ExplaGraphs — first place, and the manner of it matters.** ReGraph is the only frozen,
vector-only method in the table, and at matched 3B it still leads a row that *unfreezes the LLM*
(LoRA) and a row that gets both text and vectors (GRAG), both at 88.9. Against the closest
architectural relative, G-Retriever (83.7, frozen, text+vector), the margin is +6.73. The caveat
from §3.5 stands: +1.53 at 1.22 SE is a lead, not a significant one — significance appears only at
8B (+3.52, 3.13 SE).

**SceneGraphs — 4th of 7, and beaten by prompt tuning.** At 53.24 ReGraph loses to LoRA (85.3),
G-Retriever (82.3) and Prompt tuning (58.3), clearing only the two zero-shot rows and KAPING. Being
beaten by **prompt tuning, which sees no graph at all**, is the sharpest available statement of the
attribute-binding failure analysed in README §"Why the results split the way they do".

*With the token channel, the same dataset inverts.* Trained to convergence
(`runs/scene_graphs/dual-full`, 6 epochs, full 20,025-example test set) ReGraph reaches **89.41**,
which is **1st among the eight non-GRAFF baselines** — ahead of G-Retriever w/ LoRA by +2.58
(11.9 SE) and of LoRA by +4.11, while keeping the LLM **frozen** against their tuned one. GRAFF
stays 0.79 ahead (3.6 SE). This configuration serializes the graph and so violates
`ReGraph.md` §3.2; unlike the NLGraph case in §3.3, however, the comparison itself is fair — the
baselines it beats also put the graph in the prompt and also train — so the disqualification is
about *which method this is*, not about whether the comparison is sound. The remaining asymmetries
are the truncated full graph (1,536 tokens) versus their 8.21-node PCST subgraph, and an 8B
backbone against 7B/3B.

**WebQSP — 8th of 9 at 3B, the single worst placement in this document.** ReGraph 3B (51.17) is
beaten by **zero-shot chat (53.4)**, i.e. by the backbone with no graph and no training. Only
zero-shot base (32.7) is lower. At 8B it recovers to 5th of 9 (62.22) — which is itself the
evidence that WebQSP's score is carried by the backbone's parametric knowledge of famous entities
rather than by graph reading (README §"Backbone sensitivity": −11.06 from halving the backbone,
against ≤2 points on seven other datasets).

### 4.2 Two caveats that survive removing GRAFF

1. **Retrieved vs full graphs.** All GRAFF-paper rows evaluate on G-Retriever's PCST-*retrieved*
   subgraphs (their Table 2: WebQSP 8.39 average nodes, SceneGraph 8.21). ReGraph reads the **full**
   graph. On ExplaGraphs this is immaterial — those graphs average 5.17 nodes, so no retrieval step
   is involved and the comparison is clean. On SceneGraphs and WebQSP the difference is large, and
   it runs **in ReGraph's favour** in information terms (gold answer present in 95% of full WebQSP
   graphs versus 13–15% under our verified port of their retrieval), so the two losses are if
   anything understated.
2. **SyntheticGraph is not covered.** GRAFF's fourth dataset (G-Retriever 56.4, LoRA 58.3,
   GRAFF 79.4) has never been run here, so no claim of any kind is made on it. Its construction —
   graph properties computed from topology — is the family ReGraph is measured to fail at
   (NLGraph, §3.3), so it should be assumed unfavourable until run.

### 4.3 Net effect

Removing GRAFF converts **two** results from a loss into a win: **ExplaGraphs** for ReGraph as
specified, and **SceneGraphs** for the token-channel configuration (89.41 vs G-Retriever w/ LoRA's
86.83). It does not rescue **WebQSP**, where the binding constraint is LoRA — a baseline that
predates GRAFF and appears in every version of this table — and where no token-channel run is
possible at all (§4.2 and README footnote ‡: serializing WebQSP costs 62,703 tokens on average).

Note the asymmetry between the two wins. ExplaGraphs is won by ReGraph *as specified*, but on a
dataset where the graph contributes almost nothing. SceneGraphs is won on a dataset where the graph
is decisive, but only by a configuration that **violates §3.2**. Neither is a clean demonstration
that iterative graph reading beats prior work; the closest thing to that remains the matched A/B
in §7, claim 2.

---

## 5. The rules cut both ways

Applying A and B symmetrically has two costs that must be stated alongside the gains.

**Rule A demotes ExplaGraphs.** GRAFF's main table already uses Llama-3.2-3B. At matched 3B,
ReGraph scores 90.43 against GRAFF's 92.5 — a **2.07 loss at parity**. The 92.42 tie exists only at
8B, i.e. with a *larger* backbone than the opponent. If backbone size earns credit on Cora, it must
count as debit here; ExplaGraphs therefore moves from "tie" to "behind at parity." (It remains
ahead of every non-GRAFF baseline, and does so with no text channel at all — that is a real result,
just not a SOTA one.)

**Protocol difference favours us on the two biggest losses.** G-Retriever and GRAFF evaluate
SceneGraphs and WebQSP on **PCST-retrieved subgraphs** (8.21 / 8.39 average nodes); ReGraph reads
the **full** graph (WebQSP averages 1,371 nodes, with the gold answer present 95% of the time
versus 13–15% for our verified port of their retrieval). In information terms the setup is tilted
toward ReGraph, and it still loses by 38.4 and 11.6. These are genuine losses, arguably understated.

---

## 6. What would harden the claims

- **ogbn-arxiv:** add a supervised baseline on the identical 4,000-node subset and the same sbert
  features — a GNN or MLP probe trained on the same 20,000 nodes. This separates "the graph→LLM
  interface carries signal" from "the node features already contain the answer," which is the first
  question a reviewer will ask given that the only competitor is zero-shot. Cheap (well under an
  hour). **Not run** — per `CLAUDE.md` ground rule 7, flagged rather than built.
- **Cora:** three seeds at 3B give SEM 0.73; two more would tighten the tie claim against LLaGA's
  single reported number.
- **ExplaGraphs (GRAFF-excluded variant):** the matched-3B margin over LoRA is +1.53 at 1.22 SE.
  Additional 3B seeds would decide whether it is a win or a tie; the 8B margin (+3.52, 3.13 SE) is
  already significant.

---

## 7. Recommended framing

The defensible headline is **not** a SOTA sweep. Two claims are supported by the data as it stands:

1. **ReGraph matches or exceeds the published state of the art on ogbn-arxiv and Cora using a 3B
   frozen backbone** — half the parameters of the LLM baselines, with no text channel. This is the
   3B-only claim (§3.6, Policy 1) and it is the one to lead with, because a single small frozen
   backbone across both datasets is a cleaner sentence than a per-dataset best-of.
   If GRAFF is treated as concurrent unreleased work (§3.5), **ExplaGraphs** can be added as a
   third — but only at **8B**, where the margin is significant (+3.52, 3.13 SE) rather than at 3B
   (+1.53, n.s.). That mixes backbones, so print the backbone in every row, and frame ExplaGraphs
   as an interface/efficiency result: the number reflects the frozen backbone, not graph reading.
2. **The anchor-free path works, under a falsifiable control.** On StructuralAnomaly (§3.4)
   ReGraph reaches 99.15 at 3B and 99.05 at 8B against an analytic chance of 20.00, while the
   matched control — same generator, density contrast removed — scores 20.05 and 19.80. This is the specification's own headline
   capability (§2.1) and the only result here validated by a control that could have failed.
   Frame it as a capability demonstration, not a SOTA claim, and state the degree-cue limitation
   with it.
3. **Iterative graph reading is not subsumed by serialization.** README Table 3 row 15 minus row 16
   is +5.77 pp on ExplaGraphs (2.62 SE) and +7.00 pp on SceneGraphs (4.6 SE) — two independent
   datasets where the reading rounds add on top of putting the graph in the prompt.

The honest complement, documented at length in README §"Why the results split the way they do":
ReGraph's vector channel relays coarse semantics already present in node features, but cannot bind
attributes to objects (SceneGraphs), emit exact surface forms (WebQSP), or compute properties from
topology (NLGraph). The datasets where it reaches SOTA are exactly the datasets where the answer is
a coarse topic of one node.

---

## 8. Provenance

| Number | Source |
| --- | --- |
| ReGraph 8B | `runs/{dataset}/{fix-seed0,seed0,control}/metrics_test.json` |
| ReGraph 3B | `runs/{dataset}/llama3b*/metrics_test.json` |
| Token-channel rows | `runs/{scene_graphs,expla_graphs,nlgraph_*_dual}/dual-{A,B}/` |
| ExplaGraphs / SceneGraphs / WebQSP baselines | He et al., NeurIPS 2024 (arXiv:2402.07630) Table 3 |
| GRAFF | Chaudhary et al., Findings of EACL 2026, Table 1 |
| GraphTranslator | Zhang et al., WWW 2024, Table 1 |
| LLaGA + GNN band (Cora, PubMed) ᵍ | Chen et al., ICML 2024, Table 1, Single Focus block |
| NLGraph | Wang et al., NeurIPS 2023, Table 2 (standard set) |

Standard errors are binomial (`sqrt(p(1−p)/n)`) unless a multi-seed SEM is quoted. NLGraph averages
follow the authors' **unweighted mean of the three difficulty subsets**; size-weighted overall
accuracies are 49.33 (connectivity) and 52.88 (cycle).

**Completed 2026-08-24:** the full-scale SceneGraphs token-channel run
(`runs/scene_graphs/dual-full`) finished at **89.41** on all 20,025 test examples (6 epochs, best
validation loss 0.2239 at epoch 3), superseding the 81.93 obtained at 3,000 steps on a
1,500-example subset. It **did** change a verdict: SceneGraphs now leads every non-GRAFF baseline
(§3.5, §4.1, §4.3). Its α is also *not* collapsed — see README Table 7, where the earlier claim
that §2.3's diffusion can be removed on every dataset is withdrawn and narrowed to the
vector-only setting.

The 1-hop control for the 2-hop NeighborhoodQA variant has since completed
(`runs/arxiv_nbrqa_hop2_1hop/seed0`): 48.66 set-F1 against the full 2-hop arm's 56.68, a paired
**+7.95 (SEM 0.33, 24 SE)** over examples where gold labels are structurally invisible to the
control. It affects no SOTA claim — there is no external baseline — but it sharpens the §3.6 note
on NeighborhoodQA's analytic role, and it is where the α-collapse finding is cleanest: the hop
distribution is `[1.000, 0.000, 0.000]` in all three rounds *on a task built to require hop 2*,
so the two-hop signal comes from the graph encoder, not from §2.3's diffusion. See README Table 7.
