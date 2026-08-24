# ReGraph: Iterative Graph Reading for Open-Ended Reasoning with Large Language Models

**GraphRAG**: can not handle non-retrieval query,

**Graph Representation Alignment:** structural faithfulness under representation compression, and once alignment maybe suboptimal for different query

Combine the advantages of GraphRAG and Graph Representation Alignment: Read the embedding from the graph with the query, and then map it back to the LLM token space.

# 1. Problem Formulation

Let $G = (V,E,X)$ denote an attributed graph, where:

- $V = \{v_1, \ldots, v_n\}$ is the node set;
- $E \subseteq V \times V$ is the edge set;
- $X=\left\{x_v \mid v \in V\right\}$ contains optional node attributes.

The attributes may be textual, numerical, categorical, visual, or combinations of them. The basic method only requires that every node can be mapped to a vector.

Given a natural-language instruction with $N_q$ tokens:

$$
q=\left(q_1, q_2, \ldots, q_{N_q}\right)
$$

the model generates an open-ended response:

$$
y=\left(y_1, y_2, \ldots, y_M\right)
$$

according to

$$
p(y \mid q, G)=\prod_{j=1}^M p\left(y_j \mid y_{<j}, q, G\right)
$$

No task-specific prediction head is assumed. The response may be a class label, a path, a textual explanation or a free-form answer, and so on. A training example therefore has the general form $(G, q, y)$

# 2. Method

The retrieval things are aligned to the LLM space with aligned token embedding

## 2.1 Pre-computation

Each node attribute $x_v$ is first converted into a content representation: $c_v=f_{\text {attr }}\left(x_v\right) \in \mathbb{R}^{d_0}$, which can be a pretrained text encoder. The graph–>LLM interface does not change when the attribute encoder changes. The graph encoder produces one representation for every node:

$$
H^{\text {base }}=E_\phi(G, C) \in \mathbb{R}^{n \times d_g} = \left[h_1^{\text {base }}, \ldots, h_n^{\text {base }}\right]^{\top}
$$

where $C=\left[c_{v_1} ; c_{v_2} ; \ldots ; c_{v_n}\right]$ is the original encoded features of each node. $E_\phi$ can be a GIN, GraphSAGE, GAT, or graph transformer.

**Query-role markers:** Some instructions explicitly refer to particular graph nodes: `“Find a path from node 7 to node 19.”`  When such references can be resolved unambiguously, we construct a role indicator:

$$
r_v(q) \in\{\text { none, mentioned, source, target }\}
$$

A learned role embedding is added to the node representation:

$$
h_v=h_v^{\text {base }}+e_{\text {role }}\left(r_v(q)\right) .
$$

Thus, we have the node embeddings $H=\left[h_1 ; \ldots ; h_n\right]$. For anchor-free questions such as `“Which region of the graph is becoming structurally unstable?”`  all nodes receive the `none` marker. The model then locates relevant graph regions through semantic and structural attention rather than beginning from a predefined question entity.

Let $A \in \mathbb{R}^{n \times n}$ denote the adjacency matrix. We add self-loops as $\tilde{A}=A+I$. The row-normalized graph transition matrix is $P=D^{-1} \tilde{A}$. Given a node-attention distribution $S^{(0)}$, the distributions are computed recursively: $S^{(k+1)}=S^{(k)} P$.

## 2.2 Instruction encoding and graph-query-token initialization

The instruction is tokenized and embedded as $T_q=\left[e\left(q_1\right), e\left(q_2\right), \ldots, e\left(q_{N_q}\right)\right] \in \mathbb{R}^{L \times d}$. We introduce $m$ learnable graph-query tokens: $B_{\text {base }}=\left[b_1, \ldots, b_{N_b}\right] \in \mathbb{R}^{N_b \times d}$ where $d$ is the LLM hidden dimension. The initial LLM sequence is $Z^0=\left[T_q ; B_{\text {base}}\right]$, where the graph-query tokens are placed **after** the instruction. Under the ordinary causal mask of a decoder-only LLM, every graph-query token can attend to all instruction tokens and preceding graph-query tokens.

Suppose the LLM contains $L$ Transformer layers. We divide it into $T+1$ consecutive groups: $F_0, F_1, \ldots, F_T$. The initial sequence $Z^0=\left[T_q ; Q_{\text {base}}\right]$ is passed through the first group of LLM layers:

$$
Z_{\mathrm{pre}}^0=F_0\left(Z^0\right)
$$

Let $\mathcal{I}_B$ denote the positions of the graph-query tokens. **The initial query states are** $B_{\mathrm{pre}}^0=Z_{\mathrm{pre}}^0\left[\mathcal{I}_B\right]$**, which is conditioned only on the instruction**. It has not yet received any information from the graph. For brevity, we can write $B^0 \equiv B_{\mathrm{pre}}^0$.

- $B_{\mathrm{pre}}^t$: graph-query tokens immediately before graph-reading round $t$.
- $B_{\mathrm{post}}^t$: graph-query tokens immediately after incorporating the graph information returned at round $t$

**The first graph-reading round** takes $B_{\mathrm{pre}}^0$, $H$, $P$ as input, where $H \in \mathbb{R}^{|V| \times d_g}$ are node embeddings and $P \in \mathbb{R}^{|V| \times|V|}$ is the normalized graph transition matrix. It return graph information:

$$
R^0=\operatorname{Read}\left(B_{\mathrm{pre}}^0, H, P\right)
$$

which means **Using the instruction-conditioned query tokens** $B_{\mathrm{pre}}^0$ **to ask the graph what information** $R^0$ **is relevant, and retrieving that information from node representations** $H$ **while respecting the graph topology** $P$**.**

The return information $R^0$ is fused into the graph-query tokens:

$$
B_{\mathrm{post}}^0=\operatorname{Fuse}\left(B_{\mathrm{pre}}^0, R^0\right)
$$

At this point, $B_{\mathrm{post}}^0$ is conditioned on both:

- the instruction;
- the graph information obtained in the first read.

We replace the graph-query-token positions in the hidden sequence:

$$
Z_{\text {post }}^0=\text { Replace }\left(Z_{\text {pre }}^0, \mathcal{I}_Q, B_{\text {post }}^0\right)
$$

The resulting sequence is passed through the next LLM layer group:

$$
Z_{\text {pre }}^1=F_1\left(Z_{\text {post }}^0\right)
$$

The next graph-query-token states are extracted $B_{\mathrm{pre}}^1=Z_{\mathrm{pre}}^1\left[\mathcal{I}_B\right]$.

$$
F_0 \rightarrow \operatorname{Read}_1 \rightarrow F_1 \rightarrow \operatorname{Read}_2 \rightarrow \cdots \rightarrow \operatorname{Read}_T \rightarrow F_T
$$

**General recurrence: At graph reading round** $t$, the computation is:

$$
\begin{gathered}
R^t=\operatorname{Read}\left(B_{\mathrm{pre}}^t, H, P\right), \\
B_{\mathrm{post}}^t=\text { Fuse }\left(B_{\mathrm{pre}}^t, R^t\right), \\
Z_{\mathrm{post}}^t=\operatorname{Replace}\left(Z_{\mathrm{pre}}^t, \mathcal{I}_B, B_{\mathrm{post}}^t\right), \\
Z_{\mathrm{pre}}^{t+1}=F_{t+1}\left(Z_{\mathrm{post}}^t\right), \\
B_{\mathrm{pre}}^{t+1}=Z_{\mathrm{pre}}^{t+1}\left[\mathcal{I}_B\right] .
\end{gathered}
$$

Intuitively:

- $B_{\mathrm{pre}}^0$: what graph information appears necessary after **reading only the instruction**;
- $B_{\mathrm{pre}}^1$: what graph information appears necessary after considering the first graph read;
- $B_{\mathrm{pre}}^3$: what graph information appears necessary after considering the first two reads;

…

$$
\text { Read graph → reason in language space → form a new graph query → read again. }
$$

After the final LLM group, the graph-query tokens contain the graph information collected throughout reasoning.

## 2.3 Topology-Diffused Graph Reader $Read()$

The **Topology-Diffused Graph Reader** instantiates the graph-reading operation $R^t=\operatorname{Read}\left(B_{\mathrm{pre}}^t, H, P\right)$. The first step is to compare each current graph-query token with every node representation. We project the graph-query tokens into a reader query space:

$$
U^t=\operatorname{LN}\left(B_{\text {pre }}^t\right) W_Q \in \mathbb{R}^{N_B \times d_r}
$$

The graph memory is projected into key and value spaces:

$$
\begin{aligned}& K_H=\mathrm{LN}(H) W_K \in \mathbb{R}^{N_V \times d_r}, \\& V_H=\mathrm{LN}(H) W_V \in \mathbb{R}^{N_V \times d_r},\end{aligned}
$$

where $W_Q \in \mathbb{R}^{d \times d_r}$ and $W_K, W_V \in \mathbb{R}^{d_g \times d_r}$. The semantic compatibility between graph-query token $i$ and node $v$ is:

$$
E_{i v}^t=\frac{\left\langle U_i^t, K_{H, v}\right\rangle}{\sqrt{d_r}}
$$

We normalize over all graph nodes as:

$$
S_{i v}^{t,(0)}=\frac{\exp \left(E_{i v}^t\right)}{\sum_{u=1}^{N_V} \exp \left(E_{i u}^t\right)} .
$$

Every row is a probability distribution $\sum_{v=1}^{N_V} S_{i v}^{t,(0)}=1$. The quantity $S_{i v}^{t,(0)}$ measures how strongly the current reasoning state represented by graph-query token $i$ considers node $v$ a useful semantic seed for reading the graph. Semantic compatibility alone does not capture graph structure. Two nodes may have similar representations while occupying completely different structural positions. ReGraph therefore diffuses the seed distribution through $P$. Define $S^{t,(k)}=S^{t,(k-1)} P$ with $k = 1, \ldots, K$ and $S_{i v}^{t,(0)}$ as the semantic seed distribution. Equivalently, $S^{t,(k)}=S^{t,(0)} P^k$. Here $K$ is the maximum topology diffusion depth. For graph-query token $i$, $S_{i,:}^{t,(k)}$ is the distribution obtained after propagating its semantic seed mass through $k$ graph transitions.

Different graph questions require different structural scales. A question about a node attribute may rely mainly on the initial seeds or immediate neighbors. A connectivity, community, or influence question may require evidence propagated through several transitions. Moreover, different graph-query tokens may require different scales in the same reasoning round. ReGraph therefore predicts a hop distribution for every graph-query token:

$$
\alpha^t=\operatorname{softmax}_{\mathrm{hop}}\left(\operatorname{LN}\left(B_{\mathrm{pre}}^t\right) W_\alpha+b_\alpha\right)
$$

where $\alpha^t \in \mathbb{R}^{N_B \times (K+1)}$ and $\sum_{k=0}^K \alpha_{i k}^t=1$. $\alpha_{i k}^t$ measures how much graph-query token $i$ uses evidence propagated through $k$ topology transitions at round $t$. The final topology-diffused relevance distribution is:

$$
\widetilde{S}_{i v}^t=\sum_{k=0}^K \alpha_{i k}^t S_{i v}^{t,(k)}
$$

In matrix notation,  we have $\widetilde{S}^t=\sum_{k=0}^K \operatorname{Diag}\left(\alpha_{:, k}^t\right) S^{t,(k)}$. Substituting the diffusion equation gives

$$
\widetilde{S}^t=\sum_{k=0}^K \operatorname{Diag}\left(\alpha_{:, k}^t\right) S^{t,(0)} P^k
$$

This is a query-conditioned polynomial graph filter applied to semantic relevance rather than directly to node features. Because $S^{t,(0)}$ is row-stochastic and $P$ is row-stochastic, i.e., $\sum_v S_{i v}^{t,(k)}=1$ for every $i$ and $k$. Since the hop weights are also normalized, $\sum_{k=0}^K \alpha_{i k}^t=1$ the final distribution satisfies $\sum_v \tilde{S}_{i v}^t=1$. Therefore, $\tilde{S}_{i,:}^t$ remains a valid probability distribution over graph nodes.

The final graph evidence is obtained by weighted aggregation of graph values $\widehat{R}^t=\widetilde{S}^t V_H \in \mathbb{R}^{N_B \times d_r}$. An output projection maps the reader representation back to the LLM hidden space:

$$
R^t=\widehat{R}^t W_O=\tilde{S}^t V_H W_O \in \mathbb{R}^{N_B \times d},
$$

where $W_O \in \mathbb{R}^{d_r \times d}$. For the $i$-th graph-query token, we have

$$
R_i^t=\left(\sum_{v=1}^{N_V} \widetilde{S}_{i v}^t V_{H, v}\right) W_O
$$

Thus, $R_i^t$ is not a generic graph summary. It is the graph evidence specifically requested by the current state of graph-query token $i$. **The complete reader can be written as**

$$
\begin{aligned}
S^{t,(0)} & =\operatorname{softmax}_V\left(\frac{\operatorname{LN}\left(B_{\mathrm{pre}}^t\right) W_Q\left(\mathrm{LN}(H) W_K\right)^{\top}}{\sqrt{d_r}}\right), \\
S^{t,(k)} & =S^{t,(0)} P^k, \quad k=1, \ldots, K, \\
\alpha^t & =\operatorname{softmax}_{\mathrm{hop}}\left(\mathrm{LN}\left(B_{\mathrm{pre}}^t\right) W_\alpha+b_\alpha\right), \\
\widetilde{S}^t & =\sum_{k=0}^K \operatorname{Diag}\left(\alpha_{:, k}^t\right) S^{t,(k)}, \\
R^t & =\widetilde{S}^t V_H W_O .
\end{aligned}
$$

This is the canonical mathematical definition of the Topology-Diffused Graph Reader. A multi-head implementation can be obtained by applying the same procedure with independent query, key, and value projections for each head, concatenating the resulting evidence, and applying $W_O$. The conceptual mechanism remains unchanged. Using the notion we established, $\Gamma_t$ denotes the complete Read-Fuse-Replace operation.

For an arbitrary hidden-state sequence $Z$, let $B = Z[\mathcal{I}_B]$. Then $\Gamma_t(Z, H, P)=\operatorname{Replace}\left(Z, \mathcal{I}_B, \operatorname{Fuse}(B, \operatorname{Read}(B, H, P))\right)$. At round $t$, $Z_{\mathrm{post}}^t=\Gamma_t\left(Z_{\mathrm{pre}}^t, H, P\right)$. Although $\Gamma_t$ is indexed by $t$ because it is applied at the t-th position in the interleaved architecture, the canonical design can share the reader and fusion parameters across all rounds. In that case, the different behavior of each round comes from the evolving input $B_{\mathrm{pre}}^t$, rather than from separate round-specific readers.

## 2.4 Gated Residual Evidence Fusion $Fuse()$

The cleanest design is to make $\mathrm{Fuse}(\cdot)$ a **gated residual evidence injection**. The Topology-Diffused Graph Reader has already determined *what* graph evidence is relevant; Fuse should only determine *how much* of that evidence should be written into the current graph-query state. At graph-reading round $t$, $B_{\mathrm{pre}}^t, R^t \in \mathbb{R}^{N_B \times d}$, where $R_i^t$ is the graph evidence specifically read for the $i$-th graph-query token $B_{\mathrm{pre}, i}^t$. Fuse produces $B_{\text {post }}^t=\text { Fuse }\left(B_{\text {pre }}^t, R^t\right) \in \mathbb{R}^{N_B \times d}$. We first normalize the current language-side state and the newly read graph evidence separately:

$$
\widehat{B}^t=\mathrm{LN}_B\left(B_{\mathrm{pre}}^t\right), \quad \widehat{R}^t=\mathrm{LN}_R\left(R^t\right) .
$$

Separate normalization is useful because $B_{\mathrm{pre}}^t$ is produced by the LLM, whereas $R^t$ is produced by the graph reader. The two sources may have the different feature distributions. For each graph-query token, we predict an evidence-acceptance gate:

$$
g^t=\sigma\left(\left[\widehat{B}^t \| \widehat{R}^t\right] w_g+b_g\right),
$$

where $w_g \in \mathbb{R}^{2 d}$, $b_g \in \mathbb{R}$, and $g^t \in(0,1)^{N_B}$. Here, $\|$ denotes concatenation along the hidden dimension, and $g^t$ is one scalar gate for graph-query token $i$. The graph evidence is then injected through a residual update $B_{\text {post }}^t=B_{\text {pre }}^t+\operatorname{Diag}\left(g^t\right) \text { Dropout }\left(R^t\right)$. Equivalently, using row-wise broadcasting,

$$
B_{\text {post }}^t=B_{\text {pre }}^t+g^t \odot \text { Dropout }\left(R^t\right) .
$$

The scalar $g_i^t$ is broadcast across the $d$ dimensions of $R_i^t$. For an individual graph-query token, $b_{\mathrm{post}, i}^t=b_{\mathrm{pre}, i}^t+g_i^t r_i^t$. Ignoring dropout for clarity:

- If $g_i^t \approx 0$, the graph-query token preserves its current LLM state;
- If $g_i^t \approx 1$, the complete graph evidence is injected;
- If $0<g_i^t<1$, the evidence is integrated partially.

Thus, the gate controls the **strength of graph intervention** for every graph-query token and every reading round. **The complete Fusion function can be formulated as:**

$$
\begin{aligned}\widehat{B}^t & =\operatorname{LN}_B\left(B_{\mathrm{pre}}^t\right) \\\widehat{R}^t & =\operatorname{LN}_R\left(R^t\right) \\g^t & =\sigma\left(\left[\widehat{B}^t \| \widehat{R}^t\right] w_g+b_g\right) \\B_{\text {post }}^t & =B_{\text {pre }}^t+\operatorname{Diag}\left(g^t\right) \operatorname{Dropout}\left(R^t\right)\end{aligned}
$$

## 2.5 Answer Generation

Recall that the decoder-only LLM contain $L$ transformer layers, partitioned into $T+1$ non-overlapping consecutive groups $F_0, F_1, \ldots, F_T$. With these $T+1$ LLM blocks , there are naturally $T$ graph-reading rounds, inserted between adjacent LLM layer groups:

$$
F_0 \rightarrow \Gamma_0 \rightarrow F_1 \rightarrow \Gamma_1 \rightarrow \cdots \rightarrow \Gamma_{T-1} \rightarrow F_T \rightarrow \text { LMHead },
$$

where $\Gamma_t(Z_{\mathrm{pre}}^t, H, P) = \mathrm{Replace}(Z_{\mathrm{pre}}^t, \mathcal{I}_B,\mathrm{Fuse}(B_{\mathrm{pre}}^t, \mathrm{Read}(B_{\mathrm{pre}}^t, H, P)))$ with $B_{\mathrm{pre}}^t = Z_{\mathrm{pre}}^t[\mathcal{I}_B]$, denotes the complete **Read-Fuse-Replace retrieval operation,** which produces the input $Z_{\mathrm{post}}^t$ for the next LLM layer group $F_{t+1}$. We use $y_0$ as the answer-start token. At the autoregressive step $s \geq 1$, the model conditions on $y_{<s} = (y_1, \dots, y_{s-1})$, and the current input sequence is concetually represented as:

$$
Z_{\mathrm{in}}^{(s)}=\left[e\left(q_1\right), \ldots, e\left(q_{N_q}\right) ; B_{\mathrm{base}} ; e\left(y_0\right), e\left(y_1\right), \ldots, e\left(y_{s-1}\right)\right],
$$

where positional information is omitted for notational simplicity. The first LLM group produces $Z_{\mathrm{pre}}^{0,(s)}=F_0\left(Z_{\mathrm{in}}^{(s)}\right)$. Then for every graph-reading round $t = 0, \ldots, T-1$, we write the Read=Fuse-Replace operation compactly as $Z_{\mathrm{post}}^{t,(s)}=\Gamma_t\left(Z_{\mathrm{pre}}^{t,(s)}, H, P\right)$followed by the next LLM group $Z_{\mathrm{pre}}^{t+1,(s)}=F_{t+1}\left(Z_{\mathrm{post}}^{t,(s)}\right)$. Thus the entire graph-conditioned LLM forward pass is:

$$
Z_{\text {out }}^{(s)}=F_T \circ \Gamma_{T-1} \circ F_{T-1} \circ \cdots \circ \Gamma_0 \circ F_0\left(Z_{\text {in }}^{(s)}\right)
$$

After the final LLM group, apply the LLM’s ordinary final normalization by $\bar{Z}^{(s)}=\operatorname{LN}_{\text {final }}\left(Z_{\text {out }}^{(s)}\right)$, and the vocabulary logits are $\mathbf{\Lambda}^{(s)}=\bar{Z}^{(s)} W_{\text {vocab }}^{\top}+b_{\text {vocab }}$. Let $j_s$ denote the final position of the current input sequence, since the sequence is $\left[q_1, \ldots, q_{N_q}, B_{\text {base }}, y_0, y_1, \ldots, y_{s-1}\right]$, under one-based indexing, we have $j_s = N_q + N_B + s$. There, the next token is generated according to:

$$
p_\theta\left(y_s \mid q, G, y_{<s}\right)=\operatorname{softmax}\left(\mathbf{\Lambda}^{(s)}\left[j_s\right]\right)
$$

for $s = 1,2,\ldots$. The complete answer $Y = (y_1, \ldots, y_M)$ is consequently modeled by

$$
p_\theta(Y \mid q, G)=\prod_{s=1}^{M} p_\theta\left(y_s \mid q, G, y_{<s}\right) .
$$

Generation terminates when the model predicts $\langle\mathrm{EOS}\rangle$.

**Efficient inference with KV caching**. The formulation above describes autoregressive generation conceptually by applying the complete ReGraph forward process at every generation step $s$. In practice, however, repeatedly recomputing the instruction, graph-query tokens, and graph-reading operations is unnecessary. Because the input sequence follows the causal order $\left[q_1, \ldots, q_{N_q}, B_{\text {base }}, y_0, y_1, \ldots, y_{s-1}\right]$, the graph-query tokens always precede the generated response and therefore cannot attend to any $y_s$ under the causal attention mask. Consequently, their hidden states are invariant throughout autoregressive decoding:

$$
B_{\mathrm{pre}}^{t,(s)}=B_{\mathrm{pre}}^{t,(1)}, \quad \forall s \geq 1, t=0, \ldots, T-1 .
$$

Accordingly, all graph-reading operations $\Gamma_0, \ldots, \Gamma_{T-1}$ need to be executed only once during the **prefill stage**. Specifically, ReGraph first processes $\left[q_1, \ldots, q_{N_q}, B_{\text {base }}, y_0\right]$ through $F_0 \rightarrow \Gamma_0 \rightarrow F_1 \rightarrow \cdots \rightarrow \Gamma_{T-1} \rightarrow F_T$ producing the distribution of $y_1$ while storing the key-value states of all prefix positions at every Transformer layer. Importantly, for each $t$, the cached representations of the $B$ positions in $F_{t+1}$ are computed after $\Gamma_t$ has performed Read-Fuse-Replace, and therefore already encode the graph evidence injected at graph-reading round $t$. After sampling $y_1$, subsequent tokens are generated using standard incremental decoding. At generation step $s \geq 2$, only the newly appended token $y_{s-1}$ is propagated through $F_0, \ldots, F_T$, while attending to the previously stored KV cache. The operations $\Gamma_t$ are not recomputed during this stage, since they modify only the graph-query token positions whose graph-conditioned representations have already been cached. Nevertheless, each newly generated token can still access their information through attention to these cached $B$-token representations in the subsequent LLM groups. ReGraph performs iterative graph reading only once during prefill, while autoregressive decoding proceeds with the standard KV-cached LLM mechanism.

# 3. Experiments

We evaluate ReGraph on the GraphQA benchmark introduced by G-Retriever (He et al., 2024). GraphQA formulates graph reasoning as conditional language generation: given a text-attributed graph $G_i$ and a natural-language question $q_i$, the model autoregressively generates an answer $A_i$. We follow the official data splits, answer targets, meta-llama/Llama-3.1-8B-Instruct backbone, decoding procedure, and evaluation scripts.

## 3.1 Benchmarks and Data Splits

Each GraphQA example is a triple $(G_i,q_i,A_i)$. Nodes and edges carry textual attributes, and no task-specific prediction head is used.

| Dataset     | No. examples | Avg. nodes | Avg. edges | Prediction target             | Metric   |
| ----------- | ------------ | ---------- | ---------- | ----------------------------- | -------- |
| ExplaGraphs | 2,766        | 5.17       | 4.25       | `support` or `counter`    | Accuracy |
| SceneGraphs | 100,000      | 19.13      | 68.44      | Short natural-language answer | Accuracy |
| WebQSP      | 4,737        | 1,370.89   | 4,252.37   | One or more entity answers    | Hit@1    |

**ExplaGraphs** contains small directed explanation graphs whose nodes denote commonsense concepts and whose edges denote explanatory relations. The task is to determine whether two arguments support or counter each other. **SceneGraphs** is derived from GQA; nodes represent objects and attributes, while edges encode actions and spatial relations. Questions range from direct attribute queries to compositional relational reasoning. **WebQSP** is constructed from Freebase triples within two hops of entities mentioned in each question and may contain multiple valid entity answers. Following G-Retriever, valid WebQSP targets are concatenated with `|` during teacher-forced training.

We use the exact split indices released by G-Retriever. ExplaGraphs is split into 1,659/553/554 training, validation, and test examples. SceneGraphs follows the official 60/20/20 split over image identifiers, preventing questions from the same scene graph from crossing subsets. WebQSP preserves the original RoG train/validation/test partition; the empty validation graph removed by the official preprocessing is also excluded.

## 3.2 Training and Inference

For every example, ReGraph constructs a node memory and topology operator from the original graph:

$$
H_i=E_\phi(G_i), \qquad P_i=\widetilde D_i^{-1}(A_i+I).
$$

Node and relation text is initialized with `sentence-transformers/all-roberta-large-v1`, and a relation-aware four-layer Graph Transformer produces the 1,024-dimensional node memory. The language input contains the question $q_i$, the learnable graph-query tokens $B_{\mathrm{base}}$, and an answer boundary $y_0$; the graph is not serialized into the LLM context.

During teacher forcing, the gold answer and EOS are appended after $y_0$. ReGraph is trained with answer-only next-token likelihood:

$$
\mathcal L_{\mathrm{gen}}=-\sum_{i\in\mathcal D_{\mathrm{train}}}\sum_{s=1}^{|A_i|+1}\log p_\Theta\left(a_{i,s}\mid q_i,G_i,a_{i,<s}\right).
$$

Losses on the question, graph-query tokens, answer boundary, and padding are masked. GraphQA provides no intermediate supervision for graph-reading distributions, hop weights, fusion gates, or reasoning states; all ReGraph components are learned end to end from the final answer likelihood.

The primary setting uses meta-llama/Llama-3.1-8B-Instruct with all original LLM parameters frozen. We train the graph encoder, $B_{\mathrm{base}}$, the Topology-Diffused Graph Reader, Fuse, and graph-to-LLM projections. We train a separate model for each dataset and select the checkpoint with the lowest validation loss.

At test time, ReGraph receives only $(G_i,q_i)$. It performs all Read–Fuse–Replace rounds during prefill,

$$
F_0\rightarrow\Gamma_0\rightarrow F_1\rightarrow\cdots\rightarrow\Gamma_{T-1}\rightarrow F_T,
$$

and then generates the answer with standard KV-cached autoregressive decoding. We use greedy decoding, disable sampling, and stop at EOS or 32 generated tokens, matching G-Retriever.

## 3.3 Implementation Details

We optimize trainable parameters with AdamW using learning rate $10^{-5}$ and weight decay 0.05. Models are trained for at most 10 epochs with warm-up followed by cosine decay and early stopping with patience two. The training batch size is four. Unless otherwise stated, ReGraph uses eight graph-query tokens, three graph-reading rounds, maximum diffusion depth $K=2$, reader dimension 1,024, eight reader heads, and dropout 0.1. The LLM is divided into four consecutive groups, and Reader and Fuse parameters are shared across rounds.

We evaluate the contribution of iterative reading with controlled ablations: one graph read ($T=1$), no topology diffusion ($P=I$), fixed hop weights, resetting the query state before every read, removing the Fuse gate, and replacing ReGraph with a mean-pooled graph token. All variants use the same graph inputs, supervision, backbone, optimization budget, and evaluation procedure.
