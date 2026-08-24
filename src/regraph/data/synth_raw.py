"""StructuralAnomaly — a synthetic, anchor-free benchmark for open-ended graph reasoning.

Why this exists. `ReGraph.md` §2.1 motivates the method with an **anchor-free** query --
"Which region of the graph is becoming structurally unstable?" -- where "all nodes receive the
`none` marker" and "the model then locates relevant graph regions through semantic and
structural attention rather than beginning from a predefined question entity". Every benchmark
evaluated in this repo is *anchored*: GraphQA marks question entities, the TAG datasets and
NeighborhoodQA mark a centre node, NLGraph names both endpoints. The anchor-free path -- the
spec's headline capability -- has never been evaluated. This benchmark evaluates it.

Task. A graph contains `k` equally sized thematic communities. Exactly one of them is wired far
more densely than the others. No node is named in the question; every node carries `ROLE_NONE`.
The model must locate the structurally anomalous region and then say *what it is about*, as free
text.

What makes it a real test:

1. **Chance is provable and low.** All `k` themes are present in every graph, each is equally
   likely to be the anomalous one, and all communities have the same size. No policy that
   ignores topology can exceed `1/k`.
2. **Structure and semantics must be joined.** Topology alone identifies the region but cannot
   name it; text alone names all `k` regions but cannot say which is anomalous. Neither channel
   suffices, which is precisely the `Read` operation the spec describes.
3. **A falsifiable control ships with it.** `anomaly=False` removes the density contrast and
   assigns the label at random; any model must then score exactly `1/k`. Anything above chance
   on the control means the generator leaks and the main number is void.
4. **Difficulty is continuous.** `density_ratio`, `num_communities` and `community_size` are
   dials, so the benchmark yields a curve rather than a point.

Ground truth is exact by construction, so no human or LLM judging is involved.
"""

from __future__ import annotations

import numpy as np
import torch

EDGE_TEXT = "linked to"

# Five themes with disjoint vocabularies. Node text is a phrase, so within-community nodes are
# distinct but recognisably of one theme; sbert places them in a tight cluster.
THEMES: dict[str, list[str]] = {
    "marine biology": [
        "coral reef bleaching survey", "humpback whale migration route", "deep sea vent tubeworms",
        "plankton bloom satellite record", "sea otter foraging behaviour", "mangrove nursery habitat",
        "kelp forest canopy density", "reef fish larval dispersal", "tidal pool invertebrates",
        "shark tagging telemetry", "seagrass meadow carbon storage", "jellyfish swarm dynamics",
    ],
    "cryptography": [
        "elliptic curve key exchange", "lattice based signature scheme", "side channel timing attack",
        "zero knowledge proof system", "block cipher differential analysis", "post quantum key encapsulation",
        "homomorphic encryption circuit", "hash collision resistance bound", "secret sharing threshold scheme",
        "random oracle security proof", "stream cipher keystream bias", "authenticated encryption mode",
    ],
    "jazz music": [
        "bebop chord substitution", "walking bass line phrasing", "modal improvisation over dorian",
        "big band horn arrangement", "swing rhythm section feel", "hard bop trumpet solo",
        "jazz standard reharmonisation", "brush technique on snare", "vocal scat improvisation",
        "cool jazz nonet recording", "free jazz collective improvisation", "latin jazz clave pattern",
    ],
    "volcanology": [
        "basaltic lava flow rheology", "caldera collapse subsidence", "pyroclastic density current",
        "magma chamber seismic tomography", "volcanic ash plume dispersion", "phreatomagmatic eruption deposit",
        "fumarole gas geochemistry", "lava dome extrusion rate", "tephra layer stratigraphy",
        "volcanic tremor monitoring", "rift zone dyke intrusion", "crater lake acidity survey",
    ],
    "textile manufacturing": [
        "ring spinning yarn tension", "jacquard loom weave pattern", "reactive dye fixation rate",
        "cotton fibre staple length", "knitted fabric loop density", "warp beam sizing process",
        "mercerisation caustic treatment", "nonwoven needle punching", "worsted wool combing",
        "fabric shrinkage after washing", "twill weave float length", "spindle draft ratio",
    ],
}

QUESTION = ("This graph shows a collection of items and the links between them. One group of "
            "closely related items is far more densely interconnected than any other group. "
            "What is that group about?")
QUESTION_CONTROL = ("This graph shows a collection of items and the links between them. One "
                    "group of closely related items has been singled out. What is that group "
                    "about?")


def _sample_graph(rng: np.random.Generator, theme_names: list[str], community_size: int,
                  p_in: float, p_hot: float, p_out: float, hot: int | None):
    """Planted-partition graph; community `hot` (if any) gets internal probability `p_hot`."""
    k = len(theme_names)
    n = k * community_size
    member = np.repeat(np.arange(k), community_size)
    edges: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            same = member[i] == member[j]
            if same:
                p = p_hot if (hot is not None and member[i] == hot) else p_in
            else:
                p = p_out
            if rng.random() < p:
                edges.add((i, j))
    # keep every community connected so "dense" is a contrast, not a connectivity artefact
    for c in range(k):
        idx = np.where(member == c)[0]
        for a, b in zip(idx[:-1], idx[1:]):
            edges.add((int(a), int(b)))
    ei = np.array(sorted(edges), dtype=np.int64).T
    ei = np.concatenate([ei, ei[::-1]], axis=1)          # undirected -> both directions
    return n, member, ei


def build_synth(num_communities: int = 5, community_size: int = 12,
                p_in: float = 0.12, density_ratio: float = 5.0, p_out: float = 0.01,
                n_train: int = 8000, n_val: int = 1000, n_test: int = 2000,
                seed: int = 0, anomaly: bool = True):
    """Returns the standard (graphs, examples, texts, names, stats) tuple.

    `anomaly=False` builds the control: no density contrast, label drawn uniformly at random,
    so the task is unanswerable and any score above `1/k` indicates leakage.
    """
    names = sorted(THEMES)
    assert num_communities <= len(names), f"only {len(names)} themes available"
    rng = np.random.default_rng(seed)

    # global text table: every theme phrase, then the edge text
    texts: list[str] = []
    phrase_id: dict[tuple[str, int], int] = {}
    for t in names:
        for i, ph in enumerate(THEMES[t]):
            phrase_id[(t, i)] = len(texts)
            texts.append(ph)
    edge_tid = len(texts)
    texts.append(EDGE_TEXT)

    p_hot = min(p_in * density_ratio, 0.95)
    graphs, examples = {}, []
    hot_counts: dict[str, int] = {t: 0 for t in names}
    deg_gap = []
    gid = 0
    for split, cap in (("train", n_train), ("val", n_val), ("test", n_test)):
        for _ in range(cap):
            themes = list(rng.permutation(names)[:num_communities])
            hot = int(rng.integers(num_communities))
            n, member, ei = _sample_graph(
                rng, themes, community_size, p_in, p_hot, p_out,
                hot if anomaly else None)

            node_text_id = np.empty(n, dtype=np.int64)
            for i in range(n):
                t = themes[member[i]]
                node_text_id[i] = phrase_id[(t, int(rng.integers(len(THEMES[t]))))]

            deg = np.bincount(ei[0], minlength=n)
            in_hot = deg[member == hot].mean()
            out_hot = deg[member != hot].mean()
            deg_gap.append(float(in_hot - out_hot))

            key = str(gid); gid += 1
            graphs[key] = {
                "node_text_id": torch.tensor(node_text_id, dtype=torch.int32),
                "edge_text_id": torch.full((ei.shape[1],), edge_tid, dtype=torch.int32),
                "edge_index": torch.tensor(ei, dtype=torch.int32),
            }
            answer = themes[hot]
            hot_counts[answer] += 1
            examples.append({
                "id": int(key), "graph_key": key,
                "question": QUESTION if anomaly else QUESTION_CONTROL,
                "answer": answer, "split": split,
                # anchor-free: no roles at all -> collate fills ROLE_NONE everywhere (§2.1)
                "n_nodes": int(n), "hot_community": hot,
            })

    total = sum(hot_counts.values())
    stats = {
        "chance_setacc": round(100.0 / num_communities, 2),
        "theme_balance": {t: round(c / total * 100, 2) for t, c in hot_counts.items()},
        "mean_degree_gap_hot_vs_rest": round(float(np.mean(deg_gap)), 2),
        "p_in": p_in, "p_hot": round(p_hot, 4), "p_out": p_out,
        "anomaly": anomaly,
    }
    return graphs, examples, texts, names, stats
