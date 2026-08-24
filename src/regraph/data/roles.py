"""Query-role markers r_v(q) in {none, mentioned, source, target} (`ReGraph.md` §2.1).

Assignment happens only when the reference resolves unambiguously
(docs/components/01-data.md, docs/OPEN-QUESTIONS.md Q7):

- `mentioned`: exact normalized full-string match of the node text against a
  question span (word-boundary substring after normalization). If several nodes
  share the same normalized text the match is ambiguous -> all stay `none`.
- `source`: only where the dataset supplies it explicitly (WebQSP topic entity
  `q_entity`). Never guessed from word order.
- `target`: unused (no dataset supplies it).
- Anchor-free questions leave every node `none` — a first-class case.
"""

from __future__ import annotations

import re
import string
from collections import Counter

ROLE_NONE, ROLE_MENTIONED, ROLE_SOURCE, ROLE_TARGET = 0, 1, 2, 3
NUM_ROLES = 4

_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")


def normalize_for_roles(s: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace."""
    s = str(s).lower().translate(_PUNCT_TABLE)
    s = _ARTICLE_RE.sub(" ", s)
    return " ".join(s.split())


def assign_roles(
    question: str,
    node_texts: list[str],
    source_entities: list[str] | None = None,
) -> list[int]:
    """Role id per node. `source_entities` are dataset-supplied anchors (WebQSP q_entity)."""
    roles = [ROLE_NONE] * len(node_texts)

    q_norm = " " + normalize_for_roles(question) + " "
    norm_nodes = [normalize_for_roles(t) for t in node_texts]
    counts = Counter(norm_nodes)
    for i, nt in enumerate(norm_nodes):
        if not nt:
            continue
        if counts[nt] > 1:  # multi-node match -> ambiguous -> none
            continue
        if f" {nt} " in q_norm:
            roles[i] = ROLE_MENTIONED

    if source_entities:
        # WebQSP node texts are lowercased at graph build time; q_entity is matched
        # by exact lowercased string. Node texts are unique by construction there.
        lowered = {t.lower(): i for i, t in enumerate(node_texts)}
        for ent in source_entities:
            i = lowered.get(str(ent).lower())
            if i is not None:
                roles[i] = ROLE_SOURCE

    return roles
