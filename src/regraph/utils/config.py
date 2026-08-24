"""Config loading: YAML deep-merge with `defaults:` chaining and `${...}` interpolation.

`configs/default.yaml` holds everything; dataset files declare `defaults: default.yaml`
and override. Any value read at runtime that is *not* in the config is a bug
(docs/components/00-conventions.md).
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

_INTERP_RE = re.compile(r"\$\{([a-zA-Z0-9_.]+)\}")


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base` (override wins). Returns a new dict."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _get_dotted(cfg: dict, dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"config key not found: {dotted!r}")
        node = node[part]
    return node


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _interpolate(cfg: dict) -> dict:
    """Resolve `${a.b.c}` references against the fully merged config (one pass, repeated)."""

    def resolve(value: Any) -> Any:
        if isinstance(value, str):
            def repl(m: re.Match) -> str:
                return str(_get_dotted(cfg, m.group(1)))
            prev = None
            while prev != value and _INTERP_RE.search(value):
                prev = value
                value = _INTERP_RE.sub(repl, value)
            return value
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value

    return resolve(cfg)


def _parse_override_value(raw: str) -> Any:
    """Parse a CLI override value with YAML semantics (`true`, `[a,b]`, ...).

    PyYAML's float resolver requires a decimal point before the exponent, so it reads
    `1e-5` as the *string* `"1e-5"` — and `str(1e-05)` renders exactly that form, which
    made `train.lr=1e-5` silently produce a string. Fall back to numeric parsing so
    scientific notation behaves as written.
    """
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config, following its `defaults:` chain, then apply CLI overrides.

    `overrides` are `dotted.key=value` strings. Interpolation (`${...}`) runs last.
    """
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    chain = [cfg]
    node = cfg
    base_dir = path.parent
    while "defaults" in node:
        default_path = base_dir / node.pop("defaults")
        with open(default_path) as f:
            node = yaml.safe_load(f) or {}
        chain.append(node)
        base_dir = default_path.parent

    merged: dict = {}
    for layer in reversed(chain):  # base first, most specific last
        merged = deep_merge(merged, layer)

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must look like key=value, got {ov!r}")
        key, _, raw = ov.partition("=")
        _set_dotted(merged, key.strip(), _parse_override_value(raw.strip()))

    return _interpolate(merged)
