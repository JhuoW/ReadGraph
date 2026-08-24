"""Run-directory bookkeeping: resolved config, git SHA, seed, JSONL metric logs.

Every run writes its resolved config, git SHA, and seed to the run dir
(CLAUDE.md ground rule 6).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


def git_sha(repo_root: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)

    def log(self, record: dict[str, Any]) -> None:
        record = {"time": time.time(), **record}
        self._fh.write(json.dumps(record, default=float) + "\n")

    def close(self) -> None:
        self._fh.close()


class RunDir:
    """Creates the run directory and dumps provenance (config, git SHA, seed, argv)."""

    def __init__(self, path: str | Path, config: dict, resume: bool = False):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=(resume or not self.path.exists() or True))
        with open(self.path / "resolved_config.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        with open(self.path / "provenance.json", "w") as f:
            json.dump(
                {
                    "git_sha": git_sha(),
                    "seed": config.get("seed"),
                    "argv": sys.argv,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
                indent=2,
            )
        self.metrics = JsonlLogger(self.path / "metrics.jsonl")

    def file(self, name: str) -> Path:
        return self.path / name
