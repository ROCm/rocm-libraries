"""Reader for rpp-test-capture bundles (see README.md for the format)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT = "rpp-test-capture"
VERSION = 1

NOT_COMPARED, PASS, FAIL = 0, 1, 2


@dataclass
class Comparison:
    kind: str
    axes: list[str]
    compared: int
    failed: int
    reference: np.ndarray
    actual: np.ndarray
    diff: np.ndarray  # actual - reference, NaN where not compared
    status: np.ndarray  # NOT_COMPARED / PASS / FAIL


@dataclass
class Case:
    path: Path
    name: str
    passed: bool
    failure: str
    comparisons: list[dict]

    @property
    def failed_elements(self) -> int:
        return sum(c["failed"] for c in self.comparisons)

    def load(self, index: int) -> Comparison:
        entry = self.comparisons[index]
        arrays = {
            role: np.load(self.path / file) for role, file in entry["arrays"].items()
        }
        return Comparison(
            entry["kind"],
            list(entry["axes"]),
            entry["compared"],
            entry["failed"],
            **arrays,
        )


def read_case(manifest: Path) -> Case:
    data = json.loads(manifest.read_text())
    if data.get("format") != FORMAT or data.get("version") != VERSION:
        raise ValueError(f"{manifest}: not a {FORMAT} v{VERSION} manifest")
    return Case(
        manifest.parent,
        data["case"],
        data["passed"],
        data["failure"],
        data["comparisons"],
    )


def find_cases(root: Path) -> list[Case]:
    """Every bundle under root (or root itself), sorted by case name."""
    manifests = [root] if root.is_file() else sorted(root.rglob("manifest.json"))
    return sorted((read_case(m) for m in manifests), key=lambda c: c.name)
