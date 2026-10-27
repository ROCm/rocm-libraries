# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The differential golden file: load, validate, record and compare.

`golden/llvm_gfx_all.json` anchors the Python reference output of every parity
family's sampled configs, as one sha256 per config. This module owns its format
and nothing else; `run_diff.py` runs the emitters and hands it the hashes.

Format (schema 2)::

    {
      "schema": 2,
      "ll": {                      # LLVM IR text, one section per LLVM flavor
        "llvm20": {"<family>": {"<config index>": "<sha256>", ...}, ...},
        "llvm22": {...},
        "llvm23": {...}
      },
      "ir_canonical": {            # canonical ck.dsl.ir: the same in every flavor
        "<family>": {"<config index>": "<sha256>", ...}
      }
    }

LLVM IR text depends on the flavor it is written for, so each flavor has its
own section and a hash is only ever compared with one recorded for the same
flavor. Canonical IR is produced before lowering and has one section.

Recording or checking a subset of families touches only those families: a
partial record replaces exactly the families it ran, and a partial check
compares exactly those, so neither can wipe or falsely report the rest. A full
record replaces the whole section, so a family that no longer exists drops out.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

SCHEMA = 2
LL = "ll"
IR_CANONICAL = "ir_canonical"
KINDS = (LL, IR_CANONICAL)

# {family: {config index (as a string): sha256}}
FamilyShas = dict[str, dict[str, str]]


class GoldenFormatError(ValueError):
    """The golden file cannot be trusted as written (wrong schema, duplicate
    keys, malformed sections)."""


def _refuse_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    """json object hook: a repeated key would silently keep only its last value."""
    repeated = sorted(
        key for key, count in Counter(k for k, _ in pairs).items() if count > 1
    )
    if repeated:
        raise GoldenFormatError(
            f"duplicate keys {repeated} in the golden file -- JSON keeps only the last "
            "copy, hiding the others; merge them into one entry"
        )
    return dict(pairs)


def _check_section(kind: str, flavor: str | None) -> None:
    """`ll` is stored per LLVM flavor; `ir_canonical` has a single section."""
    if kind not in KINDS:
        raise ValueError(f"unknown golden kind {kind!r}, expected one of {list(KINDS)}")
    if kind == LL and not flavor:
        raise ValueError("the ll section needs an LLVM flavor (llvm20, llvm22, ...)")


@dataclass
class GoldenStore:
    """The parsed golden: hashes for `ll` per flavor, and for `ir_canonical`."""

    ll: dict[str, FamilyShas] = field(default_factory=dict)
    ir_canonical: FamilyShas = field(default_factory=dict)

    # ---- file I/O ---------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "GoldenStore":
        """Read `path`; a missing file is an empty store."""
        if not path.exists():
            return cls()
        document = json.loads(
            path.read_text(), object_pairs_hook=_refuse_duplicate_keys
        )
        schema = document.get("schema")
        if schema != SCHEMA:
            raise GoldenFormatError(
                f"golden file {path} has schema {schema!r}, expected {SCHEMA}; the older "
                "format stored no LLVM flavor -- re-record it with run_diff.py "
                "--record-golden for each flavor"
            )
        unknown = sorted(set(document) - {"schema", *KINDS})
        if unknown:
            raise GoldenFormatError(
                f"golden file {path} has unknown sections {unknown}"
            )
        return cls(ll=document.get(LL, {}), ir_canonical=document.get(IR_CANONICAL, {}))

    def save(self, path: Path) -> None:
        document = {"schema": SCHEMA, LL: self.ll, IR_CANONICAL: self.ir_canonical}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")

    # ---- sections ---------------------------------------------------------

    def section(self, kind: str, flavor: str | None) -> FamilyShas:
        """The hashes for `kind`; `ll` needs the flavor, `ir_canonical` takes none."""
        _check_section(kind, flavor)
        return self.ll.get(flavor, {}) if kind == LL else self.ir_canonical

    def record(
        self,
        kind: str,
        flavor: str | None,
        shas: Mapping[str, Mapping[str, str]],
        families: Iterable[str] | None = None,
    ) -> list[str]:
        """Store `shas` for `kind` (and `flavor`); return the families whose
        hashes this removed, sorted, so the caller can say so.

        `families=None` is a full record: the section becomes exactly `shas`.
        Otherwise only the named families are replaced (removed when `shas` has
        nothing for one); every other family in the section is kept.
        """
        _check_section(kind, flavor)
        before = self.section(kind, flavor)
        fresh = {family: dict(configs) for family, configs in shas.items()}
        if families is None:
            merged = fresh
        else:
            merged = dict(before)
            for family in families:
                if family in fresh:
                    merged[family] = fresh[family]
                else:
                    merged.pop(family, None)
        if kind == LL:
            self.ll[flavor] = merged
        else:
            self.ir_canonical = merged
        return sorted(set(before) - set(merged))


@dataclass(frozen=True)
class Comparison:
    """A golden check: configs whose hash changed, vanished or is new."""

    blessed: int
    changed: tuple[tuple[str, str], ...]
    missing: tuple[tuple[str, str], ...]
    new: tuple[tuple[str, str], ...]

    @property
    def ok(self) -> bool:
        """New configs are informational; changed or missing ones fail."""
        return not self.changed and not self.missing


def compare(
    blessed: FamilyShas,
    current: Mapping[str, Mapping[str, str]],
    families: Iterable[str] | None = None,
) -> Comparison:
    """Compare `current` with `blessed`, over `families` only when given (a
    family outside them is neither checked nor reported missing)."""
    scope = set(blessed) | set(current) if families is None else set(families)
    changed, missing, new = [], [], []
    for family in sorted(scope):
        old = blessed.get(family, {})
        now = current.get(family, {})
        for idx in sorted(old, key=_index_order):
            if idx not in now:
                missing.append((family, idx))
            elif now[idx] != old[idx]:
                changed.append((family, idx))
        new.extend(
            (family, idx) for idx in sorted(now, key=_index_order) if idx not in old
        )
    return Comparison(
        blessed=sum(len(blessed.get(family, {})) for family in scope),
        changed=tuple(changed),
        missing=tuple(missing),
        new=tuple(new),
    )


def _index_order(idx: str) -> tuple[int, str]:
    return (int(idx), idx) if idx.isdigit() else (1 << 62, idx)


__all__ = [
    "Comparison",
    "GoldenFormatError",
    "GoldenStore",
    "IR_CANONICAL",
    "KINDS",
    "LL",
    "SCHEMA",
    "compare",
]
