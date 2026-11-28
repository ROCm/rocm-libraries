# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The differential golden file: load, validate, record and compare.

`golden/llvm_gfx_all.json` anchors the Python reference output of every parity
family's sampled configs, as one sha256 per config. This module owns its format
and nothing else; `run_diff.py` runs the emitters and hands it the hashes.

Format (schema 2)::

    {
      "schema": 2,
      "aliases": {},              # optional lanes sharing an emission profile
      "ll": {                      # LLVM IR text, one section per LLVM profile
        "llvm20": {"<family>": {"<config index>": "<sha256>", ...}, ...},
        "llvm22": {...},
        "llvm23": {...},
        "llvm24": {...}
      },
      "ir_canonical": {            # canonical ck.dsl.ir: the same in every flavor
        "<family>": {"<config index>": "<sha256>", ...}
      },
      "fingerprint": {             # one per family, written by save()
        "<family>": {"sha256": "<fingerprint>"}
      }
    }

LLVM IR text depends on the flavor it is written for, so each flavor has its
own section and a hash is only ever compared with one recorded for the same
flavor. Canonical IR is produced before lowering and has one section.

A family's fingerprint (`rocke.core.golden_file.fingerprint`) covers its entries
`{"ll": {<flavor>: <its configs>, ...}, "ir_canonical": <its configs>}`, each
section present only when the family is in it. Two changes that re-record one
family from different trees both rewrite its fingerprint line, so git reports a
conflict instead of merging their hashes. Each fingerprint sits in its own JSON
object so neighbouring families' lines are never adjacent, and re-recording one
family never conflicts with re-recording another.

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

from rocke.core.golden_file import FINGERPRINT_KEY, fingerprint

SCHEMA = 2
LL = "ll"
IR_CANONICAL = "ir_canonical"
KINDS = (LL, IR_CANONICAL)
# The key of a family's fingerprint inside its FINGERPRINT_KEY entry.
_FINGERPRINT_SHA = "sha256"

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

    aliases: dict[str, str] = field(default_factory=dict)
    ll: dict[str, FamilyShas] = field(default_factory=dict)
    ir_canonical: FamilyShas = field(default_factory=dict)
    # The fingerprint the file records per family (see the module docstring).
    recorded_fingerprints: dict[str, str] = field(default_factory=dict)

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
        unknown = sorted(set(document) - {"schema", "aliases", FINGERPRINT_KEY, *KINDS})
        if unknown:
            raise GoldenFormatError(
                f"golden file {path} has unknown sections {unknown}"
            )
        aliases = document.get("aliases", {})
        if not isinstance(aliases, dict) or not all(
            isinstance(lane, str) and isinstance(source, str)
            for lane, source in aliases.items()
        ):
            raise GoldenFormatError(
                "golden aliases must map flavor strings to flavor strings"
            )
        ll = document.get(LL, {})
        overlap = sorted(set(aliases) & set(ll))
        if overlap:
            raise GoldenFormatError(
                f"golden flavors are both direct and aliased: {overlap}"
            )
        missing = sorted(source for source in aliases.values() if source not in ll)
        if missing:
            raise GoldenFormatError(
                f"golden aliases name missing flavor sections: {missing}"
            )
        return cls(
            aliases=aliases,
            ll=ll,
            ir_canonical=document.get(IR_CANONICAL, {}),
            recorded_fingerprints=_recorded_fingerprints(
                document.get(FINGERPRINT_KEY, {})
            ),
        )

    def save(self, path: Path) -> None:
        """Write the store with a fresh fingerprint for every family."""
        document = {
            "schema": SCHEMA,
            "aliases": self.aliases,
            LL: self.ll,
            IR_CANONICAL: self.ir_canonical,
            FINGERPRINT_KEY: {
                family: {_FINGERPRINT_SHA: value}
                for family, value in self.fingerprints().items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")

    # ---- fingerprints -----------------------------------------------------

    def fingerprints(self) -> dict[str, str]:
        """The fingerprint of every family, computed from its hashes."""
        entries: dict[str, dict] = {}
        for flavor, families in self.ll.items():
            for family, configs in families.items():
                entries.setdefault(family, {}).setdefault(LL, {})[flavor] = configs
        for family, configs in self.ir_canonical.items():
            entries.setdefault(family, {})[IR_CANONICAL] = configs
        return {family: fingerprint(entries[family]) for family in sorted(entries)}

    def stale_fingerprints(self, families: Iterable[str] | None = None) -> list[str]:
        """Families (all, or those in `families`) whose recorded fingerprint is
        missing or no longer matches their hashes, sorted. A family recorded
        only in the fingerprint map (its hashes are gone) is stale too."""
        computed = self.fingerprints()
        scope = (
            set(computed) | set(self.recorded_fingerprints)
            if families is None
            else set(families)
        )
        return sorted(
            family
            for family in scope
            if self.recorded_fingerprints.get(family) != computed.get(family)
        )

    def families(self) -> set[str]:
        """Every family the file names: in any section or the fingerprints."""
        return set(self.fingerprints()) | set(self.recorded_fingerprints)

    # ---- sections ---------------------------------------------------------

    def section(self, kind: str, flavor: str | None) -> FamilyShas:
        """The hashes for `kind`; `ll` needs the flavor, `ir_canonical` takes none."""
        _check_section(kind, flavor)
        if kind == IR_CANONICAL:
            return self.ir_canonical
        source = self.aliases.get(flavor, flavor)
        return self.ll.get(source, {})

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
            source = self.aliases.get(flavor, flavor)
            self.ll[source] = merged
        else:
            self.ir_canonical = merged
        return sorted(set(before) - set(merged))


@dataclass(frozen=True)
class Comparison:
    """A golden check: configs whose hash changed, vanished or is new (has
    output and no blessed hash)."""

    blessed: int
    changed: tuple[tuple[str, str], ...]
    missing: tuple[tuple[str, str], ...]
    new: tuple[tuple[str, str], ...]

    @property
    def ok(self) -> bool:
        """Changed, missing and new configs all fail: a new config is output
        that no golden checks until it is blessed."""
        return not self.changed and not self.missing and not self.new


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


def _recorded_fingerprints(recorded: object) -> dict[str, str]:
    """The `fingerprint` section as {family: fingerprint}; refuses any other shape."""
    if not isinstance(recorded, dict) or not all(
        isinstance(entry, dict)
        and set(entry) == {_FINGERPRINT_SHA}
        and isinstance(entry[_FINGERPRINT_SHA], str)
        for entry in recorded.values()
    ):
        raise GoldenFormatError(
            f"golden {FINGERPRINT_KEY!r} must map each family to "
            f'{{"{_FINGERPRINT_SHA}": "<fingerprint>"}}; re-record the file'
        )
    return {family: entry[_FINGERPRINT_SHA] for family, entry in recorded.items()}


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
