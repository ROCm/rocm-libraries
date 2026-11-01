# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Which MMA catalog instructions rocke's backend issues, per target.

``data/backend_coverage.json`` records, for every (catalog row, target) pair, one measured class
saying whether rocke's backend issues the row's instruction there, and the backend op
(``core.arch`` ``MmaOp.op_id``) it was matched to. The record is measured and pinned by
``tests/helpers/tiling/traits/test_mma_backend_coverage.py``; ``traits/README.md`` ("Backend
coverage") explains the classes.

Two kinds of row hold more than one backend op per target, one per *variant*:

- a ``scaled`` row, one per scale format: the (A, B) pair of scale dtypes the op decodes
  (``scale_format=("e8m0", "e5m3")``);
- an integer row whose operand signedness is ``selectable``, one per (A, B) operand dtype pair
  (``operand_dtypes=("u8", "i8")``).

Such a pair is recorded once per variant. A lookup of a scaled pair names its scale format
whenever the pair has more than one. A lookup of a selectable integer pair without
``operand_dtypes`` answers for the row's own operand dtypes (``i8`` / ``i4``, the signed forms);
``operand_dtypes`` names an unsigned twin.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ....core.lower_llvm import LLVM_FLAVORS
from .mma_traits import _INTEGER_OPERAND_DTYPES

__all__ = [
    "COVERAGE_CLASSES",
    "DEFAULT_COVERAGE_PATH",
    "BackendCoverage",
    "CoverageEntry",
    "load_backend_coverage",
]

DEFAULT_COVERAGE_PATH = (
    Path(__file__).resolve().parent / "data" / "backend_coverage.json"
)

# The classes, in the order the gap test decides them (the first that holds wins).
COVERAGE_CLASSES = (
    "no_backend_op",
    "other_instruction",
    "multi_block",
    "lowering_not_wired",
    "operand_form_words_only",
    "lane_map_differs",
    "invalid_ir",
    "issuable",
)

_ENTRY_FIELDS = frozenset(
    {"class", "backend_op", "scale_format", "operand_dtypes", "reason"}
)
_VARIANT_FIELDS = ("scale_format", "operand_dtypes")

Variant = tuple[str, str]


@dataclass(frozen=True)
class CoverageEntry:
    """The measured class of one (row, target) pair, or of one variant of it.

    ``backend_op`` is the matched backend op id (``None`` only for ``no_backend_op``; for
    ``other_instruction`` it is the op with the row's operand key that lowers to another
    instruction). ``scale_format`` / ``operand_dtypes`` name the variant, when the pair is
    recorded per variant. ``reason`` is the backend's message for ``lowering_not_wired``.
    """

    status: str
    backend_op: str | None
    scale_format: Variant | None = None
    operand_dtypes: Variant | None = None
    reason: str | None = None


def _pair_text(op_id: str, target: str, entry: CoverageEntry | None = None) -> str:
    """``('x', 'gfx942')``, with the variant appended when the entry has one."""
    text = repr((op_id, target))
    if entry is not None:
        for name in _VARIANT_FIELDS:
            value = getattr(entry, name)
            if value is not None:
                text = f"{text[:-1]}, {name}={value!r})"
    return text


@dataclass(frozen=True)
class BackendCoverage:
    """The loaded backend coverage record.

    ``llvm_flavor`` is the LLVM flavor the classes were measured at. ``entries`` maps each
    (catalog op_id, target) pair to its entries: one, or one per variant (module docstring).
    ``ops_without_row`` maps each target to the backend ops no row was matched to.
    """

    llvm_flavor: str
    entries: Mapping[tuple[str, str], tuple[CoverageEntry, ...]]
    ops_without_row: Mapping[str, tuple[str, ...]]

    def entry(
        self,
        op_id: str,
        target: str,
        *,
        scale_format: Variant | None = None,
        operand_dtypes: Variant | None = None,
    ) -> CoverageEntry:
        """The recorded entry of (``op_id``, ``target``), of the named variant.

        A pair recorded per variant takes its variant (``scale_format`` for a scaled row,
        ``operand_dtypes`` for a selectable integer row). A scale format must be named when the
        pair has more than one, and may be omitted when it has one. Omitted ``operand_dtypes``
        means the row's own operand dtypes, so ``(op_id, target)`` alone answers for the row as
        the catalog lists it. A pair recorded without variants refuses both arguments.
        """
        entries = self.entries.get((op_id, target))
        if entries is None:
            raise KeyError(
                f"no backend coverage recorded for {_pair_text(op_id, target)} -- the gap "
                f"test records every (row, target) pair; re-run it"
            )
        asked = {"scale_format": scale_format, "operand_dtypes": operand_dtypes}
        for name, value in asked.items():
            if value is None:
                continue
            recorded = [getattr(e, name) for e in entries]
            if all(v is None for v in recorded):
                raise ValueError(
                    f"{_pair_text(op_id, target)} is recorded without variants -- "
                    f"pass no {name}"
                )
            for candidate in entries:
                if getattr(candidate, name) == tuple(value):
                    return candidate
            raise KeyError(
                f"no backend coverage recorded for {_pair_text(op_id, target)} with "
                f"{name}={tuple(value)!r} -- recorded: {recorded}"
            )
        if len(entries) > 1:
            if entries[0].operand_dtypes is not None:
                return _own_operand_dtypes_entry(entries)
            name = next(
                n for n in _VARIANT_FIELDS if getattr(entries[0], n) is not None
            )
            raise ValueError(
                f"{_pair_text(op_id, target)} is recorded per {name} "
                f"({[getattr(e, name) for e in entries]}) -- pass {name}="
            )
        return entries[0]

    def status(
        self,
        op_id: str,
        target: str,
        *,
        scale_format: Variant | None = None,
        operand_dtypes: Variant | None = None,
    ) -> str:
        """The measured class of (``op_id``, ``target``), one of ``COVERAGE_CLASSES``.

        Only ``issuable`` means rocke's backend issues the row's instruction on ``target``. The
        variant arguments follow :meth:`entry`.
        """
        return self.entry(
            op_id, target, scale_format=scale_format, operand_dtypes=operand_dtypes
        ).status

    def backend_op(
        self,
        op_id: str,
        target: str,
        *,
        scale_format: Variant | None = None,
        operand_dtypes: Variant | None = None,
    ) -> str:
        """The backend op id (``core.arch`` ``MmaOp.op_id``) matched to (``op_id``, ``target``).

        Raises ``KeyError`` when the pair matched no backend op. The variant arguments follow
        :meth:`entry`.
        """
        found = self.entry(
            op_id, target, scale_format=scale_format, operand_dtypes=operand_dtypes
        )
        if found.backend_op is None:
            raise KeyError(
                f"{_pair_text(op_id, target, found)} matched no backend op "
                f"(backend coverage: {found.status})"
            )
        return found.backend_op

    def backend_ops_without_row(self, target: str) -> tuple[str, ...]:
        """The backend ops on ``target`` that no catalog row was matched to."""
        return self.ops_without_row.get(target, ())


def _variant(record: Mapping[str, object], name: str, pair: str) -> Variant | None:
    value = record.get(name)
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(v, str) for v in value)
    ):
        raise ValueError(
            f"backend_coverage.json: {pair} {name} {value!r} is not an (A, B) pair of names"
        )
    return (value[0], value[1])


def _entry_from_record(
    op_id: str, target: str, record: Mapping[str, object]
) -> CoverageEntry:
    pair = _pair_text(op_id, target)
    unexpected = sorted(set(record) - _ENTRY_FIELDS)
    if unexpected:
        raise ValueError(
            f"backend_coverage.json: {pair} has unexpected fields {unexpected} -- an entry "
            f"carries {sorted(_ENTRY_FIELDS)}"
        )
    status = record.get("class")
    if status not in COVERAGE_CLASSES:
        raise ValueError(
            f"backend_coverage.json: unknown class {status!r} -- expected one of "
            f"{list(COVERAGE_CLASSES)}"
        )
    backend_op = record.get("backend_op")
    entry = CoverageEntry(
        status=status,
        backend_op=backend_op,
        scale_format=_variant(record, "scale_format", pair),
        operand_dtypes=_variant(record, "operand_dtypes", pair),
        reason=record.get("reason"),
    )
    if status != "no_backend_op" and not isinstance(backend_op, str):
        raise ValueError(
            f"backend_coverage.json: {_pair_text(op_id, target, entry)} is {status!r} but "
            f"records no backend op"
        )
    if status == "no_backend_op" and backend_op is not None:
        raise ValueError(
            f"backend_coverage.json: {_pair_text(op_id, target, entry)} is "
            f"'no_backend_op' but records backend op {backend_op!r}"
        )
    if entry.scale_format is not None and entry.operand_dtypes is not None:
        raise ValueError(
            f"backend_coverage.json: {pair} names both a scale_format and operand_dtypes "
            f"-- a pair has one kind of variant"
        )
    return entry


def _is_own_operand_dtypes(entry: CoverageEntry) -> bool:
    """True for the entry of a selectable integer row's own operand dtypes: both signed."""
    return all(dtype in _INTEGER_OPERAND_DTYPES for dtype in entry.operand_dtypes)


def _own_operand_dtypes_entry(entries: tuple[CoverageEntry, ...]) -> CoverageEntry:
    """The one entry of a pair recorded per operand_dtypes that answers for the row's own dtypes
    (``_check_variants`` guarantees there is exactly one)."""
    return next(e for e in entries if _is_own_operand_dtypes(e))


def _check_variants(
    op_id: str, target: str, entries: tuple[CoverageEntry, ...]
) -> None:
    """A pair has one entry without a variant, or one entry per variant of one kind; a pair
    recorded per operand_dtypes also holds exactly one entry for the row's own (signed) dtypes.
    """
    pair = _pair_text(op_id, target)
    if not entries:
        raise ValueError(f"backend_coverage.json: {pair} records no entry")
    if len(entries) == 1:
        return
    for name in _VARIANT_FIELDS:
        variants = [getattr(e, name) for e in entries]
        if all(v is not None for v in variants):
            if len(set(variants)) != len(variants):
                raise ValueError(
                    f"backend_coverage.json: {pair} records a {name} twice: {variants}"
                )
            if name == "operand_dtypes":
                own = [e.operand_dtypes for e in entries if _is_own_operand_dtypes(e)]
                if len(own) != 1:
                    raise ValueError(
                        f"backend_coverage.json: {pair} records operand_dtypes {variants} "
                        f"-- expected exactly one pair of signed dtypes "
                        f"{sorted(_INTEGER_OPERAND_DTYPES)} (the row's own), found {own}"
                    )
            return
    raise ValueError(
        f"backend_coverage.json: {pair} records {len(entries)} entries that do not each "
        f"name a variant of one kind (scale_format or operand_dtypes)"
    )


def load_backend_coverage(path: Path = DEFAULT_COVERAGE_PATH) -> BackendCoverage:
    """Load ``backend_coverage.json``; any data error fails the load."""
    if not path.is_file():
        raise ValueError(f"backend_coverage table not found -- path={path}")
    document = json.loads(path.read_text())
    flavor = document.get("_meta", {}).get("llvm_flavor")
    if flavor not in LLVM_FLAVORS:
        raise ValueError(
            f"backend_coverage.json: _meta.llvm_flavor {flavor!r} is not an LLVM flavor "
            f"-- expected one of {list(LLVM_FLAVORS)}"
        )
    entries: dict[tuple[str, str], tuple[CoverageEntry, ...]] = {}
    for op_id, by_target in document.get("pairs", {}).items():
        for target, records in by_target.items():
            pair_entries = tuple(
                _entry_from_record(op_id, target, record) for record in records
            )
            _check_variants(op_id, target, pair_entries)
            entries[(op_id, target)] = pair_entries
    ops_without_row = {
        target: tuple(op_ids)
        for target, op_ids in document.get("backend_ops_without_row", {}).items()
    }
    return BackendCoverage(
        llvm_flavor=flavor, entries=entries, ops_without_row=ops_without_row
    )


@functools.cache
def committed_backend_coverage() -> BackendCoverage:
    """The committed record (``DEFAULT_COVERAGE_PATH``), loaded and validated once per process.

    It is package data, fixed while rocke runs, and :meth:`TileMmaPlan.emit_op` reads it on every MMA
    a kernel issues. Treat the result as read-only: every caller shares it.
    :func:`load_backend_coverage` reads its path afresh on every call.
    """
    return load_backend_coverage()
