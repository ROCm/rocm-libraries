# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Read side of the intrinsic arch-domain artifact.

``tools/gen_arch_domain.py`` measures, per (declaration key, gfx target),
whether that declaration links, and commits the answer as one JSON column per
LLVM flavor under ``rocke/core/arch/data/``. This module is the only reader.

A cell answers two questions at once: "does this LLVM know the name"
(``name_absent``) and "can this target lower it" (``arch_absent``).

**Negative is not the same as no data.** Only ``arch_absent`` is a negative
answer; the no-data statuses and a flavor with no committed column all mean the
question was never asked. :func:`lookup` returns ``None`` for missing data, so
silence is the default a caller gets rather than something to branch on.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import NamedTuple

SCHEMA = "rocke.intrinsic_arch_domain/v1"

# The seven answers a probe can produce. Owned here, not by the generator, so
# the writer and the readers cannot drift.
STATUS_OK = "ok"
STATUS_NAME_ABSENT = "name_absent"
STATUS_ARCH_ABSENT = "arch_absent"
STATUS_TARGET_UNSUPPORTED = "target_unsupported"
STATUS_TOOLCHAIN_CRASH = "toolchain_crash"
STATUS_TOOLCHAIN_TIMEOUT = "toolchain_timeout"
STATUS_PROBE_ERROR = "probe_error"

#: Statuses that mean "we did not get an answer"; consumers must stay silent on
#: all of them. ``probe_error`` means the generator is broken -- no committed
#: column may contain one.
NO_DATA_STATUSES = frozenset(
    {
        STATUS_TARGET_UNSUPPORTED,
        STATUS_TOOLCHAIN_CRASH,
        STATUS_TOOLCHAIN_TIMEOUT,
        STATUS_PROBE_ERROR,
    }
)

# Anchored on __file__ like ``target.py`` next door, so it resolves in an
# installed wheel (``pyproject.toml`` packages the columns via ``**/*.json``).
_DATA_DIR = Path(__file__).parent / "data"


class Cell(NamedTuple):
    """One measured (key, arch) answer.

    ``evidence`` is the first line of the toolchain diagnostic, empty for
    ``ok``. ``probe_imm`` is set only where the generator's ``immarg`` sweep had
    to move a constant operand off its default ``0`` to make the intrinsic
    lower.
    """

    status: str
    evidence: str
    verified_on: str
    probe_imm: int | None


class ArchDomain:
    """One committed column -- a single LLVM flavor, every key, every target."""

    def __init__(self, flavor: str, doc: dict) -> None:
        # Check the envelope rather than defaulting it. A document that merely
        # parses (`{}`, a v2 column) would make every lookup return None, and
        # None means "not measured" here -- so a corrupt artifact would switch
        # the warning lane off in silence, the one failure the lane prevents.
        #
        # Per-cell vocabulary is deliberately not checked: `is_negative`
        # compares for equality, so an unknown status is already inert, and
        # rejecting one would break a future column the schema string versions.
        # ValueError throughout, not TypeError: the artifact is wrong, not the
        # caller.
        if not isinstance(doc, dict):
            raise ValueError(  # noqa: TRY004
                f"arch-domain column for {flavor} is a {type(doc).__name__}, "
                "not a JSON object"
            )
        schema = doc.get("schema", "")
        if schema != SCHEMA:
            raise ValueError(
                f"arch-domain column for {flavor} declares schema "
                f"{schema!r}, expected {SCHEMA!r}. The artifact is corrupt or "
                "was written by a different rocke; regenerate it with "
                "tools/gen_arch_domain.py."
            )
        keys = doc.get("keys")
        if not isinstance(keys, dict):
            raise ValueError(  # noqa: TRY004 -- see above
                f"arch-domain column for {flavor} has no `keys` mapping "
                f"(found {type(keys).__name__}); nothing was measured."
            )
        self.flavor = flavor
        self.schema: str = schema
        self.toolchain: dict[str, object] = doc.get("toolchain", {})
        self.canonical: dict[str, str] = doc.get("canonical", {})
        self._keys: dict[str, dict[str, dict]] = keys

    def lookup(self, key: str, arch: str) -> Cell | None:
        """The cell for ``(key, arch)``, or ``None`` when it was not measured.

        ``None`` means either the key is not in the declaration table this
        column was generated from (a dynamically registered declare -- see
        ``_op_vector_smax`` in the lowerer), or the generator never swept that
        arch. Neither is evidence of anything.
        """
        row = self._keys.get(key)
        if row is None:
            return None
        cell = row.get(arch)
        if cell is None:
            return None
        imm = cell.get("probe_imm")
        return Cell(
            status=str(cell.get("status", "")),
            evidence=str(cell.get("evidence", "")),
            verified_on=str(cell.get("verified_on", self.flavor)),
            probe_imm=int(imm) if imm is not None else None,
        )

    def keys(self) -> tuple:
        """Every declaration key this column covers, in file order."""
        return tuple(self._keys)


def is_negative(status: str) -> bool:
    """Whether a status is a definite "this will not work".

    ``arch_absent`` only. ``name_absent`` is also definite, but it is a
    column-wide fact, so ``tools/check_ir_validity.py`` already catches any
    reachable one with the compiler's own diagnostic. Excluded deliberately,
    not by oversight; this function is the one place to revisit it.
    """
    return status == STATUS_ARCH_ABSENT


@functools.cache
def load(flavor: str) -> ArchDomain | None:
    """The committed column for ``flavor``, or ``None`` if there is not one.

    A host can only measure its own LLVM, so a missing column is normal, not an
    error. Callers must treat the ``None`` as no data. Only absence is: a column
    that exists but cannot be read (permissions, I/O) raises, because returning
    ``None`` for it would switch the whole lane off without a word.
    """
    path = _DATA_DIR / f"intrinsic_arch_domain.{flavor}.json"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    # A column that exists but does not parse is a corrupt artifact, not
    # no-data. Let it raise.
    return ArchDomain(flavor, json.loads(text))
