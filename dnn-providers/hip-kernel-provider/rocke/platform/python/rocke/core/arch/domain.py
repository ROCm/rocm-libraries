# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Read side of the intrinsic arch-domain artifact.

``tools/gen_arch_domain.py`` measures, per (declaration key, gfx target),
whether that declaration actually *links*, and commits the answer as one JSON
column per LLVM flavor under ``rocke/core/arch/data/``. This module is the only
thing that reads those columns.

Two axes are folded into one table::

    available(key, arch, flavor) ~= arch_domain(key, arch) and exists(key, flavor)

so a cell answers both "does this LLVM know the name" (``name_absent``) and
"can this target lower it" (``arch_absent``).

The important distinction for every consumer is *negative* versus *no data*.
Only ``arch_absent`` is a negative answer. ``target_unsupported``,
``toolchain_crash``, ``toolchain_timeout`` and a flavor with no committed column
all mean the question was never answered, and a consumer that treats them as a
"no" turns a developer on an older ROCm into a wall of warnings about intrinsics
that are perfectly fine. :func:`lookup` returns ``None`` for missing data rather
than a status, so silence is what a caller gets by default instead of something
it has to remember to branch on.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import NamedTuple

SCHEMA = "rocke.intrinsic_arch_domain/v1"

# The seven answers a probe can produce. Defined here rather than in the
# generator so the writer and the readers cannot drift: the generator imports
# these, it does not own them.
STATUS_OK = "ok"
STATUS_NAME_ABSENT = "name_absent"
STATUS_ARCH_ABSENT = "arch_absent"
STATUS_TARGET_UNSUPPORTED = "target_unsupported"
STATUS_TOOLCHAIN_CRASH = "toolchain_crash"
STATUS_TOOLCHAIN_TIMEOUT = "toolchain_timeout"
STATUS_PROBE_ERROR = "probe_error"

#: Statuses that mean "we did not get an answer". A consumer must stay silent on
#: every one of them. ``probe_error`` is here for completeness only -- it means
#: the generator is broken, and no committed column is allowed to contain one.
NO_DATA_STATUSES = frozenset(
    {
        STATUS_TARGET_UNSUPPORTED,
        STATUS_TOOLCHAIN_CRASH,
        STATUS_TOOLCHAIN_TIMEOUT,
        STATUS_PROBE_ERROR,
    }
)

# Same anchor as the arch catalog next door (``target.py``): relative to this
# file, never a repo path. ``pyproject.toml`` packages the columns via
# ``rocke = ["**/*.json"]``, so this resolves in an installed wheel too.
_DATA_DIR = Path(__file__).parent / "data"


class Cell(NamedTuple):
    """One measured (key, arch) answer.

    ``evidence`` is the first line of the toolchain diagnostic that produced a
    non-``ok`` status, and is empty for ``ok``. ``probe_imm`` is set only where
    the generator's ``immarg`` sweep had to move a constant operand off its
    default ``0`` to get the intrinsic to lower; its absence means the default
    answered.
    """

    status: str
    evidence: str
    verified_on: str
    probe_imm: int | None


class ArchDomain:
    """One committed column -- a single LLVM flavor, every key, every target."""

    def __init__(self, flavor: str, doc: dict) -> None:
        self.flavor = flavor
        self.schema: str = doc.get("schema", "")
        self.toolchain: dict[str, object] = doc.get("toolchain", {})
        self.canonical: dict[str, str] = doc.get("canonical", {})
        self._keys: dict[str, dict[str, dict]] = doc.get("keys", {})

    def lookup(self, key: str, arch: str) -> Cell | None:
        """The cell for ``(key, arch)``, or ``None`` when it was not measured.

        ``None`` covers two genuinely different situations that a caller should
        treat identically: the key is not in the declaration table this column
        was generated from (a dynamically registered declare, say -- see
        ``_op_vector_smax`` in the lowerer), or the target was not among the
        arches the generator swept. Neither is evidence of anything.
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

    Today that is ``arch_absent`` and nothing else. ``name_absent`` is also a
    definite negative -- the intrinsic does not exist in that LLVM at all -- but
    the emitted-IR validity gate already links what kernels emit and catches a
    reachable one there with a real diagnostic, so including it here would
    duplicate that coverage for no new signal. That is a recorded decision on
    AICK-2273, not an oversight; this function is the one place to revisit it.
    """
    return status == STATUS_ARCH_ABSENT


@functools.cache
def load(flavor: str) -> ArchDomain | None:
    """The committed column for ``flavor``, or ``None`` if there is not one.

    A host can only ever measure its own LLVM, so a flavor with no column is the
    normal case rather than an error -- it means nobody with that toolchain has
    blessed one yet. Callers must treat the ``None`` as no data.
    """
    path = _DATA_DIR / f"intrinsic_arch_domain.{flavor}.json"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    # A column that exists but does not parse is NOT no-data: it is a corrupt
    # committed artifact, and swallowing it here would silently switch the lane
    # off for everyone. Let it raise.
    return ArchDomain(flavor, json.loads(text))
