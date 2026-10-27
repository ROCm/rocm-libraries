# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Read side of the intrinsic arch-domain artifact.

``tools/gen_arch_domain.py`` measures, per (declaration key, gfx target),
whether that declaration links, and commits the answer as one JSON column per
LLVM flavor under ``rocke/core/arch/data/``. This module is the only reader.
The C++ engine reads no files when it lowers, so the build generates its copy
of the table through this module (``tools/gen_arch_domain_cpp.py``) -- which
is why it imports nothing outside the standard library.

A cell answers two questions at once: "does this LLVM know the name"
(``name_absent``) and "can this target lower it" (``arch_absent``).

**Negative is not the same as no data.** Only ``arch_absent`` is a negative
answer; the no-data statuses and a flavor with no committed column all mean the
question was never asked. :func:`lookup` returns ``None`` for missing data, so
silence is the default a caller gets rather than something to branch on.

The lane's policy lives here too, because both engines follow it: the modes
(:func:`mode`), the one message both engines print (:func:`message`), and the
committed waivers (:func:`waivers`).
"""

from __future__ import annotations

import functools
import json
import os
import re
from pathlib import Path
from typing import NamedTuple

SCHEMA = "rocke.intrinsic_arch_domain/v1"
WAIVERS_SCHEMA = "rocke.arch_domain_waivers/v1"

# What the lane does with a measured negative. `error` refuses the lowering
# unless the cell is waived, `warn` reports it and keeps going, `off` skips the
# lookup. The C++ engine reads the same variable by the same rules, and its
# DEFAULT_MODE is generated from this one, so changing the default here moves
# both engines.
MODE_ENV = "ROCKE_ARCH_DOMAIN"
MODE_ERROR = "error"
MODE_WARN = "warn"
MODE_OFF = "off"
MODES = (MODE_ERROR, MODE_WARN, MODE_OFF)
DEFAULT_MODE = MODE_WARN

# The C++ engine reports a refusal through its fixed error buffer
# (ROCKE_ERR_MSG_CAP: 256 bytes, NUL included). Both engines cut the message
# there, so what they say is byte-identical.
MESSAGE_CAP = 255

# ASCII whitespace only, as the C++ engine strips it.
_WHITESPACE = " \t\n\r\f\v"

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

    def arches(self, key: str) -> tuple:
        """Every target this column measured ``key`` on, in file order; empty
        for a key the column does not cover."""
        return tuple(self._keys.get(key, ()))


def is_negative(status: str) -> bool:
    """Whether a status is a definite "this will not work".

    ``arch_absent`` only. ``name_absent`` is also definite, but it is a
    column-wide fact, so ``tools/check_ir_validity.py`` already catches any
    reachable one with the compiler's own diagnostic. Excluded deliberately,
    not by oversight; this function is the one place to revisit it.
    """
    return status == STATUS_ARCH_ABSENT


_COLUMN_PREFIX = "intrinsic_arch_domain."
_COLUMN_SUFFIX = ".json"


def committed_flavors() -> tuple[str, ...]:
    """Every flavor that has a committed column, sorted.

    The C++ engine's copy of the table is generated from exactly this set
    (``tools/gen_arch_domain_cpp.py``), so a column added here reaches both
    engines with no list to update.
    """
    return tuple(
        sorted(
            p.name[len(_COLUMN_PREFIX) : -len(_COLUMN_SUFFIX)]
            for p in _DATA_DIR.glob(f"{_COLUMN_PREFIX}*{_COLUMN_SUFFIX}")
        )
    )


@functools.cache
def load(flavor: str) -> ArchDomain | None:
    """The committed column for ``flavor``, or ``None`` if there is not one.

    A host can only measure its own LLVM, so a missing column is normal, not an
    error. Callers must treat the ``None`` as no data. Only absence is: a column
    that exists but cannot be read (permissions, I/O) raises, because returning
    ``None`` for it would switch the whole lane off without a word.
    """
    path = _DATA_DIR / f"{_COLUMN_PREFIX}{flavor}{_COLUMN_SUFFIX}"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    # A column that exists but does not parse is a corrupt artifact, not
    # no-data. Let it raise.
    return ArchDomain(flavor, json.loads(text))


def mode(raw: str | None = None) -> str:
    """The lane's mode: ``raw`` if given, else ``$ROCKE_ARCH_DOMAIN``.

    Unset and empty mean :data:`DEFAULT_MODE`; case and surrounding whitespace
    are ignored. Anything else raises rather than falling back: every
    misspelling (``0``, ``false``, ``warning``) is an attempt to change what the
    lane does, and the default would quietly do something else. The C++ engine
    parses the variable by these rules and fails with this message.
    """
    value = (os.environ.get(MODE_ENV, "") if raw is None else raw).strip(_WHITESPACE)
    if not value:
        return DEFAULT_MODE
    if value.lower() not in MODES:
        raise ValueError(
            f"{MODE_ENV}='{value}' is not a mode; expected one of "
            f"{', '.join(MODES)} (unset means {DEFAULT_MODE})."
        )
    return value.lower()


def message(key: str, arch: str, flavor: str, status: str, evidence: str) -> str:
    """The sentence both engines use for a measured negative, warned or refused.

    Starts with ``arch-domain:`` so it leads to the docs from either engine,
    and is cut at :data:`MESSAGE_CAP` so the two engines print the same bytes.
    """
    detail = f": {evidence}" if evidence else ""
    text = (
        f"arch-domain: intrinsic '{key}' is not available on {arch} at {flavor} "
        f"({status}){detail}"
    )
    return text[:MESSAGE_CAP]


_MESSAGE_RE = re.compile(
    r"arch-domain: intrinsic '(?P<key>[^']+)' is not available on "
    r"(?P<arch>\S+) at (?P<flavor>\S+) \((?P<status>[a-z_]+)\)"
)


def parse_message(text: str) -> tuple[str, str, str, str] | None:
    """``(key, arch, flavor, status)`` read back from a :func:`message`.

    A refusal crosses from the C++ engine as its message alone; this recovers
    the cell. Evidence is not returned: the cap may have cut it, and the table
    has it whole. ``None`` for anything :func:`message` did not write.
    """
    m = _MESSAGE_RE.match(text)
    return (m["key"], m["arch"], m["flavor"], m["status"]) if m else None


class Waiver(NamedTuple):
    """One committed exception: report this cell, never refuse it."""

    key: str
    arch: str
    flavor: str
    reason: str


_WAIVERS_PATH = _DATA_DIR / "arch_domain_waivers.json"
_WAIVER_FIELDS = frozenset(Waiver._fields)


def parse_waivers(doc: object) -> dict[tuple[str, str, str], Waiver]:
    """Validate a waiver document and key it by ``(key, arch, flavor)``.

    The escape hatch for a wrong row: a measured negative that is not true, so
    refusing it would block every kernel that needs the intrinsic until the
    column is fixed. It downgrades a refusal to a warning for exactly one cell.
    Strict, because a waiver that does not parse as meant either waives
    nothing (and the build stays blocked) or waives the wrong cell.
    ValueError throughout: the file is wrong, not the caller.
    """
    if not isinstance(doc, dict) or doc.get("schema") != WAIVERS_SCHEMA:
        raise ValueError(
            f"{_WAIVERS_PATH.name} must be a JSON object with schema "
            f"{WAIVERS_SCHEMA!r}"
        )
    entries = doc.get("waivers")
    if not isinstance(entries, list):
        raise ValueError(  # noqa: TRY004 -- the file is wrong, not the caller
            f"{_WAIVERS_PATH.name}: `waivers` must be a list"
        )
    out: dict[tuple[str, str, str], Waiver] = {}
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != _WAIVER_FIELDS:
            raise ValueError(
                f"{_WAIVERS_PATH.name}: waiver {i} must have exactly the fields "
                f"{sorted(_WAIVER_FIELDS)}"
            )
        if not all(isinstance(v, str) and v.strip() for v in entry.values()):
            raise ValueError(
                f"{_WAIVERS_PATH.name}: waiver {i} has an empty or non-string field"
            )
        waiver = Waiver(**entry)
        cell = (waiver.key, waiver.arch, waiver.flavor)
        if cell in out:
            raise ValueError(f"{_WAIVERS_PATH.name}: {cell} is waived twice")
        out[cell] = waiver
    return out


@functools.cache
def waivers() -> dict[tuple[str, str, str], Waiver]:
    """The committed waivers. A missing file means none -- every negative is
    refused -- which is the safe way for that to fail. Only absence is none: a
    file that exists but cannot be read raises, because reading it as empty would
    quietly start refusing cells the repository waived, and bake no waivers into
    the C++ table."""
    try:
        text = _WAIVERS_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    return parse_waivers(json.loads(text))


def is_waived(key: str, arch: str, flavor: str) -> bool:
    """Whether ``(key, arch, flavor)`` has a committed waiver."""
    return (key, arch, flavor) in waivers()
