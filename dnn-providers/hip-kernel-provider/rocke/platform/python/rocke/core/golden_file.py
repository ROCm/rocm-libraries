# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The fingerprint every JSON golden records, and the command that re-records a golden.

A golden *family* is the unit one re-record writes: a shard file of a shard
directory, a whole single-family file, or one family's slice of a file shared by
several families. Its fingerprint is the sha256 of its entries in canonical JSON::

    sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")))

The record tool that writes a family writes its fingerprint beside it, one line
per family. Two changes that re-record the same family from different trees both
rewrite that line, so git reports a conflict where it would otherwise merge their
hash edits into a file that matches neither tree::

    base                 change A               change B
    "case_1": "aa..",    "case_1": "a2..",      "case_1": "aa..",
    "case_2": "bb..",    "case_2": "bb..",      "case_2": "b2..",
    "fingerprint": "f0"  "fingerprint": "fA"    "fingerprint": "fB"   <- conflict

Without the fingerprint line both edits merge cleanly and the merged golden fails
on the first tree that runs it. The fingerprint is never written by hand: a
merge resolution re-records the family on the merged tree.

A file that holds one family stores ``"fingerprint": "<hex>"`` at the top level
and its entries are the rest of the document (:func:`stamp_fingerprint`,
:func:`verify_fingerprint`). A file shared by several families stores one entry
per family; its format owner builds the entries and calls :func:`fingerprint`.

:func:`rerecord_command` is the one spelling of the command that re-records a
golden through ``rocke/tools/golden.py``; a golden test prints it on drift.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "FINGERPRINT_KEY",
    "fingerprint",
    "golden_json_text",
    "rerecord_command",
    "rerecord_hint",
    "stamp_fingerprint",
    "verify_fingerprint",
    "write_golden_json",
]

#: Top-level key a golden document records its fingerprint under.
FINGERPRINT_KEY = "fingerprint"

# The re-record front door, run from the rocke directory.
_RERECORD_TOOL = "tools/golden.py"


def fingerprint(entries: object) -> str:
    """Return the lowercase hex fingerprint of one family's ``entries``."""
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _entries(document: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a one-family document its fingerprint covers."""
    if not isinstance(document, Mapping):
        raise TypeError(
            f"golden document must be a JSON object - got {type(document).__name__}"
        )
    return {key: value for key, value in document.items() if key != FINGERPRINT_KEY}


def stamp_fingerprint(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``document`` with its fingerprint recorded under
    :data:`FINGERPRINT_KEY` (as the last key; any old fingerprint is replaced)."""
    entries = _entries(document)
    return {**entries, FINGERPRINT_KEY: fingerprint(entries)}


def verify_fingerprint(document: Mapping[str, Any]) -> tuple[bool, str]:
    """Check the fingerprint a one-family document records.

    Returns ``(True, "")`` when it matches the document's entries, else
    ``(False, reason)``: the fingerprint is missing, or stale (the entries were
    edited or merged without re-recording).
    """
    recorded = document.get(FINGERPRINT_KEY) if isinstance(document, Mapping) else None
    if recorded is None:
        return False, f"no {FINGERPRINT_KEY!r} recorded"
    computed = fingerprint(_entries(document))
    if recorded != computed:
        return False, (
            f"stale {FINGERPRINT_KEY!r} - recorded={recorded!r}, expected "
            f"{computed!r} (the entries changed without a re-record)"
        )
    return True, ""


def golden_json_text(document: Mapping[str, Any]) -> str:
    """The text of a one-family golden: ``document`` stamped with its
    fingerprint, as indented JSON with sorted keys and a final newline."""
    return json.dumps(stamp_fingerprint(document), indent=2, sort_keys=True) + "\n"


def write_golden_json(path: Path, document: Mapping[str, Any]) -> None:
    """Write ``document`` to ``path`` as :func:`golden_json_text`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(golden_json_text(document), encoding="utf-8")


def rerecord_command(golden: str, families: Iterable[str] | None = None) -> str:
    """The command, run from the rocke directory, that re-records ``golden``.

    ``golden`` is a name from the golden index (``python tools/golden.py
    list``). ``families`` narrows a multi-family golden to those families;
    without it every family is re-recorded.
    """
    if not golden or not golden.strip():
        raise ValueError(f"rerecord_command - golden={golden!r}, expected a name")
    command = f"python {_RERECORD_TOOL} rerecord {golden}"
    chosen = sorted(set(families or ()))
    if chosen:
        command += " --only " + ",".join(chosen)
    return command


def rerecord_hint(golden: str, families: Iterable[str] | None = None) -> str:
    """The lines a failing golden check prints: review, then this command."""
    return (
        "If the change is intended, review it and re-record from the rocke "
        f"directory:\n    {rerecord_command(golden, families)}"
    )
