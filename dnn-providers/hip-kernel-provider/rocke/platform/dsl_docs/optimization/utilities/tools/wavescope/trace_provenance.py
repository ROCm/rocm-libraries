# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Trace identity helpers shared by ATT capture and the sidecar producer.

The viewer binds ``inline_frames.json`` to a specific ``code.json`` and code
object by content hash and a per-capture trace id. These helpers keep the
Python scripts aligned on filenames, hash formatting, and the on-disk shape of
``wavescope-trace.json``.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

# The dispatch folders rocprofv3 writes, each a self-contained trace folder.
DISPATCH_GLOB = "ui_output_*_dispatch_*"

SIDECAR = "inline_frames.json"
SIDECAR_TMP_SUFFIX = ".tmp"
TRACE_SENTINEL = "wavescope-trace.json"
TRACE_SENTINEL_TMP_SUFFIX = ".tmp"

# Sidecar schema version written by emit_inline_frames.py.
SIDECAR_VERSION = 3

# Trace sentinel schema version. Matches the WaveScope-managed runner shape.
TRACE_SENTINEL_VERSION = 1

# code.json column indices for instructionListingHash (isa, codeobj, vaddr).
_LISTING_CODEOBJ_COL = 4
_LISTING_VADDR_COL = 5

CAPTURE_COMPLETE = "complete"
CAPTURE_TRUNCATED = "truncated"

# Files owned by sidecar generation. Capture sentinels are deliberately absent:
# only capture may change whether trace bytes are complete.
SIDECAR_FILES = (
    SIDECAR,
    f"{SIDECAR}{SIDECAR_TMP_SUFFIX}",
)


def new_trace_id() -> str:
    """Return a fresh capture-generation id."""
    return str(uuid.uuid4())


def sha256_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def instruction_listing_hash(rows: list) -> str | None:
    """Hash the semantic instruction listing, matching WaveScope ``traceShape.js``."""
    if not isinstance(rows, list) or not rows:
        return None
    lines: list[str] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            return None
        isa = row[0]
        codeobj = row[_LISTING_CODEOBJ_COL]
        vaddr = row[_LISTING_VADDR_COL]
        if (
            not isinstance(isa, str)
            or not isinstance(codeobj, int)
            or not isinstance(vaddr, int)
        ):
            return None
        lines.append(json.dumps([isa, codeobj, vaddr], separators=(",", ":")) + "\n")
    return sha256_text("".join(lines))


def instruction_listing_hash_file(path: Path) -> str | None:
    try:
        rows = json.loads(path.read_text())["code"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    return instruction_listing_hash(rows)


def dispatch_dirs(root: Path) -> list[Path]:
    if (root / "code.json").is_file():
        return [root]
    return sorted(root.glob(DISPATCH_GLOB))


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write ``payload`` beside ``path`` and rename over it."""
    tmp = path.with_name(path.name + TRACE_SENTINEL_TMP_SUFFIX)
    tmp.unlink(missing_ok=True)
    try:
        tmp.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True))
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_trace_sentinel(dispatch: Path) -> dict | None:
    path = dispatch / TRACE_SENTINEL
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def make_trace_sentinel(
    *,
    trace_id: str,
    instruction_listing_hash: str,
    capture: str,
    code_object_hash: str | None = None,
    kernel: str | None = None,
) -> dict:
    return {
        "version": TRACE_SENTINEL_VERSION,
        "traceId": trace_id,
        "instructionListingHash": instruction_listing_hash,
        "codeObjectHash": code_object_hash,
        "capture": capture,
        "kernel": kernel,
    }


def write_trace_sentinel(
    dispatch: Path,
    *,
    trace_id: str,
    instruction_listing_hash: str,
    capture: str,
    code_object_hash: str | None = None,
    kernel: str | None = None,
) -> dict:
    doc = make_trace_sentinel(
        trace_id=trace_id,
        instruction_listing_hash=instruction_listing_hash,
        capture=capture,
        code_object_hash=code_object_hash,
        kernel=kernel,
    )
    write_json_atomic(dispatch / TRACE_SENTINEL, doc)
    return doc


def enrich_trace_sentinel(dispatch: Path, **fields) -> dict | None:
    """Merge ``fields`` into an existing sentinel, or create one if absent."""
    existing = read_trace_sentinel(dispatch) or {}
    doc = {**existing, **fields}
    if "version" not in doc:
        doc["version"] = TRACE_SENTINEL_VERSION
    write_json_atomic(dispatch / TRACE_SENTINEL, doc)
    return doc


def capture_generation(root: Path, trace_id: str) -> Path:
    """Return the private output directory for one capture attempt."""
    if not trace_id or any(
        not (char.isascii() and (char.isalnum() or char in "._-")) for char in trace_id
    ):
        raise ValueError(f"invalid capture id: {trace_id!r}")
    return root / f"capture-{trace_id}"


def invalidate_sidecars(root: Path) -> int:
    """Remove sidecar-owned files without changing capture provenance."""
    dropped = 0
    for d in (root, *sorted(root.glob(DISPATCH_GLOB))):
        for name in SIDECAR_FILES:
            path = d / name
            if not path.exists():
                continue
            try:
                path.unlink()
            except OSError as exc:
                raise SystemExit(
                    f"cannot remove {path}: {exc}\n"
                    "  It is source attribution from an earlier sidecar run. "
                    "Remove it by hand before regenerating the sidecar."
                ) from exc
            print(f"  {d.name}: removed {path.name} from an earlier run")
            dropped += 1
    return dropped


def kernel_name_of(dispatch: Path) -> str | None:
    code_json = dispatch / "code.json"
    if not code_json.is_file():
        return None
    try:
        rows = json.loads(code_json.read_text())["code"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    for row in rows:
        if not row:
            continue
        isa = row[0] if row else ""
        if isinstance(isa, str) and isa.startswith(";"):
            name = isa[1:].strip()
            return name or None
    return None
