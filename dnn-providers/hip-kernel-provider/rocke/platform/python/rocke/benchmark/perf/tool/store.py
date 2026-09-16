# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Local record store - append-only JSON Lines in a user cache dir.

The one place records are persisted. Deliberately NOT a database: an append-only
`history.jsonl` under `~/.cache/rocke-perf` (one record per line), so a developer can
keep a run history and compare across runs without any server. The
data lives in a user cache dir *outside the repo* - only this code is committed,
never the results.

Resolution order for the cache dir: explicit arg > `$ROCKE_PERF_CACHE` >
`$XDG_CACHE_HOME/rocke-perf` > `~/.cache/rocke-perf`. Records are grouped by
`schema.identity` (arch, op, kernel_name, shape) for comparison. Stdlib only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from rocke.benchmark.perf import schema as _schema

_ENV = "ROCKE_PERF_CACHE"
_APP = "rocke-perf"
_HISTORY = "history.jsonl"


def cache_dir(explicit: Optional[os.PathLike | str] = None) -> Path:
    """Resolve (and create) the cache dir. See module docstring for precedence."""
    if explicit is not None:
        base = Path(explicit)
    elif os.environ.get(_ENV):
        base = Path(os.environ[_ENV])
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = (Path(xdg) if xdg else Path.home() / ".cache") / _APP
    base.mkdir(parents=True, exist_ok=True)
    return base


def history_path(cache: Optional[os.PathLike | str] = None) -> Path:
    """Path to the append-only history file inside the cache dir."""
    return cache_dir(cache) / _HISTORY


def append(
    record: Mapping[str, Any], *, cache: Optional[os.PathLike | str] = None
) -> Path:
    """Validate and append one record as a JSON line. Returns the history path."""
    _schema.validate(record)
    p = history_path(cache)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return p


def _is_readable_record(record: object) -> bool:
    """A record safe for `identity()`/comparison: a dict with the identity fields.

    Deliberately NOT a full `schema.validate` - the schema is additive-only so a
    record written under a newer version must still load. We only require the
    load-bearing structure downstream code dereferences (run.arch, kernel.*), so a
    truncated write or a JSON scalar/list is skipped instead of crashing `compare`.
    """
    if not isinstance(record, dict):
        return False
    run = record.get("run")
    kernel = record.get("kernel")
    if not isinstance(run, Mapping) or not isinstance(kernel, Mapping):
        return False
    for section in ("counters", "wall", "spread"):
        if section in record and record[section] is not None:
            if not isinstance(record[section], Mapping):
                return False
    return (
        "arch" in run
        and "op" in kernel
        and "kernel_name" in kernel
        and isinstance(kernel.get("shape"), Mapping)
    )


def load(*, cache: Optional[os.PathLike | str] = None) -> list[dict]:
    """Read all records back, in append order. Blank/corrupt lines are skipped.

    Skips lines that aren't a JSON object with the identity structure; it does NOT
    drop a well-formed record merely because its `schema` version differs (the
    schema is additive-only, so older/newer records stay readable).
    """
    p = history_path(cache)
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not _is_readable_record(record):
            continue
        out.append(record)
    return out


def group_by_identity(records: Sequence[Mapping[str, Any]]) -> dict[tuple, list[dict]]:
    """Bucket records by `(arch, op, kernel_name, shape)`, in append order."""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault(_schema.identity(r), []).append(dict(r))
    return groups


def records_for(records: Sequence[Mapping[str, Any]], identity: tuple) -> list[dict]:
    """Filter records to a single identity tuple, preserving append order."""
    return [dict(r) for r in records if _schema.identity(r) == identity]
