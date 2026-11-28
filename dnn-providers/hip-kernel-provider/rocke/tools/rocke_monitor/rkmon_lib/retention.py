# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The state folder's logs stay bounded (config `retention`).

Each full tick (`rotate`) moves an append-only log past `retention.log_max_mb` to `<name>.1`, replacing the one before,
so each takes at most twice that. The readers read `<name>.1` then `<name>` (Store.read_jsonl, Store.journal_lines,
usage's reader), so a rotation loses nothing a window of recent days needs. The service's own log (web.log), which
`rk serve` holds open for appending, is copied and then truncated instead of renamed.

The unsent items (state_store.UNSENT_FILE) are one of the logs: an item is kept until its rotated copy is replaced,
then dropped, and none is ever sent. Their file moves under the lock a keep takes (state_store.locked_in): a page's keep
writes the file whole, and would write the moved lines back.

The ledger, actions.jsonl, is never rotated: every outward write checks it whole (one post per head, the rate limits,
a ticket filed once), and it grows by one short line per outward action.

Job folders: cleanup.sweep removes a finished job's folder (it has an rc) after `retention.job_days`; one with no rc
goes too once nothing names it (jobs.referenced) and its leader is gone (a job killed or lost), after the same days.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import shutil

from . import registry, state_store
from .state_store import UNSENT_FILE

DEFAULTS = registry.package_section("retention")
LOGS = (
    "journal.md",
    "events.jsonl",
    "metrics.jsonl",
    "ci_runs.jsonl",
    "conflicts.jsonl",
    "scaler.jsonl",
    "triage.jsonl",
    "claude_usage.jsonl",
    "daylog.jsonl",
    "scoreboard_events.jsonl",
    UNSENT_FILE,
    "web.log",
)
HELD_OPEN = (
    "web.log",
)  # copied, then truncated: its writer keeps appending to the same file


def settings(cfg: dict | None) -> dict:
    return dict(DEFAULTS, **((cfg or {}).get("retention") or {}))


def older(path: pathlib.Path) -> pathlib.Path:
    """The rotated copy of a log: `<name>.1`."""
    return path.with_name(path.name + ".1")


def with_older(path: pathlib.Path) -> list[pathlib.Path]:
    """The files that hold a log, oldest first: `<name>.1` (when there is one), then `<name>`."""
    return [p for p in (older(path), path) if p.exists()]


def rotate(cfg: dict, store) -> list[str]:
    """Rotate each log past `log_max_mb`; the lines to print (one per log rotated). Never fails a tick: a log that
    cannot be moved is said and left."""
    limit = float(settings(cfg)["log_max_mb"]) * 2**20
    out = []
    for name in LOGS:
        p = pathlib.Path(store.dir) / name
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size <= limit:
            continue
        try:
            with (
                state_store.locked_in(store.dir)
                if name == UNSENT_FILE
                else contextlib.nullcontext()
            ):
                if name in HELD_OPEN:
                    shutil.copyfile(p, older(p))
                    with open(p, "r+b") as f:
                        f.truncate(0)
                else:
                    os.replace(p, older(p))
        except OSError as exc:
            out.append(f"(retention: {name} not rotated: {exc})")
            continue
        out.append(f"RETENTION {name}: {size / 2**20:.1f} MB moved to {older(p).name}")
        store.journal(
            "retention", f"{name}: {size / 2**20:.1f} MB moved to {older(p).name}"
        )
    return out
