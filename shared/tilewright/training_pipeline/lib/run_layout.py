# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Which directory a stage of a run or audit directory writes.

The benchmark planner is stage 2: `stage02/` and `stage02.log` in each round
of a training run, and at the top of an audit directory. Run and audit
directories written before the planner was renamed keep it in `stage01b/`,
and their rounds may hold the outputs of the removed probe in `stage02/`.

A directory without LAYOUT_FILE is read as its contents show (`of`): one
written before the rename keeps `stage01b/` for every reader and every
resume, the rounds a resume adds included, gets no file and has nothing
renamed or moved. The orchestrator records the current layout in
LAYOUT_FILE before the first stage it runs in any other run or audit
directory (`settle`).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .fs import write_json

LAYOUT_FILE = "run_layout.json"
PLAN_STAGE = "stage02"
LEGACY_PLAN_STAGE = "stage01b"
# Must equal lib.planner.PLAN_FILE.
PLAN_FILE = "plan.json"
_ROUND = re.compile(r"round_\d+")


@dataclass(frozen=True)
class Layout:
    """The directory names of a run or audit directory's stages."""

    version: int
    planner: str

    @property
    def legacy(self) -> bool:
        return self.planner == LEGACY_PLAN_STAGE

    def dir_name(self, key: str) -> str:
        """The directory, log and timing name of the stage `key` (a key of
        run_orchestrator.STAGE_DEFS)."""
        return self.planner if key == PLAN_STAGE else key

    def record(self) -> Dict[str, Any]:
        return {"layout": self.version, "planner": self.planner}


CURRENT = Layout(2, PLAN_STAGE)
LEGACY = Layout(1, LEGACY_PLAN_STAGE)


def read(root: Path) -> Optional[Layout]:
    """The layout LAYOUT_FILE in `root` records; None without the file.
    ValueError when the file cannot be read or records a layout this
    pipeline does not know."""
    path = Path(root) / LAYOUT_FILE
    try:
        rec = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot read {path}: {e}") from None
    for layout in (CURRENT, LEGACY):
        if rec == layout.record():
            return layout
    raise ValueError(f"{path} records {rec!r}, a layout this pipeline does not know")


def _dirs(root: Path) -> Iterator[Path]:
    """`root` and its round directories."""
    yield root
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return
    for p in entries:
        if _ROUND.fullmatch(p.name) and p.is_dir():
            yield p


def of(root: Path) -> Layout:
    """The layout of the run or audit directory `root`: the one LAYOUT_FILE
    records; without the file LEGACY when `root` or one of its rounds holds a
    `stage01b/` directory, or a `stage02/` directory without a plan (the
    removed probe's), else CURRENT. ValueError as `read`."""
    root = Path(root)
    layout = read(root)
    if layout is not None:
        return layout
    for d in _dirs(root):
        probe = d / PLAN_STAGE
        if (d / LEGACY_PLAN_STAGE).exists() or (
            probe.is_dir() and not (probe / PLAN_FILE).is_file()
        ):
            return LEGACY
    return CURRENT


def settle(root: Path) -> Layout:
    """`of(root)` for an invocation about to run stages in `root`, which
    records CURRENT in LAYOUT_FILE when the file is missing, so that a
    stage02 directory the invocation leaves without a plan is never taken
    for the probe's; a directory of the old layout is left as it is."""
    layout = of(root)
    if layout == CURRENT and read(root) is None:
        write_json(Path(root) / LAYOUT_FILE, layout.record())
    return layout


def root_of(owner: Path) -> Path:
    """The directory whose layout `owner` follows: a round directory's run
    directory, else `owner` itself (an audit directory)."""
    owner = Path(owner)
    return owner.parent if _ROUND.fullmatch(owner.name) else owner


def plan_dir(owner: Path) -> Path:
    """The planner's directory of a round directory or an audit directory.
    ValueError as `read`."""
    return Path(owner) / of(root_of(owner)).planner
