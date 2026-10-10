# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The bundles a training run deploys, in the order it deploys them: each
round's (`round_<k>/stage06`) and, with full.final_retrain, its final
retrain's (`final_retrain/stage06`), which trains every cell on the data of
the run's complete rounds once they are all done and records them, with
their completion markers, in `final_retrain/final_retrain.json`.

The final retrain is a bundle of the run (`final_retrain`) while its
decision accepted it, its stage06 completed, and every round it trained on
is complete with the completion time recorded then: a round that runs again
writes another. It deployed after those rounds and before any later one,
such as an enhance round (`bundle_key`), which builds on it: an active round
starts from the latest of the bundles before it (run_orchestrator.py
`Pipeline.base_bundle`). A final retrain trained while the one in
`final_retrain/` is a bundle of the run trains beside it, in
`final_retrain_next/`, and takes its place only once its gate accepted it
and its stage06 deployed it; until then it is no bundle of the run.

A bundle's stage05 trains some cells and carries the others from the bundle
before it (its last --prior-round-dir). `stage05/cell_origins.json` records
which bundle trained each carried cell, and what it then recorded of it
(`cell_origins`), so that the bundle still tells it once that bundle has
left the run or trained again.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

FINAL_RETRAIN_DIR = "final_retrain"
FINAL_RETRAIN_NEXT_DIR = "final_retrain_next"
CELL_ORIGINS = "cell_origins.json"
CARRIED_FORWARD = "carried_forward.json"
RECORD = "final_retrain.json"
DECISION = "decision.json"
FINAL_AUDIT = "audit.json"
AUDIT_RESULTS_DIR = "audit_results"
ROUND_MARKER = ".round_complete.json"
STAGE_MARKER = ".stage_complete.json"
DEPLOY_RECORD = "deploy_record.json"
RESULT_FILE = "audit.json"

_ROUND = re.compile(r"round_(\d+)")


def _read(path: Path) -> Optional[Any]:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def round_index(path: Path) -> Optional[int]:
    m = _ROUND.fullmatch(Path(path).name)
    return int(m.group(1)) if m else None


def round_stamps(rounds: Iterable[Path]) -> Dict[str, List[Any]]:
    """{round: [its completion time, its completion marker's mtime in ns]}
    of `rounds`, [None, None] for a round without a marker."""
    out: Dict[str, List[Any]] = {}
    for rd in rounds:
        marker = Path(rd) / ROUND_MARKER
        try:
            mtime: Optional[int] = marker.stat().st_mtime_ns
        except OSError:
            mtime = None
        rec = _read(marker)
        out[Path(rd).name] = [
            rec.get("finished") if isinstance(rec, dict) else None,
            mtime,
        ]
    return out


def trained_rounds(final_dir: Path) -> Dict[str, Any]:
    """{round: its stamp then (`round_stamps`)} of the rounds the final
    retrain in `final_dir` trained on; empty without a record."""
    rec = _read(Path(final_dir) / RECORD)
    rounds = rec.get("rounds") if isinstance(rec, dict) else None
    return dict(rounds) if isinstance(rounds, dict) else {}


def changed_rounds(run_dir: Path, recorded: Dict[str, Any]) -> List[str]:
    """The rounds of `recorded` (`trained_rounds`) that are not complete in
    `run_dir` with the completion time they had then. Only the time is
    compared, so that a copy of the run directory keeps its final retrain."""
    now = round_stamps(Path(run_dir) / name for name in recorded)
    out = []
    for name, stamp in recorded.items():
        then = stamp[0] if isinstance(stamp, list) and stamp else None
        if now[name][0] is None or now[name][0] != then:
            out.append(name)
    return out


def final_retrain(run_dir: Path) -> Optional[Path]:
    """final_retrain/ when it is a bundle of the run (see the module
    docstring), else None."""
    d = Path(run_dir) / FINAL_RETRAIN_DIR
    dec = _read(d / DECISION)
    if not (isinstance(dec, dict) and dec.get("accepted")):
        return None
    s06 = d / "stage06"
    if not ((s06 / STAGE_MARKER).is_file() and (s06 / DEPLOY_RECORD).is_file()):
        return None
    recorded = trained_rounds(d)
    return d if recorded and not changed_rounds(run_dir, recorded) else None


def bundle_key(path: Path) -> Tuple[int, int]:
    """The place of a bundle directory in the order the run deploys:
    round k at (k, 0), the final retrain at (the last round it trained on,
    1)."""
    i = round_index(path)
    if i is not None:
        return i, 0
    last = [round_index(Path(n)) for n in trained_rounds(path)]
    return max((k for k in last if k is not None), default=-1), 1


def bundles(run_dir: Path, rounds: Iterable[Path]) -> List[Path]:
    """`rounds` and the final retrain when it is a bundle of the run, in the
    order the run deploys them."""
    out = list(rounds)
    final = final_retrain(run_dir)
    if final is not None:
        out.append(final)
    return sorted(out, key=bundle_key)


def trained_entries(bundle: Path) -> List[Dict[str, Any]]:
    """`stage05/cells.json` entries of the cells the bundle's stage05
    trained; none when it carried the bundle before it forward."""
    s5 = Path(bundle) / "stage05"
    if (s5 / CARRIED_FORWARD).is_file():
        return []
    cj = _read(s5 / "cells.json")
    cells = cj.get("cells") if isinstance(cj, dict) else None
    return [c for c in cells or [] if isinstance(c, dict) and c.get("label")]


def stage05_finished(bundle: Path) -> Optional[str]:
    """When the bundle's stage05 completed (its marker's `finished`): which
    training it holds, as a round that runs again writes another."""
    rec = _read(Path(bundle) / "stage05" / STAGE_MARKER)
    return rec.get("finished") if isinstance(rec, dict) else None


def cell_origins(bundles: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    """{cell: where its model was trained} over `bundles` (bundle
    directories in the order the run deploys them), from the latest: the
    cells a bundle's stage05 trained (`trained_entries`), then those it
    carried as its `stage05/cell_origins.json` records them. Each names the
    bundle (`trained_in`) and the stage05 completion (`finished`) that
    trained it; `bundle` is that directory while it is one of `bundles`
    with that same training, whose own records then tell its cells.json
    `entry` and options (`model_options` None), else None, with the entry
    and the model-shaping options (None when unknown) the record kept. A
    bundle without that record (from before it) leaves the cells it did not
    train to the bundles before it."""
    bundles = [Path(b) for b in bundles]
    by_name = {b.name: b for b in bundles}
    entries: Dict[Path, Dict[str, Dict[str, Any]]] = {}
    stamps: Dict[Path, Optional[str]] = {}

    def trained(b: Path) -> Dict[str, Dict[str, Any]]:
        if b not in entries:
            entries[b] = {str(c["label"]): c for c in trained_entries(b)}
        return entries[b]

    def finished(b: Path) -> Optional[str]:
        if b not in stamps:
            stamps[b] = stage05_finished(b)
        return stamps[b]

    out: Dict[str, Dict[str, Any]] = {}
    for b in reversed(bundles):
        for label, c in trained(b).items():
            if label not in out:
                out[label] = {
                    "trained_in": b.name,
                    "bundle": b,
                    "finished": finished(b),
                    "entry": c,
                    "model_options": None,
                }
        rec = _read(b / "stage05" / CELL_ORIGINS)
        if not isinstance(rec, dict):
            continue
        origins = rec.get("origins") if isinstance(rec.get("origins"), dict) else {}
        cells = rec.get("cells") if isinstance(rec.get("cells"), dict) else {}
        for label, o in cells.items():
            origin = origins.get(o.get("origin")) if isinstance(o, dict) else None
            if str(label) in out or not isinstance(origin, dict):
                continue
            name, stamp = str(origin.get("bundle")), origin.get("finished")
            live = by_name.get(name)
            if live is not None and stamp and finished(live) == stamp:
                out[str(label)] = {
                    "trained_in": name,
                    "bundle": live,
                    "finished": stamp,
                    "entry": trained(live).get(str(label), o.get("entry") or {}),
                    "model_options": None,
                }
            else:
                out[str(label)] = {
                    "trained_in": name,
                    "bundle": None,
                    "finished": stamp,
                    "entry": o.get("entry") or {},
                    "model_options": origin.get("model_options"),
                }
    return out


def audit_result_files(run_dir: Path) -> List[Path]:
    """The audit records a run writes besides its rounds' stages: the
    rounds scored again at the end of the run (audit_results/<round>.json)
    and the final retrain's (its stage06 audit.json and its gate's
    audit.json)."""
    run_dir = Path(run_dir)
    final = run_dir / FINAL_RETRAIN_DIR
    files = sorted((run_dir / AUDIT_RESULTS_DIR).glob("*.json"))
    files += [
        p for p in (final / "stage06" / RESULT_FILE, final / FINAL_AUDIT) if p.is_file()
    ]
    return files
