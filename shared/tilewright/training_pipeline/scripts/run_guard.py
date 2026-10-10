#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Show and settle what a run records about its GPUs and hung lines
(lib/gpu_health.py: `<run dir>/guard/guard.json`).

    run_guard.py status  <run dir>
    run_guard.py accept  <run dir> [--gpu PCI_ADDRESS ...]
    run_guard.py release <run dir> (--shape M,N,K,B ... | --all)

status   the run's hold, whom it holds (every GPU of the run, or single
         GPUs) and why, per GPU (and the host), each GPU's baseline next to
         its readings now, its canary reference and last pass, and the
         quarantined lines
accept   take the GPUs' readings now as the run's baseline (every GPU the
         record has, or those of --gpu), drop their canary references so
         that each one's next canary becomes its reference, and drop their
         part of the hold (all of it without --gpu): for a change made on
         purpose that the run goes on under. Stage 3 benches on such a GPU
         only while its settings agree with the stage's other GPUs. Power
         management off and stuck ttm workers hold the run again at its
         next check: no GPU benches without power management, and the ttm
         workers are the host's
release  let the quarantined lines of these shapes (or all) be benched
         again in the run, in every protocol they were quarantined for

The run directory may also be the guard directory itself. The record is
changed under the lock every process of the run takes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from lib import gpu_health  # noqa: E402
from lib import kfd  # noqa: E402


def _guard(path: Path) -> gpu_health.Guard:
    d = Path(path)
    if not (d / gpu_health.GUARD_FILE).is_file():
        d = d / gpu_health.GUARD_DIR
    if not (d / gpu_health.GUARD_FILE).is_file():
        raise SystemExit(f"no {gpu_health.GUARD_FILE} in {path} or its guard/")
    return gpu_health.Guard(d, "run_guard", by="scripts/run_guard.py")


def _by_key() -> dict:
    return {gpu_health.gpu_key(g): g for g in kfd.gpus()}


def status(guard: gpu_health.Guard) -> int:
    doc = guard.read()
    hold = doc.get("hold")
    if hold:
        scope = (
            "every GPU of the run"
            if gpu_health.run_wide(hold)
            else ", ".join(gpu_health.held_keys(hold)) + " alone"
        )
        print(f"HELD since {hold.get('raised')} by {hold.get('by')}, {scope}:")
        for key, entry in sorted(gpu_health.hold_entries(hold).items()):
            for p in entry.get("problems") or []:
                print(
                    f"  {key}: {p} (since {entry.get('raised')} by {entry.get('by')})"
                )
    else:
        print(f"not held (last cleared {doc.get('last_cleared') or 'never'})")
    gpus = _by_key()
    can = doc.get("canary") or {}
    refs = can.get("references") or {}
    passed = can.get("passed") or {}
    for key, row in sorted(((doc.get("baseline") or {}).get("gpus") or {}).items()):
        g = gpus.get(key)
        now = gpu_health.read_gpu(g) if g is not None else None
        print(f"GPU {row.get('gpu')} ({key}), baseline of {row.get('recorded')}:")
        for check in gpu_health.CHECKS:
            then = row.get("readings", {}).get(check)
            shown = "not found now" if now is None else now.get(check)
            print(f"  {check:18s} {then!r:>12} -> {shown!r}")
        ref = refs.get(key)
        last = passed.get(key)
        print(
            f"  canary reference {ref.get('taken') if ref else 'none'}; last pass "
            f"{last.get('at') if last else 'none'}"
            + (f" (session {last.get('session')})" if last else "")
        )
    setup = can.get("setup")
    if setup:
        print(
            f"canary: {len(setup.get('lines') or [])} line(s) x "
            f"{setup.get('timings')}, tolerance {setup.get('tolerance')}, recorded "
            f"{setup.get('created')}"
        )
    lines = doc.get("quarantined_lines") or []
    print(f"{len(lines)} quarantined line(s)")
    for e in lines:
        print(
            f"  {'x'.join(str(v) for v in e.get('shape') or [])}: hung in "
            f"{e.get('in_flight')} on GPU {e.get('gpu')} at {e.get('at')} "
            f"({e.get('stage')}), quarantined for "
            f"{e.get('scope') or gpu_health.BENCH_SCOPE}"
        )
    let_in = gpu_health.released_lines(doc)
    if let_in:
        print(f"{len(let_in)} line(s) released (benched again)")
    return 0


def accept(guard: gpu_health.Guard, only: Sequence[str]) -> int:
    gpus = _by_key()
    wanted = {str(k).lower() for k in only}
    changed: List[str] = []
    accepted: List[str] = []
    off: List[str] = []
    with guard.update() as doc:
        base = doc.setdefault("baseline", {"recorded": gpu_health.utc(), "gpus": {}})
        can = doc.setdefault("canary", {})
        for key, row in sorted(base["gpus"].items()):
            if wanted and key.lower() not in wanted:
                continue
            g = gpus.get(key)
            if g is None:
                print(f"{key}: not found on this host; left as it is")
                continue
            now = gpu_health.read_gpu(g)
            for _, what in gpu_health.changes(row.get("readings") or {}, now):
                changed.append(f"{key}: {what}")
            if now.get("power_management") is False:
                off.append(key)
            row["previous"] = row.get("readings")
            row["readings"] = now
            row["accepted"] = gpu_health.utc()
            for part in ("references", "passed"):
                (can.get(part) or {}).pop(key, None)
            accepted.append(key)
        gpu_health.note_accepted(doc, accepted)
    named = [k for k in gpu_health.hold_entries(guard.hold()) if k.lower() in wanted]
    what, hold = guard.release(accepted + named if wanted else None, by="accept")
    for line in changed or ["no reading differs from the baseline"]:
        print(f"accepted: {line}")
    print("each GPU's next canary becomes its reference")
    for key in off:
        print(
            f"{key}: power management is off; the run's next check holds it again "
            f"until it is on"
        )
    if what == "cleared":
        print("the run's hold is lifted")
    elif hold is not None:
        print("the run stays held for: " + "; ".join(hold.get("problems") or []))
    return 0


def release(guard: gpu_health.Guard, shapes: Sequence[str], every: bool) -> int:
    want = []
    for text in shapes:
        try:
            want.append(tuple(int(v) for v in text.split(",")))
        except ValueError:
            raise SystemExit(f"--shape {text!r}: give M,N,K,B")
    gone, kept = guard.release_lines(None if every else want)
    print(f"released {len(gone)} line(s); {kept} stay quarantined")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("status")
    st.add_argument("run_dir", type=Path)
    st.add_argument("--json", action="store_true", help="print the record as is")
    ac = sub.add_parser("accept")
    ac.add_argument("run_dir", type=Path)
    ac.add_argument("--gpu", action="append", default=[], metavar="PCI_ADDRESS")
    rl = sub.add_parser("release")
    rl.add_argument("run_dir", type=Path)
    rl.add_argument("--shape", action="append", default=[], metavar="M,N,K,B")
    rl.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    guard = _guard(a.run_dir)
    if a.cmd == "status":
        if a.json:
            print(json.dumps(guard.read(), indent=2))
            return 0
        return status(guard)
    if a.cmd == "accept":
        return accept(guard, a.gpu)
    if not a.shape and not a.all:
        ap.error("release needs --shape or --all")
    return release(guard, a.shape, a.all)


if __name__ == "__main__":
    sys.exit(main())
