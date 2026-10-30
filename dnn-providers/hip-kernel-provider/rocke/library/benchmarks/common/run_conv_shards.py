# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run one MIOpenDriver case file across several GPUs, one process per card.

A cache sweep is per-case independent work, so a file of a few thousand cases
is embarrassingly parallel across cards -- and in the AOT path there is nothing
to build first, so a shard starts launching as soon as it has read the cache.
This runner splits the file, pins one process per card with
``HIP_VISIBLE_DEVICES``, and merges the shards' per-case rows at the end.

Usage (from ``library/``)::

    python benchmarks/common/run_conv_shards.py \\
        --miopen-file ../cases.txt --devices 8 --log-dir /scratch/sweep -- \\
        --run-from-cache /scratch/cache --arch gfx942 --fast-rank

Everything after ``--`` is forwarded verbatim to the benchmark script
(``--script``, default ``compare``); the runner adds ``--miopen-file`` and
``--results`` per shard.

More shards than cards, on purpose
----------------------------------
Case cost spans orders of magnitude -- a 1x1 convolution on a 200-pixel row
against a 7x7 on 224x224 -- so one static shard per card leaves cards idle for
however long the unluckiest shard runs over. ``--shards`` defaults to four per
card and a card takes the next shard off the queue when it frees up, which
bounds the tail at roughly one shard rather than one quarter of the run.

Each shard re-reads the kernel cache, so more shards add startup overhead.
Lower ``--shards`` to one per card to pay it once, at the price of a longer
tail.

Lines are dealt round-robin (line *i* to shard *i mod shards*) rather than in
blocks: a case file is usually grouped by model and resolution, so contiguous
blocks would put all the expensive shapes in one shard.

What a shard leaves behind
--------------------------
In ``--log-dir``: ``shard-NN.cases`` (its exact input, re-runnable by hand),
``shard-NN.log`` (its whole output), and ``shard-NN.jsonl`` (its per-case
rows). The runner then writes ``results.jsonl`` and ``results.csv`` over all
shards. A shard that dies keeps the rows it had written -- they are flushed
per case -- so a re-run of the remaining cases merges cleanly with them.

Progress
--------
A shard reports nothing until it exits, and one shard is minutes to hours, so
the runner counts the case keys in the shards' row files every
``PROGRESS_SECONDS`` and prints a line across the whole run::

    [progress] 142/4218 cases (3%) | 7/22 shards | 8/8 busy | 12 min elapsed | ETA 2.1 h

For one shard's detail, follow its ``shard-NN.log``: the sweeps print a banner
per case into it.

Devices (``--devices``): a count ``N`` (devices ``0..N-1``), a list ``2,3``,
or ``auto`` (default): the entries of ``HIP_VISIBLE_DEVICES``, else the count
of ``ROCR_VISIBLE_DEVICES`` (``0..k-1``), else one device. This duplicates the
rule in ``library/tests/run_device_shards.py`` rather than importing it:
``tests/`` sits above ``benchmarks/`` in the layering
``library/tests/test_library_layering.py`` enforces, so the import would be a
back-edge. The runner itself imports neither torch nor the HIP runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from benchmarks.common import case_results
from benchmarks.common.batched_timing import DEFAULT_TIMING, TIMINGS

_SCRIPT_DIR = Path(__file__).parent

SCRIPTS = {
    "compare": "benchmark_conv_compare.py",
    "direct": "benchmark_direct_conv.py",
    "implicit": "benchmark_implicit_gemm_conv.py",
}

#: Shards per card when ``--shards`` is not given. See the module docstring:
#: enough to keep the tail short, few enough that re-reading the cache per
#: shard stays a rounding error against hours of sweeping.
SHARDS_PER_DEVICE = 4

#: How often the scheduler looks for a finished shard. A shard runs for
#: minutes; a second of latency on starting the next one is not worth a busier
#: loop.
POLL_SECONDS = 1.0

#: How often the aggregate progress line is printed. Without it the console
#: says nothing between one shard finishing and the next -- a quarter of an
#: hour of silence on a long sweep, which reads the same as a hang.
PROGRESS_SECONDS = 30.0


# --------------------------------------------------------------------------- devices
def resolve_devices(spec: str, env: Optional[Mapping[str, str]] = None) -> List[str]:
    """``--devices`` to the device ids to pin shards to."""
    env = os.environ if env is None else env
    spec = spec.strip()
    if spec != "auto":
        if "," in spec:
            return [d.strip() for d in spec.split(",") if d.strip()]
        n = int(spec)
        if n < 1:
            raise ValueError("--devices needs at least one device")
        return [str(i) for i in range(n)]
    hip = env.get("HIP_VISIBLE_DEVICES", "").strip()
    if hip:
        return [d.strip() for d in hip.split(",") if d.strip()]
    rocr = env.get("ROCR_VISIBLE_DEVICES", "").strip()
    if rocr:
        return [str(i) for i in range(len([d for d in rocr.split(",") if d]))]
    return ["0"]


# --------------------------------------------------------------------------- cases
def read_cases(path) -> List[str]:
    """The case lines of a MIOpenDriver file: blanks and ``#`` comments out.

    The benchmark scripts skip those themselves, but dropping them here keeps
    a shard's line count equal to its case count, so the split is even in
    cases rather than in file lines.
    """
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def split_cases(cases: Sequence[str], shards: int) -> List[List[str]]:
    """Deal ``cases`` round-robin into ``shards`` lists (see the module doc).

    Empty shards are dropped: asking for more shards than cases is not an
    error, it just gives fewer shards -- a process with no case to run would
    read the cache for nothing.
    """
    if shards < 1:
        raise ValueError("--shards needs at least one shard")
    dealt: List[List[str]] = [[] for _ in range(shards)]
    for i, case in enumerate(cases):
        dealt[i % shards].append(case)
    return [shard for shard in dealt if shard]


# --------------------------------------------------------------------------- progress
def measured_keys(paths: Iterable[Path]) -> Set[str]:
    """The case keys recorded across *paths*, as a set.

    By key rather than by row: the comparison writes one row per engine, so a
    case both sweeps measured is one case done, not two. The rows are flushed
    per case, so reading the shard files mid-run is how the progress line sees
    inside a shard without the children reporting anything back.

    This is a floor on the cases attempted, not a count of them: a case no
    cached kernel supports finishes without writing a row. A sweep with many
    of those shows a percentage that lags the real position and so an ETA on
    the long side.
    """
    keys: Set[str] = set()
    for path in paths:
        for row in case_results.read_rows(path):
            keys.add(row["key"])
    return keys


def _duration(seconds: float) -> str:
    """Minutes up to an hour and a half, hours past it."""
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def progress_line(
    *,
    measured: int,
    total: int,
    baseline: int,
    shards_done: int,
    shards: int,
    busy: int,
    devices: int,
    elapsed: float,
) -> str:
    """One line of aggregate progress over every shard.

    ``baseline`` is what a ``--resume`` carried in. Those cases cost this run
    no time, so the rate the ETA extrapolates from counts only the cases this
    run measured itself -- otherwise resuming a nearly-finished sweep would
    report minutes left before it had timed a single case.
    """
    pct = 100.0 * measured / total if total else 0.0
    parts = [
        f"{measured}/{total} cases ({pct:.0f}%)",
        f"{shards_done}/{shards} shards",
        f"{busy}/{devices} busy",
        f"{_duration(elapsed)} elapsed",
    ]
    fresh = measured - baseline
    remaining = total - measured
    if fresh > 0 and remaining > 0:
        parts.append(f"ETA {_duration(elapsed / fresh * remaining)}")
    return "[progress] " + " | ".join(parts)


# --------------------------------------------------------------------------- runs
def previous_rows(log_dir: Path) -> List[dict]:
    """Every per-case row a previous run left in ``log_dir``.

    The shard files are overwritten by the new run, so this is read first.
    ``measured.jsonl`` -- what a previous ``--resume`` itself carried -- is
    read too, so resuming a resumed run does not lose the first attempt's
    work.
    """
    out: List[dict] = []
    for path in sorted(log_dir.glob("shard-*.jsonl")) + [log_dir / "measured.jsonl"]:
        out.extend(case_results.read_rows(path))
    return out


def forwarded_timing(forwarded: Sequence[str]) -> str:
    """The finalist timing the forwarded arguments ask the sweeps for.

    The runner forwards the benchmark's flags verbatim, but it merges the rows
    and carries a previous run's into ``--resume``, and rows of two timings are
    not comparable (:mod:`benchmarks.common.case_results`). So it reads this
    one flag back: the last ``--timing`` given, as argparse would take it, else
    the sweeps' default.
    """
    timing = DEFAULT_TIMING
    for i, arg in enumerate(forwarded):
        if arg == "--timing" and i + 1 < len(forwarded):
            timing = forwarded[i + 1]
        elif arg.startswith("--timing="):
            timing = arg.split("=", 1)[1]
    return timing


def shard_paths(log_dir: Path, index: int, width: int) -> Dict[str, Path]:
    stem = f"shard-{index:0{width}d}"
    return {
        "cases": log_dir / f"{stem}.cases",
        "log": log_dir / f"{stem}.log",
        "rows": log_dir / f"{stem}.jsonl",
    }


def shard_command(
    script: Path, forwarded: Sequence[str], paths: Mapping[str, Path]
) -> List[str]:
    """The child's argv: the user's arguments plus this shard's two files."""
    return [
        sys.executable,
        "-u",  # a piped stdout is block-buffered otherwise; the log lags hours
        str(script),
        *forwarded,
        "--miopen-file",
        str(paths["cases"]),
        "--results",
        str(paths["rows"]),
    ]


def _tail(path: Path, n: int = 30) -> str:
    try:
        return "\n".join(
            path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
        )
    except OSError:
        return ""


def run_shards(
    plan: Sequence[Tuple[int, Mapping[str, Path]]],
    *,
    devices: Sequence[str],
    command,
    popen=subprocess.Popen,
    poll_seconds: float = POLL_SECONDS,
    on_done=None,
    on_progress=None,
    progress_seconds: float = PROGRESS_SECONDS,
) -> List[Tuple[int, str, int, float]]:
    """Run every shard, at most one per device at a time.

    A device that frees up takes the next shard off the queue, so a long shard
    delays only the card running it. ``command(index, paths)`` builds the
    child's argv; ``popen`` is the process factory (a stub, under test);
    ``on_done(index, device, rc, seconds)`` is called as each shard ends and
    ``on_progress(shards_done, busy)`` every ``progress_seconds`` while any
    shard is still running.

    Returns ``(shard index, device, rc, seconds)`` per shard, in the order
    they finished.
    """
    queue = list(plan)
    running: Dict[str, Tuple[int, Mapping[str, Path], object, object, float]] = {}
    free = list(devices)
    done: List[Tuple[int, str, int, float]] = []
    last_progress = time.monotonic()

    while queue or running:
        while queue and free:
            index, paths = queue.pop(0)
            device = free.pop(0)
            env = dict(os.environ)
            env["HIP_VISIBLE_DEVICES"] = device
            fh = open(paths["log"], "w", encoding="utf-8")
            proc = popen(
                command(index, paths),
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
            )
            running[device] = (index, paths, proc, fh, time.monotonic())
        if not running:
            break
        time.sleep(poll_seconds)
        for device, (index, paths, proc, fh, started) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            del running[device]
            free.append(device)
            entry = (index, device, int(rc), time.monotonic() - started)
            done.append(entry)
            if on_done is not None:
                on_done(*entry)
        now = time.monotonic()
        if on_progress is not None and now - last_progress >= progress_seconds:
            last_progress = now
            on_progress(len(done), len(running))
    return done


# --------------------------------------------------------------------------- cli
def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Everything after -- is forwarded to the benchmark script.",
    )
    ap.add_argument(
        "--miopen-file",
        required=True,
        metavar="FILE",
        dest="miopen_file",
        help="MIOpenDriver case file to split across the devices",
    )
    ap.add_argument(
        "--devices",
        default="auto",
        help="device count, comma-separated ids, or auto (default: auto)",
    )
    ap.add_argument(
        "--shards",
        type=int,
        default=None,
        metavar="N",
        help=f"case shards to deal out (default: {SHARDS_PER_DEVICE} per "
        f"device, so a slow shard holds up one card rather than a quarter of "
        f"the run; 1 per device re-reads the cache once instead of "
        f"{SHARDS_PER_DEVICE} times)",
    )
    ap.add_argument(
        "--script",
        choices=sorted(SCRIPTS),
        default="compare",
        help="benchmark to run in each shard (default: compare, both engines)",
    )
    ap.add_argument(
        "--log-dir",
        default=None,
        dest="log_dir",
        metavar="DIR",
        help="where the shards' case files, logs and rows go "
        "(default: a new directory under the system temp dir)",
    )
    ap.add_argument(
        "--results",
        default=None,
        metavar="FILE",
        help="merged per-case table (default: results.jsonl in --log-dir); "
        "a .csv of the same rows is written beside it",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="re-run into an existing --log-dir, skipping the cases its rows "
        "already cover. A sweep that died after four hours keeps the four "
        "hours: rows are flushed per case. Each engine skips only what it "
        "recorded itself, so a case measured by one sweep and not the other "
        "finishes.",
    )
    ap.add_argument("forwarded", nargs=argparse.REMAINDER)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_parser().parse_args(argv)
    forwarded = list(args.forwarded)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    for flag in ("--miopen-file", "--results", "--skip-measured"):
        if flag in forwarded:
            print(
                f"error: {flag} is the runner's to set per shard; drop it from "
                f"the forwarded arguments",
                file=sys.stderr,
            )
            return 2
    if "--resume" in forwarded:
        # Easy to put on the wrong side of the separator, and the child's
        # "unrecognized arguments" would send the reader looking in the wrong
        # script.
        print(
            "error: --resume is the runner's own flag; it goes before --",
            file=sys.stderr,
        )
        return 2
    if not any(a.startswith("--run-from-cache") for a in forwarded):
        # Sharding is for the AOT path: a JIT sweep would build the same
        # kernels again on every card, and the per-case rows this runner
        # merges are written only by a cache run. Catching it here beats
        # starting a process per card to watch them all exit 2.
        print(
            "error: sharding runs an AOT sweep; forward --run-from-cache DIR "
            "after --",
            file=sys.stderr,
        )
        return 2

    timing = forwarded_timing(forwarded)
    if timing not in TIMINGS:
        print(
            f"error: --timing {timing!r}: expected one of {TIMINGS}",
            file=sys.stderr,
        )
        return 2

    devices = resolve_devices(args.devices)
    shards = (
        args.shards if args.shards is not None else SHARDS_PER_DEVICE * len(devices)
    )
    cases = read_cases(args.miopen_file)
    if not cases:
        print(f"error: {args.miopen_file}: no cases", file=sys.stderr)
        return 2
    dealt = split_cases(cases, shards)

    import tempfile

    if args.resume and not args.log_dir:
        # Resuming into a fresh temporary directory would find no rows and
        # silently re-run everything.
        print(
            "error: --resume needs the --log-dir of the run it continues",
            file=sys.stderr,
        )
        return 2
    log_dir = Path(args.log_dir or tempfile.mkdtemp(prefix="rocke-conv-shards-"))
    log_dir.mkdir(parents=True, exist_ok=True)

    # What the previous attempt measured, gathered before its shard files are
    # overwritten. The sweeps skip by case key, which only they can compute
    # (it comes off a parsed problem), so the runner hands them the rows and
    # does no matching of its own.
    carried: List[dict] = []
    if args.resume:
        # Every row is carried, whatever its timing, so a resume measured the
        # other way loses nothing; the sweeps skip only the cases their own
        # timing has (case_results.measured_keys), and the merge takes only
        # those rows.
        carried = previous_rows(log_dir)
        if not carried:
            print(f"[info] --resume: no rows in {log_dir}; running everything")
        else:
            measured = log_dir / "measured.jsonl"
            # --script compare's shard rows are already merged (case/winner per
            # row, no "engine"), but --skip-measured is read by the per-engine
            # sweeps underneath it -- expand before writing, or every row is
            # silently not-measured to them and --resume re-runs everything.
            skip_rows = (
                case_results.expand_merged_for_skip(carried)
                if args.script == "compare"
                else carried
            )
            measured.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in skip_rows),
                encoding="utf-8",
            )
            forwarded = list(forwarded) + ["--skip-measured", str(measured)]
            print(f"[resume] {len(carried)} row(s) carried from {log_dir}")
    width = len(str(len(dealt) - 1)) or 1
    plan = []
    for index, shard in enumerate(dealt):
        paths = shard_paths(log_dir, index, width)
        paths["cases"].write_text("\n".join(shard) + "\n", encoding="utf-8")
        plan.append((index, paths))

    script = _SCRIPT_DIR / SCRIPTS[args.script]
    print(
        f"{len(cases)} case(s) over {len(dealt)} shard(s) on "
        f"{len(devices)} device(s): {args.script}",
        flush=True,
    )
    print(f"logs: {log_dir}", flush=True)

    def _report(index: int, device: str, rc: int, seconds: float) -> None:
        paths = shard_paths(log_dir, index, width)
        rows = len(case_results.read_rows(paths["rows"]))
        print(
            f"shard {index} device={device} rc={rc} "
            f"{rows} row(s) {seconds / 60:.1f} min",
            flush=True,
        )

    started = time.monotonic()

    # The shards' own row files, re-read on each progress tick. A shard reports
    # nothing until it exits, so its rows are the only live view into it.
    row_files = [paths["rows"] for _index, paths in plan]
    baseline = {row["key"] for row in carried if case_results.row_timing(row) == timing}

    def _progress(shards_done: int, busy: int) -> None:
        print(
            progress_line(
                measured=len(baseline | measured_keys(row_files)),
                total=len(cases),
                baseline=len(baseline),
                shards_done=shards_done,
                shards=len(dealt),
                busy=busy,
                devices=len(devices),
                elapsed=time.monotonic() - started,
            ),
            flush=True,
        )

    done = run_shards(
        plan,
        devices=devices,
        command=lambda index, paths: shard_command(script, forwarded, paths),
        poll_seconds=POLL_SECONDS,
        on_done=_report,
        on_progress=_progress,
        progress_seconds=PROGRESS_SECONDS,
    )

    rc_all = 0
    for index, _device, rc, _seconds in done:
        if rc:
            rc_all = rc_all or rc
            print(f"\nshard {index} failed (rc={rc}); last lines of its log:")
            print(_tail(shard_paths(log_dir, index, width)["log"]))

    shard_rows = [
        row for _index, paths in plan for row in case_results.read_rows(paths["rows"])
    ]
    if args.script == "compare":
        # Each shard ran benchmark_conv_compare.py, whose own --results is
        # already a merged row (both engines, winner, MIOpenDriver command):
        # case_results.merge() expects the raw per-engine rows the "direct"
        # and "implicit" scripts write, so handing it these would see no
        # "engine" field on any row and silently drop them all. A fresh row
        # replaces a carried one of the same key -- resume only reaches this
        # case again because --skip-measured did not cover it.
        by_key = {row["key"]: row for row in carried}
        by_key.update({row["key"]: row for row in shard_rows})
        rows = [by_key[key] for key in sorted(by_key)]
    else:
        rows = case_results.merge(
            carried + shard_rows,
            # The shards' rows are keyed by geometry; the case file is what
            # puts each row's MIOpenDriver line back on it.
            case_results.miopen_cmds_by_key(miopen_file=args.miopen_file),
            timing=timing,
        )
    if rows:
        out = Path(args.results or (log_dir / "results.jsonl"))
        csv_path = case_results.write(rows, out)
        print(f"\n{case_results.summary(rows)}")
        print(f"results: {out} (and {csv_path})", flush=True)
    else:
        print("\n[info] no per-case rows were written by any shard")
        rc_all = rc_all or 1
    print(f"wall: {(time.monotonic() - started) / 60:.1f} min", flush=True)
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
