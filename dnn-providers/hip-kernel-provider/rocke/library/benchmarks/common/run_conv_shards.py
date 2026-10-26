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

The cost is that each shard re-reads the kernel cache: ~18 s for a 447k-entry
cache, so ~72 s per card over a run. Lower ``--shards`` to one per card to pay
it once, at the price of the tail.

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
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from benchmarks.common import case_results

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
) -> List[Tuple[int, str, int, float]]:
    """Run every shard, at most one per device at a time.

    A device that frees up takes the next shard off the queue, so a long shard
    delays only the card running it. ``command(index, paths)`` builds the
    child's argv; ``popen`` is the process factory (a stub, under test);
    ``on_done(index, device, rc, seconds)`` is called as each shard ends.

    Returns ``(shard index, device, rc, seconds)`` per shard, in the order
    they finished.
    """
    queue = list(plan)
    running: Dict[str, Tuple[int, Mapping[str, Path], object, object, float]] = {}
    free = list(devices)
    done: List[Tuple[int, str, int, float]] = []

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

    devices = resolve_devices(args.devices)
    shards = (
        args.shards
        if args.shards is not None
        else SHARDS_PER_DEVICE * len(devices)
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
        carried = previous_rows(log_dir)
        if not carried:
            print(f"[info] --resume: no rows in {log_dir}; running everything")
        else:
            measured = log_dir / "measured.jsonl"
            measured.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in carried),
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
    done = run_shards(
        plan,
        devices=devices,
        command=lambda index, paths: shard_command(script, forwarded, paths),
        poll_seconds=POLL_SECONDS,
        on_done=_report,
    )

    rc_all = 0
    for index, _device, rc, _seconds in done:
        if rc:
            rc_all = rc_all or rc
            print(f"\nshard {index} failed (rc={rc}); last lines of its log:")
            print(_tail(shard_paths(log_dir, index, width)["log"]))

    rows = case_results.merge(
        carried
        + [
            row
            for _index, paths in plan
            for row in case_results.read_rows(paths["rows"])
        ]
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
