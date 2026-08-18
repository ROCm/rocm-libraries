# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Capture a decoded ATT trace for a rocke kernel, ready to open in WaveScope.

Wraps the ``rocprofv3 --att`` invocation documented in
``skills/capture-kernel-trace-rocke.md`` so that a capture is one command
instead of a flag soup plus a hunt through the output directory. It

- preflights ``rocprofv3`` and the ``rocprof-trace-decoder`` library, failing
  with the PMC fallback pointer rather than an opaque decoder error;
- discovers the kernel name with a ``--stats`` pass when no regex is given;
- drops any inline-frame sidecar left in the output directory *before* starting
  rocprofv3, so a reused directory cannot keep the previous build's source
  attribution beside a new trace;
- runs the ATT capture over the command you pass after ``--``;
- reports every decoded ``ui_output_*_dispatch_*`` folder with the numbers that
  say whether the trace is usable at all.

The decoded folder is what the WaveScope viewer reads. Open it with the
**WaveScope: Open Trace Folder...** command, or from another extension via
``wavescope.openTraceDir``.

Usage:
    python capture_att_trace.py -- python3 -m rocke.run_manifest k.hsaco m.json
    python capture_att_trace.py --kernel-regex 'ugemm_gfx950' -- python3 bench.py
    python capture_att_trace.py --output-dir ./att_out -- python3 bench.py

Note on ``code.json``: columns ``Latency`` and ``Stall`` are hit-weighted
totals over every execution, not per-execution averages. Divide by ``Hit``
for a per-execution figure.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_WAVESCOPE = _HERE.parent / "wavescope"
_spec = importlib.util.spec_from_file_location(
    "trace_provenance", _WAVESCOPE / "trace_provenance.py"
)
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)

CAPTURE_COMPLETE = _tp.CAPTURE_COMPLETE
CAPTURE_TRUNCATED = _tp.CAPTURE_TRUNCATED
DISPATCH_GLOB = _tp.DISPATCH_GLOB
dispatch_dirs = _tp.dispatch_dirs
invalidate_provenance = _tp.invalidate_provenance
kernel_name_of = _tp.kernel_name_of
new_trace_id = _tp.new_trace_id
sha256_file = _tp.sha256_file
write_trace_sentinel = _tp.write_trace_sentinel

# rocprofv3 writes one of these per traced dispatch; each is a complete,
# self-contained trace folder.
DISPATCH_GLOB = "ui_output_*_dispatch_*"

# Files the decoder emits that a viewer never parses. Reported separately so a
# folder's real size is not confused with what a viewer has to load.
BULK_FILES = ("wstates", "realtime")

DECODER_SONAME = "librocprof-trace-decoder.so"


def _snapshot_code_hashes(out: Path) -> dict[str, str]:
    """Map dispatch folder name to the hash of its ``code.json`` before capture."""
    snap: dict[str, str] = {}
    for d in dispatch_dirs(out):
        code_json = d / "code.json"
        if code_json.is_file():
            snap[d.name] = sha256_file(code_json)
    return snap


def _stamp_dispatches(
    out: Path,
    before: dict[str, str],
    *,
    capture: str,
) -> None:
    """Write or refresh ``wavescope-trace.json`` on dispatches touched this run."""
    for d in dispatch_dirs(out):
        code_json = d / "code.json"
        if not code_json.is_file():
            continue
        digest = sha256_file(code_json)
        if d.name in before and before[d.name] == digest:
            continue
        write_trace_sentinel(
            d,
            trace_id=new_trace_id(),
            code_json_hash=digest,
            capture=capture,
            kernel=kernel_name_of(d),
        )
        print(f"[identity] stamped {d.name} as {capture}")


def _decoder_dir() -> Path | None:
    """Locate the directory holding the ATT decoder, or None if absent.

    ``ROCPROF_TRACE_DECODER_LIB`` wins, matching rocprofv3's own override; it
    may name either the library or the directory containing it.
    """
    override = os.environ.get("ROCPROF_TRACE_DECODER_LIB")
    if override:
        p = Path(override)
        if p.is_file():
            return p.parent
        if (p / DECODER_SONAME).is_file():
            return p
    roots = [os.environ.get("ROCM_PATH"), os.environ.get("ROCM_HOME"), "/opt/rocm"]
    for root in roots:
        if not root:
            continue
        cand = Path(root) / "lib"
        if (cand / DECODER_SONAME).is_file():
            return cand
    return None


def _preflight() -> Path:
    """Fail early and actionably rather than inside rocprofv3."""
    if shutil.which("rocprofv3") is None:
        raise SystemExit(
            "rocprofv3 is not on PATH. ATT capture needs a working ROCm install."
        )
    lib = _decoder_dir()
    if lib is None:
        raise SystemExit(
            f"{DECODER_SONAME} not found. ATT capture cannot decode without it.\n"
            "  Install it from https://github.com/ROCm/rocprof-trace-decoder\n"
            "  (pick the build matching this distro), or set "
            "ROCPROF_TRACE_DECODER_LIB to its path.\n"
            "  Without the decoder, use the PMC profiling path in "
            "skills/capture-kernel-trace-rocke.md instead."
        )
    return lib


def discover_kernels(command: list[str], top: int = 10) -> list[tuple[str, int]]:
    """Run a kernel-trace-only pass and return (kernel name, count) pairs.

    ATT needs a ``--kernel-include-regex``; without one every dispatch is
    traced and the interesting kernel is buried among framework kernels.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "discover"
        proc = subprocess.run(
            [
                "rocprofv3",
                "--stats",
                "--kernel-trace",
                "-f",
                "csv",
                "-o",
                str(out),
                "--",
                *command,
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr[-2000:])
            raise SystemExit(f"kernel discovery failed (exit {proc.returncode})")
        stats = sorted(Path(tmp).glob("**/*kernel_stats.csv"))
        if not stats:
            raise SystemExit("kernel discovery produced no stats CSV")
        found: list[tuple[str, int]] = []
        with stats[0].open(newline="") as fh:
            for row in csv.DictReader(fh):
                name = row.get("Name") or row.get("KernelName") or ""
                if not name:
                    continue
                try:
                    calls = int(row.get("Calls") or 0)
                except ValueError:
                    calls = 0
                found.append((name, calls))
    return found[:top]


def capture(
    command: list[str],
    kernel_regex: str,
    output_dir: Path,
    decoder_dir: Path,
    *,
    target_cu: int,
    buffer_size: str,
    iteration_range: str,
    se_mask: str,
) -> None:
    """Run the ATT capture. Decoding happens inside rocprofv3 at finalize."""
    argv = [
        "rocprofv3",
        "--att",
        "--att-library-path",
        str(decoder_dir),
        "--att-target-cu",
        str(target_cu),
        "--att-buffer-size",
        buffer_size,
        "--att-shader-engine-mask",
        se_mask,
        "--kernel-include-regex",
        kernel_regex,
        "-d",
        str(output_dir),
    ]
    if iteration_range:
        argv += ["--kernel-iteration-range", iteration_range]
    argv += ["--", *command]

    print(f"[capture] {' '.join(argv)}\n", flush=True)
    proc = subprocess.run(argv)
    if proc.returncode != 0:
        raise SystemExit(f"rocprofv3 exited {proc.returncode}")


def summarize(dispatch: Path) -> dict:
    """Read the numbers that decide whether a decoded folder is worth opening."""
    code = json.loads((dispatch / "code.json").read_text())["code"]
    n = len(code)
    mapped = sum(1 for r in code if len(r) > 3 and r[3])
    executed = [r for r in code if len(r) > 9 and r[6]]

    waves = sorted(dispatch.glob("se*_sm*_sl*_wv*.json"))
    # state % 5 -> 0 EMPTY, 1 IDLE, 2 EXEC, 3 WAIT, 4 STALL. The wave state
    # stream is authoritative for "where did the time go", independent of the
    # per-instruction columns.
    labels = ("EMPTY", "IDLE", "EXEC", "WAIT", "STALL")
    totals = dict.fromkeys(labels, 0)
    for wf in waves:
        wave = json.loads(wf.read_text())["wave"]
        for state, duration in wave["timeline"]:
            totals[labels[state % 5]] += duration

    load = sum(
        f.stat().st_size
        for f in dispatch.iterdir()
        if f.suffix == ".json" and not f.name.startswith(BULK_FILES)
    )
    hot = sorted(executed, key=lambda r: -(r[8] or 0))[:3]
    return {
        "instructions": n,
        "source_mapped": mapped,
        "waves": len(waves),
        "states": totals,
        "load_bytes": load,
        "hot": [(r[0][:56], r[6], r[8]) for r in hot],
    }


def report(dispatch: Path) -> None:
    s = summarize(dispatch)
    n, mapped = s["instructions"], s["source_mapped"]
    pct = 100 * mapped // max(n, 1)
    print(f"  {dispatch.name}")
    print(f"    instructions   {n}  ({mapped} source-mapped, {pct}%)")
    print(f"    waves          {s['waves']}")
    print(f"    viewer payload {s['load_bytes'] / 1e6:.2f} MB")

    total = sum(s["states"].values()) or 1
    breakdown = "  ".join(
        f"{k} {100 * v / total:.1f}%" for k, v in s["states"].items() if v
    )
    print(f"    wave state     {breakdown}")
    print("    top stall (hit-weighted totals, divide by hits for per-execution):")
    for asm, hits, stall in s["hot"]:
        print(f"      stall={stall:<10} hits={hits:<7} {asm}")
    if mapped == 0:
        print(
            "    note: no source mapping, so the Source tab stays empty and "
            "analysis is at ISA level. Rebuild the kernel with "
            "ROCKE_DEBUG_LOC=1 to get DWARF and re-capture."
        )
    print()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--kernel-regex",
        default=None,
        help="kernel_include_regex for ATT. Omit to run a discovery pass first.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where to write the trace (default: ./att_out beside the cwd)",
    )
    p.add_argument(
        "--target-cu", type=int, default=1, help="CU to trace (ATT is per-CU)"
    )
    p.add_argument("--buffer-size", default="0x6000000", help="per-SE trace buffer")
    p.add_argument(
        "--iteration-range",
        default="[1, [2-3]]",
        help="dispatches to trace; skips warmup. Empty string traces all.",
    )
    p.add_argument("--att-shader-engine-mask", dest="se_mask", default="0xf")
    p.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="the command to profile, after --",
    )
    args = p.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        p.error("no command given; put the command to profile after --")

    decoder_dir = _preflight()

    regex = args.kernel_regex
    if regex is None:
        print("[discover] no --kernel-regex given, running a kernel-trace pass\n")
        kernels = discover_kernels(command)
        if not kernels:
            raise SystemExit("no kernels were dispatched by that command")
        for name, calls in kernels:
            print(f"    {calls:>6}x  {name}")
        # The rocke kernel is the descriptive unmangled one; framework kernels
        # are short or C++-mangled. Longest name is a good default, but say so.
        regex = max(kernels, key=lambda kc: len(kc[0]))[0]
        print(f"\n[discover] using --kernel-regex '{regex}'")
        print("           pass --kernel-regex explicitly to override\n")

    out = args.output_dir or Path.cwd() / "att_out"
    out.mkdir(parents=True, exist_ok=True)
    before = _snapshot_code_hashes(out)
    invalidate_provenance(out)

    try:
        capture(
            command,
            regex,
            out,
            decoder_dir,
            target_cu=args.target_cu,
            buffer_size=args.buffer_size,
            iteration_range=args.iteration_range,
            se_mask=args.se_mask,
        )
    except SystemExit as exc:
        _stamp_dispatches(out, before, capture=CAPTURE_TRUNCATED)
        raise exc

    _stamp_dispatches(out, before, capture=CAPTURE_COMPLETE)

    dispatches = sorted(out.glob(DISPATCH_GLOB))
    if not dispatches:
        raise SystemExit(
            f"no {DISPATCH_GLOB} folder was decoded under {out}.\n"
            "  The regex most likely matched no dispatch -- re-check the kernel "
            "name, or widen --iteration-range."
        )

    print(f"[decoded] {len(dispatches)} dispatch folder(s) under {out}\n")
    for d in dispatches:
        report(d)

    print("Open in WaveScope: run 'WaveScope: Open Trace Folder...' and pick")
    print(f"  {dispatches[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
