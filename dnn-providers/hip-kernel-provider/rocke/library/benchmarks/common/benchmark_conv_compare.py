# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Side-by-side comparison of direct-conv and implicit-GEMM conv benchmarks.

Runs ``benchmark_direct_conv.py`` and ``benchmark_implicit_gemm_conv.py`` on the
same convolution shape, then prints a compact summary of the best kernel found
by each approach.

The two benchmark scripts are invoked as subprocesses so this script carries no
benchmark logic of its own — it only forwards arguments and aggregates results.

Usage
-----
Forward pass (fwd direction, grouped conv cpg=16):

  python benchmark_conv_compare.py \\
      --arch gfx950 --N 8 --Hi 56 --Wi 56 \\
      --C 64 --K 64 --Y 3 --X 3 --pH 1 --pW 1 --groups 4

Forward pass using a MIOpenDriver command:

  python benchmark_conv_compare.py \\
      --arch gfx950 \\
      --miopen-cmd "./MIOpenDriver convfp16 -n 8 -c 64 -H 56 -W 56 \\
          -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -F 1 -in_layout=NHWC"

Both scripts can also run purely ahead-of-time: compile each cache once with
the scripts' own ``--compile-all``, then compare out of the caches -- nothing is
compiled during the comparison:

  python benchmark_direct_conv.py --compile-all --cache-dir kernel_cache
  python benchmark_implicit_gemm_conv.py --compile-all --cache-dir kernel_cache
  python benchmark_conv_compare.py --run-from-cache kernel_cache \\
      --N 8 --Hi 56 --Wi 56 --C 64 --K 64 --groups 4

The two scripts' entries have distinct identities, so one directory can hold
both caches.

The ``--top`` / ``--warmup`` / ``--iters`` / ``--jobs`` / ``--verify`` /
``--dtype`` / ``--direction`` / ``--run-from-cache`` / ``--early-stop`` /
``--early-stop-after`` flags are forwarded to both scripts.

Early stopping carries over from one script to the other: direct conv runs
first and records each case's best time (a kernel that fails ``--verify`` is
never recorded), and the implicit-GEMM sweep starts
each case from that time instead of from nothing -- so from its first kernel it
skips any that cannot come within ``--early-stop`` times the direct result.
``--no-seed-from-direct`` turns that off.  ``--dtype`` / ``--direction`` default to fp16 / fwd for explicit shapes;
with a MIOpenDriver command they are taken from the command unless given.

Notes
-----
- The direct-conv cache holds fwd and dgrad kernels only, so a wgrad comparison
  out of the cache runs implicit-GEMM alone.
- Without a cache, direct conv requires ``cpg == kpg`` (C/groups == K/groups)
  and cpg = 1 or a multiple of 4; other forward shapes skip the direct-conv
  run. Out of the cache, each script filters its cached kernels by the shape's
  capabilities instead and reports when none fits.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent

# Regex patterns that match the "Best:" summary printed by both benchmarks.
_BEST_RE = re.compile(r"Best:\s*([\d.]+)\s*TFLOPS\s*[—\-]\s*(.+)")


def _tee(stream, sink, lines: list) -> None:
    """Echo *stream* to *sink* line by line as it arrives, keeping a copy."""
    for line in iter(stream.readline, ""):
        sink.write(line)
        sink.flush()
        lines.append(line)
    stream.close()


def _run_script(
    script: Path, extra_args: list[str], timeout: "float | None" = None
) -> tuple[float | None, str, str]:
    """Run *script* with *extra_args*; return (best_tflops, kernel_name, output).

    The script's stdout and stderr are echoed live, as it prints them; ``output``
    is the captured stdout followed by stderr."""
    import os
    import signal
    import threading

    # -u: a Python child block-buffers a piped stdout, which would hold its
    # progress back until it exits.
    cmd = [sys.executable, "-u", str(script)] + extra_args
    print(f"\n{'='*72}", flush=True)
    print(f"Running: {script.name} {' '.join(extra_args)}", flush=True)
    print(f"{'='*72}", flush=True)

    out_lines: list = []
    err_lines: list = []
    with subprocess.Popen(
        cmd,
        text=True,
        bufsize=1,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    ) as popen:
        readers = [
            threading.Thread(
                target=_tee, args=(popen.stdout, sys.stdout, out_lines), daemon=True
            ),
            threading.Thread(
                target=_tee, args=(popen.stderr, sys.stderr, err_lines), daemon=True
            ),
        ]
        for t in readers:
            t.start()
        try:
            popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(popen.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            popen.wait()
            print(
                f"\n[timeout] {script.name} exceeded {timeout:.0f}s limit — killed.",
                file=sys.stderr,
                flush=True,
            )
        for t in readers:
            t.join()

    raw_out = "".join(out_lines)
    output = raw_out + "".join(err_lines)
    m = _BEST_RE.search(raw_out)
    if m:
        return float(m.group(1)), m.group(2).strip(), output
    return None, "", output


def _cpg_valid_for_direct(C: int, K: int, groups: int) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for direct-conv cpg constraints."""
    if C % groups != 0 or K % groups != 0:
        return False, f"C={C} or K={K} not divisible by groups={groups}"
    cpg = C // groups
    kpg = K // groups
    if cpg != kpg:
        return False, f"cpg={cpg} != kpg={kpg} (direct conv requires C/g == K/g)"
    if cpg != 1 and (cpg % 4 != 0 or cpg < 4):
        return (
            False,
            f"cpg={cpg} must be 1 (depthwise) or a positive multiple of 4",
        )
    return True, ""


def _build_shape_args(args) -> list[str]:
    """Shared shape args forwarded to both scripts."""
    a: list[str] = [
        "--arch",
        args.arch,
        "--N",
        str(args.N),
        "--Hi",
        str(args.Hi),
        "--Wi",
        str(args.Wi),
        "--C",
        str(args.C),
        "--K",
        str(args.K),
        "--Y",
        str(args.Y),
        "--X",
        str(args.X),
        "--sH",
        str(args.sH),
        "--sW",
        str(args.sW),
        "--pH",
        str(args.pH),
        "--pW",
        str(args.pW),
        "--groups",
        str(args.groups),
        "--top",
        str(args.top),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--jobs",
        str(args.jobs),
    ]
    if args.verify:
        a.append("--verify")
    return a + _build_mode_args(args)


def _build_mode_args(args) -> list[str]:
    """dtype / direction / AOT-cache args forwarded to both scripts."""
    a: list[str] = []
    if args.dtype is not None:
        a += ["--dtype", args.dtype]
    if args.direction is not None:
        a += ["--direction", args.direction]
    if args.run_from_cache is not None:
        a += ["--run-from-cache", args.run_from_cache]
    if args.early_stop is not None:
        a += ["--early-stop", str(args.early_stop)]
    if args.early_stop_after is not None:
        a += ["--early-stop-after", str(args.early_stop_after)]
    return a


def _build_implicit_only_args(args) -> list[str]:
    """Args that apply only to implicit-GEMM and must not be forwarded to direct-conv."""
    a: list[str] = []
    if args.sample is not None:
        a += ["--sample", str(args.sample)]
    if args.seed != 0:
        a += ["--seed", str(args.seed)]
    if args.split_k != -1:
        a += ["--split-k", str(args.split_k)]
    if args.split_k_prune is not None:
        a += ["--split-k-prune", str(args.split_k_prune)]
    return a


def _build_miopen_args(args) -> list[str]:
    """MIOpen input args forwarded to both scripts."""
    a: list[str] = [
        "--arch",
        args.arch,
        "--top",
        str(args.top),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--jobs",
        str(args.jobs),
    ]
    if args.verify:
        a.append("--verify")
    if args.miopen_cmd:
        a += ["--miopen-cmd", args.miopen_cmd]
    elif args.miopen_file:
        a += ["--miopen-file", args.miopen_file]
    return a + _build_mode_args(args)


# The line an EarlyStop prints when a seeded sweep skipped every kernel.
_ALL_STOPPED_RE = re.compile(r"all \d+ early-stopped: .*")


def _print_summary(
    direct: tuple[float | None, str],
    implicit: tuple[float | None, str],
    implicit_note: str = "",
) -> None:
    direct_tflops, direct_name = direct
    implicit_tflops, implicit_name = implicit

    print(f"\n{'='*72}", flush=True)
    print("COMPARISON SUMMARY", flush=True)
    print(f"{'='*72}", flush=True)

    def _fmt(label, tflops, name, note=""):
        if tflops is None:
            why = note or "skipped or failed"
            print(f"  {label:<20}  (no result — {why})", flush=True)
        else:
            print(f"  {label:<20}  {tflops:7.1f} TFLOPS  {name}", flush=True)

    _fmt("direct-conv", direct_tflops, direct_name)
    _fmt("implicit-GEMM", implicit_tflops, implicit_name, implicit_note)

    if direct_tflops is not None and implicit_tflops is not None:
        ratio = direct_tflops / implicit_tflops
        winner = "direct-conv" if ratio >= 1.0 else "implicit-GEMM"
        print(
            f"\n  Speedup direct/implicit: {ratio:.3f}x  →  {winner} wins",
            flush=True,
        )
    print(f"{'='*72}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare direct-conv and implicit-GEMM benchmarks for the same conv shape. "
            "Both benchmark scripts are run as subprocesses; their output is echoed and "
            "their best TFLOPS results are printed side-by-side."
        )
    )
    parser.add_argument(
        "--arch",
        default="gfx950",
        help="gfx target (default: gfx950)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="top-N results to show per benchmark (default: 10)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="warmup iterations (default: 3)"
    )
    parser.add_argument(
        "--iters", type=int, default=10, help="timed iterations (default: 10)"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="parallel compile workers forwarded to both scripts (default: 1)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="forward --verify to both benchmark scripts",
    )
    parser.add_argument(
        "--early-stop",
        type=float,
        default=None,
        metavar="FACTOR",
        help="forwarded to both scripts (default: theirs)",
    )
    parser.add_argument(
        "--early-stop-after",
        type=int,
        default=None,
        metavar="N",
        help="forwarded to both scripts (default: theirs)",
    )
    parser.add_argument(
        "--no-seed-from-direct",
        action="store_true",
        help="do not start the implicit-GEMM early stopping from the direct-conv "
        "best times",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["fp16", "bf16"],
        help="operand dtype forwarded to both scripts (default: fp16, or the "
        "MIOpenDriver command's)",
    )
    parser.add_argument(
        "--direction",
        default=None,
        choices=["fwd", "dgrad", "wgrad"],
        help="convolution direction forwarded to both scripts (default: fwd, or "
        "the MIOpenDriver command's)",
    )
    parser.add_argument(
        "--run-from-cache",
        default=None,
        metavar="DIR",
        dest="run_from_cache",
        help="AOT: benchmark the kernels both scripts pre-compiled into DIR "
        "(their --compile-all) instead of compiling during the comparison",
    )
    parser.add_argument(
        "--skip-direct",
        action="store_true",
        dest="skip_direct",
        help="skip the direct-conv benchmark (implicit-GEMM only)",
    )
    parser.add_argument(
        "--skip-implicit",
        action="store_true",
        dest="skip_implicit",
        help="skip the implicit-GEMM benchmark (direct-conv only)",
    )

    implicit_grp = parser.add_argument_group(
        "Implicit-GEMM options",
        "Flags forwarded only to benchmark_implicit_gemm_conv.py.",
    )
    implicit_grp.add_argument(
        "--sample",
        type=float,
        default=None,
        metavar="FRAC",
        help=(
            "randomly sample FRAC of the candidate combinations before sweeping "
            "(e.g. 0.1 for ~10%%). Forwarded to implicit-GEMM only."
        ),
    )
    implicit_grp.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed used by --sample (default: 0)",
    )
    implicit_grp.add_argument(
        "--split-k",
        type=int,
        default=-1,
        dest="split_k",
        metavar="N",
        help=(
            "wgrad split-K degree forwarded to implicit-GEMM "
            "(-1 = off, 0 = auto-sweep, N>0 = fixed). Forwarded to implicit-GEMM only."
        ),
    )
    implicit_grp.add_argument(
        "--split-k-prune",
        type=float,
        default=None,
        dest="split_k_prune",
        metavar="PCT",
        help=(
            "prune split-K sweep when perf drops by PCT%% relative to the best so far. "
            "Only effective with --split-k 0. Forwarded to implicit-GEMM only."
        ),
    )

    miopen_grp = parser.add_argument_group(
        "MIOpen input",
        "Load the conv shape from a MIOpenDriver command instead of explicit flags.",
    )
    miopen_grp.add_argument(
        "--miopen-cmd",
        default=None,
        metavar="CMD",
        help="MIOpenDriver command string forwarded verbatim to both scripts.",
    )
    miopen_grp.add_argument(
        "--miopen-file",
        default=None,
        metavar="FILE",
        help="Path to a file of MIOpenDriver commands; forwarded to both scripts.",
    )

    shape_grp = parser.add_argument_group("Shape", "convolution shape parameters")
    shape_grp.add_argument("--N", type=int, default=8)
    shape_grp.add_argument("--Hi", type=int, default=56)
    shape_grp.add_argument("--Wi", type=int, default=56)
    shape_grp.add_argument("--C", type=int, default=64)
    shape_grp.add_argument("--K", type=int, default=64)
    shape_grp.add_argument("--Y", type=int, default=3)
    shape_grp.add_argument("--X", type=int, default=3)
    shape_grp.add_argument("--sH", type=int, default=1)
    shape_grp.add_argument("--sW", type=int, default=1)
    shape_grp.add_argument("--pH", type=int, default=1)
    shape_grp.add_argument("--pW", type=int, default=1)
    shape_grp.add_argument("--groups", "-g", type=int, default=1)

    args = parser.parse_args()

    using_miopen = args.miopen_cmd is not None or args.miopen_file is not None

    implicit_only = _build_implicit_only_args(args)

    if not using_miopen:
        # Explicit shapes carry no dtype/direction of their own.
        args.dtype = args.dtype or "fp16"
        args.direction = args.direction or "fwd"

    if using_miopen:
        shared_args = _build_miopen_args(args)
    else:
        shared_args = _build_shape_args(args)
    direct_args = list(shared_args)
    implicit_args = list(shared_args) + implicit_only

    if not args.skip_direct and args.run_from_cache and args.direction == "wgrad":
        print(
            "[info] direct-conv skipped: its cache holds no wgrad kernels",
            file=sys.stderr,
        )
        args.skip_direct = True
    if (
        not args.skip_direct
        and not using_miopen
        and not args.run_from_cache
        and args.direction == "fwd"
    ):
        # Validate cpg constraints for direct conv up-front so we can skip
        # gracefully rather than propagating errors through the subprocess.
        # A cache run needs no pre-check: the script filters its cached
        # kernels by the shape's capabilities itself.
        ok, reason = _cpg_valid_for_direct(args.C, args.K, args.groups)
        if not ok:
            print(
                f"[info] direct-conv skipped for this shape: {reason}",
                file=sys.stderr,
            )
            args.skip_direct = True

    direct_result: tuple[float | None, str] = (None, "")
    implicit_result: tuple[float | None, str] = (None, "")
    implicit_note = ""

    with tempfile.TemporaryDirectory() as tmp:
        # Direct conv's best time per case, handed to the implicit-GEMM sweep
        # as its starting early-stop bound.
        seed_file = str(Path(tmp) / "direct_best.json")
        seed = not args.skip_direct and not args.no_seed_from_direct
        if seed:
            direct_args += ["--early-stop-record", seed_file]

        if not args.skip_direct:
            tflops, name, _ = _run_script(
                _SCRIPT_DIR / "benchmark_direct_conv.py",
                direct_args,
                # The JIT sweep compiles; a cache run only launches.
                timeout=None if args.run_from_cache else 240,
            )
            direct_result = (tflops, name)

        if not args.skip_implicit:
            if seed and Path(seed_file).is_file():
                implicit_args += ["--early-stop-seed", seed_file]
            tflops, name, output = _run_script(
                _SCRIPT_DIR / "benchmark_implicit_gemm_conv.py",
                implicit_args,
            )
            implicit_result = (tflops, name)
            m = _ALL_STOPPED_RE.search(output)
            if tflops is None and m:
                implicit_note = m.group(0)

    _print_summary(direct_result, implicit_result, implicit_note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
