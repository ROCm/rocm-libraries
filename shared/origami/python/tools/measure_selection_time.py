#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Measure hipBLASLt solution-selection time for the Origami libraries.

This drives ``hipblaslt-bench`` over a problem set with kernel *execution* skipped
(``TENSILE_DB2=0x1``) and the per-call selection timer enabled (``TENSILE_DB=0x10000``),
then parses the ``"Solution selection time: <us> us"`` lines the library prints for
a single bench run, reports summary statistics, and can render a bar plot.

The problem set is the BF16_TN case (bf16, transA=T, transB=N) and defaults to the
ranking-regression suite's ``problem_data.csv`` (``tests/data/problem_data.csv``),
converted to a bench YAML on the fly so the numbers line up with the ranking tests.

Whatever Origami estimation path the given binary/env produces is what gets
measured -- to exercise a specific path, set its env var before running (e.g.
``ORIGAMI_LEVELED_ESTIMATION=1``). To compare two builds, run this once per build.

Examples
--------
Measure selection time on the ranking-regression problems and write a bar plot::

    python tools/measure_selection_time.py --plot selection_time.png \
        --bench /path/to/hipblaslt-install/bin/hipblaslt-bench \
        --lib-dir /path/to/hipblaslt-install/lib

Use an existing bench YAML instead of the regression CSV::

    python tools/measure_selection_time.py --yaml /path/to/problems.yaml ...
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

# tests/data/problem_data.csv, relative to this file (python/tools/ -> python/tests/...)
_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = _THIS_DIR.parent / "tests" / "data" / "problem_data.csv"

# Per-call line printed by MasterSolutionLibrary when TENSILE_DB=0x10000.
_SEL_RE = re.compile(r"Solution selection time:\s*([\d.eE+-]+)\s*us")


# --------------------------------------------------------------------------------------
# Problem set -> bench YAML
# --------------------------------------------------------------------------------------
_DTYPE_BYTES = {"f8": 1, "bf16": 2, "f16": 2, "f32": 4, "xf32": 4}


def _footprint_bytes(m: int, n: int, k: int, batch: int, dbytes: int) -> int:
    """Approximate device allocation for one problem (A + B + C + D)."""
    return (m * k + k * n + 2 * m * n) * batch * dbytes


def csv_to_bench_yaml(
    csv_path: Path,
    out_path: Path,
    dtype: str = "bf16",
    trans_a: str = "T",
    trans_b: str = "N",
    limit: int | None = None,
    max_gib: float = 4.0,
) -> int:
    """Convert a ``(m, n, k, batch_count)`` CSV into a hipBLASLt-bench YAML.

    Mirrors the field layout of the workloads we run by hand (TN bf16 GEMMs), which
    is what the bench validates before selection. Leading dimensions follow the
    proven pattern lda=M, ldb=N, ldc=ldd=M; strides are set only for batched cases.
    Kernel execution is skipped at measurement time, so the rotating buffer is off.

    Problems whose A+B+C+D footprint exceeds ``max_gib`` are skipped -- the bench
    still allocates the matrices before selection, and the ranking-regression set
    contains synthetic sizes (hundreds of GB) that would OOM. Set ``max_gib<=0`` to
    keep everything.

    Returns the number of problems written.
    """
    t = f"{dtype}_r"
    dbytes = _DTYPE_BYTES.get(dtype, 2)
    cap = int(max_gib * (1 << 30)) if max_gib and max_gib > 0 else None

    rows = []
    skipped = 0
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            try:
                m, n, k = int(row["m"]), int(row["n"]), int(row["k"])
                batch = int(row.get("batch_count", 1) or 1)
            except (KeyError, ValueError):
                continue
            if m <= 0 or n <= 0 or k <= 0:
                continue
            if cap is not None and _footprint_bytes(m, n, k, batch, dbytes) > cap:
                skipped += 1
                continue
            rows.append((m, n, k, batch))
    if limit is not None:
        rows = rows[:limit]
    if skipped:
        print(f"  ({skipped} problems skipped: footprint > {max_gib} GiB)")

    lines = []
    for (m, n, k, batch) in rows:
        # Column-major leading dims for C = op(A)*op(B):
        #   A: M x K  -> stored M x K (N) or K x M (T)  => lda = M (N) or K (T)
        #   B: K x N  -> stored K x N (N) or N x K (T)  => ldb = K (N) or N (T)
        #   C/D: M x N                                  => ldc = ldd = M
        lda = k if trans_a == "T" else m
        ldb = n if trans_b == "T" else k
        ldc = ldd = m
        sa, sb = (m * k, k * n) if batch > 1 else (0, 0)
        sc = sd = (m * n) if batch > 1 else 0
        lines.append(
            "- {{function: matmul, a_type: {t}, b_type: {t}, c_type: {t}, d_type: {t}, "
            "compute_type: c_f32_r, scale_type: f32_r, bias_type: f32_r, scaleA: 0, scaleB: 0, "
            "transA: {ta}, transB: {tb}, M: {m}, N: {n}, K: {k}, batch_count: {b}, "
            "lda: {lda}, ldb: {ldb}, ldc: {ldc}, ldd: {ldd}, initialization: trig_float, alpha: 1, beta: 0, "
            "stride_a: {sa}, stride_b: {sb}, stride_c: {sc}, stride_d: {sd}, rotating: 0, flush: 0, "
            "iters: 1, cold_iters: 0, print_kernel_info: 0, use_gpu_timer: 1}}".format(
                t=t, ta=trans_a, tb=trans_b, m=m, n=n, k=k, b=batch,
                lda=lda, ldb=ldb, ldc=ldc, ldd=ldd, sa=sa, sb=sb, sc=sc, sd=sd
            )
        )
    out_path.write_text("\n".join(lines) + "\n")
    return len(rows)


# --------------------------------------------------------------------------------------
# Run + parse
# --------------------------------------------------------------------------------------
def _build_ld_library_path(lib_dir: str | None, rocm_prefix: str, existing: str) -> str:
    """Prepend the lib under test, then the existing path, then the ROCm runtime dirs
    (for libomp/libamdhip etc.). Only existing directories are added."""
    parts: list[str] = []
    if lib_dir:
        parts.append(lib_dir)
    if existing:
        parts.append(existing)
    for sub in ("lib/llvm/lib", "lib"):
        d = os.path.join(rocm_prefix, sub)
        if os.path.isdir(d):
            parts.append(d)
    return os.pathsep.join(parts)


def measure_selection_times(
    bench: str,
    yaml_path: str,
    lib_dir: str | None = None,
    rocm_prefix: str = "/opt/rocm",
    extra_env: dict[str, str] | None = None,
) -> list[float]:
    """Run hipblaslt-bench (kernels skipped) and return the per-call selection times (us).

    The Origami estimation path (flat vs. leveled etc.) is whatever the given
    binary/env produces -- this just measures the one bench. To exercise a
    particular path, set its env var before calling (e.g. ORIGAMI_LEVELED_ESTIMATION).
    """
    env = os.environ.copy()
    env["TENSILE_PREDICTION_LIB"] = "1"  # route selection through the Origami prediction lib
    env["TENSILE_DB"] = "0x10000"        # print "Solution selection time"
    env["TENSILE_DB2"] = "0x1"           # skip kernel launch (selection only)
    env["LD_LIBRARY_PATH"] = _build_ld_library_path(lib_dir, rocm_prefix, env.get("LD_LIBRARY_PATH", ""))
    # The default Tensile library path resolves relative to libhipblaslt.so, so we
    # deliberately do NOT set HIPBLASLT_TENSILE_LIBPATH here.
    env.pop("HIPBLASLT_TENSILE_LIBPATH", None)
    if extra_env:
        env.update(extra_env)

    proc = subprocess.run(
        [bench, "--yaml", yaml_path],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    times = [float(m.group(1)) for m in _SEL_RE.finditer(proc.stdout)]
    if not times:
        sys.stderr.write(
            "WARNING: no 'Solution selection time' lines parsed. Last bench output:\n"
            + "\n".join(proc.stdout.splitlines()[-15:])
            + "\n"
        )
    return times


def summarize(times: list[float], drop_first: int = 1) -> dict[str, float]:
    """Summary stats. ``drop_first`` discards warmup calls (first selection pays the
    one-time library load / JIT, which badly skews the mean)."""
    xs = sorted(times[drop_first:] if len(times) > drop_first else times)
    if not xs:
        return {"n": 0}
    n = len(xs)
    pct = lambda p: xs[min(n - 1, int(p * n))]
    return {
        "n": n,
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "min": xs[0],
        "max": xs[-1],
    }


def print_summary(label: str, s: dict[str, float]) -> None:
    if not s.get("n"):
        print(f"{label}: no samples")
        return
    print(
        f"{label:>16}: n={s['n']:5d}  median={s['median']:7.2f}us  mean={s['mean']:7.2f}us  "
        f"p90={s['p90']:7.2f}  p99={s['p99']:7.2f}  min={s['min']:6.2f}  max={s['max']:7.2f}"
    )


# --------------------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------------------
def plot_bars(
    summary: dict[str, float],
    out_path: str,
    metrics: tuple[str, ...] = ("median", "mean", "p90", "p99", "max"),
    title: str = "hipBLASLt solution-selection time (Origami, BF16_TN)",
) -> None:
    """Bar plot of the selection-time stats for a single bench run.

    ``summary`` is a stats dict from :func:`summarize`; one bar per metric.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    vals = [summary.get(m, 0.0) for m in metrics]
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(1.6 * len(metrics) + 1, 4.5))
    bars = ax.bar(x, vals, 0.6, color="tab:blue")
    for b, v in zip(bars, vals):
        ax.annotate(
            f"{v:.1f}",
            xy=(b.get_x() + b.get_width() / 2, v),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylabel("selection time (us)")
    ax.set_title(title + f"  (n={summary.get('n', 0)})")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote plot: {out_path}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True, help="path to hipblaslt-bench")
    p.add_argument("--lib-dir", default=None, help="dir to prepend to LD_LIBRARY_PATH (the libhipblaslt.so to test)")
    p.add_argument("--rocm-prefix", default="/opt/rocm", help="ROCm prefix for runtime libs (libomp etc.)")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--yaml", default=None, help="use an existing bench YAML (skip CSV conversion)")
    src.add_argument("--csv", default=str(DEFAULT_CSV), help=f"problem CSV to convert (default: {DEFAULT_CSV})")
    # Problem case fixed to BF16_TN (bf16, transA=T, transB=N) -- the case we track.
    p.add_argument("--dtype", default="bf16", help="element dtype for generated YAML (default: bf16)")
    p.add_argument("--transA", default="T")
    p.add_argument("--transB", default="N")
    p.add_argument("--limit", type=int, default=None, help="cap number of problems")
    p.add_argument("--max-gib", type=float, default=4.0, help="skip problems whose A+B+C+D footprint exceeds this (GiB); <=0 keeps all")
    p.add_argument("--drop-first", type=int, default=1, help="drop N warmup selections from stats (default: 1)")
    p.add_argument("--plot", default=None, help="output PNG for the bar plot")
    args = p.parse_args(argv)

    # Resolve the problem YAML.
    tmp = None
    if args.yaml:
        yaml_path = args.yaml
    else:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        tmp.close()
        n = csv_to_bench_yaml(
            Path(args.csv), Path(tmp.name), args.dtype, args.transA, args.transB, args.limit, args.max_gib
        )
        yaml_path = tmp.name
        print(f"generated bench YAML from {args.csv}: {n} problems -> {yaml_path}")

    # Single bench run.
    times = measure_selection_times(
        args.bench, yaml_path, lib_dir=args.lib_dir, rocm_prefix=args.rocm_prefix
    )
    s = summarize(times, drop_first=args.drop_first)
    print_summary("selection", s)

    if args.plot:
        plot_bars(s, args.plot)

    if tmp is not None:
        os.unlink(tmp.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
