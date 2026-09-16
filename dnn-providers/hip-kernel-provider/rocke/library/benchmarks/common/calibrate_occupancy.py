# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Calibrate the static occupancy model against real hardware.

The gate in ``library/tests/test_attention_builds.py`` asserts a min-occupancy
floor computed by ``rocke.benchmark.perf.occupancy.estimate_occupancy_detail`` --
a static, no-GPU model whose per-arch caps are PROVISIONAL (gfx942 in particular
models a conservative 8 waves/SIMD, granularity 16, where CDNA3 is really 10 / 8).
This tool closes the loop: it predicts occupancy for a spread of reference kernels
and compares against rocprofv3's measured ``MeanOccupancyPerCU`` so the caps can be
corrected, then promoted into ``core/arch/data/arch_specs.json``.

Two halves, split by what needs a GPU:

  predict  (NO GPU)  -- compile the reference kernels, run the model, write a
                        predictions JSON + print a table. Runnable anywhere.
  measure  (GPU)     -- wrap each kernel's launcher with
                        ``rocprofv3 --pmc MeanOccupancyPerCU`` and parse the CSV.
  compare  (NO GPU)  -- join predictions with a measured CSV and report
                        predicted-vs-measured + the direction each cap is off.

Measurement command this tool issues (or that you can run by hand):

    rocprofv3 -i pmc.yaml --output-format csv -d <outdir> -- <launcher>
        # pmc.yaml:  jobs:\n  - pmc:\n      - MeanOccupancyPerCU

``MeanOccupancyPerCU`` = accumulate(SQ_LEVEL_WAVES)/GRBM_GUI_ACTIVE/CU_NUM, i.e.
the time-weighted average waves resident per CU -- directly comparable to the
model's ``waves_per_cu``. ``<launcher>`` is a full command that launches kernels on
the GPU -- an EXISTING driver (e.g. a prefill or decode benchmark under
``library/benchmarks/**`` / ``library/builders/**``), NOT a per-kernel thing. Run
launches many kernels; rocprofv3 records the counter for EACH, so a prefill run
covers the prefill kernels and a decode run covers the segment/reduce kernels.
This tool never launches kernels itself, so the launch path stays your existing,
tested one; ``compare`` matches whatever the CSV(s) contain, by kernel name.

Usage::

    # on any box (no GPU): predict for the reference spread.
    python -m benchmarks.common.calibrate_occupancy predict --arch gfx942 \
        --out pred_gfx942.json

    # on a gfx942/gfx950 box: run your prefill and decode drivers under rocprofv3.
    # Do this once per driver (each writes a CSV); or run rocprofv3 by hand.
    python -m benchmarks.common.calibrate_occupancy measure \
        --launcher "python -m benchmarks.gfx942.attention.prefill.benchmark_dense_prefill_live --d 128 --hq 32 --hkv 8 --dtype bf16" \
        --out meas_prefill_gfx942.csv

    # back on any box: compare, merging as many measured CSVs as you captured.
    python -m benchmarks.common.calibrate_occupancy compare \
        --pred pred_gfx942.json --measured meas_prefill_gfx942.csv meas_decode_gfx942.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

MEAN_OCCUPANCY_COUNTER = "MeanOccupancyPerCU"


# ---------------------------------------------------------------------------
# Reference kernels: a spread chosen so different limiters bind. Extend freely --
# the model is only as calibrated as the shapes you measure. Each entry builds one
# KernelDef for one arch; kept as callables so importing this module is GPU-free.
# ---------------------------------------------------------------------------
def _dense(arch: str, *, persistent: bool, head_size: int = 128, dtype: str = "bf16"):
    from kernels import AttentionDenseSpec, build_attention_dense

    spec = AttentionDenseSpec(
        batch=1,
        seqlen_q=2048,
        seqlen_kv=2048,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=head_size,
        causal=True,
        dtype=dtype,
        persistent=persistent,
    )
    return build_attention_dense(spec, arch=arch)


def _tiled(arch: str, *, head_size: int = 128, dtype: str = "fp16"):
    from unittest import mock

    import kernels.common.attention_unified as au
    from kernels import UnifiedAttentionProblem, build_unified_attention_2d_tiled

    prob = UnifiedAttentionProblem(
        total_q=2048,
        num_seqs=1,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=head_size,
        block_size=64,
        max_seqlen_q=2048,
        max_seqlen_k=2048,
        dtype=dtype,
        sliding_window=0,
    )
    with mock.patch.object(au, "_resolve_attention_arch", return_value=arch):
        return build_unified_attention_2d_tiled(
            au._tiled_spec_from_problem(prob), arch=arch
        )


def _scalar(kind: str):
    """Small scalar attention kernels -- low VGPR, so they reach high occupancy and
    exercise the ``max_waves_per_simd`` cap (the gfx942 8-vs-10 question)."""
    from kernels import (
        UnifiedAttention2DSpec,
        UnifiedAttention3DSpec,
        UnifiedAttentionProblem,
        UnifiedAttentionReduceSpec,
        build_unified_attention_2d,
        build_unified_attention_3d,
        build_unified_attention_reduce,
    )

    p = UnifiedAttentionProblem(
        total_q=3,
        num_seqs=1,
        num_query_heads=4,
        num_kv_heads=4,
        head_size=128,
        block_size=16,
        max_seqlen_q=3,
        max_seqlen_k=16,
        dtype="fp16",
    )
    if kind == "2d":
        return build_unified_attention_2d(UnifiedAttention2DSpec(p))
    if kind == "3d":
        return build_unified_attention_3d(UnifiedAttention3DSpec(p, num_segments=8))
    return build_unified_attention_reduce(UnifiedAttentionReduceSpec(p, num_segments=8))


def _tiled_decode_3d():
    """Tiled 3D decode -- LDS-limited with AGPRs allocated; exercises the LDS/AGPR
    branches the high-VGPR shipped kernels never reach."""
    from kernels import UnifiedAttention3DTiledSpec, build_unified_attention_3d_tiled

    return build_unified_attention_3d_tiled(
        UnifiedAttention3DTiledSpec(
            head_size=128,
            block_size=16,
            num_query_heads=16,
            num_kv_heads=2,
            dtype="fp16",
            use_sinks=False,
            sliding_window=0,
            has_softcap=False,
            num_segments=128,
            num_seqs=4,
        )
    )


def _tiled_small_2d(arch: str):
    from kernels import UnifiedAttention2DTiledSpec, build_unified_attention_2d_tiled

    return build_unified_attention_2d_tiled(
        UnifiedAttention2DTiledSpec(
            head_size=128,
            block_size=16,
            num_query_heads=16,
            num_kv_heads=2,
            dtype="fp16",
            use_sinks=False,
            sliding_window=0,
            has_softcap=False,
        ),
        arch=arch,
    )


# (label, arch, build_fn). Chosen to span the occupancy curve and every limiter,
# not just the high-VGPR regime the shipped kernels live in: the shipped budgeted
# kernels (what the gate floors), a low-VGPR high-occupancy point (validates the
# max waves/SIMD cap -- the gfx942 8-vs-10 question), mid-VGPR points (granularity),
# and LDS-limited kernels with AGPRs allocated (the LDS/AGPR branches). Build
# failures on an unsupported arch are skipped by ``predict`` -- extend freely.
# NB: build_attention_dense is gfx950-only; tiled-2d/3d and scalar run on both.
REFERENCE_KERNELS: list[tuple[str, str, Callable[[], object]]] = [
    # shipped, budgeted -> what the gate actually floors:
    ("dense_persist_d128_bf16", "gfx950", lambda: _dense("gfx950", persistent=True)),
    (
        "tiled_d128_fp16",
        "gfx942",
        lambda: _tiled("gfx942", head_size=128, dtype="fp16"),
    ),
    # high occupancy (~max waves/SIMD) -> validates the wave cap on each arch:
    ("scalar_reduce_d128", "gfx942", lambda: _scalar("reduce")),
    ("scalar_reduce_d128", "gfx950", lambda: _scalar("reduce")),
    # mid VGPR -> validates allocation granularity / rounding:
    ("scalar_2d_d128", "gfx942", lambda: _scalar("2d")),
    ("scalar_2d_d128", "gfx950", lambda: _scalar("2d")),
    ("scalar_3d_d128", "gfx942", lambda: _scalar("3d")),
    ("scalar_3d_d128", "gfx950", lambda: _scalar("3d")),
    # LDS-limited, AGPRs allocated -> validates the LDS and AGPR branches. gfx950
    # only: the small decode geometries hard-ABORT gfx942 codegen (an uncatchable
    # LLVM error, not a Python raise), and a ref that aborts an arch's compile must
    # not be listed for that arch -- predict() can skip a Python exception but not a
    # process abort. The LDS/AGPR branches are arch-agnostic, so gfx950 suffices.
    ("tiled_3d_decode_d128", "gfx950", _tiled_decode_3d),
    ("tiled_2d_small_d128", "gfx950", lambda: _tiled_small_2d("gfx950")),
    # contrast points (vary VGPR/LDS so the fit is not overdetermined):
    ("dense_default_d128_bf16", "gfx950", lambda: _dense("gfx950", persistent=False)),
    (
        "tiled_d128_fp16_950",
        "gfx950",
        lambda: _tiled("gfx950", head_size=128, dtype="fp16"),
    ),
    (
        "tiled_d64_bf16_942",
        "gfx942",
        lambda: _tiled("gfx942", head_size=64, dtype="bf16"),
    ),
]


@dataclass
class Prediction:
    label: str
    arch: str
    kernel_name: str
    vgpr: int
    agpr: int
    lds_bytes: int
    waves_per_wg: int
    predicted_waves_per_cu: int
    predicted_waves_per_simd: int
    limited_by: str


def predict(arch: Optional[str] = None) -> list[Prediction]:
    """Compile the reference kernels and run the static model. No GPU."""
    from rocke.benchmark.perf.occupancy import (
        estimate_occupancy_detail,
        parse_notes,
    )
    from rocke.helpers.compile import compile_kernel

    rows: list[Prediction] = []
    for label, karch, build in REFERENCE_KERNELS:
        if arch and karch != arch:
            continue
        try:
            kernel = build()
            hsaco = bytes(
                compile_kernel(kernel, arch=karch, capture_ir_text=False).hsaco
            )
        except Exception as e:  # arch-unsupported ref, etc. -- skip, don't abort
            print(
                f"# skip {label} ({karch}): {type(e).__name__}: {str(e)[:70]}",
                file=sys.stderr,
            )
            continue
        notes = parse_notes(hsaco)
        det = estimate_occupancy_detail(hsaco, karch)
        if not det:
            continue
        wg = notes.get("max_flat_workgroup_size", 0)
        rows.append(
            Prediction(
                label=label,
                arch=karch,
                kernel_name=kernel.name,
                vgpr=notes.get("vgpr", 0),
                agpr=notes.get("agpr", 0),
                lds_bytes=notes.get("lds_bytes", 0),
                waves_per_wg=max(wg // 64, 1) if wg else 1,
                predicted_waves_per_cu=det["waves_per_cu"],
                predicted_waves_per_simd=det["waves_per_simd"],
                limited_by=det["limited_by"],
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Pure parsing / comparison (unit-tested without a GPU).
# ---------------------------------------------------------------------------
def parse_measured_csv(
    text: str, counter: str = MEAN_OCCUPANCY_COUNTER
) -> dict[str, float]:
    """Mean of ``counter`` per kernel from a rocprofv3 counter-collection CSV.

    rocprofv3 emits one row per (dispatch, counter) with ``Kernel_Name`` /
    ``Counter_Name`` / ``Counter_Value`` columns; we average the requested counter
    across dispatches of the same kernel. Robust to extra columns and to the
    counter being absent (that kernel is simply omitted)."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        if (row.get("Counter_Name") or "").strip() != counter:
            continue
        name = (row.get("Kernel_Name") or "").strip()
        try:
            val = float(row.get("Counter_Value", ""))
        except (TypeError, ValueError):
            continue
        sums[name] = sums.get(name, 0.0) + val
        counts[name] = counts.get(name, 0) + 1
    return {k: sums[k] / counts[k] for k in sums if counts[k]}


def _match_measured(kernel_name: str, measured: dict[str, float]) -> Optional[float]:
    """rocprofv3 kernel names are mangled/decorated; match by substring on the
    DSL symbol (unique per shape). Returns the measured mean or None."""
    exact = measured.get(kernel_name)
    if exact is not None:
        return exact
    hits = [v for mk, v in measured.items() if kernel_name in mk]
    return hits[0] if len(hits) == 1 else None


@dataclass
class Comparison:
    label: str
    arch: str
    kernel_name: str
    predicted_waves_per_cu: int
    measured_waves_per_cu: Optional[float]
    delta: Optional[float]
    verdict: str  # MATCH / MODEL_LOW / MODEL_HIGH / NO_MEASUREMENT


def compare(
    predictions: list[Prediction], measured: dict[str, float], tol: float = 0.5
) -> list[Comparison]:
    """Join predictions with measured occupancy; flag the direction of any gap.

    ``MODEL_LOW`` (measured > predicted) means the caps under-count residency --
    e.g. gfx942 modeled at 8 waves/SIMD when hardware allows 10, or granularity too
    coarse. ``MODEL_HIGH`` means the model is optimistic (a real limiter the caps
    miss). Either way the fix is a caps edit, re-run, converge."""
    out: list[Comparison] = []
    for p in predictions:
        m = _match_measured(p.kernel_name, measured)
        if m is None:
            verdict = "NO_MEASUREMENT"
            delta = None
        else:
            delta = m - p.predicted_waves_per_cu
            if abs(delta) <= tol:
                verdict = "MATCH"
            elif delta > 0:
                verdict = "MODEL_LOW"
            else:
                verdict = "MODEL_HIGH"
        out.append(
            Comparison(
                label=p.label,
                arch=p.arch,
                kernel_name=p.kernel_name,
                predicted_waves_per_cu=p.predicted_waves_per_cu,
                measured_waves_per_cu=m,
                delta=delta,
                verdict=verdict,
            )
        )
    return out


def _rocprofv3_measure(
    launcher: str, counter: str = MEAN_OCCUPANCY_COUNTER, timeout: int = 1800
) -> str:
    """Run ``launcher`` once under rocprofv3 collecting ``counter``; return the
    concatenated counter-CSV text.

    ``launcher`` is a full command that launches kernels on the GPU -- an existing
    driver such as ``_profile_one`` or ``benchmark_dense_prefill_live``, NOT a
    per-kernel thing. rocprofv3 records the counter for EVERY kernel the run
    dispatches, so one prefill run covers the prefill kernels and one decode run
    covers the segment/reduce kernels; ``compare`` matches each by name.

    ``counter`` defaults to ``MeanOccupancyPerCU`` (what ``compare`` expects); pass
    a raw counter like ``SQ_WAVES`` to isolate whether a crash is the derived
    occupancy metric or rocprofv3 instrumentation of the kernel itself.

    GPU path -- not exercised without hardware; the CSV it returns is consumed by
    the unit-tested ``parse_measured_csv``."""
    with tempfile.TemporaryDirectory() as d:
        outdir = Path(d)
        pmc = outdir / "pmc.yaml"  # text `pmc:` form is deprecated in rocprofv3
        pmc.write_text(f"jobs:\n  - pmc:\n      - {counter}\n")
        subprocess.run(
            [
                "rocprofv3",
                "-i",
                str(pmc),
                "-d",
                str(outdir),
                "--output-format",
                "csv",
                "--",
                *launcher.split(),
            ],
            check=True,
            timeout=timeout,
        )
        parts = [
            p.read_text()
            for p in sorted(outdir.rglob("*counter_collection.csv"))
            or sorted(outdir.rglob("*.csv"))
        ]
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_predictions(rows: list[Prediction]) -> None:
    hdr = f"{'label':<28} {'arch':<7} {'vgpr':>4} {'agpr':>4} {'lds':>7} {'w/cu':>5} {'w/simd':>6} limiter"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r.label:<28} {r.arch:<7} {r.vgpr:>4} {r.agpr:>4} {r.lds_bytes:>7} "
            f"{r.predicted_waves_per_cu:>5} {r.predicted_waves_per_simd:>6} {r.limited_by}"
        )


def _print_comparisons(rows: list[Comparison]) -> None:
    hdr = f"{'label':<28} {'arch':<7} {'pred w/cu':>9} {'meas w/cu':>9} {'delta':>6} verdict"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        meas = (
            "-" if r.measured_waves_per_cu is None else f"{r.measured_waves_per_cu:.2f}"
        )
        delta = "-" if r.delta is None else f"{r.delta:+.2f}"
        print(
            f"{r.label:<28} {r.arch:<7} {r.predicted_waves_per_cu:>9} {meas:>9} "
            f"{delta:>6} {r.verdict}"
        )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_pred = sub.add_parser("predict", help="compile refs + run the model (no GPU)")
    p_pred.add_argument("--arch", default=None, help="limit to one arch")
    p_pred.add_argument("--out", type=Path, default=None, help="write predictions JSON")

    p_meas = sub.add_parser("measure", help="run a launcher under rocprofv3 (GPU)")
    p_meas.add_argument(
        "--launcher",
        required=True,
        help="full command that launches kernels on the GPU (e.g. a prefill or "
        "decode benchmark driver); rocprofv3 records MeanOccupancyPerCU for every "
        "kernel it dispatches",
    )
    p_meas.add_argument("--out", type=Path, required=True, help="write the counter CSV")
    p_meas.add_argument(
        "--counter",
        default=MEAN_OCCUPANCY_COUNTER,
        help="counter to collect (default MeanOccupancyPerCU; try SQ_WAVES to isolate a crash)",
    )

    p_cmp = sub.add_parser("compare", help="predicted vs measured (no GPU)")
    p_cmp.add_argument("--pred", type=Path, required=True)
    p_cmp.add_argument(
        "--measured",
        type=Path,
        nargs="+",
        required=True,
        help="one or more counter CSVs (e.g. a prefill CSV and a decode CSV)",
    )
    p_cmp.add_argument("--tol", type=float, default=0.5)

    args = ap.parse_args(argv)

    if args.cmd == "predict":
        rows = predict(args.arch)
        _print_predictions(rows)
        if args.out:
            args.out.write_text(json.dumps([asdict(r) for r in rows], indent=2))
            print(f"\nwrote {args.out}")
        return 0

    if args.cmd == "measure":
        try:
            text = _rocprofv3_measure(args.launcher, counter=args.counter)
        except subprocess.CalledProcessError as e:
            # rocprofv3 turns a GPU fault in the profiled app into a hard crash
            # (SIGSEGV). Almost always one of the launched kernels faulted, not the
            # counter -- restrict the driver to a single passing mode/shape.
            print(
                f"rocprofv3 exited abnormally ({e.returncode}). The profiled app "
                "likely hit a GPU fault -- restrict the launcher to a single "
                "passing mode (e.g. add '--mode persistent') and retry.",
                file=sys.stderr,
            )
            return 1
        args.out.write_text(text)
        found = parse_measured_csv(text, counter=args.counter)
        print(f"wrote {args.out}: {len(found)} kernel(s) with {args.counter}")
        return 0

    if args.cmd == "compare":
        preds = [Prediction(**d) for d in json.loads(args.pred.read_text())]
        measured: dict[str, float] = {}
        for path in args.measured:
            measured.update(parse_measured_csv(path.read_text()))
        rows = compare(preds, measured, tol=args.tol)
        _print_comparisons(rows)
        off = [r for r in rows if r.verdict in ("MODEL_LOW", "MODEL_HIGH")]
        return 1 if off else 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
