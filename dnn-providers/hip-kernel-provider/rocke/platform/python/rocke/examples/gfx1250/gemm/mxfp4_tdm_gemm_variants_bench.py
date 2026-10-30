# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Bench harness for cumulative MX FP4 TDM GEMM variants.

For each named variant: compile, report code-object resources (VGPR, SGPR,
spills, scratch, LDS), check the FP8 output against the reference at
``beta = 0`` and ``beta = 0.5``, then time the ``beta = 0`` path two ways:

* batched: ``--batch`` back-to-back enqueues per sync, host-timed, the way
  hipBLASLt's bench reports a kernel;
* median: the median of ``--iters`` single launches bracketed by HIP events.

The check uses exactly representable inputs (FP4 0 and +1, alpha 1/K). With
``--timing-inputs tensile`` (the default) the timed launches instead use the
Tensile client's inputs for the YAML run: A ``TrigSin``, B ``TrigCos``, E8M0
scales ``Random``, alpha 1. Those outputs saturate FP8 and have no exact FP32
reference, so the timed launches are not checked.

``--host-only`` compiles and reports resources without touching a device.

Absolute timings must not be committed to the repo. Use ``--out`` with a path
outside the tree.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

from ....examples.common.scaled_gemm_reference import pack_fp4_codes
from ....helpers.compile import compile_kernel
from ....instances.gfx1250.mxfp4_tdm_gemm import (
    build_mxfp4_tdm_gemm,
    is_valid_spec,
    mxfp4_tdm_gemm_grid,
)
from ....instances.gfx1250.mxfp4_tdm_gemm_variants import (
    BASE_VARIANT,
    variant_flags,
    variant_names,
    variant_spec,
    variant_step,
)
from .mxfp4_tdm_gemm import (
    YAML_TILE,
    _u8_buffer,
    kernel_inputs,
    make_inputs,
    reference,
)

_YAML_PROBLEM = (4096, 4096, 65536)
_SMOKE_PROBLEM = (1024, 1024, 2048)
"""Small enough for a full reference check, a multiple of the 4x4 cluster
grid, and four K tiles so the steady-state loop runs."""

FIELDS = [
    "step",
    "variant",
    "kernel",
    "flags",
    "check",
    "vgpr",
    "sgpr",
    "spills",
    "scratch",
    "lds",
    "batched_us",
    "median_us",
    "tflops",
    "speedup_vs_base",
]


def _parse_variants(text: str, *, trials: bool) -> list[str]:
    known = variant_names(trials=True)
    if text.strip() == "all":
        return variant_names(trials=trials)
    names = [part.strip() for part in text.split(",") if part.strip()]
    if not names:
        raise SystemExit("--variants must be 'all' or a comma list")
    for name in names:
        if name not in known:
            raise SystemExit(f"unknown variant {name!r}; known: {known}")
    return names


def _flag_summary(variant: str) -> str:
    bits = []
    for key, value in variant_flags(variant).items():
        if value is True:
            bits.append(key)
        elif isinstance(value, tuple):
            bits.append(f"{key}={'x'.join(str(v) for v in value)}")
        elif value not in (False, None):
            bits.append(f"{key}={value}")
    return "|".join(bits) if bits else "(none)"


def _resources(hsaco: bytes, emit_dir: Path | None, kernel: str) -> dict:
    """Code-object resources, or empty columns when the analyzer is unavailable."""
    import tempfile

    from ....analysis.isa import analyze_hsaco

    try:
        if emit_dir is not None:
            path = emit_dir / f"{kernel}.hsaco"
            info = analyze_hsaco(path, keep_text=True)
            (emit_dir / f"{kernel}.s").write_text(info.objdump_text)
        else:
            with tempfile.NamedTemporaryFile(suffix=".hsaco") as handle:
                handle.write(hsaco)
                handle.flush()
                info = analyze_hsaco(Path(handle.name))
    except Exception as exc:  # noqa: BLE001 - resources are best effort
        print(f"warning: resource analysis failed for {kernel}: {exc}", file=sys.stderr)
        return {}
    res = info.resources
    spills = sum(
        int(v)
        for k, v in (res.raw or {}).items()
        if "spill" in k and str(v).lstrip("-").isdigit()
    )
    return {
        "vgpr": res.vgpr_count,
        "sgpr": res.sgpr_count,
        "spills": spills,
        "scratch": res.scratch_bytes,
        "lds": res.lds_bytes,
    }


def _row(variant: str, **values) -> dict:
    row = {field: "" for field in FIELDS}
    row["step"] = variant_step(variant)
    row["variant"] = variant
    row["flags"] = _flag_summary(variant)
    row.update(values)
    return row


def _check(spec, got, inputs, *, alpha, beta, verify: str, samples: int) -> bool:
    expected = reference(
        spec, inputs, alpha=alpha, beta=beta, saturate=spec.vector_epilogue
    ).view(np.uint8)
    if verify == "full":
        return bool(np.array_equal(got, expected))
    rng = np.random.default_rng(0x32F4)
    count = min(samples, spec.M * spec.N)
    flat = rng.choice(spec.M * spec.N, size=count, replace=False)
    rows, cols = np.divmod(flat, spec.N)
    return bool(np.array_equal(got[rows, cols], expected[rows, cols]))


def _trig_fp4_codes(rows: int, cols: int, *, use_cos: bool) -> np.ndarray:
    """E2M1 codes of ``sin`` or ``cos`` of the linear element index (K
    fastest), rounded to nearest even: the client's ``initArrayTrig``."""
    import ml_dtypes

    codes = np.empty((rows, cols), dtype=np.uint8)
    trig = np.cos if use_cos else np.sin
    chunk = max(1, (1 << 24) // cols)
    for begin in range(0, rows, chunk):
        end = min(rows, begin + chunk)
        index = np.arange(begin * cols, end * cols, dtype=np.float64)
        values = trig(index).astype(np.float32).astype(ml_dtypes.float4_e2m1fn)
        codes[begin:end] = values.view(np.uint8).reshape(end - begin, cols)
    return codes


def tensile_timing_inputs(spec, seed: int = 0x32F4):
    """A, B and scales filled like the Tensile client's YAML run, in
    ``make_inputs``' canonical layout. ``Random`` E8M0 scales are
    ``rand() % 7 - 3 + 127`` there; this draws the same range, not the same
    sequence."""
    rng = np.random.default_rng(seed)
    groups = spec.K // spec.block_k
    return (
        pack_fp4_codes(_trig_fp4_codes(spec.M, spec.K, use_cos=False)),
        pack_fp4_codes(_trig_fp4_codes(spec.N, spec.K, use_cos=True)),
        rng.integers(124, 131, (spec.M, groups), dtype=np.uint8),
        rng.integers(124, 131, (groups, spec.N), dtype=np.uint8),
    )


def _bench_one(
    variant: str,
    *,
    m: int,
    n: int,
    k: int,
    geometry: dict,
    alpha: float | None,
    beta: float,
    check_betas: list[float],
    verify: str,
    samples: int,
    warmup: int,
    iters: int,
    batch: int,
    timing_inputs: str,
    host_only: bool,
    emit_dir: Path | None,
) -> dict:
    import ml_dtypes

    spec = variant_spec(variant, m, n, k, **geometry)
    ok, reason = is_valid_spec(spec)
    if not ok:
        return _row(variant, check=f"INVALID:{reason}")

    if host_only:
        arch = "gfx1250"
    else:
        from ....runtime.hip_module import get_device_arch

        arch = get_device_arch(0)
        if arch != "gfx1250":
            raise RuntimeError(f"visible HIP device 0 must be gfx1250, got {arch!r}")

    artifact = compile_kernel(
        build_mxfp4_tdm_gemm(spec, arch=arch),
        arch=arch,
        backend="python",
    )
    if emit_dir is not None:
        emit_dir.mkdir(parents=True, exist_ok=True)
        (emit_dir / f"{artifact.kernel_name}.hsaco").write_bytes(artifact.hsaco)
        if artifact.llvm_text:
            (emit_dir / f"{artifact.kernel_name}.ll").write_text(artifact.llvm_text)
    resources = _resources(artifact.hsaco, emit_dir, artifact.kernel_name)
    if host_only:
        return _row(variant, kernel=artifact.kernel_name, check="HOST", **resources)

    from ....runtime.hip_module import Runtime

    alpha32 = np.float32(1.0 / max(spec.K, 256) if alpha is None else alpha)
    inputs = make_inputs(spec)
    out = np.zeros((spec.M, spec.N), dtype=ml_dtypes.float8_e4m3fn)
    runtime = Runtime()
    module = runtime.load_module(artifact.hsaco)
    allocations = []
    try:
        function = module.get_function(artifact.kernel_name)
        for array in (*kernel_inputs(spec, inputs), out):
            pointer = runtime.alloc(array.nbytes)
            allocations.append(pointer)
            runtime.memcpy_h2d(pointer, _u8_buffer(array), array.nbytes)
        grid = mxfp4_tdm_gemm_grid(spec)
        block = (spec.block_size, 1, 1)
        cluster = spec.cluster_dim

        def _arguments(beta_value: float, alpha_value: float = alpha32) -> bytes:
            return struct.pack(
                "<QQQQQQiiiff",
                *allocations,
                spec.M,
                spec.N,
                spec.K,
                float(alpha_value),
                float(beta_value),
            )

        def _launch(arguments: bytes) -> None:
            runtime.launch(function, grid, block, arguments, cluster=cluster)

        failed = []
        for beta_value in check_betas:
            _launch(_arguments(beta_value))
            runtime.sync()
            host = (ctypes.c_uint8 * int(out.nbytes))()
            runtime.memcpy_d2h(host, allocations[-1], out.nbytes)
            got = np.frombuffer(bytes(host), dtype=np.uint8).reshape(spec.M, spec.N)
            if not _check(
                spec,
                got,
                inputs,
                alpha=alpha32,
                beta=np.float32(beta_value),
                verify=verify,
                samples=samples,
            ):
                failed.append(f"beta={beta_value:g}")
        if failed:
            return _row(
                variant,
                kernel=artifact.kernel_name,
                check="FAIL:" + ",".join(failed),
                **resources,
            )

        timed = _arguments(beta)
        if timing_inputs == "tensile":
            timing = kernel_inputs(spec, (*tensile_timing_inputs(spec), inputs[4]))
            for pointer, array in zip(allocations, timing[:4]):
                runtime.memcpy_h2d(pointer, _u8_buffer(array), array.nbytes)
            timed = _arguments(beta, alpha_value=1.0)
        for _ in range(max(warmup, 1)):
            _launch(timed)
        runtime.sync()

        batched_us = ""
        if batch > 0:
            begin = time.perf_counter()
            for _ in range(batch):
                _launch(timed)
            runtime.sync()
            batched_us = (time.perf_counter() - begin) * 1.0e6 / batch

        times: list[float] = []
        for _ in range(iters):
            start = runtime.event()
            end = runtime.event()
            start.record()
            _launch(timed)
            end.record()
            runtime.sync()
            times.append(start.elapsed_to(end) * 1.0e3)
    finally:
        module.unload()
        for pointer in allocations:
            runtime.free(pointer)

    median_us = statistics.median(times) if times else 0.0
    best_us = batched_us if batched_us != "" else median_us
    flops = 2.0 * spec.M * spec.N * spec.K
    tflops = flops / (best_us * 1.0e-6) / 1.0e12 if best_us else 0.0
    return _row(
        variant,
        kernel=artifact.kernel_name,
        check="PASS",
        batched_us=f"{batched_us:.3f}" if batched_us != "" else "",
        median_us=f"{median_us:.3f}",
        tflops=f"{tflops:.3f}",
        _time=best_us,
        **resources,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="all")
    parser.add_argument(
        "--trials",
        action="store_true",
        help="with --variants all, also run the trial variants",
    )
    parser.add_argument("--shape", choices=("smoke", "yaml"), default="smoke")
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="beta of the timed launches (default 0, as in the YAML run)",
    )
    parser.add_argument(
        "--check-betas",
        default="0,0.5",
        help="comma list of betas checked before timing",
    )
    parser.add_argument("--verify", choices=("full", "sampled"), default=None)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--batch",
        type=int,
        default=2000,
        help="enqueues per sync for the host-timed number (0 disables)",
    )
    parser.add_argument(
        "--timing-inputs",
        choices=("tensile", "check"),
        default="tensile",
        help=(
            "data of the timed launches: the Tensile client's YAML-run inputs "
            "(TrigSin/TrigCos, Random scales, alpha 1) or the check inputs"
        ),
    )
    parser.add_argument(
        "--host-only",
        action="store_true",
        help="compile and report resources only; no device is used",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--emit-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    m, n, k = _YAML_PROBLEM if args.shape == "yaml" else _SMOKE_PROBLEM
    m = args.m if args.m is not None else m
    n = args.n if args.n is not None else n
    k = args.k if args.k is not None else k
    verify = args.verify or ("full" if args.shape == "smoke" else "sampled")
    check_betas = [float(x) for x in args.check_betas.split(",") if x.strip()]

    rows = []
    base_time = None
    for variant in _parse_variants(args.variants, trials=args.trials):
        row = _bench_one(
            variant,
            m=m,
            n=n,
            k=k,
            geometry=dict(YAML_TILE),
            alpha=args.alpha,
            beta=args.beta,
            check_betas=check_betas,
            verify=verify,
            samples=args.samples,
            warmup=args.warmup,
            iters=args.iters,
            batch=args.batch,
            timing_inputs=args.timing_inputs,
            host_only=args.host_only,
            emit_dir=args.emit_dir,
        )
        if variant == BASE_VARIANT and row.get("_time"):
            base_time = row["_time"]
        rows.append(row)
        print("\t".join(str(row[f]) for f in FIELDS), flush=True)

    for row in rows:
        if base_time and row.get("_time"):
            row["speedup_vs_base"] = f"{base_time / row['_time']:.3f}"
        row.pop("_time", None)

    print("\t".join(FIELDS))
    for row in rows:
        print("\t".join(str(row[f]) for f in FIELDS))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out}", flush=True)
    ok_checks = ("PASS", "HOST")
    return 0 if all(r["check"] in ok_checks for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
