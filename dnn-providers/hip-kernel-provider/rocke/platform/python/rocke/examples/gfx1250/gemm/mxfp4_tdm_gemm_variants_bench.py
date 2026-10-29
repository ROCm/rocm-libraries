# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Bench harness for cumulative MX FP4 TDM GEMM variants.

Written for real-silicon timing later. On WSL FFM it only checks end-to-end
correctness at a small shape; FFM timings are meaningless.

Absolute timings must not be committed to the repo. Use ``--out`` with a path
outside the tree.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import dataclasses
import statistics
import struct
import sys
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
    variant_names,
    variant_spec,
)
from ....runtime.hip_module import Runtime, get_device_arch
from .mxfp4_tdm_gemm import _u8_buffer, kernel_inputs, make_inputs, reference

_YAML_PROBLEM = (4096, 4096, 65536)
_YAML_TILE = {
    "tile_m": 256,
    "tile_n": 256,
    "tile_k": 512,
    "warp_m": 2,
    "warp_n": 2,
}


# E2M1 magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4, 6; these are the midpoints.
_E2M1_MIDPOINTS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
# Tensile client InitMode::Random for E8: 127 + rand() % 7 - 3.
_TENSILE_E8_RANDOM = (124, 131)
_TRIG_CHUNK_ELEMENTS = 1 << 24


def _fp4_codes_rne(values: np.ndarray) -> np.ndarray:
    """Float32 -> E2M1 codes, round to nearest even, saturating (OCP cvt)."""
    mag = np.abs(values)
    codes = np.searchsorted(_E2M1_MIDPOINTS, mag, side="left").astype(np.uint8)
    tie = mag == _E2M1_MIDPOINTS[np.minimum(codes, len(_E2M1_MIDPOINTS) - 1)]
    codes[tie & (codes % 2 == 1)] += 1
    codes |= np.signbit(values).astype(np.uint8) << 3
    return codes


def _trig_fp4_packed(rows: int, cols: int, *, use_cos: bool) -> np.ndarray:
    """Tensile ``TrigSin``/``TrigCos``: element ``i`` in memory order is
    ``float(sin(i))`` or ``float(cos(i))``, converted to FP4."""
    trig = np.cos if use_cos else np.sin
    packed = np.empty((rows, cols // 2), dtype=np.uint8)
    chunk_rows = max(1, _TRIG_CHUNK_ELEMENTS // cols)
    for r0 in range(0, rows, chunk_rows):
        r1 = min(rows, r0 + chunk_rows)
        index = np.arange(r0 * cols, r1 * cols, dtype=np.float64)
        codes = _fp4_codes_rne(trig(index).astype(np.float32))
        packed[r0:r1] = pack_fp4_codes(codes.reshape(r1 - r0, cols))
    return packed


def tensile_data_inputs(spec, inputs, seed: int = 0):
    """``inputs`` with A/B/scales replaced by the mxf4_tn_maf.yaml client data:
    A = TrigSin, B = TrigCos, MX scales = Random. C is kept."""
    rng = np.random.default_rng(seed)
    groups = spec.K // spec.block_k
    lo, hi = _TENSILE_E8_RANDOM
    return (
        _trig_fp4_packed(spec.M, spec.K, use_cos=False),
        _trig_fp4_packed(spec.N, spec.K, use_cos=True),
        rng.integers(lo, hi, (spec.M, groups), dtype=np.uint8),
        rng.integers(lo, hi, (groups, spec.N), dtype=np.uint8),
        inputs[4],
    )


def _parse_variants(text: str) -> list[str]:
    if text.strip() == "all":
        return variant_names()
    names = [part.strip() for part in text.split(",") if part.strip()]
    if not names:
        raise SystemExit("--variants must be 'all' or a comma list")
    known = set(variant_names())
    for name in names:
        if name not in known:
            raise SystemExit(f"unknown variant {name!r}; known: {sorted(known)}")
    return names


def _sampled_ok(
    got: np.ndarray,
    spec,
    inputs,
    *,
    alpha: np.float32,
    beta: np.float32,
    samples: int,
    seed: int,
) -> bool:
    """Check a fixed-seed random set of output elements against the reference."""
    expected = reference(spec, inputs, alpha=alpha, beta=beta).view(np.uint8)
    rng = np.random.default_rng(seed)
    count = min(samples, spec.M * spec.N)
    flat = rng.choice(spec.M * spec.N, size=count, replace=False)
    rows, cols = np.divmod(flat, spec.N)
    return bool(np.array_equal(got[rows, cols], expected[rows, cols]))


def _bench_one(
    variant: str,
    *,
    m: int,
    n: int,
    k: int,
    geometry: dict,
    verify: str,
    warmup: int,
    iters: int,
    samples: int,
    emit_dir: Path | None,
    extra_flags: tuple[str, ...] = (),
    batch: int = 0,
    data: str = "exact",
) -> dict:
    import ml_dtypes

    arch = get_device_arch(0)
    if arch != "gfx1250":
        raise RuntimeError(f"visible HIP device 0 must be gfx1250, got {arch!r}")
    spec = variant_spec(variant, m, n, k, **geometry)
    if extra_flags:
        spec = dataclasses.replace(spec, **dict.fromkeys(extra_flags, True))
    ok, reason = is_valid_spec(spec)
    if not ok:
        return {
            "variant": variant,
            "kernel": "",
            "flags": "",
            "check": f"INVALID:{reason}",
            "median_ms": "",
            "min_ms": "",
            "tflops": "",
            "ratio_vs_base": "",
            "batch_ms": "",
        }

    alpha32 = np.float32(1.0 / max(spec.K, 256))
    beta32 = np.float32(0.5)
    inputs = make_inputs(spec)
    out = np.zeros((spec.M, spec.N), dtype=ml_dtypes.float8_e4m3fn)
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

    runtime = Runtime()
    module = runtime.load_module(artifact.hsaco)
    allocations = []
    times: list[float] = []
    check = "FAIL"
    try:
        function = module.get_function(artifact.kernel_name)
        for array in (*kernel_inputs(spec, inputs), out):
            pointer = runtime.alloc(array.nbytes)
            allocations.append(pointer)
            runtime.memcpy_h2d(pointer, _u8_buffer(array), array.nbytes)
        arguments = struct.pack(
            "<QQQQQQiiiff",
            *allocations,
            spec.M,
            spec.N,
            spec.K,
            float(alpha32),
            float(beta32),
        )
        grid = mxfp4_tdm_gemm_grid(spec)
        block = (spec.block_size, 1, 1)
        cluster = spec.cluster_dim

        def _launch() -> None:
            runtime.launch(function, grid, block, arguments, cluster=cluster)

        for _ in range(max(warmup, 1)):
            _launch()
        runtime.sync()

        host = (ctypes.c_uint8 * int(out.nbytes))()
        runtime.memcpy_d2h(host, allocations[-1], out.nbytes)
        got = np.frombuffer(bytes(host), dtype=np.uint8).reshape(spec.M, spec.N)
        if verify == "full":
            expected = reference(spec, inputs, alpha=alpha32, beta=beta32).view(
                np.uint8
            )
            check = "PASS" if np.array_equal(got, expected) else "FAIL"
        else:
            check = (
                "PASS"
                if _sampled_ok(
                    got,
                    spec,
                    inputs,
                    alpha=alpha32,
                    beta=beta32,
                    samples=samples,
                    seed=0x32F4,
                )
                else "FAIL"
            )
        if check != "PASS":
            return {
                "variant": variant,
                "kernel": artifact.kernel_name,
                "flags": _flag_summary(spec),
                "check": check,
                "median_ms": "",
                "min_ms": "",
                "tflops": "",
                "ratio_vs_base": "",
                "batch_ms": "",
            }

        if data == "tensile":
            # The check above runs on exactly representable data; timing runs
            # on the Tensile client's data, which has no bit-exact reference.
            timed = kernel_inputs(spec, tensile_data_inputs(spec, inputs))
            for pointer, array in zip(allocations, timed):
                runtime.memcpy_h2d(pointer, _u8_buffer(array), array.nbytes)
            for _ in range(max(warmup, 1)):
                _launch()
            runtime.sync()

        for _ in range(iters):
            start = runtime.event()
            end = runtime.event()
            start.record()
            _launch()
            end.record()
            runtime.sync()
            times.append(start.elapsed_to(end))
        # Back-to-back launches between one event pair, like the Tensile
        # client's enqueue loop: launch gaps overlap with execution.
        batch_ms = None
        if batch > 0:
            start = runtime.event()
            end = runtime.event()
            start.record()
            for _ in range(batch):
                _launch()
            end.record()
            runtime.sync()
            batch_ms = start.elapsed_to(end) / batch
    finally:
        module.unload()
        for pointer in allocations:
            runtime.free(pointer)

    median_ms = statistics.median(times)
    min_ms = min(times)
    flops = 2.0 * spec.M * spec.N * spec.K
    tflops = (flops / (median_ms * 1.0e-3)) / 1.0e12 if median_ms > 0 else 0.0
    return {
        "variant": variant,
        "kernel": artifact.kernel_name,
        "flags": _flag_summary(spec),
        "check": check,
        "median_ms": f"{median_ms:.4f}",
        "min_ms": f"{min_ms:.4f}",
        "tflops": f"{tflops:.3f}",
        "ratio_vs_base": "",
        "batch_ms": f"{batch_ms:.4f}" if batch_ms is not None else "",
        "_median": median_ms,
    }


_BOOL_FLAGS = (
    "local_read_prefetch",
    "prefetch_gl2",
    "scale_double_buffer",
    "wave_separated_tdm",
    "scale_tdm",
    "lds_pad",
    "lds_segment_interleave",
    "lds_segment_b_rotate",
    "tdm_plus_lds_buf",
    "init_c_wmma",
    "b_scale_n_major",
    "tdm_multicast",
    "cross_tile_prefetch",
    "cluster_gl2_prefetch",
    "a_scale_lds_pad",
)


def _flag_summary(spec) -> str:
    bits = []
    for name in _BOOL_FLAGS:
        if getattr(spec, name):
            bits.append(name)
    if spec.cluster_dim is not None:
        bits.append(f"cluster{spec.cluster_dim}")
    return "|".join(bits) if bits else "(none)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="all")
    parser.add_argument("--shape", choices=("smoke", "yaml"), default="smoke")
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--tile-m", type=int, default=None)
    parser.add_argument("--tile-n", type=int, default=None)
    parser.add_argument("--tile-k", type=int, default=None)
    parser.add_argument("--warp-m", type=int, default=2)
    parser.add_argument("--warp-n", type=int, default=2)
    parser.add_argument("--verify", choices=("full", "sampled"), default="sampled")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--batch",
        type=int,
        default=0,
        help="also time this many back-to-back launches with one event pair",
    )
    parser.add_argument(
        "--data",
        choices=("exact", "tensile"),
        default="exact",
        help=(
            "timing inputs: 'exact' = the verification data (FP4 0/1, scales "
            "125-128); 'tensile' = mxf4_tn_maf.yaml client data (A TrigSin, "
            "B TrigCos, scales Random 124-130), uploaded after the check"
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--emit-dir", type=Path, default=None)
    parser.add_argument(
        "--with-flag",
        action="append",
        default=[],
        choices=_BOOL_FLAGS,
        help="turn on one extra boolean spec flag for every variant (repeatable)",
    )
    args = parser.parse_args(argv)

    if args.shape == "yaml":
        m, n, k = _YAML_PROBLEM
        geometry = dict(_YAML_TILE)
    else:
        m, n, k = 128, 64, 256
        geometry = {
            "tile_m": 128,
            "tile_n": 64,
            "tile_k": 128,
            "warp_m": 2,
            "warp_n": 2,
        }
    if args.m is not None:
        m = args.m
    if args.n is not None:
        n = args.n
    if args.k is not None:
        k = args.k
    for key, attr in (
        ("tile_m", args.tile_m),
        ("tile_n", args.tile_n),
        ("tile_k", args.tile_k),
    ):
        if attr is not None:
            geometry[key] = attr
    geometry["warp_m"] = args.warp_m
    geometry["warp_n"] = args.warp_n

    if args.shape == "yaml" and args.verify == "full" and max(m, n, k) > 1024:
        print(
            "warning: full verify on large yaml shapes is expensive",
            file=sys.stderr,
        )

    rows = []
    base_median = None
    for variant in _parse_variants(args.variants):
        row = _bench_one(
            variant,
            m=m,
            n=n,
            k=k,
            geometry=geometry,
            verify=args.verify,
            warmup=args.warmup,
            iters=args.iters,
            samples=args.samples,
            emit_dir=args.emit_dir,
            extra_flags=tuple(args.with_flag),
            batch=args.batch,
            data=args.data,
        )
        row["data"] = args.data
        if variant == BASE_VARIANT and "_median" in row:
            base_median = row["_median"]
        rows.append(row)

    for row in rows:
        if base_median and row.get("_median"):
            row["ratio_vs_base"] = f"{row['_median'] / base_median:.3f}"
        row.pop("_median", None)

    fields = [
        "variant",
        "kernel",
        "flags",
        "check",
        "median_ms",
        "min_ms",
        "tflops",
        "ratio_vs_base",
        "batch_ms",
        "data",
    ]
    print("\t".join(fields))
    for row in rows:
        print("\t".join(str(row[f]) for f in fields))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out}", flush=True)
    return 0 if all(r["check"] == "PASS" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
