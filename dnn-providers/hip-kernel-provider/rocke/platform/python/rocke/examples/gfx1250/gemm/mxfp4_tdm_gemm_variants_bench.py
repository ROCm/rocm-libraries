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
import statistics
import struct
import sys
from pathlib import Path

import numpy as np

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
from .mxfp4_tdm_gemm import _u8_buffer, make_inputs, reference

_YAML_PROBLEM = (4096, 4096, 65536)
_YAML_TILE = {
    "tile_m": 256,
    "tile_n": 256,
    "tile_k": 512,
    "warp_m": 2,
    "warp_n": 2,
}


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
) -> dict:
    import ml_dtypes

    arch = get_device_arch(0)
    if arch != "gfx1250":
        raise RuntimeError(f"visible HIP device 0 must be gfx1250, got {arch!r}")
    spec = variant_spec(variant, m, n, k, **geometry)
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
        for array in (*inputs[:5], out):
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
            }

        for _ in range(iters):
            start = runtime.event()
            end = runtime.event()
            start.record()
            _launch()
            end.record()
            runtime.sync()
            times.append(start.elapsed_to(end))
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
        "_median": median_ms,
    }


def _flag_summary(spec) -> str:
    bits = []
    for name in (
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
    ):
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
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--emit-dir", type=Path, default=None)
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
        )
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
