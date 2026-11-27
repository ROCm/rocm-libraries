# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generic GPU parity harness for non-attention CK Tile counterparts.

This is the small-op analogue of the attention parity harnesses at
``rocke/library/builders/<arch>/attention/prefill/parity_unified_attention.py``.
It runs the rocKE kernels for elementwise, layernorm2d, rmsnorm2d, reduce2d,
transpose2d, batched GEMM, and grouped GEMM against numpy reference
implementations and reports::

  - max_abs error vs reference
  - rocKE kernel wall-time (HIP event timing, identical to
    :func:`rocke.runtime.launcher.time_launches`)

Inputs are fp16 normal draws from ``np.random.default_rng(--seed)``; each
reference is computed from the stored fp16 values and rounded to fp16 the
way the kernel stores its output.

Usage::

    PYTHONPATH=python python \\
        python/rocke/examples/ck_tile_parity.py [--op all]

The harness exits non-zero if any op's ``max_abs`` exceeds the per-op
tolerance, so it doubles as a smoke gate for CI.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from rocke.core.arch import ArchTarget  # noqa: E402
from rocke.core.lower_llvm import lower_kernel_to_llvm  # noqa: E402
from rocke.examples.common.parity_support import (
    device_values,
    draw,
    round_to_f16,
    values,
)
from rocke.numeric.compare import max_abs_error
from rocke.runtime.comgr import build_hsaco_from_llvm_ir  # noqa: E402
from rocke.runtime.device_array import DeviceArray
from rocke.runtime.hip_module import get_device_arch  # noqa: E402
from rocke.runtime.launcher import (  # noqa: E402
    KernelLauncher,
    LaunchConfig,
    synchronize_and_release,
    time_launches,
)


# Codegen arch for the whole harness. This is a *launch* harness, so it
# defaults to the running device's arch (falling back to gfx950); ``main``
# overrides it from ``--arch``.
_ARCH = "gfx950"


def _build_hsaco(kernel) -> bytes:
    """Lower + assemble a kernel for the harness-selected arch (``_ARCH``)."""
    isa = ArchTarget.from_gfx(_ARCH).isa_triple
    hsaco, _ = build_hsaco_from_llvm_ir(
        lower_kernel_to_llvm(kernel, arch=_ARCH), isa=isa
    )
    return hsaco


def _gemm_warp_tile_k(default: int = 16) -> int:
    """Largest legal K for an fp16 32x32 warp tile on the selected arch.

    gfx950 (CDNA4) carries 32x32x16; gfx942 (CDNA3) only 32x32x8. Asking the
    catalog keeps gfx950 output unchanged while avoiding the unselectable wide
    atom (a hard comgr crash) on gfx942.
    """
    op = ArchTarget.from_gfx(_ARCH).mma.select_largest_k(
        a_dtype="fp16", b_dtype="fp16", c_dtype="fp32", m=32, n=32
    )
    return op.k if op is not None else default


# ---------------------------------------------------------------------
# Per-op tolerance gates. These mirror what we observed in the smoke
# runs above; any regression that crosses them indicates a real bug,
# not f16/bf16 rounding noise.
# ---------------------------------------------------------------------

TOL = {
    "elementwise.linear": 0.0,  # add/sub/mul/max/min: f32 then cast - bit-exact
    "elementwise.unary": 0.0,  # copy/neg/abs/relu: bit-exact
    "elementwise.gelu_silu": 2e-4,  # ~1 ULP for f16 in [-2,2]
    "elementwise.exp2": 0.0,  # bit-exact for our input range
    # ``layernorm2d`` and ``rmsnorm2d`` share the same f16 LSB envelope:
    # at output magnitude ~1.0 the ULP is 2^-10 ~ 9.77e-4, and the LDS-
    # tree reduction order in f32 followed by f32->f16 round-down can
    # legitimately swing ~4 ULP relative to a pairwise reduction.
    # The previous 2.5e-3 gate (~ 2.5 ULP) was tight enough to oscillate
    # across fresh-randn runs (verification worker observed 1.95e-3 -> 3.91e-3
    # back-to-back). Widening to 5e-3 (~ 5 ULP) gates real regressions
    # (these would jump > 8 ULP) while ignoring inherent f16 reduction-
    # order slop.
    "layernorm2d": 5e-3,
    "rmsnorm2d": 5e-3,
    "reduce2d": 1.5e-3,  # ~1 f16 ULP near reduce magnitudes around 1.0.
    # The LDS-tree reduction order is not identical to a
    # pairwise reduction, so f32 -> f16 rounding can differ by
    # one ULP on the final cast. Catches real failures while
    # ignoring this inherent reduction-order slop.
    "transpose2d": 0.0,  # pure data movement
    "batched_gemm": 7e-2,  # standard f16 GEMM noise for K up to 256
    "grouped_gemm": 7e-2,
}


@dataclass
class OpReport:
    name: str
    max_abs: float
    ck_ms: float
    ok: bool


# ---------------------------------------------------------------------
# Op runners
# ---------------------------------------------------------------------


def _bench(fn: Callable[[], None], *, warmup: int = 3, iters: int = 30) -> float:
    return time_launches(fn, warmup=warmup, iters=iters) * 1e-3  # to seconds


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def run_elementwise(rng: np.random.Generator, N: int) -> list[OpReport]:
    from rocke.instances import (
        ElementwiseSpec,
        build_elementwise,
        elementwise_grid,
        elementwise_signature,
    )

    reports: list[OpReport] = []
    A_host, B_host = draw(rng, N), draw(rng, N)
    A, B = DeviceArray(A_host), DeviceArray(B_host)
    a, b = values(A_host, "fp16"), values(B_host, "fp16")
    # The binary ops compute in f32 and cast to fp16. f32 carries at least
    # 2*11+2 significand bits, so that double rounding of an add, subtract or
    # multiply of fp16 values equals rounding the exact (float64) result.
    cases = [
        ("add", "binary", "elementwise.linear", lambda: a + b),
        ("sub", "binary", "elementwise.linear", lambda: a - b),
        ("mul", "binary", "elementwise.linear", lambda: a * b),
        ("max", "binary", "elementwise.linear", lambda: np.maximum(a, b)),
        ("min", "binary", "elementwise.linear", lambda: np.minimum(a, b)),
        ("copy", "unary", "elementwise.unary", lambda: a),
        ("neg", "unary", "elementwise.unary", lambda: -a),
        ("abs", "unary", "elementwise.unary", lambda: np.abs(a)),
        ("relu", "unary", "elementwise.unary", lambda: np.maximum(a, 0.0)),
        ("silu", "unary", "elementwise.gelu_silu", lambda: _silu(a)),
        ("gelu_tanh", "unary", "elementwise.gelu_silu", lambda: _gelu_tanh(a)),
        ("exp2", "unary", "elementwise.exp2", lambda: np.exp2(a)),
    ]
    for op, kind, tol_key, ref_fn in cases:
        spec = ElementwiseSpec(op=op)
        k = build_elementwise(spec)
        hsaco = _build_hsaco(k)
        launcher = KernelLauncher(
            hsaco=hsaco, kernel_name=k.name, signature=elementwise_signature(spec)
        )
        C = DeviceArray.zeros((N,), "fp16")
        grid = elementwise_grid(N, spec)
        vals = {"A": A, "C": C, "N": N}
        if kind == "binary":
            vals["B"] = B
        cfg = LaunchConfig(grid=grid, block=(spec.block_size, 1, 1))
        launcher(vals, config=cfg)
        synchronize_and_release()
        ma = max_abs_error(device_values(C), round_to_f16(ref_fn()))
        ok = ma <= TOL[tol_key]
        ck_s = _bench(lambda: launcher(vals, config=cfg))
        reports.append(
            OpReport(name=f"elementwise.{op}", max_abs=ma, ck_ms=ck_s, ok=ok)
        )
    return reports


def run_layernorm(rng: np.random.Generator, M: int, N: int) -> list[OpReport]:
    from rocke.instances import (
        LayerNorm2DSpec,
        build_layernorm2d,
        layernorm2d_grid,
        layernorm2d_signature,
    )

    spec = LayerNorm2DSpec(n_per_block=N, block_size=256, vec=8)
    k = build_layernorm2d(spec)
    hsaco = _build_hsaco(k)
    launcher = KernelLauncher(
        hsaco=hsaco, kernel_name=k.name, signature=layernorm2d_signature(spec)
    )
    X_host = draw(rng, M, N)
    G_host = draw(rng, N, scale=0.1, shift=1.0)
    B_host = draw(rng, N, scale=0.1)
    X, G, B = DeviceArray(X_host), DeviceArray(G_host), DeviceArray(B_host)
    Y = DeviceArray.zeros((M, N), "fp16")
    eps = 1e-5
    grid = layernorm2d_grid(M, spec)
    vals = {"X": X, "Gamma": G, "Beta": B, "Y": Y, "M": M, "N": N, "eps": eps}
    cfg = LaunchConfig(grid=grid, block=(spec.block_size, 1, 1))
    launcher(vals, config=cfg)
    synchronize_and_release()
    x = values(X_host, "fp16")
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    ref = (x - mean) / np.sqrt(var + eps) * values(G_host, "fp16") + values(
        B_host, "fp16"
    )
    ma = max_abs_error(device_values(Y), round_to_f16(ref))
    ok = ma <= TOL["layernorm2d"]
    ck_s = _bench(lambda: launcher(vals, config=cfg))
    return [OpReport(f"layernorm2d.{M}x{N}", ma, ck_s, ok)]


def run_rmsnorm(rng: np.random.Generator, M: int, N: int) -> list[OpReport]:
    from rocke.instances import (
        RMSNorm2DSpec,
        build_rmsnorm2d,
        rmsnorm2d_grid,
        rmsnorm2d_signature,
    )

    spec = RMSNorm2DSpec(n_per_block=N, block_size=256, vec=8)
    k = build_rmsnorm2d(spec)
    hsaco = _build_hsaco(k)
    launcher = KernelLauncher(
        hsaco=hsaco, kernel_name=k.name, signature=rmsnorm2d_signature(spec)
    )
    X_host = draw(rng, M, N)
    G_host = draw(rng, N, scale=0.1, shift=1.0)
    X, G = DeviceArray(X_host), DeviceArray(G_host)
    Y = DeviceArray.zeros((M, N), "fp16")
    eps = 1e-5
    grid = rmsnorm2d_grid(M, spec)
    vals = {"X": X, "Gamma": G, "Y": Y, "M": M, "N": N, "eps": eps}
    cfg = LaunchConfig(grid=grid, block=(spec.block_size, 1, 1))
    launcher(vals, config=cfg)
    synchronize_and_release()
    x = values(X_host, "fp16")
    var = (x**2).mean(-1, keepdims=True)
    ref = x / np.sqrt(var + eps) * values(G_host, "fp16")
    ma = max_abs_error(device_values(Y), round_to_f16(ref))
    ok = ma <= TOL["rmsnorm2d"]
    ck_s = _bench(lambda: launcher(vals, config=cfg))
    return [OpReport(f"rmsnorm2d.{M}x{N}", ma, ck_s, ok)]


def run_reduce(rng: np.random.Generator, M: int, N: int) -> list[OpReport]:
    from rocke.instances import (
        Reduce2DSpec,
        build_reduce2d,
        reduce2d_grid,
        reduce2d_signature,
    )

    reports = []
    X_host = draw(rng, M, N, scale=0.1)
    X = DeviceArray(X_host)
    x = values(X_host, "fp16")
    for op, reference in [
        ("sum", x.sum(-1)),
        ("max", x.max(-1)),
        ("mean", x.mean(-1)),
    ]:
        # Thread the running arch's wave size into the spec so the
        # BlockReduce2dSync XOR-butterfly + cross-warp fold matches the
        # hardware wavefront (wave32 on RDNA3/gfx1151, wave64 on CDNA).
        # Defaulting to 64 would do a 6-stage butterfly across lanes that
        # don't exist on wave32 and miscount num_warps -> wrong reduction.
        spec = Reduce2DSpec(
            n_per_block=N,
            op=op,
            block_size=256,
            vec=8,
            wave_size=ArchTarget.from_gfx(_ARCH).wave_size,
        )
        k = build_reduce2d(spec)
        hsaco = _build_hsaco(k)
        launcher = KernelLauncher(
            hsaco=hsaco, kernel_name=k.name, signature=reduce2d_signature(spec)
        )
        Y = DeviceArray.zeros((M,), "fp16")
        grid = reduce2d_grid(M, spec)
        vals = {"X": X, "Y": Y, "M": M, "N": N}
        cfg = LaunchConfig(grid=grid, block=(spec.block_size, 1, 1))
        launcher(vals, config=cfg)
        synchronize_and_release()
        ma = max_abs_error(device_values(Y), round_to_f16(reference))
        ok = ma <= TOL["reduce2d"]
        ck_s = _bench(lambda: launcher(vals, config=cfg))
        reports.append(OpReport(f"reduce.{op}.{M}x{N}", ma, ck_s, ok))
    return reports


def run_transpose(rng: np.random.Generator, H: int, W: int) -> list[OpReport]:
    from rocke.instances import (
        Transpose2DSpec,
        build_transpose2d,
        transpose2d_grid,
        transpose2d_signature,
    )

    spec = Transpose2DSpec(tile_m=64, tile_n=64, vec=8)
    k = build_transpose2d(spec)
    hsaco = _build_hsaco(k)
    launcher = KernelLauncher(
        hsaco=hsaco, kernel_name=k.name, signature=transpose2d_signature(spec)
    )
    X_host = draw(rng, H, W)
    X = DeviceArray(X_host)
    Y = DeviceArray.zeros((W, H), "fp16")
    grid = transpose2d_grid(H, W, spec)
    vals = {"X": X, "Y": Y, "H": H, "W": W}
    cfg = LaunchConfig(grid=grid, block=(spec.block_size, 1, 1))
    launcher(vals, config=cfg)
    synchronize_and_release()
    ma = max_abs_error(device_values(Y), values(X_host, "fp16").T)
    ok = ma <= TOL["transpose2d"]
    ck_s = _bench(lambda: launcher(vals, config=cfg))
    return [OpReport(f"transpose2d.{H}x{W}", ma, ck_s, ok)]


def run_batched_gemm(
    rng: np.random.Generator, B: int, M: int, N: int, K: int
) -> list[OpReport]:
    from rocke.instances import (
        BatchedGemmSpec,
        batched_gemm_grid,
        batched_gemm_signature,
        build_batched_gemm,
    )
    from rocke.instances.common.gemm_universal import TileSpec, TraitSpec

    spec = BatchedGemmSpec(
        name="rocke_bgemm",
        tile=TileSpec(
            tile_m=128,
            tile_n=128,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=_gemm_warp_tile_k(),
        ),
        trait=TraitSpec(pipeline="compv3", epilogue="cshuffle"),
    )
    k = build_batched_gemm(spec)
    hsaco = _build_hsaco(k)
    launcher = KernelLauncher(
        hsaco=hsaco, kernel_name=k.name, signature=batched_gemm_signature(spec)
    )
    A_host, B_host = draw(rng, B, M, K), draw(rng, B, N, K)
    A, Bm = DeviceArray(A_host), DeviceArray(B_host)
    C = DeviceArray.zeros((B, M, N), "fp16")
    grid = batched_gemm_grid(B, M, N, spec)
    vals = {
        "A": A,
        "B": Bm,
        "C": C,
        "M": M,
        "N": N,
        "K": K,
        "stride_a": M * K,
        "stride_b": N * K,
        "stride_c": M * N,
    }
    cfg = LaunchConfig(grid=grid, block=(spec.block_size, 1, 1))
    launcher(vals, config=cfg)
    synchronize_and_release()
    ref = values(A_host, "fp16") @ values(B_host, "fp16").transpose(0, 2, 1)
    ma = max_abs_error(device_values(C), round_to_f16(ref))
    ok = ma <= TOL["batched_gemm"]
    ck_s = _bench(lambda: launcher(vals, config=cfg))
    return [OpReport(f"batched_gemm.B{B}_{M}x{N}x{K}", ma, ck_s, ok)]


def run_grouped_gemm(rng: np.random.Generator) -> list[OpReport]:
    from rocke.instances import (
        GroupedGemmLauncher,
        GroupedGemmSpec,
        build_grouped_gemm,
        grouped_gemm_problems,
    )
    from rocke.instances.common.gemm_universal import TileSpec, TraitSpec

    spec = GroupedGemmSpec(
        name="rocke_ggemm",
        tile=TileSpec(
            tile_m=128,
            tile_n=128,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=_gemm_warp_tile_k(),
        ),
        trait=TraitSpec(pipeline="compv3", epilogue="cshuffle"),
    )
    k = build_grouped_gemm(spec)
    hsaco = _build_hsaco(k)
    ggemm = GroupedGemmLauncher(hsaco=hsaco, spec=spec)
    groups = [
        (256, 512, 128),
        (384, 512, 128),
        (128, 512, 128),
        (256, 512, 128),
        (512, 512, 128),
    ]
    A_hosts = [draw(rng, M, K) for M, _, K in groups]
    B_hosts = [draw(rng, N, K) for _, N, K in groups]
    A_list = [DeviceArray(a) for a in A_hosts]
    B_list = [DeviceArray(b) for b in B_hosts]
    C_list = [DeviceArray.zeros((M, N), "fp16") for M, N, _ in groups]
    problems = grouped_gemm_problems(A_list, B_list, C_list)
    ggemm(problems)
    synchronize_and_release()
    max_abs = max(
        max_abs_error(
            device_values(C),
            round_to_f16(values(a_host, "fp16") @ values(b_host, "fp16").T),
        )
        for C, a_host, b_host in zip(C_list, A_hosts, B_hosts)
    )
    ok = max_abs <= TOL["grouped_gemm"]
    ck_s = _bench(lambda: ggemm(problems))
    return [OpReport(f"grouped_gemm.{len(groups)}", max_abs, ck_s, ok)]


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


_OP_RUNNERS: dict[str, Callable[[np.random.Generator], list[OpReport]]] = {
    "elementwise": lambda rng: run_elementwise(rng, N=1024 * 16),
    "layernorm2d": lambda rng: run_layernorm(rng, M=512, N=4096),
    "rmsnorm2d": lambda rng: run_rmsnorm(rng, M=512, N=4096),
    "reduce2d": lambda rng: run_reduce(rng, M=512, N=4096),
    "transpose2d": lambda rng: run_transpose(rng, H=1024, W=1024),
    "batched_gemm": lambda rng: run_batched_gemm(rng, B=16, M=512, N=512, K=128),
    "grouped_gemm": lambda rng: run_grouped_gemm(rng),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--op",
        default="all",
        help=f"op to run: 'all' or one of {sorted(_OP_RUNNERS)}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducible inputs (0 = deterministic baseline).",
    )
    parser.add_argument(
        "--arch",
        default=None,
        help="gfx target for codegen (default: the running device).",
    )
    args = parser.parse_args()

    device_arch = get_device_arch()
    if not device_arch:
        print("HIP device unavailable; exiting", file=sys.stderr)
        return 1
    global _ARCH
    _ARCH = args.arch or device_arch

    print("device:", device_arch)
    print(f"{'op':40s}  {'max_abs':>10s}  {'ck':>8s}  ok")
    print("-" * 66)

    selected: list[str]
    if args.op == "all":
        selected = list(_OP_RUNNERS)
    elif args.op in _OP_RUNNERS:
        selected = [args.op]
    else:
        print(f"unknown op {args.op!r}", file=sys.stderr)
        return 2

    # MFMA-only cases (batched_gemm / grouped_gemm) lower to MFMA intrinsics
    # that only exist on CDNA (gfx9). On a non-CDNA arch (e.g. gfx1151 RDNA3,
    # which uses WMMA, not MFMA) they are out of scope and would otherwise
    # crash the AMDGPU backend, so skip them. Mirrors the catalog-driven
    # arch gating used for the warp-tile K selection above.
    _MFMA_ONLY = {"batched_gemm", "grouped_gemm"}
    if ArchTarget.from_gfx(_ARCH).family != "cdna":
        skipped = [n for n in selected if n in _MFMA_ONLY]
        if skipped:
            for n in skipped:
                print(f"{n:40s}  SKIP (MFMA-only; {_ARCH} is not a CDNA/MFMA arch)")
            selected = [n for n in selected if n not in _MFMA_ONLY]

    all_ok = True
    for name in selected:
        try:
            # A fresh generator per op from --seed keeps every op's inputs, and
            # so its max_abs against the f16 ULP gates, stable across runs and
            # independent of which other ops run.
            reports = _OP_RUNNERS[name](np.random.default_rng(args.seed))
        except Exception as e:  # noqa: BLE001
            print(f"{name:40s}  ERROR: {e}")
            all_ok = False
            continue
        for r in reports:
            ck_ms = f"{r.ck_ms * 1e6:>7.1f}us"
            ok = "OK" if r.ok else "FAIL"
            print(f"{r.name:40s}  {r.max_abs:10.3e}  {ck_ms}  {ok}")
            if not r.ok:
                all_ok = False

    return 0 if all_ok else 3


if __name__ == "__main__":
    sys.exit(main())
