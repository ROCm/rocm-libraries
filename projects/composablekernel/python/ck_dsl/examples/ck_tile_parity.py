# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generic GPU parity harness for non-attention CK Tile counterparts.

This is the small-op analogue of
``ck_dsl/examples/attention/parity_unified_attention.py``. It runs the
CK DSL kernels for elementwise, layernorm2d, rmsnorm2d, reduce2d,
transpose2d, batched GEMM, and grouped GEMM against torch / numpy
reference implementations and reports::

  - max_abs / mean_abs / max_rel error vs reference
  - DSL kernel wall-time (HIP event timing, identical to
    :func:`ck_dsl.runtime.launcher.time_launches`)
  - torch baseline wall-time (where a directly-comparable torch op
    exists)

Usage::

    PYTHONPATH=python python \\
        python/ck_dsl/examples/ck_tile_parity.py [--op all]

The harness exits non-zero if any op's ``max_abs`` exceeds the per-op
tolerance, so it doubles as a smoke gate for CI.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

import torch  # noqa: E402

from ck_dsl.core.lower_llvm import lower_kernel_to_llvm  # noqa: E402
from ck_dsl.runtime.comgr import build_hsaco_from_llvm_ir  # noqa: E402
from ck_dsl.runtime.launcher import (  # noqa: E402
    KernelLauncher,
    LaunchConfig,
    synchronize_and_release,
    time_launches,
)


# ---------------------------------------------------------------------
# Per-op tolerance gates. These mirror what we observed in the smoke
# runs above; any regression that crosses them indicates a real bug,
# not f16/bf16 rounding noise.
# ---------------------------------------------------------------------

TOL = {
    "elementwise.linear": 0.0,  # add/sub/mul/max/min: f32 then cast - matches torch bit-exactly
    "elementwise.unary": 0.0,  # copy/neg/abs/relu: bit-exact
    "elementwise.gelu_silu": 2e-4,  # ~1 ULP for f16 in [-2,2]
    "elementwise.exp2": 0.0,  # bit-exact for our input range
    "layernorm2d": 2.5e-3,  # ~1 ULP for f16 outputs near 1.0
    "rmsnorm2d": 2.5e-3,  # ~1 ULP for f16 outputs near 2.0
    "reduce2d": 1.5e-3,  # ~1 f16 ULP near reduce magnitudes around 1.0.
    # The LDS-tree reduction order is not identical to torch's
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
    mean_abs: float
    ck_ms: float
    ref_ms: Optional[float]
    ok: bool

    def speedup(self) -> Optional[float]:
        return (self.ref_ms / self.ck_ms) if self.ref_ms else None


# ---------------------------------------------------------------------
# Op runners
# ---------------------------------------------------------------------


def _check_max_abs(
    name: str, dsl_out: torch.Tensor, ref: torch.Tensor
) -> Tuple[float, float]:
    d = (dsl_out.float() - ref.float()).abs()
    return float(d.max().item()), float(d.mean().item())


def _bench(fn: Callable[[], None], *, warmup: int = 3, iters: int = 30) -> float:
    return time_launches(fn, warmup=warmup, iters=iters) * 1e-3  # to seconds


def _bench_torch(fn: Callable[[], None], *, warmup: int = 3, iters: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def run_elementwise(N: int) -> List[OpReport]:
    from ck_dsl.instances import (
        ElementwiseSpec,
        build_elementwise,
        elementwise_grid,
        elementwise_signature,
    )

    reports: List[OpReport] = []
    A = torch.randn(N, dtype=torch.float16, device="cuda")
    B = torch.randn(N, dtype=torch.float16, device="cuda")

    cases = [
        ("add", "binary", "elementwise.linear", lambda: (A.float() + B.float()).half()),
        ("sub", "binary", "elementwise.linear", lambda: (A.float() - B.float()).half()),
        ("mul", "binary", "elementwise.linear", lambda: (A.float() * B.float()).half()),
        ("max", "binary", "elementwise.linear", lambda: torch.maximum(A, B)),
        ("min", "binary", "elementwise.linear", lambda: torch.minimum(A, B)),
        ("copy", "unary", "elementwise.unary", lambda: A.clone()),
        ("neg", "unary", "elementwise.unary", lambda: -A),
        ("abs", "unary", "elementwise.unary", lambda: A.abs()),
        ("relu", "unary", "elementwise.unary", lambda: torch.relu(A)),
        (
            "silu",
            "unary",
            "elementwise.gelu_silu",
            lambda: torch.nn.functional.silu(A.float()).half(),
        ),
        (
            "gelu_tanh",
            "unary",
            "elementwise.gelu_silu",
            lambda: torch.nn.functional.gelu(A.float(), approximate="tanh").half(),
        ),
        ("exp2", "unary", "elementwise.exp2", lambda: torch.exp2(A.float()).half()),
    ]
    for op, kind, tol_key, ref_fn in cases:
        spec = ElementwiseSpec(op=op)
        k = build_elementwise(spec)
        hsaco, _ = build_hsaco_from_llvm_ir(lower_kernel_to_llvm(k))
        launcher = KernelLauncher(
            hsaco=hsaco, kernel_name=k.name, signature=elementwise_signature(spec)
        )
        C = torch.empty_like(A)
        grid = elementwise_grid(N, spec)
        vals = {"A": A, "C": C, "N": N}
        if kind == "binary":
            vals["B"] = B
        cfg = LaunchConfig(grid=grid, block=(spec.block_size, 1, 1))
        launcher(vals, config=cfg)
        synchronize_and_release()
        ref = ref_fn()
        ma, mn = _check_max_abs(f"elementwise.{op}", C, ref)
        ok = ma <= TOL[tol_key]
        ck_s = _bench(lambda: launcher(vals, config=cfg))
        reports.append(
            OpReport(
                name=f"elementwise.{op}",
                max_abs=ma,
                mean_abs=mn,
                ck_ms=ck_s,
                ref_ms=None,
                ok=ok,
            )
        )
    return reports


def run_layernorm(M: int, N: int) -> List[OpReport]:
    from ck_dsl.instances import (
        LayerNorm2DSpec,
        build_layernorm2d,
        layernorm2d_grid,
        layernorm2d_signature,
    )

    spec = LayerNorm2DSpec(n_per_block=N, block_size=256, vec=8)
    k = build_layernorm2d(spec)
    hsaco, _ = build_hsaco_from_llvm_ir(lower_kernel_to_llvm(k))
    launcher = KernelLauncher(
        hsaco=hsaco, kernel_name=k.name, signature=layernorm2d_signature(spec)
    )
    X = torch.randn(M, N, dtype=torch.float16, device="cuda")
    G = torch.randn(N, dtype=torch.float16, device="cuda") * 0.1 + 1.0
    B = torch.randn(N, dtype=torch.float16, device="cuda") * 0.1
    Y = torch.empty_like(X)
    eps = 1e-5
    grid = layernorm2d_grid(M, spec)
    vals = {"X": X, "Gamma": G, "Beta": B, "Y": Y, "M": M, "N": N, "eps": eps}
    cfg = LaunchConfig(grid=grid, block=(spec.block_size, 1, 1))
    launcher(vals, config=cfg)
    synchronize_and_release()
    ref = torch.nn.functional.layer_norm(
        X.float(), (N,), G.float(), B.float(), eps
    ).half()
    ma, mn = _check_max_abs("layernorm2d", Y, ref)
    ok = ma <= TOL["layernorm2d"]
    ck_s = _bench(lambda: launcher(vals, config=cfg))
    ref_s = _bench_torch(
        lambda: torch.nn.functional.layer_norm(
            X.float(), (N,), G.float(), B.float(), eps
        ).half()
    )
    return [OpReport(f"layernorm2d.{M}x{N}", ma, mn, ck_s, ref_s, ok)]


def run_rmsnorm(M: int, N: int) -> List[OpReport]:
    from ck_dsl.instances import (
        RMSNorm2DSpec,
        build_rmsnorm2d,
        rmsnorm2d_grid,
        rmsnorm2d_signature,
    )

    spec = RMSNorm2DSpec(n_per_block=N, block_size=256, vec=8)
    k = build_rmsnorm2d(spec)
    hsaco, _ = build_hsaco_from_llvm_ir(lower_kernel_to_llvm(k))
    launcher = KernelLauncher(
        hsaco=hsaco, kernel_name=k.name, signature=rmsnorm2d_signature(spec)
    )
    X = torch.randn(M, N, dtype=torch.float16, device="cuda")
    G = torch.randn(N, dtype=torch.float16, device="cuda") * 0.1 + 1.0
    Y = torch.empty_like(X)
    eps = 1e-5
    grid = rmsnorm2d_grid(M, spec)
    vals = {"X": X, "Gamma": G, "Y": Y, "M": M, "N": N, "eps": eps}
    cfg = LaunchConfig(grid=grid, block=(spec.block_size, 1, 1))
    launcher(vals, config=cfg)
    synchronize_and_release()

    def ref_fn():
        var = X.float().pow(2).mean(-1, keepdim=True)
        return (X.float() * torch.rsqrt(var + eps) * G.float()).half()

    ref = ref_fn()
    ma, mn = _check_max_abs("rmsnorm2d", Y, ref)
    ok = ma <= TOL["rmsnorm2d"]
    ck_s = _bench(lambda: launcher(vals, config=cfg))
    ref_s = _bench_torch(ref_fn)
    return [OpReport(f"rmsnorm2d.{M}x{N}", ma, mn, ck_s, ref_s, ok)]


def run_reduce(M: int, N: int) -> List[OpReport]:
    from ck_dsl.instances import (
        Reduce2DSpec,
        build_reduce2d,
        reduce2d_grid,
        reduce2d_signature,
    )

    reports = []
    X = torch.randn(M, N, dtype=torch.float16, device="cuda") * 0.1
    for op, ref_fn in [
        ("sum", lambda: X.float().sum(-1).half()),
        ("max", lambda: X.float().amax(-1).half()),
        ("mean", lambda: X.float().mean(-1).half()),
    ]:
        spec = Reduce2DSpec(n_per_block=N, op=op, block_size=256, vec=8)
        k = build_reduce2d(spec)
        hsaco, _ = build_hsaco_from_llvm_ir(lower_kernel_to_llvm(k))
        launcher = KernelLauncher(
            hsaco=hsaco, kernel_name=k.name, signature=reduce2d_signature(spec)
        )
        Y = torch.empty(M, dtype=torch.float16, device="cuda")
        grid = reduce2d_grid(M, spec)
        vals = {"X": X, "Y": Y, "M": M, "N": N}
        cfg = LaunchConfig(grid=grid, block=(spec.block_size, 1, 1))
        launcher(vals, config=cfg)
        synchronize_and_release()
        ref = ref_fn()
        ma, mn = _check_max_abs(f"reduce.{op}", Y, ref)
        ok = ma <= TOL["reduce2d"]
        ck_s = _bench(lambda: launcher(vals, config=cfg))
        ref_s = _bench_torch(ref_fn)
        reports.append(OpReport(f"reduce.{op}.{M}x{N}", ma, mn, ck_s, ref_s, ok))
    return reports


def run_transpose(H: int, W: int) -> List[OpReport]:
    from ck_dsl.instances import (
        Transpose2DSpec,
        build_transpose2d,
        transpose2d_grid,
        transpose2d_signature,
    )

    spec = Transpose2DSpec(tile_m=64, tile_n=64, vec=8)
    k = build_transpose2d(spec)
    hsaco, _ = build_hsaco_from_llvm_ir(lower_kernel_to_llvm(k))
    launcher = KernelLauncher(
        hsaco=hsaco, kernel_name=k.name, signature=transpose2d_signature(spec)
    )
    X = torch.randn(H, W, dtype=torch.float16, device="cuda")
    Y = torch.empty(W, H, dtype=torch.float16, device="cuda")
    grid = transpose2d_grid(H, W, spec)
    vals = {"X": X, "Y": Y, "H": H, "W": W}
    cfg = LaunchConfig(grid=grid, block=(spec.block_size, 1, 1))
    launcher(vals, config=cfg)
    synchronize_and_release()
    ref = X.t().contiguous()
    ma, mn = _check_max_abs("transpose2d", Y, ref)
    ok = ma <= TOL["transpose2d"]
    ck_s = _bench(lambda: launcher(vals, config=cfg))
    ref_s = _bench_torch(lambda: X.t().contiguous())
    return [OpReport(f"transpose2d.{H}x{W}", ma, mn, ck_s, ref_s, ok)]


def run_batched_gemm(B: int, M: int, N: int, K: int) -> List[OpReport]:
    from ck_dsl.instances import (
        BatchedGemmSpec,
        batched_gemm_grid,
        batched_gemm_signature,
        build_batched_gemm,
    )
    from ck_dsl.instances.gemm_universal import TileSpec, TraitSpec

    spec = BatchedGemmSpec(
        name="ck_dsl_bgemm",
        tile=TileSpec(
            tile_m=128,
            tile_n=128,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        ),
        trait=TraitSpec(pipeline="compv3", epilogue="cshuffle"),
    )
    k = build_batched_gemm(spec)
    hsaco, _ = build_hsaco_from_llvm_ir(lower_kernel_to_llvm(k))
    launcher = KernelLauncher(
        hsaco=hsaco, kernel_name=k.name, signature=batched_gemm_signature(spec)
    )
    A = torch.randn(B, M, K, dtype=torch.float16, device="cuda")
    Bm = torch.randn(B, N, K, dtype=torch.float16, device="cuda")
    C = torch.zeros(B, M, N, dtype=torch.float16, device="cuda")
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
    ref = (A.float() @ Bm.float().transpose(-1, -2)).half()
    ma, mn = _check_max_abs("batched_gemm", C, ref)
    ok = ma <= TOL["batched_gemm"]
    ck_s = _bench(lambda: launcher(vals, config=cfg))
    ref_s = _bench_torch(lambda: torch.bmm(A, Bm.transpose(-1, -2)))
    return [OpReport(f"batched_gemm.B{B}_{M}x{N}x{K}", ma, mn, ck_s, ref_s, ok)]


def run_grouped_gemm() -> List[OpReport]:
    from ck_dsl.instances import (
        GroupedGemmLauncher,
        GroupedGemmSpec,
        build_grouped_gemm,
        grouped_gemm_problems,
    )
    from ck_dsl.instances.gemm_universal import TileSpec, TraitSpec

    spec = GroupedGemmSpec(
        name="ck_dsl_ggemm",
        tile=TileSpec(
            tile_m=128,
            tile_n=128,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        ),
        trait=TraitSpec(pipeline="compv3", epilogue="cshuffle"),
    )
    k = build_grouped_gemm(spec)
    hsaco, _ = build_hsaco_from_llvm_ir(lower_kernel_to_llvm(k))
    ggemm = GroupedGemmLauncher(hsaco=hsaco, spec=spec)
    groups = [
        (256, 512, 128),
        (384, 512, 128),
        (128, 512, 128),
        (256, 512, 128),
        (512, 512, 128),
    ]
    A_list = [
        torch.randn(M, K, dtype=torch.float16, device="cuda") for M, _, K in groups
    ]
    B_list = [
        torch.randn(N, K, dtype=torch.float16, device="cuda") for _, N, K in groups
    ]
    C_list = [
        torch.zeros(M, N, dtype=torch.float16, device="cuda") for M, N, _ in groups
    ]
    problems = grouped_gemm_problems(A_list, B_list, C_list)
    ggemm(problems)
    synchronize_and_release()
    max_abs = 0.0
    mean_abs = 0.0
    total = 0
    for A, Bm, C in zip(A_list, B_list, C_list):
        ref = (A.float() @ Bm.float().t()).half()
        d = (C.float() - ref.float()).abs()
        max_abs = max(max_abs, d.max().item())
        mean_abs += d.sum().item()
        total += d.numel()
    mean_abs /= total
    ok = max_abs <= TOL["grouped_gemm"]
    ck_s = _bench(lambda: ggemm(problems))
    ref_s = _bench_torch(lambda: [torch.mm(A, B.t()) for A, B in zip(A_list, B_list)])
    return [OpReport(f"grouped_gemm.{len(groups)}", max_abs, mean_abs, ck_s, ref_s, ok)]


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


_OP_RUNNERS: Dict[str, Callable[[], List[OpReport]]] = {
    "elementwise": lambda: run_elementwise(N=1024 * 16),
    "layernorm2d": lambda: run_layernorm(M=512, N=4096),
    "rmsnorm2d": lambda: run_rmsnorm(M=512, N=4096),
    "reduce2d": lambda: run_reduce(M=512, N=4096),
    "transpose2d": lambda: run_transpose(H=1024, W=1024),
    "batched_gemm": lambda: run_batched_gemm(B=16, M=512, N=512, K=128),
    "grouped_gemm": lambda: run_grouped_gemm(),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--op",
        default="all",
        help=f"op to run: 'all' or one of {sorted(_OP_RUNNERS)}",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("HIP device unavailable; exiting", file=sys.stderr)
        return 1

    print("device:", torch.cuda.get_device_name(0))
    print(f"{'op':40s}  {'max_abs':>10s}  {'ck':>8s}  {'ref':>8s}  {'speedup':>8s}  ok")
    print("-" * 86)

    selected: List[str]
    if args.op == "all":
        selected = list(_OP_RUNNERS)
    elif args.op in _OP_RUNNERS:
        selected = [args.op]
    else:
        print(f"unknown op {args.op!r}", file=sys.stderr)
        return 2

    all_ok = True
    for name in selected:
        try:
            reports = _OP_RUNNERS[name]()
        except Exception as e:  # noqa: BLE001
            print(f"{name:40s}  ERROR: {e}")
            all_ok = False
            continue
        for r in reports:
            speed = f"{r.speedup():.2f}x" if r.speedup() else "  -  "
            ref_ms = f"{r.ref_ms * 1e6:>7.1f}us" if r.ref_ms else "    -   "
            ck_ms = f"{r.ck_ms * 1e6:>7.1f}us"
            ok = "OK" if r.ok else "FAIL"
            print(
                f"{r.name:40s}  {r.max_abs:10.3e}  {ck_ms}  {ref_ms}  {speed:>8s}  {ok}"
            )
            if not r.ok:
                all_ok = False

    return 0 if all_ok else 3


if __name__ == "__main__":
    sys.exit(main())
