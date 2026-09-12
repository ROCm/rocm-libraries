"""Parametrized THROWAWAY toy RMSNorm flyDSL kernel, one HSACO per hidden size N.

This is the M2 "de-risk" kernel: a correct-but-naive RMSNorm (one thread per row,
N baked as a compile-time constexpr) whose ONLY job is to exercise the family-aware
pack machinery -- register one kernel instance per N, and have hipDNN's kernel_match
pick the instance whose baked N equals the graph's hidden size. Once the machinery is
proven, the same pack swaps to FlyDSL's real build_rmsnorm_module HSACOs (bf16,
N in {4096,8192}) with no change to the selection path.

Kernarg ABI (flyDSL 0.3.x): three Tensor args -> (ptr,i32 size) each = 44 bytes:
  out_ptr@0 out_N@8  x_ptr@16 x_N@24  w_ptr@32 w_N@40.
Grid = (rows,1,1), block = (1,1,1); thread `row` normalizes row `row`.
"""
import torch  # noqa: F401  (flyDSL import needs torch present)
from flydsl.compiler.kernel_function import kernel
from flydsl.compiler.jit_function import jit
from flydsl.expr import math, range_constexpr
from flydsl.expr.gpu import block_idx
from flydsl.expr.typing import Tensor

EPS = 1e-6


def build(N: int):
    """Return (kernel, run) for a toy RMSNorm with hidden size N baked in."""

    @kernel
    def rmsnorm(out: Tensor, x: Tensor, w: Tensor):
        row = block_idx.x  # i32 on this path (no cast, matching working vadd)
        base = row * N  # i32 * python-int -> i32
        ss = x[base] * x[base]  # seed accumulator as DSL f32
        for k in range_constexpr(1, N):  # constexpr -> unroll, k is python int
            v = x[base + k]
            ss = ss + v * v
        inv = math.rsqrt(ss * (1.0 / N) + EPS)
        for k in range_constexpr(N):
            out[base + k] = x[base + k] * inv * w[k]

    @jit
    def run(out, x, w, rows, stream):
        rmsnorm(out, x, w).launch(grid=(rows, 1, 1), block=(1, 1, 1), stream=stream)

    return rmsnorm, run
