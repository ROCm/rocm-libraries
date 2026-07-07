# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 Qwen3 fused QK-norm + RoPE kernel.

Qwen3 applies a per-head RMSNorm over ``head_dim`` to the projected Q and K
(``q_norm`` / ``k_norm``, a single learned ``[head_dim]`` weight shared across
heads) and *then* rotary position embedding -- before attention. This is a
Qwen3-specific operator with no prior rocke equivalent on any arch; the
attention kernels expect Q/K already normed + rotary'd.

This kernel fuses both steps for one Q-or-K tensor laid out
``[tokens, num_heads, head_dim]`` (contiguous): per ``(token, head)`` it

  1. RMSNorm over ``head_dim``: ``x * rsqrt(mean(x^2) + eps) * weight[d]``,
  2. RoPE (Qwen "half"/rotate-half layout by default; interleaved supported),

and writes the result to ``x_out`` in the same layout. Launch it twice per layer
(Q with ``q_norm``, K with ``k_norm``); V is a passthrough (no kernel needed).

One thread owns one ``(token, head)`` row (``head_dim`` is unrolled at compile
time), so the RMSNorm reduction is thread-local -- no cross-lane reduce. The op
is small and memory-bound (decode ``tokens`` is tiny); occupancy comes from the
``tokens * num_heads`` thread grid, not per-row parallelism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ...core.ir import BF16, F16, F32, I32, IRBuilder, KernelDef, PtrType, Type

_SUPPORTED_ARCH = "gfx1250"

__all__ = [
    "Qwen3QkNormRopeSpec",
    "build_qwen3_qk_norm_rope",
    "qwen3_qk_norm_rope_grid",
    "qwen3_qk_norm_rope_signature",
]


def _dtype_ir(dtype: str) -> Type:
    if dtype in ("fp16", "f16"):
        return F16
    if dtype == "bf16":
        return BF16
    raise ValueError(f"unsupported dtype {dtype!r} (expected fp16/bf16)")


def _require_supported(arch: str) -> Tuple[bool, str]:
    # The kernel is pure elementwise / reduction math (no WMMA, no wave-size
    # dependence), so it is arch-neutral: it targets gfx1250 but also builds and
    # runs correctly on CDNA wave64 (gfx950/gfx942), which lets the numerics be
    # validated locally on gfx950 when a gfx1250 device is unavailable.
    from ...core.arch import ArchTarget

    try:
        ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    return True, "supported"


@dataclass(frozen=True)
class Qwen3QkNormRopeSpec:
    """Fused per-head RMSNorm + RoPE for one Q-or-K tensor.

    ``num_heads`` is the head count of *this* tensor (32 for Q, 4 for K on
    Qwen3-30B-A3B). ``rope_layout`` is ``"half"`` (Qwen / LLaMA-2+ rotate-half;
    pair ``(i, i+H/2)``) or ``"interleaved"`` (pair ``(2i, 2i+1)``). The cos/sin
    tables are f32 ``[max_pos, head_dim/2]`` row-major.
    """

    num_heads: int
    head_dim: int = 64
    dtype: str = "bf16"
    eps: float = 1e-6
    rope_layout: str = "half"
    block_size: int = 64
    name: str = "rocke_gfx1250_qwen3_qk_norm_rope"

    def __post_init__(self) -> None:
        if self.dtype not in ("fp16", "f16", "bf16"):
            raise ValueError("dtype must be fp16/bf16")
        if self.head_dim <= 0 or self.head_dim % 2 != 0:
            raise ValueError("head_dim must be positive and even")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.rope_layout not in ("half", "interleaved"):
            raise ValueError("rope_layout must be 'half' or 'interleaved'")

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

        return kernel_name_join(
            self.name,
            f"h{self.num_heads}",
            f"d{self.head_dim}",
            self.dtype,
            self.rope_layout,
        )

    def _pair(self, i: int) -> Tuple[int, int]:
        if self.rope_layout == "interleaved":
            return (2 * i, 2 * i + 1)
        return (i, i + self.head_dim // 2)


def qwen3_qk_norm_rope_grid(
    num_tokens: int, spec: Qwen3QkNormRopeSpec
) -> Tuple[int, int, int]:
    """1D grid: one thread per ``(token, head)`` row."""
    total = num_tokens * spec.num_heads
    gx = (total + spec.block_size - 1) // spec.block_size
    return (gx, 1, 1)


def qwen3_qk_norm_rope_signature(spec: Qwen3QkNormRopeSpec):
    from ...helpers.spec import SignatureBuilder

    return (
        SignatureBuilder()
        .ptr("x_in", spec.dtype if spec.dtype != "f16" else "fp16")
        .ptr("weight", "f32")
        .ptr("cos", "f32")
        .ptr("sin", "f32")
        .ptr("positions", "i32")
        .ptr("x_out", spec.dtype if spec.dtype != "f16" else "fp16")
        .scalar("num_tokens", "i32")
        .build()
    )


def build_qwen3_qk_norm_rope(
    spec: Qwen3QkNormRopeSpec, *, arch: str = _SUPPORTED_ARCH
) -> KernelDef:
    ok, reason = _require_supported(arch)
    if not ok:
        raise NotImplementedError(reason)

    dt = _dtype_ir(spec.dtype)
    H = spec.head_dim
    half = H // 2
    nh = spec.num_heads
    bs = spec.block_size

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = bs

    x_in = b.param("x_in", PtrType(dt, "global"), noalias=True, readonly=True, align=16)
    weight = b.param(
        "weight", PtrType(F32, "global"), noalias=True, readonly=True, align=16
    )
    cos = b.param("cos", PtrType(F32, "global"), noalias=True, readonly=True, align=16)
    sin = b.param("sin", PtrType(F32, "global"), noalias=True, readonly=True, align=16)
    positions = b.param(
        "positions", PtrType(I32, "global"), noalias=True, readonly=True, align=16
    )
    x_out = b.param(
        "x_out", PtrType(dt, "global"), noalias=True, writeonly=True, align=16
    )
    num_tokens = b.param("num_tokens", I32)

    c_nh = b.const_i32(nh)
    c_H = b.const_i32(H)
    inv_H = b.const_f32(1.0 / H)
    c_eps = b.const_f32(spec.eps)
    c_half = b.const_i32(half)

    tid = b.add(b.mul(b.block_id_x(), b.const_i32(bs)), b.thread_id_x())
    total = b.mul(num_tokens, c_nh)

    with b.scf_if(b.cmp_lt(tid, total)):
        token = b.div(tid, c_nh)
        row_base = b.mul(tid, c_H)
        pos = b.global_load_i32(positions, token)

        # ---- load row + RMS over head_dim (thread-local reduction) ----
        xs = []
        ss = b.const_f32(0.0)
        for d in range(H):
            xd = b.cast_to_f32(
                b.global_load(x_in, b.add(row_base, b.const_i32(d)), dt, align=2)
            )
            xs.append(xd)
            ss = b.fadd(ss, b.fmul(xd, xd))
        inv = b.rsqrt(b.fadd(b.fmul(ss, inv_H), c_eps))

        # ---- normalize: xn[d] = x[d] * inv * weight[d] ----
        xn = []
        for d in range(H):
            wd = b.global_load(weight, b.const_i32(d), F32, align=4)
            xn.append(b.fmul(b.fmul(xs[d], inv), wd))

        # ---- RoPE (half / interleaved) ----
        out = [None] * H
        trig_base = b.mul(pos, c_half)
        for i in range(half):
            lo_i, hi_i = spec._pair(i)
            c = b.global_load(cos, b.add(trig_base, b.const_i32(i)), F32, align=4)
            s = b.global_load(sin, b.add(trig_base, b.const_i32(i)), F32, align=4)
            lo = xn[lo_i]
            hi = xn[hi_i]
            out[lo_i] = b.fsub(b.fmul(lo, c), b.fmul(hi, s))
            out[hi_i] = b.fadd(b.fmul(lo, s), b.fmul(hi, c))

        for d in range(H):
            b.global_store(
                x_out,
                b.add(row_base, b.const_i32(d)),
                b.cast_f32_to(out[d], dt),
                align=2,
            )

    return b.kernel
