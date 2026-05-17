# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fused add + RMSNorm + round-to-quant kernel.

DSL counterpart of CK Tile's
``example/ck_tile/11_add_rmsnorm2d_rdquant``. For two ``(M, N)`` input
tensors ``A`` and ``B`` and an ``(N,)`` ``Gamma`` per-channel scale,
the kernel produces the residual sum, the RMSNorm output, and the
quantised RMSNorm output in one pass over global memory::

 x[m, n] = a[m, n] + b[m, n]
 sum_sq[m] = sum_n(x[m, n] ^ 2)
 inv_rms[m] = 1 / sqrt(sum_sq[m] / N + eps_rms)
 y[m, n] = x[m, n] * inv_rms[m] * gamma[n]
 yscale[m] = max(amax_y, eps_q) / quant_max
 qy[m, n] = quantise(y[m, n], 1 / yscale[m])

with ``amax_y = inv_rms * max_n(|x * gamma|)``.

Two output paths:

* ``QY`` : quantised RMSNorm output (i8 / fp8e4m3 / bf8e5m2)
* ``YScale`` : per-row dynamic quant scale (fp32; optional)
* ``X`` : optional residual write-back of ``a + b`` (saves the next
 layer from recomputing the add when it needs the residual stream).

Implementation:

* One CTA per row (same as rmsnorm2d).
* Pass 1 streams ``a`` and ``b`` once: computes ``x = a + b``, the
 per-thread ``sum_sq``, and the per-thread ``amax_g`` (the max of
 ``|x * gamma|`` before the inv_rms scale). Caches ``x`` in the
 per-thread f32 register file for pass 2; writes ``x`` back to the
 residual buffer when ``save_residual=True``.
* Two block-LDS reductions (sum + max) feed the per-row constants.
* Pass 2 re-reads gamma (in L1 by now), fuses the
 ``x * inv_rms * gamma`` multiply with the quant cast, and stores
 ``qy`` per element via :func:`ck_dsl.helpers.quantize_scalar_f32`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ..core.ir import F32, I32, IRBuilder, KernelDef, PtrType
from ..helpers.io import io_ir_type, store_scalar_from_f32
from ..helpers.quant import QDType, quant_ir_type, quant_max_abs, quantize_scalar_f32
from ..helpers.reduction import block_lds_reduce
from ..helpers.spec import (
    IOSpecRule,
    SignatureBuilder,
    ceil_div_grid,
    kernel_name_join,
    validate_io,
)
from ..helpers.sweep import sweep_row_chunks
from ..helpers.tensor_view import (
    make_global_view,
    make_lds_view,
    make_naive_tensor_view_packed,
    make_tile_window,
)


DType = Literal["f16", "bf16"]


@dataclass(frozen=True)
class AddRmsnorm2DRdquantSpec:
    """One concrete fused add + RMSNorm + round-quant configuration."""

    n_per_block: int
    dtype: DType = "f16"
    out_dtype: QDType = "i8"
    block_size: int = 256
    vec: int = 4
    save_residual: bool = True  # write x = a + b to ``X``
    save_yscale: bool = True  # write per-row scale to ``YScale``
    wave_size: int = 64
    name: str = "ck_dsl_add_rmsnorm2d_rdquant"

    @property
    def elems_per_thread(self) -> int:
        return self.n_per_block // self.block_size

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.dtype,
            self.out_dtype,
            f"N{self.n_per_block}",
            f"b{self.block_size}",
            f"v{self.vec}",
            flags={"sr": self.save_residual, "ys": self.save_yscale},
        )


def is_valid_spec(spec: AddRmsnorm2DRdquantSpec) -> Tuple[bool, str]:
    if spec.out_dtype not in ("i8", "fp8e4m3", "bf8e5m2"):
        return False, f"unsupported out_dtype {spec.out_dtype!r}"
    return validate_io(
        IOSpecRule(
            dtype=spec.dtype,
            block_size=spec.block_size,
            vec=spec.vec,
            n_per_block=spec.n_per_block,
            max_elems_per_thread=64,
        )
    )


def build_add_rmsnorm2d_rdquant(spec: AddRmsnorm2DRdquantSpec) -> KernelDef:
    """Build the IR for one fused add + RMSNorm + quantise instance.

    Kernel signature (with both optional outputs enabled)::

    (A: ptr<dtype, global>, # (M, N)
    B: ptr<dtype, global>, # (M, N)
    Gamma: ptr<dtype, global>, # (N,)
    X: ptr<dtype, global>, # (M, N) optional residual out (a+b)
    QY: ptr<out_dtype, global>, # (M, N) quantised output
    YScale: ptr<f32, global>, # (M,) optional per-row scale
    M: i32, N: i32,
    eps_rms: f32, # rmsnorm epsilon
    eps_q: f32) # amax clamp epsilon (avoid /0)

    Grid: ``(M, 1, 1)`` — one CTA per row.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid add_rmsnorm2d_rdquant spec: {why}")

    io_ty = io_ir_type(spec.dtype)
    q_ty = quant_ir_type(spec.out_dtype)
    qmax = quant_max_abs(spec.out_dtype)

    BS, VEC, N = spec.block_size, spec.vec, spec.n_per_block

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    A = b.param("A", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Gamma = b.param(
        "Gamma", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    if spec.save_residual:
        X = b.param(
            "X", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16
        )
    QY = b.param("QY", PtrType(q_ty, "global"), noalias=True, writeonly=True, align=16)
    if spec.save_yscale:
        YScale = b.param(
            "YScale", PtrType(F32, "global"), noalias=True, writeonly=True, align=4
        )
    M = b.param("M", I32)  # noqa: F841 — ABI symmetry with CK Tile
    _ = b.param("N", I32)  # noqa: F841 — validated by caller
    eps_rms = b.param("eps_rms", F32)
    eps_q = b.param("eps_q", F32)

    tid = b.thread_id_x()
    row = b.block_id_x()

    a_view = make_naive_tensor_view_packed(A, shape=(1, N), dtype=io_ty)
    b_view = make_naive_tensor_view_packed(Bp, shape=(1, N), dtype=io_ty)
    g_view = make_global_view(Gamma, shape=(N,), dtype=io_ty)
    qy_view = make_naive_tensor_view_packed(QY, shape=(1, N), dtype=q_ty)
    a_tile = make_tile_window(a_view, lengths=(1, N), origin=(row, b.const_i32(0)))
    bt_tile = make_tile_window(b_view, lengths=(1, N), origin=(row, b.const_i32(0)))
    qy_tile = make_tile_window(qy_view, lengths=(1, N), origin=(row, b.const_i32(0)))
    if spec.save_residual:
        x_view = make_naive_tensor_view_packed(X, shape=(1, N), dtype=io_ty)
        x_tile = make_tile_window(x_view, lengths=(1, N), origin=(row, b.const_i32(0)))

    lds = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_red").base

    # Pass 1: stream A (which carries the f32-promoted ``a`` scalars
    # through ``sweep_row_chunks``'s ``x_scalars`` argument); per chunk
    # we manually load ``b`` from ``B`` and ``gamma`` from ``Gamma``
    # to compute ``x = a + b`` and the per-thread reductions.
    s_sq = b.const_f32(0.0)
    s_amax_g = b.const_f32(0.0)
    # The cache holds the per-thread x = a + b f32 scalars for pass 2.
    cached_x: list = []

    def pass1_body(n_off, a_scalars):
        nonlocal s_sq, s_amax_g
        b_scalars = bt_tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC)
        g_scalars = g_view.load_vec_as_f32(b, [n_off], n=VEC)
        chunk_x: list = []
        for i in range(VEC):
            x_i = b.fadd(a_scalars[i], b_scalars[i])
            chunk_x.append(x_i)
            s_sq = b.fadd(s_sq, b.fmul(x_i, x_i))
            xg = b.fmul(x_i, g_scalars[i])
            abs_xg = b.fmax(xg, b.fneg(xg))
            s_amax_g = b.fmax(s_amax_g, abs_xg)
        cached_x.extend(chunk_x)
        if spec.save_residual:
            # Per-chunk residual write-back: pack the f32 ``x`` chunk
            # back to the I/O dtype and store via the standard
            # vectorised write. Pass 2 won't re-read ``X`` (the cache
            # holds the values it needs); the write is purely so the
            # next layer's residual stream can pick up ``a + b``.
            x_tile.store_vec_from_f32(b, b.const_i32(0), n_off, values=chunk_x)

    sweep_row_chunks(
        b,
        a_tile,
        tid=tid,
        block_size=BS,
        vec=VEC,
        elems_per_thread=spec.elems_per_thread,
        body=pass1_body,
        cache=False,  # we manage the cache ourselves because ``x = a+b``
        # is the cached value, not the raw ``a``.
    )

    total_sq = block_lds_reduce(b, s_sq, lds, tid, block_size=BS, combine="sum")
    total_amax_g = block_lds_reduce(b, s_amax_g, lds, tid, block_size=BS, combine="max")

    rcp_n = b.rcp(b.const_f32(float(N)))
    mean_sq = b.fmul(total_sq, rcp_n)
    inv_rms = b.rsqrt(b.fadd(mean_sq, eps_rms))

    amax_y = b.fmul(inv_rms, total_amax_g)
    safe_amax = b.fmax(amax_y, eps_q)
    yscale = b.fmul(safe_amax, b.const_f32(1.0 / qmax))
    inv_yscale = b.rcp(yscale)

    if spec.save_yscale:
        with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
            b.global_store(YScale, row, yscale, align=4)

    # Pass 2: re-load gamma; compute ``y = x * inv_rms * gamma`` and
    # quantise per element.
    chunks = spec.elems_per_thread // VEC
    c_vec = b.const_i32(VEC)
    for k in range(chunks):
        n_off = b.add(b.mul(b.const_i32(k * BS), c_vec), b.mul(tid, c_vec))
        g_scalars = g_view.load_vec_as_f32(b, [n_off], n=VEC)
        for i in range(VEC):
            x_f32 = cached_x[k * VEC + i]
            y_f32 = b.fmul(b.fmul(x_f32, inv_rms), g_scalars[i])
            q = quantize_scalar_f32(
                b, y_f32, inv_scale=inv_yscale, qdtype=spec.out_dtype
            )
            col = b.add(n_off, b.const_i32(i))
            qy_tile.store_scalar(b, b.const_i32(0), col, value=q)

    # ``store_scalar_from_f32`` is the canonical "f32 -> dtype scalar
    # store" path; we re-export it here as a no-op import so static
    # analysers don't flag the helper as unused (the kernel uses
    # ``global_store`` for the f32 ``YScale`` write but the
    # quantisation path keeps everything in the cvt op chain).
    _ = store_scalar_from_f32  # noqa: F841 -- public-API touch

    return b.kernel


def add_rmsnorm2d_rdquant_grid(
    m: int, spec: AddRmsnorm2DRdquantSpec
) -> Tuple[int, int, int]:
    """Return the launch grid: one CTA per row."""
    return ceil_div_grid((m, 1))


def add_rmsnorm2d_rdquant_signature(spec: AddRmsnorm2DRdquantSpec):
    sb = (
        SignatureBuilder()
        .ptr("A", spec.dtype)
        .ptr("B", spec.dtype)
        .ptr("Gamma", spec.dtype)
    )
    if spec.save_residual:
        sb.ptr("X", spec.dtype)
    sb.ptr("QY", spec.out_dtype)
    if spec.save_yscale:
        sb.ptr("YScale", "f32")
    return (
        sb.scalar("M", "i32")
        .scalar("N", "i32")
        .scalar("eps_rms", "f32")
        .scalar("eps_q", "f32")
        .build()
    )


__all__ = [
    "AddRmsnorm2DRdquantSpec",
    "add_rmsnorm2d_rdquant_grid",
    "add_rmsnorm2d_rdquant_signature",
    "build_add_rmsnorm2d_rdquant",
    "is_valid_spec",
]
