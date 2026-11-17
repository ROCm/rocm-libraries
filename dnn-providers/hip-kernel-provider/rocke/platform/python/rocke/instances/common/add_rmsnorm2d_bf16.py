# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fused add + RMSNorm kernel with bf16/f16 output (no quantization).

A simplified relative of :mod:`rocke.instances.common.add_rmsnorm2d_rdquant`.
For two ``(M, N)`` activation tensors ``A`` and ``B`` and an ``(N,)``
``Gamma`` per-channel scale, the kernel produces both the residual sum
``X = A + B`` and the RMSNorm-normalized output ``Y`` in one pass over
global memory::

    x[m, n]   = a[m, n] + b[m, n]
    sum_sq[m] = sum_n(x[m, n] ^ 2)
    inv_rms[m] = 1 / sqrt(sum_sq[m] / N + eps)
    y[m, n]   = x[m, n] * inv_rms[m] * gamma[n]

This matches the pre-attention / pre-MoE residual + norm pattern in
Llama / Qwen / Mistral decoders: the kernel writes ``x`` back to ``A``
(or to a separate ``X`` buffer when ``save_residual=True``) so the
residual stream is ready for the next layer's add, and produces ``y``
as the norm input to the next projection.

Compared to ``add_rmsnorm2d_rdquant`` this drops the dynamic-quantize
pass: one block reduction (sum) instead of the twin sum+max,
``vec``-wide packed bf16 stores via ``store_vec_from_f32`` instead of
per-element quant casts, and no ``YScale`` output. Pass 1 still streams
``A`` and ``B``, computes ``x`` and the per-thread sum-of-squares, and
caches ``x`` for pass 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

from ...core.ir import F32, I32, IRBuilder, KernelDef, PtrType, Value
from ...helpers.io import io_ir_type
from ...helpers.reduction import (
    block_lds_reduce,
    block_lds_reduce_with_wave_prologue,
    tree_reduce,
)
from ...helpers.spec import (
    IOSpecRule,
    SignatureBuilder,
    ceil_div_grid,
    kernel_name_join,
    validate_io,
)
from ...helpers.tensor_view import (
    make_global_view,
    make_lds_view,
    make_naive_tensor_view_packed,
    make_tile_window,
)


DType = Literal["f16", "bf16"]


@dataclass(frozen=True)
class AddRMSNorm2DBF16Spec:
    """One concrete fused add + RMSNorm (bf16/f16 output) configuration."""

    n_per_block: int
    block_size: int = 256
    vec: int = 4
    dtype: DType = "bf16"
    save_residual: bool = True  # write x = a + b to ``X``
    wave_size: int = 64
    name: str = "rocke_add_rmsnorm2d_bf16"

    @property
    def elems_per_thread(self) -> int:
        return self.n_per_block // self.block_size

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.dtype,
            f"N{self.n_per_block}",
            f"b{self.block_size}",
            f"v{self.vec}",
            flags={"sr": self.save_residual},
        )


def is_valid_spec(spec: AddRMSNorm2DBF16Spec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for one fused add + RMSNorm config on ``arch``.

    Pure elementwise + single LDS-tree reduction (no MFMA): the only
    architecture facts that matter are the per-WG LDS capacity and max
    threads/block, both sourced from :class:`rocke.core.arch.ArchTarget`
    so an unknown arch / over-budget ``block_size`` is rejected with a
    structured reason. The one f32 reduction buffer (``block_size``
    words) fits both gfx942 (64 KiB) and gfx950 (160 KiB), so gfx950
    behavior is unchanged.
    """
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    ok, why = validate_io(
        IOSpecRule(
            dtype=spec.dtype,
            block_size=spec.block_size,
            vec=spec.vec,
            n_per_block=spec.n_per_block,
            max_elems_per_thread=64,
        )
    )
    if not ok:
        return False, why

    if spec.block_size > target.max_threads_per_block:
        return False, (
            f"block_size {spec.block_size} > max_threads_per_block "
            f"{target.max_threads_per_block} on {arch}"
        )

    # One f32 LDS reduction buffer of ``block_size`` words.
    bytes_lds = spec.block_size * 4
    if not target.fits_lds(bytes_lds):
        return False, (
            f"LDS budget {bytes_lds} > {target.lds_capacity_bytes} cap on {arch}"
        )

    return True, ""


def build_add_rmsnorm2d_bf16(
    spec: AddRMSNorm2DBF16Spec, arch: str = "gfx950"
) -> KernelDef:
    """Build the IR for one fused add + RMSNorm bf16-output instance.

    Kernel signature (with ``save_residual=True``)::

        (A: ptr<dtype, global>,      # (M, N) input
         B: ptr<dtype, global>,      # (M, N) input
         Gamma: ptr<dtype, global>,  # (N,)
         X: ptr<dtype, global>,      # (M, N) residual out (a+b)
         Y: ptr<dtype, global>,      # (M, N) normed output
         M: i32, N: i32,
         eps: f32)

    Grid: ``(M, 1, 1)`` -- one CTA per row.

    ``arch`` selects the validation target (default ``"gfx950"`` keeps
    the CDNA wave64 behavior byte-identical). The body is wave-size
    agnostic: the single cross-thread fold goes through
    :func:`block_lds_reduce`, an LDS tree that halves over
    ``block_size`` and barriers between steps -- it carries no
    wave64-only cross-lane op, so the same IR is correct on the gfx1151
    wave32 target. ``arch`` is therefore only threaded into
    :func:`is_valid_spec` (LDS / threads-per-block budgeting).
    """
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid add_rmsnorm2d_bf16 spec: {why}")

    io_ty = io_ir_type(spec.dtype)
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
    Y = b.param("Y", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    _M = b.param("M", I32)  # noqa: F841 -- ABI symmetry
    _ = b.param("N", I32)  # noqa: F841 -- validated by caller
    eps = b.param("eps", F32)

    tid = b.thread_id_x()
    row = b.block_id_x()

    a_view = make_naive_tensor_view_packed(A, shape=(1, N), dtype=io_ty)
    b_view = make_naive_tensor_view_packed(Bp, shape=(1, N), dtype=io_ty)
    g_view = make_global_view(Gamma, shape=(N,), dtype=io_ty)
    y_view = make_naive_tensor_view_packed(Y, shape=(1, N), dtype=io_ty)
    a_tile = make_tile_window(a_view, lengths=(1, N), origin=(row, b.const_i32(0)))
    bt_tile = make_tile_window(b_view, lengths=(1, N), origin=(row, b.const_i32(0)))
    y_tile = make_tile_window(y_view, lengths=(1, N), origin=(row, b.const_i32(0)))
    if spec.save_residual:
        x_view = make_naive_tensor_view_packed(X, shape=(1, N), dtype=io_ty)
        x_tile = make_tile_window(x_view, lengths=(1, N), origin=(row, b.const_i32(0)))

    lds = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_red").base

    # Pass 1: stream A and B, compute x = a + b and per-thread
    # sum-of-squares. Cache the f32 x scalars for pass 2 so we do not
    # re-read either input. (Caching x instead of x*gamma keeps pass 2's
    # gamma load on the L1-hot path -- we save one HBM round-trip on
    # gamma at the cost of one f32 register per cached element, same
    # register footprint as the rdquant variant.)
    s_sq = b.const_f32(0.0)
    cached_x: List[Value] = []
    chunks_per_thread = spec.elems_per_thread // VEC
    c_vec = b.const_i32(VEC)

    for k in range(chunks_per_thread):
        n_off = b.add(
            b.mul(b.const_i32(k * BS), c_vec),
            b.mul(tid, c_vec),
        )
        a_scalars = a_tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC)
        b_scalars = bt_tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC)
        chunk_x: List[Value] = []
        chunk_sq: List[Value] = []
        for i in range(VEC):
            x_i = b.fadd(a_scalars[i], b_scalars[i])
            chunk_x.append(x_i)
            chunk_sq.append(b.fmul(x_i, x_i))
        s_sq = b.fadd(s_sq, tree_reduce(b, b.fadd, chunk_sq))
        cached_x.extend(chunk_x)
        if spec.save_residual:
            # Write x = a + b back to the residual buffer in vec stores
            # so the next layer's residual add can pick it up without
            # re-computing.
            x_tile.store_vec_from_f32(b, b.const_i32(0), n_off, values=chunk_x)

    # Cross-thread reduction. On a wave64 target whose ``wave_size``
    # matches the spec and a block size that is a clean multiple of it,
    # use the CK Tile ``BlockReduce2dSync`` (warp XOR butterfly) +
    # ``CrossWarpSync`` (one ``num_warps``-slot LDS round) shape, the
    # same path ``instances/common/reduce.py`` and ``rmsnorm2d.py``
    # adopt: for BS=256 / wave64 this replaces the 8-round LDS tree
    # (8 syncs) with six cross-lane shuffles + one ``sync``. The butterfly
    # uses ``wave_size``-stride cross-lane ops, so we only take it when the
    # hardware wave actually matches; otherwise (e.g. the wave32 gfx1151
    # target the docstring guarantees) we keep the wave-agnostic full LDS
    # tree, leaving that correctness path byte-for-byte unchanged.
    from ...core.arch import ArchTarget

    hw_wave = ArchTarget.from_gfx(arch).wave_size
    if hw_wave == spec.wave_size and spec.block_size % spec.wave_size == 0:
        total_sq = block_lds_reduce_with_wave_prologue(
            b,
            s_sq,
            lds,
            tid,
            block_size=spec.block_size,
            combine="sum",
            wave_size=spec.wave_size,
        )
    else:
        total_sq = block_lds_reduce(b, s_sq, lds, tid, block_size=BS, combine="sum")

    rcp_n = b.rcp(b.const_f32(float(N)))
    mean_sq = b.fmul(total_sq, rcp_n)
    inv_rms = b.rsqrt(b.fadd(mean_sq, eps))

    # Pass 2: y = x * inv_rms * gamma. Re-load gamma per chunk (L1-hot
    # by now), use cached x. Vec-wide bf16 stores via
    # store_vec_from_f32. Reorder as x * (inv_rms * gv) to match the
    # rmsnorm2d pass-2 idiom (lets the AMDGPU scheduler hoist inv_rms*gv
    # alongside the cached x reads).
    for k in range(chunks_per_thread):
        n_off = b.add(
            b.mul(b.const_i32(k * BS), c_vec),
            b.mul(tid, c_vec),
        )
        gv = g_view.load_vec_as_f32(b, [n_off], n=VEC)
        y_vec = [
            b.fmul(cached_x[k * VEC + i], b.fmul(inv_rms, gv[i])) for i in range(VEC)
        ]
        y_tile.store_vec_from_f32(b, b.const_i32(0), n_off, values=y_vec)

    return b.kernel


def add_rmsnorm2d_bf16_grid(m: int, spec: AddRMSNorm2DBF16Spec) -> Tuple[int, int, int]:
    """Return the launch grid: one CTA per row."""
    return ceil_div_grid((m, 1))


def add_rmsnorm2d_bf16_signature(spec: AddRMSNorm2DBF16Spec):
    sb = (
        SignatureBuilder()
        .ptr("A", spec.dtype)
        .ptr("B", spec.dtype)
        .ptr("Gamma", spec.dtype)
    )
    if spec.save_residual:
        sb.ptr("X", spec.dtype)
    sb.ptr("Y", spec.dtype)
    return sb.scalar("M", "i32").scalar("N", "i32").scalar("eps", "f32").build()


__all__ = [
    "AddRMSNorm2DBF16Spec",
    "add_rmsnorm2d_bf16_grid",
    "add_rmsnorm2d_bf16_signature",
    "build_add_rmsnorm2d_bf16",
    "is_valid_spec",
]
