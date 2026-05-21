# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MoE-aware SmoothQuant kernel (CK Tile ``14_moe_smoothquant`` parity).

DSL counterpart of CK Tile's ``example/ck_tile/14_moe_smoothquant``.
Same per-row recipe as :func:`build_smoothquant`, with two extensions
that match the MoE router output layout:

* ``SmScale`` is now a ``(experts, N)`` per-expert smooth scale table
 (flat ``(experts * N,)`` in memory) and is gathered by the per-token
 expert id rather than shared across all rows.
* The kernel produces ``topk * tokens`` output rows: for each input
 token, the router selected ``topk`` experts; the kernel quantises
 the token once per selected expert with that expert's smscale.

Output row layout (matches CK Tile's reference)::

 out_row = i_topk * tokens + i_token # outer dim is topk
 qy[out_row, :] = quantise(x[i_token, :] * smscale[i_expert, :])
 yscale[out_row] = dynamic per-row scale
 where i_expert = topk_ids[i_token, i_topk]

The kernel launches one CTA per output row; ``block_id_x`` ranges
over ``[0, topk * tokens)`` and we decode ``(i_topk, i_token)`` from
the linear index. The expert id is read once per CTA (``tid == 0``
plus a workgroup broadcast) — a cheap single ``ds_bpermute`` against
LDS — then every thread re-uses it for its column-stride lookup into
SmScale.

What we cover today:

* ``dtype`` (input X) : ``f16`` / ``bf16``
* ``out_dtype`` (QY) : ``i8`` / ``fp8e4m3`` / ``bf8e5m2``
* ``topk`` : compile-time positive int (same as CK Tile's ``-k`` arg)
* ``experts`` : compile-time positive int (sized for SmScale stride)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ..core.ir import F32, I32, IRBuilder, KernelDef, PtrType
from ..helpers.io import io_ir_type
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
    make_lds_view,
    make_naive_tensor_view_packed,
    make_tile_window,
)


DType = Literal["f16", "bf16"]


@dataclass(frozen=True)
class MoeSmoothQuantSpec:
    """One concrete MoE-SmoothQuant kernel configuration."""

    n_per_block: int  # the hidden dim N (compile-time)
    topk: int  # router top-k
    experts: int  # total experts
    dtype: DType = "f16"
    out_dtype: QDType = "i8"
    block_size: int = 256
    vec: int = 4
    save_yscale: bool = True
    wave_size: int = 64
    name: str = "ck_dsl_moe_smoothquant"

    @property
    def elems_per_thread(self) -> int:
        return self.n_per_block // self.block_size

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.dtype,
            self.out_dtype,
            f"N{self.n_per_block}",
            f"E{self.experts}",
            f"K{self.topk}",
            f"b{self.block_size}",
            f"v{self.vec}",
            flags={"ys": self.save_yscale},
        )


def is_valid_spec(spec: MoeSmoothQuantSpec) -> Tuple[bool, str]:
    if spec.out_dtype not in ("i8", "fp8e4m3", "bf8e5m2"):
        return False, f"unsupported out_dtype {spec.out_dtype!r}"
    if spec.topk < 1:
        return False, f"topk must be >= 1 (got {spec.topk})"
    if spec.experts < 1:
        return False, f"experts must be >= 1 (got {spec.experts})"
    return validate_io(
        IOSpecRule(
            dtype=spec.dtype,
            block_size=spec.block_size,
            vec=spec.vec,
            n_per_block=spec.n_per_block,
            max_elems_per_thread=64,
        )
    )


def build_moe_smoothquant(spec: MoeSmoothQuantSpec) -> KernelDef:
    """Build the IR for one MoE-SmoothQuant instance.

    Kernel signature::

    (X: ptr<dtype, global>, # (tokens, N)
    SmScale: ptr<f32, global>, # (experts * N,) per-expert smooth scale
    TopkIds: ptr<i32, global>, # (tokens, topk) expert ids
    QY: ptr<out_dtype, global>, # (topk * tokens, N)
    [YScale: ptr<f32, global>,] # (topk * tokens,)
    tokens: i32, N: i32, eps: f32)

    Grid: ``(topk * tokens, 1, 1)`` — one CTA per output row.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid moe_smoothquant spec: {why}")

    io_ty = io_ir_type(spec.dtype)
    q_ty = quant_ir_type(spec.out_dtype)
    qmax = quant_max_abs(spec.out_dtype)

    BS, VEC, N = spec.block_size, spec.vec, spec.n_per_block
    topk = spec.topk

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    SmScale = b.param(
        "SmScale", PtrType(F32, "global"), noalias=True, readonly=True, align=16
    )
    TopkIds = b.param(
        "TopkIds", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    QY = b.param("QY", PtrType(q_ty, "global"), noalias=True, writeonly=True, align=16)
    if spec.save_yscale:
        YScale = b.param(
            "YScale", PtrType(F32, "global"), noalias=True, writeonly=True, align=4
        )
    tokens = b.param("tokens", I32)
    _ = b.param("N", I32)  # noqa: F841 — validated by caller; equals n_per_block
    eps = b.param("eps", F32)

    tid = b.thread_id_x()
    out_row = b.block_id_x()
    c_tokens = tokens  # alias for clarity

    # Decode (i_topk, i_token) from the linear ``out_row``. Layout
    # ``out_row = i_topk * tokens + i_token`` matches the CK Tile
    # reference (and the ``i_topk * tokens + i_token`` indexing in
    # ``moe_smoothquant.cpp`` line 181).
    i_topk = b.div(out_row, c_tokens)
    i_token = b.mod(out_row, c_tokens)

    # Look up the per-token expert id from the (tokens, topk) router
    # output. ``TopkIds[i_token * topk + i_topk]`` is the C-contiguous
    # flat offset. ``topkids_idx`` is wave-uniform (every lane in the
    # CTA computes the same value), so the AMDGPU backend emits one
    # ``s_load_dword`` for the whole wave -- no LDS broadcast is needed
    # to share the result. ``readfirstlane_pin`` materialises the
    # result in an SGPR so subsequent uses (the SmScale row stride
    # multiply) stay in scalar registers.
    c_topk = b.const_i32(topk)
    topkids_idx = b.add(b.mul(i_token, c_topk), i_topk)
    i_expert = b.to_sgpr_u32(b.global_load_i32(TopkIds, topkids_idx))

    # CK Tile-style views. X is a flat (tokens, N) packed view; QY
    # is a (topk*tokens, N) packed view; SmScale is a 2D (experts, N)
    # packed view (so the per-expert row stride is exactly N).
    x_view = make_naive_tensor_view_packed(X, shape=(1, N), dtype=io_ty)
    qy_view = make_naive_tensor_view_packed(QY, shape=(1, N), dtype=q_ty)
    sm_view = make_naive_tensor_view_packed(SmScale, shape=(spec.experts, N), dtype=F32)
    x_tile = make_tile_window(x_view, lengths=(1, N), origin=(i_token, b.const_i32(0)))
    qy_tile = make_tile_window(
        qy_view, lengths=(1, N), origin=(out_row, b.const_i32(0))
    )

    lds = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_amax").base

    # Pass 1: stream x, multiply by SmScale[i_expert, :], reduce amax.
    s_amax = b.const_f32(0.0)

    def pass1_body(n_off, x_scalars):
        nonlocal s_amax
        sm_scalars = sm_view.load_vec_as_f32(b, [i_expert, n_off], n=VEC)
        for i in range(VEC):
            y = b.fmul(x_scalars[i], sm_scalars[i])
            abs_y = b.fmax(y, b.fneg(y))
            s_amax = b.fmax(s_amax, abs_y)

    sweep_res = sweep_row_chunks(
        b,
        x_tile,
        tid=tid,
        block_size=BS,
        vec=VEC,
        elems_per_thread=spec.elems_per_thread,
        body=pass1_body,
        cache=True,
    )

    total_amax = block_lds_reduce(b, s_amax, lds, tid, block_size=BS, combine="max")

    safe_amax = b.fmax(total_amax, eps)
    yscale = b.fmul(safe_amax, b.const_f32(1.0 / qmax))
    inv_yscale = b.rcp(yscale)

    if spec.save_yscale:
        with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
            b.global_store(YScale, out_row, yscale, align=4)

    # Pass 2: quantise + store.
    cached = sweep_res.cached
    chunks = spec.elems_per_thread // VEC
    c_vec = b.const_i32(VEC)
    for k in range(chunks):
        n_off = b.add(b.mul(b.const_i32(k * BS), c_vec), b.mul(tid, c_vec))
        sm_scalars = sm_view.load_vec_as_f32(b, [i_expert, n_off], n=VEC)
        for i in range(VEC):
            x_f32 = cached[k * VEC + i]
            y_f32 = b.fmul(x_f32, sm_scalars[i])
            q = quantize_scalar_f32(
                b, y_f32, inv_scale=inv_yscale, qdtype=spec.out_dtype
            )
            col = b.add(n_off, b.const_i32(i))
            qy_tile.store_scalar(b, b.const_i32(0), col, value=q)

    return b.kernel


def moe_smoothquant_grid(tokens: int, spec: MoeSmoothQuantSpec) -> Tuple[int, int, int]:
    """Return the launch grid: one CTA per ``(i_topk, i_token)`` pair."""
    return ceil_div_grid((tokens * spec.topk, 1))


def moe_smoothquant_signature(spec: MoeSmoothQuantSpec):
    sb = (
        SignatureBuilder()
        .ptr("X", spec.dtype)
        .ptr("SmScale", "f32")
        .ptr("TopkIds", "i32")
        .ptr("QY", spec.out_dtype)
    )
    if spec.save_yscale:
        sb.ptr("YScale", "f32")
    return sb.scalar("tokens", "i32").scalar("N", "i32").scalar("eps", "f32").build()


__all__ = [
    "MoeSmoothQuantSpec",
    "build_moe_smoothquant",
    "is_valid_spec",
    "moe_smoothquant_grid",
    "moe_smoothquant_signature",
]
