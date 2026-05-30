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

Optimization notes (mirror the sibling :mod:`smoothquant`):

* Pass-1 per-chunk absmax uses a balanced :func:`_tree_fmax` reduction
  so the per-element ``|y|`` chain has ``O(log VEC)`` critical-path
  depth, enabling the AMDGPU backend's ``v_max3_f32`` pattern-match
  (matches CK Tile's ``UseMax3`` inline-asm path).
* Pass-2 quantise+store packs ``VEC`` f32 lanes into one i32/i64 store
  per chunk via :func:`_pack_quant_chunk_f32` +
  :func:`_store_packed_chunk`. For ``fp8e4m3`` / ``bf8e5m2`` the cvt
  uses the packed ``v_cvt_pk_{fp8,bf8}_f32`` AMDGPU intrinsics, which
  is the AITER ``q8x4_t`` vec4 path from
  ``csrc/include/quant_common.cuh``.
* The per-CTA ``i_expert`` lookup pins the result in SGPR via
  :func:`IRBuilder.to_sgpr_u32` (wave-uniform; one ``s_load_dword``).
  The ``i_expert * N`` SmScale row base is pre-computed once outside
  the chunk loop and reused as the row-stride add, so every chunk's
  SmScale gather is a single ``s_add + global_load`` rather than a
  fresh multiply-accumulate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from ...core.ir import F32, I32, I64, IRBuilder, KernelDef, PtrType, Value
from ...helpers.io import io_ir_type
from ...helpers.quant import QDType, quant_ir_type, quant_max_abs, quantize_scalar_f32
from ...helpers.reduction import block_lds_reduce
from ...helpers.spec import (
    IOSpecRule,
    SignatureBuilder,
    ceil_div_grid,
    kernel_name_join,
    validate_io,
)
from ...helpers.sweep import sweep_row_chunks
from ...helpers.tensor_view import (
    make_global_view,
    make_lds_view,
    make_naive_tensor_view_packed,
    make_tile_window,
)


# ---------------------------------------------------------------------
# Local helpers (kept inside this instance file to honour the
# "instance files only" optimisation scope). These mirror the helpers
# in :mod:`ck_dsl.instances.common.smoothquant`; we cannot share through a
# private module without expanding the optimisation scope, and the
# logic is small enough that duplication is preferable to a
# cross-file import dance.
# ---------------------------------------------------------------------


def _tree_fmax(b: IRBuilder, values: List[Value]) -> Value:
    """Balanced pairwise ``fmax`` reduction (see sibling smoothquant
    helper for the rationale)."""
    cur = list(values)
    while len(cur) > 1:
        nxt: List[Value] = []
        for i in range(0, len(cur) - 1, 2):
            nxt.append(b.fmax(cur[i], cur[i + 1]))
        if len(cur) % 2:
            nxt.append(cur[-1])
        cur = nxt
    return cur[0]


def _pack_quant_chunk_f32(
    b: IRBuilder,
    scaled_f32: List[Value],
    *,
    q_ty,
    out_dtype: QDType,
) -> Value:
    """Quantise + vec-pack a chunk of ``y * inv_yscale`` f32 scalars.

    Dispatches on ``out_dtype`` exactly like the sibling helper in
    :mod:`ck_dsl.instances.common.smoothquant`: packed ``v_cvt_pk_fp8_f32``
    / ``v_cvt_pk_bf8_f32`` for the fp8/bf8 paths, per-element
    ``cvt_f32_to_i8_sat`` (with the [-127, 127] clamp) + ``vec_pack``
    for the i8 path.
    """
    n = len(scaled_f32)
    if n not in (2, 4, 8):
        raise ValueError(f"_pack_quant_chunk_f32 expects n in {{2,4,8}}, got {n}")

    if out_dtype in ("fp8e4m3", "bf8e5m2") and n % 4 == 0:
        cvt = b.cvt_pk_fp8_f32x4 if out_dtype == "fp8e4m3" else b.cvt_pk_bf8_f32x4
        packed_chunks: List[Value] = []
        for off in range(0, n, 4):
            quad = b.vec_pack(scaled_f32[off : off + 4], F32)
            packed_chunks.append(cvt(quad))
        if len(packed_chunks) == 1:
            return packed_chunks[0]
        out = packed_chunks[0]
        for chunk in packed_chunks[1:]:
            out = b.vec_concat(out, chunk)
        return out

    qs: List[Value] = []
    for sf in scaled_f32:
        if out_dtype == "i8":
            qs.append(
                b.cvt_f32_to_i8_sat(
                    b.clamp_f32(sf, b.const_f32(-127.0), b.const_f32(127.0))
                )
            )
        elif out_dtype == "fp8e4m3":
            qs.append(b.cvt_f32_to_fp8(sf))
        elif out_dtype == "bf8e5m2":
            qs.append(b.cvt_f32_to_bf8(sf))
        else:
            raise ValueError(f"unsupported out_dtype {out_dtype!r}")
    return b.vec_pack(qs, q_ty)


def _store_packed_chunk(
    b: IRBuilder,
    qy_ptr: Value,
    byte_off: Value,
    packed: Value,
    *,
    n: int,
) -> None:
    """Bitcast ``<n x q_ty>`` (8-bit elements) to i32/i64 and emit a
    single global store (see sibling smoothquant helper for details)."""
    if n == 4:
        as_int = b.bitcast(packed, I32)
        idx = b.lshr(byte_off, b.const_i32(2))
        b.global_store(qy_ptr, idx, as_int, align=4)
    elif n == 8:
        as_int = b.bitcast(packed, I64)
        idx = b.lshr(byte_off, b.const_i32(3))
        b.global_store(qy_ptr, idx, as_int, align=8)
    else:
        raise ValueError(f"_store_packed_chunk supports n in {{4, 8}}, got {n}")


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
    # P79: compile-time ``tokens`` (optional). When set, the
    # ``(i_topk, i_token) = (out_row / tokens, out_row % tokens)``
    # decode replaces the runtime ``div`` / ``mod`` with a Hacker's-
    # Delight ``v_mul_hi_u32`` pair. Trade-off: one specialised
    # kernel per (tokens) value the caller wants to hit; for static
    # batches this is a strict win.
    tokens: Optional[int] = None

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


def is_valid_spec(spec: MoeSmoothQuantSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for one MoE-SmoothQuant config on ``arch``.

    Same shape as the sibling :func:`ck_dsl.instances.common.smoothquant.
    is_valid_spec`: pure elementwise + single LDS-tree (amax) reduction
    (no MFMA), so the architecture facts that matter are the per-WG LDS
    capacity and max threads/block, both sourced from
    :class:`ck_dsl.core.arch.ArchTarget`. The one f32 reduction buffer
    (``block_size`` words) fits both gfx950 (160 KiB) and gfx1151, so
    gfx950 behavior is unchanged.
    """
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    if spec.out_dtype not in ("i8", "fp8e4m3", "bf8e5m2"):
        return False, f"unsupported out_dtype {spec.out_dtype!r}"

    # fp8/bf8 output goes through ``v_cvt_pk_{fp8,bf8}_f32`` (the
    # ``llvm.amdgcn.cvt.pk.{fp8,bf8}.f32`` intrinsic), which only exists
    # on the CDNA (gfx9xx) ISA — RDNA3.5 (gfx1151) has no fp8/bf8 pack
    # conversion op, so selection would abort with an uncatchable
    # ``LLVM ERROR: Cannot select``. Reject it as a clean spec error so
    # callers get a structured reason. The ``i8`` path uses
    # ``v_cvt_f32_to_i8`` (available everywhere) and stays wave32-valid.
    if spec.out_dtype in ("fp8e4m3", "bf8e5m2") and target.family != "cdna":
        return False, (
            f"out_dtype {spec.out_dtype!r} needs the CDNA-only "
            f"v_cvt_pk_{{fp8,bf8}}_f32 conversion; {arch} (family "
            f"{target.family!r}) has no fp8/bf8 pack op -- use out_dtype='i8'"
        )
    if spec.topk < 1:
        return False, f"topk must be >= 1 (got {spec.topk})"
    if spec.experts < 1:
        return False, f"experts must be >= 1 (got {spec.experts})"
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

    # One f32 LDS reduction buffer (amax) of ``block_size`` words.
    bytes_lds = spec.block_size * 4
    if not target.fits_lds(bytes_lds):
        return False, (
            f"LDS budget {bytes_lds} > {target.lds_capacity_bytes} cap on {arch}"
        )

    return True, ""


def build_moe_smoothquant(spec: MoeSmoothQuantSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for one MoE-SmoothQuant instance.

    Kernel signature::

    (X: ptr<dtype, global>, # (tokens, N)
    SmScale: ptr<f32, global>, # (experts * N,) per-expert smooth scale
    TopkIds: ptr<i32, global>, # (tokens, topk) expert ids
    QY: ptr<out_dtype, global>, # (topk * tokens, N)
    [YScale: ptr<f32, global>,] # (topk * tokens,)
    tokens: i32, N: i32, eps: f32)

    Grid: ``(topk * tokens, 1, 1)`` — one CTA per output row.

    ``arch`` selects the hardware target. As with the sibling
    :func:`ck_dsl.instances.common.smoothquant.build_smoothquant`, the
    only cross-lane operation is :func:`block_lds_reduce` (an LDS tree
    over all ``block_size`` lanes, no wave shuffle), so the kernel is
    wave-size agnostic and correct on wave64 (gfx950) and wave32
    (gfx1151). The default ``"gfx950"`` keeps the wave64 build
    byte-identical.
    """
    ok, why = is_valid_spec(spec, arch)
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
    #
    # P79: when ``spec.tokens`` is a compile-time int, the divisor is
    # constant — the AMDGPU backend folds the ``div`` / ``mod`` chain
    # into one ``v_mul_hi_u32`` + ``v_sub_u32`` pair (the Hacker's-
    # Delight reciprocal-mul trick). Without ``spec.tokens`` set we
    # fall back to the runtime div/mod (which costs ~30 VALU ops vs
    # ~3 for the constant-folded form).
    if spec.tokens is not None:
        c_tok_const = b.const_i32(spec.tokens)
        i_topk = b.div(out_row, c_tok_const)
        i_token = b.mod(out_row, c_tok_const)
    else:
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
    # is a (topk*tokens, N) packed view. SmScale is exposed as a flat
    # 1D ``(experts * N,)`` view; the per-expert row-base offset
    # ``i_expert * N`` is pre-computed once below and kept in SGPR via
    # :func:`IRBuilder.to_sgpr_u32`. Every chunk's SmScale gather is
    # then just ``sm_view[i_expert*N + n_off]`` -- one ``s_add`` (the
    # per-expert base hoist) and one ``global_load_vN_f32`` (the
    # column gather), no per-chunk multiply-accumulate.
    x_view = make_naive_tensor_view_packed(X, shape=(1, N), dtype=io_ty)
    qy_view = make_naive_tensor_view_packed(QY, shape=(1, N), dtype=q_ty)
    sm_view = make_global_view(SmScale, shape=(spec.experts * N,), dtype=F32)
    x_tile = make_tile_window(x_view, lengths=(1, N), origin=(i_token, b.const_i32(0)))
    qy_tile = make_tile_window(
        qy_view, lengths=(1, N), origin=(out_row, b.const_i32(0))
    )

    sm_row_base = b.to_sgpr_u32(b.mul(i_expert, b.const_i32(N)))

    lds = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_amax").base

    # Pass 1: stream x, multiply by SmScale[i_expert, :], reduce amax.
    # Mirror :mod:`ck_dsl.instances.common.smoothquant`'s tree-fmax pattern so
    # the per-element ``|y|`` chain is ``O(log VEC)`` deep on the
    # critical path.
    s_amax = b.const_f32(0.0)

    def pass1_body(n_off, x_scalars):
        nonlocal s_amax
        sm_off = b.add(sm_row_base, n_off)
        sm_scalars = sm_view.load_vec_as_f32(b, [sm_off], n=VEC)
        abs_ys: List[Value] = []
        for i in range(VEC):
            y = b.fmul(x_scalars[i], sm_scalars[i])
            abs_ys.append(b.fmax(y, b.fneg(y)))
        chunk_amax = _tree_fmax(b, abs_ys)
        s_amax = b.fmax(s_amax, chunk_amax)

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

    # Pass 2: quantise + packed store. See sibling smoothquant for the
    # rationale; the only MoE-specific bit is the SmScale gather
    # through ``sm_row_base`` (pre-hoisted ``i_expert * N``).
    cached = sweep_res.cached
    chunks = spec.elems_per_thread // VEC
    c_vec = b.const_i32(VEC)
    use_packed_store = VEC in (4, 8)
    row_base_byte_off = b.mul(out_row, b.const_i32(N))
    for k in range(chunks):
        n_off = b.add(b.mul(b.const_i32(k * BS), c_vec), b.mul(tid, c_vec))
        sm_off = b.add(sm_row_base, n_off)
        sm_scalars = sm_view.load_vec_as_f32(b, [sm_off], n=VEC)
        if use_packed_store:
            scaled_f32: List[Value] = []
            for i in range(VEC):
                x_f32 = cached[k * VEC + i]
                y_f32 = b.fmul(x_f32, sm_scalars[i])
                scaled_f32.append(b.fmul(y_f32, inv_yscale))
            packed = _pack_quant_chunk_f32(
                b, scaled_f32, q_ty=q_ty, out_dtype=spec.out_dtype
            )
            byte_off = b.add(row_base_byte_off, n_off)
            _store_packed_chunk(b, QY, byte_off, packed, n=VEC)
        else:
            # VEC == 2: per-element scalar quant + store fallback.
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
