# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MoE activation prologue: gather into expert-block order under a block scale.

``moe_fused_mega_fp8`` reads its activations already gathered into expert-block
order, padded to ``tile_m``, with **one activation scale per (block, 128-group)**
shared by every row of the block. That second requirement is not a convenience:
the gate/up fold applies a single per-lane scalar, indexed by the A fragment's
row (``m_in_atom``), to accumulator slots spanning four *different* output rows,
so only a row-uniform scale survives it. The mega-kernel's own dynamic-quant
epilogue is built to the same constraint and says so (see the STAGE 1b Pass A
comment, ``BUILD_SPEC_FP8 OPEN RISK #1``).

A caller holding per-token-quantized activations -- which is what vLLM's
``kFp8Dynamic128Sym`` produces, and what any serving stack that quantizes before
routing will have -- therefore cannot hand them over directly. Nothing rejects
them either: per-token scales are silently wrong rather than an error, because
the fold reads a legal address and multiplies by the wrong row's scale. So this
kernel exists to make the conforming layout cheap to produce, and to put the
requirement somewhere a caller can satisfy by construction.

It emits, in one pass over the activation:

* ``A``          -- gathered, padded, and *restated* under the block scale.
                    ``A_q * s_token == A_q' * s_block`` holds by construction, so
                    this changes the fp8 rounding and nothing else.
* ``AScale``     -- the block scale, broadcast to every row including the pad
                    rows, which is the form the fold reads.
* ``SortedTokenIds`` / ``SortedWeights`` -- the epilogue's scatter metadata,
                    translated from the caller's flattened ``(token, slot)`` ids
                    and sentinel into token ids with -1 on pad rows.

Doing it as one kernel rather than in the framework is worth about 230 us of a
64-token Qwen3 layer. The framework-side version cannot avoid widening to f32 to
apply the rescale, which turns 4 MB of fp8 traffic into four passes over a 16 MB
temp; here the widening lives in registers between a vector load and a vector
store.

Pad rows are written as exact zeros. That is required, not tidiness: a pad row's
activations still enter the block-wide amax the mega-kernel reduces to set the
intermediate's scale, so anything non-zero there corrupts a real row's output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ...core.ir import F32, FP8E4M3, I32, IRBuilder, KernelDef, PtrType
from ...helpers.spec import SignatureBuilder

#: rocKE's scale-group width along the contraction. Fixed by the MoE kernels.
GROUP_K = 128


@dataclass(frozen=True)
class MoeGatherRescaleSpec:
    """One instance of the activation prologue.

    ``tile_m`` has to match the ``tile_m`` of the MoE kernel this feeds, because
    it is the row blocking the activation scale is uniform over. A mismatch is
    silently wrong in the same way per-token scales are, so callers should take
    it from the same spec they launch the GEMM with.
    """

    #: Rows per expert block. Must equal the consuming kernel's ``tile_m``.
    tile_m: int = 16
    #: Upper bound on ``hidden // GROUP_K``, for static LDS sizing. ``hidden``
    #: itself stays a runtime argument; only the scratch has to be sized ahead.
    max_n_hb: int = 32
    block_size: int = 256
    #: fp8 elements per thread per load/store. Must divide ``GROUP_K`` so a
    #: vector never straddles two scale groups, which is what lets the rescale
    #: use one scalar for the whole vector.
    vec: int = 8
    name: str = "rocke_moe_gather_rescale_a"

    def __post_init__(self) -> None:
        if GROUP_K % self.vec:
            raise ValueError(
                f"vec={self.vec} must divide GROUP_K={GROUP_K} so a vector "
                "cannot straddle two activation scale groups"
            )
        if self.tile_m * self.max_n_hb > 65536:
            raise ValueError("tile_m * max_n_hb scratch is implausibly large")

    def kernel_name(self) -> str:
        return (
            f"{self.name}_tm{self.tile_m}_b{self.block_size}"
            f"_v{self.vec}_h{self.max_n_hb}"
        )


def build_moe_gather_rescale_a(
    spec: MoeGatherRescaleSpec, *, arch: str = "gfx950"
) -> KernelDef:
    """One workgroup per expert block."""
    del arch  # No arch-specific paths; the kernel is plain vector memory work.

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size

    Aq = b.param(
        "Aq", PtrType(FP8E4M3, "global"), noalias=True, readonly=True, align=16
    )
    AqScale = b.param(
        "AqScale", PtrType(F32, "global"), noalias=True, readonly=True, align=4
    )
    SortedIds = b.param(
        "SortedIds", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    TopkWeights = b.param(
        "TopkWeights", PtrType(F32, "global"), noalias=True, readonly=True, align=4
    )
    A = b.param("A", PtrType(FP8E4M3, "global"), noalias=True, align=16)
    AScale = b.param("AScale", PtrType(F32, "global"), noalias=True, align=4)
    SortedTokenIds = b.param(
        "SortedTokenIds", PtrType(I32, "global"), noalias=True, align=4
    )
    SortedWeights = b.param(
        "SortedWeights", PtrType(F32, "global"), noalias=True, align=4
    )
    n_flat = b.param("n_flat", I32)
    topk = b.param("topk", I32)
    hidden = b.param("hidden", I32)
    n_hb = b.param("n_hb", I32)

    tid = b.thread_id_x()
    blk = b.block_id_x()
    c_tile_m = b.const_i32(spec.tile_m)
    c_vec = b.const_i32(spec.vec)
    c_group_k = b.const_i32(GROUP_K)
    c_bs = b.const_i32(spec.block_size)
    zero_i = b.const_i32(0)
    zero_f = b.const_f32(0.0)
    row_base = b.mul(blk, c_tile_m)

    # ``tok`` carries validity as a negative value, so the phases below need no
    # separate predicate array.
    tok_lds = b.smem_alloc(I32, [spec.tile_m], "tok")
    # Reused: holds the per-token scale after phase 1, the ratio after phase 3.
    scale_lds = b.smem_alloc(F32, [spec.tile_m, spec.max_n_hb], "rowscale")
    blk_lds = b.smem_alloc(F32, [spec.max_n_hb], "blkscale")

    # ---- phase 0: per-row metadata, and the epilogue's scatter arrays -------
    with b.scf_if(b.cmp_lt(tid, c_tile_m)):
        row = b.add(row_base, tid)
        sid = b.global_load_i32(SortedIds, row)
        valid = b.cmp_lt(sid, n_flat)
        # Out-of-range on a pad row, so every read of it is index-guarded.
        sid_safe = b.select(valid, sid, zero_i)
        tok = b.select(valid, b.div(sid_safe, topk), b.const_i32(-1))
        b.smem_store_vN(tok_lds, [tid], tok, 1)
        b.global_store(SortedTokenIds, row, tok, align=4)
        w = b.global_load_f32(TopkWeights, sid_safe)
        b.global_store(SortedWeights, row, b.select(valid, w, zero_f), align=4)
    b.sync()

    # ---- phase 1: gather each row's per-token scale into scratch ------------
    n_cells = b.mul(c_tile_m, n_hb)
    with b.scf_for(tid, n_cells, c_bs, iv_name="cell1") as cell:
        r = b.div(cell, n_hb)
        g = b.mod(cell, n_hb)
        tok = b.vec_extract(b.smem_load_vN(tok_lds, r, dtype=I32, n=1), 0)
        valid = b.cmp_ge(tok, zero_i)
        tok_safe = b.smax(tok, zero_i)
        s = b.global_load_f32(AqScale, b.add(b.mul(tok_safe, n_hb), g))
        # Zero on a pad row: it must not raise the block's amax.
        b.smem_store_vN(scale_lds, [r, g], b.select(valid, s, zero_f), 1)
    b.sync()

    # ---- phase 2: reduce to one scale per group, over the block's rows ------
    with b.scf_for(tid, n_hb, c_bs, iv_name="g2") as g:
        acc = zero_f
        for r in range(spec.tile_m):
            v = b.vec_extract(
                b.smem_load_vN(scale_lds, b.const_i32(r), g, dtype=F32, n=1), 0
            )
            acc = b.fmax(acc, v)
        # A block with no real rows would otherwise publish a zero scale, which
        # the consumer divides by.
        b.smem_store_vN(blk_lds, [g], b.fmax(acc, b.const_f32(1e-30)), 1)
    b.sync()

    # ---- phase 3: publish AScale, and fold the rescale into a ratio ---------
    # The ratio overwrites the per-token scale in place. Safe because phase 2's
    # readers are behind a barrier and each cell is rewritten by the one thread
    # that owns it.
    with b.scf_for(tid, n_cells, c_bs, iv_name="cell3") as cell:
        r = b.div(cell, n_hb)
        g = b.mod(cell, n_hb)
        rs = b.vec_extract(b.smem_load_vN(scale_lds, r, g, dtype=F32, n=1), 0)
        bs = b.vec_extract(b.smem_load_vN(blk_lds, g, dtype=F32, n=1), 0)
        b.global_store(AScale, b.add(b.mul(b.add(row_base, r), n_hb), g), bs, align=4)
        # An exact divide, not rcp_fast: this is computed once per cell and then
        # reused across the whole 128-wide group, so its cost is irrelevant,
        # while a 1-ulp error in it flips fp8 rounding near a tie and would make
        # the kernel disagree with the framework fallback on a few bytes.
        b.smem_store_vN(scale_lds, [r, g], b.fdiv(rs, bs), 1)
    b.sync()

    # ---- phase 4: the gather itself ----------------------------------------
    # Rows are unrolled at build time so the inner loop's only division is by
    # the compile-time GROUP_K. ``vec`` divides GROUP_K, so one ratio covers the
    # whole vector; and hidden is a multiple of GROUP_K, so a full vector is
    # always in bounds and needs no tail.
    col0 = b.mul(tid, c_vec)
    stride = b.mul(c_bs, c_vec)
    for r in range(spec.tile_m):
        c_r = b.const_i32(r)
        tok = b.vec_extract(b.smem_load_vN(tok_lds, c_r, dtype=I32, n=1), 0)
        tok_safe = b.smax(tok, zero_i)
        src_base = b.mul(tok_safe, hidden)
        dst_base = b.mul(b.add(row_base, c_r), hidden)
        with b.scf_for(col0, hidden, stride, iv_name=f"col{r}") as col:
            g = b.div(col, c_group_k)
            # Zero on a pad row, which is what drives A to exact zeros there.
            ratio = b.vec_extract(
                b.smem_load_vN(scale_lds, c_r, g, dtype=F32, n=1), 0
            )
            v = b.global_load_vN(
                Aq, b.add(src_base, col), FP8E4M3, spec.vec, align=spec.vec
            )
            out = [
                b.cvt_f32_to_fp8(
                    b.fmul(b.cvt_fp8_to_f32(b.vec_extract(v, j)), ratio)
                )
                for j in range(spec.vec)
            ]
            b.global_store_vN(
                A,
                b.add(dst_base, col),
                b.vec_pack(out, FP8E4M3),
                spec.vec,
                align=spec.vec,
            )

    return b.kernel


def moe_gather_rescale_a_grid(
    n_blocks: int, spec: MoeGatherRescaleSpec
) -> Tuple[int, int, int]:
    """One workgroup per expert block.

    ``n_blocks`` is the number of ``tile_m``-row blocks the caller's alignment
    produced, i.e. the same count it passes as the MoE launch's M blocking.
    """
    del spec
    return (int(n_blocks), 1, 1)


def moe_gather_rescale_a_signature(spec: MoeGatherRescaleSpec):
    del spec
    return (
        SignatureBuilder()
        .ptr("Aq", "fp8e4m3")
        .ptr("AqScale", "f32")
        .ptr("SortedIds", "i32")
        .ptr("TopkWeights", "f32")
        .ptr("A", "fp8e4m3")
        .ptr("AScale", "f32")
        .ptr("SortedTokenIds", "i32")
        .ptr("SortedWeights", "f32")
        .scalar("n_flat", "i32")
        .scalar("topk", "i32")
        .scalar("hidden", "i32")
        .scalar("n_hb", "i32")
        .build()
    )
