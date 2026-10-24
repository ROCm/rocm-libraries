# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""Parity subclasses of the gfx950 dualwave helpers.

`kernels/attention/flash_attn_utils.py` is imported by four production kernels
(`flash_attn_generic`, `flash_attn_gfx950`, `flash_attn_fp8_gfx950`, and the
split-K combine), so it is **imported, never edited**. Everything this port
needs that differs from it lives here as a subclass.

--- What actually differs -------------------------------------------------

Less than the feature list suggests, because the production addressing is a
*special case* of the parity one rather than a different scheme. The dualwave
kernel addresses a flattened BSHD tensor:

    element(token, head, d) = token * stride_q_n + head * HEAD_DIM + d

which is the general BHSD-strided form with two slots pinned:

    element(b, h, s, d) = b * stride_0 + h * stride_1 + s * stride_2 + d
                                              ^^^^^^^^^^^^^^^^^^^^
                              stride_1 == HEAD_DIM, stride_2 == stride_q_n

So generalizing is a **change of variables, not new machinery**. Both `b` and
`h` are workgroup-uniform, so they fold into the buffer descriptor's base
address, which the production code already rebases per batch. What remains
per-access is `s * stride_2 + d`, which is the shape the existing helpers
already compute -- they just have to be handed `stride_2` where they currently
read `stride_q_n`, and a zero head offset where they currently add
`head * HEAD_DIM`.

That is why the overrides below are small and why none of them re-implements a
loop body. Three consequences worth naming, since each removes a whole class of
change:

- **`num_records` still bounds the sequence axis.** Rebasing at
  `(b, h)` and bounding at `seqlen * stride_2` elements makes an out-of-range
  row an out-of-buffer access, which returns zero in hardware rather than
  faulting. The production kernel relies on this for the ragged tail and it
  keeps working unmodified.
- **K and V get independent strides**, which the production code cannot express
  (it has one `stride_kv_n` for both). `load_k` and `load_v` are already
  separate methods that read `self.stride_kv_n_v`, so the subclass swaps the
  attribute and delegates rather than copying either body.
- **The head remap survives.** `q_head_idx = h_kv * gqa_group + group_id` is a
  permutation of head order for locality, not a correctness device. It is kept
  with runtime head counts so a parity build schedules its workgroups exactly
  as the measured baseline does.
"""

import contextlib
from dataclasses import fields as dataclass_fields
from dataclasses import replace

from kernels.attention_bwd_cdna import fmha_common_gfx1201 as fmha
from kernels.attention_bwd_cdna.fmha_common_gfx1201 import MaskedAxis
from kernels.attention_bwd_cdna.fmha_traits_gfx950 import BF16_BYTES as _ELEM_BYTES
from kernels.attention_bwd_cdna.fmha_traits_gfx950 import DMA_BYTES as _ACCESS_BYTES
from kernels.attention_bwd_cdna.gfx950_standalone import buffer_ops, dualwave
from kernels.attention_bwd_cdna.philox import Philox

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value
from flydsl._mlir.dialects import vector
from flydsl.expr import math as fmath


# --- Port (flydsl 0.3.4 / vendored attention_cdna): float helpers --------------
# `dualwave._fadd/_fsub/_fmul/_fmax` were removed from the shared utils (FlyDSL
# daee90e4). Same ops, same explicit fast-math flags, written out as arith ops.
def _fadd(a, b, fm):
    return arith.addf(as_mlir_value(a), as_mlir_value(b), fastmath=fm)


def _fsub(a, b, fm):
    return arith.subf(as_mlir_value(a), as_mlir_value(b), fastmath=fm)


def _fmul(a, b, fm):
    return arith.mulf(as_mlir_value(a), as_mlir_value(b), fastmath=fm)


def _fmax(a, b, fm):
    return arith.MaxNumFOp(as_mlir_value(a), as_mlir_value(b), fastmath=fm).result


# Elements touched by one D-axis access. Every load and store on that axis is
# `DMA_BYTES` wide, which is the whole content of the 8xD input contract
# (`_check_8x_d_contract` states it in the same two numbers); `_slab_span_elems`
# is the one place that has to agree with it.
_D_ACCESS_ELEMS = _ACCESS_BYTES // _ELEM_BYTES

__all__ = [
    "ParityGemmHelper",
    "ParityKernelContext",
    "ParityKvGmemToLdsLoader",
    "ParityKvLdsToVgprLoader",
    "ParityQLoader",
    "ParitySoftmaxHelper",
    "ParityStoreHelper",
    "traits_cache_key",
    "wire_ptr",
    "wire_view",
]


def traits_cache_key(traits):
    """Every field of `traits`, for a builder's JIT disk-cache key.

    **Not `traits.cache_tag`.** That property is upstream's and names a
    hand-picked subset of its own fields, so an axis it does not name is
    invisible to the cache: two builds differing only on one of them hash to
    the same entry, and the second silently receives the *first one's binary*.
    At the vendored commit it omits 46 of the 93 fields, among them

    - every axis AOTriton added -- `BIAS_TYPE` (P5), `ENABLE_DROPOUT` (P6),
      `WINDOW` (P3), `LPT_TILE_ORDER` (P7);
    - `DUALWAVE_SWP_RESCALE_THRESHOLD`, and the `D_STAGES`/`QK_SHARDS`/
      `VO_SHARDS` splits.

    Measured: a `bias=True` build returns output **bit-identical** to a
    `bias=False` build of the same shape -- for a zero bias, a constant bias, a
    per-row bias and a key-varying one alike -- because it is running the
    bias-free binary. Building the bias kernel alone against an empty cache
    gives 3.7e-3 against an fp64 reference, so the kernel was never the
    problem. Sweeping `DUALWAVE_SWP_RESCALE_THRESHOLD` over 2/4/8/12/15 is
    likewise bit-identical at every value.

    The three builders already carried a partial patch for this -- the geometry
    -- and its comment describes exactly this failure ("two families of the
    same shape would collide in the JIT disk cache"). A subset is the wrong
    shape for the job, though: it has to be extended by hand every time an axis
    is added, silently returns wrong binaries when someone forgets, and nothing
    tests it. The traits are a frozen dataclass of hashable scalars that fully
    determine the build, so naming all of them cannot go stale.

    Only the *JIT* path is affected. The AOT build gives each functional its own
    invocation and does not share this cache: every `BIAS_TYPE`,
    `ENABLE_DROPOUT` and `CAUSAL_TYPE` pair in the shipped `flyc_attn_fwd.zip`
    is a distinct binary (0 of 72, 0 of 108 and 0 of 72 groups collided). What
    this breaks is in-process work -- `devtools/` sweeps, the JIT tests beside
    them, and anything varying these axes in one session.

    Field *names* as well as values: a rename or a reordering then changes the
    key, and reading a stale entry across one is the same bug in a slower form.
    """
    return tuple((f.name, getattr(traits, f.name)) for f in dataclass_fields(traits))


# --- the tensor operands are pointers on the wire --------------------------
#
# Every tensor operand of the three gfx950 parity kernels is declared
# `fx.Pointer`, not `fx.Tensor`. This is an ABI requirement, not a style
# choice: an `fx.Tensor` kernarg costs **two** slots -- the pointer, plus a
# packed shape+stride descriptor -- and AOTriton dispatches the compiled hsaco
# by filling the kernarg block itself, from a C++ struct that has a pointer and
# the strides it was already passing separately. The descriptor is 40 bytes of
# kernarg per operand that the caller has no way to fill and the kernel never
# reads.
#
# **It never reads it because every extent is already on the wire.** The
# kernels bound their buffer descriptors with products of `max_seqlen_q/k`,
# `hdim_qk/vo`, the head counts and the fifteen strides -- `seqlen_q * stride_q_seq`
# and so on -- and `batch_size` never appears as an extent at all, only as
# `batch_idx * stride_batch` with `batch_idx` coming from the grid. So the
# shapes in the descriptor duplicate scalars the ABI carries anyway, which is
# what makes dropping them safe rather than merely cheap.
#
# Measured, one build each at bf16 / h=2 / d=64 / s=256, `.kernarg_segment_size`
# from the final ISA, with VGPR counts alongside to show the wrapping is free:
#
#     forward   512 -> 296   (vgpr 164 both)
#     dQ        664 -> 352   (vgpr 190 both)
#     dK/dV     664 -> 352   (vgpr 236 both)
#
# **`Workspace` and `BlockTable` are deliberately still tensors.** Split-K and
# paged are not parity gaps -- the target is `scaled_dot_product_attention`, and
# neither is part of it -- so in every build that ships they are `Constexpr` and
# occupy no kernarg at all. There is nothing to shrink.
#
# The one thing that would have to change first if paged were ever built: the
# block table's consumer is `fx.rocdl.make_buffer_tensor(block_table)` in
# `flash_attn_utils.init_descriptors`, and that genuinely wants a tensor rather
# than an address -- unlike every operand converted here, whose only use of the
# tensor was `fx.get_iter`. A `wire_view` would not serve it, because
# `make_buffer_tensor` reads the layout it is handed. That is a real piece of
# work, not a rename.


def wire_ptr(t, elem_type=None):
    """A tensor's base address as an `fx.Pointer` argument.

    `elem_type` defaults to the tensor's **own** element type, and that default
    is load-bearing rather than cosmetic. `PointerJitArg` derives the pointer's
    assumed alignment from its element type -- `(width + 7) // 8` -- and so does
    the `fx.Tensor` path it replaces, which is what makes the two agree: a bf16
    operand keeps alignment 2 either way. Passing a byte pointer instead, as
    `fmha_abi_gfx1201.ptr_arg` does, would silently drop it to 1. Nothing fails
    when that happens; the loads just lose an alignment fact the backend was
    using, which is the kind of regression that gets attributed to the wrong
    change six weeks later.
    """
    if t is None:
        return flyc.from_c_void_p(elem_type or fx.Uint8, 0)
    if elem_type is None:
        # Keyed on the dtype's name so this module needs no torch import; it is
        # imported by the kernel builders, which run before any tensor exists.
        elem_type = _TORCH_DTYPE_TO_FX[str(t.dtype)]
    return flyc.from_c_void_p(elem_type, t.data_ptr())


_TORCH_DTYPE_TO_FX = {
    "torch.bfloat16": fx.BFloat16,
    "torch.float16": fx.Float16,
    "torch.float32": fx.Float32,
    "torch.uint8": fx.Uint8,
    "torch.int32": fx.Int32,
    "torch.int64": fx.Int64,
}


def wire_view(ptr):
    """A nominal view over `ptr`, for production code that spells `fx.get_iter`.

    `flash_attn_utils.py` is imported and never edited, and two of its methods
    reach a tensor operand through `fx.get_iter(...)`: `init_descriptors`, for
    the four dense Q/K/V/O views, and `_store_lse_row`, for LSE's base address.
    `fx.get_iter` rejects a pointer outright -- *"GetIterOp: expected
    TensorLikeType"* -- so a pointer operand cannot be handed to either
    directly. This wraps it back into something they accept.

    **The layout is a placeholder and the round trip is what makes that safe.**
    `fx.get_iter(fx.make_view(p, L))` returns `p` for any `L`, so every one of
    those call sites gets the address it wanted and none of them is affected by
    what `L` says -- all four of `init_descriptors`' views are rebased and
    re-bounded by `ParityKernelContext.init_descriptors` immediately afterwards,
    and `_store_lse_row` uses only `ptrtoint`. The value is verified to survive
    the round trip bit-exactly, and the ISA is unchanged by the wrapping.

    **What is therefore not safe is reading a size back off one of these.**
    `.shape` on a real `fx.Tensor` operand used to return the caller's true
    BHSD extents; here it returns the placeholder, which is a wrong answer
    rather than an error. No kernel does -- see the note above on why none needs
    to -- and `test_no_shape_reads_off_wire_views` is what keeps it that way.
    """
    return fx.make_view(ptr, fx.make_layout(fx.Int32(1), fx.Int32(1)))


# --- granule-general addressing --------------------------------------------
#
# Three production helpers fold constants that are only correct at granule 64.
# Each is replaced here by the same expression with the constant named, so at
# granule 64 they are the *same* arithmetic and a default build is unchanged --
# the bit-identity gate is what holds that claim.
#
# `_k_read_base` and `_ks_offset` are validated offline by
# `tooling/lds_model.py`, which reproduces family A exactly and confirms the
# granule-32 K read covers the tile once. The V pair below has no such model;
# it is measured instead.


def _anchor_v_o(traits, v_o):
    """`dualwave._anchor_v_o`, with the one-accumulator case spelled out.

    The production anchor asks for `!llvm.struct<(vector<16xf32>) x D_CHUNKS>`
    from an inline asm with `D_CHUNKS` outputs. At `D_CHUNKS == 1` LLVM rejects
    that outright -- *"inline asm with one output cannot return struct"* -- and
    the compiler aborts rather than diagnosing, so it surfaces as a crash.

    head_dim 32 is the first width to reach it: `D_CHUNKS = 32 / PV_MFMA_N = 1`.
    A single output returns the value's own type, so this is the same anchor
    with the struct wrapper dropped, not a weaker one.
    """
    if const_expr(traits.D_CHUNKS != 1):
        return dualwave._anchor_v_o(traits, v_o)
    acc = as_mlir_value(v_o[0])
    return [
        dualwave.llvm.inline_asm(acc.type, [acc], "", "=v,0", has_side_effects=True)
    ]


def mfma_operand_wait_state(pack):
    """Two wait states between a packed MFMA operand and the MFMA reading it.

    A sibling of `exp2_wait_state`, one step further down the same chain: that
    one keeps `v_exp_f32` away from the `v_cvt_pk_bf16_f32` that reads it, this
    one keeps the `v_cvt_pk_bf16_f32` away from the `v_mfma` that reads *its*
    result as SrcA or SrcB. Same barrier shape and the same reason for it -- a
    bare `_s_nop` creates no data dependence, so only an asm the value flows
    through pins the gap.

    **The hazard, as measured.** In `flyc_bwd_dkdv` at `BLOCK_DMODEL=128`,
    bf16, `CAUSAL_TYPE=0`, `ENABLE_DROPOUT=True`, `PADDED_HEAD=False`,
    `BIAS_TYPE=1` the scheduler emitted

        v_cvt_pk_bf16_f32 v70, v0, v1
        s_waitcnt lgkmcnt(1)
        v_mfma_f32_16x16x32_bf16 v[54:57], v[78:81], v[70:73], v[54:57]

    and `v[54:57]` came out as uninitialised-looking garbage -- NaN and values
    around 1e23 against a correct magnitude of 2e-3 -- in dK head-dim columns
    16..31 and nowhere else. Swapping the `v_cvt_pk_bf16_f32` with the
    `ds_read_b64_tr_b16` above it, which changes nothing but the slot distance
    (the `lgkmcnt(1)` leaves the same DS op outstanding either way), makes the
    kernel correct. That patch was applied to the shipped `.hsaco` by hand and
    five previously-failing shapes came out clean, including the two that had
    been failing hardest.

    **It is a wait-state hazard and not a memory-visibility one.** Forcing
    *every* `s_waitcnt` in the same kernel to `vmcnt(0) lgkmcnt(0)` -- all 131
    of them, again by patching the `.hsaco` -- left the corruption exactly as
    it was, so the LDS data was demonstrably present and the fault is on the
    VGPR side. `s_waitcnt` retires on the scalar pipe, so an already-satisfied
    one between the two supplies no wait state at all.

    **Scope is deliberately narrow.** A blanket "VALU write then MFMA read
    needs two wait states" is not what the hardware does: scanning all 648
    built gfx950 FlyDSL kernels finds ~7500 sites with fewer than two
    vector-pipe instructions in the gap, across 426 kernels that overwhelmingly
    pass. So this guards one shape -- a `_bf16_trunc_pack_v8` result on its way
    into an MFMA operand -- and not the general pattern. The fp16 half of that
    pack goes through it too, for symmetry and because its producer is the same
    class of instruction, though only bf16 has been seen to fail. What separates
    the failing site from the benign majority is not established, and LLVM
    models neither the hazard nor its scope.

    `s_nop 1` is two wait states. One runs per eight-element pack, against the
    128 MFMAs in the same loop iteration.
    """
    ir_val = as_mlir_value(pack)
    return dualwave.llvm.inline_asm(
        ir_val.type, [ir_val], "s_nop 1", "=v,0", has_side_effects=False
    )


def exp2_wait_state(values):
    """One wait state between a batch of `exp2` results and their consumers.

    `v_exp_f32` is a quarter-rate transcendental -- it retires 16 lanes a cycle
    -- so a VALU instruction issued in the very next slot reads a destination
    the trans unit has only partly written. CDNA requires one wait state there
    and `GCNHazardRecognizer` does not model it for gfx950, so the schedule is
    free to emit `v_exp_f32 vN, ..` immediately before the
    `v_cvt_pk_bf16_f32 vM, v(N-1), vN` of `_bf16_trunc_pack_v8`. When it does,
    that element of the eight-wide B operand carries the *pre-exp* score into
    the MFMA, which is why the symptom is a single wrong element rather than a
    wrong tile.

    **`_s_nop` alone does not fix this, and the earlier claim that it did was
    wrong.** `s_nop` is side-effecting inline asm with no operands, so it
    orders itself against memory but creates no data dependence on the values;
    LLVM moves pure VALU across it freely. Measured on the shipped kernels: of
    the eight dK/dV builds scanned, the `_s_nop(1)` stayed adjacent to an
    `exp2` in exactly one, and the two zero-gap sites in
    `BLOCK_DMODEL=192, PADDED_HEAD=False` survived it.

    So the barrier has to be one the value flows *through*. Every input is tied
    to the matching output (`"0"`, `"1"`, ... constraints), which forces the
    allocator to give each pair one register: the asm costs no moves, emits the
    single `s_nop 0` that supplies the wait state, and no consumer can be
    hoisted above it because it reads the asm's result and not the `exp2`'s.
    Only the last `exp2` in the batch is actually at risk -- the others have
    the rest of the batch between them and the barrier -- so one barrier covers
    the whole list.

    **`has_side_effects` is deliberately off, and it is the difference between
    free and a 48% regression.** A side-effecting asm is a scheduling-region
    boundary, and rung 224 sits on the 512-VGPR cliff already spilling: adding
    one there took `vgpr_spill_count` from 100 to 440 and cost 38-48% of the
    backward at head_dim 216, `causal=False`. Dropping the flag restores the
    pre-fix allocation *exactly* -- spill 100 and 468 scratch bytes at
    `PADDED_HEAD=True`, 80 and 68 at `False`, both identical to the build
    without this call -- and the ordering does not depend on the flag anyway:
    what holds a consumer below the `s_nop` is the SSA def-use edge through the
    tied operand, which no pass can break. Four other shapes were measured and
    rejected -- side-effecting at widths 16, 8, 4 and 1 all spill 438-440, and
    a `sched_barrier(0)`/`s_nop`/`sched_barrier(0)` sandwich spills 296.

    `s_nop 0` is one wait state, which is what the hazard asks for. It runs
    once per score sub-block, against 128 MFMAs in the same loop iteration.
    """
    irs = [as_mlir_value(v) for v in values]
    n = len(irs)
    if const_expr(n == 1):
        return [
            dualwave.llvm.inline_asm(
                irs[0].type, irs, "s_nop 0", "=v,0", has_side_effects=False
            )
        ]
    # Same shape as `dualwave._anchor_v_o`, and the same reason for the struct:
    # a multi-output asm returns one, and LLVM rejects a struct return from a
    # single-output asm -- hence the `n == 1` case above.
    ret_ty = dualwave.ir.Type.parse(f"!llvm.struct<({', '.join(['f32'] * n)})>")
    constraints = ",".join(["=v"] * n + [str(i) for i in range(n)])
    ret = dualwave.llvm.inline_asm(
        ret_ty, irs, "s_nop 0", constraints, has_side_effects=False
    )
    return [dualwave.llvm.extractvalue(irs[i].type, ret, [i]) for i in range(n)]


# --- the DS transpose reads, vendored ---------------------------------------
#
# `flash_attn_utils` at the pinned FlyDSL kernel commit (`FLYDSL_KERNELS_COMMIT`) emits
# `ds_read_b64_tr_b16` as inline asm and takes no alias-scope arguments. The
# gfx950 feature branch replaced that with the ROCDL op (FlyDSL `0a9c5906`),
# and that commit is on no tag and not on `upstream/main` -- so no value of the
# pin can reach it. Rather than depend on a live FlyDSL checkout to be
# numerically correct, the five emitters live here until the delta lands
# upstream. RETIRE THEM when `0a9c5906` merges into FlyDSL's `upstream/main`
# and `FLYDSL_KERNELS_COMMIT` is bumped to a tag containing it: delete the
# five, and put the `dualwave.` prefix back on the six call sites.


def _lds_ptr_ty():
    # Parsed per call rather than cached in a module global: `ir.Type.parse`
    # needs a live MLIR context, and the context differs between JIT builds.
    return dualwave.ir.Type.parse("!llvm.ptr<3>")


def _lds_ptr_with_imm(addr_i32, imm):
    """addrspace(3) pointer to `addr_i32 + imm`, for the DS transpose reads.

    A GEP off the base rather than integer arithmetic folded into an
    `inttoptr`: the DS instructions carry a 16-bit immediate `offset:` field,
    and the backend can only fold a constant back into it when it can see the
    addend as a pointer offset. Computing `inttoptr(addr + imm)` instead costs
    a `v_add_u32` per read -- ~1150 of them in a head_dim 192 build.
    """
    ty = _lds_ptr_ty()
    base = dualwave.llvm.inttoptr(ty, as_mlir_value(fx.Int32(addr_i32)))
    if imm == 0:
        return base
    return dualwave.llvm.getelementptr(
        ty,
        base,
        [],
        [imm],
        dualwave.ir.IntegerType.get_signless(8),
        dualwave.llvm.GEPNoWrapFlags.inbounds,
    )


def _tag_lds_alias(op, scope_name, scope_names):
    """Mark a DS read as touching only `scope_name` among `scope_names`.

    The same alias-scope scheme `_load_k_pack_aligned` puts on its
    `ds_read_b128`, and it is a *performance* requirement, not decoration.
    `SIInsertWaitcnts` treats `buffer_load ... lds` as an LDS-writing VMEM op
    and, before any DS read that may alias one still in flight, inserts
    `s_waitcnt vmcnt(0)` -- a full drain that collapses the KV prefetch the
    pipeline is built on. It resolves "may alias" through the machine memory
    operand's AA info, so a read scoped to the buffer it actually reads is
    provably disjoint from a DMA writing the other half of the double buffer,
    and the drain is not emitted.

    Without this the backend emits 5 extra `vmcnt(0)` at head_dim 64, worth
    ~10% -- the entire regression from moving off inline asm.
    """
    if scope_name is None:
        return op
    op.operation.attributes["alias_scopes"] = dualwave._dualwave_lds_alias_scopes(
        scope_name
    )
    op.operation.attributes["noalias_scopes"] = dualwave._dualwave_lds_noalias_scopes(
        scope_name, scope_names
    )
    return op


def _ds_read_tr16_b64_imm(
    result_type, addr_i32, imm_offset=0, scope_name=None, scope_names=()
):
    """gfx950 `ds_read_b64_tr_b16` with DUALWAVE_SWP immediate byte offset.

    **Uses the ROCDL op, not inline asm, and that is a correctness
    requirement.** `SIInsertWaitcnts` discovers outstanding LDS traffic by
    scanning the MIR for DS instructions; an inline asm is opaque to it, and a
    `~{memory}` clobber is not an lgkm event. Emitted as asm, the backend does
    not know a read is in flight and inserts no `s_waitcnt lgkmcnt` before uses
    of the result -- leaving the kernel's own cluster-boundary wait as the only
    protection.

    That is sound only while nothing reads the destination before that wait.
    Above head_dim 128 the kernel exceeds the 256 architectural-VGPR cap, the
    allocator spills to AGPRs, and it places `v_accvgpr_write` copies of the
    destination immediately after the read and ahead of the wait -- 22 such
    unwaited uses at head_dim 192, 160 at 256, against 0 at 64 and 128. The
    result was non-deterministic NaN that no amount of `s_barrier` or
    `s_waitcnt` at the DSL level could fix, because the offending read is one
    the compiler inserted.

    The op form lets the backend track the dependency and place the waits
    itself.
    """
    raw_type = dualwave.ir.VectorType.get([2], dualwave.ir.IntegerType.get_signless(32))
    ptr = _lds_ptr_with_imm(addr_i32, int(imm_offset))
    # The intrinsic is typed v4f16; `Cannot select` on a vector<2xi32> result.
    op = rocdl.ds_read_tr16_b64(
        dualwave.ir.VectorType.get([4], dualwave.ir.F16Type.get()), ptr
    )
    raw = _tag_lds_alias(op, scope_name, scope_names).result
    raw = vector.BitCastOp(raw_type, raw).result
    return vector.BitCastOp(result_type, raw).result


def _ds_read_tr_v4f16_imm(
    lds_base_elem_idx,
    imm_bytes,
    lds_kv_base_idx,
    v_lds_read_vec4_type,
    scope_name=None,
    scope_names=(),
):
    byte_offset = lds_base_elem_idx * 2 + lds_kv_base_idx
    addr_i32 = fx.Int32(byte_offset)
    return _ds_read_tr16_b64_imm(
        v_lds_read_vec4_type,
        addr_i32,
        imm_bytes,
        scope_name=scope_name,
        scope_names=scope_names,
    )


def _k_read_base(traits, lane_mod_32, lane_div_32):
    """`_k_lds_read_base_per_lane` with `SMEM_N_RPT` in place of a literal 8."""
    return (
        (lane_mod_32 % traits.SMEM_N_RPT) * traits.SMEM_K_LINE_STRIDE
        + (lane_mod_32 // traits.SMEM_N_RPT) * traits.D_128B_SIZE
        + lane_div_32 * traits.VEC_KV
    )


def _ks_offset(traits, ks):
    """`_swizzled_ks_offset` with `K_STEPS_PER_BAND` in place of a literal 4."""
    per_band = traits.K_STEPS_PER_BAND
    return (ks // per_band) * traits.K_LDS_TO_REG_KSTEP_OUTER_STRIDE + (
        ks % per_band
    ) * traits.K_LDS_TO_REG_KSTEP_INNER_STRIDE


def _v_dc_offset(traits, dc):
    """`_swizzled_v_dc_off` with `D_CHUNKS_PER_BAND` in place of a literal 2."""
    per_band = traits.D_CHUNKS_PER_BAND
    return (dc // per_band) * traits.V_LDS_TO_REG_DCHUNK_PAIR_STRIDE + (
        dc % per_band
    ) * traits.V_LDS_TO_REG_DCHUNK_IN_PAIR_STRIDE


def _v_imm_lo(traits, dc, k_substep):
    """`_swizzled_v_imm_lo`, in bytes, over the general dc offset."""
    return (
        k_substep * traits.V_LDS_TO_REG_K_SUBSTEP_STRIDE + _v_dc_offset(traits, dc)
    ) * traits.BF16_BYTES


def _bias_slab_num_records_bytes(seqlen_q, seqlen_kv, stride_b_seq_q, elem_bytes):
    """Byte bound for a `(q_row, kv_col)` bias slab, in `num_records` terms.

    `stride_b_seq_q` is the distance between bias rows, **not** the width of
    one. They coincide only when the bias is contiguous in `(.., seqlen_q,
    seqlen_k)` order; a BSHD caller hands us `(batch, seqlen_q, head,
    seqlen_k)`, where the row stride is `num_heads * seqlen_k` and each row's
    real extent is `seqlen_k` -- the rest of the stride belongs to the *other
    heads* of that row.

    So `seqlen_q * stride_b_seq_q`, the obvious bound, overshoots the slab by
    `stride_b_seq_q - seqlen_k`. For an interior head that merely reads a
    neighbouring head's bias into columns the KV tail mask then sets to `-inf`,
    which is invisible. For the **last** (batch, head) slab it runs off the end
    of the tensor, and if the allocation happens to end a mapping there it is a
    hard memory fault rather than a clamped read: `flyc_attn_fwd` faulting on
    the first byte past a `(3, 5, 32, 16)` bias with strides
    `(2560, 16, 80, 1)`, 128 bytes past, is exactly this.

    The true end of the slab is the last row's last column, so the bound is
    `(seqlen_q - 1) * stride_b_seq_q + seqlen_k`. That subtraction wraps when
    `seqlen_q` is 0 -- a varlen empty sequence -- and an index that wraps
    becomes a `num_records` covering all of memory, so an empty slab is bounded
    at 0 by a select on `seqlen_q > 0`, as `_slab_span_elems` does. The same
    form is right for a row-broadcast bias (`stride_b_seq_q == 0`, every query
    row reads one `(.., 1, seqlen_k)` row): the bound is `seqlen_k`. A `minui`
    against `seqlen_q * stride_b_seq_q` would pin that case to 0 and turn every
    bias load into 0.

    **Do not round this up, and do not widen it to `seqlen_q * stride_b_seq_q`.** The KV axis
    of a bias carries no padding contract at all -- `seqlen_k` is the width the
    caller allocated, full stop -- so there is no slack here to spend, and this
    is where `_slab_span_elems` and this function part company. That one rounds
    its D axis to `_D_ACCESS_ELEMS` because the 8xD contract makes those columns
    the caller's; nothing says the same of bias column `seqlen_k`.

    Ending on the last valid element costs the *readers* something: gfx950 range-checks a
    multi-dword buffer op per dword and drops any dword not wholly inside
    `num_records`, so for odd `seqlen_k` a wide read of the last row loses
    column `seqlen_k-1` to the out-of-range column `seqlen_k` sharing its
    dword. That is a load-side problem and `ParitySoftmaxHelper._add_bias_
    inplace` solves it load-side, with 2-byte reads on the tiles that can hold
    that column. Buying those columns back here instead would mean reading
    memory the caller never handed us -- 2 bytes for a rounded-up bound, up to
    `(stride_b_seq_q - seqlen_k) * elem_bytes` for `seqlen_q * stride_b_seq_q`, which is the fault
    described above.
    """
    rows = fx.Index(seqlen_q)
    stride = fx.Index(stride_b_seq_q)
    tight = (rows * stride - stride + fx.Index(seqlen_kv)) * fx.Index(elem_bytes)
    return as_mlir_value(fx.Index((rows > fx.Index(0)).select(tight, fx.Index(0))))


def _slab_span_elems(rows, stride_seq, hdim):
    """Elements in a `(batch, head)` slab, ending on its last **accessed** element.

    `_bias_slab_num_records_bytes`'s problem, on the Q/K/V/O/DO/DK/DV slabs,
    with the same answer: `stride_seq` is the distance between rows, the row
    itself is `ceil8(hdim)` wide, and the two coincide only when the
    tensor is contiguous in `(.., seqlen, hdim)` order. A BSHD caller hands us
    `(batch, seqlen, head, hdim)`, where the row stride is `num_heads * hdim`
    and the rest of it belongs to the *other heads* of that row -- so the
    obvious `rows * stride_seq` overshoots the slab by `stride_seq - hdim`.

    For an interior head that overshoot reads a neighbouring head's rows, which
    `PADDED_HEAD`'s `discard` throws away regardless. For the **last** (batch,
    head) slab it runs off the end of the tensor. Concretely, a `(3, 5, 64, 8)`
    Q at strides `(2560, 8, 40, 1)`: the slab at `(batch 2, head 4)` is based
    at element 5152 and bounded at 5152 + 64*40 = 7712, while the tensor ends
    at 7680. `BLOCK_DMODEL` is 32 for `hdim_qk` 8, so `ParityQLoader` issues
    the last row's load out to column 31 -- 24 elements past the allocation,
    with the hardware range check permitting every one of them because they are
    inside a `num_records` that should never have covered them. Whether that
    faults is up to the page mapping, which is why it surfaced as intermittent
    `pytest-xdist` worker deaths on the padded head dims rather than as a
    reproducible failure. Ending on the last row's last accessed element clamps
    exactly those accesses, and clamped is what they always assumed they were.

    **The row is `ceil8(hdim)` wide, not `hdim`; no axis but D rounds.** That is
    not a softening of the bound above, it is what the bound above is measuring.
    Every D-axis load and store is one `DMA_BYTES` chunk, and the 8xD input
    contract is the promise that makes that legal: `t[b, h, s, :]` is
    `ceil8(hdim)` contiguous elements for *every* `(b, h, s)`, the last one
    included, and nothing else is contiguous on any axis -- a caller may put a
    guard page between two rows, two heads or two batches. The D grant is why
    this rounds; the absence of any other grant is why nothing else does.
    `flash_attn_func_gfx950`'s module docstring states the contract in full.

    **Bounding at `hdim` stops in the middle of a real column, which is the bug
    this shape fixes.** gfx950 range-checks a multi-dword buffer op per dword
    and drops any dword not wholly inside `num_records`, so a bound ending at
    `hdim` takes the dword holding columns `hdim-1` and `hdim` with it -- the
    real column `hdim-1` included -- on the last row of *every* slab, which is
    the only row whose chunk the bound can reach. Measured on `test_prime_hdim`
    at `(3, 5, 257, 571)`, allocation and NaN-poisoned slack exactly as the
    contract asks: every one of 7/23/37/53/67/73/83/113/149/179/211/241 lost
    `O[.., seqlen_q-1, hdim-1]` (never stored, so it read back as the fill NaN),
    and with that repaired still lost `dQ[.., seqlen_q-1, hdim-1]` and
    `dK/dV[.., seqlen_kv-1, hdim-1]`. An earlier probe of this concluded the two
    forms were indistinguishable and it was wrong: it read the divergence as
    confined to the final slab, when the bound is per-slab and clips all of
    them, and a NaN in `O` then reaches every element of `dK` through `delta`.

    **Under a contiguous `(.., seqlen, ceil8(hdim))` layout this changes
    nothing, by construction.** There `stride_seq == ceil8(hdim)`, so
    `(rows - 1) * stride_seq + ceil8(hdim)` *is* `rows * stride_seq` -- the same
    descriptor, down to the `oob_off == num_records` equality the D-tail store
    suppression relies on.

    **`stride_seq` is not clamped against, and must not be.** That equality is
    the one thing tempting you to: `q_row_slab`'s `oob_off` is
    `rows * stride_seq`, and `ParityStoreHelper._final_o_global` suppresses the
    D-tail chunks by redirecting their store to it and relying on the hardware
    to drop anything at or past `num_records`, so a span *above* `oob_off` would
    put the sentinel back inside the buffer and make every suppressed tail store
    live. But the ordering already holds, and holds for the right reason: the
    contract grants `ceil8(hdim)` contiguous elements per row, so row `s + 1`
    cannot begin nearer than that and `stride_seq >= ceil8(hdim)` is guaranteed,
    which is exactly `span <= rows * stride_seq`. A `minui` on top of that is
    unreachable on every legal input and actively wrong on the rest -- where
    `stride_seq < ceil8(hdim)` it silently reinstates the clipping bug above,
    trading a diagnosable fault for wrong numbers.

    That is the same verdict a clamp against the true row pitch
    `min(s0, s1, s2)` got, for the same reason and with a second bill attached:
    reading `stride_batch` and `stride_head` here grew the live ranges
    `BwdDkDvKernelContext._init_q_head_invariants` hoists to shrink, and
    perturbed the register allocator into an `LiveIntervalUnion::extract`
    assertion failure at `BLOCK_DMODEL=224 ENABLE_DROPOUT=True BIAS_TYPE=1`, the
    highest-pressure dK/dV configuration built. **`stride_seq` is the distance
    to the next row and never the extent of one**; neither it nor any other
    stride belongs in a row width. Refusing a layout that cannot hold
    `ceil8(hdim)` belongs in `_check_8x_d_contract`, where it can say so out
    loud and costs no registers.

    The `rows > 0` select is not a clamp of that kind and stays: at `rows == 0`
    -- a varlen empty sequence -- the subtraction wraps, and an index that wraps
    becomes a `num_records` covering all of memory, which is a bigger version of
    the bug above rather than a smaller one.

    **What this bound does not do, and never did, is protect the gap between two
    rows.** Nothing about a stride promises the bytes from `hdim` to
    `stride_seq` are mapped -- a caller may legitimately have a guard page there
    -- but a single linear `num_records` cannot express a per-row extent. Even
    ending at `hdim`, the pad access on row `s` sits at `s * stride_seq + hdim`,
    which is below the bound for every `s < rows - 1`: the old form clipped the
    pad on the **last row only**, and permitted it on all the others. So a
    caller with a guard page after column `hdim - 1` faulted before this change
    too, at row 0 rather than at row `rows - 1`. The only shape that would buy
    that case is narrowing the tail *access* the way `narrow_tail` does on the
    bias KV axis; the D axis cannot, because Q/K/V staging goes through LDS DMA
    copy atoms that move whole `DMA_BYTES` chunks. What the bound does protect
    is the region past the end of the slab, which for the last `(batch, head)`
    slab is past the end of the tensor -- that is the fault it was written for.
    """
    rows_v = fx.Index(rows)
    width = fx.Index(_D_ACCESS_ELEMS)
    row_elems = ((fx.Index(hdim) + width - fx.Index(1)) // width) * width
    span = rows_v * fx.Index(stride_seq) - fx.Index(stride_seq) + row_elems
    return fx.Index((rows_v > fx.Index(0)).select(span, fx.Index(0)))


def _score_column_runs(kv_vectorized):
    """`[(element_index, column_offset, width)]` covering one score vector.

    The 16 f32 an MFMA lane holds are 16 *columns* of a single row -- the row
    is `lane_mod_32`, and `lane_div_32` shifts the column set -- so a bias read
    is a row-wise gather in principle and a handful of contiguous spans in
    practice. Which spans is exactly what the causal mask's threshold table
    already says, since that table maps element to column; this reads the same
    fact for a different purpose rather than restating it.

    Grouping the thresholds into consecutive runs gives four spans of four at
    the default granule and two of eight when the KV tile is vectorized. Both
    are single `buffer_load`s, which is the whole reason bias needs no LDS
    staging: the accumulator's layout hands back contiguity for free.
    """
    thresholds = []
    for thr_x, thr_y in dualwave._causal_pair_thresholds(kv_vectorized):
        thresholds.extend((thr_x, thr_y))
    runs, start = [], 0
    for i in range(1, len(thresholds) + 1):
        if i == len(thresholds) or thresholds[i] != thresholds[i - 1] + 1:
            runs.append((start, thresholds[start], i - start))
            start = i
    return runs


class ParityGemmHelper(dualwave.DualwaveGemmHelper):
    """The two GEMMs, addressed one D stage at a time.

    Under `D_STAGES > 1` a KV tile's D axis is covered in several passes, so
    neither GEMM sees all of it at once. The two are asymmetric about what that
    means, because D is a *reduction* axis for QK and an *output* axis for PV:

    - `qk_stage` accumulates into a running S that the caller carries across
      the stages, rather than seeding a fresh zero. All stages contribute to
      every element of S, and softmax cannot run until the last one has.
    - `pv_step_k` writes a disjoint slice of the O accumulator per stage, so
      the stages never meet; the stage only shifts which `v_o[dc]` is hit.

    Both take stage-relative register lists (that is what the loaders return)
    and map them to global indices here, so a stage index never has to be
    threaded into the loaders' addressing.
    """

    def scale_scores(self, v_s):
        """`qk_scale * S`, on the f32 accumulator, once the D axis is complete.

        **This is where `sm_scale * log2e` enters, and the Triton kernels are
        the reason it is here rather than on Q or in `sub_m`.**
        `fwd_kernel_inner.py:150` reads `qk += (Qk_scale * tl.dot(q0, k0))`:
        the scale multiplies the f32 dot product, and every later consumer --
        the bias add, the masks, the row max, `exp2`, the LSE -- sees scores
        already in the base-2 domain. Keeping that domain is the whole appeal:
        nothing downstream of this call changes at all.

        The two alternatives are both closed, each by its own measurement:

        - **Folding it into Q** costs `|S| * 2**-9` in the exponent. See
          `ParityQLoader.scale_all`, which now refuses.
        - **Folding it into `sub_m`'s subtract** as an FMA -- `m_ij =
          max(m_i, Qk_scale * max(qk))` and `exp2(fma(qk, Qk_scale, -m_ij))` --
          is free, and `fwd_kernel_inner.py:186` forbids it by name: *"DO NOT
          USE the following FMA optimization pattern, which has numerical
          errors for large inputs"*, aotriton issue 54. `_scale_sub_score_pair`
          upstream is that pattern; it is reached only by the fp8 helper, whose
          scores have a different dynamic range.

        Called **before** the masks, which is the other half of what Triton
        does: `fwd_kernel_inner.py:134` writes `-inf` into `qk` before the
        `+= Qk_scale * dot`, so no infinity is ever an FMA operand. Here the
        GEMM runs first and the masks follow, which reaches the same place --
        the multiply only ever sees finite raw dot products, so a masked `-inf`
        cannot become a `0 * inf` NaN at `sm_scale == 0.0` (a value
        `modules/flash/tests/test_forward.py` parametrizes).

        Once per tile, not once per D stage: the stages of `qk_stage` sum into
        one accumulator, so scaling the total is both cheaper and closer to the
        fp32 reference than scaling each partial sum would be. **Which means
        the caller owns the call** -- `qk_stage` cannot do it, because it
        cannot tell a middle stage from the last one, and `qk` cannot either,
        because `fmha_bwd_dq_gfx950` calls it for raw scores on purpose and
        applies its own scale in `scale_and_sub_lse`.
        """
        s_lo, s_hi = v_s
        # 16 is the forward's score-pair half width, spelled as a literal
        # everywhere else it appears (`_sub_score_pair`, `_scale_sub_score_pair`,
        # `seq_pad_mask_inplace`). `ACC_ELEMS` is dQ's name for the same thing
        # and is not a field of the forward traits.
        scale_v = Vec.from_elements(
            [fx.Float32(self.c_sm_scale_log2e)], fx.Float32
        ).broadcast_to(16)
        return (
            as_mlir_value(Vec(s_lo) * scale_v),
            as_mlir_value(Vec(s_hi) * scale_v),
        )

    def qk_stage(self, v_k, q_all_bf16, acc, stage=0):
        """One D stage of `S += Q·K^T`, with the Q pack held off the MFMA.

        Returns **raw** scores: `scale_scores` is the caller's, for the reasons
        given there.

        **`mfma_operand_wait_state` is load-bearing here**, and it no longer
        has the explanation it was written with. It used to be that Q was
        scaled and rounded back to bf16 once before the KV loop, so the
        `v_cvt_pk_bf16_f32` that built these packs looked loop-invariant -- but
        the whole chain was 32 VGPRs live across the loop, and LLVM sank it
        back in and rematerialized it per MFMA rather than pay that. What
        landed was the operand written in the slot before the MFMA that reads
        it, with only scalar instructions between:

            v_cvt_pk_bf16_f32 v100, v48, v49
            s_subb_u32 s1, s1, s9                    <- scalar, no wait state
            s_lshl_b64 s[0:1], s[0:1], 1             <- scalar, no wait state
            v_mfma_f32_32x32x16_bf16 v[18:33], v[34:37], v[100:103], v[18:33]

        which is zero effective wait states against the two the documented
        VALU-to-MFMA-SrcA/B rule asks for -- the same instruction pair
        feeding the same MFMA shape that produced a wrong answer in dQ. 28 of
        the 216 forward kernels came out this way. The `k_hi`
        MFMA reads the same quad one slot later and is shielded by the `k_lo`
        one, so a single barrier per pack covers both.

        With the pre-scale gone, Q reaches the MFMA straight from its LDS read
        and there is no convert to sink, so the *measured* chain above cannot
        form from this source any more. The barrier stays: the packs are still
        built by VALU (`_get_q_pack` shuffles), `dualwave` reaches the same
        `mfma_operand_wait_state` for dK/dV and dQ where the hazard was caught
        producing a wrong answer, and a scan of the rebuilt kernels is what
        should retire it -- not the disappearance of one of its two known
        producers. `fmha_bwd_dq_gfx950` passes an unscaled Q through this same
        method and needs it regardless.

        The `_s_nop(1)` in `qk` below is a different thing at a different
        place -- it sits *after* all the QK MFMAs and, as its comment records,
        works by perturbing register allocation rather than by supplying a wait
        state. It does not cover this gap.
        """
        k_lo, k_hi = v_k
        v_s_lo, v_s_hi = acc
        steps = self.traits.K_STEPS_PER_STAGE
        for ks in range_constexpr(steps):
            q_pack = mfma_operand_wait_state(
                dualwave._get_q_pack(self.traits, q_all_bf16, stage * steps + ks)
            )
            v_s_lo = dualwave._mfma_acc(
                k_lo[ks], q_pack, v_s_lo, self.mma_atom, self.mfma_acc_vec_type
            )
            v_s_hi = dualwave._mfma_acc(
                k_hi[ks], q_pack, v_s_hi, self.mma_atom, self.mfma_acc_vec_type
            )
        return (v_s_lo, v_s_hi)

    def qk(self, v_k, q_all_bf16, stage=0):
        """Unstaged entry point: seed at zero and run the one stage there is.

        `D_STAGES == 1` used to go through `super().qk`, which builds its own
        unbarriered Q packs. At one stage `K_STEPS_PER_STAGE == K_STEPS_QK` and
        `stage * steps + ks == ks`, so `qk_stage` from a zero seed is the same
        loop; routing both through it is what puts the barrier on every build
        rather than only the staged ones.

        **Raw, like `qk_stage`.** `fmha_bwd_dq_gfx950` is the other caller and
        wants it that way -- it applies `qk_scale` itself, fused into the LSE
        subtract. The forward reaches `scale_scores` through `_FwdGemmHelper`.
        """
        out = self.qk_stage(
            v_k, q_all_bf16, (self.c_zero_v16f32, self.c_zero_v16f32), stage
        )
        # Works, and **not for the reason it looks like.** Without it head_dim
        # 96 computes a wrong answer; with it, 96 is correct across five shapes
        # in both masking modes at ~0 cost.
        #
        # It is *not* supplying a wait state. `s_nop` here is a side-effecting
        # intrinsic present during scheduling, and stripping nops from the two
        # builds gives the same 2817 instructions with **88 differing register
        # assignments** -- so this perturbs allocation, the same way anchoring
        # the K packs does, and not the same way `amdgpu-snop-padding=1` does
        # (that one is post-RA and leaves registers byte-identical).
        #
        # The underlying defect is therefore still latent. Every *documented*
        # gfx950 MFMA hazard is satisfied in the failing build: for
        # `v_mfma_f32_32x32x16_bf16`, which is XDL 8-pass on gfx950, the
        # requirements are 12 (write -> MFMA SrcA/B), 10 (-> SrcC, non-matching
        # opcode), 12 (-> VALU/DS/VMEM) and 2 (VALU -> MFMA), and a scan finds
        # zero violations. The leading hypothesis is `ds_read_b64_tr_b16`,
        # which `GCNHazardRecognizer` does not model at all.
        #
        # Located by bisection, not from the pair: a wait state here, after
        # `exp2`, or after `reduce_sum` each fix it, while after `cast_p`,
        # `lazy_rescale_o`, `load_k`, `load_v`, `reduce_max`, `sub_m` or
        # `pv_step_k` do not -- so the producer is at or before the QK MFMAs
        # and the consumer is before the P cast. Scans of every documented
        # hazard class (VALU->MFMA SrcA/B, MFMA->VALU, MFMA->MFMA by operand
        # position, VALU->DPP) found no difference between a broken and a
        # working build, so the exact instruction pair is still unidentified.
        # See `sdpa_lore_gfx950.md`.
        #
        # Costs nothing: +1.6% at head_dim 64 and +0.8% at 128, both inside
        # run-to-run noise, and it buys head_dim 96 at 932 TF against 579 for
        # the padded 128 tile.
        dualwave._s_nop(1)
        return out

    def pv_step_k(self, step, v_p, v_v, v_o, stage=0):
        if const_expr(self.traits.D_STAGES == 1):
            return super().pv_step_k(step, v_p, v_v, v_o)
        v_p_lo, v_p_hi = v_p
        v_pk = v_v[step]
        p_pk = v_p_lo[step] if const_expr(step < 2) else v_p_hi[step - 2]
        per_stage = self.traits.D_CHUNKS_PER_STAGE
        for dc in range_constexpr(per_stage):
            out = stage * per_stage + dc
            v_o[out] = dualwave._mfma_acc(
                v_pk[dc], p_pk, v_o[out], self.mma_atom, self.mfma_acc_vec_type
            )
        return v_o

    def pv(self, v_p, v_v, v_o, stage=0):
        for step in range_constexpr(4):
            v_o = self.pv_step_k(step, v_p, v_v, v_o, stage=stage)
        return v_o


class _ParityKvStaging:
    """KV DMA addressing that allows more than one issue per wave.

    A mixin because **two objects need it**: the context builds the m0 tables
    in `init_dma_m0_tables`, and the loader computes source addresses in
    `_async_load_kv_linear`. The loader subclasses the *production* context
    rather than `ParityKernelContext`, so inheritance alone would not share
    them -- and a second copy is exactly how the write and read sides of an
    LDS layout drift apart.

    **What the production formula assumes.** It places one KV tile line per
    wave per d-band, `line = wave + d * SMEM_N_RPT`, which is correct exactly
    when `SMEM_N_RPT == NUM_WAVES`. Family A satisfies that by arithmetic
    coincidence: 8 waves, and BLOCK_N 64 at 8 tokens per issue is 8 lines. A
    4-wave family covering the same BLOCK_N needs **two issues per wave**, and
    under the production formula lines 4..7 are never written at all -- the
    reads then return whatever LDS happened to hold, which is how head_dim 192
    produced non-deterministic NaN.

    **The generalisation is a change of index, not of scheme.** The flat DMA
    index is band-major, `d_flat = band * ISSUES + issue`, and

        line  = (wave + issue * NUM_WAVES) + band * SMEM_N_RPT
        token = n_in_warp * SMEM_N_RPT + (wave + issue * NUM_WAVES)

    Both collapse to the production form at `ISSUES == 1`, where `SMEM_N_RPT`
    and `NUM_WAVES` are equal. Verified against a model of the write/read
    mapping before being written: the model reproduces family A exactly, and
    is what ruled out the BLOCK_N 128 variants -- a wave's K read covers 64
    tokens, so BLOCK_N 128 would need a doubled score accumulator.
    """

    def _issue_split(self, d_flat):
        """`(band, issue)` for a flat DMA index. Band-major."""
        return d_flat // self.ISSUES_PER_WAVE, d_flat % self.ISSUES_PER_WAVE

    def _dma_line(self, d_flat):
        """The KV tile line this wave writes for `d_flat`."""
        band, issue = self._issue_split(d_flat)
        return (
            self.wave_id_uni
            + issue * self.traits.NUM_WAVES
            + band * self.traits.SMEM_N_RPT
        )

    def _dma_m0(self, buf_base_elems, line_stride, d_flat):
        addr = (
            self.lds_kv_base_idx
            + (buf_base_elems + self._dma_line(d_flat) * line_stride)
            * self.traits.BF16_BYTES
        )
        return rocdl.readfirstlane(T.i32, as_mlir_value(fx.Int32(addr)))

    def k_dma_base(self, buf_id, d):
        return self._dma_m0(
            dualwave._k_buf_base(self.traits, buf_id), self.traits.SMEM_K_LINE_STRIDE, d
        )

    def v_dma_base(self, buf_id, d):
        return self._dma_m0(
            dualwave._v_buf_base(self.traits, buf_id), self.traits.SMEM_V_LINE_STRIDE, d
        )

    # Which D stage the next DMA reads from global. Set by the loader
    # immediately before delegating, the same way `stride_kv_n_v` is, and safe
    # for the same reason: tracing is eager, so it is read while `super()`
    # runs and no branch is open across the swap.
    dma_stage = 0

    def kv_src_elem(self, src_base, d_flat):
        """Global element index for this lane's `d_flat` chunk of a KV tile.

        `band` only spans `SMEM_D_RPT` d-bands, and under `D_STAGES > 1` that
        is one *stage* of the head dim rather than all of it -- LDS holds a
        stage at a time. So the stage's base offset is the single term that
        makes staging reach global memory; everything else is unchanged, and
        at `D_STAGES == 1` the term is zero.
        """
        band, issue = self._issue_split(d_flat)
        line_n = self.wave_id + issue * self.traits.NUM_WAVES
        n_in_tile = self.n_in_warp * self.traits.SMEM_N_RPT + line_n
        global_d = self.d_bucket * self.traits.VEC_KV + band * self.traits.D_128B_SIZE
        if const_expr(self.traits.D_STAGES > 1):
            global_d = global_d + self.dma_stage * self.traits.STAGE_DIM
        return src_base + n_in_tile * self.stride_kv_n_v + global_d


class ParityKernelContext(_ParityKvStaging, dualwave.DualwaveKernelContext):
    """Dualwave context addressing arbitrary BHSD strides, with a runtime scale."""

    def __init__(
        self,
        traits,
        *,
        strides,
        o_strides=(0, 0, 0),
        sm_scale,
        num_head_q,
        num_head_k,
        hdim_qk,
        hdim_vo,
        padded_head=False,
        hdim_qk_floor=0,
        window_left=None,
        window_right=None,
        seqinfo=(None, None, None, None),
        varlen_bits=0,
        num_seqlens=0,
        Bias=None,
        bias_strides=(0, 0, 0),
        philox=(None, None, 0, None, None),
        idropout_p=0,
        dropout_scale=1.0,
        **kwargs,
    ):
        super().__init__(traits, **kwargs)
        # 9 strides in launch order: Q, K, V, each (batch, head, seq).
        # Numerically named per `sdpa-feature-gap.md`'s porting instruction --
        # the `z/h/m/k` suffixes it warns about have caused real bugs.
        (
            self.stride_q_batch,
            self.stride_q_head,
            self.stride_q_seq,
            self.stride_k_batch,
            self.stride_k_head,
            self.stride_k_seq,
            self.stride_v_batch,
            self.stride_v_head,
            self.stride_v_seq,
        ) = strides
        # O's own three, separate from the rest because two of the three
        # kernels built on this class have no O at all: the dQ kernel writes
        # dQ and the dK/dV kernel writes dK and dV, each under its own name
        # with its own strides. Those two leave `O` at None and these at zero,
        # and `init_descriptors` builds no O view for them. Folding dQ's
        # strides in here instead is what made `stride_o_*` a name for
        # whichever tensor happened to be in the slot.
        self.stride_o_batch, self.stride_o_head, self.stride_o_seq = o_strides
        self.sm_scale_arg = sm_scale
        self.num_head_q = num_head_q
        self.num_head_k = num_head_k
        # P1. `hdim_qk` is the real reduction extent, which may be narrower
        # than the compiled tile; `hdim_vo` is the real output width. They are
        # separate because the two GEMMs are not symmetric -- see
        # `ParityQLoader` and `ParityStoreHelper`.
        self.hdim_qk = hdim_qk
        self.hdim_vo = hdim_vo
        self.PADDED_HEAD = bool(padded_head)
        # Compile-time lower bound on `hdim_qk`, exclusive. The dispatcher
        # enforces it, so D columns below it need no mask.
        self.HDIM_QK_FLOOR = int(hdim_qk_floor)
        # P3. Raw window bounds, still carrying their sentinels; resolved in
        # `init_runtime_indices`, which is the first point the sequence lengths
        # this build will use are known.
        self.window_left_arg = window_left
        self.window_right_arg = window_right
        # P4. `VarlenBits` plus the four sequence-info arrays, named by role:
        # `?0` supplies lengths, `?1` supplies positions. Unread slots are
        # **null pointers**, which is safe only because the decoder branches
        # rather than selects -- see `fmha.cond_load`.
        self.seqinfo_q0, self.seqinfo_q1, self.seqinfo_k0, self.seqinfo_k1 = seqinfo
        self.varlen_bits_arg = varlen_bits
        self.num_seqlens_arg = num_seqlens
        # P5. The bias matrix and its (batch, head, seqlen_q) strides. Slot 3
        # is the KV axis and is contractually contiguous, so it is not passed.
        self.Bias = Bias
        # `_seq_q`, not `_seq`. Bias is the one rank-4 tensor here with **two**
        # sequence axes, `(batch, head, seqlen_q, seqlen_k)`, so a bare `_seq`
        # does not say which -- and the k axis is contiguous by contract and
        # therefore never passed, which makes the ambiguity easy to miss rather
        # than obviously wrong. Not `_seqq`: the doubled letter reads as a typo
        # and is skipped. Not `_seq0`: a numeric slot is exactly what the P7
        # stride rename removed, on the grounds that nothing at runtime
        # distinguishes one stride from another and spelling the axis out is
        # the only check there is.
        self.stride_b_batch, self.stride_b_head, self.stride_b_seq_q = bias_strides
        # P6. The counter is the (pointer, immediate) pair torch splits it
        # into, not one pre-summed scalar: a captured graph re-reads the
        # pointer half. The two `*_output` slots report back what was actually
        # used, so a backward pass can regenerate the identical mask.
        (
            self.philox_seed_ptr,
            self.philox_offset1,
            self.philox_offset2,
            self.philox_seed_output,
            self.philox_offset_output,
        ) = philox
        self.idropout_p = idropout_p
        self.dropout_scale_arg = dropout_scale

    # -- runtime softmax scale -------------------------------------------
    #
    # The production kernel derives the scale from head_dim
    # (`rsqrt(head_dim) * log2e`), which is only correct when the compiled
    # tile *is* the real extent. Under a padded head it is wrong, and AOTriton
    # passes `Sm_scale` regardless, so it becomes an argument.
    #
    # `* log2e` is kept folded in: every downstream exp is `exp2`, and folding
    # here means the conversion is paid once per kernel rather than per tile.
    # Pre-scaling Q by it *before* the row max -- which the production kernel
    # already does -- is the anti-FMA correction, so nothing moves.

    def init_types_and_constants(self, head_dim_runtime=None):
        super().init_types_and_constants(head_dim_runtime=head_dim_runtime)
        # DMA issues each wave makes per d-band, and the flat count that
        # follows. `NUM_DMA_*` feed both the m0 tables and the `s_waitcnt`
        # budget the pipeline is balanced against, so they have to agree.
        self.ISSUES_PER_WAVE = self.traits.SMEM_N_RPT // self.traits.NUM_WAVES
        self.NUM_DMA_K = self.traits.SMEM_D_RPT * self.ISSUES_PER_WAVE
        self.NUM_DMA_V = self.NUM_DMA_K
        self.c_sm_scale = fx.Float32(self.sm_scale_arg)
        self.c_sm_scale_log2e = fx.Float32(
            arith.mulf(
                fx.as_ir_value(fx.Float32(self.sm_scale_arg)),
                fx.as_ir_value(fx.Float32(dualwave._LOG2E)),
                fastmath=self.fm_fast,
            )
        )

    # -- runtime head counts ---------------------------------------------

    def init_thread_mapping(self):
        super().init_thread_mapping()
        # Re-derive the four head indices with runtime counts. The *mapping* is
        # the production one verbatim: `h_idx` is decomposed against the KV head
        # count and recomposed against the group size, which groups the Q heads
        # sharing a KV head. Only the operands change from constexpr to runtime,
        # so a build with matching counts schedules identically.
        num_head_k = fx.Index(self.num_head_k)
        gqa_group = fx.Index(self.num_head_q) // num_head_k
        # Published, because the dK/dV group loop's trip count is this same
        # quantity and `q_head_idx` -- the head that loop walks from -- is
        # derived from it two lines down. Recomputing it at the loop is how a
        # trait and a kernarg drifted apart in the first place (issue 9); one
        # source is what stops them disagreeing again.
        self.gqa_group = gqa_group
        self.h_kv_idx = self.h_idx % num_head_k
        self.group_id = self.h_idx // num_head_k
        self.q_head_idx = self.h_kv_idx * gqa_group + self.group_id
        self.kv_head_idx = self.h_kv_idx
        if const_expr(self.traits.LPT_TILE_ORDER and self.traits.CAUSAL):
            # P7. Longest-processing-time-first: under a causal mask q-block
            # `i` walks about `i` KV tiles and `grid.y` issues in increasing
            # order, so the cheapest workgroups go first and the most expensive
            # land in the tail. Reversing the index puts them first.
            #
            # A bijection over the same index set, so the output is
            # bit-identical and only the schedule moves -- which is what makes
            # it safe as a knob. **Measured at 0.0% on gfx950** across every
            # width and shape tried, so it ships off; with 8 XCDs and this many
            # workgroups the tail imbalance it targets is already absorbed.
            # gfx1201's forward kernel measures 12-16% from the same change.
            #
            # Folded into this override rather than added as a second one: an
            # `init_thread_mapping` of its own silently replaced the head-index
            # derivation above, which only shows up once the runtime head
            # counts differ from the build's.
            self.q_block_idx = fx.Index(gpu.grid_dim.y) - fx.Index(1) - self.q_block_idx

    # -- granule-general staging ------------------------------------------

    def init_dma_thread_offsets(self):
        """Split a lane into (token, d-bucket) for the granule it stages.

        Production splits `lane // VEC_KV` by `lane % VEC_KV`, which is right
        only when a granule spans exactly `VEC_KV` lanes -- true at 64, which
        is `VEC_KV * VEC_KV`, and nowhere else. A lane always moves `VEC_KV`
        contiguous D elements, so `granule // VEC_KV` lanes cover one token's
        granule and the rest of the wave advances the token.
        """
        traits = self.traits
        self.lane_in_warp = self.tid % traits.WARP_SIZE
        self.n_in_warp = self.lane_in_warp // traits.SMEM_D_BUCKETS
        self.d_bucket = self.lane_in_warp % traits.SMEM_D_BUCKETS

    def init_lds_read_bases(self):
        super().init_lds_read_bases()
        # `_k_lds_read_base_per_lane` folds `SMEM_N_RPT` as a literal 8.
        self.k_lds_read_base_per_lane = _k_read_base(
            self.traits, self.lane_mod_32, self.lane_div_32
        )

    # -- per-tensor strides ----------------------------------------------

    def init_sequence_lengths(self, **kwargs):
        """Decode `VarlenBits` into the six scalars the rest of the kernel uses.

        The base class knows exactly one varlen shape -- cumulative
        `cu_seqlens` on both sides -- and reads it directly. `VarlenBits`
        generalizes that to three orthogonal axes per side (STACKED, LENGTH,
        POSITION), which is five useful configurations rather than one, so the
        decode replaces that branch rather than extending it.

        `fmha.decode_addressing` is gfx1201's, reused unedited: the bits mean
        the same thing on both architectures, and a second copy of a wire
        format is a second thing to keep in step.

        **Unconditional, and dense is not a separate case.** `varlen_bits == 0`
        decodes to BHSD / MAX / IMPLIED, for which `decode_addressing` returns
        `(max_seqlen, 0, z)` -- which is dense addressing, spelled once. Every
        array read inside it is branched over rather than selected, so the null
        `seqinfo` pointers a dense call passes are never dereferenced and the
        bits-zero path costs a not-taken scalar branch. Compiling the decode
        away for dense would buy that branch back and cost a second binary,
        which is the trade gfx1201 already declined.

        Four things about the shape of this:

        - **`z` is not `batch_idx`.** The workgroup's `z` selects a *sequence*;
          the decode says which *batch slice* that sequence lives in, which is
          `z` for a batched layout and 0 for a packed one. Overwriting
          `batch_idx` here is what keeps `_slab_byte_base` correct without a
          varlen branch inside it.
        - **`z` outlives that overwrite**, as `seq_idx_i32`. Anything indexed by
          *sequence* rather than by batch slice needs it, and after this method
          runs there is nowhere else to get it: `init_philox` is the caller that
          found this out the hard way.
        - **The reads are scalar.** `z` is workgroup-uniform, so these land in
          SGPRs and cost nothing against the VGPR budget.
        - **Row offsets stay separate from the batch index.** A packed tensor
          has `batch = 0` and a large `row_off`; a padded one has a real batch
          and `row_off = 0`. Both go through the same
          `batch * s_batch + row_off * s_seq`, which is why the descriptors
          need no varlen case at all.
        """
        z = fx.Int32(self.batch_idx)
        self.seq_idx_i32 = z
        q_len, q_row, q_batch = fmha.decode_addressing(
            self.varlen_bits_arg, 0, self.seq_len_v, self.seqinfo_q0, self.seqinfo_q1, z
        )
        k_len, k_row, k_batch = fmha.decode_addressing(
            self.varlen_bits_arg,
            8,
            self.seq_len_kv_v,
            self.seqinfo_k0,
            self.seqinfo_k1,
            z,
        )
        self.lse_tokens_i32 = fmha.lse_token_pitch(
            self.varlen_bits_arg,
            0,
            self.seq_len_v,
            self.seqinfo_q0,
            self.seqinfo_q1,
            self.num_seqlens_arg,
        )
        # **Each side owns its own batch index**, and this is not pedantry:
        # `0x040B` -- packed Q against `seqused_k` on a BHSD cache -- has Q
        # stacked (batch 0, large row offset) and K batched (batch z, no row
        # offset) in the *same call*. Reusing Q's index for K reads batch 0 of
        # the cache for every sequence, which is plan section 1.4 arriving as a
        # wrong answer. Measured before the split: correct for the four
        # configurations where both sides agree, wrong for the one that does
        # not.
        self.batch_idx = fx.Index(q_batch)
        self.kv_batch_idx = fx.Index(k_batch)
        self.varlen_q_row_off = fx.Index(q_row)
        self.varlen_kv_row_off = fx.Index(k_row)
        self.seqlen_q_v = fx.Index(q_len)
        self.seqlen_kv_v = fx.Index(k_len)
        self.seqlen_kv_i32 = fx.Int32(k_len)
        # `q_tok_base` / `q_tok_end` are the base class's names for the same
        # interval; the paged and split-K helpers read them.
        self.q_tok_base = fx.Index(0)
        self.q_tok_end = self.seqlen_q_v
        self.kv_tok_base = fx.Index(0)
        self.kv_tok_end = self.seqlen_kv_v

    def init_philox(self):
        """Seed, counter and this workgroup's plane origin. Prologue-only.

        **The offset scheme is `Philox.grid_plane`/`grid_offset`, and using
        them is the reproducibility contract** rather than a convenience. A
        dropout mask is generated here and *regenerated* by the backward pass
        and by the debug mask kernel; all three must agree bit for bit or the
        gradients are quietly wrong. They agree because they call the same two
        functions, not because three transcriptions of one formula happened to
        match.

        The consequence to write down, since it is invisible in any test run at
        a single tile size: **the mask is a function of element coordinates
        only.** `grid_plane` is given `max_seqlen_q`/`max_seqlen_k`, never
        `BLOCK_M`/`BLOCK_N`, so re-tuning the tile geometry cannot move a single
        random. From this phase onward that is a constraint on the tuner, not
        just a property of today's code.

        **The plane is indexed by the grid's sequence, not by `batch_idx`.**
        Those two differ exactly when they matter: `init_sequence_lengths`
        resolves `batch_idx` to the *batch slice* a sequence lives in, and a
        stacked layout puts every sequence in slice 0. Reading it here would
        give a whole packed batch one plane, so N sequences would draw the same
        mask -- finite, self-consistent between the forward and the backward,
        and statistically wrong in a way no allclose can see. `seq_idx_i32` is
        the raw `z`, which is what gfx1201 uses.
        """
        if const_expr(not self.traits.ENABLE_DROPOUT):
            return
        self.philox_rng = Philox.for_arch("gfx950")
        plane = self.seq_idx_i32 * fx.Int32(self.num_head_q) + fx.Int32(self.q_head_idx)
        seed = fmha.philox_seed_value(self.philox_seed_ptr)
        offset = fmha.philox_offset_base(self.philox_offset1, self.philox_offset2)
        fmha.philox_report(
            self.philox_seed_output, self.philox_offset_output, seed, offset
        )
        self.philox_seed = seed
        self.philox_plane_base, self.philox_row_stride = self.philox_rng.grid_plane(
            offset, plane, self.seq_len_v, self.seq_len_kv_v
        )

    def init_tile_bounds(self, **kwargs):
        """Resolve the window, then let the causal bound derive from it.

        Here and not in `init_runtime_indices` for an ordering reason: a
        sentinel resolves against the sequence lengths, and those are not
        settled until `init_sequence_lengths`, which runs in between. This is
        also the first place `delta_i32` is *read* -- `causal_end_raw_i32` and
        `max_num_tiles` both come from it -- so overriding it immediately
        before `super()` is what makes the tile count follow the window's right
        bound with no further change.
        """
        if const_expr(self.traits.WINDOW):
            # **`delta_i32` *is* the right bound.** The base class sets it to
            # `seqlen_kv - seqlen_q`, which is bottom-right causal spelled as a
            # diagonal offset, and every causal site downstream -- the mask's
            # `rel`, `causal_end_raw_i32`, `max_num_tiles` -- reads it from
            # here. Re-pointing it at the resolved `window_right` therefore
            # generalizes all of them at once, and is why a window is a flag on
            # top of causal rather than a second masking mode.
            #
            # Resolution is on the device, not the host: a sentinel resolves
            # against *this sequence's* lengths, and under varlen those differ
            # per sequence. `fmha.resolve_window` is gfx1201's, reused unedited.
            left, right = fmha.resolve_window(
                self.window_left_arg,
                self.window_right_arg,
                fx.Int32(self.seqlen_q_v),
                self.seqlen_kv_i32,
            )
            self.window_left_i32 = fx.Int32(left)
            self.delta_i32 = fx.Int32(right)
        super().init_tile_bounds(**kwargs)
        if const_expr(self.traits.WINDOW and not self.traits.SPLITK):
            self._skip_dead_leading_tiles()

    def _skip_dead_leading_tiles(self):
        """Start the KV walk at the window's left edge instead of at tile 0.

        The right bound already truncates the walk, through `max_num_tiles`.
        The left bound is the mirror image and needs the *base* to move, and
        the machinery for that is already here: `[split_t0, split_t_end)` is
        the tile range the whole pipeline is written against -- the prologue
        loads `split_tile(0..2)`, the loop runs from `split_tile(3)`, and only
        split-K ever moved the base before. So this is a new *value* for an
        existing knob, not new control flow.

        A dead tile is not a wrong answer, it is a masked one: every column
        fails the left bound, `exp2` gives zero, and the tile contributes
        nothing. That is exactly why correctness cannot show this works --
        skipping is invisible to the output and only a measurement sees it.

        The lowest column any row of this Q block can reach is
        `q_start - window_left`, since `q_start` is the smallest row. Tiles
        entirely below that are dead for the whole block.
        """
        traits = self.traits
        # Clamp before dividing rather than after: `q_start - window_left` is
        # negative whenever the window reaches past the start of the sequence,
        # and `fx.Index` is unsigned, so a negative would come out enormous and
        # skip the entire range. i32 until the value is known non-negative --
        # the same rule `resolve_window` states.
        first_col_i32 = fx.Int32(self.q_start) - self.window_left_i32
        first_col_i32 = fx.Int32(
            (first_col_i32 > fx.Int32(0)).select(first_col_i32, fx.Int32(0))
        )
        t0 = fx.Index(first_col_i32) // fx.Index(traits.BLOCK_N)
        # Even, because the software pipeline consumes two tiles per iteration
        # and `split_t_end` is already even; an odd base would make the segment
        # odd and leave the epilogue a tile short.
        t0 = (t0 // fx.Index(2)) * fx.Index(2)
        # The pipeline's prologue plus epilogue need four tiles to exist. The
        # base class guarantees `max_num_tiles >= 4`, so this cannot underflow.
        cap = self.split_t_end - fx.Index(4)
        self.split_t0 = fx.Index((t0 < cap).select(t0, cap))

    def init_runtime_indices(self, **kwargs):
        super().init_runtime_indices(**kwargs)
        self.stride_q_seq_v = fx.Index(self.stride_q_seq)
        self.stride_k_seq_v = fx.Index(self.stride_k_seq)
        self.stride_v_seq_v = fx.Index(self.stride_v_seq)
        self.stride_o_seq_v = fx.Index(self.stride_o_seq)
        # The production helpers read these two names. Q's seq stride is the
        # default; `ParityKvGmemToLdsLoader` swaps the KV one per tensor.
        self.stride_q_n_v = self.stride_q_seq_v
        self.stride_kv_n_v = self.stride_k_seq_v

    def _slab_byte_base(self, s0, s1, s2, row_off, head_idx, batch_idx=None):
        """Byte offset of this workgroup's (batch, head) slab.

        Both axes are workgroup-uniform, so folding them into the descriptor
        costs scalar arithmetic once instead of a per-access add.

        **`row_off` is the varlen token origin, not the batch's.** The
        production kernel's `q_tok_base` is `batch * seqlen` in dense mode,
        because there the batch axis *is* a token offset into one flat
        allocation. Here the batch has its own stride, so passing `q_tok_base`
        would count it twice. Dense passes 0; varlen will pass the cumulative
        offset with `stride_0` set to 0, which is the same decomposition
        `fmha.decode_addressing` produces on gfx1201.
        """
        if batch_idx is None:
            batch_idx = self.batch_idx
        elems = (
            batch_idx * fx.Index(s0) + head_idx * fx.Index(s1) + row_off * fx.Index(s2)
        )
        return elems * fx.Index(self.traits.BF16_BYTES)

    def _slab_view(
        self, tensor, s0, s1, s2, row_off, head_idx, rows, hdim, batch_idx=None
    ):
        """A buffer view over one (batch, head) slab, bounded at its last element.

        A row past the sequence is out of the descriptor and reads as zero
        rather than faulting -- the same mechanism the production kernel uses
        for its ragged tail, restated over a stride the caller chose.

        `hdim` is the row's real extent, `s2` the distance between rows; see
        `_slab_span_elems` for why the second is not a bound for the first.
        """
        # Span first, then base -- the order the body had before the split, and
        # the order the emitted MLIR therefore still has. Reversing it is
        # semantically free and cost 230 lines of scheduling churn in every
        # forward binary when this was first written the other way round.
        span_elems = _slab_span_elems(rows, s2, hdim)
        base_bytes = self._slab_byte_base(
            s0, s1, s2, row_off, head_idx, batch_idx=batch_idx
        )
        return self._slab_view_at(tensor, base_bytes, span_elems)

    def _slab_view_at(self, tensor, base_bytes, span_elems):
        """`_slab_view` over a base and an extent the caller already computed.

        Split out so a caller that rebinds one slab to a *different head* can
        hoist everything that does not move with the head. `_slab_byte_base` is
        affine in `head_idx` and `_slab_span_elems` does not read it at all, so
        such a caller needs one add per tensor rather than the three strides
        those two expressions between them read.

        That matters on gfx950 for a reason beyond tidiness: keeping the strides
        live across the GQA walk is what drove the backward kernel's scalar
        spilling, and the walk is the largest region in that kernel.
        `BwdDkDvKernelContext._init_q_head_invariants` is the caller this exists
        for, and carries the measurements.
        """
        return dualwave._make_rebased_view(
            fx.get_iter(tensor),
            base_bytes,
            span_elems * fx.Index(self.traits.BF16_BYTES),
            fx.make_layout(fx.Int32(span_elems), fx.Int32(1)),
            _buf_flags_i32=self.buf_flags_i32,
            _elem_ir=self.elem_ir,
        )

    def q_row_slab(self, tensor, s0, s1, s2, hdim):
        """A Q-row-shaped output slab, with the sentinel that suppresses stores.

        Returns `(view, oob_off)`. The three kernels' per-Q-row outputs -- the
        forward's O and the backward's dQ -- are all indexed
        (batch, head, q row, d) over this workgroup's own q head and row
        origin, so they differ only in the tensor, its three strides and its
        row extent. Those are the arguments; nothing else varies, and none of
        them is guessed from another tensor's.

        `oob_off` is the first element past the *untightened* span, so a store
        redirected there is dropped by the hardware bound -- which is how
        `ParityStoreHelper` suppresses the D-tail chunks without branching.
        `_slab_span_elems` can only shrink the descriptor below this, never
        grow it past. No clamp enforces that; the 8xD contract does, by
        guaranteeing `stride_seq >= ceil8(hdim)` -- see that docstring, which
        rejects a clamp outright -- so the two stay in the order the
        suppression needs even when the D axis rounds up.
        On a contiguous `(.., seqlen, ceil8(hdim))` layout they are equal, which
        is out of range and always has been.
        """
        view = self._slab_view(
            tensor, s0, s1, s2, self.q_row_off, self.q_head_idx, self.seqlen_q_v, hdim
        )
        return view, self.seqlen_q_v * fx.Index(s2)

    def init_descriptors(self, **kwargs):
        """Rebuild Q/K/V/O over arbitrary strides; everything else stays.

        `super()` runs first for the state that does not depend on the
        addressing scheme -- `delta_i32`, `buf_flags_i32`, `elem_ir`, the
        paged page-view constants and the debug resource -- and its four dense
        views are then replaced. The discarded ones are pure descriptor
        arithmetic with no side effects, so they fold away; the alternative is
        duplicating the half of the method that has nothing to do with strides.
        """
        traits = self.traits
        # `o_tensor` explicitly, because upstream defaults it to `self.O` and
        # builds one throwaway view from it (`flash_attn_utils.py:3448`) that
        # the code below replaces or drops. A kernel with no O still has to
        # hand it a live pointer for that dead view, so it lends Q's.
        super().init_descriptors(
            o_tensor=self.Q if self.O is None else self.O, **kwargs
        )

        # Token origins, straight from the decode. Dense is not a case here:
        # `decode_addressing` returns a zero row offset at `varlen_bits == 0`,
        # which is the same 0 a dense branch would have written. The pairing
        # with `batch_idx` is what makes one expression serve every mode -- a
        # packed tensor gets `batch = 0` with a large `row_off`, a padded one a
        # real batch with `row_off = 0` -- and the batch axis has a real stride
        # here, so it must not also be spent as a token offset.
        self.q_row_off = self.varlen_q_row_off
        self.kv_row_off = self.varlen_kv_row_off

        # Head folded into the base, so what remains per access is `s * stride`.
        self.q_gmem_elem_offset = self.q_start * self.stride_q_seq_v
        self.kv_gmem_elem_offset = fx.Index(0)

        if const_expr(traits.BIAS_TYPE):
            # Same slab shape as Q, which is the point: bias is indexed by
            # (batch, head, q_row, kv_col), so the varlen row origin and the
            # head fold into the base exactly as Q's do, and what remains per
            # access is `q_row * stride_b_seq_q + col`.
            #
            # A raw resource rather than a `_slab_view`: the reads here are
            # per-lane vectors of 4 or 8 elements at an address the lane
            # computes, which is `buffer_ops.buffer_load`'s shape, not the
            # copy-atom one the K/V DMA uses.
            #
            # The bound is the slab, so a row past `seqlen_q` reads zero
            # instead of faulting -- which is also the right bias for a padded
            # row -- and so does a column that runs off the last row's end.
            # `_bias_slab_num_records_bytes` is what makes the second half of
            # that true when the bias row stride is wider than the row.
            self.bias_rsrc = buffer_ops.create_buffer_resource(
                self.Bias,
                max_size=False,
                num_records_bytes=_bias_slab_num_records_bytes(
                    self.seqlen_q_v,
                    self.seqlen_kv_v,
                    self.stride_b_seq_q,
                    traits.BF16_BYTES,
                ),
                base_byte_offset=as_mlir_value(
                    self._slab_byte_base(
                        self.stride_b_batch,
                        self.stride_b_head,
                        self.stride_b_seq_q,
                        self.q_row_off,
                        self.q_head_idx,
                    )
                ),
            )
        self.q_div = self._slab_view(
            self.Q,
            self.stride_q_batch,
            self.stride_q_head,
            self.stride_q_seq,
            self.q_row_off,
            self.q_head_idx,
            self.seqlen_q_v,
            self.hdim_qk,
        )
        if self.O is not None:
            self.o_div, self.o_oob_off = self.q_row_slab(
                self.O,
                self.stride_o_batch,
                self.stride_o_head,
                self.stride_o_seq,
                self.hdim_vo,
            )
        if const_expr(not traits.PAGED):
            self.k_div = self._slab_view(
                self.K,
                self.stride_k_batch,
                self.stride_k_head,
                self.stride_k_seq,
                self.kv_row_off,
                self.kv_head_idx,
                self.seqlen_kv_v,
                self.hdim_qk,
                batch_idx=self.kv_batch_idx,
            )
            self.v_div = self._slab_view(
                self.V,
                self.stride_v_batch,
                self.stride_v_head,
                self.stride_v_seq,
                self.kv_row_off,
                self.kv_head_idx,
                self.seqlen_kv_v,
                self.hdim_vo,
                batch_idx=self.kv_batch_idx,
            )


class ParityQLoader(dualwave.DualwaveQLoader):
    """Q staging that zeroes the columns past `hdim_qk`.

    **Masking Q is what makes a padded head correct, and masking it is
    enough** for any finite pad -- `QK^T = sum_d Q[d] * K[d]`, so a zero in Q
    annihilates whatever K holds at the same column. K is left unmasked
    deliberately: Q is loaded once in the prologue, K once per KV tile, so this
    is the one of the two that is free. See `sdpa-close-gap-gfx950.md` for the
    residual case (a pad holding NaN or Inf, where `0 * NaN` is NaN) and the
    test that pins it.

    Whole 8-element chunks are still *loaded* past the extent, and are allowed
    to be: the D-axis pitch is contractually a multiple of 8 elements, so the
    chunk containing `hdim_qk` lands inside the allocation. What must not
    happen is those elements reaching the MFMA, which is what `discard` stops.
    """

    def scale_all(self, q_all):
        """Refused. `qk_scale` belongs on the f32 scores; see `scale_scores`.

        The inherited `scale_all` extends Q to f32, multiplies by
        `sm_scale * log2e` and **rounds the product back to the input dtype**.
        Q is already bf16/f16, so that second rounding is pure loss: each
        element gains a fresh relative error of up to `2**-9`, the QK dot
        multiplies it by the logit magnitude, and it arrives *in the exponent*
        of `exp2(S - m)`, where it becomes a relative error on the softmax
        weight of `2**(|S| * 2**-9)`. That is invisible at
        `sm_scale = rsqrt(head_dim)`, where `|S|` is O(1), and several percent
        at `sm_scale = 1`, where `|S|` grows as `sqrt(head_dim)`:

            head_dim   sm_scale   max |err| vs fp64, Q folded   scores scaled
              512        0.044             0.0034                 0.0034
              512        0.40              0.0632                 0.0155
              512        1.00              0.1658                 0.0145

        `BwdDqSoftmaxHelper.scale_and_sub_lse` measured the same thing in dQ
        and stopped folding there, on the grounds that the forward "tolerates
        that because O is a normalised average and the error largely cancels".
        It does not cancel far enough:
        `test_transformers.py::test_mem_eff_attention_single_query_tail` calls
        SDPA at `scale=1.0, head_dim=512, bf16` and misses on 1.4% of its
        output by up to 0.29.

        So both directions now do what the Triton kernels do --
        `fwd_kernel_inner.py:150` is `qk += (Qk_scale * tl.dot(q0, k0))` -- and
        this method exists only to make the fold unreachable. It is *one
        keystroke away* from being called again (the two forward bodies used to
        call it on the line after `load_all`), and nothing downstream checks
        shapes or magnitudes, so a reintroduction would be a silently wrong
        answer at large `sm_scale` and a correct one everywhere the tests look
        hardest. `fmha_bwd_dq_gfx950.py`'s module docstring names the same trap
        for the same reason.
        """
        raise NotImplementedError(
            "Q must not be pre-scaled: rounding `Q * sm_scale * log2e` back to bf16 costs "
            "~2**-9 of relative precision *in the exponent*. Apply `qk_scale` to the f32 "
            "scores with `ParityGemmHelper.scale_scores` instead, as the Triton kernels do."
        )

    def load_pack(self, q_row_in_block, ks):
        pack = super().load_pack(q_row_in_block, ks)
        if const_expr(not self.PADDED_HEAD):
            return pack
        col_base = fx.Index(ks * self.traits.K_STEP_QK) + self.lane_div_32 * fx.Index(
            self.traits.MFMA_LANE_K
        )
        # Q is masked once, before the KV loop, so the mask registers die
        # immediately and the bitmask form is pure win here.
        return MaskedAxis(
            fx.Index(self.hdim_qk), elem_dtype=self.elem_dtype, bitmask=True
        ).discard(pack, col_base, self.traits.MFMA_LANE_K)

    def load_all(self):
        """Q rows for this wave, for any `K_STEPS_QK`.

        The production version assembles the packs through a fixed 8 -> 16 ->
        32 tree and then takes either one 32-pack or a concatenation of two,
        which covers `K_STEPS_QK` 4 and 8 -- head_dim 64 and 128 -- and nothing
        else. head_dim 192 wants 12 packs and 256 wants 16, so the tail of that
        tree simply drops the remainder and the `Vec` constructor rejects the
        result ("shape (96,) has 96 elements, but value has type
        vector<64xbf16>"). It fails loudly, which is the good case.

        A left fold over the packs replaces the tree. `_concat_vectors` builds
        an explicit shuffle index list, so it does not need equal widths and
        the uneven steps 192 needs (32+32 -> 64, 64+32 -> 96) are fine.
        """
        traits = self.traits
        ctx = self.ctx_ref
        ctx.init_q_row()
        acc = self.load_pack(ctx.q_row_in_block, 0)
        for ks in range_constexpr(traits.K_STEPS_QK - 1):
            acc = dualwave._concat_vectors(
                acc, self.load_pack(ctx.q_row_in_block, ks + 1)
            )
        return Vec(acc, (traits.K_STEPS_QK * traits.MFMA_LANE_K,), self.elem_dtype)


class ParityKvLdsToVgprLoader(dualwave.DualwaveKvLdsToVgprLoader):
    """K register reads that zero the columns past `hdim_qk`.

    Masking Q alone is enough for a *finite* pad, since `0 * x == 0`. It is not
    enough for a pad holding NaN or Inf, and a caller's D-axis padding is
    allocation slack whose contents nothing constrains -- so K is masked too.
    V needs nothing: its D columns are O's columns, and those are suppressed at
    the store instead.

    **Why the mask is the same expression as Q's.** The K tile transits LDS in
    a swizzled layout, so the column a register holds is not obviously its
    linear D index. Working the two halves against each other: the DMA writes
    LDS element `base + w*LINE + d*N_RPT*LINE + l*8 + i` holding D column
    `(l % 8) * 8 + d * 64 + i`, while `load_k` reads at
    `(lm % 8) * LINE + (lm // 8) * 64 + ld * 8 + (ks // 4) * N_RPT * LINE +
    (ks % 4) * 16`. Matching term by term gives `w = lm % 8`, `d = ks // 4` and
    `l = (lm // 8) * 8 + ld + (ks % 4) * 2`, and since `(lm // 8) * 8` vanishes
    mod 8,

        D = (ld + (ks % 4) * 2) * 8 + (ks // 4) * 64 + i
          = ks * 16 + ld * 8 + i

    which is `_q_pack_col` exactly -- the swizzle permutes *tokens* across LDS
    lines and leaves D in linear order. `k_hi` differs from `k_lo` by
    `K_LDS_TO_REG_N_STRIP_STRIDE`, an N offset, so it carries the same columns.
    """

    @contextlib.contextmanager
    def _scoped_to_stage(self):
        """Narrow `K_STEPS_QK` / `D_CHUNKS` to one stage for the duration.

        Under `D_STAGES > 1` an LDS buffer holds one stage of the head dim, so
        the inherited readers -- which loop to `K_STEPS_QK` and `D_CHUNKS` --
        would run off the end of it. The *offsets* need no adjustment:
        `_swizzled_ks_offset` and `_swizzled_v_dc_off` address d-bands within
        the buffer, and a stage-sized buffer has exactly `SMEM_D_RPT` of them.
        Only the counts are wrong.

        So this swaps the two counts rather than reimplementing the read
        loops. Copying them is the specific thing to avoid here: the write side
        (`kv_src_elem`) and the read side have to describe one LDS layout, and
        a second copy of either is how they drift apart.

        A no-op at `D_STAGES == 1`, where the two values are already equal --
        deliberately not merely equivalent, so a default build cannot differ.
        """
        if const_expr(self.traits.D_STAGES == 1):
            yield
            return
        full = self.traits
        self.traits = replace(
            full,
            K_STEPS_QK=full.K_STEPS_PER_STAGE,
            D_CHUNKS=full.D_CHUNKS_PER_STAGE,
        )
        try:
            yield
        finally:
            self.traits = full

    def _read_k_packs(self, buf_id, urk_base):
        """The inherited non-vectorized K read, with a granule-general swizzle.

        Six lines rather than a `super()` call because the production loop
        calls `_swizzled_ks_offset`, which folds `K_STEPS_PER_BAND` as a
        literal 4 -- a module function, so there is nothing to override but the
        loop that calls it. Identical arithmetic at granule 64.
        """
        traits = self.traits
        k_base = dualwave._k_buf_base(traits, buf_id)
        k_lo = [None] * traits.K_STEPS_QK
        k_hi = [None] * traits.K_STEPS_QK
        for ks in range_constexpr(traits.K_STEPS_QK):
            k_lo[ks], k_hi[ks] = self._load_k_pair(
                buf_id, k_base + urk_base + _ks_offset(traits, ks)
            )
        return k_lo, k_hi

    def load_k(self, buf_id, urk_base=None, stage=0):
        with self._scoped_to_stage():
            if const_expr(self.traits.KV_VECTORIZED):
                k_lo, k_hi = super().load_k(buf_id, urk_base=urk_base)
            else:
                base = self.k_lds_read_base_per_lane if urk_base is None else urk_base
                k_lo, k_hi = self._read_k_packs(buf_id, base)
            steps = self.traits.K_STEPS_QK
        if const_expr(not self.PADDED_HEAD):
            return (k_lo, k_hi)
        # K is masked *inside* the KV loop, once per KV tile per K-step, so
        # unlike Q's prologue mask this one is on the hot path and its masks
        # stay live across everything else. Masking every step is what made a
        # padded head cost 27-54% against its own rung -- and near-independent
        # of how much pad there was, since 240-in-256 paid the same tax as
        # 129-in-192. Removing it entirely restored every padded build to its
        # rung's native rate exactly, which is what identifies this loop, and
        # nothing else about padding, as the whole cost.
        #
        # **Most of the steps cannot contain pad.** The build serves
        # `(HDIM_QK_FLOOR, BLOCK_DMODEL]` and the dispatcher enforces it, so a
        # step whose columns all lie at or below the floor is reading real
        # data. On the 32-spaced rungs consecutive rungs are 32 apart and
        # `K_STEP_QK` is 16, so exactly two steps survive at every width --
        # 2 of 16 at the 256 rung rather than 16.
        #
        # Skipping is `continue`, not a narrower loop bound: under `D_STAGES`
        # the surviving steps are the last ones globally but sit at arbitrary
        # stage-relative indices, so there is no contiguous stage-local range
        # to iterate.
        width = self.traits.MFMA_LANE_K
        masked = [
            ks
            for ks in range(steps)
            if ((stage * steps + ks) + 1) * self.traits.K_STEP_QK > self.HDIM_QK_FLOOR
        ]
        if const_expr(not masked):
            return (k_lo, k_hi)
        # The bitmask form ANDs one precomputed dword instead of selecting per
        # element, but each mask is a live register. Gated on the number of
        # steps that actually survive rather than on `K_STEPS_QK`: measured
        # +21% in the 64-wide tile and -43% in the 128-wide one, where 32 extra
        # live registers turned a spill-free build into 61 spills. With two
        # steps left the wide tiles are back under that limit.
        cols = MaskedAxis(
            fx.Index(self.hdim_qk),
            elem_dtype=self.elem_dtype,
            bitmask=len(masked) * (width // 2) <= 16,
        )
        for ks in masked:
            # The mask is against the *global* D column, so the stage's base
            # has to come back in here -- `ks` is stage-relative above.
            col_base = fx.Index(
                (stage * steps + ks) * self.traits.K_STEP_QK
            ) + self.lane_div_32 * fx.Index(width)
            k_lo[ks] = cols.discard(k_lo[ks], col_base, width)
            k_hi[ks] = cols.discard(k_hi[ks], col_base, width)
        return (k_lo, k_hi)

    def read_v_packs(self, buf_id, urv_base):
        """The inherited non-vectorized V read, with a granule-general swizzle.

        Same reason as `_read_k_packs`: the production loop calls
        `_swizzled_v_imm_lo`, which reaches `_swizzled_v_dc_off` and its
        literal 2. Identical arithmetic at granule 64.
        """
        traits = self.traits
        lds_base = dualwave._v_buf_base(traits, buf_id) + urv_base
        v_scope = dualwave._dualwave_lds_scope("v", buf_id)
        packs = [[None] * traits.D_CHUNKS for _ in range(4)]
        for dc in range_constexpr(traits.D_CHUNKS):
            for k_substep in range_constexpr(4):
                imm_lo = _v_imm_lo(traits, dc, k_substep)
                pair = traits.V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE * traits.BF16_BYTES
                read = lambda off: _ds_read_tr_v4f16_imm(  # noqa: E731
                    lds_base,
                    off,
                    lds_kv_base_idx=self.lds_kv_base_idx,
                    v_lds_read_vec4_type=self.v_lds_read_vec4_type,
                    scope_name=v_scope,
                    scope_names=traits.LDS_SCOPE_NAMES,
                )
                a, b = read(imm_lo), read(imm_lo + pair)
                packs[k_substep][dc] = (
                    Vec(a).shuffle(Vec(b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
                )
        return packs

    def load_v(self, buf_id, urv_base=None, stage=0):
        with self._scoped_to_stage():
            if const_expr(self.traits.KV_VECTORIZED):
                return super().load_v(buf_id, urv_base=urv_base)
            base = self.v_lds_read_base_per_lane if urv_base is None else urv_base
            return self.read_v_packs(buf_id, base)


class ParityKvGmemToLdsLoader(_ParityKvStaging, dualwave.DualwaveKvGmemToLdsLoader):
    """K/V staging with independent per-tensor sequence strides.

    The production loader has one `stride_kv_n` for both tensors, which the
    BHSD ABI can contradict -- K and V are separate allocations and a caller
    may hand us one as BHSD and the other as BSHD. `load_k` and `load_v` are
    already separate methods that read `self.stride_kv_n_v` on the way down, so
    selecting the right stride is an attribute swap before delegating, not a
    second copy of the DMA body.

    Safe because tracing is eager: the attribute is read while `super()` runs,
    and no branch is open across the swap.
    """

    def load_k(self, tile_start, buf_id, page_id=None, stage=0):
        self.stride_kv_n_v = self.stride_k_seq_v
        self.dma_stage = stage
        return super().load_k(tile_start, buf_id, page_id=page_id)

    def load_v(self, tile_start, buf_id, page_id=None, stage=0):
        self.stride_kv_n_v = self.stride_v_seq_v
        self.dma_stage = stage
        return super().load_v(tile_start, buf_id, page_id=page_id)

    # `load_*_tile` resolve a tile index to a token offset and delegate; the
    # stage has to ride along or it is lost at that hop.
    def load_k_tile(self, tile_idx, buf_id, page_id=None, stage=0):
        self.load_k(self.tile_start(tile_idx), buf_id, page_id=page_id, stage=stage)

    def load_v_tile(self, tile_idx, buf_id, page_id=None, stage=0):
        self.load_v(self.tile_start(tile_idx), buf_id, page_id=page_id, stage=stage)

    def _async_load_kv_linear(
        self, dma_m0, buf_id, src_div, src_base, soffset, num_dma
    ):
        """Issue this wave's KV DMAs, addressed through `kv_src_elem`.

        The production version calls `_linear_kv_src_elem`, which interleaves
        tokens across `NUM_WAVES` and offsets D by the flat index -- both true
        only at one issue per wave. This is the same loop with the address
        redirected, not a second copy of the DMA sequence.
        """
        for d in range_constexpr(num_dma):
            self._issue_kv_dma(
                src_div, dma_m0[buf_id][d], self.kv_src_elem(src_base, d), soffset
            )


class ParitySoftmaxHelper(dualwave.DualwaveSoftmaxHelper):
    """Softmax whose running max can never be `-inf`.

    The production `reduce_max` seeds the reduction with `-inf` and the kernel
    floors the result to `-3.0e38` only under `CAUSAL`. That is the pattern
    `sdpa-feature-gap.md` flags:

        m_i = tl.full([BLOCK_M], -3.40282e+38)   # do NOT use -inf

    A row whose scores are all `-inf` -- every key masked -- gives
    `m_i = -inf`, and then `exp2(-inf - -inf)` is `NaN` rather than 0. On
    gfx1201 this is *preventative* today and only becomes reachable with bias,
    since causal tile 0 always contains `kv = 0 <= q_row`. Here it is cheaper
    to be unconditional: seeding the reduction at the floor costs nothing (it
    replaces one constant with another) and removes the case for every masking
    mode at once, including the windows and bias still to come.

    Seeding rather than flooring afterwards is the part that matters. A floor
    applied to the *result* still lets an all-`-inf` tile reach the subtract;
    seeding means no lane ever holds `-inf` as a max in the first place.
    """

    def reduce_max(self, v_s):
        return dualwave._score_pair_max(v_s, self.c_neg_floor, self.fm_fast)

    # -- P5: bias ------------------------------------------------------------

    def _add_bias_inplace(self, v_s, tile_idx, narrow_tail=False):
        """`S += bias * log2(e)`, in place, for one KV tile.

        **After the scale and before the mask**, and both halves of that matter:

        - after, because `m_i` and the exponent live in the base-2 scaled
          domain -- Q is pre-scaled by `sm_scale * log2e` on this kernel -- so
          a bias in natural units has to cross into it. This is AOTriton's
          `qk += bias * 1.44269504089`.
        - before, because a column past `seqlen_k` must stay `-inf` rather than
          becoming `-inf + bias`. Those columns are not keys the caller hid;
          they do not exist, and neither do their bias entries.

        No runtime "does this tile need it" guard. There is nothing to skip: a
        bias build reads a bias for every live tile by definition.

        `narrow_tail` reads the run one column at a time instead of as one
        `dwordx2`/`dwordx4`. gfx950 range-checks a multi-dword buffer op **per
        dword** and drops any dword that is not wholly inside `num_records`, so
        a live column paired into the same dword as an out-of-range one is
        dropped with it. `_bias_slab_num_records_bytes` ends the slab exactly on
        the last valid element, so for odd `seqlen_k` the final dword covers
        columns `(seqlen_k-1, seqlen_k)` and straddles: the last row's last bias
        entry reads back 0, for every `(batch, head)`. Widening the bound would
        make the kernel read memory the caller never handed it, so the load
        narrows instead.

        Only the tiles that can *contain* column `seqlen_k-1` need it. A
        straddle requires `seqlen_k` odd, hence `seqlen_k % BLOCK_N != 0`, hence
        that tile is partial and takes the KV tail mask -- so
        `seq_pad_mask_if_needed` passes `narrow_tail`, and `bias_to_lists` does
        not. An interior tile is wholly in bounds, so its final dword covers
        `(c, c+1)` with `c+1 <= seqlen_k-1` and ends at or before the bound; it
        keeps the wide load.
        """
        traits = self.traits
        ctx = self.ctx_ref
        s_lo, s_hi = v_s
        lane_n_off = 8 if traits.KV_VECTORIZED else 4
        # The row is this lane's, and the descriptor already holds (batch,
        # head, row origin), so what is left is `row * pitch + column`.
        row_base = ctx.q_row * fx.Index(ctx.stride_b_seq_q)
        col_base = fx.Index(tile_idx * traits.BLOCK_N) + self.lane_div_32 * fx.Index(
            lane_n_off
        )
        log2e = fx.Float32(1.4426950408889634)
        # **Not `fm_fast` here.** `fm_fast` is MLIR's `fast`, which carries
        # `ninf` and `nnan` -- a licence to assume no infinities reach the
        # operation. A bias entry of `-inf` is how a caller spells "never
        # attend here", so bias is the first thing in this kernel that puts a
        # real infinity into *arithmetic* rather than into a select: the causal
        # mask writes `-inf` through a `cndmask` on raw bits and the KV tail
        # mask through `select`, neither of which fastmath touches.
        #
        # It happens to produce the right answer with `fast` today. That is not
        # a reason to keep it -- plan1 records `ninf` silently deleting a KV
        # tail mask on gfx1201, which is the same licence being taken up later
        # by a different pass. Two operations per element, one of them by a
        # constant, so the flag costs nothing measurable to get right.
        fm_bias = arith.FastMathFlags.contract | arith.FastMathFlags.reassoc
        for half, values in ((0, s_lo), (1, s_hi)):
            for elem0, col_off, width in _score_column_runs(traits.KV_VECTORIZED):
                run_base = row_base + col_base + fx.Index(col_off + half * 32)
                if const_expr(narrow_tail):
                    for j in range_constexpr(width):
                        one = buffer_ops.buffer_load(
                            ctx.bias_rsrc,
                            as_mlir_value(fx.Int32(run_base + fx.Index(j))),
                            vec_width=1,
                            dtype=ctx.elem_dtype,
                        )
                        # `vec_width=1` returns a **scalar** of the element
                        # type, not a one-lane vector, so it is wrapped
                        # directly rather than through `Vec(...)[0]`.
                        b = fx.Float32(ctx.elem_dtype(one).to(fx.Float32))
                        values[elem0 + j] = _fadd(
                            values[elem0 + j], _fmul(b, log2e, fm_bias), fm_bias
                        )
                    continue
                span = buffer_ops.buffer_load(
                    ctx.bias_rsrc,
                    as_mlir_value(fx.Int32(run_base)),
                    vec_width=width,
                    dtype=ctx.elem_dtype,
                )
                for j in range_constexpr(width):
                    b = fx.Float32(
                        Vec(span, (width,), ctx.elem_dtype)[j].to(fx.Float32)
                    )
                    values[elem0 + j] = _fadd(
                        values[elem0 + j], _fmul(b, log2e, fm_bias), fm_bias
                    )

    def bias_to_lists(self, v_s, tile_idx):
        """`v_s_vec_to_lists`, with the bias folded in on the way through.

        The interior tiles of the dual-wave loop only unpack the scores -- they
        need no KV tail mask, being wholly in bounds -- so this is where their
        bias goes. The tiles that *do* mask get it from the
        `seq_pad_mask_if_needed` override, which keeps every tile covered
        exactly once.
        """
        lists = self.v_s_vec_to_lists(v_s)
        if const_expr(self.traits.BIAS_TYPE):
            self._add_bias_inplace(lists, tile_idx)
        return lists

    def seq_pad_mask_if_needed(self, v_s, tile_idx):
        """The KV tail mask, with the bias applied first.

        Order is the point, and it is the same one gfx1201 records: a column
        past `seqlen_k` must come out `-inf`, not `-inf + bias`. Those columns
        are not keys the caller hid -- they do not exist, and neither do their
        bias entries.

        Overriding here rather than adding a call at each site is what makes
        the *wide* body work unchanged: it masks every tile it visits, so this
        single override is its whole bias path.
        """
        if const_expr(self.traits.BIAS_TYPE):
            lists = self.v_s_vec_to_lists(v_s)
            # `narrow_tail`: these are the tiles that can hold column
            # `seqlen_k-1`, and a wide read there loses it. See
            # `_add_bias_inplace`.
            self._add_bias_inplace(lists, tile_idx, narrow_tail=True)
            v_s = dualwave._score_lists_to_vecs(lists)
        return super().seq_pad_mask_if_needed(v_s, tile_idx)

    def rescale_o(self, v_o, m_row, l_row, m_tile_max, v_p):
        """The non-lazy rescale, anchored so one accumulator does not abort LLVM.

        `_anchor_v_o` here is the parity one, which spells out the
        `D_CHUNKS == 1` case; `rescale_o` in the shared helper reaches the
        production version, which always asks an inline asm with `D_CHUNKS`
        outputs for a struct. At one output LLVM does not diagnose that -- it
        aborts with *"inline asm with one output cannot return struct"* and an
        `UNREACHABLE`, killing the process rather than raising.

        Found by the P7 sweep, which is exactly what a sweep is for: head_dim
        32 is the only width with `D_CHUNKS == 1`, `lazy_rescale=True` is the
        default and takes the other path, so this needed the one combination
        nothing had built. The body is the shared one with that single
        substitution; there is no other difference.
        """
        if const_expr(self.traits.D_CHUNKS != 1):
            return super().rescale_o(v_o, m_row, l_row, m_tile_max, v_p)
        m_new = _fmax(m_row, m_tile_max, self.fm_fast)
        corr = rocdl.exp2(T.f32, as_mlir_value(_fsub(m_row, m_new, self.fm_fast)))
        self.scale_o(v_o, corr)
        v_o = _anchor_v_o(self.traits, v_o)
        v_p = dualwave._scale_v_p(
            self.traits, v_p, corr, elem_dtype=self.elem_dtype, fm_fast=self.fm_fast
        )
        l_row = _fmul(l_row, corr, self.fm_fast)
        return v_o, m_new, l_row, v_p

    def safe_l_inv(self, l_row):
        """`1/l`, with the dropout survivor scale folded in.

        `1/(1-p)` is a per-row constant, so it belongs here and not on the
        3200-odd scores it would otherwise multiply: one extra multiply per
        output row against one per element. `dropout_args` computes the value
        host side, and both kernel bodies reach their normalisation through
        this method, so this is the whole fold.
        """
        inv = super().safe_l_inv(l_row)
        if const_expr(self.traits.ENABLE_DROPOUT):
            inv = _fmul(inv, fx.Float32(self.ctx_ref.dropout_scale_arg), self.fm_fast)
        return inv

    def cast_p(self, v_p, tile_idx=None):
        """`cast_p`, with the dropout mask applied first.

        **After `l_row`, before the O accumulation**, which is why this hangs
        off `cast_p` and not off `exp2`. The softmax denominator must be the
        *undropped* sum, or the result stops being an expectation of the
        undropped attention and the logsumexp the backward pass reads is wrong.
        Moving this one call earlier produces plausible output that is wrong by
        a per-row factor, and no shape check notices.

        The survivors are **not** scaled here. `1/(1-p)` is a per-row constant,
        so it folds into the reciprocal of `l` at the store -- one multiply per
        output row instead of one per score. `dropout_args` computes it host
        side; gfx1201 folds it the same way.

        The column runs are the bias ones. A lane's 16 scores are 16 columns of
        one row in a few contiguous spans, and the spans start at multiples of
        `randoms_per_offset`, so each is a whole number of Philox calls with no
        partial draw.

        Every pack leaves through `mfma_operand_wait_state`. These packs are the
        PV GEMM's operand, built by the same `v_cvt_pk_bf16_f32` and handed to
        the same MFMA shape that made `BwdDqSoftmaxHelper.cast_p` return a 39%
        relative-L2 wrong answer; scanning the built binaries found 44 forward
        kernels where the scheduler left fewer than two vector-pipe
        instructions in the gap, `ENABLE_DROPOUT=True` in most of them. Unlike
        dQ this has not been caught producing a wrong answer -- but the gap is
        the same gap, the hardware does not interlock it, and
        `GCNHazardRecognizer` does not model it on gfx950.
        """
        if const_expr(self.traits.ENABLE_DROPOUT):
            traits = self.traits
            ctx = self.ctx_ref
            # `v_p` is already a pair of 16-element lists here -- `exp2` hands
            # those back, where `v_s` arrives as a vector pair. No unpacking.
            lo, hi = list(v_p[0]), list(v_p[1])
            rng = ctx.philox_rng
            lane_n_off = 8 if traits.KV_VECTORIZED else 4
            col_base = fx.Int64(tile_idx * traits.BLOCK_N) + fx.Int64(
                self.lane_div_32 * fx.Index(lane_n_off)
            )
            zero = fx.Float32(0.0)
            for half, values in ((0, lo), (1, hi)):
                for elem0, col_off, width in _score_column_runs(traits.KV_VECTORIZED):
                    first = rng.grid_offset(
                        ctx.philox_plane_base,
                        ctx.philox_row_stride,
                        fx.Int64(ctx.q_row),
                        col_base + fx.Int64(col_off + half * 32),
                    )
                    keep = rng.keep_span(ctx.philox_seed, first, width, ctx.idropout_p)
                    for j in range_constexpr(width):
                        values[elem0 + j] = keep[j].select(
                            fx.Float32(values[elem0 + j]), zero
                        )
            v_p = (lo, hi)
        p_lo_packs, p_hi_packs = super().cast_p(v_p)
        return (
            [mfma_operand_wait_state(p) for p in p_lo_packs],
            [mfma_operand_wait_state(p) for p in p_hi_packs],
        )

    # -- P3: generalized sliding window --------------------------------------
    #
    # The right bound needs no code here at all: it rides on `delta_i32`, which
    # `ParityKernelContext.init_runtime_indices` re-points at the resolved
    # `window_right`. Only the left bound is new.

    def _causal_mask_inplace(self, v_s, tile_idx, q_row_i32=None):
        """The causal right bound, then the window's left one.

        `super()` masks `col <= row + delta`. A window adds `col >= row - left`,
        and the two are independent comparisons against the same per-element
        column, so this composes rather than replacing anything.

        Plain `select`s rather than the paired inline asm the right bound uses.
        The asm packs two compares and two cndmasks per pair and exists because
        the causal mask is on the innermost path of every build; this one runs
        only in window builds, and `seq_pad_mask_inplace` next door already
        establishes the `select` form as the idiom here. Reach for the asm if a
        measurement asks for it, not before.
        """
        super()._causal_mask_inplace(v_s, tile_idx, q_row_i32=q_row_i32)
        if const_expr(not self.traits.WINDOW):
            return
        if q_row_i32 is None:
            q_row_i32 = self.ctx_ref.q_row_i32
        traits = self.traits
        s_lo, s_hi = v_s
        kv_start_i32 = fx.Int32(tile_idx * traits.BLOCK_N)
        # Same lane->column mapping the right bound uses; `s_hi` is the same
        # rows 32 columns further on, which is why its `rel` is 32 smaller.
        lane_n_off = 8 if traits.KV_VECTORIZED else 4
        lane_off_i32 = fx.Int32(self.lane_div_32) * fx.Int32(lane_n_off)
        rel_lo = fx.Int32(
            q_row_i32 - self.ctx_ref.window_left_i32 - kv_start_i32 - lane_off_i32
        )
        rel_hi = fx.Int32(rel_lo - fx.Int32(32))
        for p in range_constexpr(
            len(dualwave._causal_pair_thresholds(traits.KV_VECTORIZED))
        ):
            thr_x, thr_y = dualwave._causal_pair_thresholds(traits.KV_VECTORIZED)[p]
            ix, iy = 2 * p, 2 * p + 1
            # Keep where the column has not fallen behind the left edge.
            s_lo[ix] = (rel_lo <= fx.Int32(thr_x)).select(s_lo[ix], self.c_neg_inf)
            s_lo[iy] = (rel_lo <= fx.Int32(thr_y)).select(s_lo[iy], self.c_neg_inf)
            s_hi[ix] = (rel_hi <= fx.Int32(thr_x)).select(s_hi[ix], self.c_neg_inf)
            s_hi[iy] = (rel_hi <= fx.Int32(thr_y)).select(s_hi[iy], self.c_neg_inf)

    def causal_mask_prologue_if_needed(
        self, v_s, tile_idx=None, kv_end_pos=None, **kwargs
    ):
        """Two-sided version of "does this tile need masking at all?".

        The inherited test is one-sided -- `q_start + right < kv_end`, i.e. does
        the *first* row's right bound fall inside this tile. That is sufficient
        for causal, where the only partial tiles are the ones straddling the
        diagonal, and wrong for a window, whose left bound clips a second set
        of tiles somewhere else entirely.

        So the test gains a second term rather than being dropped. Masking
        unconditionally is also correct -- a mask on a fully live tile is a
        no-op -- but measured 197 us against causal's 126 at an unbounded left
        bound, because it forces the mask onto every interior tile in the walk.

        Each term is the worst case over the tile:

        - a column can overrun the right bound only if the *lowest* row's
          bound, `q_start + right`, lands inside the tile;
        - a column can fall behind the left bound only if the *highest* row's
          edge, `q_start + BLOCK_M - 1 - left`, is still above `kv_start`.
        """
        if const_expr(not self.traits.WINDOW):
            return super().causal_mask_prologue_if_needed(
                v_s, tile_idx=tile_idx, kv_end_pos=kv_end_pos, **kwargs
            )
        traits = self.traits
        if tile_idx is None:
            tile_idx = fx.Index(0)
        kv_end_tile = kwargs.get("kv_end_tile")
        if kv_end_pos is None:
            kv_end_pos = self.tile_start(
                tile_idx + fx.Index(1) if kv_end_tile is None else kv_end_tile
            )
        kv_start_pos = self.tile_start(tile_idx)
        q_start_pos_i32 = kwargs.get("q_start_pos_i32") or self.ctx_ref.q_start_pos_i32
        q_row_i32 = kwargs.get("q_row_i32") or self.ctx_ref.q_row_i32
        window_left_i32 = self.ctx_ref.window_left_i32
        delta_i32 = self.delta_i32
        mask_inplace = self._causal_mask_inplace
        to_lists = self.v_s_vec_to_lists

        @flyc.jit
        def _window_mask_if_needed(
            v_s, tile_idx, kv_end_pos, kv_start_pos, q_start_pos_i32, q_row_i32
        ):
            s_lo, s_hi = v_s
            clipped_right = q_start_pos_i32 + delta_i32 < fx.Int32(kv_end_pos)
            clipped_left = (
                fx.Int32(kv_start_pos)
                < q_start_pos_i32 + fx.Int32(traits.BLOCK_M - 1) - window_left_i32
            )
            if clipped_right | clipped_left:
                lo_list, hi_list = to_lists(v_s)
                mask_inplace((lo_list, hi_list), tile_idx, q_row_i32=q_row_i32)
                s_lo, s_hi = dualwave._score_lists_to_vecs((lo_list, hi_list))
            return s_lo, s_hi

        return _window_mask_if_needed(
            v_s, tile_idx, kv_end_pos, kv_start_pos, q_start_pos_i32, q_row_i32
        )

    def rescale_o_serial(self, v_o, m_row, l_row, m_tile_max):
        """`rescale_o` without the `v_p` term, for the staged (unpipelined) loop.

        The dualwave schedule rescales O *and* the previous tile's P, because
        its softmax is split across clusters and a P from the last iteration is
        still in flight. The staged loop has no such P: it finishes each tile
        before starting the next, so at rescale time the only live state is O,
        the running max and the running sum. Passing a dummy `v_p` to
        `rescale_o` would work and would also emit a real multiply over it.

        Otherwise identical to `rescale_o`, term for term.
        """
        m_new = _fmax(m_row, m_tile_max, self.fm_fast)
        corr = rocdl.exp2(T.f32, as_mlir_value(_fsub(m_row, m_new, self.fm_fast)))
        self.scale_o(v_o, corr)
        v_o = _anchor_v_o(self.traits, v_o)
        l_row = _fmul(l_row, corr, self.fm_fast)
        return v_o, m_new, l_row


class ParityStoreHelper(dualwave.DualwaveStoreHelper):
    """O stores addressed by O's own strides.

    `_final_o_base` is the one place the production code spells O's address,
    and it spells it with *Q's* token stride plus `q_head_idx * HEAD_DIM` --
    correct only where O and Q share a layout. Overriding this single method is
    the whole change; the 128-bit store path above it is untouched.
    """

    def _final_o_base(self, q_row):
        return q_row * self.stride_o_seq_v + self.lane_div_32 * 8

    def zero_o_block_if_needed(self, causal_end_raw_i32=None):
        """The inherited O zeroing, plus the LSE those same rows need.

        A Q block whose every row attends nothing never enters the kernel body
        -- `active` is `causal_end_raw > 0` -- so this method is the *only*
        writer for it. At the pinned FlyDSL (`FLYDSL_KERNELS_COMMIT`) it
        writes O and stops, which leaves LSE holding whatever the caller
        allocated. That is a wrong answer for any caller who will run a
        backward, and an unreadable one for a test harness: AOTriton's fills L
        with NaN, so the rows come back NaN and read as a store that faulted.

        `l = 0` is the whole of the addition, because `_store_lse_row_unguarded`
        already turns a zero `l` into the `+inf` those rows are contractually
        owed -- see there. Passing it rather than the value keeps one expression
        of "this row attended nothing" in the kernel instead of two.

        Not folded into the inherited `scf.if`: `super()` owns that region and
        is imported, never edited. Two regions on the same wave-uniform scalar
        compare is what an override can spell, and the second is entered only by
        blocks that took the early exit, which do no other work at all.

        **Deletable on the next flydsl bump.** Upstream has since grown the same
        store inside its own `zero_o_block_if_needed`, passing exactly this
        `m = 0, l = 0` pair (plus a `sink_log2` this build has no knob for), so
        when the pin moves past it this override becomes a duplicate store and
        should go. The `+inf` select it depends on must stay either way: upstream
        spells the value `m * ln2 + log(l)`, which is `-inf` at best.
        """
        super().zero_o_block_if_needed(causal_end_raw_i32)
        if const_expr(not self.traits.RETURN_LSE):
            return
        if causal_end_raw_i32 is None:
            causal_end_raw_i32 = self.causal_end_raw_i32
        # Rebuilt here rather than captured, so the address arithmetic stays
        # inside the branch for the blocks that skip it. This is `self.q_row`'s
        # own definition, and the inherited O store above uses it too: one row
        # mapping for both tensors and for both the early exit and the body.
        q_start = self.q_start
        wave_q_offset = self.wave_q_offset
        lane_mod_32 = self.lane_mod_32
        zero = self.c_zero_f
        store = self._store_lse_row

        @flyc.jit
        def _store_masked_lse_block():
            if causal_end_raw_i32 <= fx.Int32(0):
                store(zero, zero, q_start + wave_q_offset + lane_mod_32)

        _store_masked_lse_block()

    def _store_lse_row(self, m_row, l_row, q_row):
        """`_store_lse_row_unguarded`, skipped entirely when `LSE` is null.

        LSE is an **optional** output. `attn_fwd_params::L` is declared "Can be
        `T2::get_null_tensor()`" (include/aotriton/flash.h), and an inference
        caller that will never run a backward passes exactly that. The Triton
        kernel spells the contract as `L_not_null` -- *"Allows null L for
        training=False"*, modules/flash/kernel/fwd_kernel.py -- and gfx1201's
        FlyDSL kernel spells it as `_l_valid`
        (flash_attn_func_gfx1201_aiw.py). The gfx950 path had neither: a null
        `L` reached `_make_ws_rsrc` as a descriptor based at address 0 and the
        store faulted.

        So the optionality is a **runtime** property, not a build-time one.
        `RETURN_LSE` is a compile-time knob that deletes the store outright,
        which is the wrong instrument for it twice over: one AOT binary has to
        serve both kinds of caller, and a build with the store deleted is not
        "LSE optional", it is "LSE never written". `RETURN_LSE` therefore stays
        pinned on for AOT (modules/flash/aot/flyc_attn_fwd.py) and the null
        case is decided here, per launch.

        The condition is wave-uniform, so this is a scalar branch. Everything
        the store needs -- the log, the scale, the addressing -- stays inside
        it, for the reason gfx1201's kernel records at its own guard: hoisting
        that arithmetic out cost 8% at head_dim 256 even with the store still
        predicated, because the values then stay live across the epilogue for
        every wave, including the ones with nothing to store.
        """
        store = self._store_lse_row_unguarded
        lse_not_null = fx.Int64(fx.ptrtoint(fx.get_iter(self.LSE))) != fx.Int64(0)

        @flyc.jit
        def _store_lse_if_l_not_null():
            if lse_not_null:
                store(m_row, l_row, q_row)

        _store_lse_if_l_not_null()

    def _store_lse_row_unguarded(self, m_row, l_row, q_row):
        """LSE addressed through `VarlenBits`, which decides three things here.

        LSE is always **compact** -- it is the one tensor whose strides are not
        a free variable but a function of the bits, the head count and the
        token count, which is why no `lse_stride` is passed (plan section 4.2).
        Compact is not the same as fixed, though, and the production formula
        `q_head_idx * seq_len_v + q_row` pins all three of the things that
        vary:

        - **the token pitch.** `seq_len_v` is `max_seqlen_q`, which is right
          for a batched layout that pads every row-group to the longest
          sequence. A stacked Q side runs to the *batch total* instead, which
          is what `lse_token_pitch` decoded into `lse_tokens_i32`.
        - **the row origin.** A packed sequence starts at `q_row_off`, not 0.
        - **the layout.** Bits 17:16 choose `_HT` -- `(H, T)`, T contiguous,
          AOTriton's -- or `_TH`, which is Transformer Engine's. The production
          formula is `_HT` written out, so `_TH` was silently ignored.

        `fmha.lse_row_addressing` is gfx1201's and returns `(base, pitch)` so
        the per-row part stays a multiply-add. It is called with **batch 0**
        because the descriptor below already folds the batch in, and the two
        must not both count it; that works for either layout because a batch's
        rows are contiguous in both, `H * tokens` of them.

        The production row expression `q_head_idx * seq_len_v + q_row` is not a
        second case to keep: it is what this form *is* at `varlen_bits == 0`,
        `row_off == 0` and `tokens == seq_len_v`. One term in the descriptor is
        **not** the production one, though: the head count.

        Upstream's `_store_lse_row` sizes the per-batch slice with
        `traits.NUM_HEADS_Q`, a compile-time trait, because upstream compiles a
        kernel per shape. AOT cannot: one binary serves every head count, so
        `modules/flash/aot/flyc_attn_fwd.py` pins `num_heads=1` and the real
        count arrives as the `num_head_q` kernarg. With the trait, the
        descriptor covers `1 * tokens` rows, the batch stride advances by one
        head instead of `H`, and the hardware bound silently drops every head
        but the first -- `L` comes back written for `h == 0` and untouched
        (NaN) for the rest. So the count here is `self.num_head_q`, which is
        also what gfx1201 feeds `lse_row_addressing`. Two scalar ops; the
        stores are unchanged.
        """
        # The runtime head count, NOT `traits.NUM_HEADS_Q` -- see above.
        num_heads_q = fx.Index(self.num_head_q)
        tokens = fx.Index(self.lse_tokens_i32)
        per_batch = num_heads_q * tokens
        per_batch_bytes = per_batch * fx.Index(4)
        rsrc = dualwave._make_ws_rsrc(
            fx.Int64(fx.ptrtoint(fx.get_iter(self.LSE))),
            self.batch_idx * per_batch_bytes,
            per_batch_bytes,
        )
        lse_val = _fadd(
            _fmul(m_row, self.c_ln2_f, self.fm_fast),
            fmath.log(as_mlir_value(l_row), fastmath=self.fm_fast),
            self.fm_fast,
        )
        # A Q block that attends nothing gets **+inf**, and the formula above
        # does not produce it. Two separate things are wrong without this
        # select.
        #
        # *The sign.* The backward pass subtracts LSE from `qk`, so `+inf` is
        # what makes `exp(qk - lse)` zero for exactly the rows that must
        # contribute nothing. `-inf` would make it `+inf` instead. This is the
        # convention `fwd_kernel.py` writes at its fully-masked early exit,
        # naming that reason, and gfx1201's FlyDSL kernel writes at its LSE
        # store; gfx950 was the only forward with neither.
        #
        # *The value.* `l == 0` makes the `log` above `-inf`, but `fm_fast` is
        # `fastmath<fast>`, which carries `ninf`, so InstSimplify folds an
        # infinite operand of the `fadd` to **poison** -- and a store of poison
        # is one InstCombine may delete. Measured: the fully-masked rows came
        # back holding the harness's own NaN fill, so the defect reads as "LSE
        # was never written" rather than as a sign error.
        #
        # `l` is bit-exact 0 there and at least 1 on any live row -- the running
        # max contributes `exp2(0)` -- so test the bit pattern; integer compares
        # lower predictably under fast math, where a float compare against a
        # value the flags promise cannot exist does not.
        #
        # **This is wider than Triton's rule, deliberately, and it matches
        # gfx1201.** Triton writes `+inf` only for a whole masked *block*; a
        # masked row inside a live block falls through its ordinary epilogue,
        # which starts `l_i` at 1.0 and `m_i` at `-FLT_MAX` and so writes about
        # -2.36e38 -- Triton fixes up `Out` for those rows and leaves their LSE
        # alone. The predicate below cannot separate the two cases and does not
        # try, exactly as gfx1201's does not. `l` starting at 0 here rather than
        # at 1.0 is what makes the predicate meaningful at all.
        #
        # Both LSE writers come through here. The other is
        # `zero_o_block_if_needed`, the whole-block early exit taken when
        # `causal_end_raw <= 0`, which passes a literal `l = 0.0`: the compare
        # folds and this costs that path nothing.
        lse_val = (fmha.bitcast_i32(fx.Float32(l_row)) != fx.Int32(0)).select(
            fx.Float32(lse_val), fx.Float32(float("inf"))
        )
        base, pitch = fmha.lse_row_addressing(
            self.varlen_bits_arg,
            fx.Index(0),
            self.q_head_idx,
            num_heads_q,
            tokens,
            self.q_row_off,
        )
        lse_local = base + q_row * pitch
        # One writer per row: low half-wave, in-bounds row; everything else is
        # redirected to the sentinel the buffer bound drops.
        off_row = (q_row < self.seqlen_q_v).select(lse_local, per_batch)
        off = fx.Index((self.lane < fx.Index(32)).select(off_row, per_batch))
        dualwave._ws_store_f32(lse_val, off, rsrc)

    def _final_o_global(self, o_base, dc, g):
        """The store's element offset, redirected out of the buffer if past `hdim_vo`.

        A 128-bit store is all-or-nothing, so a chunk straddling `hdim_vo`
        cannot be partially written -- and does not need to be. The D pitch is
        contractually a multiple of 8 elements, so a chunk that *starts* inside
        `hdim_vo` ends inside the allocation, and writing its tail into the
        caller's own padding is exactly what that contract permits. Only
        chunks starting at or past `hdim_vo` must be suppressed.

        Suppressed by *address*, not by a branch: pushing the offset past the
        descriptor's `num_records` makes the hardware drop the store, which
        costs one select on a lane-varying value instead of an `scf.if` around
        a 128-bit store. `store_lse` uses the same device.
        """
        off = super()._final_o_global(o_base, dc, g)
        if const_expr(not self.PADDED_HEAD):
            return off
        col_base = fx.Index(
            dc * self.traits.D_CHUNK + 2 * g * 8
        ) + self.lane_div_32 * fx.Index(8)
        in_range = MaskedAxis(fx.Index(self.hdim_vo)).valid(col_base)
        return fx.Index(in_range.select(fx.Index(off), self.o_oob_off))
