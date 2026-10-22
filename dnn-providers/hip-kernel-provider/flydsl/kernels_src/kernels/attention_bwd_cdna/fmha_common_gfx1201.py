# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""Hardware detail for the gfx1201 (RDNA4) attention kernels.

Two boundaries, and the arch suffix is the important one.

**Not `kernels/common/`**: what is here either encodes an attention ABI
decision or has no meaning outside one. The general-purpose siblings live next
door -- `kernels/common/mem_ops.py` for pointer and global load/store,
`kernels/common/utils.py` for the scalar integer helpers.

**Not shared with gfx950 or gfx1250 either.** `flash_attn_utils.py` is the
gfx950 equivalent and this module deliberately does not import from it or
extend it. The hardware differs enough that the *algorithms* differ: gfx950
schedules around MFMA/VALU co-execution, which RDNA4 has no equivalent of, and
its dualwave traits/context machinery exists to serve that. Merging the two
would mean one abstraction serving two schedules that agree on almost nothing.
When gfx1250 FMHA arrives it gets its own `fmha_common_gfx1250.py` for the
same reason.


Why the branching helpers live here
-----------------------------------

Nothing in this file is AST-rewritten. The rewrite from Python's `if` to
`scf.if` is lexical per `@flyc.kernel` function, so a module-level helper gets
a branch only by building the `scf.IfOp` itself -- which is exactly what
`stage`, `publish`, `publish_transposed`, `write_v8` and `cond_load` do.

That is a feature, not a tax, and it is why those five are free functions
while `Aperture`'s non-branching operations are methods. Two problems
disappear at the module boundary:

**Objects can be held in ordinary variables.** In kernel code an object live
across a dynamic `if` becomes region state, which has to be MLIR-backed:

    fastmath = FastMath(FP_MODE)
    -> TypeError: state variable 'fastmath' is FastMath, not an MLIR Value

Here, `with ir.InsertionPoint(...)` is not a function boundary. An aperture
referenced inside the region is a plain Python local, and the IR values it
holds were materialised before the `IfOp`, so they dominate the region and
need no block arguments. (`flydsl.compiler.protocol`'s carry protocol --
`__get_ir_types__` and friends -- is the other way to solve this, for objects
that must cross a *rewritten* `if`. `MaskedAxis` and `Aperture` implemented it
until the branching helpers moved here; removing it broke no test, so it is
gone. Reach for it again only if kernel code has to method-call one of these
objects under a dynamic `if`.)

**The guarded and unguarded arms stop being duplicated.** A `const_expr` test
in kernel code cannot wrap a single call -- `if const_expr(needs_guard): if
pred: body()` / `else: body()` has to spell `body()` twice -- so every staging
site carried two copies. `_over_batches` writes it once.

One trap remains for kernel-side code. `ast_rewriter._collect_assigned_vars`
counts `name.method(...)` under a dynamic `if` as a use of carried state and
assigns `name` back after the region. If `name` came from an *enclosing*
scope, that makes it a local of the inner function, unbound on any sibling
path that skips the `if` -- a `const_expr` arm, typically:

    UnboundLocalError: cannot access local variable 'qk_cols'

Three ways out, best first: put the code in a module-level function; call a
free function so the base name is a module rather than the object
(`fmha.write_v8(ap, ...)` rather than `ap.write_v8(...)`); or bind a local
alias before the branch, which is what `_check_local_var` recommends in its
own warning.
"""

from dataclasses import dataclass

from kernels.common import buffer_ops, kernels_common

# Import rewrite: `wmma_ops`, and every symbol this file takes from
# `kernels.common.utils` -- ssel, smin, smax, sdiv_rd_pow2 -- are branch-local
# and absent from the released tag the build clones, so both names alias the
# polyfill wholesale and no call site below changes. `buffer_ops` and
# `kernels_common` are NOT rewritten: their symbols are in the released tree.
from kernels.attention_bwd_cdna import flyc_polyfill as wmma_ops
from kernels.attention_bwd_cdna import flyc_polyfill as common_utils

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import scf as _scf
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

__all__ = [
    "wave_lanes",
    "llvm_value",
    "split_ptr",
    "wmma_acc",
    "pointer_to_llvm_ptr",
    "load_geom",
    "acc_elem_column",
    "lse_row_addressing",
    "lds_load_v8",
    "lds_store_vx",
    "global_load_tr_v8",
    "bitcast_i32",
    "pack_bf16_pair",
    "bf16_trunc_pack_v8",
    "FastMath",
    "MaskedAxis",
    "Aperture",
    "stage",
    "publish",
    "write_v8",
    "read_batches",
    "read_batches_unmasked",
    "TransposedTiling",
    "read_transposed",
    "publish_transposed",
    "lds_f32_ptr",
    "lds_f32_store",
    "lds_f32_load",
    "reduce_s_across_shards",
    "cond_load",
    "seqinfo_addr",
    "decode_addressing",
    "lse_token_pitch",
    "WINDOW_TOPLEFT",
    "WINDOW_BOTRIGHT",
    "resolve_window",
    "CausalRegions",
    "decompose_causal_regions",
    "make_addr_pair",
    "philox_offset_base",
    "philox_seed_value",
    "philox_report",
]


def wave_lanes(warp_size):
    """`(tid, wave_id, lane, lane16, klane)` -- this thread's place in the block.

    The same five lines opened all four kernels. `lane16` and `klane` are the
    WMMA operand split rather than anything general: a 16x16x16 fragment puts
    the row/column in `lane % 16` and the K offset in `lane // 16`, which is
    why they belong together with the wave decomposition and not with the
    caller's tiling.

    **Emission order is load-bearing** and matches what the four kernels had
    line for line: `//` before `%`, then the two lane splits. Reordering the
    first two is behaviour-preserving but moves the SSA definitions, which at
    head_dim 384 -- where the kernel is pinned at 256 VGPRs -- was enough to
    reschedule the whole body. Same instruction mix and same count, different
    order; not a regression, but not something to change by accident either.
    """
    tid = fx.Index(gpu.thread_idx.x)
    wave_id = tid // warp_size
    lane = tid % warp_size
    return tid, wave_id, lane, lane % 16, lane // 16


def wmma_acc(a_v8, b_v8, c_v8):
    """One 16x16x16 WMMA into an f32 accumulator.

    Dispatch is on the operand element type, inside `wmma_ops`, rather than on
    a `dtype_str` here -- see that module for why. Was a closure over
    `v8f32_type` in all four kernels; the type is a context lookup, so making
    it here costs nothing and removes the closure.
    """
    return wmma_ops.wmma_f32_16x16x16(a_v8, b_v8, c_v8, Vec.make_type(8, fx.Float32))


def llvm_value(value):
    """Unwrap a FlyDSL scalar/vector wrapper for a raw LLVM pointer op."""
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


def split_ptr(ptr, base64, off, elem_type):
    """`ptr + base64 + off`, both 64-bit, added as two element steps.

    The split is for the addressing mode, not for arithmetic: `base64` is
    uniform across the wave and `off` is divergent, and keeping them separate
    is what lets `SelectGlobalSAddr` hold the base in SGPRs instead of forcing
    a 64-bit VGPR address pair.

    **`off` must not be narrowed to i32.** It carries `row_in_tile * s_seq`,
    and a view keeps the strides of the tensor it was taken from rather than
    of the shape the kernel sees: eight heads sliced out of a 1 GiB
    `(1, 64, 16384, 512)` f16 tensor give `s_seq = 8388608`, whose 256th row
    is exactly 2**31. The truncation wrapped and the kernel read another
    allocation. `s_seq` is already `fx.Index`, so the product was always
    computed in 64 bits; only the cast threw the high half away.
    """
    p = buffer_ops.get_element_ptr(ptr, fx.Int64(base64), elem_type=elem_type)
    return buffer_ops.get_element_ptr(p, fx.Int64(off), elem_type=elem_type)


def pointer_to_llvm_ptr(ptr) -> ir.Value:
    """A `!fly.ptr` kernel argument as an LLVM pointer, for raw load/store.

    Thin wrapper over `fx.to_llvm_ptr`, which resolves the pointer's semantic
    address space through the active backend rather than hardcoding a number.
    Kept as a named function because the reason the arguments are pointers at
    all is not local to any one call site:

    **Why the attention kernels take pointers and not tensors.** Every other
    kernel in the tree declares `fx.Tensor`, which would reach `get_llvm_ptr`
    directly and make this function unnecessary. It was measured on the
    gfx1201 SDPA kernel and rejected: each `fx.Tensor` argument adds a 40-byte
    `by_value` memref descriptor *interleaved immediately after its pointer*,
    growing the kernarg segment from 268 to 428 bytes and shifting the offset
    of every argument after the first. AOTriton dispatches that hsaco
    directly rather than through the Python wrapper, so the kernarg layout is
    an ABI, and rearranging it to delete one four-line helper is a bad trade.

    The switch would not even remove the helper: `L`, `Bias` and the four
    `seqinfo_*` arguments are optional, signalled by a null pointer the host
    builds with `flyc.from_c_void_p(..., 0)`, and there is no tensor to pass
    for a thing that is not there. They would stay pointers and keep needing
    this.
    """
    return fx.to_llvm_ptr(ptr)


def lse_row_addressing(varlen_bits, batch, head, num_head_q, tokens, row_off):
    """`(base, pitch)` for a row-wise f32 side input -- logsumexp, and delta.

    The element for absolute query row `r` is at `base + r * pitch`. Factored
    that way because `base` is loop-invariant while `r` is not: the four
    callers that had open-coded this each recomputed the whole offset per row.

    Two layouts, one select, decoded from VarlenBits bits 17:16:

        _HT   (H, T), T contiguous -- AOTriton's and this kernel's default.
              base = (batch * H + head) * tokens + row_off,   pitch = 1
        _TH   (T, H), H contiguous -- Transformer Engine's.
              base = (batch * tokens + row_off) * H + head,   pitch = H

    `tokens` is `lse_token_pitch`'s answer, not `max_seqlen_q`: a stacked
    layout runs to the batch total rather than padding each row-group.

    **Delta is required to share LSE's layout.** It is produced beside it by
    the same caller, and giving it its own decode would double the work for no
    expressiveness -- so both side inputs use this one function.

    Four callers of one fact: the forward *writes* LSE through it, and all
    three backward kernels *read* LSE and delta through it. It was two
    spellings and a third that had already factored out the base, which is the
    form kept here.
    """
    tok = fx.Index(tokens)
    nhq = fx.Index(num_head_q)
    base_ht = (batch * nhq + head) * tok + row_off
    base_th = (batch * tok + row_off) * nhq + head
    is_th = ((fx.Int32(varlen_bits) >> fx.Int32(16)) & fx.Int32(3)) != fx.Int32(0)
    base = fx.Index(is_th.select(fx.Index(base_th), fx.Index(base_ht)))
    pitch = fx.Index(is_th.select(nhq, fx.Index(1)))
    return base, pitch


def lse_row_step_per_head(varlen_bits, tokens):
    """How far `lse_row_addressing`'s `base` moves when `head` rises by one.

    `base` is affine in `head` under both layouts -- `_HT` has it inside the
    `* tokens` product, `_TH` outside it -- so a caller walking a GQA group can
    add this per head instead of re-running the decode. That is not a peephole:
    it is what stops `varlen_bits`, `batch` and `row_off` from being live across
    the walk, and on gfx950 that liveness is what drives the backward kernel's
    scalar spilling. See `BwdDkDvKernelContext._init_q_head_invariants`.

    Exactly the mirror of the `pitch` select above -- `_HT` pitches by 1 and
    steps by `tokens`, `_TH` pitches by `H` and steps by 1 -- so the two are
    written the same way, from the same bits, and cannot drift apart.

    `num_head_q` is deliberately not a parameter: it does not appear in either
    step, and taking it would suggest the caller must keep it live.
    """
    tok = fx.Index(tokens)
    is_th = ((fx.Int32(varlen_bits) >> fx.Int32(16)) & fx.Int32(3)) != fx.Int32(0)
    return fx.Index(is_th.select(fx.Index(1), tok))


def acc_elem_column(i):
    """Which column of the WMMA tile flattened accumulator element `i` holds.

    A plain Python int in, a plain Python int out -- no IR, no operands. The
    lane's own contribution (`klane * 8`) and the tile origin are the caller's
    to add, because they are traced values and this is not.

        column(i) = (i // 16) * 32 + ((i // 8) % 2) * 16 + i % 8   [+ klane * 8]

    The three terms are the three nestings of the GEMM unroll read backwards.
    `i // 16` picks the column sub-tile, 32 columns wide. `(i // 8) % 2` picks
    which half of that sub-tile -- the unroll walks (sub-tile, half) pairs, so
    the halves alternate every 8 elements. `i % 8` is the position inside a
    lane's eight, and within a 16-row WMMA block a lane holds rows
    `klane * 8 + si`, which is where the caller's `klane * 8` comes from.

    The consequence worth naming: **within each group of eight only `i % 8`
    varies**, so those eight are eight *contiguous* columns and a masked read
    of them is one v8 load rather than a gather. Three call sites in the
    forward -- the KV tail mask, bias, and dropout -- and one in `bwd_dq` had
    each open-coded the expression with a paragraph of this comment attached.
    """
    return (i // 16) * 32 + ((i // 8) % 2) * 16 + i % 8


def load_geom(width, vec_width, block_size, rows):
    """Cooperative-load geometry: `(threads_per_row, rows_per_batch, batches, needs_guard)`.

    `block_size` threads cooperate to fill `rows` rows of `width` elements,
    `vec_width` elements per thread per access.

    **`ceil` and not `floor` on the batch count.** Flooring silently drops rows
    whenever rows-per-batch neither reaches `rows` nor divides it: at
    BLOCK_DMODEL 160/192/224 the forward gets 25/21/18, so `rows // that == 1`
    and only 25/21/18 of the 32 KV rows reached LDS. The rest was stale, which
    surfaced as NaN.

    **Two independent reasons for the row guard, and both must be checked.**
    This is the whole reason the function is shared rather than copied:

    1. `nb * rpb != rows` -- the ceil() batches overshoot the tile, so the last
       batch leaves some lanes with no row.
    2. `tpr * rpb != block_size` -- when threads-per-row does not divide the
       workgroup, the leftover threads get `tid // tpr == rpb`, one row *past*
       their batch's share. Those are harmless duplicates of the next batch's
       row except on the *final* batch, where the row is past `rows` and the
       store lands in whatever tile follows.

    Reason 2 is the one that was missing, and it is not hypothetical. Three
    kernels grew a private copy of this function testing only reason 1:

    - the forward is safe, but by coincidence rather than construction. Eight
      of its thirty configs (the non-power-of-two widths 48/80/96/160/192/224/
      384) do have leftover threads, and in every one reason 1 happens to fire
      anyway; in every config where the guard is absent, `tpr` divides the
      workgroup exactly.
    - `bwd_dq` hit it and fixed it locally.
    - `bwd_dkdv` hit it and did **not**: at head_dim 224 its shipped default
      (block_m 16, 4 waves -> tpr 28, rpb 4, nb 4) gives `nb * rpb == rows`
      exactly, so no guard was emitted while 16 of 128 threads wrote one row
      past the tile. Measured 0.90 relative error on dK, silent because the
      store is in bounds for the LDS allocation as a whole, and invisible
      because that kernel's test ladder stopped at 128.

    That is three independent derivations of one fact and two of them wrong,
    which is the argument for one copy.
    """
    tpr = max(1, width // vec_width)
    rpb = max(1, block_size // tpr)
    nb = (rows + rpb - 1) // rpb
    return tpr, rpb, nb, (nb * rpb != rows) or (tpr * rpb != block_size)


# --------------------------------------------------------------------------
# LDS access, split for honest alignment
# --------------------------------------------------------------------------
#
# The two functions below exist only to avoid over-promising alignment, which
# is a property of RDNA's LDS instruction selection rather than of attention,
# and is why they live in the arch module.


def lds_load_v8(lds_ptr, lds_idx, v4_type):
    """Load 8 half-precision elements from LDS as two honest 8-byte accesses.

    **Not one v8 load.** K/V rows are `K_STRIDE * 2` bytes apart, so these
    addresses are only guaranteed 8-byte aligned. `fly.ptr_load` emits no
    alignment attribute, so LLVM falls back to the vector type's ABI
    alignment -- 16 B for v8f16, 32 B for v16f16 -- and that over-promise makes
    the backend select `ds_load_b128` on addresses that are not 16-byte
    aligned. Measured 2.2x slower (92 -> 39 TFLOPS), and undefined behaviour
    besides. Two v4f16 accesses carry a truthful `align 8` and fold back into
    `ds_load2_b64`.
    """
    lo = fx.ptr_load(lds_ptr + fx.Int32(lds_idx), result_type=v4_type)
    hi = fx.ptr_load(lds_ptr + fx.Int32(lds_idx + 4), result_type=v4_type)
    return Vec(lo).shuffle(Vec(hi), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()


def lds_store_vx(lds_ptr, vec, lds_idx, vec_width):
    """Store `vec_width` half elements to LDS in 8-byte pieces. See `lds_load_v8`."""
    v = Vec(vec)
    for _i in range_constexpr(vec_width // 4):
        part = v.shuffle(v, [_i * 4, _i * 4 + 1, _i * 4 + 2, _i * 4 + 3])
        fx.ptr_store(part, lds_ptr + fx.Int32(lds_idx + _i * 4))


def global_load_tr_v8(base_i64, base64, off32, v8_type):
    """One `global_load_tr_b128`: an 8x8 16-bit transpose per lane-group.

    Lane g_i supplies an address; the 8 contiguous elements there become
    column i of the group's output, so lane g_j receives [M_0[j] .. M_7[j]].
    Verified empirically on gfx1201. The instruction is RDNA-only, which is
    what puts this in the arch module.

    Address is split the same way the rest of the kernel splits one: the
    (batch, head, tile) origin and the intra-tile part, added in 64 bits.
    Feeding LLVM `uniform_i64 + divergent` is what lets `SelectGlobalSAddr`
    keep the base in SGPRs instead of forcing a 64-bit VGPR address pair.

    The divergent half is deliberately *not* narrowed to i32 on the way. It
    carries `row_in_tile * stride_seq`, and a view's sequence stride is bounded
    by the tensor it was taken from rather than by the shape here -- eight
    heads sliced out of a 1 GiB (1, 64, 16384, 512) f16 tensor give
    `stride_seq = 8388608`, whose 256th row is exactly 2**31. Narrowing it
    wrapped and read another allocation.
    """
    base_bytes = fx.Int64(fx.Index(base64) * 2)
    off_bytes = fx.Int64(fx.Index(off32) * 2)
    addr = fx.as_ir_value(fx.Int64(base_i64) + base_bytes + off_bytes)
    p = _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), addr).result
    return rocdl.global_load_tr_b128(v8_type, p)


def bitcast_i32(value):
    return fx.Float32(value).bitcast(fx.Int32)


def pack_bf16_pair(lo, hi, shift, mask):
    lo_i32 = bitcast_i32(lo)
    hi_i32 = bitcast_i32(hi)
    return (hi_i32 & mask) | arith.shrui(lo_i32, shift)


def bf16_trunc_pack_v8(f32_vals, elem_dtype):
    """Pack 8 f32 values into v8bf16 via bitwise truncation (upper 16 bits).

    On P precision, before anyone tries to raise it here:

    **There is no way to keep P in f32 through GEMM2 on gfx1201.** RDNA4
    WMMA has no F32xF32 form (ISA manual Table 41); A/B operands are
    f16/bf16/iu8/iu4/fp8 only. LLVM does define
    `v_wmma_f32_16x16x4_f32`, but it is real-ized under
    `VOP3P_Real_WMMA_gfx1250` -- gfx1250 only, not gfx12/gfx1201. The
    AOTriton idiom `acc += tl.dot(p, v.to(p.type.element_ty))` works on
    CDNA because that has `v_mfma_f32_16x16x4f32`; it has no gfx1201
    equivalent. Doing PV in f32 here would mean dropping to VALU FMA and
    giving up the matrix cores for GEMM2.

    Note also that V is *not* downcast: it reaches GEMM2 at the input
    tensor's native 16-bit width, so only P loses precision.

    Truncation is round-toward-zero. Measured against an fp64 reference
    (`accuracy_probe.py`, B=1 H=4 N=1024 d=128): no output bias (O sums
    P*V and V is zero-mean, so the one-sided P error cancels), but the
    RMS error is 1.6x torch SDPA's at bf16 (4.43e-3 vs 2.78e-3). f16 is
    already at exact parity. Switching to round-to-nearest-even --
    `x += 0x7FFF + ((x >> 16) & 1)` before the shift -- closes that gap
    exactly (2.79e-3) but costs 2-3% at distance 1 and 2.7-5.4% at
    ROW_SUBTILES=2, so it is deliberately not done. Truncation by
    decision, not oversight.
    """
    _c16 = fx.Int32(16)
    _cmask = fx.Int32(0xFFFF0000)
    pairs = []
    for j in range_constexpr(4):
        pairs.append(pack_bf16_pair(f32_vals[j * 2], f32_vals[j * 2 + 1], _c16, _cmask))
    return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()


class FastMath:
    """The softmax's float ops, with one `arith.FastMathFlags` set bound once.

    A class rather than four free functions taking the flags, because the flag
    set is the thing worth making visible. It is a *knob* -- `fp_mode` selects
    between "fast", "noninf" and "safe" -- and the choice is load-bearing:
    "fast" includes `ninf`, which once silently deleted the KV tail mask,
    because `exp2(-inf - m)` folds to something the flag says cannot happen.

    Takes the mode rather than a flag set, so the mapping from knob to flags
    lives here and not at each kernel that uses it. Construct it on the **host**
    side of the builder, not in the kernel body: `fp_mode` is `const_expr`, so
    this is a plain Python object the traced code captures, and assigning it
    inside the body makes the AST rewriter treat it as `scf` loop/if state
    ("state variable 'fastmath' is FastMath, not an MLIR Value").
    """

    __slots__ = ("flags",)

    def __init__(self, fp_mode: str):
        _F = arith.FastMathFlags
        if fp_mode == "fast":
            self.flags = _F.fast
        elif fp_mode == "noninf":
            self.flags = _F.reassoc | _F.nnan | _F.nsz | _F.arcp | _F.contract | _F.afn
        elif fp_mode == "safe":  # as "noninf", also dropping nnan
            self.flags = _F.reassoc | _F.nsz | _F.arcp | _F.contract | _F.afn
        else:
            raise ValueError(f"unknown fp_mode {fp_mode!r}; expected fast/noninf/safe")

    # The four operators go through `arith.fastmath`, a stable context manager,
    # rather than `arith.addf` / `subf` / `mulf` / `divf`, none of which is in
    # `arith.__all__`. It ends in the same call: `ArithValue._binary_op` reads
    # `current_fastmath()` and passes it as the `fastmath=` this class used to
    # pass by hand.
    #
    # Operands may be `ArithValue`, `Float32` or `Vector`, and all three reach
    # that path -- `Vector` because it subclasses `ArithValue`. Which is why
    # these take an operator rather than a scalar builder: `mul` is called on
    # v8f32 accumulators as well as on scalars.

    def div(self, a, b):
        with arith.fastmath(self.flags):
            return a / b

    def add(self, a, b):
        with arith.fastmath(self.flags):
            return a + b

    def sub(self, a, b):
        with arith.fastmath(self.flags):
            return a - b

    def mul(self, a, b):
        with arith.fastmath(self.flags):
            return a * b

    def max(self, a, b):
        # `arith.maxnumf` is stable and takes the flags directly. No context
        # around it: the raw builders do not consult one -- only the operators
        # do, via `_binary_op`.
        return arith.maxnumf(a, b, fastmath=self.flags)


class MaskedAxis:
    """One axis of a tile whose index can run past the real extent.

    Out-of-range indices always need two things, and keeping them together is
    the point: an address that is *safe to issue*, and a way to *discard what
    it returns*. Issuing the access unconditionally and throwing the value away
    beats branching around it, but only once the address has been redirected
    somewhere legal -- element 0 of the axis, which always exists.

    **One class covers rows and columns**, because the apparent difference
    between them is not about the axes. An access reads `width` contiguous
    elements *along one axis*; for that axis the extent boundary can fall
    inside the access, so validity is per element. For every other axis the
    index is a single scalar and the whole access stands or falls together.
    In this kernel the vector runs along the column axis, which is why columns
    look "per element" and rows "whole" -- but that is a property of the
    access, not of the axis, and `valid(idx)` is exactly `mask(idx, 1)`.

    `active=False` compiles the masking away, for an axis whose extent is known
    to be a multiple of the access width.
    """

    __slots__ = ("extent", "active", "elem_dtype", "bitmask")

    def __init__(self, extent, active=True, elem_dtype=None, bitmask=False):
        self.extent = extent
        self.active = active
        # See `discard`. Off by default: it trades in-loop instructions for
        # live registers, and only the caller knows whether that is affordable.
        self.bitmask = bitmask
        # Only `discard` needs this, so the row axis leaves it unset. It is a
        # property of the tensor and every access along an axis shares it.
        #
        # `width` is deliberately *not* bound here. It belongs to the access,
        # not the axis, and the two accesses along the QK column axis agree
        # only by coincidence: the cooperative loads are `VEC_WIDTH` wide while
        # the Q preload is 8 wide because `load_global_v8f16` is tied to the
        # WMMA operand shape. Both are 8 today from unrelated definitions, and
        # binding one would make the Q preload silently follow `VEC_WIDTH`.
        self.elem_dtype = elem_dtype

    def _bound(self):
        """The extent as a traced value.

        Resolved lazily so the object can be built on the host when the extent
        is a `const_expr` int.
        """
        return fx.Index(self.extent) if isinstance(self.extent, int) else self.extent

    def valid(self, idx):
        """`fx.Boolean`: is this index inside the extent?

        A signed compare, on every axis. Signedness is a property of the
        *type* -- `_make_binop` reads each operand's class-level `signed` --
        so `fx.Int64` is what makes `<` emit `slt` here, and it is also why
        this cannot simply compare the `fx.Index` values it is handed:
        `fx.Index` is unsigned and would give `ult`. The answer agrees, since
        every index and extent is non-negative, but the two are not the same
        code and mixing them per axis was a difference with no reason behind
        it.

        Spelled with the operator rather than `arith.cmpi(CmpIPredicate.slt,
        ...)` because `cmpi` is stable while `CmpIPredicate` is not, and
        because the `fx.Boolean` this returns has a stable `select`. The
        `index_cast` that `fx.Int64` inserts folds away -- index is already
        64-bit here -- which was measured, not assumed.
        """
        return fx.Int64(idx) < fx.Int64(self._bound())

    def mask(self, idx, width):
        """i1 vector, element j set iff `idx + j` is inside the extent.

        Built from a loop-invariant index at every current caller, so it hoists
        out of the KV loop and costs one vector select per access inside it.
        """
        return Vec.from_elements(
            [self.valid(idx + fx.Index(j)) for j in range_constexpr(width)],
            fx.Boolean,
        )

    def safe(self, idx, addressed=None):
        """`addressed` if `idx` is inside the extent, else 0.

        `addressed` defaults to `idx`, which is the column case. Rows need the
        two to differ: the bound is on the *absolute* row, `start_q + ...`,
        while the address is built from the row's offset *within the tile*, so
        the tested and the redirected quantity are not the same value.
        """
        if addressed is None:
            addressed = idx
        if not self.active:
            return addressed
        return fx.Index(self.valid(idx).select(addressed, fx.Index(0)))

    def discard(self, vec, idx, width):
        """Zero the elements of `vec` whose index is past the extent.

        **AND with a precomputed bit mask, not a per-element select**, whenever
        the elements are 16-bit and pair up evenly into dwords.

        The obvious spelling -- `mask(idx, width).select(vec, zeros)` -- asks
        for a *per-element* choice over 16-bit lanes that live packed two to a
        VGPR, so the backend has to take them apart and put them back. Measured
        on a head_dim 32 build in the 64 tile, against an unpadded 64:

            v_cndmask_b32   +419
            v_lshrrev_b32   +208     <- pure repacking
            v_perm_b32      +192     <- pure repacking
            total          +1022 instructions (3585 against 2563)

        Two thirds of that is the register file being shuffled rather than any
        masking work. A dword whose two halves are `0xFFFF` or `0` expresses the
        same choice with one `v_and_b32` and no repacking, and the mask itself
        is loop-invariant -- it depends on the extent and a loop-invariant
        column base -- so it hoists out and costs nothing inside the loop.

        Exact for any extent, including one that splits a pair: the two halves
        of a dword carry independent masks.

        **Opt-in, because it is not free and not always a win.** The masks are
        loop-invariant, so they hoist -- and then stay *live* for the whole
        loop, `width/2` registers per masked access. Measured on the gfx950
        parity kernel, that is 16 registers in the 64-wide tile and 32 in the
        128-wide one, and the 128 build already sits at 238 of 256:

            tile    spills   TFLOP/s at the padded rungs
             64        0     +21% (head_dim 16/32/48)
            128       61     -43% (head_dim 80/96/112)

        So the caller decides. `bitmask=False` keeps the per-element select,
        which is also the path the fp8 and odd-width callers were verified on.
        """
        if not self.active:
            return vec
        if (
            not self.bitmask
            or self.elem_dtype is None
            or self.elem_dtype.width != 16
            or width % 2
        ):
            zeros = Vec.filled(width, 0.0, self.elem_dtype)
            return self.mask(idx, width).select(Vec(vec), zeros).ir_value()

        keep_lo = fx.Int32(0x0000FFFF)
        keep_hi = fx.Int32(0xFFFF0000)
        zero_i32 = fx.Int32(0)
        dwords = [
            self.valid(idx + fx.Index(2 * d)).select(keep_lo, zero_i32)
            | self.valid(idx + fx.Index(2 * d + 1)).select(keep_hi, zero_i32)
            for d in range_constexpr(width // 2)
        ]
        raw = Vec(vec).bitcast(fx.Int32)
        return (
            (raw & Vec.from_elements(dwords, fx.Int32))
            .bitcast(self.elem_dtype)
            .ir_value()
        )

    def gate(self, idx, addressed=None):
        """`(valid(idx), safe(idx, addressed))` -- the two halves together.

        An out-of-range index needs both, always, and returning them as a pair
        is what stops a caller taking the address and forgetting the flag.
        """
        return self.valid(idx), self.safe(idx, addressed)


class Aperture:
    """The bounded region of one tensor this kernel may touch, and where it
    lands on chip.

    An aperture is the opening, not the light through it: it says which rows
    and columns exist and where a staged copy goes, never what was read. One
    instance per tensor, built once and handed to the movement helpers, so an
    access cannot be spelled without also naming its bounds.

    Fields are optional because the tensors are not staged the same way, and an
    absent field is a statement rather than a gap:

    `rows=None`      the row bound is inside the address closure instead. K and
                     V are like this: `make_addr_pair` was given `seqlen_k` and
                     redirects an out-of-range row itself, so a `rows` axis
                     here would be the same bound stated twice. It would also
                     not be free -- an IR-backed field crosses every dynamic
                     `if` this object is live across, and this kernel has
                     measured 6% swings from one extra value's live range.
    `lds_base=None`  never staged through LDS. Q and O go straight between VRAM
                     and registers.
    `num_batches=0`  not read cooperatively. Same two.

    The address closures are deliberately *not* fields; see `reader`. Only the
    axes hold IR values at all -- every other field is `const_expr`, which is
    what keeps an aperture free to pass around.
    """

    __slots__ = (
        "cols",
        "rows",
        "lds_base",
        "lds_stride",
        "vec_width",
        "threads_per_row",
        "rows_per_batch",
        "num_batches",
        "needs_guard",
    )

    def __init__(
        self,
        cols,
        rows=None,
        lds_base=None,
        lds_stride=None,
        vec_width=0,
        threads_per_row=0,
        rows_per_batch=0,
        num_batches=0,
        needs_guard=False,
    ):
        self.cols = cols
        self.rows = rows
        self.lds_base = lds_base
        self.lds_stride = lds_stride
        # The cooperative-load geometry, from `_load_geom` on *this* tensor's
        # width. K's and V's are computed from different widths and are not
        # interchangeable; holding each on its owner is what stops one
        # tensor's flag being read for the other.
        self.vec_width = vec_width
        self.threads_per_row = threads_per_row
        self.rows_per_batch = rows_per_batch
        self.num_batches = num_batches
        self.needs_guard = needs_guard

    def lds_index(self, row, col):
        """Element index of (row, col) within the LDS tile."""
        return self.lds_base + row * self.lds_stride + col

    def batch_row(self, base_row, batch):
        """The row `base_row` maps to in cooperative-load batch `batch`."""
        return base_row + batch * self.rows_per_batch

    def to_lds(self, lds_ptr, vec, row, col, width):
        """Publish `width` elements of `vec` at (row, col) of the LDS tile.

        `width` is the access's, not the aperture's: the staged vector is
        `vec_width` wide for a cooperative load but 8 wide for a transposed
        one, and those two are equal today only by coincidence.
        """
        lds_store_vx(lds_ptr, vec, self.lds_index(row, col), width)

    def from_lds(self, lds_ptr, row, col):
        """The 8 contiguous elements at (row, col) -- one WMMA operand."""
        v4 = Vec.make_type(4, self.cols.elem_dtype)
        return lds_load_v8(lds_ptr, self.lds_index(row, col), v4)

    def read_v8(self, fetch, row, col, row_ok):
        """One WMMA operand from VRAM at (row, col), fully masked.

        All three of the maskings this kernel needs, and none of them
        optional: the columns past `hdim` are zeroed per element, a row past
        `seqlen` is replaced wholesale, and `col` is redirected before the
        access so the address stays legal either way.

        `row` must already be the *safe* row and `row_ok` its gate flag -- take
        both from `rows.gate(...)`, once per row rather than once per column,
        because starting that compare early has measured 6% at the widest
        causal build.

        `fetch(row, col)` issues the access, and stays the caller's because
        the 64-bit-base / 32-bit-offset split is per tensor. `reader` builds
        one.
        """
        raw = self.cols.discard(fetch(row, self.cols.safe(col)), col, 8)
        zeros = Vec.filled(8, 0.0, self.cols.elem_dtype).ir_value()
        return row_ok.select(raw, zeros)

    def read_vec(self, fetch, row, col):
        """One cooperative-load vector at (row, col), columns masked.

        No row gate, unlike `read_v8`: for a tensor staged through LDS the
        row bound lives in the address closure, which redirects an
        out-of-range row to a live one rather than discarding it. What lands
        in LDS is then garbage from a real row, and the S mask -- not this --
        is what keeps it out of the answer.
        """
        return self.cols.discard(fetch(row, self.cols.safe(col)), col, self.vec_width)


# --------------------------------------------------------------------------
# Cooperative staging: VRAM -> LDS
# --------------------------------------------------------------------------


def _over_batches(aperture, base_row, block_rows, body):
    """`body(batch, row)` for each cooperative-load batch, under the row guard.

    The guard exists because `_load_geom` covers BLOCK_N rows with `ceil()`
    batches, so the last one can overshoot and leave some lanes with no row.
    Whether it is needed is `aperture.needs_guard`, a `const_expr`, so the
    unguarded configs emit no branch at all.

    An explicit `IfOp`, not Python's `if`: the rewrite from `if` to `scf.if`
    is lexical per `@flyc.kernel` function, so a module-level helper only gets
    a branch by writing one. That is also what removes the guarded/unguarded
    duplication -- the kernel had to spell both arms because the `const_expr`
    test could not wrap a single call.
    """
    for batch in range_constexpr(aperture.num_batches):
        row = aperture.batch_row(base_row, batch)
        if aperture.needs_guard:
            if_op = _scf.IfOp(fx.as_ir_value(row < block_rows))
            with kernels_common._if_then(if_op):
                body(batch, row)
        else:
            body(batch, row)


def stage(aperture, lds_ptr, read, base_row, col, block_rows):
    """VRAM -> LDS for this thread's share of one tile.

    gfx1201 has no direct global->LDS instruction, so this is load-then-store
    through VGPRs; gfx950 and gfx1250 have one. Naming the operation puts the
    seam where that ISA difference falls, rather than leaving each caller to
    open-code a pair.

    Load and store sit inside the *same* guard, which matters: when the last
    batch overshoots BLOCK_N those lanes have no row, and issuing their
    clamped, redundant global loads anyway measured -9.6%. Only a distance-0
    schedule can do this -- at distance 1 the loaded value is loop-carried, so
    it has to exist unconditionally, and the load and the store are guarded
    separately by `read_batches` and `publish`.
    """

    def body(_batch, row):
        aperture.to_lds(
            lds_ptr,
            aperture.read_vec(read, row, col),
            row,
            col,
            aperture.vec_width,
        )

    _over_batches(aperture, base_row, block_rows, body)


def publish(aperture, lds_ptr, vecs, base_row, col, block_rows):
    """Registers -> LDS: the store half, for a tile already in flight."""

    def body(batch, row):
        aperture.to_lds(lds_ptr, vecs[batch], row, col, aperture.vec_width)

    _over_batches(aperture, base_row, block_rows, body)


def reader(addr, load):
    """Bind a tensor's address split to its load: `start -> (row, col) -> value`.

    Curried on `start` because the tile origin moves every KV iteration while
    the address closure and the load instruction do not. What comes out is the
    `read` argument every movement helper here takes, so the helpers never
    need to know how a tensor is addressed or which load it uses.

    Not an `Aperture` field: an aperture is placement, which is the same on
    every iteration, while this closes over the tile origin, which is not.
    """

    def at(start):
        def read(row, col):
            base64, off32 = addr(start, row, col)
            return load(base64, off32)

        return read

    return at


def read_batches(aperture, read, base_row, col):
    """VRAM -> registers: this thread's share of one tile, columns masked.

    No row guard, unlike `stage`. A distance-1 schedule carries the loaded
    vectors through the loop, so every batch has to produce a value whether
    or not its row exists; `publish` guards the store instead.
    """
    return [
        aperture.read_vec(read, aperture.batch_row(base_row, batch), col)
        for batch in range_constexpr(aperture.num_batches)
    ]


def read_batches_unmasked(aperture, read, base_row, col):
    """`read_batches` without the column zeroing. Addresses are still safe.

    For a tensor whose out-of-range columns reach nothing. That is V and only
    V here: its garbage lands in O columns past `hdim_vo`, which the epilogue
    store drops, so zeroing it would be work with no reader -- and the
    configs that would pay are exactly the small padded ones (7-in-16,
    8-in-16, 40-in-48) where V is row-major.

    Named rather than a `mask_cols=False` argument, so the unusual case has to
    be spelled out at the call site instead of hiding in a keyword.
    """
    return [
        read(aperture.batch_row(base_row, batch), aperture.cols.safe(col))
        for batch in range_constexpr(aperture.num_batches)
    ]


# --------------------------------------------------------------------------
# The transposed V layout
# --------------------------------------------------------------------------


class TransposedTiling:
    """How V^T's 16(d) x 16(kv) blocks are spread over the waves.

    `global_load_tr_b128` transposes an 8x8 block of 16-bit elements across
    each group of 8 lanes, so one wave-wide load produces a 16(d) x 16(kv)
    block already in WMMA-operand order. This object owns the resulting
    tiling: how many blocks there are, how they map onto `l` and `wave_id`,
    and where this lane sits inside one.

    It is separate from `Aperture` for the reason §9.5 gives about K and V --
    per-tensor geometry belongs with its tensor -- taken one level further:
    this is per-*layout* geometry, live only when `V_LDS_LAYOUT` is
    "transposed", and folding it into the aperture would put fields on V that
    half the configs never read.

    The four lane offsets are two different mappings and must not be
    interchanged. The load pair says which address this lane supplies so the
    hardware transpose lands the right block; the store pair says where the
    lane's transposed result belongs in LDS.
    """

    __slots__ = (
        "d_blocks",
        "tiles",
        "loads",
        "needs_guard",
        "num_waves",
        "d_step",
        "kv_step",
        "wave_id",
        "load_d_off",
        "load_kv_off",
        "store_d_off",
        "store_kv_off",
    )

    def __init__(
        self,
        d_blocks,
        tiles,
        loads,
        needs_guard,
        num_waves,
        d_step,
        kv_step,
        wave_id,
        load_d_off,
        load_kv_off,
        store_d_off,
        store_kv_off,
    ):
        self.d_blocks = d_blocks
        self.tiles = tiles
        self.loads = loads
        self.needs_guard = needs_guard
        self.num_waves = num_waves
        self.d_step = d_step
        self.kv_step = kv_step
        self.wave_id = wave_id
        self.load_d_off = load_d_off
        self.load_kv_off = load_kv_off
        self.store_d_off = store_d_off
        self.store_kv_off = store_kv_off

    def tile(self, step):
        """Which V^T block this wave handles on step `l`."""
        return self.wave_id + fx.Index(step * self.num_waves)

    def origin(self, step):
        """`(d, kv)` of that block's top-left corner."""
        t = self.tile(step)
        return (t % self.d_blocks) * self.d_step, (t // self.d_blocks) * self.kv_step

    def overshoots(self, step):
        """const_expr: can step `l` leave some waves with no block?

        The tiling need not divide evenly across the waves; requiring it once
        forced BLOCK_DMODEL 160 down to 4 waves and cost it 89.1 -> 70.0
        TFLOPS. Only the final step or two can overshoot, so the earlier ones
        emit no branch.
        """
        return self.needs_guard and (step + 1) * self.num_waves > self.tiles


def read_transposed(aperture, tiling, read, col_extra=0):
    """VRAM -> registers through the hardware transpose, one vector per block.

    `col_extra` is the window offset (`chunk * VO_CHUNK_COLS` plus
    `D_OFFSET`); LDS indices stay window-relative, so it is added here and
    nowhere else.

    Columns are made safe but not zeroed, for the `read_batches_unmasked`
    reason -- and here there is a second one: after the transpose a lane's 8
    elements run along kv at a single d, so a column mask could only be
    whole-vector, not per element.
    """
    out = []
    for step in range_constexpr(tiling.loads):
        d_base, kv_base = tiling.origin(step)
        col = d_base + tiling.load_d_off
        if col_extra:
            col = fx.Index(col_extra) + col
        out.append(read(kv_base + tiling.load_kv_off, aperture.cols.safe(col)))
    return out


def publish_transposed(aperture, tiling, lds_ptr, vecs):
    """Registers -> LDS for the transposed layout.

    A tail wave with no block has still run its global load -- the address
    closure clamped the row, so it read in bounds -- and is simply not
    published.

    8 wide because that is what the transposed load returns, the WMMA operand
    shape, not because it is the cooperative-load width (§9.3).
    """

    def body(step):
        d_base, kv_base = tiling.origin(step)
        aperture.to_lds(
            lds_ptr,
            vecs[step],
            d_base + tiling.store_d_off,
            kv_base + tiling.store_kv_off,
            8,
        )

    for step in range_constexpr(tiling.loads):
        if tiling.overshoots(step):
            if_op = _scf.IfOp(
                fx.as_ir_value(tiling.tile(step) < fx.Index(tiling.tiles))
            )
            with kernels_common._if_then(if_op):
                body(step)
        else:
            body(step)


def write_v8(aperture, write, row, col, val):
    """Store one 8-wide chunk at (row, col), skipped if `col` is past `hdim`.

    Whole-chunk skipping rather than per-element masking, and that is exact
    rather than a compromise: the output's D pitch is a 16-byte multiple, so
    a chunk that straddles `hdim` lies entirely inside the tensor's own
    allocation. Columns in [hdim, ceil8(hdim)) receive computed-but-unused
    values, mirroring the pad region of the inputs, which the caller slices
    off. `col` is therefore *not* redirected the way a load's is -- inside the
    guard it is already in range.

    A free function and not an `Aperture` method, like `stage` and `publish`
    and unlike `read_v8`: anything that emits a branch has to build the
    `scf.IfOp` itself, which only module-level code may do. Writing
    `fmha.write_v8(ap, ...)` also keeps `ap` out of
    `_collect_assigned_vars` -- `ap.write_v8(...)` inside a dynamic `if`
    would be collected as region state and make the enclosing `scf.if` yield
    the aperture back.
    """
    if not aperture.cols.active:
        write(row, col, val)
        return
    if_op = _scf.IfOp(fx.as_ir_value(aperture.cols.valid(col)))
    with kernels_common._if_then(if_op):
        write(row, col, val)


# --------------------------------------------------------------------------
# f32 scratch aliased over the 16-bit LDS tile
# --------------------------------------------------------------------------
#
# The cross-shard S reduction needs f32 scratch, and the KV tile it borrows
# space from is `elem_dtype` (16-bit). There is no retyped view of a shared
# pointer, so the address is built by hand: `ptrtoint` on a shared pointer
# yields the 32-bit LDS offset, and the f32 element index is scaled into it.
#
# Plain functions taking both halves of the base, rather than an object: these
# are read inside the reduction's dynamic branches, and an object could not be
# live there (see "How to hand a helper object to kernel code" above). Both
# arguments are safe to hold in kernel-body variables -- one is an MLIR value,
# the other a `const_expr` int.


def lds_f32_ptr(lds_byte_base, byte0, index):
    """`!llvm.ptr<3>` at f32 element `index` of the scratch starting at `byte0`."""
    off = fx.Int32(byte0) + fx.Int32(index) * fx.Int32(4)
    addr = arith.addi(lds_byte_base, fx.as_ir_value(off))
    return _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<3>"), addr).result


def lds_f32_store(lds_byte_base, byte0, index, value):
    _llvm.StoreOp(fx.as_ir_value(value), lds_f32_ptr(lds_byte_base, byte0, index))


def lds_f32_load(lds_byte_base, byte0, index):
    return _llvm.LoadOp(
        ir.F32Type.get(), lds_f32_ptr(lds_byte_base, byte0, index)
    ).result


def reduce_s_across_shards(
    s_accs,
    *,
    lds_byte_base,
    byte0,
    wave_id,
    lane,
    shard_id,
    q_tile_in_block,
    num_shards,
    f32_per_wave,
    warp_size,
    fastmath,
):
    """Sum one Q row sub-tile's S accumulators across the QK shards, through LDS.

    Each shard-wave holds a partial sum over its own slice of BLOCK_DMODEL;
    the full S is their sum. Returns the reduced accumulators, same shape in.

    **Explicit partials, not `ds_add_f32`.** The atomic form measured 1055
    WMMA-equivalents against 54 for this, because every lane contends on the
    same address -- see `kernels/microbench/lds_reduce.py`. So each wave writes
    its own partials to a private slot, and then every wave reads the others'
    and adds them locally: two barriers and no contention.

    Called only when `num_shards > 1`, which on the current ladder is
    BLOCK_DMODEL 384 (2 shards) and 512 (4). Every other width reduces nothing
    and never reaches here -- worth knowing when gating a change to it, since
    the usual 128 build does not execute a line of this.
    """
    s_flat = [fx.as_ir_value(Vec(a)[r]) for a in s_accs for r in range_constexpr(8)]

    own = wave_id * fx.Index(f32_per_wave)
    for e in range_constexpr(len(s_flat)):
        lds_f32_store(
            lds_byte_base, byte0, own + fx.Index(e * warp_size) + lane, s_flat[e]
        )
    gpu.barrier()

    base_group = q_tile_in_block * fx.Index(num_shards * f32_per_wave)
    for e in range_constexpr(len(s_flat)):
        acc = s_flat[e]
        for k in range_constexpr(num_shards - 1):
            peer = base_group + (
                (shard_id + fx.Index(k + 1)) % fx.Index(num_shards)
            ) * fx.Index(f32_per_wave)
            acc = fastmath.add(
                acc,
                lds_f32_load(
                    lds_byte_base, byte0, peer + fx.Index(e * warp_size) + lane
                ),
            )
        s_flat[e] = acc
    gpu.barrier()

    return [
        Vec.from_elements(
            [fx.Float32(s_flat[st * 8 + r]) for r in range_constexpr(8)], fx.Float32
        ).ir_value()
        for st in range_constexpr(len(s_accs))
    ]


# --------------------------------------------------------------------------
# Varlen prologue: VarlenBits -> per-sequence addressing
# --------------------------------------------------------------------------


def cond_load(cond, addr, default):
    """Load i32 from `addr` when `cond`, else `default`. The load is skipped.

    A real `scf.if`, not a select, and that is the point: the sequence-info
    pointers are **null** whenever their mode is off, so a select -- which
    evaluates both arms -- would fault. Inside the region the load is never
    issued; verified against a null pointer.

    Built as an explicit `IfOp` rather than Python's `if`, which is what lets
    this live in a module at all: the rewrite from `if` to `scf.if` is lexical
    per `@flyc.kernel` function, but an `IfOp` written out needs no rewriting.

    `addr` is computed by the caller and may be derived from a null pointer --
    address arithmetic touches no memory.
    """
    if_op = _scf.IfOp(fx.as_ir_value(cond), results_=[T.i32], has_else=True)
    with ir.InsertionPoint(if_op.then_block):
        _scf.YieldOp([fx.as_ir_value(fx.ptr_load(addr, fx.Int32))])
    with ir.InsertionPoint(if_op.else_block):
        _scf.YieldOp([fx.as_ir_value(default)])
    return fx.Int32(if_op.results[0])


def philox_offset_base(offset1, offset2):
    """`offset2 + *offset1` -- the Philox counter, split the way torch splits it.

    `at::cuda::PhiloxCudaState` carries the offset two ways. Outside a graph
    capture it is an immediate. Under capture the counter has to advance
    between replays, so it lives in device memory that the graph re-reads,
    while the per-call increment stays baked in as an immediate;
    `at::cuda::philox::unpack` is `*offset_.ptr + offset_intragraph_`. Passing
    one pre-summed scalar instead is what freezes a captured graph onto a
    single dropout mask, because the sum is done once at capture time and the
    replay never sees the counter move.

    AOTriton spells the pair `philox_offset1` (the pointer) and
    `philox_offset2` (the immediate), and this is the same ABI so that the two
    are drop-in for each other.

    A null `offset1` is the uncaptured case. The load is *skipped*, not
    selected away -- `select` evaluates both arms and would fault. Same
    explicit `IfOp` and the same reason as `cond_load`, at i64: written out
    rather than as a Python `if` because the rewrite to `scf.if` is lexical
    per `@flyc.kernel`, and this is module level.
    """
    return fx.Int64(offset2) + _load_u64_or_zero(offset1)


def philox_seed_value(seed_ptr):
    """`*seed_ptr`, or 0 when null. The seed side of the same graph story.

    A captured graph must see the seed move too, so torch keeps it in device
    memory exactly as it keeps the offset counter, and AOTriton takes it as
    `philox_seed_ptr` rather than a value. Splitting the offset but leaving the
    seed an immediate would give a replay a moving counter under a frozen key,
    which is not the same stream torch's own RNG would have produced.

    Null reads as 0 rather than faulting, matching AOTriton's `dropout_rng`.
    """
    return _load_u64_or_zero(seed_ptr)


def philox_report(seed_output, offset_output, seed, offset_base):
    """Write back the `(seed, offset)` this launch actually drew from.

    Only the backward can say why this exists. It has to regenerate the
    forward's stream, and under graph capture the effective offset is
    `*offset1 + offset2` -- a sum formed *on the device*, from a counter the
    host cannot read without synchronising. So the forward records what it
    used and the backward is handed that, instead of both sides trying to
    re-derive it and being wrong in different ways.

    One workgroup stores, not all of them: every workgroup computed the same
    two values, so the rest would be writing the same bytes to the same two
    addresses for no reason. `block_idx` raw rather than the flipped
    `q_tile_idx`, matching AOTriton's `program_id` guard -- which workgroup is
    designated does not matter, only that exactly one is.

    Either output may be null, which is how a caller says it does not want the
    value; both are skipped independently.
    """
    first = (
        (fx.Index(gpu.block_idx.x) == fx.Index(0))
        & (fx.Index(gpu.block_idx.y) == fx.Index(0))
        & (fx.Index(gpu.block_idx.z) == fx.Index(0))
    )
    with kernels_common._if_then(_scf.IfOp(fx.as_ir_value(first))):
        _store_u64_if_nonnull(seed_output, seed)
        _store_u64_if_nonnull(offset_output, offset_base)


def _load_u64_or_zero(ptr):
    """`*ptr` as an i64, or 0 when `ptr` is null.

    The load is *skipped*, not selected away -- `select` evaluates both arms
    and would fault on the null. Same explicit `IfOp` and the same reason as
    `cond_load`, at i64: written out rather than as a Python `if` because the
    rewrite to `scf.if` is lexical per `@flyc.kernel`, and this is module
    level.
    """
    nonnull = fx.Int64(fx.ptrtoint(ptr)) != fx.Int64(0)
    if_op = _scf.IfOp(fx.as_ir_value(nonnull), results_=[T.i64], has_else=True)
    with ir.InsertionPoint(if_op.then_block):
        _scf.YieldOp(
            [
                fx.as_ir_value(
                    fx.ptr_load(fx.recast_iter(_i64_global_ptr_ty(), ptr), fx.Int64)
                )
            ]
        )
    with ir.InsertionPoint(if_op.else_block):
        _scf.YieldOp([fx.as_ir_value(fx.Int64(0))])
    return fx.Int64(if_op.results[0])


def _store_u64_if_nonnull(ptr, value):
    nonnull = fx.Int64(fx.ptrtoint(ptr)) != fx.Int64(0)
    with kernels_common._if_then(_scf.IfOp(fx.as_ir_value(nonnull))):
        fx.ptr_store(fx.Int64(value), fx.recast_iter(_i64_global_ptr_ty(), ptr))


def _i64_global_ptr_ty():
    """A u64 counter in global memory. Alignment 8, for the same reason as i32."""
    return fx.PointerType.get(
        elem_ty=fx.Int64.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=8,
    )


def _i32_global_ptr_ty():
    """An i32 pointer into global memory, alignment 4.

    Spelled out rather than `fx.recast_iter(fx.Int32, ptr)`, which inherits the
    source pointer's alignment: a kernel argument arrives as `u8` with
    alignment 1, and the shorthand then raises "alignment must be a positive
    multiple of element byte size (4), got 1". Same construction as
    `kernels/moe/moe_a8w4_mxscale_gfx1250.py` and
    `kernels/gemm/mxfp4_preshuffle.py`.

    A function, not a module constant: `PointerType.get` needs an MLIR context,
    and at import time there is none.
    """
    return fx.PointerType.get(
        elem_ty=fx.Int32.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=4,
    )


def seqinfo_addr(ptr, index):
    """`&ptr[index]` for an i32 sequence-info array. No memory is touched.

    A typed `!fly.ptr`, not an `!llvm.ptr`: that is what lets `cond_load` read
    it with the stable `fx.ptr_load` instead of a raw `llvm.LoadOp`.
    """
    return fx.recast_iter(_i32_global_ptr_ty(), ptr) + fx.Int64(index)


def decode_addressing(varlen_bits, bits_shift, max_seqlen, s0, s1, z):
    """One side of VarlenBits: where this workgroup's sequence lives.

    Returns `(seqlen, row_off, batch)` -- how long this sequence is, which row
    it starts at, and which batch index to use. Called once for Q and once for
    K. See section 3.1 of `sdpa-varlen-plan.md`; the axes are STACKED (bit 0),
    LENGTH (bits 2:1) and POSITION (bits 4:3).

    The LSE token pitch is *not* here, though it decodes from the same bits: it
    describes the logsumexp output rather than where Q or K live, and only the
    Q side needs it. See `lse_token_pitch`.

    Every load goes through `cond_load`, so the shape is flat -- fetch what
    each mode might need, then select. `s0[z]` serves both length modes and,
    under REUSE, the position too, which is why three loads cover five modes.
    """
    bits = fx.Int32(varlen_bits) >> fx.Int32(bits_shift)
    stacked = (bits & fx.Int32(1)) != fx.Int32(0)
    lenmode = (bits >> fx.Int32(1)) & fx.Int32(3)
    posmode = (bits >> fx.Int32(3)) & fx.Int32(3)

    cumulative = lenmode == fx.Int32(1)
    individual = lenmode == fx.Int32(2)
    reuse = posmode == fx.Int32(1)  # position already read as `cur`
    array = posmode == fx.Int32(2)  # position from its own array
    zero = fx.Int32(0)

    cur = cond_load(lenmode != zero, seqinfo_addr(s0, z), zero)
    nxt = cond_load(cumulative, seqinfo_addr(s0, z + fx.Int32(1)), zero)
    pos = cond_load(array, seqinfo_addr(s1, z), zero)

    seqlen = common_utils.ssel(
        cumulative,
        nxt - cur,
        common_utils.ssel(individual, cur, fx.Int32(max_seqlen)),
    )
    row_off = common_utils.ssel(
        array,
        pos,
        common_utils.ssel(
            reuse,
            cur,
            common_utils.ssel(stacked, z * fx.Int32(max_seqlen), zero),
        ),
    )
    # **An empty sequence gets row 0 of the tensor, not row 0 of itself.**
    #
    # A zero-length sequence has no row to point at. Under a packed layout a
    # *trailing* one puts `row_off` at exactly `total_tokens`, so even row 0 of
    # that sequence is one past the last row of the tensor -- and every clamp
    # downstream is powerless, because `MaskedAxis.safe` redirects to element 0
    # *of the axis* and an empty axis has none. Three kernels read past the end
    # of Q, dO and K/V that way; two of the three were reproduced as HIP
    # memory-access faults whose address was bit-for-bit the end of the tensor.
    #
    # Redirecting the offset itself is what there always is a legal answer for:
    # row 0 of the whole tensor exists whenever any workgroup runs at all, since
    # a zero-row tensor makes the grid empty.
    #
    # Nothing else needs to change, because a workgroup on an empty sequence
    # already does no work: every loop is zero-trip (`decompose_causal_regions`
    # inverts its range when `alive` is false) and every store is guarded. The
    # prologue preloads were the one access that ran regardless, and their
    # results are discarded -- so making the address legal is the whole fix,
    # and skipping the transaction as well would buy nothing.
    #
    # Inert for every non-empty sequence, and for the padded layouts where
    # `row_off` is already zero.
    row_off = common_utils.ssel(seqlen > zero, row_off, zero)
    batch = common_utils.ssel(stacked, zero, z)
    return seqlen, row_off, batch


def lse_token_pitch(varlen_bits, bits_shift, max_seqlen, s0, s1, num_seqlens):
    """Row pitch of the logsumexp output, in tokens. Q side only.

    Batched layouts pad every row-group to `max_seqlen`; stacked ones run to
    the batch total, which lives in slot [N] of whichever array supplies
    positions -- the prefix-sum assumption of plan section 9.4, asserted host
    side.

    Derived from the bits rather than passed because the logsumexp tensor,
    alone among the tensors here, is always compact: its strides are a function
    of the bits, and passing them would be a second source of truth for one
    fact (plan section 4.2).
    """
    bits = fx.Int32(varlen_bits) >> fx.Int32(bits_shift)
    stacked = (bits & fx.Int32(1)) != fx.Int32(0)
    posmode = (bits >> fx.Int32(3)) & fx.Int32(3)
    reuse = posmode == fx.Int32(1)
    array = posmode == fx.Int32(2)
    zero = fx.Int32(0)

    total_s0 = cond_load(stacked & reuse, seqinfo_addr(s0, num_seqlens), zero)
    total_s1 = cond_load(stacked & array, seqinfo_addr(s1, num_seqlens), zero)
    return common_utils.ssel(
        stacked,
        common_utils.ssel(
            reuse,
            total_s0,
            common_utils.ssel(
                array, total_s1, fx.Int32(num_seqlens) * fx.Int32(max_seqlen)
            ),
        ),
        fx.Int32(max_seqlen),
    )


# --------------------------------------------------------------------------
# Sliding-window attention: resolving the window, and cutting the KV range
# --------------------------------------------------------------------------

WINDOW_TOPLEFT = -2147483647  # 0x80000001
WINDOW_BOTRIGHT = -2147483646  # 0x80000002


def resolve_window(window_left, window_right, seqlen_q, seqlen_k):
    """`(window_left, window_right)` with the causal sentinels resolved.

    `Window_left` / `Window_right` may carry `WINDOW_TOPLEFT` or
    `WINDOW_BOTRIGHT` instead of a literal bound, and they are resolved
    against *this sequence's* lengths rather than on the host. That is the
    whole reason the sentinels exist: host resolution works only when there is
    one length to resolve against, and under varlen bottom-right needs
    `seqlen_k[z] - seqlen_q[z]`, which differs per sequence. Matches
    AOTriton's `parse_window`.

    Both sentinels give an unbounded left edge -- no row reaches further back
    than the start of its own sequence -- so they differ only in the right one.

    **Everything derived from a window stays i32.** Window bounds go negative;
    that is what a sentinel and a leading masked region are. `fx.Int32` is
    signed, so `<`/`>` emit `slt`/`sgt`, while `fx.Index` is unsigned and
    64-bit -- widening any of these even once makes the same comparison
    unsigned and a negative bound comes out enormous.
    """
    left = fx.Int32(window_left)
    right = fx.Int32(window_right)
    left_is_sentinel = (left == fx.Int32(WINDOW_TOPLEFT)) | (
        left == fx.Int32(WINDOW_BOTRIGHT)
    )
    left = common_utils.ssel(left_is_sentinel, seqlen_q, left)
    right = common_utils.ssel(right == fx.Int32(WINDOW_TOPLEFT), fx.Int32(0), right)
    right = common_utils.ssel(
        fx.Int32(window_right) == fx.Int32(WINDOW_BOTRIGHT),
        seqlen_k - seqlen_q,
        right,
    )
    return left, right


@dataclass(frozen=True, slots=True)
class CausalRegions:
    """The three contiguous KV block runs a causal/windowed Q block walks.

    Every field is a traced `fx.Int32`, not a Python int -- these are values
    the kernel computes per workgroup. Signed, because `right_col0` goes
    negative when the window admits no key at all; see
    `decompose_causal_regions`.

    A dataclass rather than a `NamedTuple` to match `Philox` next door, and
    because no caller destructures it positionally -- the kernel reads the
    seven fields by name. Being a Python object it is subject to the usual
    rule: do not let one live across a dynamic `if` (see "How to hand a helper
    object to kernel code"). The kernel unpacks it immediately, which is why
    it is safe here.
    """

    n_left: fx.Int32  # masked tiles before the full run
    n_full: fx.Int32  # tiles with no mask at all
    n_right: fx.Int32  # masked tiles after it
    left_col0: fx.Int32  # first KV column of each run
    full_col0: fx.Int32
    right_col0: fx.Int32
    masked_col0: fx.Int32  # first column of the masked run, whichever side


def decompose_causal_regions(
    start_q, q_len, k_len, window_left, window_right, block_m, block_n, alive
):
    """Cut this Q block's visited KV range into `[masked][full][masked]`.

    **Three regions, not two.** A left window kills columns at the *start* of
    the range as well as the end, so masked tiles are a prefix as well as a
    suffix and tile 0 is not automatically live. A negative `window_left` is
    the sharpest case: it pushes the whole band right of the diagonal, so the
    leading masked run can span several tiles rather than clipping one. Do not
    carry the non-causal two-region intuition in here.

    The three are contiguous and non-overlapping *by construction*, because
    they are derived by cutting one visited range rather than intersected as
    three independent intervals. That collapses two of the three special cases
    in `sdpa-gswa-plan.md` section 2.2: a window narrower than a block leaves
    the full region empty, which is detected once and turns the other two into
    a single masked run, and an irregular `seqlen_q` needs no special handling
    because `q_hi` already bounds the rows.

    A column c is live for row i iff `i - window_left <= c <= i + window_right`,
    so over the block the live columns span
    `[start_q - window_left, (q_hi - 1) + window_right]`, and a tile is *fully*
    live iff every one of its columns is live for every row -- worst case the
    largest row on the left and the smallest on the right.

    `alive` is false for a workgroup whose rows all sit past `q_len`, which the
    varlen grid dispatches because its Q extent is sized from `Max_seqlen_q`.
    The kernel is one single-exit trace and cannot return out of those (plan
    section 6.1), so the visited range is *inverted* instead and every region
    count falls to zero. Dropping this makes those workgroups walk real tiles.

    Everything here is i32 and signed, deliberately: `left_col0` and friends
    go negative when the window admits no key at all. See `resolve_window`.
    """
    one = fx.Int32(1)
    zero = fx.Int32(0)
    bn = fx.Int32(block_n)

    q_start = fx.Int32(start_q)
    q_hi = common_utils.smin(q_start + fx.Int32(block_m), q_len)
    q_last = q_hi - one

    # Blocks that exist at all, and the last block that is *whole*. Splitting
    # these is section 2.2 case 3: a ragged seqlen_k leaves a partial final
    # tile, which must be masked rather than counted as full.
    blk_last = common_utils.sdiv_rd_pow2(k_len - one, block_n)
    blk_last_whole = common_utils.sdiv_rd_pow2(k_len, block_n) - one

    # The visited range: outside it every column is dead for every row in this
    # Q block, so those tiles are not walked at all.
    v_lo = common_utils.smax(
        common_utils.sdiv_rd_pow2(q_start - window_left, block_n), zero
    )
    v_hi = common_utils.smin(
        blk_last, common_utils.sdiv_rd_pow2(q_last + window_right, block_n)
    )
    v_hi = common_utils.ssel(alive, v_hi, v_lo - one)

    # Rounded *up* on the left: a block is fully live only once its first
    # column clears the leftmost row's window. Rounding down would send a
    # partly-masked tile through the unmasked loop body -- invisible to a
    # tolerance test, not to the bitwise one.
    l_first_full = common_utils.sdiv_rd_pow2(
        q_last - window_left + fx.Int32(block_n - 1), block_n
    )
    r_first_mask = common_utils.sdiv_rd_pow2(q_start + window_right + one, block_n)

    fb_lo = common_utils.smax(l_first_full, v_lo)
    fb_hi = common_utils.smin(
        common_utils.smin(r_first_mask - one, blk_last_whole), v_hi
    )
    fb_empty = fb_lo > fb_hi

    # Cut [v_lo, v_hi] at the full region. With no full region the whole range
    # becomes one masked run -- section 2.2 case 2, the window narrower than a
    # block, falling out for free.
    lb_hi = common_utils.ssel(fb_empty, v_hi, fb_lo - one)
    rb_lo = common_utils.ssel(fb_empty, v_hi + one, fb_hi + one)

    n_left = common_utils.smax(lb_hi - v_lo + one, zero)
    n_full = common_utils.smax(fb_hi - fb_lo + one, zero)
    n_right = common_utils.smax(v_hi - rb_lo + one, zero)

    left_col0 = v_lo * bn
    right_col0 = rb_lo * bn
    full_col0 = fb_lo * bn
    # First tile of the masked run, which is also what the full loop's last
    # prefetch must fetch: the two loops are adjacent only when the left run is
    # empty. Clamped, because with a window admitting no key at all every run
    # is empty and `rb_lo` sits below zero -- and this value still reaches the
    # prologue's address computation.
    masked_col0 = common_utils.smax(
        common_utils.ssel(n_left > zero, left_col0, right_col0), zero
    )
    return CausalRegions(
        n_left, n_full, n_right, left_col0, full_col0, right_col0, masked_col0
    )


def make_addr_pair(
    strides, head, batch_index, row_off, *, seqlen_k, seq_last, hoist, clamp
):
    """Address builders for one tensor: `(tbase, toff, kv_addr)`.

    Q, K, V and O each get their own. They genuinely differ: K and V are
    whatever the caller allocated, and under MQA/GQA they carry `num_head_k`
    rather than `num_head_q`, so their head stride differs from Q's by
    construction. Assuming one shared layout is not a simplification, it is
    wrong.

    `hoist` and `clamp` are `const_expr` -- they select which code is emitted,
    not which branch runs.
    """
    # `row_off` is the varlen row offset, and it belongs in the
    # **64-bit base** rather than the 32-bit per-lane offset: on a
    # packed tensor it is a whole-batch quantity and overflows 32 bits
    # at realistic token counts (sdpa-varlen-plan.md section 5).
    # BHSD slot order: batch, head, sequence. See `_strides_of` in the kernel
    # for why the order is the ABI rather than a convenience.
    s_batch, s_head, s_seq = strides
    bh = batch_index * s_batch + head * s_head + row_off * s_seq

    def tbase(seq_start):
        """Uniform 64-bit element base for (batch, head, seq_start).

        `seq_start` is a position on whichever sequence axis this
        tensor is indexed by -- rows for Q/O, KV columns for K/V --
        since `make_addr_pair` builds one of these per tensor.
        """
        return bh + seq_start * s_seq

    def toff(row_in_tile, col):
        """Divergent 64-bit element offset inside the tile.

        64-bit because `row_in_tile * s_seq` genuinely does not fit in
        32: nothing requires the caller's tensor to be compact, and a
        view keeps its source's strides -- slicing `(1, 64, 16384,
        512)` f16, a 1 GiB tensor, down to eight heads leaves
        `s_seq = 8388608`, and 256 rows of that is exactly 2**31.

        It is worth keeping separate from `tbase` for callers outside
        the KV loop, where it is loop-invariant and LICM pays the
        64-bit width once. Inside the loop it is `kv_off` that decides
        whether that stays true.
        """
        return row_in_tile * s_seq + col

    def kv_off(ts, row_in_tile, col):
        """`toff` for a KV row, with the out-of-range row folded in.

        Two forms of the same value, and the whole difference is
        whether `row_in_tile * s_seq` stays loop-invariant.

        Recomputed (KV_ADDR_HOIST off) clamps the row first, so `row`
        depends on `ts`, which moves every KV iteration: the 64-bit
        multiply is loop-carried and re-emitted per load per
        iteration. At BLOCK_DMODEL 192 that is 14 `v_mul_lo_u32` and
        21 `v_add_co_u32` in the loop body, against 3 and 11 for the
        pre-64-bit kernel.

        Hoisted selects between two whole offsets instead, so both
        arms are loop-invariant per-lane values and the one uniform
        term is factored out of the select: the loop pays two adds and
        the select, and the multiply leaves it entirely. What it costs
        is one more 64-bit value live per cooperative load, which is
        why this is a knob and not simply the better form -- see
        `_KV_ADDR_HOIST_HEAD_DIMS` in the tuning module for where each
        one wins.

        The hoisted out-of-range arm sends the lane to its own column
        in row `ts`, the tile's first row, rather than to the last row
        of the sequence: any in-bounds address will do, since the
        value is discarded, and this one shares `ts * s_seq` with the
        in-range arm. `col` and not the literal `0`, which is equally
        in bounds and needs no register of its own -- the 0 arm holds
        one value fewer live and still spills *more*, 272 bytes of
        scratch against 44 at BLOCK_DMODEL 192, for 0.863 against
        1.172 on the same baseline. Re-measure before changing it.

        Each form states the bounds predicate its own way, and that is
        deliberate rather than untidy. `row_in_tile < seqlen_k - ts`
        puts the whole uniform half on one side, so it is one compare
        against an SGPR instead of a divergent 64-bit add and compare
        -- but only the hoisted form is free to use it, because the
        recomputed one needs `seq_last - ts` for its clamp anyway and
        because keeping it verbatim is what makes a knob-off build
        bitwise identical to the kernel before this knob existed.
        """
        if not hoist:
            in_range = (ts + row_in_tile) < seqlen_k
            row = fx.Index(in_range.select(row_in_tile, seq_last - ts))
            return toff(row, col)
        # `ts < seqlen_k` always -- it is either start_k, which the
        # caller's branch tested, or seq_last -- so this cannot wrap.
        in_range = row_in_tile < (seqlen_k - ts)
        return fx.Index(in_range.select(toff(row_in_tile, col), col))

    def kv_addr(start_k, row_in_tile, col):
        """(uniform base, divergent offset) for a KV row, clamped in bounds.

        At K_PREFETCH_DIST == 1 the loop runs one tile ahead, so the final
        iteration addresses a tile past the end of the sequence; the unguarded
        cooperative load also addresses rows past BLOCK_N. Clamp start_k
        first, then send any row still past the end to the last row of the
        sequence. The values are never consumed; the clamp exists only so the
        address stays inside the allocation.

        With both prefetch distances 0 and no load guard there is no over-read
        -- BLOCK_N divides BLOCK_M and the tail is masked -- so `clamp` is
        false and this is pure VALU saved.
        """
        if not clamp:
            return tbase(start_k), toff(row_in_tile, col)
        ts = fx.Index((start_k < seqlen_k).select(start_k, seq_last))
        return tbase(ts), kv_off(ts, row_in_tile, col)

    return tbase, toff, kv_addr
