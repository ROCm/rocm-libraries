# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.
"""Merged backward for gfx950: dK/dV and dQ (and dB) in ONE launch, no prep, no atomics.

    grid = (n_kv + n_q, 1, 1)   n_kv = Hk * ceil(Sk / BLOCK_KV) * B   dK/dV workgroups (one per KV block, KV head, batch)
                                n_q  = Hq * ceil(Sq / BLOCK_M)  * B   dQ workgroups (one per Q block, Q head, batch)

A workgroup's program id picks its role (uniform scalar branch, `role_order` below); the two roles are the existing dK/dV and dQ
bodies, emitted through `emit_dkdv_body` / `emit_dq_body`, each with its own context built from the role's (head, block, batch)
decode. Every output has exactly one writer (dK/dV: its KV workgroup, dQ and dB: its Q workgroup), so the result is bitwise
run-to-run deterministic.

**Prep folded in.**
  - Delta = rowsum(dO * O) in fp32, in place: the dQ role once per Q row (O and dO loaded in the dQ operand layout, a lane-pair
    reduction), the dK/dV role per streamed Q tile (dO read from the staged LDS tile, O from global in the same A-operand lane map,
    reduced across the lanes that split a row, parked in a per-wave LDS scratch of BLOCK_Q floats and read back in the accumulator
    row map). The dK/dV scratch is the only LDS the role adds. AOTriton's bwd_kernel_fuse recomputes Delta per streamed tile too.
  - LSE: -inf (hipDNN's empty-row value) becomes +inf before the log2 conversion in both roles, so P = 0 on those rows.

**LDS.** The roles are mutually exclusive branches, so one arena buffer (sized for the larger role) holds dK/dV's stream buffers + Delta scratch and dQ's
K/V staging (plus the staged bias tile of a bias build, behind K/V) at the same addresses (the larger sets the size).

**Block size.** One workgroup size for both roles: the dK/dV policy's num_waves. The dQ role is built at that wave count
(BLOCK_M = 32 * num_waves, policy BLOCK_N); at t128 that is 4 waves where the dQ policy alone would pick 2.

`role_order`: `concat` = all dK/dV workgroups then all dQ workgroups; `lpt` = alternate dK/dV, dQ, dK/dV, ... (dK/dV first, the
longer job) until one kind runs out, then the rest of the other kind.

ABI: the dK/dV launch's argument list with O added (pointer + three strides) and every output (DK, DV, DQ, DB) present; no Delta.
**dBias reduction.** `dbias_reduce=True` (bias builds) adds a second launch of `fmha_bwd_dbias_reduce_gfx950`'s kernel after the main one, so the object
carries two kernel symbols: the merged kernel and the reduce (`fmha_bwd_dbias_reduce_gfx950_kernel_1`). The launcher's arguments gain the reduce
launch's (`DBias`, `reduce_batch`, `reduce_head`, `stride_ws_*`, `stride_out_*`; its `DB` is the merged kernel's `DB`); they serve compilation only, since
each kernel's kernarg is packed from its own declaration. The reduce launch resets the merged module's fast-math compile hints: the reduce sums
in IEEE fp32 (its standalone build has no hints), so the merged kernel's `fast_fp_math` must not reach it. The reset changes the trace-time fast-math flags
only: the module-wide target flags and the merged kernel's LLVM options still apply to the reduce kernel.
Causal / window: each role runs its split causal body (window and bottom-right resolution are runtime wire values, as in the split kernels).
`gqa_pack` (dK/dV knob, 16-row family; causal, window and dropout included): the dK/dV role walks the packed (group, q) row axis; its in-kernel Delta reads O at
the packed row's (head, q) address.
Scope: dense and varlen layouts (the decode is shared with the split kernels), padded_head builds, the 32-row dQ family; `xcd_swizzle`,
`lpt_tile_order` with causal, and roles that differ in `daz` are refused by the builder.
"""

import contextlib
from types import SimpleNamespace as _WireArgs

from kernels.attention_bwd_cdna import (
    fmha_bwd_dbias_reduce_gfx950 as dbias_reduce_module,
)
from kernels.attention_bwd_cdna import fmha_bwd_dkdv_gfx950 as dkdv
from kernels.attention_bwd_cdna import fmha_bwd_dkdv_m16_gfx950 as m16
from kernels.attention_bwd_cdna import fmha_bwd_dq_gfx950 as dq
from kernels.attention_bwd_cdna import fmha_common_gfx1201 as fmha
from kernels.attention_bwd_cdna import philox
from kernels.attention_bwd_cdna.fmha_dualwave_gfx950 import (
    _fmul,
    _slab_span_elems,
    wire_view,
)
from kernels.attention_bwd_cdna.gfx950_standalone import buffer_ops, dualwave

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import ReductionOp, T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

__all__ = ["KERNEL_NAME", "ROLE_ORDERS", "build_fmha_bwd_merged_gfx950_module_primary"]

KERNEL_NAME = "fmha_bwd_merged_gfx950_kernel"
ROLE_ORDERS = ("concat", "lpt")
_COMPILE_HINTS = dkdv._COMPILE_HINTS
_SCRATCH_SCOPE = "lds_dscr"  # alias scope of the Delta scratch: disjoint from the four staging scopes
_NEG_INF_BITS = -8388608  # 0xFF800000


# ------------------------------------------------------------------------------------------------------------ program id
class _Dim3:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _GpuProxy:
    """`flydsl.expr.gpu` with `block_idx` replaced: what the role's context sees as its grid coordinates."""

    def __init__(self, real, block_idx):
        self._real, self.block_idx = real, block_idx

    def __getattr__(self, name):
        return getattr(self._real, name)


@contextlib.contextmanager
def _program_id(block_idx):
    """The shared context decodes (head, block, batch) from `gpu.block_idx` inside `init_thread_mapping`; the merged grid is
    one-dimensional, so each role's context is built with the decoded triple standing in for it. Scoped to that one call.
    """
    mods = [dualwave, fmha]
    saved = [(m, m.gpu) for m in mods]
    for m, g in saved:
        m.gpu = _GpuProxy(g, block_idx)
    try:
        yield
    finally:
        for m, g in saved:
            m.gpu = g


def _neg_inf_to_pos_inf(v):
    """-inf -> +inf, everything else (NaN included) unchanged; an integer compare so fast-math cannot assume it away."""
    v = fx.Float32(v)
    return (
        v.bitcast(fx.Int32)
        .__eq__(fx.Int32(_NEG_INF_BITS))
        .select(fx.Float32(float("inf")), v)
    )


# ---------------------------------------------------------------------------------------------------------- Delta scratch
def _scratch_ptr(ctx, elem_off):
    """This wave's Delta scratch, `elem_off` f32 in. Lives past the stream buffers of the dK/dV role's arena."""
    traits = ctx.traits
    base = traits.LDS_STREAM_TOTAL_SIZE * traits.BF16_BYTES
    wave = ctx.wave_id_uni * fx.Index(traits.BLOCK_Q * 4)
    return buffer_ops.get_element_ptr(
        ctx.lds_kv_base_ptr,
        byte_offset=fx.Index(base) + wave + elem_off * fx.Index(4),
        elem_type=T.i8,
    )


def _scratch_store(ctx, row, value):
    llvm.StoreOp(
        fx.Float32(value).ir_value(),
        _scratch_ptr(ctx, row),
        alignment=4,
        alias_scopes=dualwave._dualwave_lds_alias_scopes(_SCRATCH_SCOPE),
        noalias_scopes=dualwave._lds_alias_scope_array(ctx.traits.LDS_SCOPE_NAMES),
    )


def _scratch_load4(ctx, row):
    """Four contiguous f32 rows starting at `row` (a multiple of 4): one 128-bit LDS read."""
    raw = llvm.LoadOp(
        Vec.make_type(4, fx.Float32),
        _scratch_ptr(ctx, row),
        alignment=16,
        alias_scopes=dualwave._dualwave_lds_alias_scopes(_SCRATCH_SCOPE),
        noalias_scopes=dualwave._lds_alias_scope_array(ctx.traits.LDS_SCOPE_NAMES),
    ).result
    v = Vec(raw, (4,), fx.Float32)
    return [v[i] for i in range_constexpr(4)]


def _dot8(ctx, a, b, acc):
    """`acc + sum_i a[i] * b[i]` over two 8-element packs, fp32 products (what the prep does)."""
    a = Vec(a, (8,), ctx.elem_dtype).to(fx.Float32)
    b = Vec(b, (8,), ctx.elem_dtype).to(fx.Float32)
    return acc + (a * b).reduce(ReductionOp.ADD)


# ------------------------------------------------------------------------------------------------------ dK/dV role pieces
class MergedDkDvContext(dkdv.BwdDkDvKernelContext):
    """The dK/dV context with a decoded program id, the shared arena and O (a raw bounded resource per query head)."""

    def __init__(self, traits, *, arena, pid_xyz, O, o_strides, **kwargs):
        super().__init__(traits, **kwargs)
        self.arena, self.pid_xyz, self.O_ptr = arena, pid_xyz, O
        self.stride_o_batch, self.stride_o_head, self.stride_o_seq = o_strides

    def init_runtime_indices(self, **kwargs):
        super().init_runtime_indices(**kwargs)
        self.stride_o_seq_v = fx.Index(self.stride_o_seq)

    def init_lds(self, shared_storage):
        lds = self.arena
        self.lds = lds
        self.lds_kv_base_idx = fx.Index(fx.ptrtoint(lds.mem.ptr))
        self.lds_kv_base_ptr = lds.mem.ptr.llvm_ptr
        self.lds_bt_base_ptr = None

    def init_thread_mapping(self):
        with _program_id(self.pid_xyz):
            super().init_thread_mapping()

    def _init_q_head_invariants(self):
        super()._init_q_head_invariants()
        self.o_slab_span_elems = _slab_span_elems(
            self.seqlen_q_v, self.stride_o_seq, self.hdim_vo
        )
        self.o_slab_base0_bytes = self._slab_byte_base(
            self.stride_o_batch,
            self.stride_o_head,
            self.stride_o_seq,
            self.q_row_off,
            self.gqa_q_head_base,
        )
        self.o_head_step_bytes = fx.Index(self.stride_o_head) * fx.Index(
            self.traits.BF16_BYTES
        )
        if const_expr(self.GQA_PACK):
            # As the q/dO descriptors: the packed axis addresses the whole group from head 0's base; a row past the group is
            # redirected to the descriptor's own span.
            self.o_slab_span_elems = self.o_slab_span_elems + (
                self.gqa_group - fx.Index(1)
            ) * fx.Index(self.stride_o_head)
            self.pack_o_oob = self.o_slab_span_elems

    def bind_q_head(self, g):
        super().bind_q_head(g)
        self.o_rsrc = dualwave._make_ws_rsrc(
            fx.Int64(fx.ptrtoint(fx.get_iter(self.O_ptr))),
            self.o_slab_base0_bytes + g * self.o_head_step_bytes,
            self.o_slab_span_elems * fx.Index(self.traits.BF16_BYTES),
        )


class _MergedRowsMixin:
    """LSE select and in-place Delta for both dK/dV families. The family supplies `_a_row` / `_a_cols` / `_reduce` / `_rows_of`."""

    def load_neg_lse2(self, tile_base, h):
        ctx = self.ctx_ref
        lse = self.softmax.load_row_values(ctx.lse_rsrc, tile_base, h, ctx.c_one_f)
        return [
            _fmul(_neg_inf_to_pos_inf(x), ctx.c_neg_log2e, self.fm_fast) for x in lse
        ]

    def load_delta(self, tile_idx, tile_base, h, do_base, do_scope):
        ctx = self.ctx_ref
        row_in_tile = self._a_row(h)
        if const_expr(ctx.GQA_PACK):
            g, q, valid = m16.pack_split(ctx, tile_base + row_in_tile)
            o_row = fx.Index(
                valid.select(
                    g * fx.Index(ctx.stride_o_head) + q * ctx.stride_o_seq_v,
                    ctx.pack_o_oob,
                )
            )
        else:
            o_row = (tile_base + row_in_tile) * ctx.stride_o_seq_v
        acc = fx.Float32(0.0)
        for ks in range_constexpr(self._k_steps):
            d = self.reader.read_row_pack(do_base, do_scope, h, ks, self.hdim_vo)
            o = buffer_ops.buffer_load(
                ctx.o_rsrc,
                as_mlir_value(fx.Int32(o_row + self._a_col(ks))),
                vec_width=8,
                dtype=ctx.elem_dtype,
            )
            acc = _dot8(ctx, d, o, acc)
        acc = self._reduce(acc)
        _scratch_store(ctx, row_in_tile, acc)
        return self._rows_of(h)


class MergedDkDvTileBody(_MergedRowsMixin, dkdv.BwdDkDvTileBody):
    """32-row family. A-operand row = `lane % 32` of the half, columns `ks*K_STEP_QK + 8*(lane//32)`; the accumulator holds rows
    `8*(r//4) + 4*(lane//32) + r%4` of the half."""

    @property
    def _k_steps(self):
        return self.traits.K_STEPS_QK

    def _a_row(self, h):
        return fx.Index(h * 32) + self.lane_mod_32

    def _a_col(self, ks):
        t = self.traits
        return fx.Index(ks * t.K_STEP_QK) + self.lane_div_32 * fx.Index(t.MFMA_LANE_K)

    def _reduce(self, acc):
        return acc + gpu.shuffle_xor(acc, 32, 64)

    def _rows_of(self, h):
        ctx = self.ctx_ref
        out = [None] * 16
        for elem0, col_off, width in dkdv._ROW_RUNS:
            assert width == 4
            vals = _scratch_load4(
                ctx, fx.Index(h * 32 + col_off) + self.lane_div_32 * fx.Index(4)
            )
            for j in range_constexpr(4):
                out[elem0 + j] = vals[j]
        return out


class MergedM16TileBody(_MergedRowsMixin, m16.M16TileBody):
    """16-row family. A-operand row = permuted `lane % 16` of sub-block `sub`, columns `ks*32 + 8*(lane//16)`; the accumulator holds
    four contiguous rows `(sub//2)*32 + (sub%2)*4 + 8*(lane//16)`."""

    @property
    def _k_steps(self):
        return m16.k_steps32(self.traits)

    def _a_row(self, sub):
        lm = self.lane % fx.Index(16)
        return (
            fx.Index((sub // 2) * 32 + (sub % 2) * 4)
            + fx.Index(8) * (lm // fx.Index(4))
            + (lm % fx.Index(4))
        )

    def _a_col(self, ks):
        return fx.Index(ks * 32) + (self.lane // fx.Index(16)) * fx.Index(8)

    def _reduce(self, acc):
        acc = acc + gpu.shuffle_xor(acc, 16, 64)
        return acc + gpu.shuffle_xor(acc, 32, 64)

    def _rows_of(self, sub):
        return _scratch_load4(
            self.ctx_ref,
            fx.Index((sub // 2) * 32 + (sub % 2) * 4)
            + (self.lane // fx.Index(16)) * fx.Index(8),
        )


MERGED_DKDV_PARTS = dkdv.DkDvParts(
    ctx_cls=MergedDkDvContext,
    m32_tile=MergedDkDvTileBody,
    m32_softmax=dkdv.BwdDkDvSoftmaxHelper,
    m16_tile=MergedM16TileBody,
    m16_softmax=m16.M16SoftmaxHelper,
)


# ---------------------------------------------------------------------------------------------------------- dQ role pieces
class MergedDqContext(dq.BwdDqKernelContext):
    """The dQ context with a decoded program id, the shared arena, O, -inf -> +inf LSE and Delta computed from O and dO."""

    def __init__(self, traits, *, arena, pid_xyz, O, o_strides, **kwargs):
        super().__init__(traits, **kwargs)
        self.arena, self.pid_xyz, self.O_ptr = arena, pid_xyz, O
        self.stride_o_batch, self.stride_o_head, self.stride_o_seq = o_strides

    def init_runtime_indices(self, **kwargs):
        super().init_runtime_indices(**kwargs)
        self.stride_o_seq_v = fx.Index(self.stride_o_seq)

    def init_lds(self, shared_storage):
        lds = self.arena
        self.lds = lds
        self.lds_kv_base_idx = fx.Index(fx.ptrtoint(lds.mem.ptr))
        self.lds_kv_base_ptr = lds.mem.ptr.llvm_ptr
        self.lds_bt_base_ptr = None

    def bias_lds_base(self):
        """The staged bias region sits behind the dQ role's K/V staging in the arena."""
        off = self.traits.LDS_KV_TOTAL_SIZE * self.traits.BF16_BYTES
        ptr = buffer_ops.get_element_ptr(
            self.lds_kv_base_ptr, byte_offset=fx.Index(off), elem_type=T.i8
        )
        return self.lds_kv_base_idx + fx.Index(off), ptr

    def init_thread_mapping(self):
        with _program_id(self.pid_xyz):
            super().init_thread_mapping()

    def init_descriptors(self, **kwargs):
        super().init_descriptors(**kwargs)
        self.o_div = self._slab_view(
            self.O_ptr,
            self.stride_o_batch,
            self.stride_o_head,
            self.stride_o_seq,
            self.q_row_off,
            self.q_head_idx,
            self.seqlen_q_v,
            self.hdim_vo,
        )
        self.o_gmem_elem_offset = self.q_start * self.stride_o_seq_v

    def row_lse(self, lse):
        return _neg_inf_to_pos_inf(lse)

    def row_delta(self, q_row, off_i32):
        """rowsum(dO * O) for this lane's row: both loaded in the dQ operand map (row = lane % 32, half the columns per lane
        half), fp32 products, partner lane (lane ^ 32) added."""
        hdim = self.hdim_vo
        o_all = dq.BwdSecondaryQLoader(
            self, self.o_div, self.stride_o_seq_v, self.o_gmem_elem_offset, hdim
        ).load_all()
        do_all = dq.BwdSecondaryQLoader(
            self, self.do_div, self.stride_do_seq_v, self.do_gmem_elem_offset, hdim
        ).load_all()
        n = self.traits.K_STEPS_QK * self.traits.MFMA_LANE_K
        o = Vec(o_all, (n,), self.elem_dtype).to(fx.Float32)
        d = Vec(do_all, (n,), self.elem_dtype).to(fx.Float32)
        acc = (o * d).reduce(ReductionOp.ADD)
        return acc + gpu.shuffle_xor(acc, 32, 64)


# ------------------------------------------------------------------------------------------------------------- the builder
def _wire(**kw):
    return _WireArgs(**kw)


def build_fmha_bwd_merged_gfx950_module_primary(
    dk_meta, dk_knobs, dq_meta, dq_knobs, role_order="concat", dbias_reduce=False
):
    """Build the merged kernel from the two roles' resolved (meta, knobs) pairs. `dq_knobs` must be at the dK/dV block size."""
    if role_order not in ROLE_ORDERS:
        raise ValueError(f"role_order {role_order!r} not in {ROLE_ORDERS}")
    dk = dkdv.build_fmha_bwd_dkdv_gfx950_module_primary(dk_meta, dk_knobs)
    dqb = dq.build_fmha_bwd_dq_gfx950_module_primary(dq_meta, dq_knobs)
    kt, qt = dk.traits, dqb.traits
    if kt.BLOCK_SIZE != qt.BLOCK_SIZE:
        raise ValueError(
            f"the roles disagree on the workgroup size: dK/dV {kt.BLOCK_SIZE}, dQ {qt.BLOCK_SIZE}"
        )
    if qt.MFMA_N != 32:
        raise NotImplementedError(
            "the merged kernel's dQ role is the 32-row family only"
        )
    if bool(dk_knobs.gqa_pack) and kt.MFMA_ROWS != 16:
        raise ValueError("merged gqa_pack: 16-row dK/dV role")
    if (
        kt.DTYPE_STR != qt.DTYPE_STR
        or kt.BIAS_TYPE != qt.BIAS_TYPE
        or kt.ENABLE_DROPOUT != qt.ENABLE_DROPOUT
    ):
        raise ValueError("the roles must agree on dtype, bias and dropout")
    if dbias_reduce and not (kt.BIAS_TYPE and qt.STORE_DB):
        raise ValueError("dbias_reduce: a bias build whose dQ role stores dB only")
    # Each role's context decodes its (head, block, batch) from a proxy that replaces only `gpu.block_idx`; `gpu.grid_dim` is the
    # merged launch grid (total, 1, 1), which the swizzled / LPT-reversed block decodes below would read.
    for t, role in ((kt, "dK/dV"), (qt, "dQ")):
        if t.XCD_SWIZZLE:
            raise NotImplementedError(
                f"merged {role} role: xcd_swizzle reads gpu.grid_dim of the role's own grid"
            )
        if t.LPT_TILE_ORDER and t.CAUSAL:
            raise NotImplementedError(
                f"merged {role} role: causal lpt_tile_order reads gpu.grid_dim of the role's own grid"
            )
    if kt.DAZ != qt.DAZ:
        raise ValueError(
            f"the roles must agree on daz: dK/dV {kt.DAZ}, dQ {qt.DAZ} (one function attribute set covers both)"
        )
    if kt.ENABLE_DROPOUT and philox.default_width("gfx950") != 32:
        raise ValueError(
            "the gfx950 dropout stream is Philox-4x32 (the forward's); philox.default_width('gfx950') must be 32"
        )
    BLOCK_SIZE = kt.BLOCK_SIZE
    WPE = min(kt.WAVES_PER_EU, qt.WAVES_PER_EU)
    ELEM_BYTES = kt.BF16_BYTES
    DK_ELEMS = kt.LDS_STREAM_TOTAL_SIZE + kt.NUM_WAVES * kt.BLOCK_Q * 4 // ELEM_BYTES
    DQ_ELEMS = qt.LDS_KV_TOTAL_SIZE + qt.BIAS_LDS_BYTES // ELEM_BYTES
    REDUCE_VEC = dbias_reduce_module.VEC
    REDUCE_BLOCK = dbias_reduce_module.BLOCK_THREADS
    reduce_kernel = (
        dbias_reduce_module.build_fmha_bwd_dbias_reduce_gfx950_module_primary(
            kt.DTYPE_STR
        ).kernel
        if dbias_reduce
        else None
    )
    _lds_elem_dtype = dualwave.dtype_to_elem_type(kt.DTYPE_STR)
    _cache_tag = (dk.cache_tag, dqb.cache_tag, role_order, WPE, bool(dbias_reduce))

    @fx.struct
    class Arena:
        # One buffer, base address shared by the roles: dK/dV reads it as stream buffers + Delta scratch, dQ as K/V staging.
        mem: fx.Array[_lds_elem_dtype, max(DK_ELEMS, DQ_ELEMS), 16]

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def fmha_bwd_merged_gfx950_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        B: fx.Pointer,
        O: fx.Pointer,
        DO: fx.Pointer,
        DK: fx.Pointer,
        DV: fx.Pointer,
        DQ: fx.Pointer,
        DB: fx.Pointer,
        LSE: fx.Pointer,
        batch_size: fx.Int32,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        seqinfo_k0: fx.Pointer,
        seqinfo_k1: fx.Pointer,
        varlen_bits: fx.Int32,
        num_seqlens: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        philox_seed_ptr: fx.Pointer,
        philox_offset1: fx.Pointer,
        philox_offset2: fx.Int64,
        idropout_p: fx.Int32,
        dropout_scale: fx.Float32,
        num_head_q: fx.Int32,
        num_head_k: fx.Int32,
        hdim_qk: fx.Int32,
        hdim_vo: fx.Int32,
        sm_scale: fx.Float32,
        stride_q_batch: fx.Int64,
        stride_q_head: fx.Int64,
        stride_q_seq: fx.Int64,
        stride_k_batch: fx.Int64,
        stride_k_head: fx.Int64,
        stride_k_seq: fx.Int64,
        stride_v_batch: fx.Int64,
        stride_v_head: fx.Int64,
        stride_v_seq: fx.Int64,
        stride_o_batch: fx.Int64,
        stride_o_head: fx.Int64,
        stride_o_seq: fx.Int64,
        stride_do_batch: fx.Int64,
        stride_do_head: fx.Int64,
        stride_do_seq: fx.Int64,
        stride_dk_batch: fx.Int64,
        stride_dk_head: fx.Int64,
        stride_dk_seq: fx.Int64,
        stride_dv_batch: fx.Int64,
        stride_dv_head: fx.Int64,
        stride_dv_seq: fx.Int64,
        stride_dq_batch: fx.Int64,
        stride_dq_head: fx.Int64,
        stride_dq_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
        stride_db_batch: fx.Int64,
        stride_db_head: fx.Int64,
        stride_db_seq_q: fx.Int64,
        rt_flags: fx.Int32,
    ):
        arena = fx.SharedAllocator().allocate(Arena).peek()
        O = wire_view(O)

        # ---- program id -> role and (head, block, batch). Index arithmetic, workgroup-uniform.
        pid = fx.Index(gpu.block_idx.x)
        nkv_blocks = (fx.Index(max_seqlen_k) + fx.Index(kt.BLOCK_KV - 1)) // fx.Index(
            kt.BLOCK_KV
        )
        nq_blocks = (fx.Index(max_seqlen_q) + fx.Index(qt.BLOCK_M - 1)) // fx.Index(
            qt.BLOCK_M
        )
        hk, hq = fx.Index(num_head_k), fx.Index(num_head_q)
        batches = fx.Index(num_seqlens if num_seqlens != fx.Int32(0) else batch_size)
        n_kv = hk * nkv_blocks * batches
        n_q = hq * nq_blocks * batches
        if const_expr(role_order == "concat"):
            is_dk = pid < n_kv
            idx_dk = pid
            idx_dq = pid - n_kv
        else:
            m = fx.Index((n_kv < n_q).select(n_kv, n_q))
            paired = pid < m * fx.Index(2)
            is_dk = paired.select((pid & fx.Index(1)) == fx.Index(0), n_kv > n_q)
            idx_dk = paired.select(pid // fx.Index(2), pid - m)
            idx_dq = idx_dk
        dk_pid = _Dim3(
            idx_dk % hk, (idx_dk // hk) % nkv_blocks, idx_dk // (hk * nkv_blocks)
        )
        dq_pid = _Dim3(
            idx_dq % hq, (idx_dq // hq) % nq_blocks, idx_dq // (hq * nq_blocks)
        )

        @flyc.jit
        def _dispatch():
            if is_dk:
                dk.emit(
                    _wire(
                        Q=Q,
                        K=K,
                        V=V,
                        B=B,
                        DO=DO,
                        DK=DK,
                        DV=DV,
                        LSE=LSE,
                        Delta=LSE,
                        seqinfo_q0=seqinfo_q0,
                        seqinfo_q1=seqinfo_q1,
                        seqinfo_k0=seqinfo_k0,
                        seqinfo_k1=seqinfo_k1,
                        varlen_bits=varlen_bits,
                        num_seqlens=num_seqlens,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_k=max_seqlen_k,
                        window_left=window_left,
                        window_right=window_right,
                        philox_seed_ptr=philox_seed_ptr,
                        philox_offset1=philox_offset1,
                        philox_offset2=philox_offset2,
                        idropout_p=idropout_p,
                        dropout_scale=dropout_scale,
                        num_head_q=num_head_q,
                        num_head_k=num_head_k,
                        hdim_qk=hdim_qk,
                        hdim_vo=hdim_vo,
                        sm_scale=sm_scale,
                        stride_q_batch=stride_q_batch,
                        stride_q_head=stride_q_head,
                        stride_q_seq=stride_q_seq,
                        stride_k_batch=stride_k_batch,
                        stride_k_head=stride_k_head,
                        stride_k_seq=stride_k_seq,
                        stride_v_batch=stride_v_batch,
                        stride_v_head=stride_v_head,
                        stride_v_seq=stride_v_seq,
                        stride_do_batch=stride_do_batch,
                        stride_do_head=stride_do_head,
                        stride_do_seq=stride_do_seq,
                        stride_dk_batch=stride_dk_batch,
                        stride_dk_head=stride_dk_head,
                        stride_dk_seq=stride_dk_seq,
                        stride_dv_batch=stride_dv_batch,
                        stride_dv_head=stride_dv_head,
                        stride_dv_seq=stride_dv_seq,
                        stride_b_batch=stride_b_batch,
                        stride_b_head=stride_b_head,
                        stride_b_seq_q=stride_b_seq_q,
                        rt_flags=rt_flags,
                    ),
                    parts=MERGED_DKDV_PARTS,
                    ctx_kwargs=dict(
                        arena=arena,
                        pid_xyz=dk_pid,
                        O=O,
                        o_strides=(stride_o_batch, stride_o_head, stride_o_seq),
                    ),
                )
            else:
                dqb.emit(
                    _wire(
                        Q=Q,
                        K=K,
                        V=V,
                        B=B,
                        DO=DO,
                        DQ=DQ,
                        DB=DB,
                        LSE=LSE,
                        Delta=LSE,
                        seqinfo_q0=seqinfo_q0,
                        seqinfo_q1=seqinfo_q1,
                        seqinfo_k0=seqinfo_k0,
                        seqinfo_k1=seqinfo_k1,
                        varlen_bits=varlen_bits,
                        num_seqlens=num_seqlens,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_k=max_seqlen_k,
                        window_left=window_left,
                        window_right=window_right,
                        philox_seed_ptr=philox_seed_ptr,
                        philox_offset1=philox_offset1,
                        philox_offset2=philox_offset2,
                        idropout_p=idropout_p,
                        dropout_scale=dropout_scale,
                        num_head_q=num_head_q,
                        num_head_k=num_head_k,
                        hdim_qk=hdim_qk,
                        hdim_vo=hdim_vo,
                        sm_scale=sm_scale,
                        stride_q_batch=stride_q_batch,
                        stride_q_head=stride_q_head,
                        stride_q_seq=stride_q_seq,
                        stride_k_batch=stride_k_batch,
                        stride_k_head=stride_k_head,
                        stride_k_seq=stride_k_seq,
                        stride_v_batch=stride_v_batch,
                        stride_v_head=stride_v_head,
                        stride_v_seq=stride_v_seq,
                        stride_do_batch=stride_do_batch,
                        stride_do_head=stride_do_head,
                        stride_do_seq=stride_do_seq,
                        stride_dq_batch=stride_dq_batch,
                        stride_dq_head=stride_dq_head,
                        stride_dq_seq=stride_dq_seq,
                        stride_b_batch=stride_b_batch,
                        stride_b_head=stride_b_head,
                        stride_b_seq_q=stride_b_seq_q,
                        stride_db_batch=stride_db_batch,
                        stride_db_head=stride_db_head,
                        stride_db_seq_q=stride_db_seq_q,
                        rt_flags=rt_flags,
                    ),
                    ctx_cls=MergedDqContext,
                    ctx_kwargs=dict(
                        arena=arena,
                        pid_xyz=dq_pid,
                        O=O,
                        o_strides=(stride_o_batch, stride_o_head, stride_o_seq),
                    ),
                )

        _dispatch()

    @flyc.jit
    def launch_fmha_bwd_merged_gfx950(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        Bias: fx.Pointer,
        O: fx.Pointer,
        DO: fx.Pointer,
        DK: fx.Pointer,
        DV: fx.Pointer,
        DQ: fx.Pointer,
        DB: fx.Pointer,
        LSE: fx.Pointer,
        batch_size: fx.Int32,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        seqinfo_k0: fx.Pointer,
        seqinfo_k1: fx.Pointer,
        varlen_bits: fx.Int32,
        num_seqlens: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        philox_seed_ptr: fx.Pointer,
        philox_offset1: fx.Pointer,
        philox_offset2: fx.Int64,
        idropout_p: fx.Int32,
        dropout_scale: fx.Float32,
        num_head_q: fx.Int32,
        num_head_k: fx.Int32,
        hdim_qk: fx.Int32,
        hdim_vo: fx.Int32,
        sm_scale: fx.Float32,
        stride_q_batch: fx.Int64,
        stride_q_head: fx.Int64,
        stride_q_seq: fx.Int64,
        stride_k_batch: fx.Int64,
        stride_k_head: fx.Int64,
        stride_k_seq: fx.Int64,
        stride_v_batch: fx.Int64,
        stride_v_head: fx.Int64,
        stride_v_seq: fx.Int64,
        stride_o_batch: fx.Int64,
        stride_o_head: fx.Int64,
        stride_o_seq: fx.Int64,
        stride_do_batch: fx.Int64,
        stride_do_head: fx.Int64,
        stride_do_seq: fx.Int64,
        stride_dk_batch: fx.Int64,
        stride_dk_head: fx.Int64,
        stride_dk_seq: fx.Int64,
        stride_dv_batch: fx.Int64,
        stride_dv_head: fx.Int64,
        stride_dv_seq: fx.Int64,
        stride_dq_batch: fx.Int64,
        stride_dq_head: fx.Int64,
        stride_dq_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
        stride_db_batch: fx.Int64,
        stride_db_head: fx.Int64,
        stride_db_seq_q: fx.Int64,
        rt_flags: fx.Int32,
        DBias: fx.Pointer,
        reduce_batch: fx.Int32,
        reduce_head: fx.Int32,
        stride_ws_batch: fx.Int64,
        stride_ws_head: fx.Int64,
        stride_ws_seq: fx.Int64,
        stride_out_batch: fx.Int64,
        stride_out_head: fx.Int64,
        stride_out_seq: fx.Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = _cache_tag
        bs_idx = fx.Index(num_seqlens if num_seqlens != fx.Int32(0) else batch_size)
        nkv_blocks = (fx.Index(max_seqlen_k) + fx.Index(kt.BLOCK_KV - 1)) // fx.Index(
            kt.BLOCK_KV
        )
        nq_blocks = (fx.Index(max_seqlen_q) + fx.Index(qt.BLOCK_M - 1)) // fx.Index(
            qt.BLOCK_M
        )
        total = bs_idx * (
            fx.Index(num_head_k) * nkv_blocks + fx.Index(num_head_q) * nq_blocks
        )
        passthrough_entries = (
            [
                ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                ["no-nans-fp-math", "true"],
                ["unsafe-fp-math", "true"],
            ]
            if const_expr(kt.DAZ)
            else None
        )
        fmha_bwd_merged_gfx950_kernel(
            Q,
            K,
            V,
            Bias,
            O,
            DO,
            DK,
            DV,
            DQ,
            DB,
            LSE,
            batch_size,
            seqinfo_q0,
            seqinfo_q1,
            seqinfo_k0,
            seqinfo_k1,
            varlen_bits,
            num_seqlens,
            max_seqlen_q,
            max_seqlen_k,
            window_left,
            window_right,
            philox_seed_ptr,
            philox_offset1,
            philox_offset2,
            idropout_p,
            dropout_scale,
            num_head_q,
            num_head_k,
            hdim_qk,
            hdim_vo,
            sm_scale,
            stride_q_batch,
            stride_q_head,
            stride_q_seq,
            stride_k_batch,
            stride_k_head,
            stride_k_seq,
            stride_v_batch,
            stride_v_head,
            stride_v_seq,
            stride_o_batch,
            stride_o_head,
            stride_o_seq,
            stride_do_batch,
            stride_do_head,
            stride_do_seq,
            stride_dk_batch,
            stride_dk_head,
            stride_dk_seq,
            stride_dv_batch,
            stride_dv_head,
            stride_dv_seq,
            stride_dq_batch,
            stride_dq_head,
            stride_dq_seq,
            stride_b_batch,
            stride_b_head,
            stride_b_seq_q,
            stride_db_batch,
            stride_db_head,
            stride_db_seq_q,
            rt_flags,
            value_attrs={
                "rocdl.waves_per_eu": WPE,
                "rocdl.flat_work_group_size": f"{BLOCK_SIZE},{BLOCK_SIZE}",
                "passthrough": passthrough_entries,
            },
        ).launch(grid=(total, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)
        if const_expr(dbias_reduce):
            nvec = (fx.Index(max_seqlen_k) + fx.Index(REDUCE_VEC - 1)) // fx.Index(
                REDUCE_VEC
            )
            nblk = (
                fx.Index(max_seqlen_q) * nvec + fx.Index(REDUCE_BLOCK - 1)
            ) // fx.Index(REDUCE_BLOCK)
            out_b = (reduce_batch != fx.Int32(0)).select(fx.Int32(1), batch_size)
            out_h = (reduce_head != fx.Int32(0)).select(fx.Int32(1), num_head_q)
            with CompilationContext.compile_hints(
                {"fastmath": None, "fast_fp_math": False, "unsafe_fp_math": False}
            ):
                reduce_kernel(
                    DB,
                    DBias,
                    batch_size,
                    num_head_q,
                    max_seqlen_q,
                    max_seqlen_k,
                    reduce_batch,
                    reduce_head,
                    stride_ws_batch,
                    stride_ws_head,
                    stride_ws_seq,
                    stride_out_batch,
                    stride_out_head,
                    stride_out_seq,
                ).launch(
                    grid=(nblk, fx.Index(out_b) * fx.Index(out_h), 1),
                    block=(REDUCE_BLOCK, 1, 1),
                    stream=stream,
                )

    def _launch(*args, **kwargs):
        raise NotImplementedError(
            "launch through the jit entry point (`_launch.jit`): the merged kernel takes named arguments"
        )

    _launch.traits = (kt, qt)
    _launch.role_order = role_order
    _launch.jit = launch_fmha_bwd_merged_gfx950
    return _launch
