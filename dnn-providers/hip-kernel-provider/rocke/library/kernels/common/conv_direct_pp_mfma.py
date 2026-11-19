# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The ping-pong direct convolution pipeline on wave64 MFMA with LDS DMA (gfx950).

A dense NHWC KxK stride-1 pass as a channel-chunked direct kernel, shared by
the forward (:mod:`kernels.common.conv_direct_fwd_pp_mfma`) and the dgrad
(:mod:`kernels.common.conv_direct_dgrad_pp_mfma`, the forward pass over the
transposed problem). The two differ only in how the filter reaches LDS and
how the A (weight) fragments are read back; a :class:`PpWeights` supplies
that half, and this module emits everything else:

* **LDS DMA.** Every operand byte reaches LDS through ``buffer_load ... lds``
  (16 bytes per lane; 4 for the narrow tail of an input row), never through
  VGPRs and ``ds_write``. A DMA instruction writes 1 KiB lane-contiguously, so
  the LDS swizzle is applied on the *global* side. An input row of 64
  channels is 8 granules of 16 bytes, granule ``g`` of column ``x`` stored at
  slot ``g ^ (x % 8)``: by the slot arithmetic, a fragment read (16 columns x
  one granule) hits 8 distinct slots per 8 lanes (a layout argument, not a
  counter measurement).
* **Two wave ranks.** The 8 waves of a block form two ranks of 4. Rank 0
  streams the weights of the *next filter tap* into a double buffer; rank 1
  streams the *next channel chunk's* input tile (the ``(TH + KH - 1) x (TW +
  KW - 1)`` halo, 64 channels) into the other input buffer, a few 1 KiB
  pieces per step, so the loads trickle under the MFMAs instead of arriving
  as one burst.
* **Ping-pong schedule.** Rank 1 starts one barrier late. Every step (one tap
  and one 32-channel half of the chunk) is ``reads + DMA issue; barrier;
  setprio 1; MFMAs; setprio 0; barrier``, so while one rank's MFMAs run at
  raised priority the other rank reads its fragments and issues its DMA.

The DMA completion protocol is explicit (``s_waitcnt vmcnt``): rank 0 waits
for the next tap's weights after the second half of the current tap, rank 1
for the next chunk's input before the last step of the chunk. Every buffer is
its own LDS allocation with its own alias scope, and the chunk loop is
unrolled twice so each read names its buffer statically: the waitcnt pass then
only waits for DMA into the buffer a read actually touches.

Wave tile: rank-agnostic ``wave_j = wave % 2`` owns half the pixel rows of the
16x16 output tile (8 rows of 16 pixels: 8 B fragments), ``wave_k = wave / 2``
a quarter of the block's output channels (``block_k / 64`` A fragments); 16
``mfma_f32_16x16x32`` per step and A fragment.

AOT: batch, extents, C and K are kernargs (the forward direct-conv kernarg
block); the filter, padding, dtype and tile are baked. The C++ mirror is
``rocke/instance_conv_direct_pp_mfma_internal.hpp``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from rocke.core.arch import ArchTarget
from rocke.core.ir import F32, I64, IRBuilder, KernelDef, Value, VectorType
from rocke.helpers.spec import ceil_div

from kernels.common.conv_direct_grouped import (
    I32_MAX,
    DirectConvProblem,
    _buf_store_vN,
    _io_type,
    _mfma,
    _trunc_f32,
    emit_direct_params,
    one_padding_reason,
    problem_dims_reason,
)

__all__ = [
    "PpCtx",
    "PpMfmaSpec",
    "PpWeights",
    "emit_pp_mfma",
    "pp_mfma_reason",
    "pp_mfma_tile_efficiency",
]

WAVE = 64
CK = 64  # channels per chunk
GRAN = 8  # halves per 16-byte granule
TH = 16  # output rows per block
TW = 16  # output columns per block
WAVES = 8
RANK_WAVES = 4
ATOM = 16
PARTS_C = CK // 32  # 32-channel MFMA steps per chunk and tap


@dataclass(frozen=True)
class PpMfmaSpec:
    """The configuration both ping-pong builders take; each family subclasses
    it with its own ``name``."""

    problem: DirectConvProblem
    name: str = "pp_mfma"
    block_k: int = 128  # output channels per block (4 waves x wave_k)
    wave_size: int = 64

    @property
    def wave_k(self) -> int:
        return self.block_k // 4

    @property
    def k_frags(self) -> int:  # A fragments (16 channels) per wave
        return self.wave_k // ATOM

    @property
    def in_h(self) -> int:
        return TH + self.problem.KH - 1

    @property
    def in_w(self) -> int:
        return TW + self.problem.KW - 1

    @property
    def in_rows(self) -> int:  # staged rows, rounded up to whole DMA groups
        return ceil_div(self.in_h, RANK_WAVES) * RANK_WAVES

    @property
    def in_elems(self) -> int:  # one input buffer
        return self.in_rows * self.in_w * CK

    @property
    def w_elems(self) -> int:  # one weight buffer
        return self.block_k * CK

    def lds_bytes(self) -> int:
        return 2 * (2 * self.in_elems + 2 * self.w_elems)

    @property
    def threads_per_block(self) -> int:
        return WAVES * self.wave_size

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            f"f{p.KH}x{p.KW}",
            f"p{p.PAD}",
            f"bk{self.block_k}",
            flags={"bf16": p.dtype == "bf16"},
        )

    def grid(self) -> Tuple[int, int, int]:
        p = self.problem
        n = (
            p.N
            * ceil_div(p.Ho, TH)
            * ceil_div(p.Wo, TW)
            * ceil_div(p.total_k, self.block_k)
        )
        return (n, 1, 1)

    def launch_geometry(self):
        return self.grid(), (self.threads_per_block, 1, 1)


def pp_mfma_tile_efficiency(spec: PpMfmaSpec) -> float:
    """Real work over the padded tile grid (pixels x output channels)."""
    p = spec.problem
    return (p.Ho * p.Wo * p.total_k) / (
        ceil_div(p.Ho, TH)
        * TH
        * ceil_div(p.Wo, TW)
        * TW
        * ceil_div(p.total_k, spec.block_k)
        * spec.block_k
    )


def pp_mfma_reason(
    spec: PpMfmaSpec,
    arch: str,
    *,
    family: str,
    block_ks: Sequence[int],
    block_k_reason: str,
    needs_ds_read_tr: bool = False,
) -> "str | None":
    """Why ``spec`` cannot build on ``arch`` (None when it can).

    ``family`` names the kernel in the reasons; ``block_ks`` are the block_k
    values its weight staging takes (``block_k_reason`` refuses the rest);
    ``needs_ds_read_tr`` gates on the transposing LDS read.
    """
    p = spec.problem
    target = ArchTarget.from_gfx(arch)
    if target.wave_size != 64 or spec.wave_size != 64:
        return f"{family} is a wave64 MFMA kernel"
    if needs_ds_read_tr and not target.memory.has_ds_read_tr:
        return f"{family} needs ds_read_b64_tr_b16 (absent on {arch})"
    if target.async_lds_max_dwords < 4:
        return f"{family} needs the 16-byte LDS DMA (absent on {arch})"
    if p.dtype not in ("fp16", "bf16"):
        return "fp16/bf16 only"
    ab = "bf16" if p.dtype == "bf16" else "f16"
    if not target.mma.has_shape(
        a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=16, n=16, k=32
    ):
        return f"missing mfma_f32_16x16x32_{ab} on {arch}"
    why = problem_dims_reason(p)
    if why is not None:
        return why
    if p.groups != 1:
        return "groups must be 1"
    if p.stride != 1:
        return "stride must be 1"
    if p.dil_h != 1 or p.dil_w != 1:
        return "dilation must be 1"
    if p.KH != p.KW or p.KH < 3 or p.KH % 2 == 0:
        return "square odd filters >= 3 only"
    if p.PAD != (p.KH - 1) // 2:
        return f"'same' padding only: PAD must be {(p.KH - 1) // 2} (got {p.PAD})"
    why = one_padding_reason(p)
    if why is not None:
        return why
    if p.cpg % GRAN:
        return f"C {p.cpg} must be a multiple of {GRAN} (16-byte granules)"
    if p.kpg % 4:
        return f"K {p.kpg} must be a multiple of 4 (8-byte output writes)"
    if spec.block_k not in block_ks:
        return block_k_reason
    if not target.fits_lds(spec.lds_bytes()):
        return f"LDS {spec.lds_bytes()} B > {target.lds_capacity_bytes}"
    return _pp_mfma_limits_reason(spec)


def _pp_mfma_limits_reason(spec: PpMfmaSpec) -> "str | None":
    """The i32 limits: the buffer descriptors span one input image, one
    output image and the whole filter, and a masked lane reads at byte offset
    ``I32_MAX``, which is out of range only while each of them is
    at most ``I32_MAX`` bytes; the whole input and output sizes are i32
    kernargs. The flat block id is an i32 too, but at most ``N * Ho * Wo * K``
    blocks fit in it once the output does."""
    p = spec.problem
    for what, nbytes in (
        ("input image", 2 * p.H * p.W * p.total_c),
        ("output image", 2 * p.Ho * p.Wo * p.total_k),
        ("filter", 2 * p.total_k * p.KH * p.KW * p.cpg),
    ):
        if nbytes > I32_MAX:
            return f"{what} exceeds {I32_MAX} B, the i32 buffer range"
    for what, nbytes in (
        ("input", 2 * p.N * p.H * p.W * p.total_c),
        ("output", 2 * p.N * p.Ho * p.Wo * p.total_k),
    ):
        if nbytes > I32_MAX:
            return f"{what} exceeds {I32_MAX} B, the i32 byte-count kernarg"
    return None


def _input_pieces(in_rows: int, in_w: int) -> List[List[Tuple[int, int, int]]]:
    """The DMA pieces of one input buffer, grouped 4 per step slot.

    A piece is ``(row, first column, dwords per lane)``: 8 columns (1 KiB) per
    16-byte piece, and the row's last ``in_w % 8`` (even) columns as 2-column
    4-byte pieces. The 4 pieces of a group share the column and width and take
    consecutive rows, one per rank-1 wave, so a wave's piece is ``group row +
    wave``; ``in_rows`` is a whole number of groups (the rows past the halo
    are never read).
    """
    cols = [(c0, 4) for c0 in range(0, in_w - 7, 8)]
    cols += [(c0, 1) for c0 in range(in_w - in_w % 8, in_w, 2)]
    groups = []
    for c0, dw in cols:
        for r0 in range(0, in_rows, RANK_WAVES):
            groups.append([(r0 + g, c0, dw) for g in range(RANK_WAVES)])
    return groups


@dataclass
class PpCtx:
    """What the pipeline has emitted that a :class:`PpWeights` builds on."""

    b: IRBuilder
    spec: PpMfmaSpec
    io_type: object
    lane: Value
    wave_k: Value
    wave_g: Value
    k0: Value
    C: Value
    K: Value
    c0: Value
    c2: Value
    oob: Value
    w_rsrc: Value
    w_bufs: List[Value]
    w_addr: List[Value]
    lds_at: Callable[[Value, Value], Value]
    # Lane values a PpWeights computed and the pipeline reuses (see
    # PpWeights.dma_geometry and PpWeights.a_geometry).
    shared: Dict[str, Value] = field(default_factory=dict)

    def ci(self, v: int) -> Value:
        return self.b.const_i32(v)


class PpWeights:
    """The family half of a ping-pong kernel: weight staging and A reads.

    Each method is called once, at a fixed point of the pipeline, so a
    family's IR is emitted in the same order as when it was a whole builder.
    """

    def dma_geometry(self, ctx: PpCtx) -> None:
        """Emit the rank-0 weight DMA geometry. May set ``ctx.shared`` keys
        ``l_row`` (lane // 8), ``l_gran`` ((lane % 8) ^ l_row) and ``lane_c``
        (l_gran * 8), which the input DMA geometry then reuses instead of
        emitting its own."""
        raise NotImplementedError

    def issue(self, ctx: PpCtx, buf: int, tap: int, c_base: Value) -> None:
        """DMA the weights of ``tap`` for the chunk at ``c_base`` into weight
        buffer ``buf``."""
        raise NotImplementedError

    def a_geometry(self, ctx: PpCtx) -> None:
        """Emit the A fragment read geometry. Must set ``ctx.shared['lane_hi']``
        (lane // 16, the 4-channel group of the output a lane holds); may set
        ``l16`` (lane % 16), which the B geometry then reuses."""
        raise NotImplementedError

    def read_a(self, ctx: PpCtx, buf: int, kf: int, part: int) -> Value:
        """A fragment ``kf`` of 32-channel half ``part`` from buffer ``buf``."""
        raise NotImplementedError


def emit_pp_mfma(spec: PpMfmaSpec, weights: PpWeights) -> KernelDef:
    """Emit the ping-pong kernel of ``spec`` (already validated) with the
    weight half ``weights``."""
    p = spec.problem
    KH, KW, PAD = p.KH, p.KW, p.PAD
    dtype = p.dtype
    io_type = _io_type(dtype)
    BK = spec.block_k
    KF = spec.k_frags
    IN_H, IN_W = spec.in_h, spec.in_w
    ROW_E = IN_W * CK  # elements per staged input row
    TAPS = KH * KW
    STEPS = TAPS * PARTS_C  # MFMA steps per chunk
    in_groups = _input_pieces(spec.in_rows, IN_W)
    # Spread the input groups over the first steps of a chunk; the last steps
    # leave the DMA time to land before the chunk-end wait.
    in_sched: List[List[int]] = [[] for _ in range(STEPS)]
    span = max(1, STEPS - 4)
    for i in range(len(in_groups)):
        in_sched[i * span // len(in_groups)].append(i)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads_per_block
    b.kernel.attrs["lds_alias_scopes"] = True
    b.kernel.attrs["waves_per_eu"] = 2
    params = emit_direct_params(b, io_type=io_type)

    def ci(v: int) -> Value:
        return b.const_i32(v)

    c0, c1, c2 = ci(0), ci(1), ci(2)
    oob = ci(I32_MAX)

    p_Hi, p_Wi = params["p_Hi"], params["p_Wi"]
    p_Ho, p_Wo = params["p_Ho"], params["p_Wo"]
    C, K = params["p_total_c"], params["p_total_k"]
    sa_n, sa_h, sa_w = (
        params["p_A_stride_n"],
        params["p_A_stride_hi"],
        params["p_A_stride_wi"],
    )
    sd_n, sd_h, sd_w = (
        params["p_D_stride_n"],
        params["p_D_stride_ho"],
        params["p_D_stride_wo"],
    )

    tid = b.thread_id_x()
    lane = b.mod(tid, ci(WAVE))
    wave = b.readfirstlane(b.div(tid, ci(WAVE)))
    wave_j = b.mod(wave, c2)
    wave_k = b.div(wave, c2)
    wave_g = b.mod(wave, ci(RANK_WAVES))
    rank = b.div(wave, ci(RANK_WAVES))

    # ---- grid decode: block -> (k tile fastest, w tile, h tile, image) ------
    n_kt = b.div(b.add(K, ci(BK - 1)), ci(BK))
    n_wt = b.div(b.add(p_Wo, ci(TW - 1)), ci(TW))
    n_ht = b.div(b.add(p_Ho, ci(TH - 1)), ci(TH))
    blk = b.block_id_x()
    k_tile = b.mod(blk, n_kt)
    rest = b.div(blk, n_kt)
    w_tile = b.mod(rest, n_wt)
    rest = b.div(rest, n_wt)
    h_tile = b.mod(rest, n_ht)
    n_img = b.div(rest, n_ht)
    oh0 = b.mul(h_tile, ci(TH))
    ow0 = b.mul(w_tile, ci(TW))
    k0 = b.mul(k_tile, ci(BK))

    # Per-image buffer bases keep every offset inside 32 bits.
    a_img = b.global_ptr_offset(
        params["A"], b.mul(b.zext(n_img, I64), b.zext(b.mul(sa_n, c2), I64))
    )
    d_img = b.global_ptr_offset(
        params["D"], b.mul(b.zext(n_img, I64), b.zext(b.mul(sd_n, c2), I64))
    )
    a_rsrc = b.buffer_rsrc(a_img, b.mul(sa_n, c2))
    w_rsrc = b.buffer_rsrc(params["B"], params["B_bytes"])
    d_rsrc = b.buffer_rsrc(d_img, b.mul(sd_n, c2))

    # ---- LDS ----------------------------------------------------------------
    in_bufs = [
        b.smem_alloc(io_type, [1, spec.in_elems], name_hint=f"in{i}") for i in range(2)
    ]
    w_bufs = [
        b.smem_alloc(io_type, [1, spec.w_elems], name_hint=f"w{i}") for i in range(2)
    ]
    in_addr = [b.smem_addr_of(s) for s in in_bufs]
    w_addr = [b.smem_addr_of(s) for s in w_bufs]

    def lds_at(base: Value, byte_off: Value) -> Value:
        return b.smem_ptr_add(base, b.zext(byte_off, I64))

    ctx = PpCtx(
        b=b,
        spec=spec,
        io_type=io_type,
        lane=lane,
        wave_k=wave_k,
        wave_g=wave_g,
        k0=k0,
        C=C,
        K=K,
        c0=c0,
        c2=c2,
        oob=oob,
        w_rsrc=w_rsrc,
        w_bufs=w_bufs,
        w_addr=w_addr,
        lds_at=lds_at,
    )

    # ---- rank 0: weight DMA geometry (the family's) -------------------------
    weights.dma_geometry(ctx)

    # ---- rank 1: input DMA geometry -----------------------------------------
    # A 16-byte piece covers 8 columns: lane l -> column c0 + l / 8, slot l % 8,
    # logical granule (l % 8) ^ ((c0 + l / 8) % 8) = (l % 8) ^ (l / 8).
    # A 4-byte piece covers 2 columns: lane l -> column c0 + l / 32, element
    # (l % 32) * 2 of the column, slot (l % 32) / 4, logical granule
    # slot ^ ((c0 + l / 32) % 8).
    if "l_row" in ctx.shared:
        l_row, l_gran = ctx.shared["l_row"], ctx.shared["l_gran"]
    else:
        l_row = b.div(lane, ci(8))
        l_gran = b.xor(b.mod(lane, ci(8)), l_row)
    lane_c: Optional[Value] = ctx.shared.get("lane_c")
    ih0 = b.sub(oh0, ci(PAD))
    iw0 = b.sub(ow0, ci(PAD))
    lane_geo = {}
    for c_first, dw in sorted({(g[0][1], g[0][2]) for g in in_groups}):
        if dw == 4:
            col = b.add(ci(c_first), l_row)
            ch = lane_c if lane_c is not None else b.mul(l_gran, ci(GRAN))
        else:
            col = b.add(ci(c_first), b.div(lane, ci(32)))
            e = b.mul(b.mod(lane, ci(32)), c2)
            gran = b.xor(b.div(e, ci(GRAN)), b.mod(col, ci(8)))
            ch = b.add(b.mul(gran, ci(GRAN)), b.mod(e, ci(GRAN)))
        iw = b.add(iw0, col)
        ok = b.land(b.cmp_ge(iw, c0), b.cmp_lt(iw, p_Wi))
        lane_geo[(c_first, dw)] = (b.add(b.mul(iw, sa_w), ch), ch, ok)

    def issue_input_group(buf: int, gi: int, c_base: Value):
        """DMA input piece group ``gi`` of the chunk at ``c_base`` into ``buf``."""
        r_first, c_first, dw = in_groups[gi][0]
        off, ch, ok = lane_geo[(c_first, dw)]
        row = b.add(ci(r_first), wave_g)
        ih = b.add(ih0, row)
        row_ok = b.land(
            b.cmp_lt(row, ci(IN_H)), b.land(b.cmp_ge(ih, c0), b.cmp_lt(ih, p_Hi))
        )
        c_ok = b.cmp_lt(b.add(c_base, ch), C)
        g_off = b.add(b.add(off, b.mul(ih, sa_h)), c_base)
        voff = b.select(b.land(b.land(ok, row_ok), c_ok), b.mul(g_off, c2), oob)
        lds_off = b.add(b.mul(row, ci(ROW_E * 2)), ci(c_first * CK * 2))
        b.async_buffer_load_lds(a_rsrc, lds_at(in_addr[buf], lds_off), voff, c0, dw)

    # ---- fragment reads -----------------------------------------------------
    weights.a_geometry(ctx)
    lane_hi = ctx.shared["lane_hi"]
    # B (input): pixel row h = wave_j * 8 + jb, column w = lane % 16; tap (r, s)
    # reads staged row h + r, column w + s, slot granule ^ ((w + s) % 8).
    l16 = ctx.shared["l16"] if "l16" in ctx.shared else b.mod(lane, ci(16))
    b_lane = {}
    for s in range(KW):
        col = b.add(l16, ci(s))
        for part in range(PARTS_C):
            g = b.add(ci(part * 4), lane_hi)
            slot = b.xor(g, b.mod(col, ci(8)))
            b_lane[(s, part)] = b.add(b.mul(col, ci(CK)), b.mul(slot, ci(GRAN)))
    row0 = b.mul(wave_j, ci(8 * ROW_E))

    def read_b(buf: int, jb: int, r: int, s: int, part: int) -> Value:
        idx = b.add(b_lane[(s, part)], b.add(row0, ci((jb + r) * ROW_E)))
        return b.smem_load_vN(in_bufs[buf], c0, idx, dtype=io_type, n=GRAN)

    # ---- one chunk ----------------------------------------------------------
    n_acc = KF * 8

    def chunk(accs: List[Value], par: int, c_base: Value, my_rank: int) -> List[Value]:
        """One 64-channel chunk for wave rank ``my_rank``; ``par`` = chunk parity.

        Input buffer ``par`` holds this chunk; weight buffer of tap t is
        ``(par * TAPS + t) % 2`` (taps alternate buffers across chunks).
        """
        accs = list(accs)
        c_next = b.add(c_base, ci(CK))
        for step in range(STEPS):
            tap, part = divmod(step, PARTS_C)
            r, s = divmod(tap, KW)
            wbuf = (par * TAPS + tap) % 2
            a_fr = [weights.read_a(ctx, wbuf, kf, part) for kf in range(KF)]
            b_fr = [read_b(par, jb, r, s, part) for jb in range(8)]
            if my_rank == 0 and part == 0:
                # Next tap's weights (tap 0 of the next chunk after the last).
                if tap + 1 < TAPS:
                    weights.issue(ctx, 1 - wbuf, tap + 1, c_base)
                else:
                    weights.issue(ctx, 1 - wbuf, 0, c_next)
            if my_rank == 1:
                for gi in in_sched[step]:
                    issue_input_group(1 - par, gi, c_next)
                if step == STEPS - 1:
                    # The next chunk's input, before rank 0 reads it.
                    b.s_waitcnt(vmcnt=0)
            b.sched_barrier(0)
            b.s_barrier_bare()
            b.sched_barrier(0)
            b.s_setprio(1)
            order = range(8) if step % 2 == 0 else range(7, -1, -1)
            for kf in range(KF):
                for jb in order:
                    i = kf * 8 + jb
                    accs[i] = _mfma(b, dtype, "16x16x32", a_fr[kf], b_fr[jb], accs[i])
            b.s_setprio(0)
            b.sched_barrier(0)
            if my_rank == 0 and part == PARTS_C - 1:
                # The next tap's weights, before either rank reads them.
                b.s_waitcnt(vmcnt=0)
            b.s_barrier_bare()
            b.sched_barrier(0)
        return accs

    n_chunks = b.div(b.add(C, ci(CK - 1)), ci(CK))
    n_pairs = b.div(n_chunks, c2)
    zero = b.zero_vec_f32(4)
    acc_t = VectorType(F32, 4)

    def rank_body(my_rank: int) -> List[Value]:
        # Prologue: tap 0 weights (rank 0) and chunk 0's input (rank 1).
        if my_rank == 0:
            weights.issue(ctx, 0, 0, c0)
        else:
            for gi in range(len(in_groups)):
                issue_input_group(0, gi, c0)
        b.s_waitcnt(vmcnt=0)
        b.s_barrier_bare()
        if my_rank == 1:
            b.s_barrier_bare()  # rank 1 runs one barrier behind
        b.sched_barrier(0)
        loop = b.scf_for_iter(
            c0,
            n_pairs,
            c1,
            [(f"acc{my_rank}_{i}", zero) for i in range(n_acc)],
            iv_name=f"pair{my_rank}",
            elide_trailing_barrier=False,
        )
        with loop as (pi, carried):
            c_even = b.mul(pi, ci(2 * CK))
            accs = chunk(list(carried), 0, c_even, my_rank)
            accs = chunk(accs, 1, b.add(c_even, ci(CK)), my_rank)
            b.scf_yield(*accs)
        accs = list(loop.results)
        odd = b.cmp_eq(b.mod(n_chunks, c2), c1)
        tail = b.scf_if_else(odd, result_types=[acc_t] * n_acc)
        with tail as (then_ctx, else_ctx):
            with then_ctx:
                b.scf_yield(*chunk(accs, 0, b.mul(n_pairs, ci(2 * CK)), my_rank))
            with else_ctx:
                b.scf_yield(*accs)
        accs = list(tail.results)
        if my_rank == 0:
            b.s_barrier_bare()  # pair rank 1's last barrier
        return accs

    # ``rank`` is wave-uniform (readfirstlane), so this is a branch per wave,
    # not per lane. s_barrier counts waves, so the two rank schedules only
    # need the same barrier count: 2 per step in both, plus prologue / epilogue
    # barriers that add to 2 per rank (pinned by
    # test_rank_branches_hit_the_same_barriers).
    branch = b.scf_if_else(b.cmp_eq(rank, c0), result_types=[acc_t] * n_acc)
    with branch as (then_ctx, else_ctx):
        with then_ctx:
            b.scf_yield(*rank_body(0))
        with else_ctx:
            b.scf_yield(*rank_body(1))
    accs = list(branch.results)

    # ---- epilogue: lane holds channels lane_hi*4 .. +3 of pixel (h, l16) ----
    k_lane = b.add(b.add(k0, b.mul(wave_k, ci(spec.wave_k))), b.mul(lane_hi, ci(4)))
    ow = b.add(ow0, l16)
    w_ok = b.cmp_lt(ow, p_Wo)
    for jb in range(8):
        oh = b.add(oh0, b.add(b.mul(wave_j, ci(8)), ci(jb)))
        hw_ok = b.land(w_ok, b.cmp_lt(oh, p_Ho))
        pix = b.add(b.mul(oh, sd_h), b.mul(ow, sd_w))
        for kf in range(KF):
            k = b.add(k_lane, ci(kf * ATOM))
            ok = b.land(hw_ok, b.cmp_lt(k, K))
            voff = b.select(ok, b.mul(b.add(pix, k), c2), oob)
            val = _trunc_f32(b, dtype, accs[kf * 8 + jb])
            _buf_store_vN(b, dtype, d_rsrc, voff, c0, val, 2)
    return b.kernel
