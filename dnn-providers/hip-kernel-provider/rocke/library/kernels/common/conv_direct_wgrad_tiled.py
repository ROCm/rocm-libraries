# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Direct wgrad with multi-atom wave tiles and an LDS-DMA row pipeline.

Computes the weight gradient of a stride-1 NHWC convolution::

    dW[k, r, s, c] = sum_{n, ho, wo} dY[n, ho, wo, k] * X[n, ho + r - PAD, wo + s - PAD, c]

The row loop walks INPUT rows ``hi`` of one 32-column strip. Input row ``hi``
pairs with output rows ``hi + PAD - r`` for every filter row ``r``, so each dY
row is read out of LDS once and kept in a ``KH``-deep register ring, and one X
row strip of ``32 + KW - 1`` columns serves every filter column ``s`` at a
one-column shift.

What sets it apart from :func:`build_direct_conv_wgrad` (16x16 per wave):

* A wave owns ``wave_k16*16 x wave_c16*16`` channels for all ``KH*KW`` taps
  (``32 x 32 x 9`` = 144 fp32 accumulator registers at 3x3), so one dY
  fragment and one X fragment feed ``wave_c16`` resp. ``wave_k16`` MFMAs.
* The rows are workgroup-shared: every wave DMAs one 1 KiB slice of each row
  straight into LDS (``buffer_load ... lds``), with ``prefetch`` rows in
  flight, so no VGPR staging and one barrier per row.
* A 16-byte channel chunk lands at ``chunk ^ swizzle(row)``; the swizzle is
  chosen at build time so the transpose reads are bank-conflict free.
* A persistent grid: ``ceil(target_blocks / channel_tiles)`` blocks walk the
  (image, column strip, row segment) list with a stride, accumulate in
  registers across segments, and atomically add once at the end.
* ``rows_per_seg = 0`` picks the launch from the runtime shape: a segment is
  a whole strip (all H rows, rounded up to the unroll), and the grid is at
  most ``target_blocks`` blocks (``floor``, not ``ceil``), so it runs in one
  round when ``target_blocks`` is the number of blocks the device holds.

dW is fp32; the caller zero-initialises it.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import List, Optional, Tuple

from rocke.core.ir import I64, IRBuilder, KernelDef, Value
from rocke.helpers.spec import ceil_div

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    _io_type,
    _mfma,
    dilation_reason,
    emit_direct_params,
)

# MFMA 16x16x32: M = output channels (k), N = input channels (c), K = columns.
_ATOM = 16
_WO = 32
_VEC = 8  # 16-bit elements per 16-byte chunk
_WAVE = 64
# Bank model used to pick the swizzle: 64 four-byte banks, a 32-lane half of
# a ds_read_b64 serviced per pass.
_BANKS = 64
_HALF = 32


@dataclass(frozen=True)
class DirectWgradTiledSpec:
    problem: DirectConvProblem
    name: str = "direct_wgrad_tiled"
    wave_k16: int = 2
    wave_c16: int = 2
    waves_k: int = 4
    waves_c: int = 2
    # Input rows per segment (a multiple of the unroll); 0 = the whole strip
    # with a one-round grid, both derived from the runtime shape.
    rows_per_seg: int = 30
    prefetch: int = 2
    target_blocks: int = 256
    # Main-loop unroll override (a multiple of n_buffers); 0 = default.
    row_unroll: int = 0
    # Images packed into one 32-column strip: the MFMA reduction runs over
    # unfold_n images x 32 / unfold_n columns, so a narrow image (W = 40, say)
    # does not leave most of a 32-wide strip empty. Above 4 (up to 32: one
    # column per image) only for KW == 1, e.g. a 1xN layer run as its Nx1
    # exchanged view (H = 1: the same bytes with H and W swapped).
    unfold_n: int = 1
    # Two wave groups per block, each streaming its own segments of the same
    # channel tile; their partial gradients meet in LDS before the atomics,
    # which halves the atomic count. With ``pingpong`` the second group runs
    # half a row behind the first (one extra barrier at the start), so one
    # group's MFMA phase covers the other's DMA issue and fragment reads.
    # With ``q_groups == 1`` the two ping-pong groups are the two halves of
    # the block's waves, which share one row stream. ``q_groups == 4``
    # runs four segments at once, one per wave group (the group index is the
    # wave index's fastest digit), and ping-pongs the halves of every group:
    # the first half of the block's waves against the second, so each SIMD
    # holds one wave of either half. Its partial gradients meet in LDS as with
    # two groups.
    q_groups: int = 1
    pingpong: bool = False
    # Ping-pong in halves: no extra row buffer; each half waits for its
    # own LDS reads at the barrier that ends its read phase instead, so a
    # buffer is free again one barrier after its last read.
    pp_tight: bool = False
    # Auto launch only: run the strip's H rows as whole unrolled blocks
    # plus a tail of nested row tests, instead of rounding H up to the
    # unroll (a 15-row unroll on a 32-row strip runs 13 empty rows).
    row_tail: bool = False
    # Small images: one block per channel tile walks every segment, so
    # nothing else adds into its dW tile and the epilogue stores instead of
    # adding atomically (dW-bound layers: big C x K, few pixels).
    one_split: bool = False
    # one_split only: store dW in the activation dtype, so the call is
    # this one launch (no fp32 dW to zero and cast afterwards).
    dw16: bool = False
    # Atomic tiles: add packed 16-bit pairs (adjacent c, lane pairs
    # joined by a DPP swap) into the 16-bit dW: the call is a zero-fill and
    # this launch, with half the atomics and no cast.
    pk16: bool = False
    # Grouped, small groups: every wave owns a whole group (kpg ==
    # wave_k, cpg == wave_c), so a block covers n_waves consecutive groups,
    # whose dY / X channels are one contiguous run each.
    gpack: bool = False
    # Atomic tiles: block ``split`` stores (no atomics) its fp32 tile
    # into slab ``split`` of a ``[n_splits][dW]`` scratch, which the slab fold
    # (WgradSlabFoldSpec) sums into the 16-bit dW: the call is this launch and
    # the fold, with no zero-fill and no cast.
    slab: bool = False
    # L64: each segment addresses dY / X through descriptors rebased
    # (64-bit) to its first image, so one launch covers a batch whose
    # operands pass the 2^31-byte buffer range (A_bytes / B_bytes unused).
    l64: bool = False
    # Ping-pong: schedule fences keep the next row's DMA setup (row
    # masks, addresses) off the MFMA phase's tail and inside the memory
    # phase, which issues its fragment reads before the DMAs; a row's DMA
    # addresses are all formed before its first DMA, so no address register
    # is rewritten while an earlier DMA still reads it.
    pp_sched: bool = False

    @property
    def wave_k(self) -> int:
        return _ATOM * self.wave_k16

    @property
    def wave_c(self) -> int:
        return _ATOM * self.wave_c16

    @property
    def block_k(self) -> int:
        return self.wave_k * (self.n_waves if self.gpack else self.waves_k)

    @property
    def block_c(self) -> int:
        return self.wave_c * (self.n_waves if self.gpack else self.waves_c)

    @property
    def n_waves(self) -> int:
        return self.waves_k * self.waves_c

    @property
    def threads_per_block(self) -> int:
        return self.n_waves * self.q_groups * _WAVE

    @property
    def strip_w(self) -> int:
        """Output columns of one image in a strip."""
        return _WO // self.unfold_n

    @property
    def pingpong_halves(self) -> bool:
        """Single-group ping-pong: the two groups are the block's wave halves."""
        return self.pingpong and self.q_groups in (1, 4)

    @property
    def n_buffers(self) -> int:
        # The q_groups == 1 ping-pong keeps one more row buffer: a row's
        # buffer is then refilled a full step after its reads were issued,
        # so the reads need not complete before the mid barrier.
        if self.pingpong_halves and not self.pp_tight:
            return self.prefetch + 2
        return self.prefetch + 1

    @property
    def unroll(self) -> int:
        # The LDS buffer index has to fold to a constant, so the unroll is a
        # multiple of n_buffers. The dY ring is SSA renaming and would work at
        # any unroll, but one that is not a multiple of KH rotates the ring
        # through the loop-carried values, and the copies cost registers.
        if self.row_unroll:
            return self.row_unroll
        return math.lcm(self.n_buffers, self.problem.KH)

    def acc_vgprs(self) -> int:
        p = self.problem
        return p.KH * p.KW * self.wave_k16 * self.wave_c16 * 4

    def dma_instrs(self) -> Tuple[int, int]:
        """1 KiB DMA instructions per wave and row: (dY, X)."""
        seg = self.strip_w + self.problem.KW - 1
        ik = ceil_div(ceil_div(_WO * self.block_k // _VEC, _WAVE), self.n_waves)
        ic = ceil_div(
            ceil_div(self.unfold_n * seg * self.block_c // _VEC, _WAVE), self.n_waves
        )
        return ik, ic

    def lds_bytes(self) -> int:
        """The padded dY / X ring buffers, plus the q-group reduction buffers."""
        ik, ic = self.dma_instrs()
        slot = self.q_groups * self.n_waves * (ik + ic) * _WAVE * _VEC * 2
        red = 0
        if self.q_groups > 1:
            red = self.red_buffers() * self.red_slices() * self.red_slice_bytes()
        return self.n_buffers * slot + red

    def red_slice_bytes(self) -> int:
        """One wave group's accumulators of one tap."""
        return self.n_waves * self.wave_k16 * self.wave_c16 * _WAVE * 16

    def red_slices(self) -> int:
        # Two groups: the sender's slice. Four: every group's, each group
        # then sums a quarter of the tap's atoms.
        return 1 if self.q_groups == 2 else self.q_groups

    def red_buffers(self) -> int:
        """LDS buffers of the q-group reduction: two (one barrier per tap)
        where they fit, else one (two barriers per tap)."""
        if self.q_groups == 2:
            return 2
        ik, ic = self.dma_instrs()
        rings = (
            self.n_buffers * self.q_groups * self.n_waves * (ik + ic) * _WAVE * _VEC * 2
        )
        one = self.red_slices() * self.red_slice_bytes()
        return 2 if rings + 2 * one <= 160 * 1024 else 1

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        flags = {"bf16": True} if p.dtype == "bf16" else {}
        if self.row_unroll:
            flags[f"u{self.row_unroll}"] = True
        if self.unfold_n > 1:
            flags[f"un{self.unfold_n}"] = True
        if self.q_groups > 1:
            flags[f"qg{self.q_groups}"] = True
        if self.pingpong:
            flags["pp"] = True
        if self.pp_tight:
            flags["ppt"] = True
        if self.row_tail:
            flags["rt"] = True
        if self.one_split:
            flags["os"] = True
        if self.dw16:
            flags["d16"] = True
        if self.pk16:
            flags["pk"] = True
        if self.gpack:
            flags["gp"] = True
        if self.slab:
            flags["sl"] = True
        if self.l64:
            flags["l64"] = True
        if self.pp_sched:
            flags["pps"] = True
        return kernel_name_join(
            self.name,
            f"f{p.KH}x{p.KW}p{p.PAD}" + (f"x{p.PAD_W}" if p.PAD_W != p.PAD else ""),
            f"wt{self.wave_k}x{self.wave_c}",
            f"w{self.waves_k}x{self.waves_c}",
            f"rps{self.rows_per_seg}" if self.rows_per_seg else "rpsH",
            f"pf{self.prefetch}",
            f"tb{self.target_blocks}",
            flags=flags,
        )

    # ---- host-side launch geometry (mirrors the in-kernel decode) ----
    def n_channel_tiles(self) -> int:
        p = self.problem
        if self.gpack:
            return ceil_div(p.groups, self.n_waves)
        return p.groups * ceil_div(p.kpg, self.block_k) * ceil_div(p.cpg, self.block_c)

    def n_segments(self) -> int:
        p = self.problem
        return (
            ceil_div(p.N, self.unfold_n)
            * ceil_div(p.Wo, self.strip_w)
            * (ceil_div(p.H, self.rows_per_seg) if self.rows_per_seg else 1)
        )

    def n_splits(self) -> int:
        if self.one_split:
            return 1
        tiles = self.n_channel_tiles()
        if self.rows_per_seg:
            per_tile = ceil_div(self.target_blocks, tiles)
        else:
            per_tile = max(1, self.target_blocks // tiles)
        return min(ceil_div(self.n_segments(), self.q_groups), per_tile)

    def launch_geometry(self):
        return (self.n_channel_tiles() * self.n_splits(), 1, 1), (
            self.threads_per_block,
            1,
            1,
        )


def is_valid_wgrad_tiled_spec(spec: DirectWgradTiledSpec, arch: str = "gfx950"):
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    p = spec.problem
    why = dilation_reason(p)
    if why is not None:
        return False, why
    if p.dtype not in ("fp16", "bf16"):
        return False, f"unsupported dtype {p.dtype!r}"
    if p.stride != 1:
        return False, f"stride-1 algorithm (got stride={p.stride})"
    if target.wave_size != 64 or not target.memory.has_ds_read_tr:
        return False, f"needs wave64 + ds_read_b64_tr_b16 (absent on {arch})"
    if target.async_lds_max_dwords < 4:
        return False, f"needs 16-byte buffer_load_lds (absent on {arch})"
    ab = "bf16" if p.dtype == "bf16" else "f16"
    if not target.mma.has_shape(
        a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=16, n=16, k=32
    ):
        return False, f"missing mfma_f32_16x16x32_{ab} on {arch}"
    if spec.wave_k16 not in (1, 2, 4) or spec.wave_c16 not in (1, 2, 4):
        return False, "wave_k16 / wave_c16 must be 1, 2 or 4"
    for n, v in (("waves_k", spec.waves_k), ("waves_c", spec.waves_c)):
        if v < 1 or v & (v - 1):
            return False, f"{n} must be a power of two"
    if spec.unfold_n not in (1, 2, 4, 8, 16, 32):
        return False, "unfold_n must be a power of two up to 32"
    if spec.unfold_n > 4 and p.KW != 1:
        # A lane's 8 reduction elements must be 8 consecutive X tile rows:
        # with fewer than 8 columns per image that holds only without a
        # column halo (KW == 1: the packed images' columns are contiguous).
        return False, "unfold_n above 4 needs KW == 1"
    if spec.q_groups not in (1, 2, 4):
        return False, "q_groups must be 1, 2 or 4"
    if spec.pp_sched and not spec.pingpong:
        return False, "pp_sched needs the ping-pong"
    if spec.pp_tight and not spec.pingpong_halves:
        return False, "pp_tight needs the ping-pong in halves"
    if spec.pingpong_halves and spec.n_waves % 2:
        return False, "pingpong in halves needs an even wave count per group"
    if spec.threads_per_block > target.max_threads_per_block:
        return False, "too many threads per block"
    if p.cpg % _VEC or p.kpg % 4:
        # dY's 16-byte chunks need only 8-byte alignment (kpg % 4): a chunk
        # that runs past kpg reads the next pixel's channels into dW rows
        # the epilogue does not store.
        return False, "cpg must be a multiple of 8 and kpg of 4"
    if spec.dw16 and not spec.one_split:
        return False, "dw16 stores dW whole: needs one_split"
    if spec.pk16 and p.dtype != "fp16":
        return False, "pk16: bf16 pair atomics lose too much precision"
    if spec.pk16 and spec.one_split:
        return False, "pk16 adds atomically: not with one_split"
    if spec.gpack and (p.groups < 2 or p.kpg != spec.wave_k or p.cpg != spec.wave_c):
        return False, "gpack needs groups > 1 with kpg == wave_k and cpg == wave_c"
    if p.kpg < spec.wave_k or p.cpg < spec.wave_c:
        return False, "wave tile wider than the group's channels"
    if spec.acc_vgprs() > 192:
        return False, f"accumulators need {spec.acc_vgprs()} VGPRs (> 192)"
    if spec.slab and (spec.one_split or spec.dw16 or spec.pk16):
        return False, "slab stores fp32 partials: not with one_split / dw16 / pk16"
    if spec.row_tail and spec.rows_per_seg:
        return False, "row_tail needs the auto launch (rows_per_seg 0)"
    if (
        spec.l64
        and spec.unfold_n * 2 * max(p.Ho * p.Wo * p.total_k, p.H * p.W * p.total_c)
        >= (1 << 31) - 1
    ):
        return False, "l64: a segment's images exceed the i32 buffer range"
    if spec.row_unroll < 0:
        return False, "row_unroll must be >= 0"
    if spec.target_blocks < 1:
        return False, "target_blocks must be >= 1"
    # Before the n_buffers check: prefetch -1 (-2 with the single-group
    # ping-pong's extra buffer) makes n_buffers 0.
    if spec.prefetch < 1:
        return False, "prefetch must be >= 1"
    if spec.row_unroll % spec.n_buffers:
        return False, "row_unroll must be a multiple of n_buffers"
    if spec.rows_per_seg < 0 or spec.rows_per_seg % spec.unroll:
        return False, f"rows_per_seg must be 0 or a positive multiple of {spec.unroll}"
    if p.KH > 8 or p.KW > 8:
        return False, "filter extent above 8"
    if spec.lds_bytes() > 160 * 1024:
        return False, f"LDS {spec.lds_bytes()} B > 160 KiB"
    return True, "ok"


# ---------------------------------------------------------------------------
# LDS swizzle
# ---------------------------------------------------------------------------


def _frag_row(lane: int, rd: int, shift: int, wu: int, seg: int) -> int:
    """LDS row a ``ds_read_b64_tr_b16`` lane reads for reduction index block rd.

    Lane ``l`` supplies MFMA reduction elements ``(l // 16) * 8 + 0..7``; with
    ``unfold_n`` images in a strip those are columns of image ``kk // wu``,
    whose rows start at ``image * seg`` in the tile.
    """
    kk0 = (lane // 16) * 8
    return (kk0 // wu) * seg + kk0 % wu + rd * 4 + (lane // 4) % 4 + shift


def _swizzle_ok(swz, chunks: int, rows_of, col_origins) -> bool:
    """No two lanes of a 32-lane half hit one bank, for every read the kernel makes."""
    for row_of in rows_of:
        for origin in col_origins:
            for rd in (0, 1):
                for half in (0, 1):
                    banks = set()
                    for lane in range(half * _HALF, (half + 1) * _HALF):
                        row = row_of(lane, rd)
                        col = origin + (lane % 4) * 4
                        phys = (col // _VEC) ^ swz(row)
                        byte = (row * chunks + phys) * 16 + (col % _VEC) * 2
                        for w in range(2):
                            banks.add((byte // 4 + w) % _BANKS)
                    if len(banks) != 2 * _HALF:
                        return False
    return True


@lru_cache(maxsize=None)
def pick_swizzle(chunks: int, shifts: Tuple[int, ...], wu: int, seg: int, col_origins):
    """``bits``: chunk XOR ``2 * sum(((row >> bits[j]) & 1) << j)``.

    The XOR moves 32-byte chunk PAIRS so a lane's 8-byte half-chunk read stays
    inside its chunk. Returns the first bit selection that makes every read
    conflict-free, or no swizzle when none does.
    """
    rows_of = [
        (lambda lane, rd, sh=sh: _frag_row(lane, rd, sh, wu, seg)) for sh in shifts
    ]
    n_pairs = chunks // 2
    width = int(math.log2(n_pairs)) if n_pairs > 1 else 0

    def make(bits):
        def swz(row):
            v = 0
            for j, bit in enumerate(bits):
                v |= ((row >> bit) & 1) << j
            return 2 * v

        return swz

    if _swizzle_ok(make(()), chunks, rows_of, col_origins):
        return ()
    for nbits in range(1, width + 1):
        for bits in itertools.permutations(range(6), nbits):
            if _swizzle_ok(make(bits), chunks, rows_of, col_origins):
                return bits
    return ()


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_direct_wgrad_tiled(
    spec: DirectWgradTiledSpec, arch: str = "gfx950"
) -> KernelDef:
    """Build the tiled direct wgrad kernel (see the module docstring).

    Grid: ``n_channel_tiles * n_splits`` blocks (``launch_geometry``); block
    ``split`` walks segments ``split, split + n_splits, ...`` of the
    (image group, column strip, row block) list. A segment streams
    ``rows_per_seg`` input rows of one strip through the row pipeline:

      step t (input row hi = h0 - (KH - 1) + t):
        wait for row t's DMA, barrier, DMA row t + prefetch into the free
        buffer, transpose-read the dY fragment into the ring and the KW X
        fragments, then KH * KW * wave_k16 * wave_c16 MFMAs.

    The first KH - 1 steps of a segment only prime the dY ring.
    """
    ok, why = is_valid_wgrad_tiled_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid DirectWgradTiledSpec for {arch}: {why}")

    p = spec.problem
    KH, KW, PAD = p.KH, p.KW, p.PAD
    io_dtype = p.dtype
    io_type = _io_type(io_dtype)
    NW = spec.n_waves
    WK16, WC16 = spec.wave_k16, spec.wave_c16
    BK, BC = spec.block_k, spec.block_c
    CHK, CHC = BK // _VEC, BC // _VEC
    UN = spec.unfold_n
    WU = spec.strip_w
    SEG = WU + KW - 1  # X tile rows (columns of the image) per packed image
    STRIP = UN * SEG
    D = spec.prefetch
    NB = spec.n_buffers
    HPB = spec.rows_per_seg
    AUTO = HPB == 0
    U = spec.unroll
    T_PRO = KH - 1

    # 1 KiB DMA instructions per wave and row, per operand. Slots past the
    # tile load zero (OOB sentinel) into the buffer's pad.
    IK, IC = spec.dma_instrs()
    N_INSTR = IK + IC
    QG = spec.q_groups
    # A ring buffer holds one slice per wave group, group g's at g * G*_ELEMS.
    GDY_ELEMS = NW * IK * _WAVE * _VEC
    GX_ELEMS = NW * IC * _WAVE * _VEC
    DY_ELEMS = QG * GDY_ELEMS
    X_ELEMS = QG * GX_ELEMS

    GP = spec.gpack
    k_origins = tuple(
        wk * spec.wave_k + mi * _ATOM
        for wk in range(NW if GP else spec.waves_k)
        for mi in range(WK16)
    )
    c_origins = tuple(
        wc * spec.wave_c + ni * _ATOM
        for wc in range(NW if GP else spec.waves_c)
        for ni in range(WC16)
    )
    swz_k = pick_swizzle(CHK, (0,), WU, WU, k_origins)
    swz_c = pick_swizzle(CHC, tuple(range(KW)), WU, SEG, c_origins)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads_per_block
    # Separate scopes per row buffer, or the waitcnt pass drains the DMA ring.
    b.kernel.attrs["lds_alias_scopes"] = True
    # Two waves per SIMD: the register budget the 32x32 tile is sized for.
    b.kernel.attrs["waves_per_eu"] = (2, 8)
    params = emit_direct_params(
        b, direction="wgrad", io_type=io_type, d16=spec.dw16, d16_atomic=spec.pk16
    )

    def ci(v: int) -> Value:
        return b.const_i32(v)

    c0, c1, c_two = ci(0), ci(1), ci(2)
    oob = ci((1 << 31) - 1)

    p_N = params["p_N"]
    p_H = params["p_Hi"]
    p_W = params["p_Wi"]
    p_Ho = params["p_Ho"]
    p_Wo = params["p_Wo"]
    p_groups = params["p_groups"]
    c_cpg = b.div(params["p_total_c"], p_groups)
    c_kpg = b.div(params["p_total_k"], p_groups)

    tid = b.thread_id_x()
    lane = b.mod(tid, ci(_WAVE))
    wave_all = b.readfirstlane(b.div(tid, ci(_WAVE)))
    if QG > 2:
        # Group fastest: wave w's group is w % QG, so the block's first half
        # (the ping waves) holds wave 0 of every group.
        qg = b.mod(wave_all, ci(QG))
        wave = b.div(wave_all, ci(QG))
    elif QG > 1:
        qg = b.div(wave_all, ci(NW))
        wave = b.mod(wave_all, ci(NW))
    else:
        qg, wave = c0, wave_all

    # ---- grid decode --------------------------------------------------------
    if GP:
        n_ch_tiles = b.div(b.add(p_groups, ci(NW - 1)), ci(NW))
    else:
        n_k_tiles = b.div(b.add(c_kpg, ci(BK - 1)), ci(BK))
        n_c_tiles = b.div(b.add(c_cpg, ci(BC - 1)), ci(BC))
        n_ch_tiles = b.mul(b.mul(p_groups, n_k_tiles), n_c_tiles)
    n_q = b.div(b.add(p_Wo, ci(WU - 1)), ci(WU))
    n_ng = b.div(b.add(p_N, ci(UN - 1)), ci(UN))
    if AUTO:
        # One segment per strip: all H rows, the row loop's trip count
        # rounded up to the unroll.
        if spec.row_tail:
            n_rb = b.div(p_H, ci(U))
            n_tail = b.sub(p_H, b.mul(n_rb, ci(U)))
        else:
            n_rb = b.div(b.add(p_H, ci(U - 1)), ci(U))
        total_segs = b.mul(n_ng, n_q)
    else:
        n_hb = b.div(b.add(p_H, ci(HPB - 1)), ci(HPB))
        total_segs = b.mul(b.mul(n_ng, n_q), n_hb)
    # Same formula as DirectWgradTiledSpec.n_splits, which sizes the grid.
    seg_splits = b.div(b.add(total_segs, ci(spec.q_groups - 1)), ci(spec.q_groups))
    if AUTO:
        per_tile = b.smax(b.div(ci(spec.target_blocks), n_ch_tiles), c1)
    else:
        per_tile = b.div(
            b.add(ci(spec.target_blocks), b.sub(n_ch_tiles, c1)), n_ch_tiles
        )
    n_splits = c1 if spec.one_split else b.smin(seg_splits, per_tile)

    # Dispatch deals block ids round-robin over 8 XCDs; renumber so each XCD
    # gets a contiguous run, i.e. the channel tiles of the same segments,
    # which then share that XCD's L2.
    lin = b.block_id_x()
    n_blocks = b.mul(n_ch_tiles, n_splits)
    xcd = b.mod(lin, ci(8))
    pos = b.div(lin, ci(8))
    per = b.div(n_blocks, ci(8))
    rem = b.mod(n_blocks, ci(8))
    lin = b.add(b.add(b.mul(xcd, per), b.smin(xcd, rem)), pos)
    ch_tile = b.mod(lin, n_ch_tiles)
    split = b.div(lin, n_ch_tiles)
    if GP:
        # Block = NW consecutive groups; wave w owns group gb * NW + w.
        g_blk0 = b.mul(ch_tile, ci(NW))
        k_blk0 = b.mul(g_blk0, c_kpg)  # global k
        c_glb0 = b.mul(g_blk0, c_cpg)
        c_blk0 = c0
        group = b.add(g_blk0, wave)
        wave_k_id = wave
        wave_c_id = wave
    else:
        c_tile = b.mod(ch_tile, n_c_tiles)
        gk = b.div(ch_tile, n_c_tiles)
        k_tile = b.mod(gk, n_k_tiles)
        group = b.div(gk, n_k_tiles)
        k_blk0 = b.add(b.mul(group, c_kpg), b.mul(k_tile, ci(BK)))  # global k
        c_blk0 = b.mul(c_tile, ci(BC))  # in-group c
        c_glb0 = b.add(b.mul(group, c_cpg), c_blk0)

        wave_k_id = b.mod(wave, ci(spec.waves_k))
        wave_c_id = b.div(wave, ci(spec.waves_k))

    L64 = spec.l64
    PPS = spec.pp_sched
    if not L64:
        a_rsrc = b.buffer_rsrc(params["A"], params["A_bytes"])
        b_rsrc = b.buffer_rsrc(params["B"], params["B_bytes"])

    # ---- LDS: one allocation per ring buffer (one alias scope each) ---------
    dy_bufs = [
        b.smem_alloc(io_type, [1, DY_ELEMS], name_hint=f"dy{i}") for i in range(NB)
    ]
    x_bufs = [
        b.smem_alloc(io_type, [1, X_ELEMS], name_hint=f"xs{i}") for i in range(NB)
    ]
    dy_addr = [b.smem_addr_of(s) for s in dy_bufs]
    x_addr = [b.smem_addr_of(s) for s in x_bufs]

    if QG > 1:
        qg_dy_bytes = b.zext(b.mul(qg, ci(GDY_ELEMS * 2)), I64)
        qg_x_bytes = b.zext(b.mul(qg, ci(GX_ELEMS * 2)), I64)

    def swz_val(row: Value, bits) -> Optional[Value]:
        v = None
        for j, bit in enumerate(bits):
            term = b.shl(b.land(b.lshr(row, ci(bit)), c1), ci(j + 1))
            v = term if v is None else b.lor(v, term)
        return v

    # ---- DMA lane geometry (segment invariant part) -------------------------
    # Slot L = (wave * I + i) * 64 + lane of a 16-byte chunk; tile row = L / CH,
    # physical chunk = L % CH, logical chunk = physical ^ swizzle(row). A tile
    # row is (packed image, column) = divmod(row, rows per image).
    def dma_lanes(n_instr: int, chunks: int, rows: int, per_img: int, bits):
        out = []
        for i in range(n_instr):
            slot = b.add(b.mul(b.add(b.mul(wave, ci(n_instr)), ci(i)), ci(_WAVE)), lane)
            row = b.div(slot, ci(chunks))
            phys = b.mod(slot, ci(chunks))
            sw = swz_val(row, bits)
            log = phys if sw is None else b.xor(phys, sw)
            img = b.div(row, ci(per_img)) if UN > 1 else c0
            col = b.mod(row, ci(per_img)) if UN > 1 else row
            out.append((img, col, b.mul(log, ci(_VEC)), b.cmp_lt(row, ci(rows))))
        return out

    dy_lanes = dma_lanes(IK, CHK, _WO, WU, swz_k)
    x_lanes = dma_lanes(IC, CHC, STRIP, SEG, swz_c)

    sdy_n = params["p_dY_stride_n"]
    sdy_h = params["p_dY_stride_ho"]
    sdy_w = params["p_dY_stride_wo"]
    sx_n = params["p_X_stride_n"]
    sx_h = params["p_X_stride_hi"]
    sx_w = params["p_X_stride_wi"]

    def seg_rsrcs(n0: Value):
        """L64: dY / X descriptors of the segment's images n0 .. n0 + UN - 1
        (a 64-bit rebased base; images past N are masked by the lanes)."""
        n0 = b.readfirstlane(n0)
        out = []
        for ptr, sn in ((params["A"], sdy_n), (params["B"], sx_n)):
            img_b = b.mul(sn, c_two)
            base = b.global_ptr_offset(ptr, b.mul(b.zext(n0, I64), b.zext(img_b, I64)))
            out.append(b.buffer_rsrc(base, b.mul(ci(UN), img_b)))
        return out

    def seg_lane_offsets(n0: Value, q0: Value):
        """Per-instruction (lane offset in elements, lane valid) for one strip."""
        dy = []
        for img, col, ch, okr in dy_lanes:
            n_i = b.add(n0, img)
            wo = b.add(q0, col)
            ok = b.land(okr, b.land(b.cmp_lt(wo, p_Wo), b.cmp_lt(n_i, p_N)))
            off = b.add(
                b.add(b.mul(img if L64 else n_i, sdy_n), b.mul(wo, sdy_w)),
                b.add(k_blk0, ch),
            )
            dy.append((off, ok))
        xs = []
        for img, col, ch, okr in x_lanes:
            n_i = b.add(n0, img)
            wi = b.add(q0, b.sub(col, ci(p.PAD_W)))
            ok = b.land(
                okr,
                b.land(b.land(b.cmp_ge(wi, c0), b.cmp_lt(wi, p_W)), b.cmp_lt(n_i, p_N)),
            )
            off = b.add(
                b.add(b.mul(img if L64 else n_i, sx_n), b.mul(wi, sx_w)),
                b.add(c_glb0, ch),
            )
            xs.append((off, ok))
        if L64:
            return dy, xs, *seg_rsrcs(n0)
        return dy, xs, a_rsrc, b_rsrc

    def issue_row(buf: int, t: Value, h0: Value, h_end: Value, lane_offs, t_end: Value):
        """DMA row ``t`` of the segment into buffer ``buf``."""
        dy_offs, x_offs, a_rsrc, b_rsrc = lane_offs
        hi = b.add(h0, b.sub(t, ci(T_PRO)))
        ho = b.add(hi, ci(PAD))
        live = b.cmp_lt(t, t_end)
        dy_ok = b.land(live, b.land(b.cmp_ge(ho, c0), b.cmp_lt(ho, p_Ho)))
        x_ok = b.land(
            live,
            b.land(
                b.cmp_ge(t, ci(T_PRO)), b.land(b.cmp_ge(hi, c0), b.cmp_lt(hi, h_end))
            ),
        )
        dy_row = b.mul(ho, sdy_h)
        x_row = b.mul(hi, sx_h)
        # pp_sched forms every DMA address of the row before the first DMA;
        # otherwise each DMA issues as soon as its address is formed.
        loads = []

        def dma(rsrc, lds, voff):
            if PPS:
                loads.append((rsrc, lds, voff))
            else:
                b.async_buffer_load_lds(rsrc, lds, voff, c0, 4)

        for i, (off, ok) in enumerate(dy_offs):
            voff = b.select(b.land(ok, dy_ok), b.mul(b.add(off, dy_row), c_two), oob)
            lds = b.smem_ptr_add(
                dy_addr[buf] if QG == 1 else b.smem_ptr_add(dy_addr[buf], qg_dy_bytes),
                b.zext(b.mul(b.add(b.mul(wave, ci(IK)), ci(i)), ci(_WAVE * 16)), I64),
            )
            dma(a_rsrc, lds, voff)
        for i, (off, ok) in enumerate(x_offs):
            voff = b.select(b.land(ok, x_ok), b.mul(b.add(off, x_row), c_two), oob)
            lds = b.smem_ptr_add(
                x_addr[buf] if QG == 1 else b.smem_ptr_add(x_addr[buf], qg_x_bytes),
                b.zext(b.mul(b.add(b.mul(wave, ci(IC)), ci(i)), ci(_WAVE * 16)), I64),
            )
            dma(b_rsrc, lds, voff)
        if PPS:
            # Every address in its own register before the first DMA.
            b.sched_barrier(0)
            for rsrc, lds, voff in loads:
                b.async_buffer_load_lds(rsrc, lds, voff, c0, 4)

    # ---- fragment read addresses (element index into a buffer) -------------
    # See _frag_row: a lane's 8 reduction elements are 8 columns of one image.
    lane_hi = b.div(lane, ci(16))
    lane_q = b.mod(b.div(lane, ci(4)), ci(4))
    lane_c4 = b.mod(lane, ci(4))
    kk0 = b.mul(lane_hi, ci(8))
    if UN > 1:
        img_l = b.div(kk0, ci(WU))
        jl = b.mod(kk0, ci(WU))
    else:
        img_l, jl = c0, kk0

    def frag_index(
        shift: int, rd: int, col0: Value, chunks: int, bits, seg: int, g_elems: int
    ) -> Value:
        row = b.add(b.add(b.mul(img_l, ci(seg)), b.add(jl, lane_q)), ci(shift + rd * 4))
        col = b.add(col0, b.mul(lane_c4, ci(4)))
        chunk = b.div(col, ci(_VEC))
        sw = swz_val(row, bits)
        phys = chunk if sw is None else b.xor(chunk, sw)
        idx = b.add(
            b.mul(b.add(b.mul(row, ci(chunks)), phys), ci(_VEC)), b.mod(col, ci(_VEC))
        )
        if QG > 1:
            idx = b.add(idx, b.mul(qg, ci(g_elems)))
        return idx

    k_wave0 = b.mul(wave_k_id, ci(spec.wave_k))
    c_wave0 = b.mul(wave_c_id, ci(spec.wave_c))
    dy_idx = [
        [
            frag_index(0, rd, b.add(k_wave0, ci(mi * _ATOM)), CHK, swz_k, WU, GDY_ELEMS)
            for rd in (0, 1)
        ]
        for mi in range(WK16)
    ]
    x_idx = [
        [
            [
                frag_index(
                    s, rd, b.add(c_wave0, ci(ni * _ATOM)), CHC, swz_c, SEG, GX_ELEMS
                )
                for rd in (0, 1)
            ]
            for ni in range(WC16)
        ]
        for s in range(KW)
    ]

    def read_frag(smem: Value, idx: List[Value]) -> Value:
        lo = b.ds_read_tr16_b64(smem, c0, idx[0], dtype=io_type)
        hi = b.ds_read_tr16_b64(smem, c0, idx[1], dtype=io_type)
        return b.vec_concat(lo, hi)

    zero_acc = b.zero_vec_f32(4)
    zero_frag = b.zero_vec(io_type, _VEC)
    n_acc = KH * KW * WK16 * WC16

    def acc_index(r, s, mi, ni):
        return ((r * KW + s) * WK16 + mi) * WC16 + ni

    def wait_rows(lead: bool):
        """Before a barrier: lgkm, and the row DMA wait where it is needed.

        With q_groups == 1 the groups share the row buffers and every barrier
        of the lagging group (1) pairs with the leading group's (0) next one:
        group 1 must hold its slice of row t + 1 at its mid barrier, which
        pairs with group 0's row barrier of t + 1. The wait immediates are
        static, and a per-group branch around them costs the waitcnt pass
        its LDS-DMA tracking (or, duplicating the loop, the registers), so
        both groups wait at both barriers.
        """
        if spec.pingpong_halves:
            if lead:
                # Row t was already waited for at the previous mid barrier
                # (or, for a segment's first row, before the stagger).
                b.s_waitcnt(lgkmcnt=0)
            else:
                # Both groups wait (a per-group branch costs ~40 VGPRs here).
                # No lgkm wait: the extra row buffer (n_buffers) covers the
                # reads still in flight, which the MFMAs then wait for one
                # fragment at a time. Without it (pp_tight) the reads retire
                # here.
                if spec.pp_tight:
                    b.s_waitcnt(vmcnt=(D - 1) * N_INSTR, lgkmcnt=0)
                else:
                    b.s_waitcnt(vmcnt=(D - 1) * N_INSTR)
        elif lead:
            b.s_waitcnt(vmcnt=(D - 1) * N_INSTR, lgkmcnt=0)
        else:
            b.s_waitcnt(lgkmcnt=0)

    def mid_barrier():
        """The ping-pong barrier between a group's reads and its MFMAs."""
        wait_rows(False)
        b.s_barrier_bare()
        if spec.pingpong_halves or PPS:
            b.sched_barrier(0)

    def step(buf: int, t: Value, ring, accs, compute: bool, issue):
        """One row: wait, barrier, DMA row t + D, read, MFMA."""
        wait_rows(True)
        b.s_barrier_bare()
        if PPS:
            # The row's setup stays after the barrier (not in the MFMA tail).
            b.sched_barrier(0)
        # Ping-pong pins this phase with scheduling fences; there the DMA goes
        # after the reads, or the waitcnt pass (which cannot tell the X buffer
        # reads from the new DMA's target) drains the DMAs before every read.
        late = spec.pingpong_halves or PPS
        if not late:
            issue((buf + D) % NB, b.add(t, ci(D)))
        newest = [read_frag(dy_bufs[buf], dy_idx[mi]) for mi in range(WK16)]
        ring = ring[1:] + [newest]
        xf = None
        if compute:
            xf = [
                [read_frag(x_bufs[buf], x_idx[s][ni]) for ni in range(WC16)]
                for s in range(KW)
            ]
        if late:
            # Pinned after the reads, so its address math and issue run
            # under the LDS read latency.
            b.sched_barrier(0)
            issue((buf + D) % NB, b.add(t, ci(D)))
        if not compute:
            if spec.pingpong_halves:
                # Keep two barriers per row: the groups share the buffers, so
                # the lagging group must stay half a row (not a row) behind.
                mid_barrier()
            return ring, accs
        if spec.pingpong:
            # The other group's half-step: its reads run under these MFMAs,
            # which take issue priority while they do.
            mid_barrier()
            b.s_setprio(1)
        accs = list(accs)
        for r in range(KH):
            dyf = ring[KH - 1 - r]
            for s in range(KW):
                for mi in range(WK16):
                    for ni in range(WC16):
                        i = acc_index(r, s, mi, ni)
                        accs[i] = _mfma(
                            b, io_dtype, "16x16x32", dyf[mi], xf[s][ni], accs[i]
                        )
        if spec.pingpong:
            if spec.pingpong_halves or PPS:
                b.sched_barrier(0)
            b.s_setprio(0)
        elif AUTO:
            # A runtime trip count lets the scheduler sink the MFMAs into the
            # last rows of the unrolled body; keep each row's MFMAs in its row.
            b.sched_barrier(0)
        return ring, accs

    # ---- persistent segment loop -------------------------------------------
    def segment(accs, n0, q, h0, h_end, valid=None):
        """Rows [h0, h_end) of strip q of images n0 .. n0 + unfold_n - 1.

        ``valid`` false runs it with every load masked (a wave group with no
        segment left keeps step with the other).
        """
        lane_offs = seg_lane_offsets(n0, b.mul(q, ci(WU)))
        # A whole strip needs no tail cut: the DMAs past the last row are
        # masked by the row bounds or never read. A constant keeps the
        # row-loop schedule close to the fixed-rows one.
        t_end = ci(1 << 30) if AUTO else ci(T_PRO + HPB)
        if valid is not None:
            t_end = b.select(valid, t_end, c0)

        def issue(buf, t):
            issue_row(buf, t, h0, h_end, lane_offs, t_end)

        for t in range(D):
            issue(t % NB, ci(t))
        if spec.pingpong_halves:
            # Group 1 starts the segment one barrier late, holding row 0.
            b.s_waitcnt(vmcnt=(D - 1) * N_INSTR)
            with b.scf_if(b.cmp_eq(pp_grp, c1)):
                b.s_barrier_bare()

        ring = [[zero_frag] * WK16 for _ in range(KH)]
        for t in range(T_PRO):
            ring, accs = step(t % NB, ci(t), ring, accs, False, issue)

        n_iter_args = [(f"a{i}", v) for i, v in enumerate(accs)]
        n_iter_args += [
            (f"r{j}_m{mi}", ring[j][mi]) for j in range(1, KH) for mi in range(WK16)
        ]
        row_loop = b.scf_for_iter(
            c0,
            n_rb if AUTO else ci(HPB // U),
            c1,
            n_iter_args,
            iv_name="rb",
            elide_trailing_barrier=False,
        )
        with row_loop as (rb, rc):
            accs_i = list(rc[:n_acc])
            ring_i = [[zero_frag] * WK16] + [
                list(rc[n_acc + (j - 1) * WK16 : n_acc + j * WK16])
                for j in range(1, KH)
            ]
            t_base = b.add(b.mul(rb, ci(U)), ci(T_PRO))
            for u in range(U):
                ring_i, accs_i = step(
                    (T_PRO + u) % NB, b.add(t_base, ci(u)), ring_i, accs_i, True, issue
                )
            b.scf_yield(*accs_i, *[v for j in range(1, KH) for v in ring_i[j]])
        accs_out = list(row_loop.results[:n_acc])
        if spec.row_tail and U > 1:
            ring_t = [[zero_frag] * WK16] + [
                list(row_loop.results[n_acc + (j - 1) * WK16 : n_acc + j * WK16])
                for j in range(1, KH)
            ]
            t_tail = b.add(b.mul(n_rb, ci(U)), ci(T_PRO))
            acc_type = zero_acc.type

            def tail(u, ring_u, accs_u):
                """Tail rows u.. of the strip, each nested in the previous
                row's test (one straight path with exits off it)."""
                branch = b.scf_if_else(b.cmp_lt(ci(u), n_tail), [acc_type] * n_acc)
                with branch as (then_ctx, else_ctx):
                    with then_ctx:
                        ring_n, accs_n = step(
                            (T_PRO + u) % NB,
                            b.add(t_tail, ci(u)),
                            ring_u,
                            accs_u,
                            True,
                            issue,
                        )
                        if u + 1 < U - 1:
                            accs_n = tail(u + 1, ring_n, accs_n)
                        b.scf_yield(*accs_n)
                    with else_ctx:
                        b.scf_yield(*accs_u)
                return list(branch.results)

            accs_out = tail(0, ring_t, accs_out)
        # Drain the dummy tail DMAs before the next segment reuses the buffers.
        b.s_waitcnt(vmcnt=0, lgkmcnt=0)
        b.s_barrier_bare()
        if spec.pingpong_halves:
            # Group 0 catches up: both groups leave the segment aligned.
            with b.scf_if(b.cmp_eq(pp_grp, c0)):
                b.s_barrier_bare()
        return accs_out

    def seg_origin(seg):
        """Segment ``seg`` -> (n0, q, h0, h_end): first image, strip, and the
        row range. Segments run row block fastest, then strip, then image
        group; in auto mode a segment is a whole strip."""
        if AUTO:
            rest, h0 = seg, c0
        else:
            hb = b.mod(seg, n_hb)
            rest = b.div(seg, n_hb)
            h0 = b.mul(hb, ci(HPB))
        n0 = b.mul(b.div(rest, n_q), ci(UN))
        q = b.mod(rest, n_q)
        h_end = p_H if AUTO else b.smin(b.add(h0, ci(HPB)), p_H)
        return n0, q, h0, h_end

    # Ping-pong group: the q group, or the half of the block's waves.
    pp_grp = qg
    if spec.pingpong_halves:
        pp_grp = b.div(wave, ci(NW // 2))
    pp_groups = spec.pingpong and not spec.pingpong_halves
    if pp_groups:
        # Group 1 starts one barrier late: from here on its every barrier
        # pairs with group 0's next one, half a row ahead of it.
        with b.scf_if(b.cmp_eq(pp_grp, c1)):
            b.s_barrier_bare()
    zero_accs = [(f"acc{i}", zero_acc) for i in range(n_acc)]
    if QG == 1:
        seg_loop = b.scf_for_iter(
            split,
            total_segs,
            n_splits,
            zero_accs,
            iv_name="seg",
            elide_trailing_barrier=False,
        )
        with seg_loop as (seg, carried):
            accs_out = segment(list(carried), *seg_origin(seg))
            b.scf_yield(*accs_out)
    else:
        # Group g of block `split` takes virtual split split * QG + g; both
        # groups run group 0's segment count (the larger) in lockstep.
        n_vsplits = b.mul(n_splits, ci(QG))
        vsplit0 = b.mul(split, ci(QG))
        n_iter = b.div(
            b.add(b.sub(total_segs, vsplit0), b.sub(n_vsplits, c1)), n_vsplits
        )
        seg_loop = b.scf_for_iter(
            c0, n_iter, c1, zero_accs, iv_name="segi", elide_trailing_barrier=False
        )
        with seg_loop as (segi, carried):
            seg = b.add(b.add(vsplit0, qg), b.mul(segi, n_vsplits))
            accs_out = segment(
                list(carried), *seg_origin(seg), valid=b.cmp_lt(seg, total_segs)
            )
            b.scf_yield(*accs_out)
    accs = list(seg_loop.results)
    if pp_groups:
        with b.scf_if(b.cmp_eq(pp_grp, c0)):
            b.s_barrier_bare()

    # ---- epilogue: atomic add into dW[total_k, KH, KW, cpg] -----------------
    D_ptr = params["D"]

    rs_stride = c_cpg
    k_stride = b.mul(c_cpg, ci(KH * KW))
    if spec.slab:
        # Slab ``split`` of the scratch starts dW's element count past slab 0.
        ws_shift = b.mul(split, b.mul(params["p_total_k"], k_stride))

    def _put(ptr, off, val):
        if spec.slab:
            b.global_store(ptr, b.add(off, ws_shift), val, align=4)
        elif spec.dw16:
            b.global_store(ptr, off, b.cast_f32_to(val, io_type), align=2)
        elif spec.one_split:
            b.global_store(ptr, off, val, align=4)
        else:
            b.global_atomic_add(ptr, off, val)

    def _emit(ptr, off, val, ok):
        """``_put`` under ``ok``; pk16: even lanes add (own, odd neighbour's)
        as one packed pair (c even, cpg % 8 == 0: the pair shares ``ok``).
        ``val`` may be a thunk, built inside the guard where no lane needs it
        outside (the pk16 swap needs it before)."""
        if not spec.pk16:
            with b.scf_if(ok):
                _put(ptr, off, val() if callable(val) else val)
            return
        if callable(val):
            val = val()
        from rocke.helpers.io import pack_f32_to

        nb = b.warp_shuffle_xor_dpp(val, 1)
        even = b.cmp_eq(b.mod(lane, ci(2)), c0)
        with b.scf_if(b.land(ok, even)):
            pair = pack_f32_to(b, [val, nb], dtype=io_dtype)
            if io_dtype == "bf16":
                b.global_atomic_add_pk_bf16(ptr, off, pair)
            else:
                b.global_atomic_add_pk_f16(ptr, off, pair)

    k_lane0 = b.add(b.add(k_blk0, k_wave0), b.mul(lane_hi, ci(4)))
    k_in_grp0 = b.sub(k_lane0, b.mul(group, c_kpg))
    if GP:
        # In-group c is the lane's column alone; groups past the last are
        # masked through k (k_in >= kpg never holds, so test the group).
        c_lane0 = b.mod(lane, ci(16))
        c_cpg_ok = b.select(b.cmp_lt(group, p_groups), c_cpg, c0)
    else:
        c_lane0 = b.add(b.add(c_blk0, c_wave0), b.mod(lane, ci(16)))
        c_cpg_ok = c_cpg
    if QG == 4:
        # Every group hands a tap's accumulators over through LDS, and group
        # g sums its quarter of the tap's floats over the four slices, so all
        # waves issue the atomics. Double-buffered on the tap parity (one
        # barrier per tap) where it fits.
        n_atoms = WK16 * WC16
        per_g = n_atoms * 4 // QG  # floats of a lane a group sums per tap
        vw = min(per_g, 4)
        slice_f = NW * n_atoms * _WAVE * 4
        red = [
            b.smem_alloc_f32([1, QG * slice_f], name_hint=f"red{i}")
            for i in range(spec.red_buffers())
        ]

        def red_index(src: Value, flt: Value) -> Value:
            # flt = atom * 4 + slot of this lane
            atom = b.div(flt, ci(4))
            return b.add(
                b.mul(src, ci(slice_f)),
                b.add(
                    b.mul(
                        b.add(
                            b.mul(b.add(b.mul(wave, ci(n_atoms)), atom), ci(_WAVE)),
                            lane,
                        ),
                        ci(4),
                    ),
                    b.mod(flt, ci(4)),
                ),
            )

        for r in range(KH):
            for s in range(KW):
                tap = r * KW + s
                buf = red[tap % len(red)]
                for mi in range(WK16):
                    for ni in range(WC16):
                        b.smem_store_vN_f32(
                            buf,
                            [c0, red_index(qg, ci((mi * WC16 + ni) * 4))],
                            accs[acc_index(r, s, mi, ni)],
                            4,
                        )
                b.s_waitcnt(lgkmcnt=0)
                b.s_barrier_bare()
                for q in range(per_g // vw):
                    flt = b.add(b.mul(qg, ci(per_g)), ci(q * vw))
                    atom = b.div(flt, ci(4))
                    mi_v = b.div(atom, ci(WC16))
                    ni_v = b.mod(atom, ci(WC16))
                    tot = None
                    for src in range(QG):
                        v = b.smem_load_vN_f32(buf, c0, red_index(ci(src), flt), n=vw)
                        tot = v if tot is None else b.fadd(tot, v)
                    c_in = b.add(c_lane0, b.mul(ni_v, ci(_ATOM)))
                    c_ok = b.cmp_lt(c_in, c_cpg_ok)
                    for e in range(vw):
                        k_off = b.add(
                            b.mul(mi_v, ci(_ATOM)), b.add(b.mod(flt, ci(4)), ci(e))
                        )
                        k_in = b.add(k_in_grp0, k_off)
                        k_glb = b.add(k_lane0, k_off)
                        ok = b.land(c_ok, b.cmp_lt(k_in, c_kpg))
                        off = b.add(
                            b.add(b.mul(k_glb, k_stride), c_in),
                            b.mul(ci(tap), rs_stride),
                        )
                        val = b.vec_extract(tot, e)
                        _emit(D_ptr, off, val, ok)
                if len(red) == 1:
                    # The reads retire before the next tap's stores.
                    b.s_waitcnt(lgkmcnt=0)
                    b.s_barrier_bare()
        return b.kernel
    if QG > 1:
        # Tap t is summed and written by group t % 2; the other group hands
        # its accumulators over through LDS, double-buffered on the tap
        # parity, one barrier per tap.
        n_atoms = WK16 * WC16
        red = [
            b.smem_alloc_f32([1, NW * n_atoms * _WAVE * 4], name_hint=f"red{i}")
            for i in range(2)
        ]

        def red_index(atom: int) -> Value:
            return b.mul(
                b.add(
                    b.mul(b.add(b.mul(wave, ci(n_atoms)), ci(atom)), ci(_WAVE)), lane
                ),
                ci(4),
            )

        for r in range(KH):
            for s in range(KW):
                tap = r * KW + s
                buf = red[tap % 2]
                is_recv = b.cmp_eq(qg, ci(tap % 2))
                with b.scf_if(b.lnot(is_recv)):
                    for mi in range(WK16):
                        for ni in range(WC16):
                            b.smem_store_vN_f32(
                                buf,
                                [c0, red_index(mi * WC16 + ni)],
                                accs[acc_index(r, s, mi, ni)],
                                4,
                            )
                b.s_waitcnt(lgkmcnt=0)
                b.s_barrier_bare()
                with b.scf_if(is_recv):
                    for ni in range(WC16):
                        c_in = b.add(c_lane0, ci(ni * _ATOM))
                        c_ok = b.cmp_lt(c_in, c_cpg_ok)
                        for mi in range(WK16):
                            tot = b.fadd(
                                accs[acc_index(r, s, mi, ni)],
                                b.smem_load_vN_f32(
                                    buf, c0, red_index(mi * WC16 + ni), n=4
                                ),
                            )
                            for slot in range(4):
                                k_in = b.add(k_in_grp0, ci(mi * _ATOM + slot))
                                k_glb = b.add(k_lane0, ci(mi * _ATOM + slot))
                                ok = b.land(c_ok, b.cmp_lt(k_in, c_kpg))
                                off = b.add(
                                    b.add(b.mul(k_glb, k_stride), c_in),
                                    b.mul(ci(tap), rs_stride),
                                )
                                _emit(D_ptr, off, partial(b.vec_extract, tot, slot), ok)
        return b.kernel
    for ni in range(WC16):
        c_in = b.add(c_lane0, ci(ni * _ATOM))
        c_ok = b.cmp_lt(c_in, c_cpg_ok)
        for mi in range(WK16):
            for slot in range(4):
                k_in = b.add(k_in_grp0, ci(mi * _ATOM + slot))
                k_glb = b.add(k_lane0, ci(mi * _ATOM + slot))
                ok = b.land(c_ok, b.cmp_lt(k_in, c_kpg))
                base = b.add(b.mul(k_glb, k_stride), c_in)
                for r in range(KH):
                    for s in range(KW):
                        val = b.vec_extract(accs[acc_index(r, s, mi, ni)], slot)
                        off = b.add(base, b.mul(ci(r * KW + s), rs_stride))
                        _emit(D_ptr, off, val, ok)
    return b.kernel


# ---------------------------------------------------------------------------
# Slab fold: dW[e] = sum_split ws[split][e] (the ``slab`` epilogue's reduce)
# ---------------------------------------------------------------------------

_FOLD_THREADS = 256
# Slab loads each thread keeps in flight.
_FOLD_UNROLL = 4


@dataclass(frozen=True)
class WgradSlabFoldSpec:
    """Sum ``parts`` fp32 slabs of ``n`` elements into dW (``out`` dtype).

    A block owns ``4 * 256 / slices`` consecutive dW elements: its threads are
    ``slices`` slices of quads, each slice loads every ``slices``-th slab as
    16-byte vectors through a buffer descriptor (an index past the last slab
    loads out of range, i.e. 0), and the slices meet in LDS. Shape-generic: ``parts`` and ``n`` are
    kernargs.
    """

    out: str = "fp16"
    slices: int = 8
    name: str = "direct_wgrad_slab_fold"

    @property
    def quads(self) -> int:
        return _FOLD_THREADS // self.slices

    @property
    def elems_per_block(self) -> int:
        return 4 * self.quads

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        return kernel_name_join(self.name, f"s{self.slices}", self.out)

    def launch_geometry(self, n_elems: int):
        return (ceil_div(n_elems, self.elems_per_block), 1, 1), (_FOLD_THREADS, 1, 1)


# Kernargs after the A/D pointers and byte sizes, in order.
WGRAD_SLAB_FOLD_ARGS = ("parts", "n_elems")


def wgrad_slab_fold_signature(spec: WgradSlabFoldSpec) -> list:
    """A = the fp32 slabs, D = dW, as the AOT plans bind helper steps."""
    from rocke.helpers.spec import SignatureBuilder

    sb = (
        SignatureBuilder()
        .ptr("A", "fp32")
        .ptr("D", spec.out)
        .scalar("A_bytes", "i32")
        .scalar("D_bytes", "i32")
    )
    for name in WGRAD_SLAB_FOLD_ARGS:
        sb = sb.scalar(name, "i32")
    return sb.build()


def is_valid_slab_fold_spec(spec: WgradSlabFoldSpec):
    if spec.out not in ("fp32", "fp16", "bf16"):
        return False, f"unsupported out dtype {spec.out!r}"
    if spec.slices < 1 or _FOLD_THREADS % spec.slices or spec.slices > 64:
        return False, "slices must divide 256 (at most 64)"
    return True, "ok"


def build_wgrad_slab_fold(spec: WgradSlabFoldSpec, arch: str = "gfx950") -> KernelDef:
    from rocke.core.ir import F32, I32, PtrType
    from rocke.helpers.io import io_ir_type

    ok, why = is_valid_slab_fold_spec(spec)
    if not ok:
        raise ValueError(f"invalid WgradSlabFoldSpec for {arch}: {why}")
    S, Q, E, U = spec.slices, spec.quads, spec.elems_per_block, _FOLD_UNROLL
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = _FOLD_THREADS
    ws = b.param("A", PtrType(F32, "global"), noalias=True, readonly=True, align=16)
    out_t = F32 if spec.out == "fp32" else io_ir_type(spec.out)
    dw = b.param(
        "D",
        PtrType(out_t, "global"),
        noalias=True,
        writeonly=True,
        align=4 if spec.out == "fp32" else 2,
    )
    ws_bytes = b.param("A_bytes", I32)
    b.param("D_bytes", I32)
    parts = b.param("parts", I32)
    n_el = b.param("n_elems", I32)

    def ci(v):
        return b.const_i32(v)

    rsrc = b.buffer_rsrc(ws, ws_bytes)
    tid = b.thread_id_x()
    q = b.mod(tid, ci(Q))
    slc = b.div(tid, ci(Q))
    e0 = b.mul(b.block_id_x(), ci(E))
    # Byte offset of this thread's quad in slab 0 and the slab stride. A slab
    # index past the last loads from offset -16 (past any buffer), which
    # reads 0: the scratch may be bigger than this call's slabs.
    q_bytes = b.mul(b.add(e0, b.mul(q, ci(4))), ci(4))
    slab_bytes = b.mul(n_el, ci(4))
    zero4 = b.vec_pack([b.const_f32(0.0)] * 4, F32)
    n_it = b.div(b.add(parts, ci(S * U - 1)), ci(S * U))
    loop = b.scf_for_iter(
        ci(0),
        n_it,
        ci(1),
        [(f"a{j}", zero4) for j in range(U)],
        iv_name="it",
        elide_trailing_barrier=False,
    )
    with loop as (it, acc):
        acc = list(acc)
        vals = []
        for j in range(U):
            pidx = b.add(slc, b.mul(b.add(b.mul(it, ci(U)), ci(j)), ci(S)))
            voff = b.select(
                b.cmp_lt(pidx, parts),
                b.add(q_bytes, b.mul(pidx, slab_bytes)),
                ci(-16),
            )
            vals.append(b.buffer_load_vN(rsrc, voff, ci(0), F32, 4))
        for j in range(U):
            acc[j] = b.fadd(acc[j], vals[j])
        b.scf_yield(*acc)
    tot = loop.results[0]
    for v in loop.results[1:]:
        tot = b.fadd(tot, v)
    if S == 1:
        e = b.add(e0, b.mul(q, ci(4)))
        with b.scf_if(b.cmp_lt(e, n_el)):
            if spec.out == "fp32":
                b.global_store_vN(dw, e, tot, 4)
            else:
                from rocke.helpers.io import pack_f32_to

                vals = [b.vec_extract(tot, i) for i in range(4)]
                b.global_store_vN(dw, e, pack_f32_to(b, vals, dtype=spec.out), 4)
        return b.kernel
    red = b.smem_alloc_f32([1, S * E], name_hint="slabfold")
    b.smem_store_vN_f32(red, [ci(0), b.add(b.mul(slc, ci(E)), b.mul(q, ci(4)))], tot, 4)
    b.sync()
    for i in range(ceil_div(E, _FOLD_THREADS)):
        le = b.add(tid, ci(i * _FOLD_THREADS))
        e = b.add(e0, le)
        ok = b.cmp_lt(e, n_el)
        if E % _FOLD_THREADS:
            ok = b.land(ok, b.cmp_lt(le, ci(E)))
        with b.scf_if(ok):
            t = None
            for sl in range(S):
                v = b.vec_extract(
                    b.smem_load_vN_f32(red, ci(0), b.add(le, ci(sl * E)), n=1), 0
                )
                t = v if t is None else b.fadd(t, v)
            if spec.out == "fp32":
                b.global_store(dw, e, t, align=4)
            else:
                b.global_store(dw, e, b.cast_f32_to(t, out_t), align=2)
    return b.kernel
