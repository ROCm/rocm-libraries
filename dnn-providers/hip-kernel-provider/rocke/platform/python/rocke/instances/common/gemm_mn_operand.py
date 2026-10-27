# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MN-contiguous GEMM operands for the gfx950 DTL ring bodies.

The universal GEMM reads both operands K-contiguous (layout ``RCR``: A is
``[M, K]``, B is ``[N, K]``). The BLAS NN / TN / TT layouts store one or both
operands MN-contiguous (A as ``[K, M]``, B as ``[K, N]``). This module gives
the alias-scoped DTL ring (``dtl_stages >= 2``, 16x16x32 atoms, 16-bit data)
a native reader for such an operand, so those layouts need no transpose pass:

* **DTL (global -> LDS).** Every wave's 16-byte-per-lane ``buffer_load ...
  lds`` fills one 1 KiB LDS block: ``R = 512 / W`` K rows of ``W`` consecutive
  MN elements (``W`` = the MN chunk width). A row is ``2W`` contiguous bytes
  in global memory (one K index, ``W`` consecutive M / N), so the loads stay
  coalesced. The stage is ``tile_mn / W`` chunks of ``tile_k * W * 2 / 1024``
  blocks; blocks are ``1024 + pad`` bytes apart.
* **Bank-conflict-free transpose reads.** The fragments come from
  ``ds_read_b64_tr_b16`` (two per 16x16x32 fragment: K rows ``4r + 0..3`` of
  the lane's 8-row K group). A half-wave of one read touches the 8 K rows
  whose bits 0, 1 and 3 differ. Those three K bits select the block within a
  group of 8 consecutive blocks, so with ``pad = 32`` the 8 rows sit in 8
  distinct 32-byte bank groups. The remaining K bits (2, 4, 5, ...) select
  the row in the block and the higher blocks.
* **Addressing.** K bits map to (block, row) bit by bit, so a fragment's byte
  address is ONE lane value per (operand, stage) plus an immediate per
  (atom, K atom, read): the K atom and read index only set immediate K bits,
  the atom index only moves the MN position.
* **MFMA layout.** Lane ``l`` of fragment ``idx`` holds MN row
  ``woff + 16 idx + l % 16`` and K ``8 (l // 16) .. + 8``: exactly the layout
  of the K-contiguous ``ds_read_b128`` fragment, so the MFMA stream and the
  epilogue are unchanged.
* **Bounds.** Rows past the MN extent read the next K row's data (valid
  memory) into LDS columns that only feed masked output rows / columns; the
  tail of the operand is bounded by the tight DTL resource (``dtl_tight_rsrc``)
  of the padded bodies. K rows past K land past the operand's end, so with the
  tight resource they read zeros.

Pure Python geometry (``dtl_lane``, ``read_addr``) is used by the unit tests to
check that every (k, mn) of a stage is written exactly once and that the
reader fetches the element the MFMA expects; the IR helpers emit the same
arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

MN_BLOCK_BYTES = 1024
# the K bits whose 8 combinations one half-wave of ds_read_b64_tr_b16 reads
# (16x16x32 fragment: lane bits g -> K bits 0, 1 and a -> K bit 3)
_TR_LOW_BITS = (0, 1, 3)
# MnLayout.sub_perm: MN quad addressed by the lanes of quad slot s (an
# involution: s -> 2 * (s % 2) + s // 2)
QUAD_PERM = (0, 2, 1, 3)


def _ilog2(x: int) -> int:
    if x <= 0 or x & (x - 1):
        raise ValueError(f"{x} is not a power of two")
    return x.bit_length() - 1


def mn_layout_letters(layout: str) -> Tuple[bool, bool]:
    """``(a_mn, b_mn)`` of a GEMM data layout string.

    ``RCR`` (A ``[M, K]``, B ``[N, K]``) is the K-contiguous default. The
    first letter is A's storage: ``R`` = row-major ``[M, K]`` (K-contiguous),
    ``C`` = column-major ``[M, K]`` i.e. ``[K, M]`` (M-contiguous). The second
    letter is B's storage seen as ``[K, N]``: ``C`` = column-major
    (``[N, K]``, K-contiguous), ``R`` = row-major ``[K, N]`` (N-contiguous).
    C is always row-major (``R``). BLAS row-major names: NT = ``RCR``,
    NN = ``RRR``, TN = ``CRR``, TT = ``CCR``."""
    lay = layout.upper()
    if len(lay) != 3 or lay[0] not in "RC" or lay[1] not in "RC" or lay[2] != "R":
        raise ValueError(f"unsupported GEMM layout {layout!r}")
    return lay[0] == "C", lay[1] == "R"


def default_mn_chunk(tile_mn: int, tile_k: int, warp_ext: int) -> int:
    """MN chunk width ``W``.

    * Tiles wider than 128: one wave's MN extent (or the narrowest
      conflict-free width ``4096 / tile_k`` when that is wider). On gfx950
      this beats the widest chunk on the 256-wide beat, B3 and 16-wave ring
      bodies (the widest chunk, 256, is the slowest form there).
    * Otherwise: the widest power of two <= min(tile_mn, 256) that divides
      ``tile_mn``, keeps the transpose reads conflict free (``W >= 4096 /
      tile_k``) and is commensurate with one wave's MN extent.
    """
    if tile_mn > 128:
        w = max(warp_ext, MN_BLOCK_BYTES * 4 // tile_k, 16)
        if (
            w <= 256
            and w & (w - 1) == 0
            and tile_mn % w == 0
            and (warp_ext % w == 0 or w % warp_ext == 0)
        ):
            return w
    best = 0
    w = 16
    while w <= min(tile_mn, 256):
        ok = tile_mn % w == 0 and (warp_ext % w == 0 or w % warp_ext == 0)
        if ok and tile_k * w * 2 >= MN_BLOCK_BYTES:
            best = w
        w *= 2
    if not best:
        raise ValueError(f"no MN chunk width for tile_mn {tile_mn}, tile_k {tile_k}")
    return best


@dataclass(frozen=True)
class MnLayout:
    """LDS stage layout of one MN-contiguous operand (see the module doc)."""

    tile_mn: int
    tile_k: int
    block_size: int
    wave_size: int
    warp_ext: int  # MN elements one wave's fragments span (mfmas * 16)
    pad: int = 32  # bytes after every 1 KiB block
    chunk: int = 0  # MN chunk width W (0: default_mn_chunk)
    # ``TraitSpec.sab_vec16``: the transpose-read lanes of
    # MN quad ``s`` (lane % 4) address quad ``QUAD_PERM[s]`` instead of ``s``,
    # so fragment lanes 4s..4s+3 hold MN ``4 * QUAD_PERM[s] + j``: the
    # accumulator rows of lane groups q and q + 2 (lanes L and L + 32) are then
    # 8 consecutive MN, joined by one v_permlane32_swap per dword (16-byte C
    # stores). Same address set per 16-lane group (same banks).
    sub_perm: bool = False
    W: int = field(init=False)
    R: int = field(init=False)
    row_bits: Tuple[int, ...] = field(init=False)
    blk_bits: Tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        W = self.chunk or default_mn_chunk(self.tile_mn, self.tile_k, self.warp_ext)
        object.__setattr__(self, "W", W)
        if self.wave_size != 64:
            raise ValueError("MN operand DTL is built for wave64 (1 KiB per wave)")
        if self.tile_k < 32:
            raise ValueError("MN operand reader needs tile_k >= 32 (16x16x32 atoms)")
        if self.tile_mn % W or 2 * W > MN_BLOCK_BYTES or W < 16:
            raise ValueError(f"MN chunk {W} does not tile tile_mn {self.tile_mn}")
        if not (self.warp_ext % W == 0 or W % self.warp_ext == 0):
            raise ValueError(f"MN chunk {W} vs wave extent {self.warp_ext}")
        R = MN_BLOCK_BYTES // (2 * W)
        object.__setattr__(self, "R", R)
        tk = _ilog2(self.tile_k)
        _ilog2(W)
        low = [bt for bt in _TR_LOW_BITS if bt < tk]
        rest = [bt for bt in range(tk) if bt not in low]
        rb = _ilog2(R)
        row = rest[:rb]
        rest = rest[rb:]
        while len(row) < rb:
            # tile too shallow for a conflict-free block: steal low bits
            # (two-way bank conflicts on the transpose reads)
            row.append(low.pop())
        object.__setattr__(self, "row_bits", tuple(row))
        object.__setattr__(self, "blk_bits", tuple(low + rest))
        if self.n_blocks % self.n_waves:
            raise ValueError(
                f"{self.n_blocks} LDS blocks are not whole DTL passes of "
                f"{self.n_waves} waves"
            )

    # ------------------------------------------------------------------ sizes
    @property
    def n_waves(self) -> int:
        return self.block_size // self.wave_size

    @property
    def block_bytes(self) -> int:
        return MN_BLOCK_BYTES + self.pad

    @property
    def blocks_per_chunk(self) -> int:
        return self.tile_k * self.W * 2 // MN_BLOCK_BYTES

    @property
    def n_blocks(self) -> int:
        return self.tile_mn * self.tile_k * 2 // MN_BLOCK_BYTES

    @property
    def n_pass(self) -> int:
        return self.n_blocks // self.n_waves

    @property
    def stage_bytes(self) -> int:
        return self.n_blocks * self.block_bytes

    @property
    def wave_bytes(self) -> int:
        """LDS bytes between two waves' blocks of one DTL pass."""
        return self.block_bytes

    @property
    def pass_bytes(self) -> int:
        return self.n_waves * self.block_bytes

    @property
    def conflict_free(self) -> bool:
        return self.blk_bits[:3] == _TR_LOW_BITS

    # ------------------------------------------------------------------ bit maps
    def k_of(self, bb: int, ri: int) -> int:
        """K row (in the tile) of block ``bb`` of a chunk, row ``ri``."""
        k = 0
        for j, bt in enumerate(self.blk_bits):
            k |= ((bb >> j) & 1) << bt
        for j, bt in enumerate(self.row_bits):
            k |= ((ri >> j) & 1) << bt
        return k

    def k_contrib(self, k: int) -> int:
        """Byte offset (block + row part) of K row ``k``; additive over
        disjoint K bits."""
        off = 0
        for j, bt in enumerate(self.blk_bits):
            off += ((k >> bt) & 1) * (1 << j) * self.block_bytes
        for j, bt in enumerate(self.row_bits):
            off += ((k >> bt) & 1) * (1 << j) * 2 * self.W
        return off

    def mn_contrib(self, mn: int) -> int:
        """Byte offset (chunk + in-row part) of MN position ``mn``."""
        return (mn // self.W) * self.blocks_per_chunk * self.block_bytes + (
            mn % self.W
        ) * 2

    # ------------------------------------------------------------------ python geometry
    def dtl_lane(self, p: int, tid: int) -> Tuple[int, int]:
        """(k, mn) in the tile of the first element DTL lane ``tid`` of pass
        ``p`` loads (8 consecutive MN elements from there)."""
        w, l = divmod(tid, self.wave_size)
        beta = p * self.n_waves + w
        ch, bb = divmod(beta, self.blocks_per_chunk)
        ri, rem = divmod(l * 16, 2 * self.W)
        return self.k_of(bb, ri), ch * self.W + rem // 2

    def dtl_lds_byte(self, p: int, tid: int) -> int:
        """LDS byte (in the stage) DTL lane ``tid`` of pass ``p`` writes."""
        w, l = divmod(tid, self.wave_size)
        return p * self.pass_bytes + w * self.wave_bytes + 16 * l

    def lds_byte(self, k: int, mn: int) -> int:
        return self.k_contrib(k) + self.mn_contrib(mn)

    def read_lane_k_mn(self, lane: int, r: int, kk: int) -> Tuple[int, int]:
        """(first K row, MN offset in the 16-row atom) lane ``lane`` addresses
        with transpose read ``r`` (0/1) of K atom ``kk``."""
        g = (lane % 16) // 4
        a = (lane // 16) % 2
        c = lane // 32
        s = lane % 4
        if self.sub_perm:
            s = QUAD_PERM[s]
        return 32 * kk + 16 * c + 8 * a + 4 * r + g, 4 * s

    def read_addr(self, lane: int, woff: int, idx: int, r: int, kk: int) -> int:
        k, sub = self.read_lane_k_mn(lane, r, kk)
        return self.lds_byte(k, woff + 16 * idx + sub)

    def read_lane_bytes(self, lane: int) -> int:
        """The lane part of every transpose-read address."""
        k, sub = self.read_lane_k_mn(lane, 0, 0)
        return self.k_contrib(k) + 2 * sub

    def read_imm_bytes(self, idx: int, r: int, kk: int) -> int:
        """The (atom, read, K atom) immediate, relative to the wave's MN
        origin (see :meth:`warp_bytes`)."""
        return self.k_contrib(32 * kk + 4 * r) + self._mn_atom(idx)

    def _mn_atom(self, idx: int) -> int:
        if self.warp_ext % self.W == 0:
            return self.mn_contrib(16 * idx)
        return 32 * idx  # the wave's atoms sit in one chunk

    def warp_bytes(self, warp_idx: int) -> int:
        """Byte offset of wave row/column ``warp_idx``'s MN origin."""
        return self.mn_contrib(warp_idx * self.warp_ext)


# ---------------------------------------------------------------------------
# IR helpers (gemm_universal's staged DTL ring)
# ---------------------------------------------------------------------------


def emit_dtl_voffs(b, lay: MnLayout, tid, ld) -> List:
    """Per-pass per-lane global ELEMENT offsets ``k * ld + mn`` (i32) of the
    DTL loads of one MN-contiguous operand tile (the tile / K-tile origin goes
    into the resource base / soffset)."""
    ws = b.const_i32(lay.wave_size)
    w = b.div(tid, ws)
    lane = b.mod(tid, ws)
    lb = b.mul(lane, b.const_i32(16))
    ri = b.div(lb, b.const_i32(2 * lay.W))
    mnin = b.div(b.mod(lb, b.const_i32(2 * lay.W)), b.const_i32(2))
    # K contribution of the in-block row (lane only)
    k_ri = None
    for j, bt in enumerate(lay.row_bits):
        bit = (
            b.mod(b.div(ri, b.const_i32(1 << j)), b.const_i32(2))
            if j
            else b.mod(ri, b.const_i32(2))
        )
        term = b.mul(bit, b.const_i32(1 << bt)) if bt else bit
        k_ri = term if k_ri is None else b.add(k_ri, term)
    out = []
    for p in range(lay.n_pass):
        beta = b.add(w, b.const_i32(p * lay.n_waves)) if p else w
        ch = b.div(beta, b.const_i32(lay.blocks_per_chunk))
        bb = b.mod(beta, b.const_i32(lay.blocks_per_chunk))
        k = k_ri
        for j, bt in enumerate(lay.blk_bits):
            bit = (
                b.mod(b.div(bb, b.const_i32(1 << j)), b.const_i32(2))
                if j
                else b.mod(bb, b.const_i32(2))
            )
            term = b.mul(bit, b.const_i32(1 << bt)) if bt else bit
            k = term if k is None else b.add(k, term)
        mn = b.add(b.mul(ch, b.const_i32(lay.W)), mnin)
        out.append(b.add(b.mul(k, ld), mn) if k is not None else mn)
    return out


def emit_dtl_lane_k_mn(b, lay: MnLayout, tid) -> List[Tuple]:
    """Per pass, the ``(k, mn)`` in the tile (i32 values) of the first of the
    8 consecutive MN elements DTL lane ``tid`` loads: :meth:`MnLayout.dtl_lane`
    as IR. Used by callers that address the operand themselves (the tensor
    GETT's per-lane mode offsets); :func:`emit_dtl_voffs` is the plain
    ``k * ld + mn`` form."""
    ws = b.const_i32(lay.wave_size)
    w = b.div(tid, ws)
    lane = b.mod(tid, ws)
    lb = b.mul(lane, b.const_i32(16))
    ri = b.div(lb, b.const_i32(2 * lay.W))
    mnin = b.div(b.mod(lb, b.const_i32(2 * lay.W)), b.const_i32(2))
    k_ri = None
    for j, bt in enumerate(lay.row_bits):
        bit = b.mod(b.div(ri, b.const_i32(1 << j)), b.const_i32(2))
        term = b.mul(bit, b.const_i32(1 << bt))
        k_ri = term if k_ri is None else b.add(k_ri, term)
    out = []
    for p in range(lay.n_pass):
        beta = b.add(w, b.const_i32(p * lay.n_waves))
        ch = b.div(beta, b.const_i32(lay.blocks_per_chunk))
        bb = b.mod(beta, b.const_i32(lay.blocks_per_chunk))
        k = k_ri
        for j, bt in enumerate(lay.blk_bits):
            bit = b.mod(b.div(bb, b.const_i32(1 << j)), b.const_i32(2))
            term = b.mul(bit, b.const_i32(1 << bt))
            k = term if k is None else b.add(k, term)
        if k is None:
            k = b.const_i32(0)
        mn = b.add(b.mul(ch, b.const_i32(lay.W)), mnin)
        out.append((k, mn))
    return out


def emit_read_lane_elems(b, lay: MnLayout, lane, warp_idx):
    """Lane + wave part (ELEMENTS) of every transpose-read address of one
    operand: K bits of the lane's (g, a, c), its 4-element MN sub-chunk, and
    the wave's MN origin."""
    c4, c16 = b.const_i32(4), b.const_i32(16)
    g = b.div(b.mod(lane, c16), c4)
    a = b.mod(b.div(lane, c16), b.const_i32(2))
    c = b.div(lane, b.const_i32(32))
    sub = b.mod(lane, c4)
    if lay.sub_perm:
        # QUAD_PERM[s] = 2 * (s % 2) + s // 2
        sub = b.add(
            b.mul(b.mod(sub, b.const_i32(2)), b.const_i32(2)),
            b.div(sub, b.const_i32(2)),
        )
    terms = []
    for v, kb in ((g, None), (a, 3), (c, 4)):
        if kb is None:
            # g = K bits 0 and 1
            g0 = b.mod(v, b.const_i32(2))
            g1 = b.div(v, b.const_i32(2))
            terms.append(b.mul(g0, b.const_i32(lay.k_contrib(1) // 2)))
            terms.append(b.mul(g1, b.const_i32(lay.k_contrib(2) // 2)))
        else:
            terms.append(b.mul(v, b.const_i32(lay.k_contrib(1 << kb) // 2)))
    terms.append(b.mul(sub, c4))
    # wave MN origin
    if lay.warp_ext % lay.W == 0:
        terms.append(b.mul(warp_idx, b.const_i32(lay.mn_contrib(lay.warp_ext) // 2)))
    else:
        mn0 = b.mul(warp_idx, b.const_i32(lay.warp_ext))
        ch = b.div(mn0, b.const_i32(lay.W))
        terms.append(
            b.mul(ch, b.const_i32(lay.blocks_per_chunk * lay.block_bytes // 2))
        )
        terms.append(b.mod(mn0, b.const_i32(lay.W)))
    out = terms[0]
    for t_ in terms[1:]:
        out = b.add(out, t_)
    return out


def emit_tr_fragment(
    b, smem, lay: MnLayout, base_elems, idx: int, kk: int, dtype, sub_elems: int = 0
):
    """The ``<8 x dtype>`` 16x16x32 fragment ``idx`` of K atom ``kk``: two
    ``ds_read_b64_tr_b16`` at ``base + imm(idx, r, kk)`` (``sub_elems``: a
    constant the caller folded into ``base``)."""
    parts = []
    for r in range(2):
        imm = lay.read_imm_bytes(idx, r, kk) // 2 - sub_elems
        addr = b.add(base_elems, b.const_i32(imm)) if imm else base_elems
        parts.append(b.ds_read_tr16_b64(smem, b.const_i32(0), addr, dtype=dtype))
    return b.vec_concat(parts[0], parts[1])


# ---------------------------------------------------------------------------
# spec-level helpers (gemm_universal)
# ---------------------------------------------------------------------------

GEMM_LAYOUTS = ("RCR", "RRR", "CRR", "CCR")
# targets with ds_read_b64_tr_b16 and the 16-byte DTL
MN_OPERAND_ARCHS = ("gfx950",)


# targets of the generic (VGPR-staged) MN operand path: transposing LDS writes
# into the K-contiguous LDS tile (no transpose read needed)
MN_GENERIC_ARCHS = ("gfx942", "gfx950")


def uses_mn_dtl(spec) -> bool:
    """True when the spec's operands go through the alias-scoped DTL ring
    (``dtl_stages >= 2``): an MN-contiguous operand then uses the
    :class:`MnLayout` block layout and transpose reads; otherwise the
    VGPR-staged loader writes it transposed into the K-contiguous LDS tile."""
    tr = spec.trait
    return bool(tr.direct_to_lds and tr.dtl_prefetch and int(tr.dtl_stages) >= 2)


def spec_mn_layouts(spec) -> Tuple[Optional[MnLayout], Optional[MnLayout]]:
    """(A layout, B layout) of the spec's MN-contiguous operands on the DTL
    ring (``None`` for a K-contiguous operand, and for both when the spec
    does not use the DTL ring). Raises ``ValueError`` for a geometry the
    layout cannot express."""
    a_mn, b_mn = mn_layout_letters(spec.data.layout)
    if not uses_mn_dtl(spec):
        return None, None
    t, tr = spec.tile, spec.trait
    out = []
    for nm, is_mn, tile_mn, warps in (
        ("A", a_mn, t.tile_m, t.warp_m),
        ("B", b_mn, t.tile_n, t.warp_n),
    ):
        if not is_mn:
            out.append(None)
            continue
        out.append(
            MnLayout(
                tile_mn=tile_mn,
                tile_k=t.tile_k,
                block_size=spec.block_size,
                wave_size=spec.wave_size,
                warp_ext=tile_mn // warps,
                pad=int(tr.mn_pad),
                chunk=int(tr.mn_chunk),
                # the N-contiguous B of a sab_vec16 body (see MnLayout); the
                # M-contiguous A of a tensor GETT with the 16-byte epilogue
                sub_perm=(nm == "B" and bool(getattr(tr, "sab_vec16", False)))
                or (
                    nm == "A"
                    and bool(getattr(getattr(spec, "_gett", None), "a_sub_perm", False))
                ),
            )
        )
    return out[0], out[1]


def mn_layout_error(spec, arch: str) -> Optional[str]:
    """Why ``spec`` cannot use its MN-contiguous operand layout (``None``:
    it can, or the layout is ``RCR``)."""
    try:
        a_mn, b_mn = mn_layout_letters(spec.data.layout)
    except ValueError as e:
        return str(e)
    if not (a_mn or b_mn):
        if getattr(spec.trait, "mn_ld_any", False):
            return "mn_ld_any needs an MN-contiguous operand (layout RRR / CRR / CCR)"
        return None
    t, tr = spec.tile, spec.trait
    why = f"layout {spec.data.layout}"
    if getattr(tr, "mn_ld_any", False):
        # the shifted last tile (TraitSpec.mn_ld_any) recomputes rows of
        # the tile before it: every output path must store, never accumulate
        if not uses_mn_dtl(spec):
            return f"{why}: mn_ld_any is a DTL-ring lever (dtl_stages >= 2)"
        if int(tr.split_k) > 1 or int(getattr(tr, "split_k_serial", 0)) or tr.streamk:
            return (
                f"{why}: mn_ld_any shifts the last tile onto rows of the one before "
                "it (split-K / stream-K would add those partials twice)"
            )
        if spec.batched:
            return f"{why}: mn_ld_any is built for the non-batched body"
    if not uses_mn_dtl(spec):
        return _mn_generic_error(spec, arch, a_mn, b_mn, why)
    if arch not in MN_OPERAND_ARCHS:
        return f"{why}: MN-contiguous operands need ds_read_b64_tr_b16 ({MN_OPERAND_ARCHS})"
    if spec.data.dtype_a not in ("fp16", "bf16"):
        return f"{why}: 16-bit A/B only"
    if not (tr.direct_to_lds and tr.dtl_prefetch and int(tr.dtl_stages) >= 2):
        return f"{why}: needs the alias-scoped DTL ring (direct_to_lds, dtl_prefetch, dtl_stages >= 2)"
    if (t.warp_tile_m, t.warp_tile_n, t.warp_tile_k) != (16, 16, 32):
        return f"{why}: transpose fragments are built for the 16x16x32 atom"
    if t.warp_k != 1:
        return f"{why}: warp_k must be 1"
    blocked = [
        ("pingpong", int(tr.pingpong) > 0),
        ("asm_loop", bool(tr.asm_loop)),
        ("asm_loop_rawrd", bool(tr.asm_loop_rawrd)),
        # the vector epilogue stores 8 columns per lane: B's LDS row
        # interleave order, which transpose reads cannot produce
        ("rilv_vec_epilogue with an N-contiguous B", b_mn and tr.rilv_vec_epilogue),
        ("preshuffle_b", tr.preshuffle_b),
        ("streamk", tr.streamk),
    ]
    for nm, on in blocked:
        if on:
            return f"{why}: {nm} is K-contiguous-only"
    if tr.lds_row_interleave:
        # verified only with the register-minimising scheduler: with the
        # default one the unrolled interleave loop shuffles the AGPR
        # accumulators through VGPRs / scratch, and one such kernel (CCR
        # beat-pad body) returned wrong results on gfx950
        if a_mn and b_mn:
            return f"{why}: lds_row_interleave with both operands MN-contiguous (register spills)"
        if tr.sched_strategy != "iterative-minreg":
            return f"{why}: lds_row_interleave with an MN-contiguous operand requires sched_strategy='iterative-minreg'"
    if not (tr.mn_pad in (0, 16, 32, 64)):
        return f"{why}: mn_pad must be 0 / 16 / 32 / 64 (got {tr.mn_pad})"
    try:
        spec_mn_layouts(spec)
    except ValueError as e:
        return f"{why}: {e}"
    # rows past the MN extent / K read past the operand's end on the last
    # tile: only the tight DTL resource bounds them
    # (dtl_tight_ragged_only bounds only the operands pad_m / pad_n flag)
    if tr.pad_k and not (tr.dtl_tight_rsrc and not tr.dtl_tight_ragged_only):
        return f"{why}: pad_k with an MN-contiguous operand needs dtl_tight_rsrc on every operand"
    if a_mn and tr.pad_m and not tr.dtl_tight_rsrc:
        return f"{why}: pad_m with an M-contiguous A needs dtl_tight_rsrc"
    if b_mn and tr.pad_n and not tr.dtl_tight_rsrc:
        return f"{why}: pad_n with an N-contiguous B needs dtl_tight_rsrc"
    return None


def mn_launch_error(spec, M: int, N: int, K: int) -> Optional[str]:
    """Shape rules of an MN-contiguous operand (``None``: none apply / ok).

    The leading dimension
    of an MN-contiguous operand (M for A, N for B) bounds every load only at
    whole vectors -- the VGPR-staged loader checks the first element of each
    ``mn_vec``-wide load (a ragged last vector read past the operand: GPU
    faults at 128x110x768 RRR and 347x240x16 CRR with mn_vec 8), and the DTL
    ring's 16-byte DMA needs 16-byte-aligned rows (wrong elements otherwise).
    The dispatcher's native forms already pick such widths; this makes the
    launch gate say so for any caller."""
    try:
        a_mn, b_mn = mn_layout_letters(spec.data.layout)
    except ValueError as e:
        return str(e)
    dtl = uses_mn_dtl(spec)
    v = 8 if dtl else int(spec.trait.mn_vec)
    any_ld = dtl and bool(getattr(spec.trait, "mn_ld_any", False))
    for is_mn, nm, ld, tile in (
        (a_mn, "A", int(M), int(spec.tile.tile_m)),
        (b_mn, "B", int(N), int(spec.tile.tile_n)),
    ):
        if is_mn and any_ld:
            # TraitSpec.mn_ld_any: 2-byte-aligned DMA rows, the last
            # tile shifted back to end at the extent (origin ld - tile >= 0)
            if ld < tile:
                return (
                    f"layout {spec.data.layout}: mn_ld_any shifts the last tile to "
                    f"end at the extent; the MN-contiguous {nm}'s leading dimension "
                    f"{ld} is below the tile ({tile})"
                )
            continue
        if is_mn and ld % v:
            return (
                f"layout {spec.data.layout}: the MN-contiguous {nm}'s leading "
                f"dimension {ld} is not a multiple of "
                + ("8 (16-byte DMA rows)" if dtl else f"mn_vec={v}")
            )
    return None


def _mn_generic_error(
    spec, arch: str, a_mn: bool, b_mn: bool, why: str
) -> Optional[str]:
    """Rules of the VGPR-staged MN operand path (non-DTL-ring bodies)."""
    t, tr = spec.tile, spec.trait
    if arch not in MN_GENERIC_ARCHS:
        return f"{why}: the VGPR-staged MN operand path is built for {MN_GENERIC_ARCHS}"
    if spec.data.dtype_a not in ("fp16", "bf16"):
        return f"{why}: 16-bit A/B only"
    if tr.direct_to_lds:
        return f"{why}: direct_to_lds without the alias-scoped ring (dtl_stages >= 2) is K-contiguous-only"
    blocked = [
        ("preshuffle_b", tr.preshuffle_b),
        ("streamk", tr.streamk),
        ("pipeline wsp3", tr.pipeline == "wsp3"),
        # the register-staged prefetch body has its own loader; its MN
        # operand path is the micro-block register transpose (vp_mn_kv)
        (
            "vgpr_prefetch without vp_mn_kv",
            int(getattr(tr, "vgpr_prefetch", 0)) > 0
            and not int(getattr(tr, "vp_mn_kv", 0)),
        ),
    ]
    for nm, on in blocked:
        if on:
            return f"{why}: {nm} is K-contiguous-only"
    v = int(tr.mn_vec)
    if v not in (1, 2, 4, 8):
        return f"{why}: mn_vec must be 1 / 2 / 4 / 8 (got {v})"
    kv = int(getattr(tr, "vp_mn_kv", 0))
    if kv:
        return _vp_mn_error(spec, a_mn, b_mn, why, v, kv)
    for is_mn, tile_mn, nm in ((a_mn, t.tile_m, "A"), (b_mn, t.tile_n, "B")):
        if not is_mn:
            continue
        if tile_mn % v or (tile_mn * t.tile_k // v) % spec.block_size:
            return (
                f"{why}: {nm} tile {tile_mn}x{t.tile_k} is not a whole number of "
                f"{v}-element loads per thread ({spec.block_size} threads)"
            )
    return None


def _vp_mn_error(spec, a_mn: bool, b_mn: bool, why: str, v: int, kv: int):
    """Rules of the vgpr_prefetch MN operand path (``vp_mn_kv``): micro-blocks
    of ``kv`` K rows x ``v`` M / N elements, whole micro-blocks per thread."""
    t, tr = spec.tile, spec.trait
    if kv not in (2, 4, 8):
        return f"{why}: vp_mn_kv must be 2 / 4 / 8 (got {kv})"
    if v not in (2, 4, 8):
        return f"{why}: vp_mn_kv needs mn_vec 2 / 4 / 8 (whole-dword loads; got {v})"
    if int(tr.vp_mn_order) not in (0, 1, 2, 3):
        return f"{why}: vp_mn_order must be 0..3 (got {tr.vp_mn_order})"
    if int(tr.vp_mn_worder) not in (0, 1, 2, 3):
        return f"{why}: vp_mn_worder must be 0..3 (got {tr.vp_mn_worder})"
    if int(tr.vp_mn_pipe) not in (0, 1):
        return f"{why}: vp_mn_pipe must be 0 / 1 (got {tr.vp_mn_pipe})"
    if int(tr.vp_mn_wrap) not in (0, 1):
        return f"{why}: vp_mn_wrap must be 0 / 1 (got {tr.vp_mn_wrap})"
    for is_mn, tile_mn, nm in ((a_mn, t.tile_m, "A"), (b_mn, t.tile_n, "B")):
        if not is_mn:
            continue
        if tile_mn % v or t.tile_k % kv:
            return (
                f"{why}: {nm} tile {tile_mn}x{t.tile_k} is not a whole number of "
                f"{kv}x{v} micro-blocks"
            )
        if ((tile_mn // v) * (t.tile_k // kv)) % spec.block_size and not int(
            tr.vp_mn_wrap
        ):  # (vp_mn_rounds)
            return (
                f"{why}: {nm} tile {tile_mn}x{t.tile_k} is not a whole number of "
                f"{kv}x{v} micro-blocks per thread ({spec.block_size} threads)"
            )
        q = vp_mn_order_group(int(tr.vp_mn_order))
        if (tile_mn // v) % q:
            return (
                f"{why}: vp_mn_order {tr.vp_mn_order} groups {q} M / N blocks; "
                f"{nm} has {tile_mn // v}"
            )
    return None


# ---------------------------------------------------------------------------
# vgpr_prefetch MN operands (vp_mn_kv): index math shared by
# the kernel (IRBuilder ops) and the CPU tests (IntOps)
# ---------------------------------------------------------------------------


class IntOps:
    """Python-int mirror of the IRBuilder ops the vp_mn index math uses."""

    def const_i32(self, v):
        return int(v)

    def add(self, a, b):
        return a + b

    def mul(self, a, b):
        return a * b

    def div(self, a, b):
        return a // b

    def mod(self, a, b):
        return a % b

    def xor(self, a, b):
        return a ^ b

    def shl(self, a, b):
        return a << b

    def lshr(self, a, b):
        return a >> b


def vp_mn_order_group(order: int) -> int:
    """M / N blocks per lane group of vp_mn_order (1: none)."""
    return {2: 4, 3: 2}.get(int(order), 1)


def vp_mn_rounds(tile_mn: int, tile_k: int, kv: int, mv: int, threads: int) -> int:
    """Micro-blocks per thread (the last round wraps with vp_mn_wrap)."""
    return -(-((tile_mn // mv) * (tile_k // kv)) // threads)


def vp_mn_micro(b, u: int, tid, *, threads, tile_mn, tile_k, kv, mv, order):
    """(k0, mn0) in the tile of micro-block ``u`` of thread ``tid``: vp_mn_order
    0 = K blocks fastest, 1 = M / N blocks fastest, 2 / 3 = groups of 4 / 2
    M / N blocks, then the K blocks. A count that is not a multiple of the
    thread count wraps (the surplus threads repeat micro-blocks from the
    start: identical reads and identical LDS writes)."""
    nkb, nmb = tile_k // kv, tile_mn // mv
    vi = b.add(b.mul(b.const_i32(u), b.const_i32(threads)), tid)
    if (nkb * nmb) % threads:
        vi = b.mod(vi, b.const_i32(nkb * nmb))
    o = int(order)
    if o == 0:
        kb, mb = b.mod(vi, b.const_i32(nkb)), b.div(vi, b.const_i32(nkb))
    elif o == 1:
        mb, kb = b.mod(vi, b.const_i32(nmb)), b.div(vi, b.const_i32(nmb))
    else:
        q = vp_mn_order_group(o)
        r_ = b.div(vi, b.const_i32(q))
        kb = b.mod(r_, b.const_i32(nkb))
        mb = b.add(
            b.mul(b.div(r_, b.const_i32(nkb)), b.const_i32(q)),
            b.mod(vi, b.const_i32(q)),
        )
    return b.mul(kb, b.const_i32(kv)), b.mul(mb, b.const_i32(mv))


def swizzle_col(b, col, row, swz):
    """The LDS XOR swizzle of the K-contiguous tile: ``col ^ (((row >> R) %
    2^W) << L)`` (``swz`` = (R, W, L); None: no swizzle)."""
    if swz is None:
        return col
    r, w, l_ = swz
    return b.xor(
        col,
        b.shl(b.mod(b.lshr(row, b.const_i32(r)), b.const_i32(1 << w)), b.const_i32(l_)),
    )


def vp_mn_fast_col_ok(kv: int, mv: int, swz) -> bool:
    """True when :func:`vp_mn_col` may split the swizzle (R <= log2(mn_vec)
    and L + log2(mn_vec) - R >= log2(vp_mn_kv))."""
    if swz is None:
        return True
    r, _w, l_ = swz
    mvb, kvb = mv.bit_length() - 1, kv.bit_length() - 1
    return r <= mvb and l_ + mvb - r >= kvb


def vp_mn_col(b, k0, mn0, i: int, j0: int, *, kv: int, mv: int, swz, cache: dict):
    """Swizzled LDS column of (row mn0 + i, col k0 + j0) for a micro-block
    (mn0 % mn_vec == 0, k0 % vp_mn_kv == 0, i < mn_vec, j0 < vp_mn_kv) without
    one address register per row (:func:`vp_mn_fast_col_ok` must hold). The
    XOR key is (Mk | (i >> R)) % 2^W with Mk = (mn0 >> R) % 2^W, so col' =
    D ^ S with D = k0 ^ (Mk << L) (per micro-block, ``cache``) and S = j0 ^
    (((i >> R) % 2^W) << L) (static); S's bits at or above log2(vp_mn_kv)
    are XORed into D once per distinct value, the bits below are disjoint
    from D's and become an add (an immediate offset)."""
    kvb = kv.bit_length() - 1
    if swz is None:
        s_ = j0
    else:
        r, w, l_ = swz
        s_ = j0 ^ (((i >> r) % (1 << w)) << l_)
    h, lo = (s_ >> kvb) << kvb, s_ & ((1 << kvb) - 1)
    if "D" not in cache:
        if swz is None:
            cache["D"] = k0
        else:
            r, w, l_ = swz
            cache["D"] = b.xor(
                k0,
                b.shl(
                    b.mod(b.lshr(mn0, b.const_i32(r)), b.const_i32(1 << w)),
                    b.const_i32(l_),
                ),
            )
    if h not in cache:
        cache[h] = b.xor(cache["D"], b.const_i32(h)) if h else cache["D"]
    return b.add(cache[h], b.const_i32(lo)) if lo else cache[h]


def mn_lds_extra_bytes(spec) -> int:
    """LDS bytes the MN-contiguous stages add over the plain [tile_mn, tile_k]
    stages (the per-block pad), over all stages."""
    try:
        la, lb = spec_mn_layouts(spec)
    except ValueError:
        return 0
    S = int(spec.trait.dtl_stages)
    return sum(l.n_blocks * l.pad * S for l in (la, lb) if l is not None)
