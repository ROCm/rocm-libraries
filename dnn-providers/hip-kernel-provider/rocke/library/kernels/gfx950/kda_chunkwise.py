"""Chunkwise gated delta-rule linear attention (KDA) prefill kernels for gfx950.

KDA is a gated delta-rule linear attention: a per-channel decay gate plus a
delta-rule state write, evaluated chunkwise so the token-serial recurrence
collapses into a handful of dense matmuls per chunk.

Contract
--------
Per head, tokens are grouped into chunks of ``C`` rows. Writing the per-channel
cumulative log decay within a chunk as ``Gamma_i = exp(sum_{j<=i} g_j)`` and the
whole-chunk decay as ``gamma_C = Gamma_{C-1}``, the chunk body factorizes into
six state-independent tiles

.. code-block:: text

    A    = (I + StrictTril(Diag(beta) Akk))^-1 Diag(beta)      C x C
           Akk_ij = k_i . (k_j * Gamma_i / Gamma_j)
    GK   = K * Gamma                                          C x DK
    GQ   = Q * Gamma * scale                                  C x DK
    Aqk  = Tril(GQ (K / Gamma)^T)                (i >= j)     C x C
    Kt   = (K * gamma_C / Gamma)^T                            DK x C
    dec  = gamma_C                                            DK

and a serial walk over chunks carrying the state ``S`` (DK x DV) in fp32

.. code-block:: text

    Vt = A (V - GK S)
    O  = GQ S + Aqk Vt
    S  = Diag(dec) S + Kt^T Vt

Only ``S`` is serial, so the tile construction above is one workgroup per chunk
and fully parallel over the sequence.

Numerics
--------
``Akk`` and ``Aqk`` both need the ratio ``Gamma_i / Gamma_j``, which spans the
whole chunk's decay range and overflows fp32 if formed directly: at the
reference ``gate_lower_bound = -5`` a 32-token chunk accumulates up to 160 nats.
Both are
therefore built factored against the chunk's midpoint row ``CREF = C // 2``,

.. code-block:: text

    Akk = (K * e^(Gc - Gref)) (K * e^(Gref - Gc))^T

so each factor's exponent is bounded by half the chunk range, and the product
reconstructs the ratio exactly. Every exponential is additionally clamped to the
fp32 exp2 range. The cumulative sum is kept scaled by ``log2(e)`` throughout so
the hardware ``v_exp_f32`` (base 2) is used directly with no extra multiply.

Layout
------
Inputs arrive already packed by chunk: ``tile = bh * NC + n`` indexes a chunk of
one (batch, head). q/k are bf16, the gate is fp32 per channel (not pre-summed),
beta is fp32 per token, and q/k are expected pre-normalized -- the L2 norm, gate
activation and beta sigmoid belong to the host-side pack pass, not here.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Tuple

from rocke.core.ir import BF16, F32, I32, IRBuilder, KernelDef, PtrType
from rocke.helpers.atoms import MfmaAtom
from rocke.helpers.spec import SignatureBuilder, kernel_name_join

# Cumulative log decay is kept in the log2 domain so exp2 is a bare v_exp_f32.
LOG2E = 1.4426950408889634
# v_exp_f32 saturates past this; the clamp keeps a saturated gate finite instead
# of turning a whole chunk into NaN.
EXP2_CLAMP = 126.0

_DTYPE_IR = {"bf16": BF16}
# gfx950 LDS budget per workgroup.
LDS_LIMIT = 160 * 1024


@dataclass(frozen=True)
class KdaTileSpec:
    """Tiling and LDS-layout knobs for the chunkwise KDA kernels.

    ``chunk`` is the algorithmic chunk length: it sets the ``C x C`` triangular
    solve cost (O(C^2) serial) against the number of chunks (O(T/C) serial in
    the scan), so it is the primary throughput knob rather than a free choice.

    The pads are bank-conflict padding, in elements, on the LDS row pitch. The
    MFMA operand reads are ``ds_read_b128`` at a row stride of ``DK``; an
    unpadded 128-element bf16 row puts every lane's read in the same 16 banks,
    so the pitch is padded to spread them.
    """

    chunk: int = 32
    block_size: int = 256
    # LDS row padding, in elements, for the (C x DK) MFMA operand staging tiles.
    pad_dk: int = 8
    # LDS row padding, in elements, for the scalar-accessed fp32 (C x C) tiles.
    pad_c: int = 4
    # LDS row padding, in elements, for the bf16 (C x C) tiles that feed the
    # rank-update MFMA. These are read with ds_read_b128, so the padded row
    # pitch must stay a multiple of 8 elements (16 B) or odd rows land on an
    # 8-byte boundary and silently break the read's alignment contract.
    pad_cb: int = 8
    # solve_block: row-block size of the triangular solve. The solve's arithmetic
    # splits into per-block substitution (serial, scalar VALU) and the rank
    # update against already-solved blocks (a matmul, so MFMA). Only the
    # substitution part is irreducibly scalar, and it shrinks as the square of
    # the block size, so smaller blocks move more of the O(C^3) work onto the
    # MFMA pipe -- at the cost of one more block step. ``solve_block == chunk``
    # is the degenerate single-block case: one unblocked scalar substitution and
    # no MFMA. Must be a multiple of 8 (the accumulator holds a contiguous run
    # of 8 output rows per group of 4 slots, which is what lets a block step
    # write back only its own rows) and must divide ``chunk``.
    solve_block: int = 8
    waves_per_eu: int = 0  # 0 = leave the occupancy hint off

    @property
    def wave_size(self) -> int:
        return 64

    @property
    def num_waves(self) -> int:
        return self.block_size // self.wave_size

    def name_parts(self) -> Tuple[str, ...]:
        parts = (f"c{self.chunk}", f"b{self.block_size}", f"sb{self.solve_block}")
        if (self.pad_dk, self.pad_c) != (8, 4):
            parts += (f"p{self.pad_dk}x{self.pad_c}",)
        if self.waves_per_eu:
            parts += (f"wpe{self.waves_per_eu}",)
        return parts


@dataclass(frozen=True)
class KdaChunkPrepSpec:
    """Compile-time spec for the state-independent per-chunk tile builder.

    One workgroup per chunk, grid ``(BH * NC, 1, 1)``. Every shape field is
    baked into the kernel as a constant, so the ABI carries pointers and the
    softmax scale only.
    """

    head_k: int = 128
    head_v: int = 128
    dtype: str = "bf16"
    tile: KdaTileSpec = KdaTileSpec()
    name: str = "rocke_kda_chunk_prep"

    @property
    def atom(self) -> MfmaAtom:
        """The bf16 hero atom. Both C x C products contract over the full head
        dim, so the 32x32x16 shape covers one output tile per wave with K
        stepping, and ``chunk`` must match its M/N extent."""
        return MfmaAtom.bf16_32x32x16()

    @property
    def k_steps(self) -> int:
        return self.head_k // self.atom.k

    def lds_bytes(self) -> int:
        c, t = self.tile.chunk, self.tile
        pdk, pc = self.head_k + t.pad_dk, c + t.pad_c
        return (
            4 * c * self.head_k  # g_cum       fp32 (C x DK)
            + 2 * c * self.head_k  # k_s       bf16
            + 2 * c * self.head_k  # q_s       bf16
            + 2 * c * pdk  # x_s               bf16 Akk A operand
            + 2 * c * pdk  # y_s               bf16 shared B operand
            + 2 * c * pdk  # xq_s              bf16 Aqk A operand
            + 4 * c * pc  # m_mat              fp32 T', in-block substitution
            + 4 * c * pc  # a_mat              fp32 solve result -> A
            + 4 * c * pc  # aqk_mat            fp32 Aqk
            + 2 * c * (c + t.pad_cb)  # tb_s   bf16 rank-update A operand
            + 2 * c * (c + t.pad_cb)  # zs_s   bf16 solved-so-far, transposed
            + 4 * c  # beta_s                  fp32
            + 4 * self.head_k  # gl_s          fp32 whole-chunk log decay
        )

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            f"dk{self.head_k}",
            f"dv{self.head_v}",
            self.dtype,
            *self.tile.name_parts(),
        )


def is_valid_spec(spec: KdaChunkPrepSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a prep spec on ``arch``."""
    if arch != "gfx950":
        return False, f"kda_chunk_prep is gfx950-only (got {arch})"
    if spec.dtype not in _DTYPE_IR:
        return False, f"unsupported dtype {spec.dtype!r} (bf16 only)"

    t = spec.tile
    atom = spec.atom
    if t.chunk != atom.m or t.chunk != atom.n:
        return False, (
            f"chunk ({t.chunk}) must equal the MFMA M/N extent "
            f"({atom.m}x{atom.n}); the C x C products are one atom tile"
        )
    if spec.head_k % atom.k:
        return False, (
            f"head_k ({spec.head_k}) must be a multiple of the MFMA K step "
            f"({atom.k})"
        )
    # The cumulative sum splits the chunk in half and gives one thread a whole
    # (half, channel) column, which is what pins the block to two channels'
    # worth of threads.
    if t.block_size != 2 * spec.head_k:
        return False, (
            f"block_size ({t.block_size}) must be 2*head_k ({2 * spec.head_k}) "
            "for the two-pass cumulative sum"
        )
    if t.chunk % 2:
        return False, f"chunk ({t.chunk}) must be even for the split cumsum"
    if t.solve_block % 8 or t.chunk % t.solve_block:
        return False, (
            f"solve_block ({t.solve_block}) must be a multiple of 8 and divide "
            f"chunk ({t.chunk})"
        )
    if t.block_size % t.wave_size:
        return False, f"block_size ({t.block_size}) must be a wave multiple"

    # Every global access is a 128-bit transaction; that requires each vector to
    # sit inside one row and to tile the payload exactly.
    if spec.head_k % 8 or t.chunk % 8:
        return False, "head_k and chunk must be multiples of 8 for 128-bit access"
    if (t.chunk * spec.head_k) % (8 * t.block_size):
        return False, (
            f"C*DK ({t.chunk * spec.head_k}) must be divisible by "
            f"8*block_size ({8 * t.block_size}) for the staging sweep"
        )
    if t.pad_dk % 8:
        return False, (
            f"pad_dk ({t.pad_dk}) must be a multiple of 8 so the padded row "
            "pitch keeps ds_read_b128 alignment"
        )
    if t.pad_c % 4:
        return False, f"pad_c ({t.pad_c}) must be a multiple of 4"
    if (t.chunk + t.pad_cb) % 8:
        return False, (
            f"chunk + pad_cb ({t.chunk + t.pad_cb}) must be a multiple of 8 so "
            "the bf16 C x C row pitch keeps ds_read_b128 alignment"
        )

    lds = spec.lds_bytes()
    if lds > LDS_LIMIT:
        return False, f"LDS request {lds} B exceeds the {LDS_LIMIT} B budget"
    return True, "ok"


class _ChunkCtx:
    """Everything the per-chunk tile emitter needs that does not depend on
    *which* chunk is being built: the LDS tiles, the thread/lane decomposition,
    and the shared access helpers.

    Split out from the emitter so a kernel that walks many chunks in one
    workgroup pays for this setup once, outside its loop, while the per-chunk
    work stays a single reusable emission.
    """

    def __init__(self, b: IRBuilder, spec: KdaChunkPrepSpec, inputs):
        t = spec.tile
        self.b = b
        self.spec = spec
        self.q_ptr, self.k_ptr, self.g_ptr, self.beta_ptr, self.scale = inputs
        self.C = C = t.chunk
        self.DK = DK = spec.head_k
        self.BLOCK = t.block_size
        self.PDK = DK + t.pad_dk
        self.PC = C + t.pad_c
        self.PCB = C + t.pad_cb
        self.CREF = C // 2
        self.HALF = C // 2
        self.ELEM = _DTYPE_IR[spec.dtype]
        self.atom = spec.atom
        self.N_CD = C * DK
        ELEM = self.ELEM

        # ---- LDS ----
        self.g_lds = b.smem_alloc(F32, [C, DK], "g_cum")
        self.k_lds = b.smem_alloc(ELEM, [C, DK], "k_s")
        self.q_lds = b.smem_alloc(ELEM, [C, DK], "q_s")
        self.x_lds = b.smem_alloc(ELEM, [C, self.PDK], "x_s")
        self.y_lds = b.smem_alloc(ELEM, [C, self.PDK], "y_s")
        self.xq_lds = b.smem_alloc(ELEM, [C, self.PDK], "xq_s")
        self.m_lds = b.smem_alloc(F32, [C, self.PC], "m_mat")
        self.a_lds = b.smem_alloc(F32, [C, self.PC], "a_mat")
        self.qk_lds = b.smem_alloc(F32, [C, self.PC], "aqk_mat")
        self.tb_lds = b.smem_alloc(ELEM, [C, self.PCB], "tb_s")
        self.zs_lds = b.smem_alloc(ELEM, [C, self.PCB], "zs_s")
        self.beta_lds = b.smem_alloc(F32, [C], "beta_s")
        self.gl_lds = b.smem_alloc(F32, [DK], "gl_s")

        self.tid = b.thread_id_x()
        self.lane = lane = b.mod(self.tid, b.const_i32(64))
        self.lane_m = b.mod(lane, b.const_i32(32))
        self.lane_h = lane_h = b.div(lane, b.const_i32(32))
        self.frag_k_off = b.mul(lane_h, b.const_i32(8))

        self.c_clamp = b.const_f32(-EXP2_CLAMP)
        self.c_log2e = b.const_f32(LOG2E)
        self.c_clamp_hi = b.const_f32(EXP2_CLAMP)

    def ex2(self, x):
        """exp2 with the argument clamped into the fp32 exponent range.

        A saturated gate can drive the factored exponents past the fp32 range;
        clamping keeps the tile finite instead of letting one channel turn the
        whole chunk into NaN.
        """
        b = self.b
        return b.exp2(b.fmin(b.fmax(x, self.c_clamp), self.c_clamp_hi))

    def lds_get(self, smem, idx, dtype=F32):
        """One scalar LDS read."""
        b = self.b
        return b.vec_extract(b.smem_load_vN(smem, *idx, dtype=dtype, n=1), 0)

    def lds_get8_f32(self, smem, row, col):
        """Eight consecutive fp32 from LDS as two ``ds_read_b128``."""
        b = self.b
        out = []
        for h in range(2):
            v = b.smem_load_vN(
                smem, row, b.add(col, b.const_i32(4 * h)), dtype=F32, n=4
            )
            out += [b.vec_extract(v, j) for j in range(4)]
        return out

    def lds_get8_elem(self, smem, row, col):
        """Eight consecutive bf16 from LDS as one ``ds_read_b128``, as f32."""
        b = self.b
        v = b.smem_load_vN(smem, row, col, dtype=self.ELEM, n=8)
        return [b.cast_to_f32(b.vec_extract(v, j)) for j in range(8)]

    def lds_put8(self, smem, row, col, vals_f32):
        """Eight f32 truncated to bf16 and written as one ``ds_write_b128``."""
        b = self.b
        b.smem_store_vN(
            smem,
            [row, col],
            b.vec_pack([b.cast_f32_to(v, self.ELEM) for v in vals_f32], self.ELEM),
            8,
        )


class _ChunkOffsets:
    """The one chunk's base offsets into each flat per-chunk array."""

    def __init__(self, ctx: _ChunkCtx, tile):
        b, C, DK = ctx.b, ctx.C, ctx.DK
        self.tile = tile
        self.cd = b.mul(tile, b.const_i32(ctx.N_CD))
        self.cc = b.mul(tile, b.const_i32(C * C))
        self.c = b.mul(tile, b.const_i32(C))
        self.dk = b.mul(tile, b.const_i32(DK))


class _GlobalTileSink:
    """Tile destination: HBM, for a separate scan kernel to read back.

    The hooks fire at the point each tile's values become available, so the
    store keeps the 128-bit shape the producing pass already had its values in.
    """

    # GK/GQ go straight out of the fused elementwise pass; nothing is competing
    # for the destination, so there is no reason to defer them.
    deferred_gk_gq = False

    def __init__(self, a_ptr, gk_ptr, gq_ptr, aqk_ptr, kt_ptr, dec_ptr):
        self.a_ptr = a_ptr
        self.gk_ptr = gk_ptr
        self.gq_ptr = gq_ptr
        self.aqk_ptr = aqk_ptr
        self.kt_ptr = kt_ptr
        self.dec_ptr = dec_ptr

    def dec(self, ctx, ch, col4, dec4):
        b = ctx.b
        b.global_store_vN(self.dec_ptr, b.add(ch.dk, col4), b.vec_pack(dec4, F32), 4)

    def gk_gq(self, ctx, ch, row, col, off, gk8, gq8):
        b, ELEM = ctx.b, ctx.ELEM
        gidx = b.add(ch.cd, off)
        b.global_store_vN(
            self.gk_ptr,
            gidx,
            b.vec_pack([b.cast_f32_to(v, ELEM) for v in gk8], ELEM),
            8,
        )
        b.global_store_vN(
            self.gq_ptr,
            gidx,
            b.vec_pack([b.cast_f32_to(v, ELEM) for v in gq8], ELEM),
            8,
        )

    def _cxc(self, ctx, ch, out_ptr, off, row, col, vals):
        b, ELEM = ctx.b, ctx.ELEM
        b.global_store_vN(
            out_ptr,
            b.add(ch.cc, off),
            b.vec_pack([b.cast_f32_to(v, ELEM) for v in vals], ELEM),
            8,
        )

    def a(self, ctx, ch, off, row, col, vals):
        self._cxc(ctx, ch, self.a_ptr, off, row, col, vals)

    def aqk(self, ctx, ch, off, row, col, vals):
        self._cxc(ctx, ch, self.aqk_ptr, off, row, col, vals)

    def kt(self, ctx, ch, off, dch, r8, vals):
        b = ctx.b
        b.global_store_vN(self.kt_ptr, b.add(ch.cd, off), b.vec_pack(vals, ctx.ELEM), 8)


def _emit_gk_gq_pass(ctx: _ChunkCtx, ch: "_ChunkOffsets", sink) -> None:
    """GK = K * Gamma and GQ = Q * Gamma * scale, as a standalone sweep.

    Same access shape as the main elementwise pass -- one thread owns eight
    consecutive channels of a row, so every LDS transaction is 128-bit -- but
    emitted late, for a sink whose GK/GQ destination is only free once the
    chunk's MFMA operands have been consumed.
    """
    b = ctx.b
    tid, scale = ctx.tid, ctx.scale
    DK, BLOCK, N_CD = ctx.DK, ctx.BLOCK, ctx.N_CD
    ew_col = b.mod(b.mul(tid, b.const_i32(8)), b.const_i32(DK))
    ew_row = b.div(b.mul(tid, b.const_i32(8)), b.const_i32(DK))
    ew_rstep = (BLOCK * 8) // DK
    for p in range(N_CD // (8 * BLOCK)):
        row = b.add(ew_row, b.const_i32(p * ew_rstep))
        off = b.add(b.mul(tid, b.const_i32(8)), b.const_i32(p * 8 * BLOCK))
        g8 = ctx.lds_get8_f32(ctx.g_lds, row, ew_col)
        k8 = ctx.lds_get8_elem(ctx.k_lds, row, ew_col)
        q8 = ctx.lds_get8_elem(ctx.q_lds, row, ew_col)
        gk8, gq8 = [], []
        for j in range(8):
            eg = ctx.ex2(g8[j])
            gk8.append(b.fmul(k8[j], eg))
            gq8.append(b.fmul(b.fmul(q8[j], scale), eg))
        sink.gk_gq(ctx, ch, row, ew_col, off, gk8, gq8)


def _emit_chunk_tiles(ctx: _ChunkCtx, tile, sink) -> None:
    """Emit the six state-independent tiles for the chunk indexed by ``tile``.

    Reads ``q/k/g/beta`` for the chunk out of HBM and hands each finished tile
    to ``sink``. Nothing here depends on the state recurrence, so this is the
    whole parallel part of a chunkwise KDA forward.
    """
    b, spec = ctx.b, ctx.spec
    C, DK, BLOCK = ctx.C, ctx.DK, ctx.BLOCK
    CREF, HALF, N_CD = ctx.CREF, ctx.HALF, ctx.N_CD
    ELEM, atom, scale = ctx.ELEM, ctx.atom, ctx.scale
    tid, lane, lane_m, frag_k_off = ctx.tid, ctx.lane, ctx.lane_m, ctx.frag_k_off
    g_lds, k_lds, q_lds = ctx.g_lds, ctx.k_lds, ctx.q_lds
    x_lds, y_lds, xq_lds = ctx.x_lds, ctx.y_lds, ctx.xq_lds
    m_lds, a_lds, qk_lds = ctx.m_lds, ctx.a_lds, ctx.qk_lds
    tb_lds, zs_lds = ctx.tb_lds, ctx.zs_lds
    beta_lds, gl_lds = ctx.beta_lds, ctx.gl_lds
    c_log2e = ctx.c_log2e
    ex2, lds_get = ctx.ex2, ctx.lds_get
    lds_get8_f32, lds_get8_elem = ctx.lds_get8_f32, ctx.lds_get8_elem
    lds_put8 = ctx.lds_put8

    ch = _ChunkOffsets(ctx, tile)
    q_ptr, k_ptr, g_ptr, beta_ptr = ctx.q_ptr, ctx.k_ptr, ctx.g_ptr, ctx.beta_ptr
    tile_cd, tile_c = ch.cd, ch.c

    # =================================================================
    # 1. stage g / k / q into LDS with 128-bit transactions
    # =================================================================
    # A 128-bit vector never straddles a row: DK is a multiple of both 4 (fp32)
    # and 8 (bf16), so (row, col) decomposition of the flat vector index is
    # exact and the LDS destination stays lane-contiguous.
    for i in range(N_CD // (4 * BLOCK)):
        vidx = b.add(tid, b.const_i32(i * BLOCK))
        off = b.mul(vidx, b.const_i32(4))
        row = b.div(off, b.const_i32(DK))
        col = b.mod(off, b.const_i32(DK))
        vec = b.global_load_vN(g_ptr, b.add(tile_cd, off), F32, 4)
        b.smem_store_vN(g_lds, [row, col], vec, 4)

    for i in range(N_CD // (8 * BLOCK)):
        vidx = b.add(tid, b.const_i32(i * BLOCK))
        off = b.mul(vidx, b.const_i32(8))
        row = b.div(off, b.const_i32(DK))
        col = b.mod(off, b.const_i32(DK))
        gidx = b.add(tile_cd, off)
        b.smem_store_vN(k_lds, [row, col], b.global_load_vN(k_ptr, gidx, ELEM, 8), 8)
        b.smem_store_vN(q_lds, [row, col], b.global_load_vN(q_ptr, gidx, ELEM, 8), 8)

    with b.scf_if(b.cmp_gt(b.const_i32(C), tid)):
        bv = b.global_load_f32(beta_ptr, b.add(tile_c, tid))
        b.smem_store_vN(beta_lds, [tid], bv, 1)
    b.sync()

    # =================================================================
    # 2. in-place cumulative log decay, scaled to log2
    # =================================================================
    # One thread owns a whole (half-chunk, channel) column, so the running sum
    # stays in a register and the only cross-thread step is folding the first
    # half's total into the second.
    d_ch = b.mod(tid, b.const_i32(DK))
    half = b.div(tid, b.const_i32(DK))
    row0 = b.mul(half, b.const_i32(HALF))
    acc = b.const_f32(0.0)
    for i in range(HALF):
        rr = b.add(row0, b.const_i32(i))
        acc = b.fadd(acc, lds_get(g_lds, [rr, d_ch]))
        b.smem_store_vN(g_lds, [rr, d_ch], b.fmul(acc, c_log2e), 1)
    b.sync()

    # Second half += first half's total. Two threads per channel, so each takes
    # every other row of the upper half.
    base_lo = lds_get(g_lds, [b.const_i32(HALF - 1), d_ch])
    for i in range(HALF // 2):
        rr = b.add(b.add(b.const_i32(HALF), half), b.const_i32(i * 2))
        cur = lds_get(g_lds, [rr, d_ch])
        b.smem_store_vN(g_lds, [rr, d_ch], b.fadd(cur, base_lo), 1)
    b.sync()

    # =================================================================
    # 3. dec = gamma_C, and cache the whole-chunk log decay for Kt
    # =================================================================
    with b.scf_if(b.cmp_gt(b.const_i32(DK // 4), tid)):
        col4 = b.mul(tid, b.const_i32(4))
        gl = [
            lds_get(g_lds, [b.const_i32(C - 1), b.add(col4, b.const_i32(j))])
            for j in range(4)
        ]
        b.smem_store_vN(gl_lds, [col4], b.vec_pack(gl, F32), 4)
        sink.dec(ctx, ch, col4, [ex2(v) for v in gl])

    # =================================================================
    # 4. fused elementwise pass: GK / GQ out, and all three MFMA operands
    # =================================================================
    # Every one of these five tiles is a pointwise function of the same
    # (k, q, Gcum) element, so they are built in a single sweep: g/k/q are read
    # once, and one thread owns eight consecutive channels of a row so every
    # LDS and global access in the pass is a 128-bit transaction.
    #
    #   GK = K * Gamma                 -> global
    #   GQ = Q * Gamma * scale         -> global
    #   X  = K * e^(Gc - Gref)         -> LDS, Akk's A operand
    #   Y  = K * e^(Gref - Gc)         -> LDS, shared B operand
    #   XQ = Q * scale * e^(Gc - Gref) -> LDS, Aqk's A operand
    #
    # X/Y/XQ are factored against the chunk's midpoint row so that X Y^T
    # reconstructs k_i . (k_j * Gamma_i / Gamma_j) with each factor's exponent
    # bounded by half the chunk's decay range instead of all of it.
    ew_col = b.mod(b.mul(tid, b.const_i32(8)), b.const_i32(DK))
    ew_row = b.div(b.mul(tid, b.const_i32(8)), b.const_i32(DK))
    ew_rstep = (BLOCK * 8) // DK
    gref8 = lds_get8_f32(g_lds, b.const_i32(CREF), ew_col)

    for p in range(N_CD // (8 * BLOCK)):
        row = b.add(ew_row, b.const_i32(p * ew_rstep))
        off = b.add(b.mul(tid, b.const_i32(8)), b.const_i32(p * 8 * BLOCK))
        g8 = lds_get8_f32(g_lds, row, ew_col)
        k8 = lds_get8_elem(k_lds, row, ew_col)
        q8 = lds_get8_elem(q_lds, row, ew_col)

        gk8, gq8, x8, y8, xq8 = [], [], [], [], []
        for j in range(8):
            gc, kv = g8[j], k8[j]
            qv = b.fmul(q8[j], scale)
            dref = b.fsub(gc, gref8[j])
            emr = ex2(dref)
            erm = ex2(b.fsub(gref8[j], gc))
            if not sink.deferred_gk_gq:
                eg = ex2(gc)
                gk8.append(b.fmul(kv, eg))
                gq8.append(b.fmul(qv, eg))
            x8.append(b.fmul(kv, emr))
            y8.append(b.fmul(kv, erm))
            xq8.append(b.fmul(qv, emr))

        if not sink.deferred_gk_gq:
            sink.gk_gq(ctx, ch, row, ew_col, off, gk8, gq8)
        lds_put8(x_lds, row, ew_col, x8)
        lds_put8(y_lds, row, ew_col, y8)
        lds_put8(xq_lds, row, ew_col, xq8)
    b.sync()

    def cxc_mfma(a_smem, b_smem):
        """One C x C = (C x DK)(C x DK)^T product, K-stepped over the head dim.

        Every wave computes the whole tile. There is only one 32x32 output tile
        here, so splitting K across waves would need a cross-wave reduction
        through LDS; the product is 32x32x128, small enough that the redundant
        issue costs less than the reduction's barriers.
        """
        acc = atom.zero_acc(b)
        for ks in range(spec.k_steps):
            kb = b.add(b.const_i32(ks * atom.k), frag_k_off)
            av = b.smem_load_vN(a_smem, lane_m, kb, dtype=ELEM, n=8)
            bv = b.smem_load_vN(b_smem, lane_m, kb, dtype=ELEM, n=8)
            acc = atom.emit(b, av, bv, acc)
        return acc

    # Both products share the Y operand and neither depends on the other, so
    # they are issued back to back off the one staging barrier -- the second
    # MFMA chain fills the first's latency, and the operand rebuild plus the two
    # extra barriers the sequential version needed are gone.
    acc_kk = cxc_mfma(x_lds, y_lds)
    acc_qk = cxc_mfma(xq_lds, y_lds)

    if sink.deferred_gk_gq:
        # A fused consumer has no spare LDS for GK/GQ, so they are built into the
        # X/XQ staging tiles now that both MFMAs have consumed them. The barrier
        # is what makes that safe: every wave issues the products above, so a
        # wave running ahead would otherwise overwrite an operand another wave is
        # still reading. The recomputed exp2 is cheaper than the tiles it saves.
        b.sync()
        _emit_gk_gq_pass(ctx, ch, sink)

    # =================================================================
    # 5. T' = StrictTril(Diag(beta) Akk), RHS = Diag(beta), Aqk = Tril(.)
    # =================================================================
    # Both accumulators are already in the atom's output layout, so the masks
    # and the beta scale are applied in-register on the way to LDS.
    # ``zs`` accumulates the solved columns for the rank update and must read as
    # zero for rows not yet solved, so the unsolved part contributes nothing.
    n_vec_zs = (C * C) // 8
    zero8 = b.vec_pack([b.cast_f32_to(b.const_f32(0.0), ELEM)] * 8, ELEM)
    with b.scf_if(b.cmp_gt(b.const_i32(n_vec_zs), tid)):
        zoff = b.mul(tid, b.const_i32(8))
        b.smem_store_vN(
            zs_lds,
            [b.div(zoff, b.const_i32(C)), b.mod(zoff, b.const_i32(C))],
            zero8,
            8,
        )

    with b.scf_if(b.cmp_gt(b.const_i32(64), tid)):
        for i in range(atom.c_per_lane):
            row, col = atom.lane_to_output(b, lane, i)
            bet = lds_get(beta_lds, [row])
            tp = b.select(
                b.cmp_gt(row, col),
                b.fmul(bet, b.vec_extract(acc_kk, i)),
                b.const_f32(0.0),
            )
            # fp32 for the in-block substitution (it multiplies the solved
            # values directly, so it is the precision-critical copy) and bf16
            # for the rank-update MFMA operand.
            b.smem_store_vN(m_lds, [row, col], tp, 1)
            b.smem_store_vN(tb_lds, [row, col], b.cast_f32_to(tp, ELEM), 1)
            # The substitution reads its starting value straight out of a_mat,
            # so seed it with the right-hand side Diag(beta) here; each later
            # block overwrites its own rows with (RHS - rank update).
            b.smem_store_vN(
                a_lds,
                [row, col],
                b.select(b.cmp_eq(row, col), bet, b.const_f32(0.0)),
                1,
            )
            b.smem_store_vN(
                qk_lds,
                [row, col],
                b.select(
                    b.cmp_gt(b.add(row, b.const_i32(1)), col),  # row >= col
                    b.vec_extract(acc_qk, i),
                    b.const_f32(0.0),
                ),
                1,
            )
    b.sync()

    # =================================================================
    # 6. A = (I + T')^-1 Diag(beta) by blocked forward substitution
    # =================================================================
    # Per block: rank-update the block's rows against every already-solved row
    # (a matmul, issued on MFMA), then substitute within the block (scalar, and
    # the only irreducibly serial part). The whole loop runs on wave 0 alone, so
    # the LDS hand-offs between the two halves need no s_barrier -- they are
    # ordered by the wave's own lgkmcnt -- and the other waves wait once at the
    # end instead of twice per block.
    BS = spec.tile.solve_block
    NB = C // BS
    ks_solve = C // atom.k
    with b.scf_if(b.cmp_gt(b.const_i32(64), tid)):
        for bi in range(NB):
            if bi > 0:
                # The previous block wrote its solved rows into zs, and this
                # rank update reads them at a row another lane wrote. Being on
                # one wave orders the two only once the write has retired, and
                # the LDS addresses are computed independently enough that the
                # backend does not pair them on its own -- so drain lgkmcnt
                # explicitly. Cheaper than an s_barrier, and correctness here
                # does not survive without it once blocks are adjacent.
                b.s_waitcnt(lgkmcnt=0)
                # U = T' @ Zs over the full C x C x C shape. Rows outside this
                # block and columns of already-solved-but-irrelevant rows cost
                # nothing to include: Zs is zero wherever a row is unsolved, so
                # the extra lanes of the product are exact zeros.
                accu = atom.zero_acc(b)
                for ks in range(ks_solve):
                    kb = b.add(b.const_i32(ks * atom.k), frag_k_off)
                    accu = atom.emit(
                        b,
                        b.smem_load_vN(tb_lds, lane_m, kb, dtype=ELEM, n=8),
                        b.smem_load_vN(zs_lds, lane_m, kb, dtype=ELEM, n=8),
                        accu,
                    )
                # Slots [4*bi*t, 4*(bi+1)*t) are exactly this block's rows, so
                # the update folds straight into a_mat's seeded right-hand side
                # and the substitution below needs no separate staging tile.
                for i in range(bi * 4 * (BS // 8), (bi + 1) * 4 * (BS // 8)):
                    row, col = atom.lane_to_output(b, lane, i)
                    b.smem_store_vN(
                        a_lds,
                        [row, col],
                        b.fsub(lds_get(a_lds, [row, col]), b.vec_extract(accu, i)),
                        1,
                    )
                # Same hand-off in the other direction: the substitution below
                # reads rows this loop just wrote from a different lane.
                b.s_waitcnt(lgkmcnt=0)

            with b.scf_if(b.cmp_gt(b.const_i32(C), lane)):
                zblk = []
                for r in range(bi * BS, (bi + 1) * BS):
                    cr = b.const_i32(r)
                    val = lds_get(a_lds, [cr, lane])
                    for j in range(bi * BS, r):
                        val = b.fsub(
                            val,
                            b.fmul(
                                lds_get(m_lds, [cr, b.const_i32(j)]),
                                zblk[j - bi * BS],
                            ),
                        )
                    zblk.append(val)
                    b.smem_store_vN(a_lds, [cr, lane], val, 1)
                # Transposed, so a thread's whole solved block is contiguous and
                # goes out as one ds_write_b128 -- and lands in the (n, k) order
                # the next rank update's B operand wants.
                if bi + 1 < NB:
                    for h in range(BS // 8):
                        lds_put8(
                            zs_lds,
                            lane,
                            b.const_i32(bi * BS + h * 8),
                            zblk[h * 8 : h * 8 + 8],
                        )
    b.sync()

    # =================================================================
    # 7. A and Aqk out, 128-bit stores
    # =================================================================
    n_vec_cc = (C * C) // 8

    def store_cxc(src, hook):
        """One C x C fp32 LDS tile out as bf16, 128-bit per thread."""
        for i in range(max(1, n_vec_cc // BLOCK)):
            vidx = b.add(tid, b.const_i32(i * BLOCK))
            with b.scf_if(b.cmp_gt(b.const_i32(n_vec_cc), vidx)):
                off = b.mul(vidx, b.const_i32(8))
                row = b.div(off, b.const_i32(C))
                col = b.mod(off, b.const_i32(C))
                vals = [
                    lds_get(src, [row, b.add(col, b.const_i32(j))]) for j in range(8)
                ]
                hook(ctx, ch, off, row, col, vals)

    store_cxc(a_lds, sink.a)
    store_cxc(qk_lds, sink.aqk)

    # =================================================================
    # 8. Kt = (K * gamma_C / Gamma)^T
    # =================================================================
    # Transposed output: a thread owns eight consecutive chunk rows at one
    # channel, so the eight LDS reads are strided but the global store is a
    # single transaction -- the orientation the scan wants as an MFMA operand.
    for i in range(N_CD // (8 * BLOCK)):
        vidx = b.add(tid, b.const_i32(i * BLOCK))
        off = b.mul(vidx, b.const_i32(8))
        dch = b.div(off, b.const_i32(C))
        r8 = b.mod(off, b.const_i32(C))
        glc = lds_get(gl_lds, [dch])
        vals = []
        for j in range(8):
            rj = b.add(r8, b.const_i32(j))
            kv = b.cast_to_f32(lds_get(k_lds, [rj, dch], dtype=ELEM))
            gc = lds_get(g_lds, [rj, dch])
            vals.append(b.cast_f32_to(b.fmul(kv, ex2(b.fsub(glc, gc))), ELEM))
        sink.kt(ctx, ch, off, dch, r8, vals)


def build_kda_chunk_prep(spec: KdaChunkPrepSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for the per-chunk tile builder.

    Kernel signature::

        (q, k: ptr<bf16>,      # [NT, C * DK]
         g:    ptr<f32>,       # [NT, C * DK]  per-channel log decay
         beta: ptr<f32>,       # [NT, C]
         a_out, gk_out, gq_out, aqk_out, kt_out: ptr<bf16>,
         dec_out: ptr<f32>,    # [NT, DK]
         scale: f32)

    Grid ``(NT, 1, 1)`` where ``NT = BH * NC``; block ``(block_size, 1, 1)``.
    """
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid kda_chunk_prep spec for {arch}: {why}")

    ELEM = _DTYPE_IR[spec.dtype]
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.tile.block_size
    if spec.tile.waves_per_eu:
        b.kernel.attrs["waves_per_eu"] = (
            spec.tile.waves_per_eu,
            spec.tile.waves_per_eu,
        )

    q_ptr = b.param("q_ptr", PtrType(ELEM, "global"), readonly=True, align=16)
    k_ptr = b.param("k_ptr", PtrType(ELEM, "global"), readonly=True, align=16)
    g_ptr = b.param("g_ptr", PtrType(F32, "global"), readonly=True, align=16)
    beta_ptr = b.param("beta_ptr", PtrType(F32, "global"), readonly=True, align=4)
    a_ptr = b.param("a_ptr", PtrType(ELEM, "global"), writeonly=True, align=16)
    gk_ptr = b.param("gk_ptr", PtrType(ELEM, "global"), writeonly=True, align=16)
    gq_ptr = b.param("gq_ptr", PtrType(ELEM, "global"), writeonly=True, align=16)
    aqk_ptr = b.param("aqk_ptr", PtrType(ELEM, "global"), writeonly=True, align=16)
    kt_ptr = b.param("kt_ptr", PtrType(ELEM, "global"), writeonly=True, align=16)
    dec_ptr = b.param("dec_ptr", PtrType(F32, "global"), writeonly=True, align=16)
    scale = b.param("scale", F32)

    ctx = _ChunkCtx(b, spec, (q_ptr, k_ptr, g_ptr, beta_ptr, scale))
    sink = _GlobalTileSink(a_ptr, gk_ptr, gq_ptr, aqk_ptr, kt_ptr, dec_ptr)
    _emit_chunk_tiles(ctx, b.block_id_x(), sink)

    b.ret()
    return b.kernel


@dataclass(frozen=True)
class KdaChunkFusedSpec:
    """Compile-time spec for the fused per-chunk-tiles + state-scan kernel.

    One workgroup owns a whole (batch, head) and walks its chunks in order, so
    the six per-chunk tiles are built and consumed in LDS and never reach HBM.
    That is the entire point of the fusion: on the split path the tiles are
    written once by the tile builder and read back once per v-split by the scan,
    and that round-trip is several times the traffic the problem actually needs.

    The trade is parallelism. The split tile builder has one workgroup per
    chunk; this kernel has one per (batch, head), so it needs ``BH`` to be
    comfortably above the CU count to fill the device. It is the right choice
    for prefill at batch scale, not for a single short sequence.
    """

    head_k: int = 128
    head_v: int = 128
    dtype: str = "bf16"
    tile: KdaTileSpec = KdaTileSpec()
    has_initial_state: bool = False
    store_final_state: bool = True
    name: str = "rocke_kda_chunk_fused"

    @property
    def prep(self) -> KdaChunkPrepSpec:
        """The per-chunk tile builder whose emission this kernel reuses."""
        return KdaChunkPrepSpec(
            head_k=self.head_k,
            head_v=self.head_v,
            dtype=self.dtype,
            tile=self.tile,
        )

    @property
    def atom(self) -> MfmaAtom:
        return self.prep.atom

    @property
    def state_tiles(self) -> int:
        """Atom tiles per wave across the state's ``DK`` extent.

        Each wave owns one ``atom.m``-row band of ``S^T`` and the full head
        dimension, which is what makes every one of the five per-chunk products
        partition the same way and keeps the state in that wave's registers.
        """
        return self.head_k // self.atom.n

    def lds_bytes(self) -> int:
        """Prep's tiles plus the three the scan adds.

        GK/GQ land in the X/XQ staging tiles and A/Aqk in the two bf16 C x C
        solve tiles, all of which are dead by the time the scan phase runs, so
        the fusion only pays for the transposed K tile, the bf16 mirror of the
        state, and the ``V~`` tile.
        """
        t = self.tile
        C, DK, EV = t.chunk, self.head_k, self.head_v
        return (
            self.prep.lds_bytes()
            + 2 * DK * (C + t.pad_cb)  # kt_s   bf16 (DK x C), B operand
            + 2 * EV * (DK + t.pad_dk)  # stb_s bf16 mirror of S^T
            + 2 * EV * (C + t.pad_cb)  # vn_s   bf16 residual, then V~^T
        )

    def kernel_name(self) -> str:
        parts = (f"dk{self.head_k}", f"dv{self.head_v}", self.dtype)
        parts += self.tile.name_parts()
        if self.has_initial_state:
            parts += ("h0",)
        if not self.store_final_state:
            parts += ("noht",)
        return kernel_name_join(self.name, *parts)


def is_valid_fused_spec(
    spec: KdaChunkFusedSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a fused spec on ``arch``."""
    ok, why = is_valid_spec(spec.prep, arch=arch)
    if not ok:
        return False, why

    atom = spec.atom
    t = spec.tile
    # Every product in the scan body is partitioned by giving each wave one
    # atom-row band of the state, so the v extent has to be exactly covered by
    # the waves. Anything else needs a second partitioning rule for the same
    # accumulator, which is how cross-wave reductions creep in.
    waves = t.num_waves
    if spec.head_v != atom.m * waves:
        return False, (
            f"head_v ({spec.head_v}) must equal atom.m * waves "
            f"({atom.m} * {waves} = {atom.m * waves}); each wave owns one "
            "row band of the state"
        )
    if spec.head_k % atom.n:
        return False, (
            f"head_k ({spec.head_k}) must be a multiple of the atom N extent "
            f"({atom.n})"
        )
    if t.chunk % atom.k:
        return False, (
            f"chunk ({t.chunk}) must be a multiple of the MFMA K step "
            f"({atom.k}); it is the contraction extent of three of the products"
        )
    lds = spec.lds_bytes()
    if lds > LDS_LIMIT:
        return False, f"LDS request {lds} B exceeds the {LDS_LIMIT} B budget"
    return True, "ok"


class _LdsTileSink:
    """Tile destination: LDS, for a scan fused into the same workgroup.

    Each tile lands in bf16 in the exact layout its consumer wants as an MFMA
    operand. Nothing is allocated for GK/GQ, A or Aqk: they overwrite staging
    tiles that are dead by the time they are produced (the X/XQ MFMA operands
    and the two bf16 C x C tiles the triangular solve used for its rank
    updates), which is what keeps the fused kernel inside the LDS budget.
    ``dec`` needs no destination at all -- the emitter already caches the
    whole-chunk log decay in ``gl_s``, and the state decay exponentiates it
    where it is applied.
    """

    deferred_gk_gq = True

    def __init__(self, gk_lds, gq_lds, ab_lds, aqb_lds, kt_lds):
        self.gk_lds = gk_lds
        self.gq_lds = gq_lds
        self.ab_lds = ab_lds
        self.aqb_lds = aqb_lds
        self.kt_lds = kt_lds

    def dec(self, ctx, ch, col4, dec4):
        pass

    def gk_gq(self, ctx, ch, row, col, off, gk8, gq8):
        ctx.lds_put8(self.gk_lds, row, col, gk8)
        ctx.lds_put8(self.gq_lds, row, col, gq8)

    def a(self, ctx, ch, off, row, col, vals):
        ctx.lds_put8(self.ab_lds, row, col, vals)

    def aqk(self, ctx, ch, off, row, col, vals):
        ctx.lds_put8(self.aqb_lds, row, col, vals)

    def kt(self, ctx, ch, off, dch, r8, vals):
        b = ctx.b
        b.smem_store_vN(self.kt_lds, [dch, r8], b.vec_pack(vals, ctx.ELEM), 8)


class _ScanCtx:
    """LDS tiles, lane decomposition and helpers for the state-scan body.

    The scan body is identical whether the six per-chunk tiles were just built
    in this workgroup's LDS or staged in from HBM, so both kernels share it.
    The only differences are how the tiles arrive and whether the whole-chunk
    decay arrives in the log domain (the tile emitter caches the log; the
    materialized ``dec`` tile is already exponentiated), which is what
    ``dec_is_log`` selects.

    Every wave owns one ``atom.m``-row band of ``S^T`` and the full head
    dimension. That single rule partitions all five products, which is what
    keeps the state in that wave's accumulators with no cross-wave reduction.
    """

    def __init__(
        self,
        b: IRBuilder,
        *,
        atom: MfmaAtom,
        chunk: int,
        head_k: int,
        head_v: int,
        block_size: int,
        elem,
        tiles,
        stb_lds,
        vn_lds,
        v_ptr,
        o_ptr,
        tid,
        lane,
        lane_m,
        frag_k_off,
        dec_is_log: bool,
        ex2=None,
    ):
        self.b = b
        self.ex2 = ex2
        self.atom = atom
        self.C, self.DK, self.EV = chunk, head_k, head_v
        self.BLOCK = block_size
        self.ELEM = elem
        (
            self.gk_lds,
            self.gq_lds,
            self.ab_lds,
            self.aqb_lds,
            self.kt_lds,
            self.dec_lds,
        ) = tiles
        self.stb_lds, self.vn_lds = stb_lds, vn_lds
        self.v_ptr, self.o_ptr = v_ptr, o_ptr
        self.tid, self.lane, self.lane_m = tid, lane, lane_m
        self.frag_k_off = frag_k_off
        self.dec_is_log = dec_is_log
        self.NS = head_k // atom.n
        self.KS_DK, self.KS_C = head_k // atom.k, chunk // atom.k
        self.CPL = atom.c_per_lane
        self.N_CV = chunk * head_v
        # This wave's band of S^T rows.
        self.wrow = b.mul(b.div(tid, b.const_i32(64)), b.const_i32(atom.m))

    def slot(self, i):
        """Slot ``i``'s (row, col) inside the atom's output tile.

        Recomputed at each use rather than hoisted: the mapping is a couple of
        VALU ops off ``lane``, but holding all ``c_per_lane`` pairs live spans
        the whole chunk loop at two registers each, and this kernel's occupancy
        is register-sensitive.
        """
        return self.atom.lane_to_output(self.b, self.lane, i)

    def gemm(self, a_smem, a_row, b_smem, b_row, ksteps, acc):
        """``acc += A B^T`` over ``ksteps`` atom steps, both operands from LDS."""
        b = self.b
        for ks in range(ksteps):
            kb = b.add(b.const_i32(ks * self.atom.k), self.frag_k_off)
            av = b.smem_load_vN(a_smem, a_row, kb, dtype=self.ELEM, n=8)
            bv = b.smem_load_vN(b_smem, b_row, kb, dtype=self.ELEM, n=8)
            acc = self.atom.emit(b, av, bv, acc)
        return acc

    def state_idx(self, base, i, ti):
        """Global (ev, dk) offset of accumulator slot ``i`` of state tile ``ti``."""
        b = self.b
        row, col = self.slot(i)
        ev = b.add(self.wrow, row)
        dk = b.add(b.const_i32(ti * self.atom.n), col)
        return b.add(base, b.add(b.mul(ev, b.const_i32(self.DK)), dk))

    def publish_state(self, state):
        """S^T -> its bf16 mirror, the operand form both consumers read.

        One scalar write per element: a lane's slots are consecutive *rows* of
        ``S^T``, so in the operand layout they sit a row pitch apart and cannot
        be packed. The mirror itself is unavoidable -- the accumulator and
        A-operand lane mappings are transposes of each other, so this round
        trip through LDS is what performs the relayout.
        """
        b = self.b
        for ti in range(self.NS):
            for i in range(self.CPL):
                row, col = self.slot(i)
                b.smem_store_vN(
                    self.stb_lds,
                    [
                        b.add(self.wrow, row),
                        b.add(b.const_i32(ti * self.atom.n), col),
                    ],
                    b.cast_f32_to(b.vec_extract(state[ti], i), self.ELEM),
                    1,
                )

    def tiles_for_sink(self):
        """The five tile destinations a fused tile emitter writes into.

        ``dec`` is excluded: the emitter already caches the whole-chunk log
        decay itself, so the sink has nothing to do for it.
        """
        return (self.gk_lds, self.gq_lds, self.ab_lds, self.aqb_lds, self.kt_lds)

    def zero_state(self):
        return [self.atom.zero_acc(self.b) for _ in range(self.NS)]

    def load_state(self, ptr, base):
        """Seed the accumulators from an ``[BH, DV, DK]`` fp32 state."""
        b = self.b
        out = []
        for ti in range(self.NS):
            vals = [
                b.global_load_f32(ptr, self.state_idx(base, i, ti))
                for i in range(self.CPL)
            ]
            out.append(b.vec_pack(vals, F32))
        return out

    def store_state(self, ptr, base, state):
        b = self.b
        for ti in range(self.NS):
            for i in range(self.CPL):
                b.global_store_vN(
                    ptr, self.state_idx(base, i, ti), b.vec_extract(state[ti], i), 1
                )


def _emit_scan_body(sc: _ScanCtx, state, tile):
    """One chunk of the state recurrence; returns the updated state.

    .. code-block:: text

        Z^T  = S^T GK^T                    EV x C
        R^T  = V^T - Z^T                   EV x C   (in-register)
        V~^T = R^T A^T                     EV x C
        O    = GQ S + Aqk V~               C x EV
        S^T <- Diag(dec) S^T + V~^T Kt^T   EV x DK

    Working transposed keeps every product in ``A B^T`` form with the
    contraction on the fastest axis, so no operand ever needs an LDS transpose.
    """
    b, atom = sc.b, sc.atom
    lane_m, wrow, CPL = sc.lane_m, sc.wrow, sc.CPL
    ELEM = sc.ELEM
    cv_base = b.mul(tile, b.const_i32(sc.N_CV))

    # ---- Z^T = S^T GK^T, then R^T = V^T - Z^T into the V~ tile ----------
    acc_z = sc.gemm(
        sc.stb_lds, b.add(wrow, lane_m), sc.gk_lds, lane_m, sc.KS_DK, atom.zero_acc(b)
    )
    # A lane's slots are four runs of four consecutive v channels at one fixed
    # chunk row, so V arrives in four 4-wide loads and the residual never needs
    # an fp32 staging tile.
    for grp in range(CPL // 4):
        row0, col = sc.slot(4 * grp)
        vvec = b.global_load_vN(
            sc.v_ptr,
            b.add(
                cv_base,
                b.add(b.mul(col, b.const_i32(sc.EV)), b.add(wrow, row0)),
            ),
            ELEM,
            4,
        )
        for j in range(4):
            row, _ = sc.slot(4 * grp + j)
            res = b.fsub(
                b.cast_to_f32(b.vec_extract(vvec, j)),
                b.vec_extract(acc_z, 4 * grp + j),
            )
            b.smem_store_vN(
                sc.vn_lds, [b.add(wrow, row), col], b.cast_f32_to(res, ELEM), 1
            )
    b.sync()

    # ---- V~^T = R^T A^T -------------------------------------------------
    acc_v = sc.gemm(
        sc.vn_lds, b.add(wrow, lane_m), sc.ab_lds, lane_m, sc.KS_C, atom.zero_acc(b)
    )
    b.sync()
    for i in range(CPL):
        row, col = sc.slot(i)
        b.smem_store_vN(
            sc.vn_lds,
            [b.add(wrow, row), col],
            b.cast_f32_to(b.vec_extract(acc_v, i), ELEM),
            1,
        )
    b.sync()

    # ---- O = GQ S + Aqk V~ ----------------------------------------------
    # This wave owns a band of output *columns* here (the same band of v
    # channels it owns of the state), so GQ and Aqk are the A operands and the
    # state mirror and V~ are the B operands.
    acc_o = sc.gemm(
        sc.gq_lds, lane_m, sc.stb_lds, b.add(wrow, lane_m), sc.KS_DK, atom.zero_acc(b)
    )
    acc_o = sc.gemm(sc.aqb_lds, lane_m, sc.vn_lds, b.add(wrow, lane_m), sc.KS_C, acc_o)
    # Straight to HBM, no staging tile: a slot's column index is ``lane % 32``,
    # so one store instruction covers 32 consecutive v channels per half-wave
    # and is already coalesced.
    for i in range(CPL):
        row, col = sc.slot(i)
        b.global_store_vN(
            sc.o_ptr,
            b.add(
                cv_base,
                b.add(b.mul(row, b.const_i32(sc.EV)), b.add(wrow, col)),
            ),
            b.cast_f32_to(b.vec_extract(acc_o, i), ELEM),
            1,
        )

    # ---- S^T <- Diag(dec) S^T + V~^T Kt^T -------------------------------
    # dec is the whole-chunk decay per k channel, i.e. the state's column
    # index, so it is read once per accumulator slot.
    new_state = []
    for ti in range(sc.NS):
        scaled = []
        for i in range(CPL):
            _, col = sc.slot(i)
            d = b.vec_extract(
                b.smem_load_vN(
                    sc.dec_lds,
                    b.add(b.const_i32(ti * atom.n), col),
                    dtype=F32,
                    n=1,
                ),
                0,
            )
            if sc.dec_is_log:
                d = sc.ex2(d)
            scaled.append(b.fmul(b.vec_extract(state[ti], i), d))
        acc = sc.gemm(
            sc.vn_lds,
            b.add(wrow, lane_m),
            sc.kt_lds,
            b.add(b.const_i32(ti * atom.n), lane_m),
            sc.KS_C,
            b.vec_pack(scaled, F32),
        )
        new_state.append(acc)
    b.sync()
    sc.publish_state(new_state)
    b.sync()
    return new_state


def build_kda_chunk_fused(spec: KdaChunkFusedSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for the fused chunkwise KDA forward.

    Kernel signature::

        (q, k: ptr<bf16>,      # [NT, C * DK]   NT = BH * NC
         g:    ptr<f32>,       # [NT, C * DK]   per-channel log decay
         beta: ptr<f32>,       # [NT, C]
         v:    ptr<bf16>,      # [NT, C * DV]
         o:    ptr<bf16>,      # [NT, C * DV]
         h0:   ptr<f32>,       # [BH, DV, DK]   S^T, read iff has_initial_state
         ht:   ptr<f32>,       # [BH, DV, DK]   S^T, written iff store_final_state
         scale: f32,
         nc:    i32)           # chunks per (batch, head)

    Grid ``(BH, 1, 1)``; block ``(block_size, 1, 1)``.

    Per chunk the body is the tile emission followed by

    .. code-block:: text

        Z^T  = S^T GK^T                    EV x C
        R^T  = V^T - Z^T                   EV x C   (in-register, see below)
        V~^T = R^T A^T                     EV x C
        O    = GQ S + Aqk V~               C x EV
        S^T <- Diag(dec) S^T + V~^T Kt^T   EV x DK

    Working transposed is what keeps this cheap: every product is then
    ``A B^T`` with the contraction on the fastest axis, so no operand ever
    needs an LDS transpose. ``R^T`` never reaches LDS as fp32 either -- the
    accumulator's lane mapping puts each lane's 16 slots at one chunk row and
    four runs of four consecutive v channels, so ``V`` is subtracted straight
    into the accumulator with four short vector loads per lane.
    """
    ok, why = is_valid_fused_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid kda_chunk_fused spec for {arch}: {why}")

    prep = spec.prep
    t = spec.tile
    C, DK, EV = t.chunk, spec.head_k, spec.head_v
    BLOCK = t.block_size
    PDK, PCB = DK + t.pad_dk, C + t.pad_cb
    ELEM = _DTYPE_IR[spec.dtype]
    atom = spec.atom

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BLOCK
    if t.waves_per_eu:
        b.kernel.attrs["waves_per_eu"] = (t.waves_per_eu, t.waves_per_eu)

    q_ptr = b.param("q_ptr", PtrType(ELEM, "global"), readonly=True, align=16)
    k_ptr = b.param("k_ptr", PtrType(ELEM, "global"), readonly=True, align=16)
    g_ptr = b.param("g_ptr", PtrType(F32, "global"), readonly=True, align=16)
    beta_ptr = b.param("beta_ptr", PtrType(F32, "global"), readonly=True, align=4)
    v_ptr = b.param("v_ptr", PtrType(ELEM, "global"), readonly=True, align=16)
    o_ptr = b.param("o_ptr", PtrType(ELEM, "global"), writeonly=True, align=16)
    h0_ptr = b.param("h0_ptr", PtrType(F32, "global"), readonly=True, align=16)
    ht_ptr = b.param("ht_ptr", PtrType(F32, "global"), writeonly=True, align=16)
    scale = b.param("scale", F32)
    nc = b.param("nc", I32)

    ctx = _ChunkCtx(b, prep, (q_ptr, k_ptr, g_ptr, beta_ptr, scale))
    kt_lds = b.smem_alloc(ELEM, [DK, PCB], "kt_s")
    stb_lds = b.smem_alloc(ELEM, [EV, PDK], "stb_s")
    vn_lds = b.smem_alloc(ELEM, [EV, PCB], "vn_s")
    sink = _LdsTileSink(ctx.x_lds, ctx.xq_lds, ctx.tb_lds, ctx.zs_lds, kt_lds)

    sc = _ScanCtx(
        b,
        atom=atom,
        chunk=C,
        head_k=DK,
        head_v=EV,
        block_size=BLOCK,
        elem=ELEM,
        tiles=(
            ctx.x_lds,  # GK  overwrites the X staging tile
            ctx.xq_lds,  # GQ  overwrites the XQ staging tile
            ctx.tb_lds,  # A   overwrites the solve's rank-update operand
            ctx.zs_lds,  # Aqk overwrites the solved-so-far tile
            kt_lds,
            ctx.gl_lds,  # the whole-chunk decay, still in the log domain
        ),
        stb_lds=stb_lds,
        vn_lds=vn_lds,
        v_ptr=v_ptr,
        o_ptr=o_ptr,
        tid=ctx.tid,
        lane=ctx.lane,
        lane_m=ctx.lane_m,
        frag_k_off=ctx.frag_k_off,
        dec_is_log=True,
        ex2=ctx.ex2,
    )
    sink = _LdsTileSink(*sc.tiles_for_sink())

    bh = b.block_id_x()
    state_base = b.mul(bh, b.const_i32(EV * DK))
    s_init = (
        sc.load_state(h0_ptr, state_base) if spec.has_initial_state else sc.zero_state()
    )
    sc.publish_state(s_init)
    b.sync()

    loop = b.scf_for_iter(
        b.const_i32(0),
        nc,
        b.const_i32(1),
        [(f"s{ti}", s_init[ti]) for ti in range(sc.NS)],
        iv_name="chunk",
        elide_trailing_barrier=False,
    )
    with loop as (n, carried):
        tile = b.add(b.mul(bh, nc), n)
        _emit_chunk_tiles(ctx, tile, sink)
        b.sync()
        b.scf_yield(*_emit_scan_body(sc, list(carried), tile))

    if spec.store_final_state:
        sc.store_state(ht_ptr, state_base, loop.results)

    b.ret()
    return b.kernel


def kda_chunk_fused_grid(spec: KdaChunkFusedSpec, bh: int) -> Tuple[int, int, int]:
    """One workgroup per (batch, head)."""
    return (int(bh), 1, 1)


def kda_chunk_fused_signature(spec: KdaChunkFusedSpec):
    return (
        SignatureBuilder()
        .ptr("q_ptr", spec.dtype)
        .ptr("k_ptr", spec.dtype)
        .ptr("g_ptr", "f32")
        .ptr("beta_ptr", "f32")
        .ptr("v_ptr", spec.dtype)
        .ptr("o_ptr", spec.dtype)
        .ptr("h0_ptr", "f32")
        .ptr("ht_ptr", "f32")
        .scalar("scale", "f32")
        .scalar("nc", "i32")
        .build()
    )


@dataclass(frozen=True)
class KdaChunkScanSpec:
    """Compile-time spec for the standalone state scan over materialized tiles.

    The second half of the split path: :func:`build_kda_chunk_prep` writes the
    six per-chunk tiles to HBM, and this kernel walks one (batch, head)'s chunks
    in order, staging each chunk's tiles into LDS and running the same
    recurrence the fused kernel runs in registers.

    Fusing the two would remove that HBM round trip entirely, which sounds
    strictly better and is not: holding the tile builder's staging tiles and the
    scan's operands live at once puts the workgroup over half the LDS budget, so
    only one fits per CU. The scan is a latency-bound chain of small matmuls, and
    at one workgroup per CU there is no second workgroup to cover it. Paying for
    the tile traffic to keep two resident is the cheaper side of that trade,
    which is why ``lds_bytes`` here is checked against *half* the budget rather
    than all of it -- occupancy is the spec's binding constraint, not capacity.
    """

    head_k: int = 128
    head_v: int = 128
    dtype: str = "bf16"
    tile: KdaTileSpec = KdaTileSpec()
    has_initial_state: bool = False
    store_final_state: bool = True
    # Workgroups resident per CU that the LDS request must leave room for.
    min_occupancy: int = 2
    name: str = "rocke_kda_chunk_scan"

    @property
    def atom(self) -> MfmaAtom:
        return KdaChunkPrepSpec(
            head_k=self.head_k,
            head_v=self.head_v,
            dtype=self.dtype,
            tile=self.tile,
        ).atom

    @property
    def state_tiles(self) -> int:
        """Atom tiles per wave across the state's ``DK`` extent."""
        return self.head_k // self.atom.n

    def lds_bytes(self) -> int:
        """The six staged tiles plus the state mirror and ``V~``.

        Every tile is staged in the exact layout its consumer wants as an MFMA
        operand, so staging is a straight copy and the scan body is identical to
        the fused one. Unlike the fused kernel there is nothing to overlap them
        with, so each is its own allocation -- which is still the smaller
        footprint, because none of the tile builder's staging tiles exist here.
        """
        t = self.tile
        C, DK, EV = t.chunk, self.head_k, self.head_v
        PDK, PCB = DK + t.pad_dk, C + t.pad_cb
        return (
            2 * C * PDK  # gk_s   bf16 (C x DK)
            + 2 * C * PDK  # gq_s  bf16 (C x DK)
            + 2 * C * PCB  # a_s   bf16 (C x C)
            + 2 * C * PCB  # aqk_s bf16 (C x C)
            + 2 * DK * PCB  # kt_s bf16 (DK x C)
            + 2 * EV * PDK  # stb_s bf16 mirror of S^T
            + 2 * EV * PCB  # vn_s  bf16 (EV x C)
            + 4 * DK  # dec_s   fp32 (DK)
        )

    def kernel_name(self) -> str:
        t = self.tile
        return kernel_name_join(
            self.name,
            f"dk{self.head_k}",
            f"dv{self.head_v}",
            self.dtype,
            f"c{t.chunk}",
            f"b{t.block_size}",
        )


def is_valid_scan_spec(
    spec: KdaChunkScanSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a standalone scan spec on ``arch``."""
    if arch != "gfx950":
        return False, f"unsupported arch {arch}"
    if spec.dtype not in _DTYPE_IR:
        return False, f"unsupported dtype {spec.dtype}"

    atom = spec.atom
    t = spec.tile
    # Same single partitioning rule as the fused scan: one atom-row band of the
    # state per wave, covering the v extent exactly. Anything else needs a
    # second rule for the same accumulator, which is how cross-wave reductions
    # creep in.
    waves = t.num_waves
    if spec.head_v != atom.m * waves:
        return False, (
            f"head_v ({spec.head_v}) must equal atom.m * waves "
            f"({atom.m} * {waves} = {atom.m * waves}); each wave owns one "
            "row band of the state"
        )
    if spec.head_k % atom.n:
        return False, (
            f"head_k ({spec.head_k}) must be a multiple of the atom N extent "
            f"({atom.n})"
        )
    if t.chunk % atom.k:
        return False, (
            f"chunk ({t.chunk}) must be a multiple of the MFMA K step "
            f"({atom.k}); it is the contraction extent of three of the products"
        )
    # Staging is 128-bit per thread throughout, so both padded row pitches have
    # to keep 8-element alignment and each tile has to divide evenly across the
    # workgroup's 8-element slots.
    if (spec.head_k + t.pad_dk) % 8:
        return False, (
            f"padded head_k pitch ({spec.head_k + t.pad_dk}) must be a multiple "
            "of 8 elements; the staging copies are ds_write_b128"
        )
    if (t.chunk + t.pad_cb) % 8:
        return False, (
            f"padded chunk pitch ({t.chunk + t.pad_cb}) must be a multiple of 8 "
            "elements; the staging copies are ds_write_b128"
        )
    for what, n in (
        ("C x DK", t.chunk * spec.head_k),
        ("C x C", t.chunk * t.chunk),
        ("DK x C", spec.head_k * t.chunk),
    ):
        if n % 8:
            return False, (
                f"{what} tile ({n} elements) must be a multiple of 8; staging "
                "moves one 128-bit slot per thread"
            )
    if spec.head_k % 4:
        return False, (
            f"dec tile ({spec.head_k}) must be a multiple of 4; it is staged as "
            "one fp32 4-vector per thread"
        )

    lds = spec.lds_bytes()
    budget = LDS_LIMIT // spec.min_occupancy
    if lds > budget:
        return False, (
            f"LDS request {lds} B exceeds the {budget} B budget for "
            f"{spec.min_occupancy} workgroups per CU; the split path only pays "
            "for its tile traffic if it keeps that occupancy"
        )
    return True, "ok"


def build_kda_chunk_scan(spec: KdaChunkScanSpec, arch: str = "gfx950") -> "KernelDef":
    """Serial state scan over per-chunk tiles already materialized in HBM.

    One workgroup per (batch, head) walks that head's chunks in order. Per
    chunk it stages the six tiles from HBM into LDS and runs
    :func:`_emit_scan_body`, so the recurrence is bit-for-bit the fused
    kernel's -- the only difference is where the operands came from.

    The state stays in accumulators across the whole loop, so the sequence
    length costs nothing in registers, and ``dec`` arrives already
    exponentiated (the tile builder stored it that way), which is the one place
    this body diverges from the fused one.
    """
    ok, why = is_valid_scan_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid kda_chunk_scan spec for {arch}: {why}")

    t = spec.tile
    C, DK, EV = t.chunk, spec.head_k, spec.head_v
    BLOCK = t.block_size
    PDK, PCB = DK + t.pad_dk, C + t.pad_cb
    ELEM = _DTYPE_IR[spec.dtype]
    atom = spec.atom

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BLOCK
    if t.waves_per_eu:
        b.kernel.attrs["waves_per_eu"] = (t.waves_per_eu, t.waves_per_eu)

    a_ptr = b.param("a_ptr", PtrType(ELEM, "global"), readonly=True, align=16)
    gk_ptr = b.param("gk_ptr", PtrType(ELEM, "global"), readonly=True, align=16)
    gq_ptr = b.param("gq_ptr", PtrType(ELEM, "global"), readonly=True, align=16)
    aqk_ptr = b.param("aqk_ptr", PtrType(ELEM, "global"), readonly=True, align=16)
    kt_ptr = b.param("kt_ptr", PtrType(ELEM, "global"), readonly=True, align=16)
    dec_ptr = b.param("dec_ptr", PtrType(F32, "global"), readonly=True, align=16)
    v_ptr = b.param("v_ptr", PtrType(ELEM, "global"), readonly=True, align=16)
    o_ptr = b.param("o_ptr", PtrType(ELEM, "global"), writeonly=True, align=16)
    h0_ptr = b.param("h0_ptr", PtrType(F32, "global"), readonly=True, align=16)
    ht_ptr = b.param("ht_ptr", PtrType(F32, "global"), writeonly=True, align=16)
    nc = b.param("nc", I32)

    gk_lds = b.smem_alloc(ELEM, [C, PDK], "gk_s")
    gq_lds = b.smem_alloc(ELEM, [C, PDK], "gq_s")
    ab_lds = b.smem_alloc(ELEM, [C, PCB], "a_s")
    aqb_lds = b.smem_alloc(ELEM, [C, PCB], "aqk_s")
    kt_lds = b.smem_alloc(ELEM, [DK, PCB], "kt_s")
    dec_lds = b.smem_alloc(F32, [DK], "dec_s")
    stb_lds = b.smem_alloc(ELEM, [EV, PDK], "stb_s")
    vn_lds = b.smem_alloc(ELEM, [EV, PCB], "vn_s")

    tid = b.thread_id_x()
    lane = b.mod(tid, b.const_i32(64))
    lane_m = b.mod(lane, b.const_i32(32))
    frag_k_off = b.mul(b.div(lane, b.const_i32(32)), b.const_i32(8))

    sc = _ScanCtx(
        b,
        atom=atom,
        chunk=C,
        head_k=DK,
        head_v=EV,
        block_size=BLOCK,
        elem=ELEM,
        tiles=(gk_lds, gq_lds, ab_lds, aqb_lds, kt_lds, dec_lds),
        stb_lds=stb_lds,
        vn_lds=vn_lds,
        v_ptr=v_ptr,
        o_ptr=o_ptr,
        tid=tid,
        lane=lane,
        lane_m=lane_m,
        frag_k_off=frag_k_off,
        dec_is_log=False,
    )

    bh = b.block_id_x()
    state_base = b.mul(bh, b.const_i32(EV * DK))
    s_init = (
        sc.load_state(h0_ptr, state_base) if spec.has_initial_state else sc.zero_state()
    )
    sc.publish_state(s_init)
    b.sync()

    def stage(src, dst, rows, cols, base):
        """One flat ``rows x cols`` HBM tile into its padded LDS tile.

        Both sides are 128-bit: the source row length is a multiple of 8, so a
        thread's eight consecutive elements never straddle a row and the only
        difference between the two addresses is the destination's pad. A tile
        smaller than one workgroup sweep (the ``C x C`` pair, at half) just
        leaves the upper threads idle rather than giving them a second, narrower
        access pattern.
        """
        n_slot = rows * cols // 8
        for i in range(max(1, n_slot // BLOCK)):
            vidx = b.add(tid, b.const_i32(i * BLOCK))
            guard = (
                nullcontext()
                if n_slot >= BLOCK
                else b.scf_if(b.cmp_gt(b.const_i32(n_slot), vidx))
            )
            with guard:
                off = b.mul(vidx, b.const_i32(8))
                b.smem_store_vN(
                    dst,
                    [b.div(off, b.const_i32(cols)), b.mod(off, b.const_i32(cols))],
                    b.global_load_vN(src, b.add(base, off), ELEM, 8),
                    8,
                )

    loop = b.scf_for_iter(
        b.const_i32(0),
        nc,
        b.const_i32(1),
        [(f"s{ti}", s_init[ti]) for ti in range(sc.NS)],
        iv_name="chunk",
        elide_trailing_barrier=False,
    )
    with loop as (n, carried):
        tile = b.add(b.mul(bh, nc), n)
        cd = b.mul(tile, b.const_i32(C * DK))
        cc = b.mul(tile, b.const_i32(C * C))
        stage(gk_ptr, gk_lds, C, DK, cd)
        stage(gq_ptr, gq_lds, C, DK, cd)
        stage(kt_ptr, kt_lds, DK, C, cd)
        stage(a_ptr, ab_lds, C, C, cc)
        stage(aqk_ptr, aqb_lds, C, C, cc)
        with b.scf_if(b.cmp_gt(b.const_i32(DK // 4), tid)):
            col4 = b.mul(tid, b.const_i32(4))
            b.smem_store_vN(
                dec_lds,
                [col4],
                b.global_load_vN(
                    dec_ptr, b.add(b.mul(tile, b.const_i32(DK)), col4), F32, 4
                ),
                4,
            )
        b.sync()
        b.scf_yield(*_emit_scan_body(sc, list(carried), tile))

    if spec.store_final_state:
        sc.store_state(ht_ptr, state_base, loop.results)

    b.ret()
    return b.kernel


def kda_chunk_scan_grid(spec: KdaChunkScanSpec, bh: int) -> Tuple[int, int, int]:
    """One workgroup per (batch, head)."""
    return (int(bh), 1, 1)


def kda_chunk_scan_signature(spec: KdaChunkScanSpec):
    return (
        SignatureBuilder()
        .ptr("a_ptr", spec.dtype)
        .ptr("gk_ptr", spec.dtype)
        .ptr("gq_ptr", spec.dtype)
        .ptr("aqk_ptr", spec.dtype)
        .ptr("kt_ptr", spec.dtype)
        .ptr("dec_ptr", "f32")
        .ptr("v_ptr", spec.dtype)
        .ptr("o_ptr", spec.dtype)
        .ptr("h0_ptr", "f32")
        .ptr("ht_ptr", "f32")
        .scalar("nc", "i32")
        .build()
    )


def kda_chunk_prep_grid(spec: KdaChunkPrepSpec, num_tiles: int) -> Tuple[int, int, int]:
    """One workgroup per chunk. ``num_tiles = BH * NC``."""
    return (int(num_tiles), 1, 1)


def kda_chunk_prep_signature(spec: KdaChunkPrepSpec):
    return (
        SignatureBuilder()
        .ptr("q_ptr", spec.dtype)
        .ptr("k_ptr", spec.dtype)
        .ptr("g_ptr", "f32")
        .ptr("beta_ptr", "f32")
        .ptr("a_ptr", spec.dtype)
        .ptr("gk_ptr", spec.dtype)
        .ptr("gq_ptr", spec.dtype)
        .ptr("aqk_ptr", spec.dtype)
        .ptr("kt_ptr", spec.dtype)
        .ptr("dec_ptr", "f32")
        .scalar("scale", "f32")
        .build()
    )


__all__ = [
    "EXP2_CLAMP",
    "KdaChunkFusedSpec",
    "KdaChunkPrepSpec",
    "KdaChunkScanSpec",
    "KdaTileSpec",
    "LOG2E",
    "build_kda_chunk_fused",
    "build_kda_chunk_prep",
    "build_kda_chunk_scan",
    "is_valid_fused_spec",
    "is_valid_scan_spec",
    "is_valid_spec",
    "kda_chunk_fused_grid",
    "kda_chunk_fused_signature",
    "kda_chunk_prep_grid",
    "kda_chunk_prep_signature",
    "kda_chunk_scan_grid",
    "kda_chunk_scan_signature",
]
