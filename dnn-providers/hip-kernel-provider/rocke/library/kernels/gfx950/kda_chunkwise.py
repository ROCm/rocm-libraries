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
    S  = Diag(dec) S + Kt Vt

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

Synchronization
---------------
Every phase boundary here is an LDS-visibility rendezvous, so all of them use
``sync_lds_only`` (``lgkmcnt(0)`` + ``s_barrier``) rather than the full
``sync``. There is no direct-to-LDS path in this kernel: a global load always
lands in a register and reaches LDS through a ``ds_write``, so waiting on
``lgkmcnt`` transitively waits on the load that feeds it. The global *stores*
are kernel outputs, consumed across a dispatch boundary rather than across one
of these barriers.

This is the minimum legal wait, not a faster one. Dropping ``vmcnt(0)`` off all
14 barriers is exactly neutral end to end, and ATT says why: the stall is
conserved, moving from the vmcnt bucket (22.3% -> 18.7% of total) into the lgkm
bucket (23.7% -> 27.5%) with the barrier bucket and the total both unchanged.
The drain was never redundant work -- it was the same load dependency, charged
to a different counter. Cheapening these waits is therefore not a lever on this
kernel; the barrier waits on wave arrival, and wave arrival waits on the loads
either way.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Tuple

from rocke.core.ir import BF16, F32, I8, I32, IRBuilder, KernelDef, PtrType
from rocke.helpers.activations import _sigmoid_via_exp2
from rocke.helpers.atoms import MfmaAtom
from rocke.helpers.spec import SignatureBuilder, kernel_name_join

_RAW_VALUE_SPLITS = (1, 2, 4, 8)

# Cumulative log decay is kept in the log2 domain so exp2 is a bare v_exp_f32.
LOG2E = 1.4426950408889634
# v_exp_f32 saturates past this; the clamp keeps a saturated gate finite instead
# of turning a whole chunk into NaN.
EXP2_CLAMP = 126.0

_DTYPE_IR = {"bf16": BF16}
# Declared coverage, exported so a dispatch candidate can state what it serves
# by importing these rather than transcribing them. The validators below read
# the same names, so the two cannot drift apart.
KDA_DTYPES: Tuple[str, ...] = tuple(sorted(_DTYPE_IR))
# The only chunk lengths the emitted tile-atom schedules cover.
KDA_CHUNK_SIZES: Tuple[int, ...] = (16, 32)
# gfx950 LDS budget per workgroup.
LDS_LIMIT = 160 * 1024
# The K-packed bf16 atoms, by M/N extent. Both are gfx950 shapes; the K extent
# is whatever that shape packs, since the scan's contractions are all K-stepped.
_SCAN_ATOMS = {
    32: MfmaAtom.bf16_32x32x16,
    16: MfmaAtom.bf16_16x16x32,
}


class _LdsTile:
    """A 2-D (or 1-D) view onto a 1-D ``i8`` LDS pool.

    The fused kernel's tile phase and scan phase are mutually exclusive in
    time, but they live in one ``scf.for``, so the LDS packer's loop liveness
    will not overlay them. A single pool plus byte-offset views lets the
    scan-only ``S^T`` mirror and ``V~`` reuse tile-only storage explicitly.
    """

    __slots__ = ("pool", "off", "elem_bytes", "cols", "rank")

    def __init__(self, pool, off: int, elem, shape: Tuple[int, ...]):
        self.pool = pool
        self.off = off
        self.elem_bytes = 4 if elem.name == "f32" else 2
        self.rank = len(shape)
        self.cols = int(shape[-1]) if shape else 1

    def index(self, b: IRBuilder, *idx):
        if self.rank == 1:
            (col,) = idx
            return b.add(
                b.const_i32(self.off), b.mul(col, b.const_i32(self.elem_bytes))
            )
        row, col = idx
        return b.add(
            b.const_i32(self.off),
            b.add(
                b.mul(row, b.const_i32(self.cols * self.elem_bytes)),
                b.mul(col, b.const_i32(self.elem_bytes)),
            ),
        )


def _ld(b: IRBuilder, smem, *idx, dtype, n: int):
    """Vector LDS load that accepts either a typed alloc or an ``_LdsTile``."""
    if isinstance(smem, _LdsTile):
        return b.smem_load_vN(smem.pool, smem.index(b, *idx), dtype=dtype, n=n)
    return b.smem_load_vN(smem, *idx, dtype=dtype, n=n)


def _st(b: IRBuilder, smem, *idx, value, n: int) -> None:
    """Vector LDS store that accepts either a typed alloc or an ``_LdsTile``."""
    if isinstance(smem, _LdsTile):
        b.smem_store_vN(smem.pool, [smem.index(b, *idx)], value, n)
    else:
        b.smem_store_vN(smem, list(idx), value, n)


def _fused_lds_layout(head_k: int, head_v: int, tile: "KdaTileSpec") -> dict:
    """Byte offsets of every fused LDS tile inside one pool.

    Three groups have different lifetimes:

    * V/Y, Kt, and GK/GQ/A/Aqk/dec survive from the tile phase into the scan.
      Kt must be dedicated even though the scan consumes it later: helper waves
      produce it during the solve while ``m/a/qk`` are still live.
    * g/k/q/m/a/qk/beta are tile-only once the post-solve copies finish.
    * the ``S^T`` mirror and ``V~`` are scan-only. Together they fit inside the
      tile-only span and reuse it exactly after the tile/scan rendezvous.

    Default-shape accounting::

        V/Y + Kt + GK/GQ/A/Aqk/dec      41,984 B
        tile-only region                 46,720 B
        S^T mirror + V~ (overlaid)       45,056 B
        pool                              88,704 B

    This is down from 133,760 B. It is intentionally not the tempting 78,464-B
    layout: that version overlapped both V and Kt before their last use and
    produced NaNs.

    The barrier-free next-input prefetch is also incompatible with this layout:
    it writes g/k/q/beta during the scan, exactly while the overlaid scan extras
    occupy those addresses. ``is_valid_fused_spec`` rejects that combination.
    """
    C, DK, EV = tile.chunk, head_k, head_v
    pdk, pc, pcb = DK + tile.pad_dk, C + tile.pad_c, C + tile.pad_cb
    off = 0

    def take(n: int) -> int:
        nonlocal off
        o = off
        off += n
        return o

    # Live from tile production into the scan.
    lay = {
        "y": take(2 * C * pdk),
        "kt": take(2 * DK * pcb),
        "x": take(2 * C * pdk),
        "xq": take(2 * C * pdk),
        "tb": take(2 * C * pcb),
        "zs": take(2 * C * pcb),
        "gl": take(4 * DK),
    }

    # Tile-only, one contiguous reuse span.
    tile_only = off
    lay.update(
        {
            "g": take(4 * C * DK),
            "k": take(2 * C * DK),
            "q": take(2 * C * DK),
            "m": take(4 * C * pc),
            "a": take(4 * C * pc),
            "qk": take(4 * C * pc),
            "beta": take(4 * C),
        }
    )
    pool_end = off

    # Scan-only views reuse the tile-only span.
    stb_n, vn_n = 2 * EV * pdk, 2 * EV * pcb
    lay["stb"], lay["vn"] = tile_only, tile_only + stb_n
    if lay["vn"] + vn_n > pool_end:
        raise ValueError(
            f"scan-only tiles ({stb_n + vn_n} B) exceed reusable tile-only "
            f"region ({pool_end - tile_only} B)"
        )
    off = pool_end
    lay["pool"] = off
    return lay


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
    # M/N extent of the tile phase's C x C products. 32 keeps the original
    # one-atom tile; 16 maps the eight waves onto the four 16x16 inner panels
    # (two waves per panel) while preserving the algorithmic C=32 chunk.
    tile_atom_m: int = 32
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
    # M/N extent of the atom the *state scan* uses, which need not be the tile
    # phase's. The C x C tile products want an atom as wide as the chunk, but the
    # scan's partitioning rule is independent: one wave owns one ``atom.m``-row
    # band of S^T, so a narrower atom subtiles the state into shorter bands.
    # Twice as many waves then cover the v extent and each lane carries half as
    # many state accumulators -- which is the only way to fit the fused kernel's
    # loop-carried state under the 256-VGPR ceiling that two waves per SIMD
    # needs. 0 = reuse the chunk-wide atom.
    scan_atom_m: int = 0
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
        if self.pad_cb != 8:
            parts += (f"pcb{self.pad_cb}",)
        if self.scan_atom_m:
            parts += (f"sa{self.scan_atom_m}",)
        if self.tile_atom_m != 32:
            parts += (f"ta{self.tile_atom_m}",)
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
    # Default-off raw token-major path: inputs are [B,T,H,D] / [B,T,H] and the
    # producer may fuse the Aiter-equivalent q/k L2 norm, gate, and beta sigmoid.
    raw_inputs: bool = False
    # Raw beta has two separately compiled ABIs. False is the framework-native
    # BF16 path; True preserves direct FP32 input without a conversion.
    fp32_beta_dtype: bool = False
    fuse_qk_l2norm: bool = False
    fuse_gate: bool = False
    fuse_beta_sigmoid: bool = False
    has_dt_bias: bool = False
    lower_bound: float = -5.0

    @property
    def atom(self) -> MfmaAtom:
        """The bf16 atom used by the tile phase's C x C products."""
        make = _SCAN_ATOMS.get(self.tile.tile_atom_m)
        if make is None:
            raise ValueError(
                f"no bf16 tile atom with M/N extent {self.tile.tile_atom_m}; "
                f"have {sorted(_SCAN_ATOMS)}"
            )
        return make()

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
        parts = [
            self.name,
            f"dk{self.head_k}",
            f"dv{self.head_v}",
            self.dtype,
            *self.tile.name_parts(),
        ]
        if self.raw_inputs:
            parts.extend(("raw", "bfp32" if self.fp32_beta_dtype else "bbf16"))
            if self.fuse_qk_l2norm:
                parts.append("nl2")
            if self.fuse_gate:
                parts.append("fg")
            if self.fuse_beta_sigmoid:
                parts.append("fb")
            if self.has_dt_bias:
                parts.append("db")
            if self.lower_bound != -5.0:
                parts.append(f"lb{self.lower_bound:g}")
        return kernel_name_join(*parts)


def _scan_atom(tile: KdaTileSpec, default: MfmaAtom) -> MfmaAtom:
    """The atom the state scan runs on, per ``KdaTileSpec.scan_atom_m``."""
    if not tile.scan_atom_m:
        return default
    make = _SCAN_ATOMS.get(tile.scan_atom_m)
    if make is None:
        raise ValueError(
            f"no bf16 atom with M/N extent {tile.scan_atom_m}; "
            f"have {sorted(_SCAN_ATOMS)}"
        )
    return make()


def is_valid_spec(spec: KdaChunkPrepSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a prep spec on ``arch``."""
    if arch != "gfx950":
        return False, f"kda_chunk_prep is gfx950-only (got {arch})"
    if spec.dtype not in _DTYPE_IR:
        return False, f"unsupported dtype {spec.dtype!r} (bf16 only)"

    if spec.has_dt_bias and not spec.fuse_gate:
        return False, "has_dt_bias requires fuse_gate"
    if spec.fp32_beta_dtype and not spec.raw_inputs:
        return False, "fp32_beta_dtype requires raw_inputs=True"
    for flag, name in (
        (spec.fuse_qk_l2norm, "fuse_qk_l2norm"),
        (spec.fuse_gate, "fuse_gate"),
        (spec.fuse_beta_sigmoid, "fuse_beta_sigmoid"),
    ):
        if flag and not spec.raw_inputs:
            return False, f"{name} requires raw_inputs=True"
    if spec.raw_inputs:
        if not (spec.fuse_qk_l2norm or spec.fuse_gate or spec.fuse_beta_sigmoid):
            return False, (
                "raw_inputs requires at least one fused preprocessing flag "
                "(fuse_qk_l2norm, fuse_gate, fuse_beta_sigmoid)"
            )

    t = spec.tile
    if t.tile_atom_m not in _SCAN_ATOMS:
        return False, (
            f"unsupported tile_atom_m {t.tile_atom_m}; have {sorted(_SCAN_ATOMS)}"
        )
    atom = spec.atom
    if t.chunk not in KDA_CHUNK_SIZES:
        return False, (
            f"chunk must be one of {list(KDA_CHUNK_SIZES)} for the emitted "
            "tile-atom schedules"
        )
    if t.chunk % atom.m or t.chunk % atom.n:
        return False, (f"tile atom ({atom.m}x{atom.n}) must divide chunk ({t.chunk})")
    panels = (t.chunk // atom.m) * (t.chunk // atom.n)
    if t.num_waves % panels:
        return False, (
            f"waves ({t.num_waves}) must be a multiple of C x C panels "
            f"({panels}) so every panel has equal wave coverage"
        )
    if spec.head_k % atom.k:
        return False, (
            f"head_k ({spec.head_k}) must be a multiple of the MFMA K step ({atom.k})"
        )
    solve_atom = MfmaAtom.bf16_16x16x32() if t.chunk == 16 else MfmaAtom.bf16_32x32x16()
    if t.chunk + t.pad_cb < solve_atom.k:
        return False, (
            f"chunk + pad_cb ({t.chunk + t.pad_cb}) must cover solve MFMA K "
            f"({solve_atom.k}) so the C=16 tail can be zero padded"
        )
    # The cumulative sum gives one thread a whole (row-group, channel) column,
    # so the block has to be a whole number of copies of the channel extent, and
    # the chunk has to split evenly into that many groups.
    if t.block_size % spec.head_k or t.block_size < 2 * spec.head_k:
        return False, (
            f"block_size ({t.block_size}) must be a multiple of head_k "
            f"({spec.head_k}), at least 2x, for the grouped cumulative sum"
        )
    n_groups = t.block_size // spec.head_k
    if t.chunk % n_groups:
        return False, (
            f"chunk ({t.chunk}) must split evenly into the {n_groups} cumsum "
            f"row groups implied by block_size/head_k"
        )
    # The fold deals rows out round-robin across a channel's threads and folds a
    # compile-time set of group totals into each, which needs every step's rows
    # to land inside one group.
    if (t.chunk // n_groups) % n_groups:
        return False, (
            f"cumsum group height ({t.chunk // n_groups}) must be a multiple of "
            f"the group count ({n_groups}) so the fold's offsets stay uniform"
        )
    if t.solve_block % 8 or t.chunk % t.solve_block:
        return False, (
            f"solve_block ({t.solve_block}) must be a multiple of 8 and divide "
            f"chunk ({t.chunk})"
        )
    if t.block_size % t.wave_size:
        return False, f"block_size ({t.block_size}) must be a wave multiple"

    # Every global access is a 128-bit transaction; vector starts must stay
    # inside rows. C=16 leaves half the 512-thread block inactive in the bf16
    # staging sweep, which the emitter predicates explicitly.
    if spec.head_k % 8 or t.chunk % 8:
        return False, "head_k and chunk must be multiples of 8 for 128-bit access"
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


def _reduce16_fadd(b: IRBuilder, v):
    """Sum ``v`` across the 16-lane row subgroup (XOR butterfly)."""
    for mask in (1, 2, 4, 8):
        v = b.fadd(v, b.warp_shuffle_xor(v, mask))
    return v


def _l2norm_scale8(b: IRBuilder, vals8):
    """One in-register L2 norm over eight bf16-as-f32 values in a 16-lane group."""
    ss = b.const_f32(0.0)
    for x in vals8:
        ss = b.fadd(ss, b.fmul(x, x))
    ss = _reduce16_fadd(b, ss)
    return b.rsqrt(b.fadd(ss, b.const_f32(1e-6)))


def _apply_gate_fused(
    b: IRBuilder,
    raw_g,
    a_log,
    dt_bias,
    head,
    col,
    lower_bound,
    *,
    has_dt_bias: bool,
    dk: int,
):
    """``lower_bound * sigmoid(exp(A_log[h]) * (g + dt_bias[h,d]))``."""
    g = b.cast_to_f32(raw_g)
    if has_dt_bias:
        g = b.fadd(
            g,
            b.global_load_f32(dt_bias, b.add(b.mul(head, b.const_i32(dk)), col)),
        )
    a = b.global_load_f32(a_log, head)
    x = b.fmul(b.exp2(b.fmul(b.const_f32(LOG2E), a)), g)
    return b.fmul(b.const_f32(lower_bound), _sigmoid_via_exp2(b, x))


class _RawTokenAddr:
    """Token-major [B,T,H,D] / [B,T,H] addressing for one chunk tile."""

    def __init__(
        self,
        b: IRBuilder,
        *,
        heads,
        tseq,
        nc,
        chunk,
        dk,
        a_log=None,
        dt_bias=None,
        beta_stride_batch=None,
        beta_stride_token=None,
        beta_stride_head=None,
    ):
        self.b = b
        self.H = heads
        self.T = tseq
        self.nc = nc
        self.C = chunk
        self.DK = dk
        self.a_log = a_log
        self.dt_bias = dt_bias
        self.stride_token_qk = b.mul(heads, b.const_i32(dk))
        self.stride_batch_qk = b.mul(tseq, self.stride_token_qk)
        self.stride_head_beta = (
            b.const_i32(1) if beta_stride_head is None else beta_stride_head
        )
        self.stride_token_beta = (
            heads if beta_stride_token is None else beta_stride_token
        )
        self.stride_batch_beta = (
            b.mul(tseq, heads) if beta_stride_batch is None else beta_stride_batch
        )

    def _parts(self, tile, row):
        b = self.b
        bh = b.div(tile, self.nc)
        chunk_n = b.mod(tile, self.nc)
        batch = b.div(bh, self.H)
        head = b.mod(bh, self.H)
        token = b.add(b.mul(chunk_n, b.const_i32(self.C)), row)
        return batch, head, token

    def qk_off(self, tile, row, col):
        b = self.b
        batch, head, token = self._parts(tile, row)
        return b.add(
            b.add(
                b.mul(batch, self.stride_batch_qk),
                b.mul(token, self.stride_token_qk),
            ),
            b.add(b.mul(head, b.const_i32(self.DK)), col),
        )

    def beta_off(self, tile, row):
        b = self.b
        batch, head, token = self._parts(tile, row)
        return b.add(
            b.add(
                b.mul(batch, self.stride_batch_beta),
                b.mul(token, self.stride_token_beta),
            ),
            b.mul(head, self.stride_head_beta),
        )

    def v_off(self, tile, chunk_row, ev, ev_dim):
        b = self.b
        batch, head, token = self._parts(tile, chunk_row)
        stride_token = b.mul(self.H, b.const_i32(ev_dim))
        stride_batch = b.mul(self.T, stride_token)
        return b.add(
            b.add(
                b.mul(batch, stride_batch),
                b.mul(token, stride_token),
            ),
            b.add(b.mul(head, b.const_i32(ev_dim)), ev),
        )


class _ScanTokenIo:
    """Token-major v/o addressing with batch/head bound once per scan WG.

    The standalone scan owns one (batch, head) for the whole recurrence, so the
    per-chunk tile index only needs to contribute ``chunk_n * C + row`` to the
    token coordinate.  Hoisting ``batch`` and ``head`` out of the inner loop
    removes two div/mod pairs from every V/O access.
    """

    def __init__(
        self,
        b: IRBuilder,
        *,
        heads,
        tseq,
        chunk: int,
        ev_stride: int,
        batch,
        head,
        v_row_base,
    ):
        self.b = b
        self.C = chunk
        self.stride_token = b.mul(heads, b.const_i32(ev_stride))
        self.stride_batch = b.mul(tseq, self.stride_token)
        self.base = b.add(
            b.mul(batch, self.stride_batch),
            b.mul(head, b.const_i32(ev_stride)),
        )
        self.v_row_base = v_row_base

    def v_off(self, chunk_n, chunk_row, gev):
        b = self.b
        token = b.add(b.mul(chunk_n, b.const_i32(self.C)), chunk_row)
        return b.add(self.base, b.add(b.mul(token, self.stride_token), gev))


class _ChunkCtx:
    """Everything the per-chunk tile emitter needs that does not depend on
    *which* chunk is being built: the LDS tiles, the thread/lane decomposition,
    and the shared access helpers.

    Split out from the emitter so a kernel that walks many chunks in one
    workgroup pays for this setup once, outside its loop, while the per-chunk
    work stays a single reusable emission.
    """

    def __init__(
        self, b: IRBuilder, spec: KdaChunkPrepSpec, inputs, *, overlay: bool = False
    ):
        t = spec.tile
        self.b = b
        self.spec = spec
        self.q_ptr, self.k_ptr, self.g_ptr, self.beta_ptr, self.scale = inputs[:5]
        self.raw = inputs[5] if len(inputs) > 5 else None
        self.C = C = t.chunk
        self.DK = DK = spec.head_k
        self.BLOCK = t.block_size
        self.PDK = DK + t.pad_dk
        self.PC = C + t.pad_c
        self.PCB = C + t.pad_cb
        self.CREF = C // 2
        self.HALF = C // 2
        # Row groups for the cumulative sum: one thread owns a whole
        # (row-group, channel) column, so the number of groups is however many
        # copies of the channel extent the block holds.
        self.NG = self.BLOCK // DK
        self.GROUP = C // self.NG
        self.ELEM = _DTYPE_IR[spec.dtype]
        self.atom = spec.atom
        self.N_CD = C * DK
        ELEM = self.ELEM

        # ---- LDS ----
        if overlay:
            self._init_overlay_lds()
        else:
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
            self.kt_lds = self.stb_lds = self.vn_lds = None

        self.tid = b.thread_id_x()
        self.lane = lane = b.mod(self.tid, b.const_i32(64))
        self.lane_m = b.mod(lane, b.const_i32(32))
        self.lane_h = lane_h = b.div(lane, b.const_i32(32))
        self.frag_k_off = b.mul(lane_h, b.const_i32(8))

        self.c_clamp = b.const_f32(-EXP2_CLAMP)
        self.c_log2e = b.const_f32(LOG2E)
        self.c_clamp_hi = b.const_f32(EXP2_CLAMP)

    def _init_overlay_lds(self) -> None:
        """One i8 pool; tile-only buffers and scan extras share the dead region."""
        b, spec, t = self.b, self.spec, self.spec.tile
        C, DK, EV = self.C, self.DK, spec.head_v
        ELEM = self.ELEM
        lay = _fused_lds_layout(DK, EV, t)
        pool = b.smem_alloc(I8, [lay["pool"]], "lds_pool")

        def view(key, elem, shape):
            return _LdsTile(pool, lay[key], elem, shape)

        self.g_lds = view("g", F32, (C, DK))
        self.k_lds = view("k", ELEM, (C, DK))
        self.q_lds = view("q", ELEM, (C, DK))
        self.y_lds = view("y", ELEM, (C, self.PDK))
        self.m_lds = view("m", F32, (C, self.PC))
        self.a_lds = view("a", F32, (C, self.PC))
        self.qk_lds = view("qk", F32, (C, self.PC))
        self.beta_lds = view("beta", F32, (C,))
        self.x_lds = view("x", ELEM, (C, self.PDK))
        self.xq_lds = view("xq", ELEM, (C, self.PDK))
        self.tb_lds = view("tb", ELEM, (C, self.PCB))
        self.zs_lds = view("zs", ELEM, (C, self.PCB))
        self.gl_lds = view("gl", F32, (DK,))
        self.stb_lds = view("stb", ELEM, (EV, self.PDK))
        self.kt_lds = view("kt", ELEM, (DK, self.PCB))
        self.vn_lds = view("vn", ELEM, (EV, self.PCB))

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
        return self.b.vec_extract(_ld(self.b, smem, *idx, dtype=dtype, n=1), 0)

    def lds_get8_f32(self, smem, row, col):
        """Eight consecutive fp32 from LDS as two ``ds_read_b128``."""
        b = self.b
        out = []
        for h in range(2):
            v = _ld(b, smem, row, b.add(col, b.const_i32(4 * h)), dtype=F32, n=4)
            out += [b.vec_extract(v, j) for j in range(4)]
        return out

    def lds_get8_elem(self, smem, row, col):
        """Eight consecutive bf16 from LDS as one ``ds_read_b128``, as f32."""
        b = self.b
        v = _ld(b, smem, row, col, dtype=self.ELEM, n=8)
        return [b.cast_to_f32(b.vec_extract(v, j)) for j in range(8)]

    def lds_put8(self, smem, row, col, vals_f32):
        """Eight f32 truncated to bf16 and written as one ``ds_write_b128``."""
        b = self.b
        _st(
            b,
            smem,
            row,
            col,
            value=b.vec_pack(
                [b.cast_f32_to(v, self.ELEM) for v in vals_f32], self.ELEM
            ),
            n=8,
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
    for p in range((N_CD + 8 * BLOCK - 1) // (8 * BLOCK)):
        vidx = b.add(tid, b.const_i32(p * BLOCK))
        guard = (
            b.scf_if(b.cmp_gt(b.const_i32(N_CD // 8), vidx))
            if N_CD % (8 * BLOCK)
            else nullcontext()
        )
        with guard:
            row = b.add(ew_row, b.const_i32(p * ew_rstep))
            off = b.mul(vidx, b.const_i32(8))
            g8 = ctx.lds_get8_f32(ctx.g_lds, row, ew_col)
            k8 = ctx.lds_get8_elem(ctx.k_lds, row, ew_col)
            q8 = ctx.lds_get8_elem(ctx.q_lds, row, ew_col)
            gk8, gq8 = [], []
            for j in range(8):
                eg = ctx.ex2(g8[j])
                gk8.append(b.fmul(k8[j], eg))
                gq8.append(b.fmul(b.fmul(q8[j], scale), eg))
            sink.gk_gq(ctx, ch, row, ew_col, off, gk8, gq8)


def _emit_kt_slot(ctx: _ChunkCtx, ch, sink, vidx) -> None:
    """One thread's eight-row Kt column at flat slot ``vidx``."""
    b, C, ELEM = ctx.b, ctx.C, ctx.ELEM
    off = b.mul(vidx, b.const_i32(8))
    dch = b.div(off, b.const_i32(C))
    r8 = b.mod(off, b.const_i32(C))
    glc = ctx.lds_get(ctx.gl_lds, [dch])
    vals = []
    for j in range(8):
        rj = b.add(r8, b.const_i32(j))
        kv = b.cast_to_f32(ctx.lds_get(ctx.k_lds, [rj, dch], dtype=ELEM))
        gc = ctx.lds_get(ctx.g_lds, [rj, dch])
        vals.append(b.cast_f32_to(b.fmul(kv, ctx.ex2(b.fsub(glc, gc))), ELEM))
    sink.kt(ctx, ch, off, dch, r8, vals)


def _emit_kt_all_waves(ctx: _ChunkCtx, ch, sink) -> None:
    """Kt with the original uniform ``tid + i*BLOCK`` mapping, every wave."""
    b, tid, BLOCK, N_CD = ctx.b, ctx.tid, ctx.BLOCK, ctx.N_CD
    for i in range(N_CD // (8 * BLOCK)):
        _emit_kt_slot(ctx, ch, sink, b.add(tid, b.const_i32(i * BLOCK)))


def _emit_stage_issue(ctx: _ChunkCtx, ch):
    """Issue the HBM loads for one chunk's g / k / q / beta.

    Returns each loaded value next to the staging slot it belongs in, so a
    caller can place the ``ds_write``s further along and let whatever runs in
    between cover the load latency. Slots are ``tid + i*BLOCK``, which is exact
    for every legal block size.

    ``beta`` is loaded on every thread against a clamped index rather than under
    the ``tid < C`` predicate its store needs: the value has to outlive the issue
    point, and a predicated load would leave it trapped inside the ``scf.if``.
    The redundant lanes re-read 128 B that is already in cache.
    """
    b = ctx.b
    C, DK, BLOCK, N_CD = ctx.C, ctx.DK, ctx.BLOCK, ctx.N_CD
    ELEM = ctx.ELEM
    tid = ctx.tid
    tile_cd, tile_c = ch.cd, ch.c
    tile = ch.tile
    staged = []
    raw = ctx.raw

    for i in range((N_CD + 4 * BLOCK - 1) // (4 * BLOCK)):
        vidx = b.add(tid, b.const_i32(i * BLOCK))
        valid = None
        safe = vidx
        if N_CD % (4 * BLOCK):
            valid = b.cmp_gt(b.const_i32(N_CD // 4), vidx)
            safe = b.select(valid, vidx, b.const_i32(N_CD // 4 - 1))
        off = b.mul(safe, b.const_i32(4))
        row = b.div(off, b.const_i32(DK))
        col4 = b.mod(off, b.const_i32(DK))
        if raw is not None:
            gidx = raw.qk_off(tile, row, col4)
            gval = b.global_load_vN(ctx.g_ptr, gidx, ELEM, 4)
            staged.append((ctx.g_lds, row, col4, gval, 4, valid, "g"))
        else:
            staged.append(
                (
                    ctx.g_lds,
                    row,
                    col4,
                    b.global_load_vN(ctx.g_ptr, b.add(tile_cd, off), F32, 4),
                    4,
                    valid,
                    None,
                )
            )
    for i in range((N_CD + 8 * BLOCK - 1) // (8 * BLOCK)):
        vidx = b.add(tid, b.const_i32(i * BLOCK))
        valid = None
        safe = vidx
        if N_CD % (8 * BLOCK):
            valid = b.cmp_gt(b.const_i32(N_CD // 8), vidx)
            safe = b.select(valid, vidx, b.const_i32(N_CD // 8 - 1))
        off = b.mul(safe, b.const_i32(8))
        row = b.div(off, b.const_i32(DK))
        col = b.mod(off, b.const_i32(DK))
        if raw is not None:
            gidx = raw.qk_off(tile, row, col)
            kval = b.global_load_vN(ctx.k_ptr, gidx, ELEM, 8)
            qval = b.global_load_vN(ctx.q_ptr, gidx, ELEM, 8)
        else:
            gidx = b.add(tile_cd, off)
            kval = b.global_load_vN(ctx.k_ptr, gidx, ELEM, 8)
            qval = b.global_load_vN(ctx.q_ptr, gidx, ELEM, 8)
        staged.append((ctx.k_lds, row, col, kval, 8, valid, "k"))
        staged.append((ctx.q_lds, row, col, qval, 8, valid, "q"))
    if raw is not None:
        bcol = b.select(b.cmp_gt(b.const_i32(C), tid), tid, b.const_i32(C - 1))
        beta_off = raw.beta_off(tile, bcol)
        if ctx.spec.fp32_beta_dtype:
            beta = b.global_load_f32(ctx.beta_ptr, beta_off)
        else:
            beta = b.cast_to_f32(
                b.global_load(
                    ctx.beta_ptr,
                    beta_off,
                    ELEM,
                    align=2,
                )
            )
    else:
        bcol = b.select(b.cmp_gt(b.const_i32(C), tid), tid, b.const_i32(C - 1))
        beta = b.global_load_f32(ctx.beta_ptr, b.add(tile_c, bcol))
    return staged, beta, tile


def _emit_stage_commit(ctx: _ChunkCtx, issued) -> None:
    """Write what :func:`_emit_stage_issue` loaded into the staging tiles."""
    b = ctx.b
    staged, beta, tile = issued
    spec = ctx.spec
    raw = ctx.raw
    bh = None
    head = None
    if raw is not None:
        bh = b.div(tile, raw.nc)
        head = b.mod(bh, raw.H)

    for lds, row, col, value, n, valid, kind in staged:
        with b.scf_if(valid) if valid is not None else nullcontext():
            if kind == "g" and raw is not None and spec.fuse_gate:
                for j in range(4):
                    raw_g = b.vec_extract(value, j)
                    col_j = b.add(col, b.const_i32(j))
                    gate = _apply_gate_fused(
                        b,
                        raw_g,
                        raw.a_log,
                        raw.dt_bias,
                        head,
                        col_j,
                        spec.lower_bound,
                        has_dt_bias=spec.has_dt_bias,
                        dk=ctx.DK,
                    )
                    _st(b, lds, row, col_j, value=gate, n=1)
            elif kind in ("k", "q") and raw is not None and spec.fuse_qk_l2norm:
                vals = [b.cast_to_f32(b.vec_extract(value, j)) for j in range(8)]
                inv = _l2norm_scale8(b, vals)
                vals = [b.fmul(v, inv) for v in vals]
                _st(
                    b,
                    lds,
                    row,
                    col,
                    value=b.vec_pack(
                        [b.cast_f32_to(v, ctx.ELEM) for v in vals], ctx.ELEM
                    ),
                    n=8,
                )
            else:
                _st(b, lds, row, col, value=value, n=n)
    if spec.fuse_beta_sigmoid and raw is not None:
        beta = _sigmoid_via_exp2(b, beta)
    with b.scf_if(b.cmp_gt(b.const_i32(ctx.C), ctx.tid)):
        _st(b, ctx.beta_lds, ctx.tid, value=beta, n=1)


def _emit_stage_inputs(ctx: _ChunkCtx, ch) -> None:
    """g / k / q / beta for ``ch`` from HBM into their staging tiles."""
    _emit_stage_commit(ctx, _emit_stage_issue(ctx, ch))


class _InputPrefetch:
    """Stage the *next* chunk's g / k / q / beta from inside this chunk's scan.

    All four staging tiles are dead for the whole scan phase -- g and k are last
    read by Kt in the solve window, q by the elementwise sweep, beta by the T'
    construction -- so the next chunk's inputs are written straight back into
    them and no second buffer is needed.

    The hand-off also needs no barrier of its own. The tile-phase rendezvous
    ahead of the scan separates the write from this chunk's last read (WAR), and
    the scan's own rendezvous separate it from the next chunk's first read
    (RAW). Both already exist for other reasons. The chunk loop therefore drops
    its post-staging barrier outright, and the load retires behind the scan's
    matmuls instead of at the head of the next chunk.
    """

    def __init__(self, ctx: _ChunkCtx, ch_next):
        self.ctx = ctx
        self.ch = ch_next

    def issue(self):
        return _emit_stage_issue(self.ctx, self.ch)

    def commit(self, issued) -> None:
        _emit_stage_commit(self.ctx, issued)


def _emit_v_issue(ctx: _ChunkCtx, ch, v_ptr):
    """Issue this chunk's V loads well before their LDS commit.

    The fused 512-thread shape has exactly one 128-bit V vector per thread.
    Keep the generic loop for other legal compile-time shapes; tail threads
    read the final vector again and predicate only the eventual LDS write so
    the loaded SSA value can live outside an ``scf.if``.
    """
    b = ctx.b
    C, EV, BLOCK = ctx.C, ctx.spec.head_v, ctx.BLOCK
    n_v = (C * EV) // 8
    cv = b.mul(ch.tile, b.const_i32(C * EV))
    issued = []
    for i in range((n_v + BLOCK - 1) // BLOCK):
        vidx = b.add(ctx.tid, b.const_i32(i * BLOCK))
        safe = b.select(b.cmp_gt(b.const_i32(n_v), vidx), vidx, b.const_i32(n_v - 1))
        off = b.mul(safe, b.const_i32(8))
        issued.append(
            (
                vidx,
                off,
                b.global_load_vN(v_ptr, b.add(cv, off), ctx.ELEM, 8),
            )
        )
    return issued


def _emit_v_commit(ctx: _ChunkCtx, v_lds, issued) -> None:
    """Park hoisted V vectors after both C x C MFMAs release the V tile."""
    b = ctx.b
    EV = ctx.spec.head_v
    n_v = (ctx.C * EV) // 8
    for vidx, off, value in issued:
        with b.scf_if(b.cmp_gt(b.const_i32(n_v), vidx)):
            _st(
                b,
                v_lds,
                b.div(off, b.const_i32(EV)),
                b.mod(off, b.const_i32(EV)),
                value=value,
                n=8,
            )


def _emit_idle_during_solve(ctx: _ChunkCtx, ch, sink) -> None:
    """Work the non-wave-0 threads do while wave 0 runs the triangular solve.

    Those waves would otherwise sit on the trailing ``s_barrier``. Kt does not
    read the solve tiles (it is ``(K * gamma_C / Gamma)^T`` off ``k``, ``g``,
    ``gl``), and V is not a tile-phase operand at all, so both are safe to
    issue in this window. Wave 0 does not participate: the original Kt loop
    dealt slots out as ``tid + i*BLOCK``, so the coverage is remapped onto
    ``BLOCK - 64`` workers.

    V used to be loaded here as well, but that put a ``vmcnt`` drain directly
    after each load. It is now issued before the elementwise/MFMA section and
    committed once the two C x C MFMAs release Y, leaving this window
    exclusively for Kt.
    """
    b = ctx.b
    tid, BLOCK, N_CD = ctx.tid, ctx.BLOCK, ctx.N_CD
    workers = BLOCK - 64
    wid = b.sub(tid, b.const_i32(64))

    n_kt = N_CD // 8
    for i in range((n_kt + workers - 1) // workers):
        vidx = b.add(wid, b.const_i32(i * workers))
        with b.scf_if(b.cmp_gt(b.const_i32(n_kt), vidx)):
            _emit_kt_slot(ctx, ch, sink, vidx)


def _emit_chunk_tiles(
    ctx: _ChunkCtx,
    tile,
    sink,
    v_ptr=None,
    v_lds=None,
    *,
    overlap_solve: bool = True,
    stage_inputs: bool = True,
) -> None:
    """Emit the six state-independent tiles for the chunk indexed by ``tile``.

    Reads ``q/k/g/beta`` for the chunk out of HBM and hands each finished tile
    to ``sink``. Nothing here depends on the state recurrence, so this is the
    whole parallel part of a chunkwise KDA forward.

    ``overlap_solve`` (prep, and fused at ``block_size >= 512``) runs Kt on the
    idle waves during the wave-0 solve. A 256-thread fused group has only three
    helper waves; the extra control flow in that already-fat kernel lost more
    than the overlap saved, so that path keeps the uniform post-solve Kt.
    ``v_ptr`` is fused-only and only set with ``overlap_solve``.

    ``stage_inputs=False`` says the inputs are already in their staging tiles --
    put there by the previous chunk's scan, see :class:`_InputPrefetch` -- so
    both the staging loads and the barrier that published them are skipped.
    """
    b, spec = ctx.b, ctx.spec
    C, DK, BLOCK = ctx.C, ctx.DK, ctx.BLOCK
    CREF, HALF, N_CD = ctx.CREF, ctx.HALF, ctx.N_CD
    ELEM, atom, scale = ctx.ELEM, ctx.atom, ctx.scale
    tid, lane, lane_m, frag_k_off = ctx.tid, ctx.lane, ctx.lane_m, ctx.frag_k_off
    # The triangular rank update covers the whole algorithmic chunk on wave 0.
    # C=16 uses the K=32 atom and zero-pads the contraction's upper half.
    solve_atom = MfmaAtom.bf16_16x16x32() if C == 16 else MfmaAtom.bf16_32x32x16()
    solve_lane_m = b.mod(lane, b.const_i32(solve_atom.m))
    solve_frag_k_off = b.mul(
        b.div(lane, b.const_i32(solve_atom.m)),
        b.const_i32(solve_atom.a_per_lane),
    )
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
    v_issued = None

    # =================================================================
    # 1. stage g / k / q into LDS with 128-bit transactions
    # =================================================================
    # A 128-bit vector never straddles a row: DK is a multiple of both 4 (fp32)
    # and 8 (bf16), so (row, col) decomposition of the flat vector index is
    # exact and the LDS destination stays lane-contiguous.
    if stage_inputs:
        _emit_stage_inputs(ctx, ch)
        b.sync_lds_only()

    # =================================================================
    # 2. in-place cumulative log decay, scaled to log2
    # =================================================================
    # One thread owns a whole (half-chunk, channel) column, so the running sum
    # stays in a register and the only cross-thread step is folding the first
    # half's total into the second.
    NG, GROUP = ctx.NG, ctx.GROUP
    d_ch = b.mod(tid, b.const_i32(DK))
    grp = b.div(tid, b.const_i32(DK))
    row0 = b.mul(grp, b.const_i32(GROUP))
    acc = b.const_f32(0.0)
    for i in range(GROUP):
        rr = b.add(row0, b.const_i32(i))
        acc = b.fadd(acc, lds_get(g_lds, [rr, d_ch]))
        _st(b, g_lds, rr, d_ch, value=b.fmul(acc, c_log2e), n=1)
    b.sync_lds_only()

    # Fold each group's running total into every later group. A group's local
    # total sits at its last row, so the offsets are read up front -- the writes
    # below land on some of those same rows once NG > 2, and the correct offset
    # is the pre-fold local total.
    totals = [
        lds_get(g_lds, [b.const_i32((j + 1) * GROUP - 1), d_ch]) for j in range(NG - 1)
    ]
    if NG > 2:
        b.sync_lds_only()
    # Only rows above the first group need folding, and they are dealt out
    # round-robin across the NG threads of a channel, so every thread writes the
    # same count. ``NG`` divides ``GROUP``, so all NG rows of one step fall in
    # the same group and the offset for a step is a compile-time sum.
    for i in range((C - GROUP) // NG):
        base = GROUP + i * NG
        rr = b.add(b.const_i32(base), grp)
        off = totals[0]
        for j in range(1, base // GROUP):
            off = b.fadd(off, totals[j])
        _st(b, g_lds, rr, d_ch, value=b.fadd(lds_get(g_lds, [rr, d_ch]), off), n=1)
    b.sync_lds_only()

    # =================================================================
    # 3. dec = gamma_C, and cache the whole-chunk log decay for Kt
    # =================================================================
    with b.scf_if(b.cmp_gt(b.const_i32(DK // 4), tid)):
        col4 = b.mul(tid, b.const_i32(4))
        gl = [
            lds_get(g_lds, [b.const_i32(C - 1), b.add(col4, b.const_i32(j))])
            for j in range(4)
        ]
        _st(b, gl_lds, col4, value=b.vec_pack(gl, F32), n=4)
        sink.dec(ctx, ch, col4, [ex2(v) for v in gl])

    # V is independent of the remaining tile work. Issue it here -- after the
    # cumsum's temporaries die, but before the long elementwise + dual-MFMA
    # section -- and consume it only after those MFMAs release Y. This leaves
    # hundreds of instructions to hide the old ~508-cycle HBM round trip
    # without extending four VGPRs across the entire tile phase.
    if v_ptr is not None and v_lds is not None:
        v_issued = _emit_v_issue(ctx, ch, v_ptr)

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

    for p in range((N_CD + 8 * BLOCK - 1) // (8 * BLOCK)):
        vidx = b.add(tid, b.const_i32(p * BLOCK))
        guard = (
            b.scf_if(b.cmp_gt(b.const_i32(N_CD // 8), vidx))
            if N_CD % (8 * BLOCK)
            else nullcontext()
        )
        with guard:
            row = b.add(ew_row, b.const_i32(p * ew_rstep))
            off = b.mul(vidx, b.const_i32(8))
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
    b.sync_lds_only()

    panelized = atom.m < C
    if panelized:
        # Four 16x16 panels, assigned round-robin to eight waves. Duplicating
        # each panel once keeps every wave useful and avoids a reduction: the
        # duplicate waves write identical values to the same LDS addresses.
        wave = b.div(tid, b.const_i32(64))
        panel = b.mod(wave, b.const_i32((C // atom.m) ** 2))
        panel_m = b.div(panel, b.const_i32(C // atom.n))
        panel_n = b.mod(panel, b.const_i32(C // atom.n))
        panel_row = b.mul(panel_m, b.const_i32(atom.m))
        panel_col = b.mul(panel_n, b.const_i32(atom.n))
        tile_lane_m = b.mod(lane, b.const_i32(atom.m))
        tile_frag_k_off = b.mul(
            b.div(lane, b.const_i32(atom.m)), b.const_i32(atom.a_per_lane)
        )
    else:
        panel_row = panel_col = b.const_i32(0)
        tile_lane_m = b.mod(lane, b.const_i32(atom.m))
        tile_frag_k_off = b.mul(
            b.div(lane, b.const_i32(atom.m)),
            b.const_i32(atom.a_per_lane),
        )

    def tile_output(i):
        row, col = atom.lane_to_output(b, lane, i)
        return b.add(panel_row, row), b.add(panel_col, col)

    def cxc_mfma(a_smem, b_smem):
        """This wave's panel of ``C x C = (C x DK)(C x DK)^T``."""
        acc = atom.zero_acc(b)
        for ks in range(spec.k_steps):
            kb = b.add(b.const_i32(ks * atom.k), tile_frag_k_off)
            av = _ld(b, a_smem, b.add(panel_row, tile_lane_m), kb, dtype=ELEM, n=8)
            bv = _ld(b, b_smem, b.add(panel_col, tile_lane_m), kb, dtype=ELEM, n=8)
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
        b.sync_lds_only()
        _emit_gk_gq_pass(ctx, ch, sink)
    if v_issued is not None:
        _emit_v_commit(ctx, v_lds, v_issued)

    # =================================================================
    # 5. T' = StrictTril(Diag(beta) Akk), RHS = Diag(beta), Aqk = Tril(.)
    # =================================================================
    # Both accumulators are already in the atom's output layout, so the masks
    # and the beta scale are applied in-register on the way to LDS.
    # ``zs`` accumulates the solved columns for the rank update and must read as
    # zero for rows not yet solved, so the unsolved part contributes nothing.
    zs_zero_cols = ctx.PCB if C < solve_atom.k else C
    n_vec_zs = (C * zs_zero_cols) // 8
    zero8 = b.vec_pack([b.cast_f32_to(b.const_f32(0.0), ELEM)] * 8, ELEM)
    with b.scf_if(b.cmp_gt(b.const_i32(n_vec_zs), tid)):
        zoff = b.mul(tid, b.const_i32(8))
        _st(
            b,
            zs_lds,
            b.div(zoff, b.const_i32(zs_zero_cols)),
            b.mod(zoff, b.const_i32(zs_zero_cols)),
            value=zero8,
            n=8,
        )
    # The C=16 solve/scan contract over a K=32 atom. Keep only the valid lower
    # half of T'; the padded columns must read as exact zero.
    if ctx.PCB > C:
        n_pad = (C * (ctx.PCB - C)) // 8
        with b.scf_if(b.cmp_gt(b.const_i32(n_pad), tid)):
            poff = b.mul(tid, b.const_i32(8))
            _st(
                b,
                tb_lds,
                b.div(poff, b.const_i32(ctx.PCB - C)),
                b.add(
                    b.const_i32(C),
                    b.mod(poff, b.const_i32(ctx.PCB - C)),
                ),
                value=zero8,
                n=8,
            )
        if hasattr(sink, "kt_lds"):
            n_kt_pad = (DK * (ctx.PCB - C)) // 8
            for i in range((n_kt_pad + BLOCK - 1) // BLOCK):
                vidx = b.add(tid, b.const_i32(i * BLOCK))
                with b.scf_if(b.cmp_gt(b.const_i32(n_kt_pad), vidx)):
                    poff = b.mul(vidx, b.const_i32(8))
                    _st(
                        b,
                        sink.kt_lds,
                        b.div(poff, b.const_i32(ctx.PCB - C)),
                        b.add(
                            b.const_i32(C),
                            b.mod(poff, b.const_i32(ctx.PCB - C)),
                        ),
                        value=zero8,
                        n=8,
                    )

    # The chunk-wide atom is redundantly computed by every wave, so wave 0 is
    # its sole writer. In panel mode every wave owns one panel (duplicated once)
    # and all waves must publish their panel.
    writer = nullcontext() if panelized else b.scf_if(b.cmp_gt(b.const_i32(64), tid))
    with writer:
        for i in range(atom.c_per_lane):
            row, col = tile_output(i)
            bet = lds_get(beta_lds, [row])
            tp = b.select(
                b.cmp_gt(row, col),
                b.fmul(bet, b.vec_extract(acc_kk, i)),
                b.const_f32(0.0),
            )
            # fp32 for the in-block substitution (it multiplies the solved
            # values directly, so it is the precision-critical copy) and bf16
            # for the rank-update MFMA operand.
            _st(b, m_lds, row, col, value=tp, n=1)
            _st(b, tb_lds, row, col, value=b.cast_f32_to(tp, ELEM), n=1)
            # The substitution reads its starting value straight out of a_mat,
            # so seed it with the right-hand side Diag(beta) here; each later
            # block overwrites its own rows with (RHS - rank update).
            _st(
                b,
                a_lds,
                row,
                col,
                value=b.select(b.cmp_eq(row, col), bet, b.const_f32(0.0)),
                n=1,
            )
            _st(
                b,
                qk_lds,
                row,
                col,
                value=b.select(
                    b.cmp_gt(b.add(row, b.const_i32(1)), col),  # row >= col
                    b.vec_extract(acc_qk, i),
                    b.const_f32(0.0),
                ),
                n=1,
            )
    b.sync_lds_only()

    # =================================================================
    # 6. A = (I + T')^-1 Diag(beta) by blocked forward substitution
    # =================================================================
    # Per block: rank-update the block's rows against every already-solved row
    # (a matmul, issued on MFMA), then substitute within the block (scalar, and
    # the only irreducibly serial part). The whole loop runs on wave 0 alone, so
    # the LDS hand-offs between the two halves need no s_barrier -- they are
    # ordered by the wave's own lgkmcnt -- and the other waves wait once at the
    # end instead of twice per block. Those waiting waves emit Kt (and, on the
    # fused path, prefetch V into the dead Y tile) in the same window: both are
    # independent of the solve, so they come off the critical path without an
    # extra barrier.
    #
    # Raising wave 0 to ``s_setprio(3)`` across this region measures as nothing
    # (9.599 ms vs 9.587 ms at 32x16x2048, inside a ~0.02 ms spread). Priority
    # only decides issue arbitration between waves that are both ready, and the
    # substitution below is bound by its own serial lgkmcnt round trips, so
    # there is no issue slot for wave 0 to win off the one helper sharing its
    # SIMD. Do not retry without first making the solve issue-bound.
    BS = spec.tile.solve_block
    NB = C // BS
    ks_solve = (C + solve_atom.k - 1) // solve_atom.k
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
                accu = solve_atom.zero_acc(b)
                for ks in range(ks_solve):
                    kb = b.add(b.const_i32(ks * solve_atom.k), solve_frag_k_off)
                    accu = solve_atom.emit(
                        b,
                        _ld(b, tb_lds, solve_lane_m, kb, dtype=ELEM, n=8),
                        _ld(b, zs_lds, solve_lane_m, kb, dtype=ELEM, n=8),
                        accu,
                    )
                # Slots [4*bi*t, 4*(bi+1)*t) are exactly this block's rows, so
                # the update folds straight into a_mat's seeded right-hand side
                # and the substitution below needs no separate staging tile.
                if solve_atom.m == 32:
                    update_slots = range(bi * 4 * (BS // 8), (bi + 1) * 4 * (BS // 8))
                    for i in update_slots:
                        row, col = solve_atom.lane_to_output(b, lane, i)
                        _st(
                            b,
                            a_lds,
                            row,
                            col,
                            value=b.fsub(
                                lds_get(a_lds, [row, col]),
                                b.vec_extract(accu, i),
                            ),
                            n=1,
                        )
                else:
                    for i in range(solve_atom.c_per_lane):
                        row, col = solve_atom.lane_to_output(b, lane, i)
                        with b.scf_if(b.cmp_ge(row, b.const_i32(bi * BS))):
                            with b.scf_if(b.cmp_lt(row, b.const_i32((bi + 1) * BS))):
                                _st(
                                    b,
                                    a_lds,
                                    row,
                                    col,
                                    value=b.fsub(
                                        lds_get(a_lds, [row, col]),
                                        b.vec_extract(accu, i),
                                    ),
                                    n=1,
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
                    _st(b, a_lds, cr, lane, value=val, n=1)
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
    if overlap_solve:
        with b.scf_if(b.cmp_ge(tid, b.const_i32(64))):
            _emit_idle_during_solve(ctx, ch, sink)
    b.sync_lds_only()

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
    if not overlap_solve:
        _emit_kt_all_waves(ctx, ch, sink)


def build_kda_chunk_prep(spec: KdaChunkPrepSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for the per-chunk tile builder.

    Kernel signature::

        (q, k: ptr<bf16>,      # [NT, C * DK] or raw [B,T,H,D]
         g:    ptr<f32>,       # [NT, C * DK] per-channel log decay, or raw bf16
         beta: ptr<f32>,       # [NT,C], or compile-time raw bf16/f32 [B,T,H]
         a_out, gk_out, gq_out, aqk_out, kt_out: ptr<bf16>,
         dec_out: ptr<f32>,    # [NT, DK]
         scale: f32)

    Raw mode adds ``a_log``, optional ``dt_bias``, dimensions, and the three
    element strides of the potentially non-contiguous beta view.

    Grid ``(NT, 1, 1)`` where ``NT = BH * NC``; block ``(block_size, 1, 1)``.
    """
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid kda_chunk_prep spec for {arch}: {why}")

    ELEM = _DTYPE_IR[spec.dtype]
    t = spec.tile
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = t.block_size
    if t.waves_per_eu:
        b.kernel.attrs["waves_per_eu"] = (
            t.waves_per_eu,
            t.waves_per_eu,
        )

    q_ptr = b.param("q_ptr", PtrType(ELEM, "global"), readonly=True, align=16)
    k_ptr = b.param("k_ptr", PtrType(ELEM, "global"), readonly=True, align=16)
    g_elem = ELEM if spec.raw_inputs else F32
    g_ptr = b.param("g_ptr", PtrType(g_elem, "global"), readonly=True, align=16)
    beta_elem = F32 if not spec.raw_inputs or spec.fp32_beta_dtype else ELEM
    beta_ptr = b.param(
        "beta_ptr",
        PtrType(beta_elem, "global"),
        readonly=True,
        align=2 if spec.raw_inputs and not spec.fp32_beta_dtype else 4,
    )
    a_ptr = b.param("a_ptr", PtrType(ELEM, "global"), writeonly=True, align=16)
    gk_ptr = b.param("gk_ptr", PtrType(ELEM, "global"), writeonly=True, align=16)
    gq_ptr = b.param("gq_ptr", PtrType(ELEM, "global"), writeonly=True, align=16)
    aqk_ptr = b.param("aqk_ptr", PtrType(ELEM, "global"), writeonly=True, align=16)
    kt_ptr = b.param("kt_ptr", PtrType(ELEM, "global"), writeonly=True, align=16)
    dec_ptr = b.param("dec_ptr", PtrType(F32, "global"), writeonly=True, align=16)
    scale = b.param("scale", F32)

    raw = None
    if spec.raw_inputs:
        a_log_ptr = b.param("a_log_ptr", PtrType(F32, "global"), readonly=True, align=4)
        dt_bias_ptr = b.param(
            "dt_bias_ptr", PtrType(F32, "global"), readonly=True, align=4
        )
        batch = b.param("batch", I32)
        heads = b.param("heads", I32)
        tseq = b.param("tseq", I32)
        nc = b.param("nc", I32)
        beta_stride_batch = b.param("beta_stride_batch", I32)
        beta_stride_token = b.param("beta_stride_token", I32)
        beta_stride_head = b.param("beta_stride_head", I32)
        raw = _RawTokenAddr(
            b,
            heads=heads,
            tseq=tseq,
            nc=nc,
            chunk=t.chunk,
            dk=spec.head_k,
            a_log=a_log_ptr,
            dt_bias=dt_bias_ptr,
            beta_stride_batch=beta_stride_batch,
            beta_stride_token=beta_stride_token,
            beta_stride_head=beta_stride_head,
        )

    ctx = _ChunkCtx(b, spec, (q_ptr, k_ptr, g_ptr, beta_ptr, scale, raw))
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
    # Fused-only tuned schedule. The larger row pitches improve LDS bank
    # distribution while the 512-thread panel mapping keeps eight waves
    # resident. Do not copy these pads to the split scan: its LDS budget is
    # deliberately capped for two workgroups per CU.
    tile: KdaTileSpec = KdaTileSpec(
        block_size=512,
        pad_dk=16,
        pad_cb=16,
        tile_atom_m=16,
        scan_atom_m=16,
    )
    has_initial_state: bool = False
    store_final_state: bool = True
    # Stage each chunk's inputs from the previous chunk's scan phase rather than
    # at the head of its own tile phase. Costs no LDS (the staging tiles are
    # dead across the scan) and removes the per-chunk staging barrier; see
    # :class:`_InputPrefetch`. Worth 7-9% at 32x16x2048 and bitwise identical,
    # so it is on by default; the flag exists to keep the A/B measurable.
    prefetch_inputs: bool = True
    # Reuse tile-only LDS for the scan extras. This cannot be combined with the
    # barrier-free input prefetch because that prefetch deliberately writes the
    # same staging addresses while the scan is live.
    overlay_lds: bool = False
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
    def scan_atom(self) -> MfmaAtom:
        """The atom the state scan runs on; see ``KdaTileSpec.scan_atom_m``."""
        return _scan_atom(self.tile, self.atom)

    @property
    def state_tiles(self) -> int:
        """Atom tiles per wave across the state's ``DK`` extent.

        Each wave owns one ``scan_atom.m``-row band of ``S^T`` and the full head
        dimension, which is what makes every one of the five per-chunk products
        partition the same way and keeps the state in that wave's registers.
        """
        return self.head_k // self.scan_atom.n

    def lds_bytes(self) -> int:
        """Exact fused allocation for the selected LDS schedule.

        The normal schedule uses typed allocations. The overlay schedule uses
        one byte pool with explicit views so loop liveness cannot keep
        tile-only and scan-only buffers simultaneously live.
        """
        t = self.tile
        C, DK, EV = t.chunk, self.head_k, self.head_v
        if self.overlay_lds:
            return _fused_lds_layout(DK, EV, t)["pool"]
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
        if not self.prefetch_inputs:
            parts += ("nopf",)
        if self.overlay_lds:
            parts += ("ovl",)
        return kernel_name_join(self.name, *parts)


def is_valid_fused_spec(
    spec: KdaChunkFusedSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a fused spec on ``arch``."""
    ok, why = is_valid_spec(spec.prep, arch=arch)
    if not ok:
        return False, why

    t = spec.tile
    if spec.overlay_lds and spec.prefetch_inputs:
        return False, (
            "overlay_lds aliases g/k/q/beta with live scan extras; "
            "set prefetch_inputs=False"
        )
    if spec.overlay_lds and t.chunk != 32:
        return False, "overlay_lds is only modeled for the C32 fused layout"
    ok, why = _check_scan_partition(spec.scan_atom, t, spec.head_k, spec.head_v)
    if not ok:
        return False, why
    lds = spec.lds_bytes()
    if lds > LDS_LIMIT:
        return False, f"LDS request {lds} B exceeds the {LDS_LIMIT} B budget"
    return True, "ok"


def _check_scan_partition(
    atom: MfmaAtom, t: KdaTileSpec, head_k: int, head_v: int
) -> Tuple[bool, str]:
    """Validate the one rule that partitions every product in the scan body.

    Each wave owns one ``atom.m``-row band of ``S^T`` and the full head
    dimension, so the v extent has to be exactly covered by the waves. Anything
    else needs a second partitioning rule for the same accumulator, which is how
    cross-wave reductions creep in.
    """
    waves = t.num_waves
    if head_v != atom.m * waves:
        return False, (
            f"head_v ({head_v}) must equal scan atom.m * waves "
            f"({atom.m} * {waves} = {atom.m * waves}); each wave owns one "
            "row band of the state"
        )
    if head_k % atom.n:
        return False, (
            f"head_k ({head_k}) must be a multiple of the atom N extent ({atom.n})"
        )
    if t.chunk < atom.k and t.chunk + t.pad_cb < atom.k:
        return False, (
            f"chunk + pad_cb ({t.chunk + t.pad_cb}) must cover scan MFMA K "
            f"({atom.k}) for zero padding"
        )
    # The chunk extent lands on the N side of two products and the M side of a
    # third, so tiling it needs a square atom that divides it.
    if atom.m != atom.n or t.chunk % atom.m:
        return False, (
            f"scan atom ({atom.m}x{atom.n}) must be square and divide chunk "
            f"({t.chunk}); the chunk extent is tiled on both operand sides"
        )
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
        _st(
            ctx.b,
            self.kt_lds,
            dch,
            r8,
            value=ctx.b.vec_pack(vals, ctx.ELEM),
            n=8,
        )


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
        dec_is_log: bool,
        ex2=None,
        v_lds=None,
        v_row_base=0,
        ev_stride=None,
        io=None,
        prefetch_v: bool = False,
        wave_local_intermediates: bool = False,
    ):
        self.b = b
        self.ex2 = ex2
        self.atom = atom
        self.C, self.DK = chunk, head_k
        self.EV = head_v
        self.EV_stride = head_v if ev_stride is None else ev_stride
        self.v_row_base = v_row_base
        self.io = io
        self.prefetch_v = prefetch_v
        self.wave_local_intermediates = wave_local_intermediates
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
        # Fused parks V in the dead Y tile during the solve; the residual then
        # reads this instead of ``v_ptr``. None on the split scan.
        self.v_lds = v_lds
        self.tid, self.lane = tid, lane
        self.dec_is_log = dec_is_log
        self.NS = head_k // atom.n
        self.KS_DK = head_k // atom.k
        self.KS_C = (chunk + atom.k - 1) // atom.k
        self.CPL = atom.c_per_lane
        self.N_CV = chunk * head_v
        # Atom tiles spanning the chunk extent. The three products whose output
        # carries a C extent (Z^T, V~^T and O) are one atom tile per wave only
        # while the atom is as wide as the chunk; a narrower atom -- which is
        # how the state subtiles down to fewer registers per lane -- splits
        # them into this many tiles instead.
        self.NC_T = chunk // atom.m
        # Operand lane mapping, derived from *this* atom rather than inherited
        # from the tile phase's: the two need not be the same atom, and an A
        # operand's row/K split is a function of the atom's M extent and
        # per-lane K width. Holds for every supported MFMA shape.
        self.lane_m = b.mod(lane, b.const_i32(atom.m))
        self.frag_k_off = b.mul(
            b.div(lane, b.const_i32(atom.m)), b.const_i32(atom.a_per_lane)
        )
        # This wave's band of S^T rows.
        self.wrow = b.mul(b.div(tid, b.const_i32(64)), b.const_i32(atom.m))

    def cv_base(self, tile):
        """Flat chunk base for v/o in the chunk-packed layout."""
        return self.b.mul(tile, self.b.const_i32(self.C * self.EV_stride))

    def global_ev(self, local_ev):
        b = self.b
        if isinstance(self.v_row_base, int):
            if self.v_row_base == 0:
                return local_ev
            return b.add(b.const_i32(self.v_row_base), local_ev)
        return b.add(self.v_row_base, local_ev)

    def v_global(self, tile, chunk_row, local_ev, *, chunk_n=None):
        """Absolute index of one v/o element."""
        b = self.b
        gev = self.global_ev(local_ev)
        if self.io is not None:
            if chunk_n is not None:
                return self.io.v_off(chunk_n, chunk_row, gev)
            return self.io.v_off(tile, chunk_row, gev, self.EV_stride)
        return b.add(
            self.cv_base(tile),
            b.add(b.mul(chunk_row, b.const_i32(self.EV_stride)), gev),
        )

    def o_global(self, tile, chunk_row, ev_col, v_row, *, chunk_n=None):
        """Absolute index of one output element (chunk row x v channel)."""
        b = self.b
        local_ev = b.add(v_row, ev_col)
        return self.v_global(tile, chunk_row, local_ev, chunk_n=chunk_n)

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
        apl = self.atom.a_per_lane
        for ks in range(ksteps):
            kb = b.add(b.const_i32(ks * self.atom.k), self.frag_k_off)
            av = _ld(b, a_smem, a_row, kb, dtype=self.ELEM, n=apl)
            bv = _ld(b, b_smem, b_row, kb, dtype=self.ELEM, n=apl)
            acc = self.atom.emit(b, av, bv, acc)
        return acc

    def crow(self, j, off=None):
        """Row ``lane_m`` of chunk-extent atom tile ``j``, i.e. ``j*m + lane_m``.

        The operand row for tile ``j`` of a product whose output spans the chunk
        extent. ``off`` shifts within the tile for the state's ``DK`` extent.
        """
        base = j * self.atom.m if off is None else off
        return self.b.add(self.b.const_i32(base), self.lane_m)

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
                _st(
                    b,
                    self.stb_lds,
                    b.add(self.wrow, row),
                    b.add(b.const_i32(ti * self.atom.n), col),
                    value=b.cast_f32_to(b.vec_extract(state[ti], i), self.ELEM),
                    n=1,
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

    def load_state_from_lds(self, lds):
        """Seed accumulators from a cooperatively staged fp32 ``S^T`` slice."""
        b = self.b
        out = []
        for ti in range(self.NS):
            vals = []
            for i in range(self.CPL):
                row, col = self.slot(i)
                ev = b.add(self.wrow, row)
                dk = b.add(b.const_i32(ti * self.atom.n), col)
                vals.append(b.vec_extract(_ld(b, lds, ev, dk, dtype=F32, n=1), 0))
            out.append(b.vec_pack(vals, F32))
        return out

    def store_state(self, ptr, base, state):
        b = self.b
        for ti in range(self.NS):
            for i in range(self.CPL):
                b.global_store_vN(
                    ptr, self.state_idx(base, i, ti), b.vec_extract(state[ti], i), 1
                )


def _emit_scan_body(
    sc: _ScanCtx,
    state,
    tile,
    *,
    chunk_n=None,
    prefetch: "_InputPrefetch" = None,
    tile_prefetch=None,
):
    """One chunk of the state recurrence; returns the updated state.

    .. code-block:: text

        Z^T  = S^T GK^T                    EV x C
        R^T  = V^T - Z^T                   EV x C   (in-register)
        V~^T = R^T A^T                     EV x C
        O    = GQ S + Aqk V~               C x EV
        S^T <- S^T Diag(dec) + V~^T Kt^T   EV x DK

    Working transposed keeps every product in ``A B^T`` form with the
    contraction on the fastest axis, so no operand ever needs an LDS transpose.

    ``prefetch`` stages the next chunk's inputs across this body: the loads go
    out first and the writes land after the ``V~`` rendezvous, so the scan's own
    matmuls sit between them and the closing barrier publishes the result.
    """
    b, atom = sc.b, sc.atom
    lane_m, wrow, CPL = sc.lane_m, sc.wrow, sc.CPL
    ELEM = sc.ELEM
    NC_T = sc.NC_T
    srow = b.add(wrow, lane_m)  # this wave's S^T operand row

    def cidx(jt, col):
        """Global chunk index of ``col`` inside chunk-extent tile ``jt``."""
        return col if jt == 0 else b.add(b.const_i32(jt * atom.m), col)

    def sync_intermediate():
        if sc.wave_local_intermediates:
            b.s_waitcnt(lgkmcnt=0)
        else:
            b.sync_lds_only()

    # Issued before anything else in the scan so the whole body covers the HBM
    # latency; the staging tiles are already dead by here.
    issued = prefetch.issue() if prefetch is not None else None

    # The standalone scan reads V from HBM.  Issue those loads before publishing
    # the state so that the mirror stores, rendezvous, and Z MFMAs cover their
    # latency.  ATT otherwise places an exposed vmcnt drain immediately before
    # every residual group.  The fused 256-thread schedule deliberately keeps
    # its old late-load path: carrying these vectors there was measured to lose.
    issued_v = None
    if sc.prefetch_v and sc.v_lds is None:
        issued_v = []
        for jt in range(NC_T):
            jt_values = []
            for grp in range(CPL // 4):
                row0, col = sc.slot(4 * grp)
                cg = cidx(jt, col)
                jt_values.append(
                    b.global_load_vN(
                        sc.v_ptr,
                        sc.v_global(tile, cg, b.add(wrow, row0), chunk_n=chunk_n),
                        ELEM,
                        4,
                    )
                )
            issued_v.append(jt_values)

    # Materialized scan tiles are much larger than V and remain live until the
    # end of the body.  Their prefetch therefore issues here but commits only
    # after the closing LDS rendezvous has retired every read of the current
    # tiles.  The next iteration's publish-state rendezvous makes those writes
    # visible, so this adds neither a second LDS buffer nor another barrier.
    # SA16 value-split scans have twice as many per-thread tile loads.  Issuing
    # that larger burst here delays the older V loads at the residual use, so
    # defer it until V has retired.  SA32 keeps the longer prefetch distance.
    issued_tiles = None
    if tile_prefetch is not None and atom.m == 32:
        issued_tiles = tile_prefetch.issue()

    # C=16 uses the 16x16x32 atom. V~/residual has only 16 real chunk columns,
    # so zero the padded K half before any of the three C-contracted products.
    if sc.C < atom.k:
        pad = atom.k - sc.C
        n_pad = (sc.EV * pad) // 8
        for p in range((n_pad + sc.BLOCK - 1) // sc.BLOCK):
            vidx = b.add(sc.tid, b.const_i32(p * sc.BLOCK))
            with b.scf_if(b.cmp_gt(b.const_i32(n_pad), vidx)):
                poff = b.mul(vidx, b.const_i32(8))
                _st(
                    b,
                    sc.vn_lds,
                    b.div(poff, b.const_i32(pad)),
                    b.add(
                        b.const_i32(sc.C),
                        b.mod(poff, b.const_i32(pad)),
                    ),
                    value=b.vec_pack(
                        [b.cast_f32_to(b.const_f32(0.0), ELEM)] * 8,
                        ELEM,
                    ),
                    n=8,
                )

    # The mirror is a derived copy of the state the caller already carries in
    # registers, so it is rebuilt here rather than left behind by the previous
    # chunk. That is what keeps its live range inside the scan phase: published
    # at the end instead, it would span the tile phase and the LDS packer could
    # not alias it onto the staging tiles that are dead by then.
    sc.publish_state(state)
    b.sync_lds_only()

    # The chunk extent lands on the N side of two products and the M side of
    # the third, so tiling it both ways needs a square atom.
    assert atom.m == atom.n, "chunk-extent tiling needs a square atom"

    # ---- Z^T = S^T GK^T, then R^T = V^T - Z^T into the V~ tile ----------
    acc_z = [
        sc.gemm(
            sc.stb_lds,
            srow,
            sc.gk_lds,
            sc.crow(jt),
            sc.KS_DK,
            atom.zero_acc(b),
        )
        for jt in range(NC_T)
    ]
    # A lane's slots are runs of four consecutive v channels at one fixed chunk
    # row, so V arrives in 4-wide loads and the residual never needs an fp32
    # staging tile. On the fused path those loads hit the Y tile idle waves
    # filled during the solve; the split scan still reads V from HBM. Every
    # residual tile has to land before the next product reads them, so the
    # whole chunk extent is written before the barrier.
    for jt in range(NC_T):
        for grp in range(CPL // 4):
            row0, col = sc.slot(4 * grp)
            cg = cidx(jt, col)
            if sc.v_lds is not None:
                vvec = _ld(b, sc.v_lds, cg, b.add(wrow, row0), dtype=ELEM, n=4)
            elif issued_v is not None:
                vvec = issued_v[jt][grp]
            else:
                vvec = b.global_load_vN(
                    sc.v_ptr,
                    sc.v_global(tile, cg, b.add(wrow, row0), chunk_n=chunk_n),
                    ELEM,
                    4,
                )
            for j in range(4):
                row, _ = sc.slot(4 * grp + j)
                res = b.fsub(
                    b.cast_to_f32(b.vec_extract(vvec, j)),
                    b.vec_extract(acc_z[jt], 4 * grp + j),
                )
                _st(
                    b,
                    sc.vn_lds,
                    b.add(wrow, row),
                    cg,
                    value=b.cast_f32_to(res, ELEM),
                    n=1,
                )
    sync_intermediate()
    if tile_prefetch is not None and issued_tiles is None:
        issued_tiles = tile_prefetch.issue()

    # ---- V~^T = R^T A^T -------------------------------------------------
    # Contracts over the full chunk extent, so every tile reads the whole
    # residual: all of them are computed before any overwrites it in place.
    acc_v = [
        sc.gemm(
            sc.vn_lds,
            srow,
            sc.ab_lds,
            sc.crow(jt),
            sc.KS_C,
            atom.zero_acc(b),
        )
        for jt in range(NC_T)
    ]
    if not sc.wave_local_intermediates:
        b.sync_lds_only()
    for jt in range(NC_T):
        for i in range(CPL):
            row, col = sc.slot(i)
            _st(
                b,
                sc.vn_lds,
                b.add(wrow, row),
                cidx(jt, col),
                value=b.cast_f32_to(b.vec_extract(acc_v[jt], i), ELEM),
                n=1,
            )
    sync_intermediate()

    # The next chunk's inputs land here: late enough that the loads issued at
    # the top of this body have had the Z and V~ matmul groups to retire behind,
    # and still ahead of the rendezvous that closes the body, which is what
    # publishes them to every wave.
    if prefetch is not None:
        prefetch.commit(issued)

    # ---- O = GQ S + Aqk V~ ----------------------------------------------
    # This wave owns a band of output *columns* here (the same band of v
    # channels it owns of the state), so GQ and Aqk are the A operands and the
    # state mirror and V~ are the B operands -- which puts the chunk extent on
    # the M side, so a chunk-extent tile picks the A operand's row.
    acc_os = []
    for jt in range(NC_T):
        acc_o = sc.gemm(
            sc.gq_lds,
            sc.crow(jt),
            sc.stb_lds,
            srow,
            sc.KS_DK,
            atom.zero_acc(b),
        )
        acc_os.append(
            sc.gemm(
                sc.aqb_lds,
                sc.crow(jt),
                sc.vn_lds,
                srow,
                sc.KS_C,
                acc_o,
            )
        )
    for jt, acc_o in enumerate(acc_os):
        # Straight to HBM, no staging tile: a slot's column index is the lane's
        # position in the atom's N extent, so one store instruction covers a
        # contiguous run of v channels per lane group and is already coalesced.
        for i in range(CPL):
            row, col = sc.slot(i)
            b.global_store_vN(
                sc.o_ptr,
                sc.o_global(tile, cidx(jt, row), col, wrow, chunk_n=chunk_n),
                b.cast_f32_to(b.vec_extract(acc_o, i), ELEM),
                1,
            )

    # ---- S^T <- S^T Diag(dec) + V~^T Kt^T -------------------------------
    # dec is the whole-chunk decay per k channel, i.e. the state's column
    # index.  Every accumulator slot owned by a lane has the same column for
    # both supported scan atoms, so one LDS read scales the whole fragment.
    scaled_states = []
    kt_rows = []
    _, decay_col = sc.slot(0)
    for ti in range(sc.NS):
        d = b.vec_extract(
            _ld(
                b,
                sc.dec_lds,
                b.add(b.const_i32(ti * atom.n), decay_col),
                dtype=F32,
                n=1,
            ),
            0,
        )
        if sc.dec_is_log:
            d = sc.ex2(d)
        scaled = [b.fmul(b.vec_extract(state[ti], i), d) for i in range(CPL)]
        scaled_states.append(b.vec_pack(scaled, F32))
        kt_rows.append(b.add(b.const_i32(ti * atom.n), lane_m))
    new_state = [
        sc.gemm(
            sc.vn_lds,
            b.add(wrow, lane_m),
            sc.kt_lds,
            kt_rows[ti],
            sc.KS_C,
            scaled_states[ti],
        )
        for ti in range(sc.NS)
    ]
    b.sync_lds_only()
    if tile_prefetch is not None:
        tile_prefetch.commit(issued_tiles)
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
        S^T <- S^T Diag(dec) + V~^T Kt^T   EV x DK

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
    ELEM = _DTYPE_IR[spec.dtype]
    atom = spec.scan_atom

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

    ctx = _ChunkCtx(
        b,
        prep,
        (q_ptr, k_ptr, g_ptr, beta_ptr, scale),
        overlay=spec.overlay_lds,
    )
    if spec.overlay_lds:
        kt_lds, stb_lds, vn_lds = ctx.kt_lds, ctx.stb_lds, ctx.vn_lds
    else:
        kt_lds = b.smem_alloc(ELEM, [DK, C + t.pad_cb], "kt_s")
        stb_lds = b.smem_alloc(ELEM, [EV, DK + t.pad_dk], "stb_s")
        vn_lds = b.smem_alloc(ELEM, [EV, C + t.pad_cb], "vn_s")
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
        dec_is_log=True,
        ex2=ctx.ex2,
        # V is issued before the elementwise/MFMA section and parked in Y once
        # its C x C consumers finish. Restrict this to the 512-thread schedule:
        # the 256-thread path did not benefit from the extra live vectors.
        v_lds=ctx.y_lds if (ctx.PDK >= EV and BLOCK >= 512) else None,
    )
    sink = _LdsTileSink(*sc.tiles_for_sink())

    bh = b.block_id_x()
    state_base = b.mul(bh, b.const_i32(EV * DK))
    # No initial publish: the scan body mirrors whatever state it is handed.
    s_init = (
        sc.load_state(h0_ptr, state_base) if spec.has_initial_state else sc.zero_state()
    )

    # With the prefetch on, every chunk finds its inputs already staged by its
    # predecessor's scan, so chunk 0 is the one that has to stage its own and is
    # peeled out here. This is the only staging barrier left in the kernel.
    pf = spec.prefetch_inputs
    if pf:
        _emit_stage_inputs(ctx, _ChunkOffsets(ctx, b.mul(bh, nc)))
        b.sync_lds_only()

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
        overlap = BLOCK >= 512
        _emit_chunk_tiles(
            ctx,
            tile,
            sink,
            v_ptr=v_ptr if sc.v_lds is not None else None,
            v_lds=sc.v_lds,
            overlap_solve=overlap,
            stage_inputs=not pf,
        )
        # The scan does not consume tile outputs until after publish_state()'s
        # rendezvous, so that later barrier can publish both the tile outputs
        # and the state mirror. The explicit overlay is the exception: its
        # mirror writes reuse tile-only addresses and must not race outstanding
        # reads from the final A/Aqk copy.
        if spec.overlay_lds:
            b.sync_lds_only()
        prefetch = None
        if pf:
            # The last chunk has no successor, so it re-stages itself rather
            # than reading off the end of the inputs. Clamping beats a branch:
            # the loads are redundant, not predicated, and this is one chunk in
            # a sequence of NC.
            nxt = b.add(n, b.const_i32(1))
            nxt = b.select(b.cmp_gt(nc, nxt), nxt, n)
            prefetch = _InputPrefetch(
                ctx, _ChunkOffsets(ctx, b.add(b.mul(bh, nc), nxt))
            )
        b.scf_yield(*_emit_scan_body(sc, list(carried), tile, prefetch=prefetch))

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
    # Split the v extent across ``value_splits`` independent workgroups per
    # (batch, head), mirroring FlashKDA K2's ``ceil(V/BW)`` grid dimension.
    value_splits: int = 1
    # Read/write token-major [B,T,H,D] tensors instead of chunk-packed views.
    token_major_io: bool = False
    # Issue the next chunk's materialized tiles across the current scan body,
    # then reuse the same LDS allocations after their current contents die.
    prefetch_tiles: bool = True
    # The residual and V~ LDS rows are private to one wave. Their phase
    # boundaries need an lgkmcnt drain, but not a workgroup barrier.
    wave_local_intermediates: bool = False
    name: str = "rocke_kda_chunk_scan"

    @property
    def prep(self) -> KdaChunkPrepSpec:
        """The tile builder whose output this scan stages.

        Derived rather than passed alongside, because the scan's staging
        copies assume exactly the layouts that builder's global sink writes:
        two specs that could drift apart would drift into a silent misread.
        """
        tile = self.tile
        if self.value_splits != 1:
            # The scan's smaller block and narrower atom manufacture recurrence
            # parallelism; the independent tile producer keeps its 256-thread
            # schedule.  Only the pads/chunk/tile atom define their shared HBM
            # layout, so discard scan-only geometry here.
            tile = replace(tile, block_size=256, scan_atom_m=0)
        return KdaChunkPrepSpec(
            head_k=self.head_k,
            head_v=self.head_v,
            dtype=self.dtype,
            tile=tile,
        )

    @property
    def atom(self) -> MfmaAtom:
        return self.prep.atom

    @property
    def scan_atom(self) -> MfmaAtom:
        """The atom the state scan runs on; see ``KdaTileSpec.scan_atom_m``."""
        return _scan_atom(self.tile, self.atom)

    @property
    def state_tiles(self) -> int:
        """Atom tiles per wave across the state's ``DK`` extent."""
        return self.head_k // self.scan_atom.n

    def _base_lds_bytes(self) -> int:
        """LDS footprint excluding the optional fp32 h0 staging slice."""
        t = self.tile
        C, DK, EV = t.chunk, self.head_k, self.head_v
        ev = EV // self.value_splits
        PDK, PCB = DK + t.pad_dk, C + t.pad_cb
        return (
            2 * C * PDK  # gk_s   bf16 (C x DK)
            + 2 * C * PDK  # gq_s  bf16 (C x DK)
            + 2 * C * PCB  # a_s   bf16 (C x C)
            + 2 * C * PCB  # aqk_s bf16 (C x C)
            + 2 * DK * PCB  # kt_s bf16 (DK x C)
            + 2 * ev * PDK  # stb_s bf16 mirror of S^T
            + 2 * ev * PCB  # vn_s  bf16 (EV x C)
            + 4 * DK  # dec_s   fp32 (DK)
        )

    @property
    def stages_h0_in_lds(self) -> bool:
        """Whether h0 staging preserves the requested LDS occupancy."""
        if not self.has_initial_state:
            return False
        ev = self.head_v // self.value_splits
        pdk = self.head_k + self.tile.pad_dk
        return self._base_lds_bytes() + 4 * ev * pdk <= LDS_LIMIT // self.min_occupancy

    def lds_bytes(self) -> int:
        """The six staged tiles plus the state mirror, ``V~``, and optional h0.

        Every tile is staged in the exact layout its consumer wants as an MFMA
        operand, so staging is a straight copy and the scan body is identical to
        the fused one. Unlike the fused kernel there is nothing to overlap them
        with, so each is its own allocation -- which is still the smaller
        footprint, because none of the tile builder's staging tiles exist here.
        """
        total = self._base_lds_bytes()
        if self.stages_h0_in_lds:
            ev = self.head_v // self.value_splits
            total += 4 * ev * (self.head_k + self.tile.pad_dk)
        return total

    def kernel_name(self) -> str:
        parts = (f"dk{self.head_k}", f"dv{self.head_v}", self.dtype)
        parts += self.tile.name_parts()
        if self.has_initial_state:
            parts += ("h0",)
        if not self.store_final_state:
            parts += ("noht",)
        if self.value_splits != 1:
            parts += (f"vs{self.value_splits}",)
        if self.token_major_io:
            parts += ("tm",)
        if not self.prefetch_tiles:
            parts += ("nopf",)
        if self.wave_local_intermediates:
            parts += ("wlocal",)
        return kernel_name_join(self.name, *parts)


def tuned_kda_chunk_scan_spec(
    workgroups: int,
    *,
    head_k: int = 128,
    head_v: int = 128,
    dtype: str = "bf16",
    chunk: int = 32,
    has_initial_state: bool = False,
    store_final_state: bool = True,
    token_major_io: bool = False,
) -> KdaChunkScanSpec:
    """Return the measured gfx950 scan geometry for a recurrence-stream count.

    The Kimi K3 shape (DK=DV=128, C=32) is grid-starved below one workgroup per
    CU.  Four value bands win through 96 streams, two narrower-atom bands win
    through 192, and the unsplit scan wins once the base grid is saturated.
    Other contracts retain the conservative default geometry.
    """
    tile = KdaTileSpec(chunk=chunk)
    value_splits = 1
    wave_local_intermediates = False
    if head_k == 128 and head_v == 128 and dtype.lower() == "bf16" and chunk == 32:
        if int(workgroups) <= 96:
            tile = replace(
                tile,
                block_size=128,
                scan_atom_m=16,
                pad_dk=16,
                pad_cb=0,
            )
            value_splits = 4
            wave_local_intermediates = True
        elif int(workgroups) <= 192:
            tile = replace(tile, block_size=256, scan_atom_m=16)
            value_splits = 2
    return KdaChunkScanSpec(
        head_k=head_k,
        head_v=head_v,
        dtype=dtype,
        tile=tile,
        has_initial_state=has_initial_state,
        store_final_state=store_final_state,
        value_splits=value_splits,
        token_major_io=token_major_io,
        wave_local_intermediates=wave_local_intermediates,
    )


def is_valid_scan_spec(
    spec: KdaChunkScanSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a standalone scan spec on ``arch``."""
    if arch != "gfx950":
        return False, f"unsupported arch {arch}"
    if spec.dtype not in _DTYPE_IR:
        return False, f"unsupported dtype {spec.dtype}"
    t = spec.tile

    if spec.value_splits not in _RAW_VALUE_SPLITS:
        return False, (
            f"value_splits must be one of {_RAW_VALUE_SPLITS}, got {spec.value_splits}"
        )
    if spec.head_v % spec.value_splits:
        return False, (
            f"head_v ({spec.head_v}) must divide evenly by value_splits "
            f"({spec.value_splits})"
        )
    ev_slice = spec.head_v // spec.value_splits
    if ev_slice % spec.scan_atom.n:
        return False, (
            f"v slice ({ev_slice}) must divide scan atom n ({spec.scan_atom.n})"
        )
    num_waves = spec.tile.num_waves
    if ev_slice != num_waves * spec.scan_atom.m:
        return False, (
            f"v slice ({ev_slice}) must equal waves ({num_waves}) * scan atom m "
            f"({spec.scan_atom.m}); adjust block_size, scan_atom_m, or value_splits"
        )

    t = spec.tile
    ok, why = _check_scan_partition(spec.scan_atom, t, spec.head_k, ev_slice)
    if not ok:
        return False, why
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
    for rows, cols in (
        (t.chunk, spec.head_k),
        (t.chunk, t.chunk),
        (spec.head_k, t.chunk),
    ):
        n_slot = rows * cols // 8
        if n_slot > t.block_size and n_slot % t.block_size:
            return False, (
                f"tile {rows}x{cols} needs {n_slot} slots, not a multiple of "
                f"{t.block_size}"
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
    if spec.head_k > 4 * t.block_size:
        return False, (
            f"dec tile ({spec.head_k}) exceeds one guarded pass of "
            f"{4 * t.block_size} elements for block_size ({t.block_size})"
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
    ev_slice = EV // spec.value_splits
    BLOCK = t.block_size
    PDK, PCB = DK + t.pad_dk, C + t.pad_cb
    ELEM = _DTYPE_IR[spec.dtype]
    atom = spec.scan_atom
    pipeline_tiles = spec.prefetch_tiles and C == 32 and BLOCK >= 128

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

    io = None
    heads = None
    if spec.token_major_io:
        batch = b.param("batch", I32)
        heads = b.param("heads", I32)
        tseq = b.param("tseq", I32)

    gk_lds = b.smem_alloc(ELEM, [C, PDK], "gk_s")
    gq_lds = b.smem_alloc(ELEM, [C, PDK], "gq_s")
    ab_lds = b.smem_alloc(ELEM, [C, PCB], "a_s")
    aqb_lds = b.smem_alloc(ELEM, [C, PCB], "aqk_s")
    kt_lds = b.smem_alloc(ELEM, [DK, PCB], "kt_s")
    dec_lds = b.smem_alloc(F32, [DK], "dec_s")
    stb_lds = b.smem_alloc(ELEM, [ev_slice, PDK], "stb_s")
    vn_lds = b.smem_alloc(ELEM, [ev_slice, PCB], "vn_s")
    h0_lds = (
        b.smem_alloc(F32, [ev_slice, PDK], "h0_s") if spec.stages_h0_in_lds else None
    )

    tid = b.thread_id_x()
    lane = b.mod(tid, b.const_i32(64))

    wg = b.block_id_x()
    if spec.value_splits == 1:
        bh = wg
        v_row_base = b.const_i32(0)
    else:
        bh = b.div(wg, b.const_i32(spec.value_splits))
        v_split = b.mod(wg, b.const_i32(spec.value_splits))
        v_row_base = b.mul(v_split, b.const_i32(ev_slice))

    if spec.token_major_io:
        io = _ScanTokenIo(
            b,
            heads=heads,
            tseq=tseq,
            chunk=C,
            ev_stride=EV,
            batch=b.div(bh, heads),
            head=b.mod(bh, heads),
            v_row_base=v_row_base,
        )

    sc = _ScanCtx(
        b,
        atom=atom,
        chunk=C,
        head_k=DK,
        head_v=ev_slice,
        block_size=BLOCK,
        elem=ELEM,
        tiles=(gk_lds, gq_lds, ab_lds, aqb_lds, kt_lds, dec_lds),
        stb_lds=stb_lds,
        vn_lds=vn_lds,
        v_ptr=v_ptr,
        o_ptr=o_ptr,
        tid=tid,
        lane=lane,
        dec_is_log=False,
        v_row_base=v_row_base,
        ev_stride=EV,
        io=io,
        prefetch_v=True,
        wave_local_intermediates=spec.wave_local_intermediates,
    )

    state_base = b.add(
        b.mul(bh, b.const_i32(EV * DK)),
        b.mul(v_row_base, b.const_i32(DK)),
    )
    # When the LDS budget allows, h0 is cooperatively staged and consumed only
    # after chunk zero's operands arrive. Larger value slices keep the direct
    # load path so the two-workgroup LDS occupancy contract remains intact.
    s_init = (
        sc.zero_state()
        if h0_lds is not None
        else (
            sc.load_state(h0_ptr, state_base)
            if spec.has_initial_state
            else sc.zero_state()
        )
    )

    def stage_h0_to_lds():
        """Cooperative global->LDS copy of this WG's value-split state slice."""
        n_elem = ev_slice * DK
        n_slot = n_elem // 8
        for i in range(max(1, n_slot // BLOCK)):
            vidx = b.add(tid, b.const_i32(i * BLOCK))
            guard = (
                nullcontext()
                if n_slot >= BLOCK
                else b.scf_if(b.cmp_gt(b.const_i32(n_slot), vidx))
            )
            with guard:
                off = b.mul(vidx, b.const_i32(8))
                ev = b.div(off, b.const_i32(DK))
                col8 = b.mod(off, b.const_i32(DK))
                gbase = b.add(
                    state_base,
                    b.add(b.mul(ev, b.const_i32(DK)), col8),
                )
                _st(
                    b,
                    h0_lds,
                    ev,
                    col8,
                    value=b.global_load_vN(h0_ptr, gbase, F32, 8),
                    n=8,
                )

    if h0_lds is not None:
        stage_h0_to_lds()
        b.sync_lds_only()

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
                _st(
                    b,
                    dst,
                    b.div(off, b.const_i32(cols)),
                    b.mod(off, b.const_i32(cols)),
                    value=b.global_load_vN(src, b.add(base, off), ELEM, 8),
                    n=8,
                )

    def stage_all(tile):
        """Stage one complete materialized chunk with immediate load-to-LDS copies."""
        cd = b.mul(tile, b.const_i32(C * DK))
        cc = b.mul(tile, b.const_i32(C * C))
        stage(gk_ptr, gk_lds, C, DK, cd)
        stage(gq_ptr, gq_lds, C, DK, cd)
        stage(kt_ptr, kt_lds, DK, C, cd)
        stage(a_ptr, ab_lds, C, C, cc)
        stage(aqk_ptr, aqb_lds, C, C, cc)
        with b.scf_if(b.cmp_gt(b.const_i32(DK // 4), tid)):
            col4 = b.mul(tid, b.const_i32(4))
            _st(
                b,
                dec_lds,
                col4,
                value=b.global_load_vN(
                    dec_ptr, b.add(b.mul(tile, b.const_i32(DK)), col4), F32, 4
                ),
                n=4,
            )

    def issue_stage(src, rows, cols, base):
        """Issue a tile's HBM loads now and return values for a later LDS commit."""
        n_slot = rows * cols // 8
        issued = []
        for i in range(max(1, n_slot // BLOCK)):
            vidx = b.add(tid, b.const_i32(i * BLOCK))
            # A value issued under scf.if cannot escape to the late commit.
            # Clamp inactive lanes to the final valid vector instead, then
            # predicate their LDS writes in commit_stage.  The pipelined
            # schedules keep this duplication smaller than the latency hidden.
            safe = (
                vidx
                if n_slot >= BLOCK
                else b.select(
                    b.cmp_gt(b.const_i32(n_slot), vidx),
                    vidx,
                    b.const_i32(n_slot - 1),
                )
            )
            off = b.mul(safe, b.const_i32(8))
            issued.append(
                (
                    vidx,
                    off,
                    b.global_load_vN(src, b.add(base, off), ELEM, 8),
                )
            )
        return issued

    def commit_stage(dst, cols, n_slot, issued):
        """Commit values from :func:`issue_stage` after the old tile is dead."""
        for vidx, off, value in issued:
            guard = (
                nullcontext()
                if n_slot >= BLOCK
                else b.scf_if(b.cmp_gt(b.const_i32(n_slot), vidx))
            )
            with guard:
                _st(
                    b,
                    dst,
                    b.div(off, b.const_i32(cols)),
                    b.mod(off, b.const_i32(cols)),
                    value=value,
                    n=8,
                )

    class _TilePrefetch:
        """Register-held next-chunk loads that reuse the current chunk's LDS."""

        def __init__(self, tile):
            self.tile = tile

        def _bases(self):
            return (
                b.mul(self.tile, b.const_i32(C * DK)),
                b.mul(self.tile, b.const_i32(C * C)),
            )

        def _issue_dec(self):
            dec_slot = DK // 4
            dec_vidx = tid
            dec_safe = b.select(
                b.cmp_gt(b.const_i32(dec_slot), dec_vidx),
                dec_vidx,
                b.const_i32(dec_slot - 1),
            )
            dec_col4 = b.mul(dec_safe, b.const_i32(4))
            return (
                dec_vidx,
                dec_col4,
                b.global_load_vN(
                    dec_ptr,
                    b.add(b.mul(self.tile, b.const_i32(DK)), dec_col4),
                    F32,
                    4,
                ),
            )

        def _commit_dec(self, dec):
            dec_vidx, dec_col4, dec_value = dec
            with b.scf_if(b.cmp_gt(b.const_i32(DK // 4), dec_vidx)):
                _st(b, dec_lds, dec_col4, value=dec_value, n=4)

        def issue(self):
            cd, cc = self._bases()
            return (
                issue_stage(gk_ptr, C, DK, cd),
                issue_stage(gq_ptr, C, DK, cd),
                issue_stage(kt_ptr, DK, C, cd),
                issue_stage(a_ptr, C, C, cc),
                issue_stage(aqk_ptr, C, C, cc),
                self._issue_dec(),
            )

        def commit(self, issued):
            gk, gq, kt, aa, aqk, dec = issued
            commit_stage(gk_lds, DK, C * DK // 8, gk)
            commit_stage(gq_lds, DK, C * DK // 8, gq)
            commit_stage(kt_lds, C, DK * C // 8, kt)
            commit_stage(ab_lds, C, C * C // 8, aa)
            commit_stage(aqb_lds, C, C * C // 8, aqk)
            self._commit_dec(dec)

    first_tile = b.mul(bh, nc)
    chunk0_n = b.const_i32(0)
    if pipeline_tiles:
        # Peel chunk zero.  Every later chunk is loaded by its predecessor.
        stage_all(first_tile)
        b.sync_lds_only()

    peel_h0 = h0_lds is not None
    loop_init = s_init
    loop_start = b.const_i32(0)
    if peel_h0:
        if not pipeline_tiles:
            stage_all(first_tile)
            b.sync_lds_only()
        loop_init = _emit_scan_body(
            sc,
            sc.load_state_from_lds(h0_lds),
            first_tile,
            chunk_n=chunk0_n if io is not None else None,
            tile_prefetch=None,
        )
        loop_start = b.const_i32(1)
        if pipeline_tiles:
            # Keep chunk zero's large next-tile vectors out of the h0 live
            # range. Stage chunk one only after h0 has become the carried state.
            next_n = b.select(
                b.cmp_gt(nc, b.const_i32(1)),
                b.const_i32(1),
                b.const_i32(0),
            )
            next_tile = b.add(first_tile, next_n)
            stage_all(next_tile)
            b.sync_lds_only()

    loop = b.scf_for_iter(
        loop_start,
        nc,
        b.const_i32(1),
        [(f"s{ti}", loop_init[ti]) for ti in range(sc.NS)],
        iv_name="chunk",
        elide_trailing_barrier=False,
    )
    with loop as (n, carried):
        tile = b.add(b.mul(bh, nc), n)
        tile_prefetch = None
        if pipeline_tiles:
            # Clamp the last iteration to a valid tile.  One redundant prefetch
            # is cheaper than a divergent branch in every recurrence stream.
            nxt = b.add(n, b.const_i32(1))
            nxt = b.select(b.cmp_gt(nc, nxt), nxt, n)
            tile_prefetch = _TilePrefetch(b.add(b.mul(bh, nc), nxt))
        else:
            stage_all(tile)
            b.sync_lds_only()
        b.scf_yield(
            *_emit_scan_body(
                sc,
                list(carried),
                tile,
                chunk_n=n if io is not None else None,
                tile_prefetch=tile_prefetch,
            )
        )

    if spec.store_final_state:
        sc.store_state(ht_ptr, state_base, loop.results)

    b.ret()
    return b.kernel


def kda_chunk_scan_grid(spec: KdaChunkScanSpec, bh: int) -> Tuple[int, int, int]:
    """One workgroup per (batch, head), times ``value_splits`` v bands."""
    return (int(bh) * int(spec.value_splits), 1, 1)


def kda_chunk_scan_signature(spec: KdaChunkScanSpec):
    sb = (
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
    )
    if spec.token_major_io:
        sb = sb.scalar("batch", "i32").scalar("heads", "i32").scalar("tseq", "i32")
    return sb.build()


def kda_chunk_prep_grid(spec: KdaChunkPrepSpec, num_tiles: int) -> Tuple[int, int, int]:
    """One workgroup per chunk. ``num_tiles = BH * NC``."""
    return (int(num_tiles), 1, 1)


def kda_chunk_prep_signature(spec: KdaChunkPrepSpec):
    sb = (
        SignatureBuilder()
        .ptr("q_ptr", spec.dtype)
        .ptr("k_ptr", spec.dtype)
        .ptr("g_ptr", "bf16" if spec.raw_inputs else "f32")
        .ptr(
            "beta_ptr",
            ("f32" if not spec.raw_inputs or spec.fp32_beta_dtype else spec.dtype),
        )
        .ptr("a_ptr", spec.dtype)
        .ptr("gk_ptr", spec.dtype)
        .ptr("gq_ptr", spec.dtype)
        .ptr("aqk_ptr", spec.dtype)
        .ptr("kt_ptr", spec.dtype)
        .ptr("dec_ptr", "f32")
        .scalar("scale", "f32")
    )
    if spec.raw_inputs:
        sb = (
            sb.ptr("a_log_ptr", "f32")
            .ptr("dt_bias_ptr", "f32")
            .scalar("batch", "i32")
            .scalar("heads", "i32")
            .scalar("tseq", "i32")
            .scalar("nc", "i32")
            .scalar("beta_stride_batch", "i32")
            .scalar("beta_stride_token", "i32")
            .scalar("beta_stride_head", "i32")
        )
    return sb.build()


__all__ = [
    "EXP2_CLAMP",
    "KDA_CHUNK_SIZES",
    "KDA_DTYPES",
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
    "tuned_kda_chunk_scan_spec",
]
