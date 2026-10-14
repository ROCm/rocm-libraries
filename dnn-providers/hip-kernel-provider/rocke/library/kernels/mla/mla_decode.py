# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MLA decode kernel, weight-absorbed: gfx942 (CDNA3) and gfx950 (CDNA4).

Decode is one query per sequence, so the query tile that prefill puts on the
MFMA M/N axis is gone. What fills it instead is the **head** axis: every head
of an absorbed MLA layer attends to the *same* 576-wide key row
``[c_kv | k_rope]``, so ``BLOCK_H`` heads of one sequence form the GEMM's head
dimension and share every staged key tile.

Contract (the same tensors as the prefill ``absorbed_query`` kernel)::

    qa     [total_q, H, r_kv + d_rope]   Qa = [Q_nope @ W_UK_nopeᵀ | Q_rope]
    c_kv   [num_pages, page, r_kv]       latent cache
    k_rope [num_pages, page, d_rope]     head-shared RoPE keys
    out    [total_q, H, r_kv]            normalized latent, before W_UV
    lse    [total_q, H]                  natural log

Both projections, ``W_UK_nope`` into the query and ``W_UV`` out of the latent,
are batched GEMMs outside the kernel: with heads on the MFMA axis, per-head
weights vary along that axis and cannot be in-wave MFMA operands.

Layout. A workgroup owns ``BLOCK_H = heads_per_wave * num_warps`` consecutive
heads of one sequence and one key segment; wave ``w`` owns 16 of those heads
outright. Per key tile::

    kv_lds[:, :r_kv] <- c_kv[page]          [Bk, r_kv]   shared by every wave
    kv_lds[:, r_kv:] <- k_rope[page]        [Bk, d_rope]
    Sᵀ   <- [C | K_rope] · Qaᵀ              m = key, n = head, K = 576
    P    <- online_softmax(Sᵀ)              per head = per n column
    accᵀ += Cᵀ · Pᵀ                         m = latent r, n = head

This is the one-wave-per-head prefill pipeline (``_build_mla_prefill_fwd_heads``
with ``absorbed_query``) with its n axis relabelled from query to head, and its
fragment layouts carry over unchanged: the absorbed query is read straight
into the score's B layout, the accumulator never leaves registers, and the PV
reads the latent from the natural ``kv_lds`` tile (the read-path gather the
prefill measured ahead of a second, transposed copy). Waves split heads only:
the latent is the score's K axis *and* the PV's N axis, so splitting it
across waves would turn the score into a split-K with a cross-wave combine on
every tile.

Masking. One query sees its whole cache (the bottom-right causal mask admits
every key at ``S_q = 1``), so the only mask is the sequence's last, partly
filled page.

Split-KV. With ``num_seg > 1`` the grid's z axis cuts each sequence's key
tiles into ``num_seg`` contiguous runs and the kernel writes unnormalized
fp32 partials ``(acc, m, l)`` to a workspace instead of the latent; the
arch-neutral :mod:`kernels.mla.mla_decode_combine` merges them. ``num_seg``
itself is a kernel argument, so one compiled split kernel serves every
segment count -- only ``num_seg > 1`` versus ``== 1`` changes the IR.

Per arch. :class:`MlaDecodeSpec` is the gfx942 kernel and the shared spec;
:class:`kernels.mla.mla_decode_gfx950.MlaDecodeGfx950Spec` is its gfx950
subclass. The class, not a field, fixes the two arch-specific emission
choices, so a spec can never name one arch's IR with the other's kernel name:

* the score atom: ``mfma_f32_16x16x16_bf16`` on gfx942 (CDNA3 has no wide-K
  bf16 atom), ``mfma_f32_16x16x32_bf16`` on gfx950, which halves the score's
  MFMA count and reads 16 B operand fragments;
* the PV's latent operand: the read-path gather from the natural ``kv_lds``
  tile on gfx942 (no transpose read there), ``ds_read_tr16_b64`` on gfx950,
  which hands each lane the transposed fragment directly and keeps the latent
  tiles in natural order.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import ClassVar

from rocke.core.ir import BF16, F32, I32, I64, IRBuilder, KernelDef, PtrType
from rocke.helpers.attention import (
    mfma_16x16x16_for_dtype as _mfma_16x16x16,
    mfma_16x16x32_for_dtype as _mfma_16x16x32,
    safe_inv_l as _safe_inv_l,
)
from rocke.helpers.layouts import TransposeLdsReader
from rocke.helpers.spec import SignatureBuilder

from .mla_decode_combine import NEG_INIT, MlaDecodeCombineSpec

# Shared with the prefill kernel on purpose: these helpers and constants are
# its own, imported rather than copied so that both kernels' IR moves together.
# Once the prefill lands, they can move to a common kernels/mla module and lose
# their leading underscores.
from .mla_prefill_gfx942 import (
    C_PER_LANE,
    C_STAGE_W,
    HPW_AGPR_ALLOC,
    HPW_LAZY_TAU,
    HPW_PV_PREFETCH,
    HPW_SCORE_FENCE_EVERY,
    HPW_SCORE_PREFETCH,
    LDS_BYTES_PER_WORKGROUP,
    MFMA_K,
    MFMA_N,
    V8_PAD,
    WAVE_SIZE,
    WT_PAD,
    _bf16x4_rne_finite,
    _pick4,
    _query_reduce,
    _shift,
)

__all__ = [  # noqa: RUF022  -- ordered spec -> gate -> launch geometry -> build
    "MlaDecodeSpec",
    "supports_mla_decode",
    "require_mla_decode",
    "mla_decode_block",
    "mla_decode_grid",
    "mla_decode_signature",
    "mla_decode_workspace_shapes",
    "mla_decode_combine_spec",
    "build_mla_decode",
]


@dataclass(frozen=True)
class MlaDecodeSpec:
    """Immutable description of one MLA decode configuration (gfx942).

    Only ``num_heads`` is problem shape. ``num_seg`` is a launch parameter
    chosen per request by dispatch: it sizes the grid's z axis and the
    workspace, but only ``num_seg > 1`` versus ``== 1`` changes the emitted
    kernel (:attr:`split_kv`), so the kernel name records that and not the
    count.
    """

    #: Target this spec class emits for; also the kernel name's stem.
    ARCH: ClassVar[str] = "gfx942"
    #: K of the score MFMA atom: 16 (``16x16x16``) or 32 (``16x16x32``).
    SCORE_K: ClassVar[int] = 16
    #: PV latent operand via ``ds_read_tr16_b64`` rather than the gather.
    PV_TRANSPOSE_READ: ClassVar[bool] = False
    #: Per-workgroup LDS the admission check allows.
    LDS_BYTES: ClassVar[int] = LDS_BYTES_PER_WORKGROUP

    num_heads: int
    d_rope: int = 64
    r_kv: int = 512
    block_k: int = 16
    page_block_size: int = 16
    # Heads on the MFMA n axis of one wave. 16 is the bf16 16x16x16 atom's n:
    # a 16-head x 512-latent fp32 accumulator is 128 VGPRs, which leaves room
    # for the absorbed query without AGPRs.
    heads_per_wave: int = 16
    num_warps: int = 4
    num_seg: int = 1
    dtype: str = "bf16"

    def __post_init__(self) -> None:
        """Reject specs that are impossible on *any* arch.

        Arch- and atom-specific admission stays in :func:`supports_mla_decode`,
        which reports rather than raises so a dispatcher can fall through.
        """
        if self.dtype != "bf16":
            raise ValueError(f"MLA decode supports bf16 only, got dtype={self.dtype!r}")
        for name in (
            "num_heads",
            "d_rope",
            "r_kv",
            "block_k",
            "page_block_size",
            "heads_per_wave",
            "num_seg",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError(f"num_warps must be one of 1/2/4/8, got {self.num_warps}")
        if self.page_block_size != self.block_k:
            # One k-tile must land inside exactly one page, otherwise the tile
            # straddles two block_table entries and the single-page gather is
            # wrong.
            raise ValueError(
                "page_block_size must equal block_k so a k-tile maps to one page, "
                f"got page_block_size={self.page_block_size}, block_k={self.block_k}"
            )

    @property
    def block_h(self) -> int:
        """Heads per workgroup."""
        return self.heads_per_wave * self.num_warps

    @property
    def qa_cols(self) -> int:
        """Width of one absorbed query / key row: the latent then the RoPE half."""
        return self.r_kv + self.d_rope

    @property
    def threads(self) -> int:
        return WAVE_SIZE * self.num_warps

    @property
    def split_kv(self) -> bool:
        return self.num_seg > 1

    @property
    def lds_pad(self) -> int:
        """Trailing bf16 columns on ``kv_lds`` / ``qr_lds`` rows.

        ``WT_PAD`` (see the prefill kernel) for 8 B score reads. The 16 B reads
        of the wide-K atom need every row 16 B-aligned, hence ``V8_PAD``.
        """
        return V8_PAD if self.SCORE_K == 32 else WT_PAD

    def identity(self) -> dict:
        """What the compiled binary depends on, for dispatch's spec hash.

        Every field but ``num_seg``, which is a launch argument (only
        :attr:`split_kv` reaches the IR), plus the class's emission choices,
        which ``asdict`` cannot see: a gfx942 and a gfx950 spec with equal
        fields are different binaries.
        """
        fields = dataclasses.asdict(self)
        del fields["num_seg"]
        return {
            **fields,
            "arch": self.ARCH,
            "score_k": self.SCORE_K,
            "pv_transpose_read": self.PV_TRANSPOSE_READ,
            "split_kv": self.split_kv,
        }

    def kernel_name(self) -> str:
        return (
            f"mla_decode_{self.ARCH}"
            f"_h{self.num_heads}"
            f"_rkv{self.r_kv}r{self.d_rope}"
            f"_k{self.block_k}"
            f"_hw{self.heads_per_wave}"
            f"_w{self.num_warps}"
            + ("_split" if self.split_kv else "")
            + f"_{self.dtype}"
        )


def _lds_bytes(spec: MlaDecodeSpec) -> int:
    """The LDS pool :func:`build_mla_decode` asks for.

    Two buffers live through the whole loop: the shared key tile and every
    wave's parked ``Q_rope``. The lowerer rounds the pool up to 16 B.
    """
    elem = 2  # bf16
    kv_lds = elem * spec.block_k * (spec.qa_cols + spec.lds_pad)
    qr_lds = elem * spec.block_h * (spec.d_rope + spec.lds_pad)
    return -(-(kv_lds + qr_lds) // 16) * 16


def _kr_stage_width(spec: MlaDecodeSpec) -> int:
    """``k_rope`` staging chunk: a quarter of the c_kv tile, so at eight waves a
    4-wide chunk per thread would over-cover it and the chunk narrows to 2."""
    return 4 if (spec.block_k * spec.d_rope) % (spec.threads * 4) == 0 else 2


def supports_mla_decode(spec: MlaDecodeSpec, *, arch: str) -> tuple[bool, str]:
    """Admission check for the MLA decode kernel ``spec``'s class emits.

    Returns ``(ok, reason)``; ``reason`` is empty when ``ok``. Every rejection
    names the offending field and what would be accepted.
    """
    if arch.split(":")[0] != spec.ARCH:
        return False, (
            f"{type(spec).__name__} emits the {spec.ARCH} MLA decode kernel, got "
            f"arch={arch!r}"
        )
    if spec.heads_per_wave != MFMA_N:
        return False, (
            f"heads_per_wave must be the MFMA n width {MFMA_N}, got "
            f"{spec.heads_per_wave}"
        )
    if spec.num_heads % spec.block_h != 0:
        return False, (
            f"num_heads={spec.num_heads} must be a multiple of BLOCK_H = "
            f"heads_per_wave x num_warps = {spec.block_h}"
        )
    if spec.block_k % MFMA_N != 0:
        return False, f"block_k must be a multiple of {MFMA_N}, got {spec.block_k}"
    if spec.d_rope % spec.SCORE_K != 0:
        return False, (
            f"d_rope must be a multiple of the score atom's K={spec.SCORE_K}, got "
            f"{spec.d_rope}"
        )
    # The PV gathers four latent tiles per read and the epilogue stores the
    # latent in 64-wide blocks.
    if spec.r_kv % 64 != 0:
        return False, f"r_kv must be a multiple of 64, got {spec.r_kv}"
    threads = spec.threads
    c_chunks_per_row = spec.r_kv // C_STAGE_W
    kr_w = _kr_stage_width(spec)
    kr_chunks_per_row = spec.d_rope // kr_w
    stages = {
        "c_kv tile": (spec.block_k * spec.r_kv, C_STAGE_W),
        "k_rope tile": (spec.block_k * spec.d_rope, kr_w),
    }
    for name, (count, width) in stages.items():
        if count % (threads * width) != 0:
            return False, (
                f"{name} has {count} elements, which does not divide evenly "
                f"across {threads} threads at {width} elements per chunk"
            )
    # Every staging chunk of a thread is its first one shifted by whole rows.
    for name, per_row in (
        ("c_kv", c_chunks_per_row),
        ("k_rope", kr_chunks_per_row),
    ):
        if threads % per_row != 0:
            return False, (
                f"{name} staging needs the {threads} threads to cover whole rows of "
                f"{per_row} chunks"
            )
    lds = _lds_bytes(spec)
    if lds > spec.LDS_BYTES:
        return False, (
            f"the decode kernel needs {lds} B of LDS, over the "
            f"{spec.LDS_BYTES} B per-workgroup budget"
        )
    return True, ""


def require_mla_decode(spec: MlaDecodeSpec, *, arch: str) -> None:
    """Raise :class:`NotImplementedError` if ``spec``/``arch`` is unsupported."""
    ok, reason = supports_mla_decode(spec, arch=arch)
    if not ok:
        raise NotImplementedError(reason)


def mla_decode_block(spec: MlaDecodeSpec) -> tuple[int, int, int]:
    """Workgroup shape: one flat dimension of ``64 * num_warps`` threads."""
    return (spec.threads, 1, 1)


def mla_decode_grid(spec: MlaDecodeSpec, *, num_seqs: int) -> tuple[int, int, int]:
    """``(head groups, sequences, key segments)``.

    Head groups vary fastest, so the workgroups that read the same sequence's
    key tiles are dispatched together. A sequence with no query exits at once.
    """
    if num_seqs <= 0:
        raise ValueError(f"num_seqs must be positive, got {num_seqs}")
    return (spec.num_heads // spec.block_h, num_seqs, spec.num_seg)


def mla_decode_signature(spec: MlaDecodeSpec):
    """Launch signature, in declaration order.

    With ``num_seg == 1`` this is the prefill ``absorbed_query`` kernel's
    12-argument pack, unchanged: a caller holds one set of tensors whichever
    kernel dispatch picks. ``scale`` is the raw host scale (``log2(e)`` is
    folded in-kernel) and ``max_k`` is unused but kept for that symmetry.

    With ``num_seg > 1`` the two outputs are the fp32 workspace instead (see
    :func:`mla_decode_workspace_shapes`) and ``num_seg`` is appended.
    """
    sig = SignatureBuilder()
    if spec.split_kv:
        sig = sig.ptr("part_acc_ptr", "f32").ptr("part_ml_ptr", "f32")
    else:
        sig = sig.ptr("out_ptr", spec.dtype).ptr("lse_ptr", "f32")
    sig = (
        sig.ptr("qa_ptr", spec.dtype)
        .ptr("c_kv_ptr", spec.dtype)
        .ptr("k_rope_ptr", spec.dtype)
        .ptr("cu_seqlens_q_ptr", "i32")
        .ptr("cu_seqlens_k_ptr", "i32")
        .ptr("block_table_ptr", "i32")
        .scalar("scale", "f32")
        .scalar("num_seqs", "i32")
        .scalar("block_table_stride", "i32")
        .scalar("max_k", "i32")
    )
    if spec.split_kv:
        sig = sig.scalar("num_seg", "i32")
    return sig.build()


def mla_decode_workspace_shapes(
    spec: MlaDecodeSpec, *, total_q: int
) -> dict[str, tuple[int, ...]]:
    """fp32 workspace of the split kernel, keyed by argument name.

    ``part_acc`` is the unnormalized latent accumulator and ``part_ml`` holds
    ``(m, l)`` per (token, segment, head), ``m`` in log2 units. Empty with
    ``num_seg == 1``. The kernels index it with i32 element offsets, so
    ``total_q * num_seg * num_heads * r_kv`` must stay below ``2**31``;
    dispatch clamps ``num_seg`` to keep it there.
    """
    if not spec.split_kv:
        return {}
    return {
        "part_acc_ptr": (total_q, spec.num_seg, spec.num_heads, spec.r_kv),
        "part_ml_ptr": (total_q, spec.num_seg, spec.num_heads, 2),
    }


def mla_decode_combine_spec(spec: MlaDecodeSpec) -> MlaDecodeCombineSpec:
    """The combine that merges ``spec``'s split-KV partials.

    The two kernels share the workspace layout, so the head count, latent width
    and output dtype must agree; this is the one place that says so.
    """
    return MlaDecodeCombineSpec(
        num_heads=spec.num_heads, r_kv=spec.r_kv, dtype=spec.dtype
    )


def build_mla_decode(spec: MlaDecodeSpec, *, arch: str | None = None) -> KernelDef:
    """Emit the MLA decode latent kernel (single pass, or split-KV partials).

    ``arch`` defaults to the spec class's own target.
    """
    arch = spec.ARCH if arch is None else arch
    require_mla_decode(spec, arch=arch)
    dtype = BF16
    BK = spec.block_k
    D_ROPE = spec.d_rope
    R_KV = spec.r_kv
    QA_COLS = spec.qa_cols
    PAGE = spec.page_block_size
    THREADS = spec.threads
    H = spec.num_heads
    HPW = spec.heads_per_wave
    BLOCK_H = spec.block_h
    SPLIT = spec.split_kv

    # Score atom: K = SK per MFMA, SW bf16 operand elements per lane.
    SK = spec.SCORE_K
    SW = SK // 4
    score_mfma = _mfma_16x16x32 if SK == 32 else _mfma_16x16x16
    TR_PV = spec.PV_TRANSPOSE_READ
    PAD = spec.lds_pad
    # The score read-ahead and fence spacing are the prefill's, held constant
    # in latent *columns*: a K = 32 k-step's fragments are twice as wide, and
    # the prefill's 8-step read-ahead at that width spills.
    SCORE_PREFETCH = HPW_SCORE_PREFETCH * MFMA_K // SK
    SCORE_FENCE_EVERY = max(1, HPW_SCORE_FENCE_EVERY * MFMA_K // SK)

    K_TILES = BK // MFMA_N
    SCORE_K_ITERS = QA_COLS // SK
    N_LAT_STEPS = R_KV // SK
    ROPE_K_ITERS = D_ROPE // SK
    N_R_TILES = R_KV // MFMA_N

    LOG2E = math.log2(math.e)
    LN2 = math.log(2.0)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS
    # No AGPRs: the 128-VGPR accumulator stays in the VGPR file, where the
    # per-tile rescale does not pay an accvgpr read/write pair per register.
    b.kernel.attrs["agpr_alloc"] = HPW_AGPR_ALLOC

    if SPLIT:
        part_acc = b.param(
            "part_acc_ptr",
            PtrType(F32, "global"),
            noalias=True,
            writeonly=True,
            align=16,
        )
        part_ml = b.param(
            "part_ml_ptr", PtrType(F32, "global"), noalias=True, writeonly=True, align=8
        )
    else:
        out = b.param(
            "out_ptr", PtrType(dtype, "global"), noalias=True, writeonly=True, align=16
        )
        lse = b.param(
            "lse_ptr", PtrType(F32, "global"), noalias=True, writeonly=True, align=16
        )
    qa_ptr = b.param(
        "qa_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    c_kv = b.param(
        "c_kv_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    k_rope = b.param(
        "k_rope_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    cu_q = b.param("cu_seqlens_q_ptr", PtrType(I32, "global"), readonly=True, align=4)
    cu_k = b.param("cu_seqlens_k_ptr", PtrType(I32, "global"), readonly=True, align=4)
    block_table = b.param(
        "block_table_ptr", PtrType(I32, "global"), readonly=True, align=4
    )
    scale_p = b.param("scale", F32)
    b.param("num_seqs", I32)  # the grid's y extent; kept for the shared pack
    bt_stride_p = b.param("block_table_stride", I32)
    b.param("max_k", I32)  # unused; kept for the shared pack
    num_seg_p = b.param("num_seg", I32) if SPLIT else None

    # ---- prologue: identify (head group, sequence, segment) ----------------
    head_grp = b.block_id_x()
    seq_idx = b.block_id_y()
    tid = b.thread_id_x()
    lane = b.mod(tid, b.const_i32(WAVE_SIZE))
    wave = b.div(tid, b.const_i32(WAVE_SIZE))
    lane_row = b.mod(lane, b.const_i32(16))
    # lane_kq: the C fragment's row group (keys of the score, latent rows of
    # the PV). lane_ks: the lane's first K column of a score operand -- the
    # same value for the 16x16x16 atom, twice it for the K-packed 16x16x32.
    lane_kq = b.mul(b.div(lane, b.const_i32(16)), b.const_i32(4))
    lane_ks = (
        lane_kq if SW == 4 else b.mul(b.div(lane, b.const_i32(16)), b.const_i32(SW))
    )
    # The wave's first head is wave-uniform (an SGPR); the lane's own head is
    # the n index of every fragment below, lane % 16.
    h0 = b.readfirstlane(
        b.add(b.mul(head_grp, b.const_i32(BLOCK_H)), b.mul(wave, b.const_i32(HPW)))
    )
    head = b.add(h0, lane_row)

    token = b.global_load_i32(cu_q, seq_idx)
    q_stop = b.global_load_i32(cu_q, b.add(seq_idx, b.const_i32(1)))
    # A sequence with no query this step: nothing to write.
    with b.scf_if(b.cmp_ge(token, q_stop)):
        b.ret()

    cu_k_start = b.global_load_i32(cu_k, seq_idx)
    cu_k_stop = b.global_load_i32(cu_k, b.add(seq_idx, b.const_i32(1)))
    s_k = b.sub(cu_k_stop, cu_k_start)
    n_k_tiles = b.div(b.add(s_k, b.const_i32(BK - 1)), b.const_i32(BK))

    def _min(x, y):
        return b.select(b.cmp_lt(x, y), x, y)

    if SPLIT:
        seg = b.block_id_z()
        per_seg = b.div(b.add(n_k_tiles, b.sub(num_seg_p, b.const_i32(1))), num_seg_p)
        t_lo = _min(b.mul(seg, per_seg), n_k_tiles)
        t_hi = _min(b.add(t_lo, per_seg), n_k_tiles)
    else:
        t_lo = b.const_i32(0)
        t_hi = n_k_tiles
    # Tiles wholly inside S_k run unmasked; at most one tail tile follows.
    t_full = _min(b.div(s_k, b.const_i32(BK)), t_hi)
    t_full = b.select(b.cmp_lt(t_full, t_lo), t_lo, t_full)

    # ---- absorbed query, read straight into the score's B layout ------------
    # SW columns at SK t + lane_ks of head lane % 16: exactly the B operand of
    # score k-step t, so there is no transform at all.
    q_base = b.add(
        b.mul(b.add(b.mul(token, b.const_i32(H)), head), b.const_i32(QA_COLS)),
        lane_ks,
    )
    qa_frags = [
        b.global_load_vN(qa_ptr, _shift(b, t * SK, q_base), dtype, SW)
        for t in range(N_LAT_STEPS)
    ]
    rope_frags = [
        b.global_load_vN(qa_ptr, _shift(b, R_KV + j * SK, q_base), dtype, SW)
        for j in range(ROPE_K_ITERS)
    ]
    # Q_rope is parked in LDS and re-read every key tile, as in the prefill:
    # the VGPRs it would pin are what keeps the loop inside 256. Each lane
    # reads back exactly what it wrote, so no barrier is involved.
    qr_lds = b.smem_alloc(dtype, [BLOCK_H, D_ROPE + PAD], name_hint="qr_lds")
    qr_row = b.add(b.mul(wave, b.const_i32(HPW)), lane_row)
    for j in range(ROPE_K_ITERS):
        b.smem_store_vN(qr_lds, [qr_row, _shift(b, j * SK, lane_ks)], rope_frags[j], SW)

    scale_log2 = b.fmul(scale_p, b.const_f32(LOG2E))
    neg_init = b.const_f32(NEG_INIT)
    zero_f = b.const_f32(0.0)

    iter_args = [("m", neg_init), ("l", zero_f)]
    iter_args += [(f"acc{t}", b.zero_vec_f32(4)) for t in range(N_R_TILES)]

    if TR_PV:
        # ds_read_tr16_b64 lane addressing for a K = 16 (one key tile) operand.
        tr = TransposeLdsReader(K=MFMA_K).bind(b, lane)
    else:
        lane_row4 = b.mul(lane_row, b.const_i32(4))
    kv_lds = b.smem_alloc(dtype, [BK, QA_COLS + PAD], name_hint="kv_lds")

    c_chunks = (BK * R_KV) // (THREADS * C_STAGE_W)
    c_chunks_per_row = R_KV // C_STAGE_W
    KR_W = _kr_stage_width(spec)
    kr_chunks = (BK * D_ROPE) // (THREADS * KR_W)
    kr_chunks_per_row = D_ROPE // KR_W
    # Chunk j of a thread is chunk 0 shifted by whole rows, so every chunk
    # shares one per-thread offset and the shift is a constant (SGPR soffset
    # for the loads, immediate for the LDS stores).
    C_ROWS_PER_CHUNK = THREADS // c_chunks_per_row
    KR_ROWS_PER_CHUNK = THREADS // kr_chunks_per_row
    C_PAGE_BYTES = PAGE * R_KV * 2
    KR_PAGE_BYTES = PAGE * D_ROPE * 2
    c_m0 = b.div(tid, b.const_i32(c_chunks_per_row))
    c_r0 = b.mul(b.mod(tid, b.const_i32(c_chunks_per_row)), b.const_i32(C_STAGE_W))
    c_voff = b.mul(b.add(b.mul(c_m0, b.const_i32(R_KV)), c_r0), b.const_i32(2))
    kr_m0 = b.div(tid, b.const_i32(kr_chunks_per_row))
    kr_d0 = b.mul(b.mod(tid, b.const_i32(kr_chunks_per_row)), b.const_i32(KR_W))
    kr_voff = b.mul(b.add(b.mul(kr_m0, b.const_i32(D_ROPE)), kr_d0), b.const_i32(2))

    bt_row = b.mul(seq_idx, bt_stride_p)

    def _load_page(tile, *, guard=None):
        page = b.global_load_i32(block_table, b.add(bt_row, tile))
        if guard is not None:
            # A zero-trip segment's slot may be uninitialised; page 0 is always
            # mapped, and its tile is never used.
            page = b.select(guard, page, b.const_i32(0))
        return page

    def _stage_loads(page):
        # The page's base is a 64-bit pointer (scalar math: a kernel argument
        # plus a wave-uniform page), and each descriptor bounds that one page.
        # An i32 byte offset from the cache base would wrap past 2 GiB of cache.
        page64 = b.zext(b.readfirstlane(page), I64)
        c_rsrc = b.buffer_rsrc(
            b.global_ptr_add(c_kv, b.mul(page64, b.const_i64(C_PAGE_BYTES))),
            b.const_i32(C_PAGE_BYTES),
        )
        kr_rsrc = b.buffer_rsrc(
            b.global_ptr_add(k_rope, b.mul(page64, b.const_i64(KR_PAGE_BYTES))),
            b.const_i32(KR_PAGE_BYTES),
        )
        staged = [
            b.buffer_load_vN(
                c_rsrc,
                c_voff,
                b.const_i32(j * C_ROWS_PER_CHUNK * R_KV * 2),
                dtype,
                C_STAGE_W,
            )
            for j in range(c_chunks)
        ]
        staged += [
            b.buffer_load_vN(
                kr_rsrc,
                kr_voff,
                b.const_i32(j * KR_ROWS_PER_CHUNK * D_ROPE * 2),
                dtype,
                KR_W,
            )
            for j in range(kr_chunks)
        ]
        return staged

    def _stage_store(staged, valid_rows=None):
        for j in range(c_chunks):
            m = _shift(b, j * C_ROWS_PER_CHUNK, c_m0)
            v = staged[j]
            if valid_rows is not None:
                # Rows past S_k on the tail page hold whatever the cache does.
                # The score masks them, but the PV contracts over keys, where
                # P = 0 times a stale NaN or Inf is still NaN for every head.
                v = b.select(b.cmp_lt(m, valid_rows), v, b.zero_vec(dtype, C_STAGE_W))
            b.smem_store_vN(kv_lds, [m, c_r0], v, C_STAGE_W)
        kr_col = b.add(kr_d0, b.const_i32(R_KV))
        for j in range(kr_chunks):
            m = _shift(b, j * KR_ROWS_PER_CHUNK, kr_m0)
            b.smem_store_vN(kv_lds, [m, kr_col], staged[c_chunks + j], KR_W)

    # Page lookups never index past the segment, and an empty segment (or
    # S_k = 0) stages page 0 instead of trusting its unused slot.
    has_tiles = b.cmp_lt(t_lo, t_hi)
    last_tile = b.select(has_tiles, b.sub(t_hi, b.const_i32(1)), b.const_i32(0))
    first_tile = b.select(has_tiles, t_lo, b.const_i32(0))
    prologue = _stage_loads(_load_page(first_tile, guard=has_tiles))
    n_stage = len(prologue)
    iter_args += [(f"stg{j}", v) for j, v in enumerate(prologue)]
    iter_args += [
        (
            "pgn",
            _load_page(
                _min(b.add(first_tile, b.const_i32(1)), last_tile), guard=has_tiles
            ),
        )
    ]

    def _k_tile_body(k_tile, state, *, masked: bool):
        m_run = state[0]
        l_run = state[1]
        accs = list(state[2 : 2 + N_R_TILES])
        staged = list(state[-(n_stage + 1) : -1])
        pg_next = state[-1]
        kb_start = b.mul(k_tile, b.const_i32(BK))

        b.sync()  # WAR: every wave's readers of the previous tile
        _stage_store(staged, b.sub(s_k, kb_start) if masked else None)
        b.sync()

        pg_after = _load_page(_min(b.add(k_tile, b.const_i32(2)), last_tile))

        # ---- Sᵀ = [C | K_rope] · Qaᵀ, the wave's whole K reduction ----------
        # Software-pipelined as in the prefill (SCORE_PREFETCH): k-step
        # kk + D's LDS reads are emitted before k-step kk's MFMA, and the fences
        # stop the scheduler from sinking them back down.
        s_acc = [b.zero_vec_f32(4) for _ in range(K_TILES)]

        def _score_operands(kk):
            if kk < N_LAT_STEPS:
                q_v = qa_frags[kk]
            else:
                q_v = b.smem_load_vN(
                    qr_lds,
                    qr_row,
                    _shift(b, (kk - N_LAT_STEPS) * SK, lane_ks),
                    dtype=dtype,
                    n=SW,
                )
            col = _shift(b, kk * SK, lane_ks)
            k_vs = [
                b.smem_load_vN(
                    kv_lds, _shift(b, kt * MFMA_N, lane_row), col, dtype=dtype, n=SW
                )
                for kt in range(K_TILES)
            ]
            return q_v, k_vs

        score_pre = {
            kk: _score_operands(kk) for kk in range(min(SCORE_PREFETCH, SCORE_K_ITERS))
        }
        for kk in range(SCORE_K_ITERS):
            if kk + SCORE_PREFETCH < SCORE_K_ITERS:
                score_pre[kk + SCORE_PREFETCH] = _score_operands(kk + SCORE_PREFETCH)
            q_v, k_vs = score_pre.pop(kk)
            for kt in range(K_TILES):
                s_acc[kt] = score_mfma(b, dtype, k_vs[kt], q_v, s_acc[kt])
            if (kk + 1) % SCORE_FENCE_EVERY == 0:
                b.sched_barrier(0)

        # The next tile's staging loads go out only now: issued before the
        # score, their registers would be live across it.
        staged_next = _stage_loads(pg_next)

        # ---- PV read groups ---------------------------------------------------
        # A group is the four latent tiles 4 w .. 4 w + 3 of one key tile.
        # Gather (gfx942): row m of latent tile 4 w + e is r = 64 w + 4 m + e,
        # so one 8 B read of a key row hands a lane its element of four tiles
        # (the prefill's PV permutation; the epilogue stores through the same
        # one). Transpose read (gfx950): ds_read_tr16_b64 hands a lane the
        # fragment of one tile in natural order, r = 16 t + m.
        pv_groups = [(kt, w) for kt in range(K_TILES) for w in range(N_R_TILES // 4)]

        def _pv_reads(kt, w):
            if TR_PV:
                row = tr.row(b, k_offset=kt * MFMA_N)
                return [
                    b.ds_read_tr16_b64(
                        kv_lds,
                        row,
                        _shift(b, (4 * w + e) * MFMA_N, tr.col),
                        dtype=dtype,
                    )
                    for e in range(4)
                ]
            a_col = _shift(b, kt * MFMA_N, lane_kq)
            return [
                b.smem_load_vN(
                    kv_lds,
                    _shift(b, i, a_col),
                    _shift(b, 64 * w, lane_row4),
                    dtype=dtype,
                    n=4,
                )
                for i in range(4)
            ]

        pv_pre = {
            gi: _pv_reads(*pv_groups[gi])
            for gi in range(min(HPW_PV_PREFETCH, len(pv_groups)))
        }

        # ---- online softmax, one head per n column ----------------------------
        # Lane l holds head l % 16 and keys 4 (l // 16) + r of each key tile;
        # _query_reduce folds the four lane groups, i.e. over the keys.
        s_v = []
        keep = []
        for kt in range(K_TILES):
            for r in range(C_PER_LANE):
                s_scaled = b.fmul(b.vec_extract(s_acc[kt], r), scale_log2)
                if masked:
                    k_abs = b.add(kb_start, _shift(b, kt * MFMA_N + r, lane_kq))
                    keep.append(b.cmp_lt(k_abs, s_k))
                    s_v.append(b.select(keep[-1], s_scaled, neg_init))
                else:
                    s_v.append(s_scaled)
        local_max = s_v[0]
        for v in s_v[1:]:
            local_max = b.fmax(local_max, v)
        m_new = b.fmax(m_run, _query_reduce(b, local_max, combine="max"))
        # Lazy rescale (HPW_LAZY_TAU): keep the max in use unless the new one is
        # far enough above it. The partials and the final 1/l use the same max,
        # so a stale one changes nothing.
        need = b.fcmp("ogt", m_new, b.fadd(m_run, b.const_f32(HPW_LAZY_TAU)))
        m_new = b.select(need, m_new, m_run)
        alpha = b.exp2_fast(b.fsub(m_run, m_new))
        if masked:
            p_v = [
                b.select(k, b.exp2_fast(b.fsub(v, m_new)), zero_f)
                for k, v in zip(keep, s_v)
            ]
        else:
            p_v = [b.exp2_fast(b.fsub(v, m_new)) for v in s_v]
        local_sum = p_v[0]
        for v in p_v[1:]:
            local_sum = b.fadd(local_sum, v)
        l_new = b.fadd(b.fmul(l_run, alpha), _query_reduce(b, local_sum, combine="sum"))
        # The accumulator rescale runs only when some lane of the wave moved its
        # max: a zero- or one-trip loop on a wave-uniform count, which carries
        # the accumulator as its iteration arguments. Lanes that did not move
        # have alpha == 1.
        any_need = b.wave_any(b.select(need, b.const_i32(1), b.const_i32(0)))
        rs_loop = b.scf_for_iter(
            b.const_i32(0),
            b.select(
                b.cmp_ne(any_need, b.const_i32(0)), b.const_i32(1), b.const_i32(0)
            ),
            b.const_i32(1),
            iter_args=[(f"rs{int(masked)}_{t}", accs[t]) for t in range(N_R_TILES)],
            iv_name=f"rescale{int(masked)}",
        )
        with rs_loop as (_, rs_state):
            scaled = []
            for t in range(N_R_TILES):
                v = rs_state[t]
                for r in range(C_PER_LANE):
                    v = b.vec_insert(v, b.fmul(b.vec_extract(v, r), alpha), r)
                scaled.append(v)
            b.scf_yield(*scaled)
        accs = list(rs_loop.results)
        p_frags = [
            _bf16x4_rne_finite(b, p_v[kt * C_PER_LANE : (kt + 1) * C_PER_LANE])
            for kt in range(K_TILES)
        ]

        # ---- accᵀ += Cᵀ · Pᵀ over every latent tile ---------------------------
        for gi, (kt, w) in enumerate(pv_groups):
            if gi + HPW_PV_PREFETCH < len(pv_groups):
                pv_pre[gi + HPW_PV_PREFETCH] = _pv_reads(
                    *pv_groups[gi + HPW_PV_PREFETCH]
                )
            xs = pv_pre.pop(gi)
            for e in range(4):
                c_t = xs[e] if TR_PV else _pick4(b, xs, e, dtype)
                accs[4 * w + e] = _mfma_16x16x16(
                    b, dtype, c_t, p_frags[kt], accs[4 * w + e]
                )
            b.sched_barrier(0)

        return [m_new, l_new, *accs, *staged_next, pg_after]

    main_loop = b.scf_for_iter(
        t_lo, t_full, b.const_i32(1), iter_args=iter_args, iv_name="k_main"
    )
    with main_loop as (k_tile, state):
        b.scf_yield(*_k_tile_body(k_tile, state, masked=False))
    tail_loop = b.scf_for_iter(
        t_full,
        t_hi,
        b.const_i32(1),
        iter_args=[
            (f"{name}_t", v) for (name, _), v in zip(iter_args, main_loop.results)
        ],
        iv_name="k_tail",
    )
    with tail_loop as (k_tile, state):
        b.scf_yield(*_k_tile_body(k_tile, state, masked=True))

    # ---- epilogue -----------------------------------------------------------
    res = tail_loop.results
    m_fin = res[0]
    l_fin = res[1]
    accs = list(res[2 : 2 + N_R_TILES])
    # Each store writes four consecutive latent columns of the lane's head:
    # ``(offset from r_lane, [(tile, register) per column])``.
    if TR_PV:
        # Natural order: register reg of tile t holds r = 16 t + 4 g + reg in
        # lane group g, so one tile is one store.
        r_lane = lane_kq
        chunks = [
            (MFMA_N * t, [(t, reg) for reg in range(C_PER_LANE)])
            for t in range(N_R_TILES)
        ]
    else:
        # With the PV's row permutation, register reg of tile 4 w + e holds
        # r = 64 w + 16 g + 4 reg + e in lane group g, so the four tiles of one
        # w hand a lane four consecutive r: one store per (w, reg).
        r_lane = b.mul(lane_kq, b.const_i32(4))
        chunks = [
            (64 * w + 4 * reg, [(4 * w + e, reg) for e in range(4)])
            for w in range(N_R_TILES // 4)
            for reg in range(C_PER_LANE)
        ]
    if SPLIT:
        # Unnormalized partials, relative to this segment's (possibly stale)
        # max. An empty segment writes (acc, m, l) = (0, NEG_INIT, 0).
        row_id = b.add(b.mul(b.add(b.mul(token, num_seg_p), seg), b.const_i32(H)), head)
        row = b.add(b.mul(row_id, b.const_i32(R_KV)), r_lane)
        for off, elems in chunks:
            vals = b.vec_pack([b.vec_extract(accs[t], reg) for t, reg in elems], F32)
            b.global_store_vN(part_acc, _shift(b, off, row), vals, 4, align=16)
        # One writer per head: the wave's first 16-lane group.
        with b.scf_if(b.cmp_lt(lane, b.const_i32(16))):
            b.global_store_vN(
                part_ml,
                b.mul(row_id, b.const_i32(2)),
                b.vec_pack([m_fin, l_fin], F32),
                2,
                align=8,
            )
    else:
        inv = _safe_inv_l(b, l_fin)
        row = b.add(
            b.mul(b.add(b.mul(token, b.const_i32(H)), head), b.const_i32(R_KV)), r_lane
        )
        for off, elems in chunks:
            vals = b.vec_pack(
                [b.fmul(b.vec_extract(accs[t], reg), inv) for t, reg in elems], F32
            )
            b.global_store_vN(
                out, _shift(b, off, row), b.vec_trunc_f32_to_bf16(vals), 4, align=8
            )
        # S_k = 0 leaves l = 0, so log2(l) = -inf and the LSE is -inf.
        with b.scf_if(b.cmp_lt(lane, b.const_i32(16))):
            value = b.fmul(b.fadd(m_fin, b.log2(l_fin)), b.const_f32(LN2))
            b.global_store(
                lse, b.add(b.mul(token, b.const_i32(H)), head), value, align=4
            )

    return b.kernel
