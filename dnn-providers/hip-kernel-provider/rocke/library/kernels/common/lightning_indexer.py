# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Lightning indexer forward (DSA relevance scorer) -- gfx942, bf16, v1.

The lightning indexer is the first stage of DeepSeek Sparse Attention. It is a
score-only kernel: for each query position ``t`` it scores every allowed key
position ``s`` and emits a single relevance score

    I(t, s) = sum over heads h of  w_h * ReLU( index_q[t, h] . index_k[s] )

with no softmax and no value output. The scores feed top-k selection (a separate
stage). The index-query is per head; the index-key is shared across heads
(one D_I vector per token, MQA-style), so ``index_k`` carries no head axis.

Scope of this module (v1):

* Score-only. The projection, RoPE, and index-key cache write are caller-side
  (the reference's fused qk-norm-rope-quant op); this kernel consumes an already
  projected + RoPE'd ``index_q`` and an already populated ``index_k``.
* Scalar reduction body. The per-key dot is a straight FMA reduction over D_I,
  not an MFMA. This is the correctness-first bf16 baseline; the MFMA score body
  is a later perf hoist under the optimization runbook.
* gfx942, bf16 only. fp8 and gfx950 are later phases.

Two seams are kept so later work is additive, not a rewrite:

* ``_emit_score`` is the only place a score reaches memory. The fused
  indexer+top-k kernel replaces this hook with an on-chip top-k tail; the score
  loop above it does not change.
* ``_load_index_q_elem`` / ``_load_index_k_elem`` are the only places inputs are
  read. Folding the pre-step (projection/RoPE) into the kernel later replaces
  these with a project-then-score prologue that produces the same values.

Grid: ``(seqlen_q, 1, 1)`` -- one workgroup per query row. The workgroup's
threads stride over the key range; each thread owns keys ``tid, tid + block,
...`` and writes their scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from rocke.core.ir import BF16, F32, I32, IRBuilder, KernelDef, PtrType
from rocke.helpers.spec import SignatureBuilder, kernel_name_join

# gfx942 has 64 KiB LDS per CU. The scalar v1 body uses no LDS; the ceiling is
# carried so the validator has one place to check once the MFMA/fused forms
# (which do allocate an indexer tile plus a top-k candidate reserve) arrive.
LDS_LIMIT = 64 * 1024
# Causal mask sentinel. A future key must never be selectable by top-k, so its
# score is driven far below any real score rather than merely to zero (a real
# ReLU-weighted score is >= 0, so zero would still be a selection candidate).
NEG_INF_SCORE = -3.0e38
# Declared coverage, exported so a dispatch candidate states what it serves by
# importing these rather than transcribing them.
INDEXER_DTYPES: Tuple[str, ...] = ("bf16",)


@dataclass(frozen=True)
class IndexerTileSpec:
    """Scalar-body tiling knobs.

    v1 is a scalar FMA reduction, so the only knob is how many threads cover a
    query row's key range. Bq / Bk / head-streaming appear with the MFMA score
    body, where the per-head query tile is the binding LDS resource.
    """

    block_size: int = 256
    waves_per_eu: int = 0

    def name_parts(self) -> str:
        wp = "" if self.waves_per_eu == 0 else f"_w{self.waves_per_eu}"
        return f"b{self.block_size}{wp}"


@dataclass(frozen=True)
class IndexerSpec:
    """One lightning-indexer forward configuration.

    ``seqlen_q`` (total query tokens) and ``seqlen_k`` (key length Sk) are
    compile-time so the spec can size the launch grid, matching the tiled FMHA
    spec convention; runtime variability lifts via a derived spec later.
    """

    n_index_heads: int
    index_head_dim: int = 128
    seqlen_q: int = 1
    seqlen_k: int = 1
    dtype: str = "bf16"
    body: str = "scalar"
    tile: IndexerTileSpec = field(default_factory=IndexerTileSpec)
    name: str = "rocke_lightning_indexer"

    def kernel_name(self) -> str:
        parts = [
            self.name,
            f"HI{self.n_index_heads}",
            f"D{self.index_head_dim}",
            self.dtype,
            f"Q{self.seqlen_q}",
            f"K{self.seqlen_k}",
            self.tile.name_parts(),
        ]
        # Only a non-default body appends a tag, so the scalar kernel_name (and
        # its already-blessed golden) stays unchanged.
        if self.body != "scalar":
            parts.append(self.body)
        return kernel_name_join(*parts)

    def lds_bytes(self) -> int:
        # Both bodies use no shared memory: the scalar reduction has no tiles,
        # and the MFMA score body runs register-to-register (score-only, no value
        # path means no P->LDS round-trip). LDS enters only with the fused top-k.
        return 0


def is_valid_spec(spec: IndexerSpec, arch: str = "gfx942") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for one indexer config on ``arch``.

    Supported on the CDNA wave64 targets gfx942 and gfx950 in bf16: the kernel is
    arch-neutral (the 16x16x16 bf16 atom and the register-only MFMA body exist on
    both, and neither body uses LDS). Architecture facts (wave size, LDS capacity)
    are sourced from :class:`rocke.core.arch.ArchTarget` so an unknown or
    unsupported arch (e.g. an RDNA wave32 part) is rejected with a structured
    reason instead of crashing comgr at lower time.
    """
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    if arch not in ("gfx942", "gfx950"):
        return False, f"lightning indexer supports gfx942/gfx950 (got {arch!r})"
    if spec.dtype not in INDEXER_DTYPES:
        return False, f"unsupported dtype {spec.dtype!r} (only bf16 in v1)"
    if spec.n_index_heads <= 0:
        return False, f"n_index_heads must be positive (got {spec.n_index_heads})"
    if spec.index_head_dim <= 0:
        return False, f"index_head_dim must be positive (got {spec.index_head_dim})"
    if spec.seqlen_q <= 0 or spec.seqlen_k <= 0:
        return (
            False,
            f"seqlen_q/seqlen_k must be positive (got {spec.seqlen_q}, {spec.seqlen_k})",
        )
    bs = spec.tile.block_size
    wave = target.wave_size
    if bs <= 0 or bs > 1024:
        return False, f"block_size {bs} outside (0, 1024]"
    if bs % wave != 0:
        return False, f"block_size {bs} not a multiple of wave_size {wave} for {arch}"
    if spec.lds_bytes() > LDS_LIMIT:
        return False, f"lds_bytes {spec.lds_bytes()} exceeds {LDS_LIMIT} on {arch}"
    if spec.body not in ("scalar", "mfma"):
        return False, f"unknown body {spec.body!r} (scalar | mfma)"
    if spec.body == "mfma":
        # The 16x16x16 bf16 atom uses one wave64 and 16-wide Q/K/score tiles, and
        # v1 has no tile-tail masking, so shapes must be 16-aligned. Unaligned
        # requests fall to the scalar candidate.
        if bs != wave:
            return (
                False,
                f"mfma body requires block_size == wave_size {wave} (got {bs})",
            )
        if spec.index_head_dim % 16 != 0:
            return (
                False,
                f"mfma body requires index_head_dim % 16 == 0 (got {spec.index_head_dim})",
            )
        if spec.seqlen_q % 16 != 0 or spec.seqlen_k % 16 != 0:
            return False, (
                f"mfma body requires seqlen_q/seqlen_k multiples of 16 "
                f"(got {spec.seqlen_q}, {spec.seqlen_k})"
            )
    return True, "ok"


def _load_index_q_elem(b: IRBuilder, index_q, base, d):
    """Load one f32-promoted element of the per-head index-query.

    Input seam: folding the pre-step (projection/RoPE) into the kernel later
    replaces this with a value produced on-chip from the hidden state.
    """
    return b.cast_to_f32(b.global_load_bf16(index_q, b.add(base, d)))


def _load_index_k_elem(b: IRBuilder, index_k, base, d):
    """Load one f32-promoted element of the shared index-key.

    Input seam, mirror of :func:`_load_index_q_elem`.
    """
    return b.cast_to_f32(b.global_load_bf16(index_k, b.add(base, d)))


def _emit_score(b: IRBuilder, scores, out_idx, value) -> None:
    """Write one query/key score to the standalone score buffer.

    Emit seam: the fused indexer+top-k kernel replaces this with an on-chip
    top-k tail so the score never reaches HBM. The score loop is unchanged.
    """
    b.global_store(scores, out_idx, value, align=4)


def _emit_scalar_body(b, spec, index_q, index_k, w, scores, q_pos_base):
    """Scalar-reduction score loop (correctness-first v1).

    One workgroup per query row; threads stride the key range and, per key, sum
    the per-head ReLU-weighted dot products over D_I, apply the causal bound, and
    emit the score. No matrix cores, no LDS.
    """
    H_I = spec.n_index_heads
    D_I = spec.index_head_dim
    Sk = spec.seqlen_k
    bs = spec.tile.block_size

    c_DI = b.const_i32(D_I)
    c_Sk = b.const_i32(Sk)
    c_HI = b.const_i32(H_I)
    c_block = b.const_i32(bs)
    c_zero_f = b.const_f32(0.0)
    c_neg_inf = b.const_f32(NEG_INF_SCORE)

    q = b.block_id_x()
    q_pos = b.add(q_pos_base, q)
    q_row_base = b.mul(q, b.const_i32(H_I * D_I))
    scores_row_base = b.mul(q, c_Sk)

    tid = b.thread_id_x()
    key_loop = b.scf_for(tid, c_Sk, c_block, iv_name="s")
    with key_loop as s:
        k_row_base = b.mul(s, c_DI)
        head_loop = b.scf_for_iter(
            b.const_i32(0),
            c_HI,
            b.const_i32(1),
            [("score", c_zero_f)],
            iv_name="h",
        )
        with head_loop as (h, carried):
            (score,) = carried
            q_head_base = b.add(q_row_base, b.mul(h, c_DI))
            dot = c_zero_f
            for d in range(D_I):
                c_d = b.const_i32(d)
                qv = _load_index_q_elem(b, index_q, q_head_base, c_d)
                kv = _load_index_k_elem(b, index_k, k_row_base, c_d)
                dot = b.fma(qv, kv, dot)
            relu = b.fmax(dot, c_zero_f)
            w_h = b.global_load_f32(w, h)
            b.scf_yield(b.fma(w_h, relu, score))
        score = head_loop.results[0]
        causal_ok = b.cmp_le(s, q_pos)
        out = b.select(causal_ok, score, c_neg_inf)
        _emit_score(b, scores, b.add(scores_row_base, s), out)


def _emit_mfma_body(b, spec, index_q, index_k, w, scores, q_pos_base):
    """MFMA (matrix-core) score body.

    Per head h, scores_h[16, 16] = Q_h[16, D_I] @ K[16, D_I]^T over the 16x16x16
    bf16 atom (D_I contracts in D_I/16 accumulated MMA steps), then a per-element
    ReLU, the per-head weight, and a head sum accumulate the running score tile.
    Score-only means no value matmul, so no P->LDS round-trip and no LDS at all:
    the whole thing runs register-to-register. Head-streamed with the shared
    index-key tile loaded once per key tile and reused across heads.

    One wave64 per output block; requires 16-aligned seqlens (validator-gated).
    Lane decode: m_in_atom = lane%16 (A/B operand row), m_blk = lane//16 (output
    row block and the 4-K slot). Output cell r is matrix (row = m_blk*4 + r,
    col = m_in_atom).
    """
    from rocke.helpers.atoms import MfmaAtom

    H_I = spec.n_index_heads
    D_I = spec.index_head_dim
    atom = MfmaAtom.bf16_16x16x16()
    apl = atom.a_per_lane
    cpl = atom.c_per_lane
    n_katoms = D_I // atom.k

    c16 = b.const_i32(16)
    c_zero_f = b.const_f32(0.0)
    c_neg_inf = b.const_f32(NEG_INF_SCORE)
    c_DI = b.const_i32(D_I)
    c_Sk = b.const_i32(spec.seqlen_k)
    c_apl = b.const_i32(apl)
    c_atomk = b.const_i32(atom.k)
    c_cpl = b.const_i32(cpl)
    c_qrow_stride = b.const_i32(H_I * D_I)

    lane = b.thread_id_x()
    m_in_atom = b.mod(lane, c16)
    m_blk = b.div(lane, c16)
    k_lane_start = b.mul(m_blk, c_apl)

    # Each workgroup owns one (16-query, 16-key) output block: block_id_x is the
    # query tile, block_id_y the key tile. This parallelizes over both dimensions
    # (grid = seqlen_q/16 x seqlen_k/16) rather than looping the key range
    # serially in one workgroup.
    q_tile_base = b.mul(b.block_id_x(), c16)
    k_tile_base = b.mul(b.block_id_y(), c16)
    q_row_base = b.mul(b.add(q_tile_base, m_in_atom), c_qrow_stride)
    k_row_base = b.mul(b.add(k_tile_base, m_in_atom), c_DI)

    # Shared index-key: one tile load, reused across all heads.
    k_vecs = []
    for ka in range(n_katoms):
        d_start = b.add(b.mul(b.const_i32(ka), c_atomk), k_lane_start)
        k_vecs.append(
            b.global_load_vN(
                index_k, b.add(k_row_base, d_start), BF16, apl, align=apl * 2
            )
        )
    head_loop = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(H_I),
        b.const_i32(1),
        [(f"acc{r}", c_zero_f) for r in range(cpl)],
        iv_name="h",
    )
    with head_loop as (h, acc_cells):
        q_head_base = b.add(q_row_base, b.mul(h, c_DI))
        score = atom.zero_acc(b)
        for ka in range(n_katoms):
            d_start = b.add(b.mul(b.const_i32(ka), c_atomk), k_lane_start)
            q_vec = b.global_load_vN(
                index_q, b.add(q_head_base, d_start), BF16, apl, align=apl * 2
            )
            score = b.mfma_f32_16x16x16_bf16(q_vec, k_vecs[ka], score)
        w_h = b.global_load_f32(w, h)
        new_cells = []
        for r in range(cpl):
            relu = b.fmax(b.vec_extract(score, r), c_zero_f)
            new_cells.append(b.fadd(acc_cells[r], b.fmul(w_h, relu)))
        b.scf_yield(*new_cells)
    acc = head_loop.results
    # Causal bound + write. Output cell r is matrix (row = m_blk*4 + r,
    # col = m_in_atom); k_col is the same for all r cells of this lane.
    k_col = b.add(k_tile_base, m_in_atom)
    for r in range(cpl):
        row_q = b.add(q_tile_base, b.add(b.mul(m_blk, c_cpl), b.const_i32(r)))
        pos = b.add(q_pos_base, row_q)
        out = b.select(b.cmp_le(k_col, pos), acc[r], c_neg_inf)
        _emit_score(b, scores, b.add(b.mul(row_q, c_Sk), k_col), out)


def build_lightning_indexer(spec: IndexerSpec, *, arch: str = "gfx942") -> KernelDef:
    """Build the IR for one lightning-indexer forward instance.

    Kernel signature::

        (index_q: ptr<bf16, global>,   # [seqlen_q, H_I, D_I], projected + RoPE'd
         index_k: ptr<bf16, global>,   # [seqlen_k, D_I], shared across heads
         w:       ptr<f32,  global>,   # [H_I] per-head indexer weights
         scores:  ptr<f32,  global>,   # [seqlen_q, seqlen_k] output score matrix
         q_pos_base: i32)              # causal base: pos(q) = q_pos_base + q

    The scalar body (default) runs one workgroup per query row; the MFMA body
    runs one wave64 per 16-query tile over the matrix cores. Both emit the same
    signature and score matrix.
    """
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid lightning_indexer spec for {arch}: {why}")

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.tile.block_size

    index_q = b.param(
        "index_q", PtrType(BF16, "global"), noalias=True, readonly=True, align=16
    )
    index_k = b.param(
        "index_k", PtrType(BF16, "global"), noalias=True, readonly=True, align=16
    )
    w = b.param("w", PtrType(F32, "global"), noalias=True, readonly=True, align=16)
    scores = b.param(
        "scores", PtrType(F32, "global"), noalias=True, writeonly=True, align=16
    )
    q_pos_base = b.param("q_pos_base", I32)

    if spec.body == "mfma":
        _emit_mfma_body(b, spec, index_q, index_k, w, scores, q_pos_base)
    else:
        _emit_scalar_body(b, spec, index_q, index_k, w, scores, q_pos_base)

    b.ret()
    return b.kernel


def lightning_indexer_grid(spec: IndexerSpec) -> Tuple[int, int, int]:
    """Launch grid: scalar = one workgroup per query row; mfma = one wave64 per
    (16-query, 16-key) output block, parallelized over both dimensions."""
    if spec.body == "mfma":
        return (spec.seqlen_q // 16, spec.seqlen_k // 16, 1)
    return (spec.seqlen_q, 1, 1)


def lightning_indexer_signature(spec: IndexerSpec):
    """Manifest-style signature for :class:`rocke.runtime.launcher.KernelLauncher`."""
    return (
        SignatureBuilder()
        .ptr("index_q", spec.dtype)
        .ptr("index_k", spec.dtype)
        .ptr("w", "f32")
        .ptr("scores", "f32")
        .scalar("q_pos_base", "i32")
        .build()
    )


__all__ = [
    "IndexerSpec",
    "IndexerTileSpec",
    "build_lightning_indexer",
    "lightning_indexer_grid",
    "lightning_indexer_signature",
    "is_valid_spec",
    "INDEXER_DTYPES",
    "LDS_LIMIT",
    "NEG_INF_SCORE",
]
