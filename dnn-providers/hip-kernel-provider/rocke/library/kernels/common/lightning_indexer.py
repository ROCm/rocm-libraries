# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Lightning indexer forward (DSA relevance scorer) -- gfx942 / gfx950, bf16.

The lightning indexer is the first stage of DeepSeek Sparse Attention. It is a
score-only kernel: for each query position ``t`` it scores every allowed key
position ``s`` and emits a single relevance score

    I(t, s) = sum over heads h of  w[t, h] * ReLU( index_q[t, h] . index_k[s] )

with no softmax and no value output. The scores feed top-k selection (a separate
stage). The index-query is per head; the index-key is shared across heads (one
D_I vector per token, MQA-style), so ``index_k`` carries no head axis; the weight
``w[t, h]`` is per query token and head (shape [seqlen_q, H_I]) and signed.

Scope:

* Score-only. The projection, RoPE, and index-key cache write are caller-side
  (the reference's fused qk-norm-rope-quant op); this kernel consumes an already
  projected + RoPE'd ``index_q`` and an already populated ``index_k``.
* Two bodies behind ``IndexerSpec.body``: a scalar FMA reduction (the correctness
  oracle and the fallback for unaligned shapes) and an MFMA matrix-core body for
  16-aligned shapes. Both run on gfx942 (CDNA3) and gfx950 (CDNA4) in bf16 and
  use no LDS. fp8 is a later phase.

Two seams keep later work additive for the SCALAR body. They do not hold for the
MFMA body, which loads inputs directly through ``global_load_vN`` (not the load
seams) and splits one query row's keys across ``seqlen_k/16`` workgroups, so
fusing a per-row top-k there is a change to the decomposition, not a hook swap:

* ``_emit_score`` is the only place a score reaches memory. The fused
  indexer+top-k kernel replaces this hook with an on-chip top-k tail; the scalar
  score loop above it does not change.
* ``_load_index_q_elem`` / ``_load_index_k_elem`` are the only places the scalar
  body reads inputs. Folding the pre-step (projection/RoPE) into the kernel later
  replaces these with a project-then-score prologue that produces the same values.

Grid is body-dependent: the scalar body launches ``(seqlen_q, 1, 1)`` (one
workgroup per query row, threads striding the key range); the MFMA body launches
``(seqlen_q/16, seqlen_k/16, 1)`` (one wave64 per 16x16 output block).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from rocke.core.ir import BF16, F32, I32, IRBuilder, KernelDef, PtrType
from rocke.helpers.spec import SignatureBuilder, kernel_name_join

# Conservative LDS ceiling (the gfx942 64 KiB figure). Both bodies use no LDS, so
# the validator's real check sources the arch's capacity from ArchTarget; this
# constant is kept for the export and the spec test until a fused top-k needs a
# per-arch budget.
LDS_LIMIT = 64 * 1024
# Causal mask sentinel. A future key must never be selectable by top-k, so its
# score is driven far below any real score. Real scores can be negative (the
# per-head weights are signed), so the sentinel sits far below the most negative
# possible real score, not merely at zero.
NEG_INF_SCORE = -3.0e38
# Declared coverage, exported so a dispatch candidate states what it serves by
# importing these rather than transcribing them.
INDEXER_DTYPES: Tuple[str, ...] = ("bf16",)


@dataclass(frozen=True)
class IndexerTileSpec:
    """Tiling knobs.

    The only knob today is ``block_size`` (threads per workgroup): 256 for the
    scalar body, one wave64 (64) for the MFMA body. Bq / Bk and head-streaming
    would appear with an LDS-staged MFMA variant; the current register-only MFMA
    body needs none.
    """

    block_size: int = 256

    def name_parts(self) -> str:
        return f"b{self.block_size}"


@dataclass(frozen=True)
class IndexerSpec:
    """One lightning-indexer forward configuration.

    ``seqlen_q`` (total query tokens) and ``seqlen_k`` (key length Sk) are runtime
    kernel arguments, not baked into the IR: the spec carries them only to size
    the launch grid host-side and to describe the dispatch request. One compiled
    kernel per ``(n_index_heads, index_head_dim, body)`` therefore serves every
    sequence length, and decode does not recompile as the cache grows.
    """

    n_index_heads: int
    seqlen_q: int
    seqlen_k: int
    index_head_dim: int = 128
    dtype: str = "bf16"
    body: str = "scalar"
    tile: IndexerTileSpec = field(default_factory=IndexerTileSpec)
    name: str = "rocke_lightning_indexer"

    def kernel_name(self) -> str:
        # seqlen_q and seqlen_k are runtime kernel arguments, not baked into the
        # IR, so they are NOT in the name: one binary per (H_I, D_I, body) serves
        # every sequence length, and decode no longer recompiles as the cache
        # grows. The spec still carries them to size the launch grid host-side.
        parts = [
            self.name,
            f"HI{self.n_index_heads}",
            f"D{self.index_head_dim}",
            self.dtype,
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
    # Element offsets are i32 in both bodies. Reject shapes whose largest offset
    # would overflow 2**31 (a silent wrap into an OOB load/store): the score store
    # index is row_q*seqlen_k + k_col, the index-query base is
    # q*n_index_heads*index_head_dim, and the index-key base is s*index_head_dim.
    if spec.seqlen_q * spec.seqlen_k >= 2**31:
        return (
            False,
            f"seqlen_q*seqlen_k ({spec.seqlen_q * spec.seqlen_k}) overflows the i32 "
            f"score-matrix offset",
        )
    if spec.seqlen_q * spec.n_index_heads * spec.index_head_dim >= 2**31:
        return (
            False,
            "seqlen_q*n_index_heads*index_head_dim overflows the i32 index-query offset",
        )
    if spec.seqlen_k * spec.index_head_dim >= 2**31:
        return (
            False,
            "seqlen_k*index_head_dim overflows the i32 index-key offset",
        )
    bs = spec.tile.block_size
    wave = target.wave_size
    if bs <= 0 or bs > 1024:
        return False, f"block_size {bs} outside (0, 1024]"
    if bs % wave != 0:
        return False, f"block_size {bs} not a multiple of wave_size {wave} for {arch}"
    if spec.lds_bytes() > target.lds_capacity_bytes:
        return (
            False,
            f"lds_bytes {spec.lds_bytes()} exceeds {target.lds_capacity_bytes} on {arch}",
        )
    if spec.body not in ("scalar", "mfma"):
        return False, f"unknown body {spec.body!r} (scalar | mfma)"
    if spec.body == "mfma":
        # The 16x16x16 bf16 atom uses one wave64. seqlen_q / seqlen_k need not be
        # 16-aligned: the ceil grid clamps tail loads and predicates tail stores.
        # index_head_dim must stay 16-aligned (the D_I contraction is untiled).
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


def _emit_scalar_body(b, spec, index_q, index_k, w, scores, q_pos_base, seqlen_k):
    """Scalar-reduction score loop (correctness-first v1).

    One workgroup per query row (block_id_x) and per key slice (block_id_y);
    within a slice one thread owns one key and sums the per-head ReLU-weighted dot
    products over D_I, applies the causal bound, and emits the score. No matrix
    cores, no LDS. Grid ``(seqlen_q, ceil(seqlen_k/block_size), 1)``.
    """
    H_I = spec.n_index_heads
    D_I = spec.index_head_dim
    bs = spec.tile.block_size

    # seqlen_k is a runtime kernel argument (c_Sk), not a compiled constant.
    c_Sk = seqlen_k
    c_DI = b.const_i32(D_I)
    c_HI = b.const_i32(H_I)
    c_block = b.const_i32(bs)
    c_zero_f = b.const_f32(0.0)
    c_neg_inf = b.const_f32(NEG_INF_SCORE)

    q = b.block_id_x()
    q_pos = b.add(q_pos_base, q)
    q_row_base = b.mul(q, b.const_i32(H_I * D_I))
    scores_row_base = b.mul(q, c_Sk)

    tid = b.thread_id_x()
    # Keys are parallelized across grid.y: workgroup block_id_y owns the
    # block_size-wide key slice starting at block_id_y*block_size, one key per
    # thread. The loop runs at most once (step == seqlen_k), so it is a bounded
    # "process key s only if s < seqlen_k": a tail thread with s >= seqlen_k
    # iterates zero times and issues no index_k load and no store.
    s0 = b.add(b.mul(b.block_id_y(), c_block), tid)
    key_loop = b.scf_for(s0, c_Sk, c_Sk, iv_name="s")
    c_zero_i = b.const_i32(0)
    with key_loop as s:
        k_row_base = b.mul(s, c_DI)
        # Causal skip: a future key (s > q_pos) runs zero head iterations, so its
        # per-head dot and the K/Q loads inside the loop are skipped; the score
        # stays at the 0.0 init and the select below writes the sentinel.
        causal_ok = b.cmp_le(s, q_pos)
        h_bound = b.select(causal_ok, c_HI, c_zero_i)
        head_loop = b.scf_for_iter(
            c_zero_i,
            h_bound,
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
            # Per-query weight: w has shape [seqlen_q, H_I], indexed by this
            # workgroup's query row q and head h (DSA weight is w_{t,h}, not a
            # per-head constant).
            w_h = b.global_load_f32(w, b.add(b.mul(q, c_HI), h))
            b.scf_yield(b.fma(w_h, relu, score))
        score = head_loop.results[0]
        out = b.select(causal_ok, score, c_neg_inf)
        _emit_score(b, scores, b.add(scores_row_base, s), out)


def _emit_mfma_body(
    b, spec, index_q, index_k, w, scores, q_pos_base, seqlen_q, seqlen_k
):
    """MFMA (matrix-core) score body.

    Per head h, scores_h[16, 16] = Q_h[16, D_I] @ K[16, D_I]^T over the 16x16x16
    bf16 atom (D_I contracts in D_I/16 accumulated MMA steps), then a per-element
    ReLU, the per-head weight, and a head sum accumulate the running score tile.
    Score-only means no value matmul, so no P->LDS round-trip and no LDS at all:
    the whole thing runs register-to-register. Head-streamed with the shared
    index-key tile loaded once per key tile and reused across heads.

    One wave64 per output block. Non-16-aligned seqlen_q / seqlen_k are handled
    by the ceil grid: a tail tile's loads are clamped in-bounds and its stores are
    predicated out, so only real (row, key) cells are written. Requires 16-aligned
    index_head_dim (the D_I contraction is untiled).
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
    # seqlen_q / seqlen_k are runtime kernel arguments, not compiled constants.
    c_Sq = seqlen_q
    c_Sk = seqlen_k
    c_apl = b.const_i32(apl)
    c_atomk = b.const_i32(atom.k)
    c_cpl = b.const_i32(cpl)
    c_HI = b.const_i32(H_I)
    c_qrow_stride = b.const_i32(H_I * D_I)

    lane = b.thread_id_x()
    m_in_atom = b.mod(lane, c16)
    m_blk = b.div(lane, c16)
    k_lane_start = b.mul(m_blk, c_apl)

    # Each workgroup owns one (16-query, 16-key) output block: block_id_x is the
    # query tile, block_id_y the key tile, so the grid is
    # (ceil(seqlen_q/16), ceil(seqlen_k/16)) and parallelizes over both dims. A
    # tail tile may address rows/keys past the real seqlen_q/seqlen_k, so clamp
    # the load row indices in-bounds; the garbage those lanes compute is discarded
    # by the predicated store below.
    c1 = b.const_i32(1)
    q_tile_base = b.mul(b.block_id_x(), c16)
    k_tile_base = b.mul(b.block_id_y(), c16)
    q_row_idx = b.smin(b.add(q_tile_base, m_in_atom), b.sub(c_Sq, c1))
    k_row_idx = b.smin(b.add(k_tile_base, m_in_atom), b.sub(c_Sk, c1))
    q_row_base = b.mul(q_row_idx, c_qrow_stride)
    k_row_base = b.mul(k_row_idx, c_DI)

    # Shared index-key: one tile load, reused across all heads.
    k_vecs = []
    for ka in range(n_katoms):
        d_start = b.add(b.mul(b.const_i32(ka), c_atomk), k_lane_start)
        k_vecs.append(
            b.global_load_vN(
                index_k, b.add(k_row_base, d_start), BF16, apl, align=apl * 2
            )
        )
    # Causal skip: a fully-future output block (the smallest key exceeds the
    # largest query position in the tile) runs zero head iterations, so every
    # H_I MMA and the per-head Q loads are skipped; the accumulator stays at 0 and
    # the per-cell causal select below writes the sentinel to every cell. This is
    # wave-uniform (depends only on block ids and q_pos_base). The shared K load
    # above still runs.
    max_pos = b.add(q_pos_base, b.add(q_tile_base, b.const_i32(15)))
    fully_future = b.cmp_lt(max_pos, k_tile_base)
    h_bound = b.select(fully_future, b.const_i32(0), c_HI)
    head_loop = b.scf_for_iter(
        b.const_i32(0),
        h_bound,
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
        new_cells = []
        for r in range(cpl):
            relu = b.fmax(b.vec_extract(score, r), c_zero_f)
            # Per-query weight: output cell r is query row q_tile_base + m_blk*cpl
            # + r, so each cell takes its own w[row_q, h] = w[row_q*H_I + h]. Clamp
            # the w row in-bounds like the q/k loads above: a ceil-grid tail cell
            # (row_q >= seqlen_q) would otherwise read up to 15 rows past w, and its
            # result is discarded by the predicated store below.
            row_q = b.add(q_tile_base, b.add(b.mul(m_blk, c_cpl), b.const_i32(r)))
            w_row = b.smin(row_q, b.sub(c_Sq, c1))
            w_hr = b.global_load_f32(w, b.add(b.mul(w_row, c_HI), h))
            new_cells.append(b.fadd(acc_cells[r], b.fmul(w_hr, relu)))
        b.scf_yield(*new_cells)
    acc = head_loop.results
    # Causal bound + write, guarded so a ceil-grid tail cell writes nothing.
    # Output cell r is matrix (row = m_blk*4 + r, col = m_in_atom); k_col is the
    # same for all r cells of this lane, so the key-bound check hoists out of the
    # r loop, leaving only the per-row bound inside.
    k_col = b.add(k_tile_base, m_in_atom)
    with b.scf_if(b.cmp_lt(k_col, c_Sk)):
        for r in range(cpl):
            row_q = b.add(q_tile_base, b.add(b.mul(m_blk, c_cpl), b.const_i32(r)))
            pos = b.add(q_pos_base, row_q)
            out = b.select(b.cmp_le(k_col, pos), acc[r], c_neg_inf)
            with b.scf_if(b.cmp_lt(row_q, c_Sq)):
                _emit_score(b, scores, b.add(b.mul(row_q, c_Sk), k_col), out)


def build_lightning_indexer(spec: IndexerSpec, arch: str = "gfx942") -> KernelDef:
    """Build the IR for one lightning-indexer forward instance.

    Kernel signature::

        (index_q: ptr<bf16, global>,   # [seqlen_q, H_I, D_I], projected + RoPE'd
         index_k: ptr<bf16, global>,   # [seqlen_k, D_I], shared across heads
         w:       ptr<f32,  global>,   # [seqlen_q, H_I] per-query signed weights
         scores:  ptr<f32,  global>,   # [seqlen_q, seqlen_k] output score matrix
         q_pos_base: i32,              # causal base: pos(q) = q_pos_base + q
         seqlen_q:   i32,              # runtime query count (grid + tail bound)
         seqlen_k:   i32)              # runtime key length (grid + stride + tail)

    seqlen_q / seqlen_k are runtime arguments, not compiled in, so one binary per
    (H_I, D_I, body) serves every sequence length and decode does not recompile as
    the cache grows. The scalar body (default) runs one workgroup per (query row,
    key slice); the MFMA body runs one wave64 per (16-query, 16-key) output block.
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
    seqlen_q = b.param("seqlen_q", I32)
    seqlen_k = b.param("seqlen_k", I32)

    if spec.body == "mfma":
        _emit_mfma_body(
            b, spec, index_q, index_k, w, scores, q_pos_base, seqlen_q, seqlen_k
        )
    else:
        _emit_scalar_body(b, spec, index_q, index_k, w, scores, q_pos_base, seqlen_k)

    b.ret()
    return b.kernel


def lightning_indexer_grid(spec: IndexerSpec) -> Tuple[int, int, int]:
    """Launch grid. Scalar: one workgroup per (query row, key slice), so
    ``(seqlen_q, ceil(seqlen_k/block_size), 1)``. MFMA: one wave64 per
    (16-query, 16-key) output block, so ``(ceil(seqlen_q/16), ceil(seqlen_k/16), 1)``.
    Both ceil, so non-16 / non-block-multiple seqlens get a tail tile."""
    if spec.body == "mfma":
        return ((spec.seqlen_q + 15) // 16, (spec.seqlen_k + 15) // 16, 1)
    bs = spec.tile.block_size
    return (spec.seqlen_q, (spec.seqlen_k + bs - 1) // bs, 1)


def lightning_indexer_signature(spec: IndexerSpec):
    """Manifest-style signature for :class:`rocke.runtime.launcher.KernelLauncher`."""
    return (
        SignatureBuilder()
        .ptr("index_q", spec.dtype)
        .ptr("index_k", spec.dtype)
        .ptr("w", "f32")
        .ptr("scores", "f32")
        .scalar("q_pos_base", "i32")
        .scalar("seqlen_q", "i32")
        .scalar("seqlen_k", "i32")
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
