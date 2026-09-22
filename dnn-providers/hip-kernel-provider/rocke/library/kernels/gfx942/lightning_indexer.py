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
    tile: IndexerTileSpec = field(default_factory=IndexerTileSpec)
    name: str = "rocke_lightning_indexer"

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            f"HI{self.n_index_heads}",
            f"D{self.index_head_dim}",
            self.dtype,
            f"Q{self.seqlen_q}",
            f"K{self.seqlen_k}",
            self.tile.name_parts(),
        )

    def lds_bytes(self) -> int:
        # Scalar v1 uses no shared memory. The MFMA/fused forms will return the
        # indexer tile plus the top-k candidate reserve here.
        return 0


def is_valid_spec(spec: IndexerSpec, arch: str = "gfx942") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for one indexer config on ``arch``.

    v1 is gfx942 + bf16 only. The architecture facts (wave size, LDS capacity)
    are sourced from :class:`rocke.core.arch.ArchTarget` so an unknown or
    unsupported arch is rejected with a structured reason instead of crashing
    comgr at lower time.
    """
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    if arch != "gfx942":
        return False, f"lightning indexer v1 is gfx942 only (got {arch!r})"
    if spec.dtype not in INDEXER_DTYPES:
        return False, f"unsupported dtype {spec.dtype!r} (only bf16 in v1)"
    if spec.n_index_heads <= 0:
        return False, f"n_index_heads must be positive (got {spec.n_index_heads})"
    if spec.index_head_dim <= 0:
        return False, f"index_head_dim must be positive (got {spec.index_head_dim})"
    if spec.seqlen_q <= 0 or spec.seqlen_k <= 0:
        return False, f"seqlen_q/seqlen_k must be positive (got {spec.seqlen_q}, {spec.seqlen_k})"
    bs = spec.tile.block_size
    wave = target.wave_size
    if bs <= 0 or bs > 1024:
        return False, f"block_size {bs} outside (0, 1024]"
    if bs % wave != 0:
        return False, f"block_size {bs} not a multiple of wave_size {wave} for {arch}"
    if spec.lds_bytes() > LDS_LIMIT:
        return False, f"lds_bytes {spec.lds_bytes()} exceeds {LDS_LIMIT} on {arch}"
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


def build_lightning_indexer(spec: IndexerSpec, *, arch: str = "gfx942") -> KernelDef:
    """Build the IR for one lightning-indexer forward instance.

    Kernel signature::

        (index_q: ptr<bf16, global>,   # [seqlen_q, H_I, D_I], projected + RoPE'd
         index_k: ptr<bf16, global>,   # [seqlen_k, D_I], shared across heads
         w:       ptr<f32,  global>,   # [H_I] per-head indexer weights
         scores:  ptr<f32,  global>,   # [seqlen_q, seqlen_k] output score matrix
         q_pos_base: i32)              # causal base: pos(q) = q_pos_base + q

    Grid ``(seqlen_q, 1, 1)``; ``block_id_x`` is the query row. Each thread owns
    keys ``tid, tid + block_size, ...`` and, per key, sums the per-head
    ReLU-weighted dot products, applies the causal bound, and emits the score.
    """
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid lightning_indexer spec for {arch}: {why}")

    H_I = spec.n_index_heads
    D_I = spec.index_head_dim
    Sk = spec.seqlen_k
    bs = spec.tile.block_size

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = bs

    index_q = b.param("index_q", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    index_k = b.param("index_k", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    w = b.param("w", PtrType(F32, "global"), noalias=True, readonly=True, align=16)
    scores = b.param("scores", PtrType(F32, "global"), noalias=True, writeonly=True, align=16)
    q_pos_base = b.param("q_pos_base", I32)

    c_DI = b.const_i32(D_I)
    c_Sk = b.const_i32(Sk)
    c_HI = b.const_i32(H_I)
    c_block = b.const_i32(bs)
    c_zero_f = b.const_f32(0.0)
    c_neg_inf = b.const_f32(NEG_INF_SCORE)

    q = b.block_id_x()
    # pos(q) is the query's absolute position for the causal bound. In decode
    # q_pos_base is the context length (one query); in single-sequence prefill
    # it is 0 so pos(q) == q.
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
        # Causal: a query only scores keys at or before its position; future
        # keys are driven to the sentinel so top-k never selects them.
        causal_ok = b.cmp_le(s, q_pos)
        out = b.select(causal_ok, score, c_neg_inf)
        _emit_score(b, scores, b.add(scores_row_base, s), out)

    b.ret()
    return b.kernel


def lightning_indexer_grid(spec: IndexerSpec) -> Tuple[int, int, int]:
    """Launch grid: one workgroup per query row."""
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
