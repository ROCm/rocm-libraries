# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Numerical reference for MLA (Multi-head Latent Attention) prefill.

Test oracle for the ``mla_prefill_fwd`` kernel family. numpy only -- torch is
optional across this repo and numpy is the only hard dependency, so the
reference must not need torch even though ``DESIGN.md`` section 8.1 sketches it
in torch notation.

This models the **whole op** (pre-kernel query projection + RoPE, then the flash
loop), not the flash kernel alone. It deliberately does not reproduce the kernel
split.

Conventions this file pins down
-------------------------------

**Weight layout.** Every weight is stored ``[in, out]`` row-major; an
up-projection is a plain ``x @ W`` with no transpose. ``W_UK`` and ``W_UV`` are
**per head** (leading ``H_q`` axis) -- the shared latent ``c_kv`` expands to a
*different* K/V for every head. ``k_rope`` is the one part of K that **is**
head-shared: it is broadcast across heads, never expanded.

**RoPE.** ``k_rope`` is stored post-rotation in the cache, so the query rotation
is mandatory and ``positions`` is a required input. Rotary tables are required
arguments, not derived here: a geometry-derived default would silently agree
with a kernel that derived it the same wrong way. Layout follows
``rocke.helpers.rotary``: ``"half"`` pairs ``(i, i + D/2)`` (default),
``"interleaved"`` pairs ``(2i, 2i+1)``.

**Softmax scale.** ``scale`` is a required host-supplied argument, never derived
from a head dimension. For the models this family targets it is
``1/sqrt(d_nope + d_rope) = 1/sqrt(192)`` -- not ``1/sqrt(576)``, and not that
value at all under YaRN rope scaling. The kernel ABI takes the base-2 form
``scale * log2(e)`` (see :func:`mla_scale_log2`), which is a different number
from ``log2(scale)``.

**Causal mask.** Bottom-right aligned from each sequence's *runtime* Q/K
lengths: key ``j`` is visible to query ``i`` iff ``j <= i + (S_k - S_q)``. A
static offset is not sufficient for packed varlen, where every sequence has its
own pair of lengths.

**softmax_lse.** Natural log, over the *scaled and masked* scores:
``lse = log(sum_j exp(s_j))``. This is the FlashAttention / vLLM convention that
cross-chunk merging consumes. A kernel that tracks ``(m, l)`` in the log2 domain
(as the unified attention kernels here do) converts on the way out with
``lse = (m + log2(l)) * ln(2)``. A fully masked row yields ``lse = -inf`` and a
zero output row.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Ordered by pipeline stage (geometry -> scale -> rope -> gather -> mask ->
# entry points), not alphabetically: this list doubles as a reading order for
# someone following the op through the file.
__all__ = [  # noqa: RUF022
    "MlaGeometry",
    "DEEPSEEK_V3_GEOMETRY",
    "KIMI_K2_GEOMETRY",
    "mla_scale_log2",
    "rope_pair_indices",
    "apply_rope",
    "gather_paged_kv",
    "bottom_right_causal_mask",
    "ref_mla_prefill_scores",
    "ref_mla_prefill",
]

ROPE_LAYOUTS = ("half", "interleaved")
DEFAULT_ROPE_LAYOUT = "half"


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MlaGeometry:
    """Shape constants for one MLA configuration.

    ``head_dim_qk = d_nope + d_rope`` is the width of the score-side
    contraction. It is unrelated to ``d_v`` and to decode-absorb's
    ``r_kv + d_rope``.
    """

    num_heads: int
    d_nope: int = 128
    d_rope: int = 64
    d_v: int = 128
    r_kv: int = 512
    r_q: int = 1536

    @property
    def head_dim_qk(self) -> int:
        return self.d_nope + self.d_rope

    @property
    def default_scale(self) -> float:
        """The ``1/sqrt(192)`` value, exposed for *tests* to pass explicitly.

        Deliberately not used as a default anywhere in this module: the parity
        gate is only meaningful if the reference and the kernel are handed the
        same host-supplied number.
        """
        return 1.0 / math.sqrt(self.head_dim_qk)

    def __post_init__(self) -> None:
        for name in ("num_heads", "d_nope", "d_rope", "d_v", "r_kv", "r_q"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"MlaGeometry.{name} must be a positive int, got {value!r}"
                )
        if self.d_rope % 2 != 0:
            raise ValueError(
                f"d_rope must be even for rotary pairing, got {self.d_rope}"
            )

    @classmethod
    def from_tensors(
        cls,
        q_latent: np.ndarray,
        c_kv: np.ndarray,
        k_rope: np.ndarray,
        w_uq: np.ndarray,
        w_uk: np.ndarray,
    ) -> MlaGeometry:
        """Recover the geometry from operand shapes, validating consistency.

        Shapes are recovered rather than assumed so that a caller which passes a
        trap-shaped operand (head axis dropped from ``W_UK``, head axis *added*
        to ``k_rope``) gets a named error instead of a silently different
        operator.
        """
        if q_latent.ndim != 2:
            raise ValueError(
                f"q_latent must be [total_q, r_Q] (2-D); got shape {q_latent.shape}. "
                "q_latent has no head axis -- the latent query is shared across heads."
            )
        if c_kv.ndim != 3:
            raise ValueError(
                f"c_kv must be paged [num_blocks, block_size, r_KV] (3-D); "
                f"got shape {c_kv.shape}."
            )
        if k_rope.ndim != 3:
            raise ValueError(
                f"k_rope must be paged [num_blocks, block_size, d_rope] (3-D); "
                f"got shape {k_rope.shape}. k_rope is head-shared -- it is the only "
                "part of K that is, and giving it a head axis is a different operator."
            )
        if w_uq.ndim != 3:
            raise ValueError(
                f"W_UQ must be [H_q, r_Q, d_nope + d_rope] (3-D); got shape {w_uq.shape}."
            )
        if w_uk.ndim != 3:
            raise ValueError(
                f"W_UK must be per head [H_q, r_KV, d_nope + d_V] (3-D); got shape "
                f"{w_uk.shape}. Dropping the head axis is not a simplification, it is a "
                "different operator: the shared latent expands to a different K/V for "
                "every head."
            )

        total_q, r_q = q_latent.shape
        num_blocks, block_size, r_kv = c_kv.shape
        rope_blocks, rope_block_size, d_rope = k_rope.shape
        num_heads, w_uq_r_q, head_dim_qk = w_uq.shape
        w_uk_heads, w_uk_r_kv, d_nope_plus_v = w_uk.shape

        if (rope_blocks, rope_block_size) != (num_blocks, block_size):
            raise ValueError(
                f"k_rope paging {(rope_blocks, rope_block_size)} must match c_kv paging "
                f"{(num_blocks, block_size)}: both index the same cache blocks."
            )
        if w_uq_r_q != r_q:
            raise ValueError(f"W_UQ r_Q={w_uq_r_q} does not match q_latent r_Q={r_q}")
        if w_uk_r_kv != r_kv:
            raise ValueError(f"W_UK r_KV={w_uk_r_kv} does not match c_kv r_KV={r_kv}")
        if w_uk_heads != num_heads:
            raise ValueError(
                f"W_UK head count {w_uk_heads} does not match W_UQ head count {num_heads}"
            )
        d_nope = head_dim_qk - d_rope
        if d_nope <= 0:
            raise ValueError(
                f"W_UQ output width {head_dim_qk} must exceed d_rope={d_rope}; "
                "it is d_nope + d_rope."
            )
        d_v = d_nope_plus_v - d_nope
        if d_v <= 0:
            raise ValueError(
                f"W_UK output width {d_nope_plus_v} must exceed d_nope={d_nope}; "
                "it is d_nope + d_V."
            )
        del total_q
        return cls(
            num_heads=num_heads,
            d_nope=d_nope,
            d_rope=d_rope,
            d_v=d_v,
            r_kv=r_kv,
            r_q=r_q,
        )


DEEPSEEK_V3_GEOMETRY = MlaGeometry(num_heads=128)
KIMI_K2_GEOMETRY = MlaGeometry(num_heads=64)


def mla_scale_log2(scale: float) -> float:
    """Convert a host softmax ``scale`` to the kernel ABI's ``scale_log2``.

    Every rocKE attention kernel multiplies the raw dot product by an f32
    ``scale_log2`` and feeds an ``exp2`` softmax, so the ABI wants
    ``scale * log2(e)`` -- **not** ``log2(scale)``. At ``scale = 1/sqrt(192)``
    those are ``0.1041`` and ``-3.7924``; confusing them is uniformly, not
    marginally, wrong, which makes it a cheap first smoke test on a red bring-up
    run.
    """
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"scale must be a finite positive float, got {scale!r}. A negative value is "
            "the signature of passing log2(scale) where scale*log2(e) was wanted."
        )
    return scale * math.log2(math.e)


# --------------------------------------------------------------------------
# Rotary
# --------------------------------------------------------------------------


def rope_pair_indices(
    d_rope: int, layout: str = DEFAULT_ROPE_LAYOUT
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lo, hi)`` index arrays for the rotary pairs of one head.

    Mirrors ``rocke.helpers.rotary.pair_indices``: ``"interleaved"`` pairs
    ``(2i, 2i+1)``; ``"half"`` pairs ``(i, i + d_rope/2)``.
    """
    if layout not in ROPE_LAYOUTS:
        raise ValueError(f"rope layout must be one of {ROPE_LAYOUTS}, got {layout!r}")
    if d_rope % 2 != 0:
        raise ValueError(f"d_rope must be even, got {d_rope}")
    half = d_rope // 2
    idx = np.arange(half, dtype=np.int64)
    if layout == "interleaved":
        return 2 * idx, 2 * idx + 1
    return idx, idx + half


def apply_rope(
    x: np.ndarray,
    positions: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    layout: str = DEFAULT_ROPE_LAYOUT,
) -> np.ndarray:
    """Rotate ``x`` in place-free fashion at ``positions``.

    ``x`` is ``[T, d_rope]`` or ``[T, H, d_rope]`` with the token axis first.
    ``cos_table`` / ``sin_table`` are ``[max_pos, d_rope // 2]`` f32, indexed by
    absolute token position -- the host-supplied tables of the repo rotary
    contract, passed in rather than derived so that an incorrect table is
    visible to the gate.
    """
    if x.ndim not in (2, 3):
        raise ValueError(
            f"apply_rope expects [T, d_rope] or [T, H, d_rope]; got {x.shape}"
        )
    positions = np.asarray(positions)
    if positions.ndim != 1 or positions.shape[0] != x.shape[0]:
        raise ValueError(
            f"positions must be [T] matching the token axis of x ({x.shape[0]}); "
            f"got shape {positions.shape}"
        )
    d_rope = x.shape[-1]
    half = d_rope // 2
    for name, table in (("cos_table", cos_table), ("sin_table", sin_table)):
        if table.ndim != 2 or table.shape[1] != half:
            raise ValueError(
                f"{name} must be [max_pos, d_rope//2] = [*, {half}]; got {table.shape}"
            )
    max_pos = cos_table.shape[0]
    if positions.min() < 0 or positions.max() >= max_pos:
        raise ValueError(
            f"positions span [{positions.min()}, {positions.max()}] but the rotary "
            f"tables only cover [0, {max_pos - 1}]"
        )

    lo, hi = rope_pair_indices(d_rope, layout)
    cos = cos_table[positions].astype(np.float32)  # [T, half]
    sin = sin_table[positions].astype(np.float32)
    if x.ndim == 3:
        cos = cos[:, None, :]
        sin = sin[:, None, :]

    xf = x.astype(np.float32)
    x_lo = xf[..., lo]
    x_hi = xf[..., hi]
    out = np.empty_like(xf)
    out[..., lo] = x_lo * cos - x_hi * sin
    out[..., hi] = x_lo * sin + x_hi * cos
    return out


# --------------------------------------------------------------------------
# Paged gather
# --------------------------------------------------------------------------


def gather_paged_kv(
    c_kv: np.ndarray,
    k_rope: np.ndarray,
    block_table: np.ndarray,
    cu_seqlens_k: Sequence[int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialize dense per-sequence ``(c_kv, k_rope)`` from the paged cache.

    Returns one ``([S_k, r_KV], [S_k, d_rope])`` pair per sequence, truncated to
    the sequence's runtime length so that a partially filled final block (a page
    tail) contributes only its valid rows.
    """
    if c_kv.ndim != 3 or k_rope.ndim != 3:
        raise ValueError(
            "gather_paged_kv expects paged c_kv [num_blocks, block_size, r_KV] and "
            f"k_rope [num_blocks, block_size, d_rope]; got {c_kv.shape} and {k_rope.shape}"
        )
    num_blocks, block_size, _ = c_kv.shape
    block_table = np.asarray(block_table)
    if block_table.ndim != 2:
        raise ValueError(
            f"block_table must be [B, max_blocks]; got {block_table.shape}"
        )

    cu_seqlens_k = np.asarray(cu_seqlens_k, dtype=np.int64)
    batch = cu_seqlens_k.shape[0] - 1
    if block_table.shape[0] != batch:
        raise ValueError(
            f"block_table batch {block_table.shape[0]} does not match cu_seqlens_k "
            f"batch {batch}"
        )

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for b in range(batch):
        seq_len = int(cu_seqlens_k[b + 1] - cu_seqlens_k[b])
        if seq_len < 0:
            raise ValueError(f"cu_seqlens_k is not monotonic at sequence {b}")
        needed = (seq_len + block_size - 1) // block_size
        if needed > block_table.shape[1]:
            raise ValueError(
                f"sequence {b} needs {needed} blocks for S_k={seq_len} at block_size="
                f"{block_size}, but block_table only has {block_table.shape[1]} columns"
            )
        ids = block_table[b, :needed].astype(np.int64)
        if needed and (ids.min() < 0 or ids.max() >= num_blocks):
            raise ValueError(
                f"block_table row {b} references block ids outside [0, {num_blocks - 1}]"
            )
        latent = c_kv[ids].reshape(needed * block_size, -1)[:seq_len]
        rope = k_rope[ids].reshape(needed * block_size, -1)[:seq_len]
        out.append((latent, rope))
    return out


# --------------------------------------------------------------------------
# Mask
# --------------------------------------------------------------------------


def bottom_right_causal_mask(s_q: int, s_k: int) -> np.ndarray:
    """``[S_q, S_k]`` bool mask, True where the key is visible.

    Bottom-right aligned: key ``j`` is visible to query ``i`` iff
    ``j <= i + (S_k - S_q)``. With ``S_k > S_q`` (chunked prefill) every query
    sees the whole preceding context; with ``S_k < S_q`` the leading queries see
    nothing, which is a fully masked row, not an error.
    """
    if s_q < 0 or s_k < 0:
        raise ValueError(
            f"sequence lengths must be non-negative; got S_q={s_q}, S_k={s_k}"
        )
    offset = s_k - s_q
    i = np.arange(s_q, dtype=np.int64)[:, None]
    j = np.arange(s_k, dtype=np.int64)[None, :]
    return j <= i + offset


# --------------------------------------------------------------------------
# Reference
# --------------------------------------------------------------------------


def _validate_packing(
    q_latent: np.ndarray,
    cu_seqlens_q: np.ndarray,
    cu_seqlens_k: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cu_q = np.asarray(cu_seqlens_q, dtype=np.int64)
    cu_k = np.asarray(cu_seqlens_k, dtype=np.int64)
    if cu_q.ndim != 1 or cu_k.ndim != 1:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must both be 1-D [B+1] arrays")
    if cu_q.shape != cu_k.shape:
        raise ValueError(
            f"cu_seqlens_q {cu_q.shape} and cu_seqlens_k {cu_k.shape} must describe the "
            "same batch. They are separate inputs because packed varlen prefill has "
            "S_q != S_k per sequence, not because the batch differs."
        )
    if cu_q[0] != 0 or cu_k[0] != 0:
        raise ValueError("cu_seqlens arrays must start at 0")
    if np.any(np.diff(cu_q) < 0) or np.any(np.diff(cu_k) < 0):
        raise ValueError("cu_seqlens arrays must be non-decreasing")
    if int(cu_q[-1]) != q_latent.shape[0]:
        raise ValueError(
            f"cu_seqlens_q[-1]={int(cu_q[-1])} does not match total_q="
            f"{q_latent.shape[0]}"
        )
    positions = np.asarray(positions)
    if positions.ndim != 1 or positions.shape[0] != q_latent.shape[0]:
        raise ValueError(
            f"positions must be [total_q] = [{q_latent.shape[0]}]; got {positions.shape}. "
            "RoPE is not optional on the query side: k_rope is cached post-rotation, so "
            "the query must be rotated at its absolute position."
        )
    return cu_q, cu_k


def _project_query(
    q_latent: np.ndarray,
    w_uq: np.ndarray,
    positions: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    geom: MlaGeometry,
    rope_layout: str,
) -> tuple[np.ndarray, np.ndarray]:
    """The pre-kernel: expand the latent query per head, split, rotate.

    Returns ``(q_nope [total_q, H_q, d_nope], q_rope [total_q, H_q, d_rope])``.
    """
    q = np.einsum(
        "tr,hro->tho",
        q_latent.astype(np.float32),
        w_uq.astype(np.float32),
        optimize=True,
    )
    q_nope = q[..., : geom.d_nope]
    q_rope = q[..., geom.d_nope :]
    q_rope = apply_rope(q_rope, positions, cos_table, sin_table, rope_layout)
    return q_nope, q_rope


def _expand_latent_kv(
    latent: np.ndarray,
    w_uk: np.ndarray,
    geom: MlaGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-head latent expansion of one sequence's dense ``c_kv``.

    ``W_UK`` is ``[H_q, r_KV, d_nope + d_V]``, so the head axis survives into
    both outputs: ``K_nope [S_k, H_q, d_nope]`` and ``V [S_k, H_q, d_V]``.
    """
    latent_f = latent.astype(np.float32)
    w = w_uk.astype(np.float32)
    k_nope = np.einsum("sr,hro->sho", latent_f, w[:, :, : geom.d_nope], optimize=True)
    v = np.einsum("sr,hro->sho", latent_f, w[:, :, geom.d_nope :], optimize=True)
    return k_nope, v


def ref_mla_prefill_scores(
    q_latent: np.ndarray,
    c_kv: np.ndarray,
    k_rope: np.ndarray,
    w_uq: np.ndarray,
    w_uk: np.ndarray,
    cu_seqlens_q: Sequence[int],
    cu_seqlens_k: Sequence[int],
    block_table: np.ndarray,
    positions: np.ndarray,
    scale: float,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    rope_layout: str = DEFAULT_ROPE_LAYOUT,
    causal: bool = True,
    geometry: MlaGeometry | None = None,
) -> list[np.ndarray]:
    """Scaled (and optionally masked) QK scores, one ``[S_q, H_q, S_k]`` per sequence.

    This is the score-tile oracle: it stops short of softmax so a bring-up
    kernel can be compared on the QK half alone, before PV is written. Pass
    ``causal=False`` to compare a probe that has not yet grown a mask.

    Masked entries are ``-inf``.
    """
    geom = geometry or MlaGeometry.from_tensors(q_latent, c_kv, k_rope, w_uq, w_uk)
    cu_q, cu_k = _validate_packing(q_latent, cu_seqlens_q, cu_seqlens_k, positions)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"scale must be a finite positive float, got {scale!r}. It is host-supplied "
            "and must be the same value the kernel receives; do not derive it here. "
            "A non-positive value is the signature of passing log2(scale) where the "
            "linear scale was wanted -- see mla_scale_log2 for the ABI's base-2 form."
        )

    q_nope, q_rope = _project_query(
        q_latent, w_uq, positions, cos_table, sin_table, geom, rope_layout
    )
    per_seq = gather_paged_kv(c_kv, k_rope, block_table, cu_k)

    scores: list[np.ndarray] = []
    for b, (latent, rope) in enumerate(per_seq):
        q_lo, q_hi = int(cu_q[b]), int(cu_q[b + 1])
        s_q = q_hi - q_lo
        s_k = latent.shape[0]
        k_nope, _ = _expand_latent_kv(latent, w_uk, geom)

        # k_rope is head-shared: contracted without a head axis and broadcast
        # across h, unlike K_nope which carries one per head.
        s = scale * (
            np.einsum("thd,shd->ths", q_nope[q_lo:q_hi], k_nope, optimize=True)
            + np.einsum(
                "thd,sd->ths", q_rope[q_lo:q_hi], rope.astype(np.float32), optimize=True
            )
        )
        if causal:
            visible = bottom_right_causal_mask(s_q, s_k)
            s = np.where(visible[:, None, :], s, np.float32(-np.inf))
        scores.append(s.astype(np.float32))
    return scores


def ref_mla_prefill(
    q_latent: np.ndarray,
    c_kv: np.ndarray,
    k_rope: np.ndarray,
    w_uq: np.ndarray,
    w_uk: np.ndarray,
    cu_seqlens_q: Sequence[int],
    cu_seqlens_k: Sequence[int],
    block_table: np.ndarray,
    positions: np.ndarray,
    scale: float,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    rope_layout: str = DEFAULT_ROPE_LAYOUT,
    causal: bool = True,
    geometry: MlaGeometry | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Full ``mla_prefill_fwd`` reference.

    Returns ``(out [total_q, H_q, d_V] f32, softmax_lse [total_q, H_q] f32)``.
    Outputs are f32; the caller rounds ``out`` to bf16 when comparing against a
    bf16 kernel. ``softmax_lse`` is in natural log over the scaled, masked
    scores.
    """
    geom = geometry or MlaGeometry.from_tensors(q_latent, c_kv, k_rope, w_uq, w_uk)
    cu_q, cu_k = _validate_packing(q_latent, cu_seqlens_q, cu_seqlens_k, positions)

    scores = ref_mla_prefill_scores(
        q_latent,
        c_kv,
        k_rope,
        w_uq,
        w_uk,
        cu_q,
        cu_k,
        block_table,
        positions,
        scale,
        cos_table,
        sin_table,
        rope_layout=rope_layout,
        causal=causal,
        geometry=geom,
    )
    per_seq = gather_paged_kv(c_kv, k_rope, block_table, cu_k)

    total_q = q_latent.shape[0]
    out = np.zeros((total_q, geom.num_heads, geom.d_v), dtype=np.float32)
    lse = np.full((total_q, geom.num_heads), -np.inf, dtype=np.float32)

    for b, (latent, _rope) in enumerate(per_seq):
        q_lo, q_hi = int(cu_q[b]), int(cu_q[b + 1])
        if q_hi == q_lo:
            continue
        _, v = _expand_latent_kv(latent, w_uk, geom)
        s = scores[b]  # [S_q, H_q, S_k], -inf where masked

        # Stabilized softmax. A fully masked row has m = -inf; guard it so the
        # subtraction does not produce NaN, and leave its lse at -inf.
        m = s.max(axis=-1)  # [S_q, H_q]
        alive = np.isfinite(m)
        m_safe = np.where(alive, m, np.float32(0.0))
        p = np.exp(s - m_safe[:, :, None])
        p = np.where(np.isfinite(s), p, np.float32(0.0))
        denom = p.sum(axis=-1)  # [S_q, H_q]

        acc = np.einsum("ths,shv->thv", p, v, optimize=True)
        safe_denom = np.where(alive, denom, np.float32(1.0))
        acc = acc / safe_denom[:, :, None]

        out[q_lo:q_hi] = np.where(alive[:, :, None], acc, np.float32(0.0))
        lse[q_lo:q_hi] = np.where(
            alive,
            m_safe + np.log(np.maximum(denom, np.finfo(np.float32).tiny)),
            -np.inf,
        )

    return out, lse
