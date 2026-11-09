# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Adversarial coverage for the MLA numerical reference
# (``builders/mla/ref_mla_attn.py``).
#
# The reference is the *only* oracle the gfx942 MLA kernel will be graded
# against, so testing that it agrees with itself is worthless. Every test in the
# "traps" class instead re-implements the op with one specific mistake baked in
# and asserts the reference **disagrees** with that mutant. A trap the reference
# shares with the kernel is a trap the parity gate cannot see -- which is the
# whole reason DESIGN.md section 9 enumerates them.
#
# The four traps, from DESIGN.md section 9:
#   1. W_UK / W_UV are per head. Dropping the head axis is a different operator.
#   2. k_rope IS head-shared -- the only part of K that is. Giving it a head
#      axis is the opposite error.
#   3. RoPE is not optional on the query side; k_rope is cached post-rotation,
#      so `positions` is required.
#   4. `scale` is host-supplied and the ABI wants scale * log2(e), not
#      log2(scale).
#
# Dimensions here are deliberately small. The traps live in the *head* and
# *layout* axes, not in the latent ranks, so r_Q / r_KV are shrunk to keep the
# suite fast while d_nope / d_rope / d_v and the head count stay structural.

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import pytest

from builders.mla.ref_mla_attn import (
    DEFAULT_ROPE_LAYOUT,
    MlaGeometry,
    apply_rope,
    bottom_right_causal_mask,
    gather_paged_kv,
    mla_scale_log2,
    ref_mla_prefill,
    ref_mla_prefill_scores,
    rope_pair_indices,
)

BLOCK_SIZE = 16


class Problem(NamedTuple):
    geom: MlaGeometry
    q_latent: np.ndarray
    c_kv: np.ndarray
    k_rope: np.ndarray
    w_uq: np.ndarray
    w_uk: np.ndarray
    cu_seqlens_q: np.ndarray
    cu_seqlens_k: np.ndarray
    block_table: np.ndarray
    positions: np.ndarray
    scale: float
    cos_table: np.ndarray
    sin_table: np.ndarray

    def kwargs(self, **overrides):
        """Every field except ``geom`` is a reference-function argument.

        Derived from ``_asdict()`` rather than a hand-written list so a new
        operand cannot be added to the fixture and silently left unpassed.
        ``geom`` is dropped because the reference re-infers geometry from the
        operand shapes -- handing it back would mask a shape bug.
        """
        base = self._asdict()
        del base["geom"]
        base.update(overrides)
        return base


def _rope_tables(d_rope: int, max_pos: int, base: float = 10000.0):
    half = d_rope // 2
    inv_freq = 1.0 / (base ** (np.arange(half, dtype=np.float64) / half))
    angle = np.arange(max_pos, dtype=np.float64)[:, None] * inv_freq[None, :]
    return np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)


def make_problem(
    *,
    num_heads: int = 4,
    q_lens=(5, 7),
    k_lens=(20, 19),
    d_nope: int = 16,
    d_rope: int = 8,
    d_v: int = 16,
    r_kv: int = 32,
    r_q: int = 48,
    seed: int = 0,
) -> Problem:
    """Build a packed-varlen MLA problem.

    Default ``k_lens`` are deliberately not multiples of ``BLOCK_SIZE`` so every
    sequence ends in a partially filled page -- the page-tail case is the common
    path here, not a special one.
    """
    rng = np.random.default_rng(seed)
    geom = MlaGeometry(
        num_heads=num_heads, d_nope=d_nope, d_rope=d_rope, d_v=d_v, r_kv=r_kv, r_q=r_q
    )

    cu_q = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    cu_k = np.concatenate([[0], np.cumsum(k_lens)]).astype(np.int32)
    total_q = int(cu_q[-1])
    batch = len(q_lens)

    max_blocks = max((k + BLOCK_SIZE - 1) // BLOCK_SIZE for k in k_lens)
    num_blocks = batch * max_blocks
    # Shuffled so a kernel that assumes sequence-contiguous pages fails.
    block_table = (
        rng.permutation(num_blocks).astype(np.int32).reshape(batch, max_blocks)
    )

    q_latent = rng.standard_normal((total_q, r_q)).astype(np.float32)
    c_kv = rng.standard_normal((num_blocks, BLOCK_SIZE, r_kv)).astype(np.float32)
    k_rope = rng.standard_normal((num_blocks, BLOCK_SIZE, d_rope)).astype(np.float32)
    w_uq = (rng.standard_normal((num_heads, r_q, d_nope + d_rope)) * 0.1).astype(
        np.float32
    )
    w_uk = (rng.standard_normal((num_heads, r_kv, d_nope + d_v)) * 0.1).astype(
        np.float32
    )

    # Nonzero, per-sequence absolute positions: the query of a chunk starts at
    # the end of its context, not at 0. A reference that ignored `positions`
    # would agree with one that used zeros, so zeros must not appear here.
    #
    # The clamp only fires for S_k < S_q, which is not a real serving shape --
    # a chunk cannot hold more queries than the context has keys. Those configs
    # exist here solely to force fully-masked rows, and their leading queries
    # are masked out anyway, so pinning them at position 0 changes no result.
    positions = np.concatenate(
        [
            np.arange(max(k - q, 0), max(k - q, 0) + q, dtype=np.int32)
            for q, k in zip(q_lens, k_lens)
        ]
    )
    max_pos = int(positions.max()) + 1
    cos_table, sin_table = _rope_tables(d_rope, max_pos)

    return Problem(
        geom=geom,
        q_latent=q_latent,
        c_kv=c_kv,
        k_rope=k_rope,
        w_uq=w_uq,
        w_uk=w_uk,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        block_table=block_table,
        positions=positions,
        scale=geom.default_scale,
        cos_table=cos_table,
        sin_table=sin_table,
    )


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.nanmax(np.abs(np.asarray(a) - np.asarray(b))))


# --------------------------------------------------------------------------
# Geometry / shape contract
# --------------------------------------------------------------------------


class TestGeometry:
    def test_inferred_from_operand_shapes(self):
        p = make_problem()
        got = MlaGeometry.from_tensors(p.q_latent, p.c_kv, p.k_rope, p.w_uq, p.w_uk)
        assert got == p.geom

    def test_default_scale_is_over_sqrt_head_dim_qk(self):
        geom = MlaGeometry(num_heads=128)
        assert geom.head_dim_qk == 192
        assert geom.default_scale == pytest.approx(1.0 / math.sqrt(192))
        # Not the latent width: decode-absorb's r_kv + d_rope must not leak in.
        assert geom.default_scale != pytest.approx(1.0 / math.sqrt(576))

    @pytest.mark.parametrize("num_heads", [64, 128])
    def test_real_head_counts(self, num_heads):
        p = make_problem(num_heads=num_heads, r_q=16, r_kv=16)
        out, lse = ref_mla_prefill(**p.kwargs())
        assert out.shape == (int(p.cu_seqlens_q[-1]), num_heads, p.geom.d_v)
        assert lse.shape == (int(p.cu_seqlens_q[-1]), num_heads)
        assert np.isfinite(out).all() and np.isfinite(lse).all()

    def test_q_latent_head_axis_rejected(self):
        p = make_problem()
        bad = np.broadcast_to(
            p.q_latent[:, None, :], (p.q_latent.shape[0], 4, p.geom.r_q)
        )
        with pytest.raises(ValueError, match="q_latent"):
            ref_mla_prefill(**p.kwargs(q_latent=np.ascontiguousarray(bad)))

    def test_dense_unpaged_kv_rejected(self):
        p = make_problem()
        with pytest.raises(ValueError, match="c_kv"):
            ref_mla_prefill(**p.kwargs(c_kv=p.c_kv.reshape(-1, p.geom.r_kv)))

    def test_cu_seqlens_batch_mismatch_rejected(self):
        p = make_problem()
        with pytest.raises(ValueError, match="same batch"):
            ref_mla_prefill(**p.kwargs(cu_seqlens_k=p.cu_seqlens_k[:-1]))


# --------------------------------------------------------------------------
# The four DESIGN.md section 9 traps
# --------------------------------------------------------------------------


class TestTrap1PerHeadLatentWeights:
    """W_UK / W_UV carry a head axis. Collapsing it is a different operator."""

    def test_head_collapsed_w_uk_shape_is_rejected(self):
        p = make_problem()
        with pytest.raises(ValueError, match="per head"):
            ref_mla_prefill(**p.kwargs(w_uk=p.w_uk[0]))

    def test_head_broadcast_w_uk_is_detected_numerically(self):
        """A kernel that expands the latent once and shares it across heads.

        Shape-legal (the head axis is still there), so only the numbers catch
        it. This is the realistic form of trap 1: an implementation that hoists
        the latent expansion out of the head loop as an "optimization".
        """
        p = make_problem()
        shared = np.repeat(p.w_uk[:1], p.geom.num_heads, axis=0)
        ref_out, ref_lse = ref_mla_prefill(**p.kwargs())
        bad_out, bad_lse = ref_mla_prefill(**p.kwargs(w_uk=shared))

        assert _max_abs(ref_out, bad_out) > 1e-2
        assert _max_abs(ref_lse, bad_lse) > 1e-2
        # And the give-away: under the mutant every head is identical.
        assert _max_abs(bad_out[:, 0], bad_out[:, -1]) > 0.0 or p.geom.num_heads == 1
        head_spread = _max_abs(ref_out[:, 0], ref_out[:, -1])
        assert (
            head_spread > 1e-3
        ), "heads must not be degenerate, or trap 1 is untestable"

    def test_v_slice_is_the_upper_half_of_w_uk(self):
        """Swapping the K and V halves of W_UK is a silent slicing error."""
        p = make_problem()
        swapped = np.concatenate(
            [p.w_uk[:, :, p.geom.d_nope :], p.w_uk[:, :, : p.geom.d_nope]], axis=-1
        )
        ref_out, _ = ref_mla_prefill(**p.kwargs())
        bad_out, _ = ref_mla_prefill(**p.kwargs(w_uk=swapped))
        assert _max_abs(ref_out, bad_out) > 1e-2


class TestTrap2HeadSharedKRope:
    """k_rope is head-shared and broadcast -- never expanded per head."""

    def test_head_axis_on_k_rope_is_rejected(self):
        p = make_problem()
        bad = np.repeat(p.k_rope[:, :, None, :], p.geom.num_heads, axis=2)
        with pytest.raises(ValueError, match="head-shared"):
            ref_mla_prefill(**p.kwargs(k_rope=bad))

    def test_rope_contribution_is_identical_across_heads(self):
        """Two heads with equal q_rope and zero q_nope must score identically.

        If k_rope were expanded per head, the shared rotary query would still
        produce different scores per head and this equality would break. That
        makes it a positive test for broadcast semantics, not just a shape gate.
        """
        p = make_problem()
        w_uq = p.w_uq.copy()
        w_uq[:, :, : p.geom.d_nope] = 0.0  # kill the nope contribution entirely
        w_uq[1] = w_uq[0]  # heads 0 and 1 now share the rope projection

        scores = ref_mla_prefill_scores(**p.kwargs(w_uq=w_uq), causal=False)
        for seq in scores:
            np.testing.assert_allclose(seq[:, 0, :], seq[:, 1, :], rtol=0, atol=0)
            # Heads that do NOT share the projection must still differ, else the
            # equality above is vacuous.
            assert _max_abs(seq[:, 0, :], seq[:, -1, :]) > 1e-4


class TestTrap3QueryRope:
    """RoPE is mandatory on the query side; k_rope is cached post-rotation."""

    def test_positions_is_a_required_argument(self):
        p = make_problem()
        kwargs = p.kwargs()
        kwargs.pop("positions")
        with pytest.raises(TypeError):
            ref_mla_prefill(**kwargs)

    def test_skipping_the_rotation_is_detected(self):
        """Identity tables (cos=1, sin=0) == not rotating at all."""
        p = make_problem()
        ones = np.ones_like(p.cos_table)
        zeros = np.zeros_like(p.sin_table)
        ref_out, _ = ref_mla_prefill(**p.kwargs())
        bad_out, _ = ref_mla_prefill(**p.kwargs(cos_table=ones, sin_table=zeros))
        assert _max_abs(ref_out, bad_out) > 1e-2

    def test_zero_positions_is_detected(self):
        """Rotating everything at position 0 is the same mistake, subtler."""
        p = make_problem()
        ref_out, _ = ref_mla_prefill(**p.kwargs())
        bad_out, _ = ref_mla_prefill(**p.kwargs(positions=np.zeros_like(p.positions)))
        assert _max_abs(ref_out, bad_out) > 1e-2

    def test_wrong_rope_layout_is_detected(self):
        p = make_problem()
        ref_out, _ = ref_mla_prefill(**p.kwargs(), rope_layout="half")
        bad_out, _ = ref_mla_prefill(**p.kwargs(), rope_layout="interleaved")
        assert _max_abs(ref_out, bad_out) > 1e-2

    def test_out_of_range_position_is_rejected(self):
        p = make_problem()
        bad = p.positions.copy()
        bad[0] = p.cos_table.shape[0] + 5
        with pytest.raises(ValueError, match="rotary tables"):
            ref_mla_prefill(**p.kwargs(positions=bad))

    def test_rope_is_norm_preserving(self):
        p = make_problem()
        x = (
            np.random.default_rng(1)
            .standard_normal((7, 3, p.geom.d_rope))
            .astype(np.float32)
        )
        pos = np.arange(7, dtype=np.int32)
        rot = apply_rope(x, pos, p.cos_table, p.sin_table, DEFAULT_ROPE_LAYOUT)
        np.testing.assert_allclose(
            np.linalg.norm(rot, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-5
        )

    @pytest.mark.parametrize(
        "layout,expected",
        [("interleaved", ([0, 2, 4], [1, 3, 5])), ("half", ([0, 1, 2], [3, 4, 5]))],
    )
    def test_pair_indices_match_the_platform_rotary_helper(self, layout, expected):
        lo, hi = rope_pair_indices(6, layout)
        assert lo.tolist() == expected[0]
        assert hi.tolist() == expected[1]


class TestTrap4Scale:
    """The ABI wants scale * log2(e), not log2(scale)."""

    def test_scale_log2_is_not_log2_of_scale(self):
        scale = 1.0 / math.sqrt(192)
        assert mla_scale_log2(scale) == pytest.approx(0.10411754627697266)
        assert math.log2(scale) == pytest.approx(-3.792481250360578)
        # Not merely unequal -- opposite sign, which is why a kernel that
        # confuses them is uniformly rather than marginally wrong.
        assert mla_scale_log2(scale) > 0 > math.log2(scale)

    def test_scale_log2_round_trips_through_exp2(self):
        """exp2(s * scale_log2) must equal exp(s * scale)."""
        scale = 1.0 / math.sqrt(192)
        s = np.linspace(-8.0, 8.0, 33)
        np.testing.assert_allclose(
            np.exp2(s * mla_scale_log2(scale)), np.exp(s * scale), rtol=1e-6
        )

    def test_log2_scale_as_scale_is_rejected(self):
        p = make_problem()
        with pytest.raises(ValueError, match="log2"):
            ref_mla_prefill(**p.kwargs(scale=math.log2(p.scale)))

    def test_scale_is_required_and_not_derived(self):
        p = make_problem()
        kwargs = p.kwargs()
        kwargs.pop("scale")
        with pytest.raises(TypeError):
            ref_mla_prefill(**kwargs)

    def test_wrong_scale_changes_the_result(self):
        """Guards against a kernel deriving scale from d_v or r_kv + d_rope."""
        p = make_problem()
        ref_out, _ = ref_mla_prefill(**p.kwargs())
        for wrong in (p.geom.d_v**-0.5, (p.geom.r_kv + p.geom.d_rope) ** -0.5):
            bad_out, _ = ref_mla_prefill(**p.kwargs(scale=wrong))
            assert _max_abs(ref_out, bad_out) > 1e-3


# --------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------


class TestBottomRightCausalMask:
    def test_square_case_is_lower_triangular(self):
        np.testing.assert_array_equal(
            bottom_right_causal_mask(4, 4), np.tril(np.ones((4, 4), bool))
        )

    def test_chunked_case_keeps_the_whole_prefix(self):
        m = bottom_right_causal_mask(2, 5)
        # offset = 3: query 0 sees keys 0..3, query 1 sees keys 0..4.
        assert m[0].tolist() == [True, True, True, True, False]
        assert m[1].tolist() == [True] * 5

    def test_short_context_leaves_fully_masked_rows(self):
        m = bottom_right_causal_mask(4, 2)
        # offset = -2: queries 0 and 1 see nothing.
        assert not m[0].any() and not m[1].any()
        assert m[2].tolist() == [True, False]
        assert m[3].tolist() == [True, True]

    def test_top_left_alignment_is_detected(self):
        """A static `j <= i` mask diverges as soon as S_q != S_k."""
        p = make_problem(q_lens=(4,), k_lens=(20,))
        ref = ref_mla_prefill_scores(**p.kwargs())[0]
        unmasked = ref_mla_prefill_scores(**p.kwargs(), causal=False)[0]
        s_q, s_k = 4, 20
        top_left = np.arange(s_k)[None, :] <= np.arange(s_q)[:, None]
        bad = np.where(top_left[:, None, :], unmasked, -np.inf)
        assert not np.array_equal(np.isfinite(ref), np.isfinite(bad))

    def test_static_offset_across_mixed_varlen_is_detected(self):
        """One shared S_k - S_q offset cannot serve a packed batch.

        Sequence 0 has offset 15, sequence 1 has offset 12. Applying either to
        both is wrong, which is the reason the kernel must read cu_seqlens per
        sequence instead of taking a launch-time constant.
        """
        p = make_problem(q_lens=(5, 7), k_lens=(20, 19))
        per_seq = ref_mla_prefill_scores(**p.kwargs())
        offsets = {20 - 5, 19 - 7}
        assert len(offsets) == 2
        unmasked = ref_mla_prefill_scores(**p.kwargs(), causal=False)
        for b, (s_q, s_k) in enumerate([(5, 20), (7, 19)]):
            wrong_offset = 12 if b == 0 else 15
            j = np.arange(s_k)[None, :]
            i = np.arange(s_q)[:, None]
            bad = np.where((j <= i + wrong_offset)[:, None, :], unmasked[b], -np.inf)
            assert not np.array_equal(np.isfinite(per_seq[b]), np.isfinite(bad))

    def test_fully_masked_rows_are_zero_not_nan(self):
        p = make_problem(q_lens=(5,), k_lens=(2,))
        out, lse = ref_mla_prefill(**p.kwargs())
        assert not np.isnan(out).any()
        # offset = -3: queries 0..2 see nothing.
        np.testing.assert_array_equal(out[:3], np.zeros_like(out[:3]))
        assert np.isneginf(lse[:3]).all()
        assert np.isfinite(lse[3:]).all()
        assert np.abs(out[3:]).max() > 0.0


# --------------------------------------------------------------------------
# Paging
# --------------------------------------------------------------------------


class TestPagedGather:
    def test_page_tail_rows_are_dropped(self):
        p = make_problem(q_lens=(3,), k_lens=(19,))
        ((latent, rope),) = gather_paged_kv(
            p.c_kv, p.k_rope, p.block_table, p.cu_seqlens_k
        )
        assert latent.shape == (19, p.geom.r_kv)
        assert rope.shape == (19, p.geom.d_rope)
        # The 19th row is the 3rd row of the 2nd page, not of the 1st.
        second_page = p.block_table[0, 1]
        np.testing.assert_array_equal(latent[18], p.c_kv[second_page, 2])

    def test_gather_follows_the_block_table_indirection(self):
        p = make_problem(q_lens=(3,), k_lens=(BLOCK_SIZE,))
        ((latent, _),) = gather_paged_kv(
            p.c_kv, p.k_rope, p.block_table, p.cu_seqlens_k
        )
        np.testing.assert_array_equal(latent, p.c_kv[p.block_table[0, 0]])

    def test_exact_multiple_needs_no_extra_page(self):
        p = make_problem(q_lens=(3,), k_lens=(2 * BLOCK_SIZE,))
        ((latent, _),) = gather_paged_kv(
            p.c_kv, p.k_rope, p.block_table, p.cu_seqlens_k
        )
        assert latent.shape[0] == 2 * BLOCK_SIZE

    def test_short_block_table_is_rejected(self):
        p = make_problem(q_lens=(3,), k_lens=(19,))
        with pytest.raises(ValueError, match="block_table"):
            gather_paged_kv(p.c_kv, p.k_rope, p.block_table[:, :1], p.cu_seqlens_k)

    def test_out_of_range_block_id_is_rejected(self):
        p = make_problem(q_lens=(3,), k_lens=(19,))
        bad = p.block_table.copy()
        bad[0, 0] = p.c_kv.shape[0] + 1
        with pytest.raises(ValueError, match="outside"):
            gather_paged_kv(p.c_kv, p.k_rope, bad, p.cu_seqlens_k)


# --------------------------------------------------------------------------
# Softmax / LSE
# --------------------------------------------------------------------------


class TestSoftmaxAndLse:
    def test_matches_an_independent_per_sequence_recompute(self):
        """Recompute sequence by sequence, head by head, without the batched path."""
        p = make_problem()
        out, lse = ref_mla_prefill(**p.kwargs())
        per_seq = gather_paged_kv(p.c_kv, p.k_rope, p.block_table, p.cu_seqlens_k)
        g = p.geom
        half = g.d_rope // 2

        for b, (latent, rope) in enumerate(per_seq):
            q_lo, q_hi = int(p.cu_seqlens_q[b]), int(p.cu_seqlens_q[b + 1])
            s_q, s_k = q_hi - q_lo, latent.shape[0]
            for h in range(g.num_heads):
                k_nope = latent @ p.w_uk[h][:, : g.d_nope]
                v = latent @ p.w_uk[h][:, g.d_nope :]
                for local_i, t in enumerate(range(q_lo, q_hi)):
                    proj = p.q_latent[t] @ p.w_uq[h]
                    q_nope, q_rope = proj[: g.d_nope], proj[g.d_nope :]
                    c = p.cos_table[p.positions[t]]
                    s = p.sin_table[p.positions[t]]
                    rot = np.empty_like(q_rope)
                    rot[:half] = q_rope[:half] * c - q_rope[half:] * s
                    rot[half:] = q_rope[:half] * s + q_rope[half:] * c

                    score = p.scale * (k_nope @ q_nope + rope @ rot)
                    score[local_i + (s_k - s_q) + 1 :] = -np.inf
                    m = score.max()
                    prob = np.exp(score - m)
                    prob[~np.isfinite(score)] = 0.0
                    denom = prob.sum()

                    np.testing.assert_allclose(
                        out[t, h], (prob @ v) / denom, rtol=1e-4, atol=1e-5
                    )
                    np.testing.assert_allclose(
                        lse[t, h], m + np.log(denom), rtol=1e-5, atol=1e-5
                    )

    def test_lse_is_natural_log_not_log2(self):
        p = make_problem()
        _, lse = ref_mla_prefill(**p.kwargs())
        scores = ref_mla_prefill_scores(**p.kwargs())
        ref = np.concatenate(
            [
                np.log(np.exp(s - s.max(-1, keepdims=True)).sum(-1)) + s.max(-1)
                for s in scores
            ]
        )
        np.testing.assert_allclose(lse, ref, rtol=1e-5, atol=1e-5)
        # A log2-domain LSE would be off by exactly 1/ln(2).
        assert _max_abs(lse, ref / math.log(2.0)) > 1e-2

    def test_lse_matches_the_kernel_log2_accumulator_conversion(self):
        """(m2 + log2(l)) * ln2, the form the kernel will convert on write.

        The kernel tracks m and l in log2 domain against scale * log2(e)-scaled
        scores. This asserts that path lands on the same number as the natural
        log reference, so the conversion is validated before any kernel exists.
        """
        p = make_problem()
        _, lse = ref_mla_prefill(**p.kwargs())
        unscaled = ref_mla_prefill_scores(**p.kwargs(scale=1.0))
        log2e = math.log2(math.e)

        converted = []
        for s in unscaled:
            s2 = np.where(np.isfinite(s), s * (p.scale * log2e), -np.inf)
            m2 = s2.max(-1)
            l2 = np.exp2(s2 - m2[..., None])
            l2 = np.where(np.isfinite(s2), l2, 0.0).sum(-1)
            converted.append((m2 + np.log2(l2)) * math.log(2.0))
        np.testing.assert_allclose(lse, np.concatenate(converted), rtol=1e-4, atol=1e-5)

    def test_softmax_rows_are_normalized(self):
        """exp(scores - lse) must be a probability distribution over the row.

        This is the statement that the output is a convex combination of the
        visible V rows, expressed through the LSE so it does not need a
        hand-constructed W_UV.
        """
        p = make_problem(q_lens=(4,), k_lens=(20,))
        _, lse = ref_mla_prefill(**p.kwargs())
        scores = ref_mla_prefill_scores(**p.kwargs())[0]
        prob = np.exp(scores - lse[..., None])
        np.testing.assert_allclose(prob.sum(-1), 1.0, rtol=1e-5, atol=1e-5)

    def test_scores_probe_matches_the_full_reference_masking(self):
        p = make_problem()
        masked = ref_mla_prefill_scores(**p.kwargs())
        unmasked = ref_mla_prefill_scores(**p.kwargs(), causal=False)
        for b, (s_q, s_k) in enumerate([(5, 20), (7, 19)]):
            visible = bottom_right_causal_mask(s_q, s_k)
            np.testing.assert_array_equal(
                np.isfinite(masked[b]),
                np.broadcast_to(visible[:, None, :], masked[b].shape),
            )
            np.testing.assert_allclose(
                masked[b][:, :, :][np.isfinite(masked[b])],
                unmasked[b][np.isfinite(masked[b])],
                rtol=0,
                atol=0,
            )


# --------------------------------------------------------------------------
# Packed varlen
# --------------------------------------------------------------------------


class TestPackedVarlen:
    def test_sequences_are_independent(self):
        """Running a batch must equal running each sequence alone."""
        p = make_problem(q_lens=(5, 7), k_lens=(20, 19))
        batched_out, batched_lse = ref_mla_prefill(**p.kwargs())

        for b, (s_q, s_k) in enumerate([(5, 20), (7, 19)]):
            q_lo, q_hi = int(p.cu_seqlens_q[b]), int(p.cu_seqlens_q[b + 1])
            solo_out, solo_lse = ref_mla_prefill(
                **p.kwargs(
                    q_latent=p.q_latent[q_lo:q_hi],
                    cu_seqlens_q=np.array([0, s_q], np.int32),
                    cu_seqlens_k=np.array([0, s_k], np.int32),
                    block_table=p.block_table[b : b + 1],
                    positions=p.positions[q_lo:q_hi],
                )
            )
            np.testing.assert_allclose(
                solo_out, batched_out[q_lo:q_hi], rtol=1e-5, atol=1e-6
            )
            np.testing.assert_allclose(
                solo_lse, batched_lse[q_lo:q_hi], rtol=1e-5, atol=1e-6
            )

    def test_empty_query_sequence_is_allowed(self):
        p = make_problem(q_lens=(0, 6), k_lens=(BLOCK_SIZE, 19))
        out, lse = ref_mla_prefill(**p.kwargs())
        assert out.shape[0] == 6 and lse.shape[0] == 6

    def test_square_prefill_is_the_s_q_equals_s_k_case(self):
        p = make_problem(q_lens=(12, 9), k_lens=(12, 9))
        out, lse = ref_mla_prefill(**p.kwargs())
        assert np.isfinite(out).all()
        # Bottom-right and top-left coincide only here.
        for s in (12, 9):
            np.testing.assert_array_equal(
                bottom_right_causal_mask(s, s), np.tril(np.ones((s, s), bool))
            )
        assert np.isfinite(lse).all()
