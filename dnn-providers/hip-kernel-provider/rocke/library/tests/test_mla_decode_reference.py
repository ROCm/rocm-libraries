# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU coverage for the MLA *decode* reference (``builders/mla/ref_mla_decode.py``)
and for the on-device parity gate's own rules.

The decode kernel is graded against ``ref_mla_decode_latent``, and the decode
op against ``ref_mla_decode_absorb``. Both are written in the absorbed form
the kernel computes, which is exactly the situation where a shared
misconception would pass the parity gate. So nothing here checks the
reference against itself. Each test ties it to something computed a
different way:

  * the *expanded* form (per-head K_nope / V materialized through W_UK),
  * the already-reviewed prefill oracle, at one query per sequence,
  * a single-pass softmax, for the split-KV merge rule,
  * a mutant with one ``builders/mla/DESIGN.md`` section 9 trap baked in,
    which must disagree.

Dimensions are small for speed; the head axis and the layout axes, where the
traps live, stay structural.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from builders.mla.ref_mla_attn import (
    MlaGeometry,
    _project_query,
    mla_scale_log2,
    ref_mla_prefill,
)
from builders.mla.ref_mla_decode import (
    absorb_query,
    ref_mla_decode_absorb,
    ref_mla_decode_latent,
    ref_mla_decode_merge,
    ref_mla_decode_partials,
)

BLOCK_SIZE = 16


def _make(
    *,
    num_heads: int = 4,
    q_lens=(1, 1, 1),
    k_lens=(20, 37, 5),
    d_nope: int = 16,
    d_rope: int = 8,
    d_v: int = 16,
    r_kv: int = 32,
    seed: int = 0,
    dtype=np.float64,
) -> dict:
    """A packed decode batch, paged with a shuffled block table.

    ``k_lens`` default to non-multiples of the page so every sequence ends in a
    partial page, the common case rather than a special one.
    """
    rng = np.random.default_rng(seed)
    cu_q = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    cu_k = np.concatenate([[0], np.cumsum(k_lens)]).astype(np.int32)
    total_q = int(cu_q[-1])
    batch = len(q_lens)
    max_blocks = max(1, max(-(-k // BLOCK_SIZE) for k in k_lens))
    num_blocks = batch * max_blocks
    block_table = rng.permutation(num_blocks).astype(np.int32).reshape(batch, -1)
    return {
        "q_nope": rng.standard_normal((total_q, num_heads, d_nope)).astype(dtype),
        "q_rope": rng.standard_normal((total_q, num_heads, d_rope)).astype(dtype),
        "c_kv": rng.standard_normal((num_blocks, BLOCK_SIZE, r_kv)).astype(dtype),
        "k_rope": rng.standard_normal((num_blocks, BLOCK_SIZE, d_rope)).astype(dtype),
        "w_uk": (rng.standard_normal((num_heads, r_kv, d_nope + d_v)) * 0.2).astype(
            dtype
        ),
        "block_table": block_table,
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
        "scale": 1.0 / math.sqrt(d_nope + d_rope),
    }


def _latent_kwargs(p: dict) -> dict:
    qa = absorb_query(p["q_nope"], p["q_rope"], p["w_uk"])
    return {
        "qa": qa,
        "c_kv": p["c_kv"],
        "k_rope": p["k_rope"],
        "block_table": p["block_table"],
        "cu_seqlens_q": p["cu_seqlens_q"],
        "cu_seqlens_k": p["cu_seqlens_k"],
        "scale": p["scale"],
    }


def _expanded(p: dict, *, shared_w_uk: bool = False, per_head_rope: bool = False):
    """The decode op in its *expanded* form, written independently.

    Per head, ``K_nope = C @ W_UK[h, :, :d_nope]`` and ``V = C @ W_UK[h, :,
    d_nope:]`` are materialized and attention runs at width ``d_nope +
    d_rope``. The two flags inject ``builders/mla/DESIGN.md`` section 9 traps
    1 and 2 into this implementation only, for the mutant tests.
    """
    q_nope, q_rope, w_uk = p["q_nope"], p["q_rope"], p["w_uk"]
    d_nope = q_nope.shape[-1]
    h = q_nope.shape[1]
    cu_q, cu_k = p["cu_seqlens_q"], p["cu_seqlens_k"]
    out = np.zeros((q_nope.shape[0], h, w_uk.shape[2] - d_nope))
    lse = np.full((q_nope.shape[0], h), -np.inf)
    for b in range(len(cu_q) - 1):
        if cu_q[b + 1] == cu_q[b]:
            continue
        t = int(cu_q[b])
        s_k = int(cu_k[b + 1] - cu_k[b])
        ids = p["block_table"][b, : -(-s_k // BLOCK_SIZE)]
        c = p["c_kv"][ids].reshape(-1, p["c_kv"].shape[2])[:s_k]
        kr = p["k_rope"][ids].reshape(-1, p["k_rope"].shape[2])[:s_k]
        for hh in range(h):
            w = w_uk[0] if shared_w_uk else w_uk[hh]
            k_nope = c @ w[:, :d_nope]
            v = c @ w[:, d_nope:]
            rope = kr * (1.0 + hh) if per_head_rope else kr
            s = p["scale"] * (k_nope @ q_nope[t, hh] + rope @ q_rope[t, hh])
            m = s.max()
            e = np.exp(s - m)
            out[t, hh] = (e @ v) / e.sum()
            lse[t, hh] = m + math.log(e.sum())
    return out, lse


class TestAbsorbedEqualsExpanded:
    """The identity every absorbed kernel relies on, in fp64."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize("num_heads", [1, 4])
    def test_absorb_matches_expanded(self, seed, num_heads):
        p = _make(seed=seed, num_heads=num_heads)
        out, lse = ref_mla_decode_absorb(**p)
        want_out, want_lse = _expanded(p)
        np.testing.assert_allclose(out, want_out, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(lse, want_lse, rtol=1e-12)

    def test_shared_w_uk_mutant_disagrees(self):
        # Trap 1: one W_UK for every head is a different operator.
        p = _make(num_heads=4)
        out, _ = ref_mla_decode_absorb(**p)
        mutant, _ = _expanded(p, shared_w_uk=True)
        assert np.max(np.abs(out - mutant)) > 1e-2

    def test_per_head_rope_mutant_disagrees(self):
        # Trap 2: k_rope is head-shared; a per-head k_rope is a different operator.
        p = _make(num_heads=4)
        out, _ = ref_mla_decode_absorb(**p)
        mutant, _ = _expanded(p, per_head_rope=True)
        assert np.max(np.abs(out - mutant)) > 1e-2


class TestSplitMerge:
    """``ref_mla_decode_merge`` of kernel-shaped partials equals the single pass."""

    @pytest.mark.parametrize("num_seg", [1, 2, 3, 7, 16, 64])
    def test_merge_is_exact(self, num_seg):
        # k_lens span 0..5 tiles, so num_seg >= 3 leaves trailing segments empty
        # and 64 empties most of them; one sequence has no query, one no keys.
        p = _make(q_lens=(1, 0, 1, 1, 1, 1), k_lens=(20, 50, 77, 5, 64, 0))
        kw = _latent_kwargs(p)
        want_lat, want_lse = ref_mla_decode_latent(**kw)
        parts = ref_mla_decode_partials(**kw, num_seg=num_seg, block_k=BLOCK_SIZE)
        got_lat, got_lse = ref_mla_decode_merge(*parts)
        np.testing.assert_allclose(got_lat, want_lat, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(got_lse, want_lse, rtol=1e-13)

    @pytest.mark.parametrize("num_seg", [3, 64])
    def test_merge_agrees_with_the_kernels_finite_sentinel(self, num_seg):
        # The kernels mark an empty segment with the finite NEG_INIT rather
        # than -inf, so an all-empty row merges without -inf - -inf = NaN.
        from kernels.mla.mla_decode_combine import NEG_INIT

        p = _make(k_lens=(20, 0, 5))
        acc, m, lsum = ref_mla_decode_partials(**_latent_kwargs(p), num_seg=num_seg)
        want = ref_mla_decode_merge(acc, m, lsum)
        got = ref_mla_decode_merge(acc, np.where(np.isfinite(m), m, NEG_INIT), lsum)
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])

    def test_empty_segments_are_marked(self):
        p = _make(k_lens=(20,), q_lens=(1,))  # two tiles
        parts = ref_mla_decode_partials(**_latent_kwargs(p), num_seg=4)
        part_acc, part_m, part_l = parts
        assert np.all(np.isfinite(part_m[0, :2]))
        assert np.all(part_m[0, 2:] == -np.inf)
        assert np.all(part_l[0, 2:] == 0.0)
        assert np.all(part_acc[0, 2:] == 0.0)

    def test_merge_needs_only_consistent_partials(self):
        # A kernel with a lazy (stale) running max hands over partials relative
        # to a smaller m. Shifting every (m, l, acc) consistently by delta must
        # not move the merged result.
        p = _make()
        kw = _latent_kwargs(p)
        part_acc, part_m, part_l = ref_mla_decode_partials(**kw, num_seg=2)
        delta = np.random.default_rng(3).uniform(-20.0, 0.0, size=part_m.shape)
        delta = np.where(np.isfinite(part_m), delta, 0.0)
        shifted = (
            part_acc * np.exp2(-delta)[..., None],
            part_m + delta,
            part_l * np.exp2(-delta),
        )
        a = ref_mla_decode_merge(part_acc, part_m, part_l)
        b = ref_mla_decode_merge(*shifted)
        np.testing.assert_allclose(b[0], a[0], rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(b[1], a[1], rtol=1e-13)

    def test_spike_crosses_the_lazy_rescale_threshold(self):
        # The GPU lane's spike scenario, at the real geometry: half of every
        # wave's heads see their running max jump past the kernel's lazy
        # threshold at key 300, the other half do not -- so the device takes
        # the rescale on a live accumulator for some lanes of a wave only. The
        # reference's split merge stays exact across the jump.
        from builders.mla import verify_mla_decode as v
        from builders.mla.ref_mla_attn import gather_paged_kv
        from kernels.mla.mla_decode import DECODE_LAZY_TAU

        p = v.make_problem(heads=64, q_lens=(1,), k_lens=(400,), spike_key=300)
        qa = absorb_query(p.q_nope, p.q_rope, p.w_uk)
        ((latent, rope),) = gather_paged_kv(
            p.c_kv, p.k_rope, p.block_table, p.cu_seqlens_k
        )
        keys = np.concatenate([latent, rope], axis=-1).astype(np.float64)
        s2 = mla_scale_log2(p.scale) * (qa[0] @ keys.T)  # [H, S_k], log2 units
        jump = s2[:, 300] - s2[:, :300].max(axis=1)
        spiked = np.arange(64) % 16 < 8
        assert np.all(jump[spiked] > DECODE_LAZY_TAU)
        assert np.all(jump[~spiked] < DECODE_LAZY_TAU)

        kw = {
            "qa": qa,
            "c_kv": p.c_kv,
            "k_rope": p.k_rope,
            "block_table": p.block_table,
            "cu_seqlens_q": p.cu_seqlens_q,
            "cu_seqlens_k": p.cu_seqlens_k,
            "scale": p.scale,
        }
        want_lat, want_lse = ref_mla_decode_latent(**kw)
        for num_seg in (2, 7):
            parts = ref_mla_decode_partials(**kw, num_seg=num_seg, block_k=BLOCK_SIZE)
            got_lat, got_lse = ref_mla_decode_merge(*parts)
            np.testing.assert_allclose(got_lat, want_lat, rtol=1e-12, atol=1e-13)
            np.testing.assert_allclose(got_lse, want_lse, rtol=1e-13)

    @pytest.mark.parametrize("num_cus", [80, 256, 304])
    @pytest.mark.parametrize("heads", [64, 128])
    def test_spike_lands_on_a_live_accumulator(self, monkeypatch, heads, num_cus):
        # Every split the GPU lane runs (forced 1 / 7 / 64 and the auto count
        # at representative CU counts) must put key 300's tile after at least
        # one earlier tile of its segment in some sequence; otherwise the
        # rescale only ever scales a zero accumulator.
        monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm22")
        from builders.mla import verify_mla_decode as v
        from dispatch.mla import MLARequest, dispatch_mla

        sc = v.SCENARIOS[f"spike_h{heads}"]
        auto = dispatch_mla(
            MLARequest(
                num_heads=heads,
                total_q=sum(sc.q_lens),
                num_seqs=len(sc.k_lens),
                max_seqlen_q=1,
                max_seqlen_k=max(sc.k_lens),
                arch="gfx942",
                num_cus=num_cus,
            )
        ).spec.num_seg
        spike_tile = sc.spike_key // BLOCK_SIZE
        for num_seg in sorted({1, 7, 64, auto}):
            prior = []
            for k in sc.k_lens:
                n_tiles = -(-k // BLOCK_SIZE)
                per_seg = -(-n_tiles // num_seg)
                prior.append(spike_tile % per_seg)
            assert max(prior) > 0, (num_seg, prior)


class TestDegenerateSequences:
    def test_empty_key_range(self):
        # S_k = 0: zero latent, lse = -inf, from both the single pass and the merge.
        p = _make(k_lens=(20, 0, 5))
        kw = _latent_kwargs(p)
        lat, lse = ref_mla_decode_latent(**kw)
        assert np.all(lat[1] == 0.0) and np.all(lse[1] == -np.inf)
        m_lat, m_lse = ref_mla_decode_merge(*ref_mla_decode_partials(**kw, num_seg=3))
        assert np.all(m_lat[1] == 0.0) and np.all(m_lse[1] == -np.inf)
        assert np.all(np.isfinite(lse[[0, 2]]))

    def test_sequence_without_a_query(self):
        # max_seqlen_q == 1 does not mean total_q == num_seqs: a packed decode
        # batch may hold a sequence with no query. Its keys must not land on the
        # next sequence's token.
        p = _make(q_lens=(1, 0, 1), k_lens=(20, 37, 5))
        out, _ = ref_mla_decode_absorb(**p)
        want, _ = _expanded(p)
        assert out.shape[0] == 2
        np.testing.assert_allclose(out, want, rtol=1e-10, atol=1e-12)

    def test_two_queries_refused(self):
        p = _make(q_lens=(1, 2), k_lens=(20, 37))
        with pytest.raises(ValueError, match="at most one query per sequence"):
            ref_mla_decode_absorb(**p)


class TestTraps:
    """``builders/mla/DESIGN.md`` section 9: each trap is refused with a named error."""

    def test_w_uk_without_head_axis_refused(self):
        p = _make()
        p["w_uk"] = p["w_uk"][0]
        with pytest.raises(ValueError, match="per head"):
            ref_mla_decode_absorb(**p)

    def test_head_axis_k_rope_refused(self):
        p = _make()
        kw = _latent_kwargs(p)
        kw["k_rope"] = np.repeat(kw["k_rope"][:, :, None, :], 4, axis=2)
        with pytest.raises(ValueError, match="head-shared"):
            ref_mla_decode_latent(**kw)

    def test_scale_has_no_default(self):
        kw = _latent_kwargs(_make())
        del kw["scale"]
        with pytest.raises(TypeError):
            ref_mla_decode_latent(**kw)

    def test_log2_scale_refused(self):
        # log2(1/sqrt(24)) is negative: the signature of the wrong base-2 form.
        kw = _latent_kwargs(_make())
        kw["scale"] = math.log2(kw["scale"])
        with pytest.raises(ValueError, match="log2"):
            ref_mla_decode_latent(**kw)

    def test_unabsorbed_query_refused(self):
        p = _make()
        kw = _latent_kwargs(p)
        kw["qa"] = np.concatenate([p["q_nope"], p["q_rope"]], axis=-1)
        with pytest.raises(ValueError, match="absorb_query"):
            ref_mla_decode_latent(**kw)

    def test_partials_use_scale_times_log2e(self):
        # m is in the kernel ABI's log2 units: scale * log2(e) times the dot
        # product, not log2(scale) times it.
        p = _make()
        kw = _latent_kwargs(p)
        _, part_m, _ = ref_mla_decode_partials(**kw, num_seg=1)
        _, lse = ref_mla_decode_latent(**kw)
        t = 0
        s_k = int(p["cu_seqlens_k"][1])
        ids = p["block_table"][0, : -(-s_k // BLOCK_SIZE)]
        keys = np.concatenate(
            [
                p["c_kv"][ids].reshape(-1, p["c_kv"].shape[2])[:s_k],
                p["k_rope"][ids].reshape(-1, p["k_rope"].shape[2])[:s_k],
            ],
            axis=-1,
        )
        dots = kw["qa"][t] @ keys.T
        np.testing.assert_allclose(
            part_m[t, 0], mla_scale_log2(p["scale"]) * dots.max(axis=-1), rtol=1e-13
        )
        wrong = math.log2(p["scale"]) * dots.max(axis=-1)
        assert np.max(np.abs(part_m[t, 0] - wrong)) > 1.0
        assert np.all(np.isfinite(lse[t]))


def _rope_tables(d_rope: int, max_pos: int, base: float = 10000.0):
    half = d_rope // 2
    inv_freq = 1.0 / (base ** (np.arange(half, dtype=np.float64) / half))
    angle = np.arange(max_pos, dtype=np.float64)[:, None] * inv_freq[None, :]
    return np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)


class TestConsistencyWithPrefill:
    """Decode equals the reviewed prefill oracle at one query per sequence."""

    @pytest.mark.parametrize("seed", [0, 1])
    def test_last_position_query(self, seed):
        rng = np.random.default_rng(seed)
        k_lens = (20, 37, 5)
        num_heads, d_nope, d_rope, d_v, r_kv, r_q = 4, 16, 8, 16, 32, 48
        geom = MlaGeometry(
            num_heads=num_heads,
            d_nope=d_nope,
            d_rope=d_rope,
            d_v=d_v,
            r_kv=r_kv,
            r_q=r_q,
        )
        p = _make(
            num_heads=num_heads,
            k_lens=k_lens,
            d_nope=d_nope,
            d_rope=d_rope,
            d_v=d_v,
            r_kv=r_kv,
            seed=seed,
            dtype=np.float32,
        )
        q_latent = rng.standard_normal((3, r_q)).astype(np.float32)
        w_uq = (rng.standard_normal((num_heads, r_q, d_nope + d_rope)) * 0.1).astype(
            np.float32
        )
        # The decode query sits at the last position of its sequence.
        positions = np.array([k - 1 for k in k_lens], dtype=np.int32)
        cos_t, sin_t = _rope_tables(d_rope, max(k_lens))
        common = {
            "c_kv": p["c_kv"],
            "k_rope": p["k_rope"],
            "block_table": p["block_table"],
            "cu_seqlens_q": p["cu_seqlens_q"],
            "cu_seqlens_k": p["cu_seqlens_k"],
            "scale": p["scale"],
        }
        want_out, want_lse = ref_mla_prefill(
            q_latent=q_latent,
            w_uq=w_uq,
            w_uk=p["w_uk"],
            positions=positions,
            cos_table=cos_t,
            sin_table=sin_t,
            **common,
        )
        q_nope, q_rope = _project_query(
            q_latent, w_uq, positions, cos_t, sin_t, geom, "half"
        )
        out, lse = ref_mla_decode_absorb(
            q_nope=q_nope, q_rope=q_rope, w_uk=p["w_uk"], **common
        )
        np.testing.assert_allclose(out, want_out, rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(lse, want_lse, rtol=1e-5)


class TestParityGateIsNotVacuous:
    """The on-device gate in ``builders.mla.verify_mla_decode`` fails what it must.

    Pure numpy, so it runs on CPU CI: a gate that passes everything would make
    every GPU result meaningless.
    """

    def test_gate(self):
        from builders.mla import verify_mla_decode as v

        # Sequence 1 has no query, sequence 2 no keys: token 1 is the dead row.
        p = v.make_problem(heads=64, q_lens=(1, 0, 1), k_lens=(20, 5, 0))
        ref_out = np.zeros((2, 64, 512))
        ref_lse = np.array([np.zeros(64), np.full(64, -np.inf)])

        def ok(o, lse):
            return v.compare(p, o, lse, ref_out, ref_lse)[0]

        assert ok(ref_out.copy(), ref_lse.copy())
        o = ref_out.copy()
        o[0, 3, 7] = np.nan
        assert not ok(o, ref_lse)  # an unwritten slot
        o = ref_out.copy()
        o[0, 0, 0] = 0.05
        assert not ok(o, ref_lse)  # over atol where ref = 0
        o = ref_out.copy()
        o[1, 0, 0] = 1e-3
        assert not ok(o, ref_lse)  # a dead row must be exactly 0
        lse = ref_lse.copy()
        lse[1, 0] = 0.0
        assert not ok(ref_out, lse)  # a dead lse must be -inf
        lse = ref_lse.copy()
        lse[0, 5] = 0.05
        assert not ok(ref_out, lse)  # a live lse error

    def test_relative_term_scales_with_the_output(self):
        from builders.mla import verify_mla_decode as v

        p = v.make_problem(heads=64, q_lens=(1,), k_lens=(20,))
        ref_out = np.full((1, 64, 512), 8.0)
        ref_lse = np.zeros((1, 64))
        # One bf16 ulp at 8 (0.0625) is over the absolute bound alone but
        # inside atol + rtol * |ref| = 0.2; ten times that is not.
        assert v.compare(p, ref_out + 0.0625, ref_lse, ref_out, ref_lse)[0]
        assert not v.compare(p, ref_out + 0.625, ref_lse, ref_out, ref_lse)[0]

    def test_unused_cache_rows_are_poisoned(self):
        from builders.mla import verify_mla_decode as v

        p = v.make_problem(heads=64, q_lens=(1, 1), k_lens=(20, 16))
        # Pages hold 40 rows' worth of 16-row pages for 36 keys: 12 unowned rows.
        assert np.isnan(p.c_kv).any(axis=-1).sum() == p.c_kv.shape[0] * 16 - 36

    def test_unused_block_table_slots_are_poisoned(self):
        # Slots past a sequence's last page are -1, a page id that faults if
        # a lookup (the empty-segment guard, the prefetch clamps) ever uses it.
        from builders.mla import verify_mla_decode as v

        p = v.make_problem(heads=64, q_lens=(1, 1, 1), k_lens=(20, 16, 0))
        assert np.all(p.block_table[0] >= 0)  # two pages, both used
        assert p.block_table[1, 0] >= 0 and p.block_table[1, 1] == -1
        assert np.all(p.block_table[2] == -1)  # S_k = 0 owns no page
        clean = v.make_problem(
            heads=64, q_lens=(1, 1, 1), k_lens=(20, 16, 0), poison_unused=False
        )
        assert np.all(clean.block_table >= 0)
