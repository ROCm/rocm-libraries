# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Element-path + arch-gating tests for the fused-MoE dispatcher family."""

from __future__ import annotations

import unittest

import random

from rocke.dispatch.families.moe import (
    MOE_REGISTRY,
    PROLOGUE_SPEC_ID,
    MoeRequest,
    dispatch_moe,
    dispatch_moe_plan,
    moe_candidates,
    moe_launch_grid,
    moe_launch_kind,
    moe_sweep_space,
)
from rocke.dispatch.families.moe import (
    _deferred_perf_knobs,
    _num_m_blocks,
    _PINNED_PERF_KNOBS,
)
from rocke.instances.common.moe_fused_mega_fp8_tuned import (
    BAND_GEOMETRY,
    BAND_RANGE,
    MAX_TUNED_TOKENS,
    TOKEN_BANDS,
    TUNED_SHAPE,
    band_for,
)


def _moe(arch="gfx950", dtype="fp16", **kw):
    base = dict(
        num_tokens=128,
        hidden=7168,
        intermediate=2048,
        num_experts=256,
        top_k=8,
        arch=arch,
        dtype=dtype,
    )
    base.update(kw)
    return MoeRequest(**base)


class TestMoeDispatch(unittest.TestCase):
    def test_fp16_selects_f16_mega(self):
        r = dispatch_moe(_moe(dtype="fp16"))
        self.assertEqual(r.candidate.spec_id, "mega_f16")

    def test_bf16_selects_f16_mega(self):
        r = dispatch_moe(_moe(dtype="bf16"))
        self.assertEqual(r.candidate.spec_id, "mega_f16")
        self.assertEqual(r.spec.dtype, "bf16")

    def test_fp8_selects_fp8_mega(self):
        r = dispatch_moe(_moe(dtype="fp8"))
        self.assertEqual(r.candidate.spec_id, "mega_fp8")
        # fp8 hero atom K=128.
        self.assertEqual(r.spec.gate_up_k, 128)

    def test_rejects_unknown_dtype(self):
        with self.assertRaises(ValueError):
            dispatch_moe(_moe(dtype="i8"))

    def test_rejects_topk_gt_experts(self):
        with self.assertRaises(ValueError):
            dispatch_moe(_moe(num_experts=4, top_k=8))

    def test_rejects_gfx942_no_atom(self):
        # gfx942 lacks the 16x16x32 MoE f16 atom -> unsupported.
        with self.assertRaises(ValueError):
            dispatch_moe(_moe(arch="gfx942", dtype="fp16"))

    def test_rejects_rdna_arch(self):
        with self.assertRaises(ValueError):
            dispatch_moe(_moe(arch="gfx1151"))

    def test_rejects_unknown_arch(self):
        with self.assertRaises(ValueError):
            dispatch_moe(_moe(arch="gfx000"))

    def test_candidates_dtype_exclusive(self):
        req = _moe(dtype="fp8")
        supported = [c for c in moe_candidates() if c.admits(req)[0]]
        self.assertEqual([c.spec_id for c in supported], ["mega_fp8"])

    def test_unique_candidate_names(self):
        names = [c.name for c in moe_candidates()]
        self.assertEqual(len(names), len(set(names)))

    def test_block_size_default(self):
        r = dispatch_moe(_moe(dtype="fp16"))
        # warp_m=1 * warp_n=4 * wave_size=64 = 256.
        self.assertEqual(r.spec.block_size, 256)


def _tuned_req(tokens, **kw):
    """The tuned cohort's shape at one token count; ``kw`` overrides it.

    Read off :data:`TUNED_SHAPE` rather than spelled out, so a test that means
    "inside the cohort" cannot drift outside it when the record changes.
    """
    shape = dict(
        num_tokens=tokens,
        hidden=TUNED_SHAPE.hidden,
        intermediate=TUNED_SHAPE.intermediate,
        num_experts=TUNED_SHAPE.num_experts,
        top_k=TUNED_SHAPE.top_k,
        dtype="fp8",
    )
    shape.update(kw)
    return _moe(**shape)


class TestMoeTokenBands(unittest.TestCase):
    """num_tokens as a selection knob.

    The token count used to be a runtime argument that selection ignored. It
    cannot be: ``tile_m`` trades weight traffic against the row padding an
    expert's block count forces, and the two cross over three times between
    decode and prefill. Each boundary below is a measured crossover, so a test
    that only checked "some fp8 kernel comes back" would pass while the
    dispatcher handed prefill a decode kernel.
    """

    def test_each_band_selects_its_measured_winner(self):
        for tokens, expected in (
            (1, "fused_tm16"),
            (8, "fused_tm16"),
            (16, "split_coop_tm16"),
            (256, "split_coop_tm16"),
            (512, "split_coop_tm32"),
            (1024, "split_coop_tm64"),
            (4096, "split_coop_tm64"),
        ):
            with self.subTest(tokens=tokens):
                r = dispatch_moe(_tuned_req(tokens))
                self.assertEqual(r.candidate.algorithm, expected)

    def test_boundaries_are_where_they_were_measured(self):
        """The band edges specifically, since an off-by-one is invisible above."""
        for lo, hi in ((8, 16), (256, 512), (512, 1024)):
            with self.subTest(edge=(lo, hi)):
                self.assertNotEqual(
                    dispatch_moe(_tuned_req(lo)).candidate.algorithm,
                    dispatch_moe(_tuned_req(hi)).candidate.algorithm,
                )

    def test_geometry_matches_the_band(self):
        for tokens, tile_m, warp_m in ((8, 16, 1), (128, 16, 1), (512, 32, 2), (4096, 64, 4)):
            with self.subTest(tokens=tokens):
                spec = dispatch_moe(_tuned_req(tokens)).spec
                self.assertEqual((spec.tile_m, spec.warp_m), (tile_m, warp_m))
                self.assertEqual(spec.block_size, warp_m * spec.warp_n * 64)

    def test_off_cohort_shape_falls_back_rather_than_extrapolating(self):
        """A crossover measured at I=768 says nothing about another shape."""
        for kw in (
            dict(intermediate=1536),
            dict(num_experts=64),
            dict(top_k=4),
            dict(hidden=4096),
        ):
            with self.subTest(**kw):
                r = dispatch_moe(_tuned_req(4096, **kw))
                self.assertEqual(r.candidate.spec_id, "mega_fp8")

    def test_split_bands_report_two_launches(self):
        self.assertEqual(len(dispatch_moe_plan(_tuned_req(4))), 1)
        for tokens in (16, 512, 4096):
            with self.subTest(tokens=tokens):
                plan = dispatch_moe_plan(_tuned_req(tokens))
                self.assertEqual(len(plan), 2)
                # Both stages must agree on tile_m or the second reads the
                # first's intermediate with the wrong row blocking.
                self.assertEqual(plan[0].spec.tile_m, plan[1].spec.tile_m)
                # ...and disagree on warp_n, which is the point of splitting.
                self.assertEqual((plan[0].spec.warp_n, plan[1].spec.warp_n), (1, 4))

    def test_stage2_is_not_an_answer_to_auto(self):
        """Half an algorithm must not be returned as the whole one."""
        for c in moe_candidates():
            if c.name.endswith("_stage2"):
                self.assertFalse(c.admits(_tuned_req(4096))[0], c.name)

    def test_bands_do_not_collide_in_the_kernel_id(self):
        """Distinct kernels need distinct spec hashes, or the sweep space and
        the by-identifier replay both silently fold them together."""
        hashes = set()
        for tokens in (4, 16, 512, 4096):
            for r in dispatch_moe_plan(_tuned_req(tokens)):
                hashes.add(r.kernel_id.spec_hash)
        self.assertEqual(len(hashes), 7)  # 1 fused + 3 split x 2 stages

    def test_static_scale_selection_states_its_precondition(self):
        """These specs read the intermediate's scale instead of deriving it, so
        an uninitialised buffer yields NaN at full speed -- no launch error and
        no suspicious latency. The obligation rides with the selection."""
        r = dispatch_moe(_tuned_req(4096))
        self.assertTrue(r.spec.static_inter_scale)
        self.assertTrue(
            any("InterScale" in line for line in r.explanation), r.explanation
        )

    def test_no_precondition_claimed_when_none_applies(self):
        r = dispatch_moe(_tuned_req(4))
        self.assertFalse(r.spec.static_inter_scale)
        self.assertFalse(any("InterScale" in line for line in r.explanation))


#: The routing contract for the reference shape, as one table: token count ->
#: (candidate, launches in the plan, tile_m). Written out rather than derived
#: from ``TOKEN_BANDS`` on purpose -- deriving it would make the test agree
#: with whatever the table says, and the thing under test is that the table
#: still says what was measured.
#:
#: T=32 is the contested row. A brief handed to this review, and the (already
#: red) ``library/tests/serve/test_serve_moe.py``, both expect T=32 to select
#: the single fused launch. That is the pre-banding behaviour: the recorded
#: sweep has the fused kernel behind at T=16, T=32 and T=64, and the boundary
#: was deliberately moved from 64 down to 8 because of it. The row below states
#: what the dispatcher does; changing it is a re-measurement, not an edit.
_TUNED_ROUTING = (
    (1, "moe_fused_tm16", 1, 16),
    (32, "moe_split_coop_tm16", 2, 16),
    (64, "moe_split_coop_tm16", 2, 16),
    (128, "moe_split_coop_tm16", 2, 16),
    (256, "moe_split_coop_tm16", 2, 16),
    (512, "moe_split_coop_tm32", 2, 32),
)


class TestTunedRoutingContract(unittest.TestCase):
    """The selection for the reference shape, pinned end to end."""

    def test_candidate_launch_count_and_tile_m(self):
        for tokens, candidate, launches, tile_m in _TUNED_ROUTING:
            with self.subTest(tokens=tokens):
                plan = dispatch_moe_plan(_tuned_req(tokens))
                self.assertEqual(plan[0].candidate.name, candidate)
                self.assertEqual(len(plan), launches)
                for stage in plan:
                    self.assertEqual(stage.spec.tile_m, tile_m)

    def test_the_split_plan_is_gate_up_then_down(self):
        """Order is load-bearing: stage 2 reads what stage 1 publishes."""
        for tokens, _, launches, _ in _TUNED_ROUTING:
            if launches != 2:
                continue
            with self.subTest(tokens=tokens):
                plan = dispatch_moe_plan(_tuned_req(tokens))
                self.assertFalse(plan[0].candidate.name.endswith("_stage2"))
                self.assertTrue(plan[1].candidate.name.endswith("_stage2"))

    def test_t32_spec_is_fully_pinned(self):
        """Every field the reference configuration names, at one token count.

        A per-field assertion rather than a spec_hash, so a regression says
        which knob moved instead of only that something did.
        """
        spec = dispatch_moe(_tuned_req(32)).spec
        for field, want in (
            ("tile_m", 16),
            ("tile_n_inter", 128),
            ("tile_n_down", 128),
            ("tile_k_gu", 32),
            ("tile_k_down", 64),
            ("warp_m", 1),
            ("warp_n", 1),
            ("gate_up_k", 128),
            ("down_k", 128),
            ("hidden_group_k", 128),
            ("use_fused_kloop", True),
            ("swizzle_gu", True),
            ("swizzle_down", True),
        ):
            with self.subTest(field=field):
                self.assertEqual(getattr(spec, field), want)

    def test_t512_widens_the_tile_and_goes_cooperative(self):
        spec = dispatch_moe(_tuned_req(512)).spec
        self.assertEqual((spec.tile_m, spec.warp_m), (32, 2))
        self.assertTrue(spec.coop_b_lds)
        self.assertEqual(len(dispatch_moe_plan(_tuned_req(512))), 2)

    def test_selection_is_reproducible(self):
        """Same request, same answer -- the property the compile cache rests on."""
        for tokens, _, _, _ in _TUNED_ROUTING:
            with self.subTest(tokens=tokens):
                first = dispatch_moe(_tuned_req(tokens))
                again = dispatch_moe(_tuned_req(tokens))
                self.assertEqual(first.candidate.name, again.candidate.name)
                self.assertEqual(
                    first.kernel_id.selection_key, again.kernel_id.selection_key
                )


class TestBandTableIsWellFormed(unittest.TestCase):
    """Structural properties of the band table, independent of its values.

    A band table that overlaps admits two candidates at one token count and
    resolves the tie by name; one with a gap admits none and drops the request
    to the untuned generic candidate. Neither shows up as an error, so neither
    is caught by asking "did some kernel come back".
    """

    def test_bands_are_contiguous_and_cover_the_measured_range(self):
        self.assertEqual(TOKEN_BANDS[0][0], 1)
        self.assertEqual(TOKEN_BANDS[-1][1], MAX_TUNED_TOKENS)
        for (_, prev_hi, *_), (next_lo, *_) in zip(TOKEN_BANDS, TOKEN_BANDS[1:]):
            self.assertEqual(next_lo, prev_hi + 1)

    def test_geometry_and_ranges_are_projections_of_one_table(self):
        ids = [spec_id for _, _, spec_id, _, _ in TOKEN_BANDS]
        self.assertEqual(sorted(BAND_GEOMETRY), sorted(ids))
        self.assertEqual(sorted(BAND_RANGE), sorted(ids))
        self.assertEqual(len(set(ids)), len(ids))

    def test_every_token_count_in_range_lands_in_exactly_one_band(self):
        edges = {lo for lo, _, _, _, _ in TOKEN_BANDS}
        edges |= {hi for _, hi, _, _, _ in TOKEN_BANDS}
        probes = sorted(
            {1, 2, 7, 9, 15, 33, 255, 257, 511, 513, MAX_TUNED_TOKENS}
            | edges
            | {e + 1 for e in edges if e < MAX_TUNED_TOKENS}
        )
        for tokens in probes:
            with self.subTest(tokens=tokens):
                hits = [
                    spec_id
                    for lo, hi, spec_id, _, _ in TOKEN_BANDS
                    if lo <= tokens <= hi
                ]
                self.assertEqual(len(hits), 1, hits)
                self.assertEqual(band_for(tokens), hits[0])

    def test_no_band_is_claimed_past_the_measured_bound(self):
        """The table used to end in an open-ended sentinel, which turned
        "never measured" into "the widest tile wins"."""
        self.assertIsNone(band_for(MAX_TUNED_TOKENS + 1))
        self.assertIsNone(band_for(10**9))

    def test_selection_is_total_over_the_whole_token_range(self):
        """Every count is served, and at most one band ever claims it.

        Totality is the generic fp8 candidate's job -- it admits any valid
        request -- and exclusivity is the band table's. Two bands claiming one
        count would be resolved by candidate name, which is deterministic and
        meaningless.
        """
        banded = {f"moe_{spec_id}" for spec_id in BAND_GEOMETRY}
        for tokens in (1, 8, 9, 256, 512, 4096, 4097, 100_000):
            with self.subTest(tokens=tokens):
                req = _tuned_req(tokens)
                admitted = [c.name for c in moe_candidates() if c.admits(req)[0]]
                self.assertIn("moe_fused_mega_fp8", admitted)
                self.assertLessEqual(len(banded & set(admitted)), 1, admitted)
                self.assertEqual(dispatch_moe(req).candidate.name, admitted[0])

    def test_out_of_range_falls_back_instead_of_extrapolating(self):
        r = dispatch_moe(_tuned_req(MAX_TUNED_TOKENS + 1))
        self.assertEqual(r.candidate.spec_id, "mega_fp8")


class TestPinnedPerfKnobs(unittest.TestCase):
    """The dispatched spec must not inherit a scheduling knob from the instance.

    These were environment variables inside the kernel, so the same request
    built different ISA on two machines and a spec_hash from one replayed the
    wrong binary on the other.
    """

    def test_every_dispatched_fp8_spec_states_them(self):
        deferred = set(_deferred_perf_knobs())
        for tokens, _, _, _ in _TUNED_ROUTING:
            for stage in dispatch_moe_plan(_tuned_req(tokens)):
                for knob, want in _PINNED_PERF_KNOBS.items():
                    if knob in deferred:
                        continue
                    with self.subTest(tokens=tokens, knob=knob):
                        self.assertEqual(getattr(stage.spec, knob), want)

    def test_the_untuned_fallback_states_them_too(self):
        spec = dispatch_moe(_tuned_req(4096, num_experts=64)).spec
        for knob, want in _PINNED_PERF_KNOBS.items():
            if knob in _deferred_perf_knobs():
                continue
            with self.subTest(knob=knob):
                self.assertEqual(getattr(spec, knob), want)

    def test_no_pinned_knob_is_dead(self):
        """Pinning a name the spec does not declare is a no-op the lookup hides.

        The tolerance below existed because the instance was growing these
        fields on a parallel branch, so a pin could legitimately lead the
        field. Now that it has landed, a name still on the deferred list is not
        early -- it is pinning something that no longer exists, which reads as
        dispatcher control over codegen that nothing implements.
        """
        self.assertEqual(sorted(_deferred_perf_knobs()), [])

    def test_a_knob_the_instance_lacks_is_reported_not_dropped(self):
        """The instance is growing these fields on another branch. Until one
        lands, the selection has to admit it does not control it."""
        deferred = _deferred_perf_knobs()
        r = dispatch_moe(_tuned_req(32))
        for knob in deferred:
            with self.subTest(knob=knob):
                self.assertFalse(hasattr(r.spec, knob))
                self.assertTrue(
                    any(knob in line for line in r.explanation), r.explanation
                )
        if not deferred:
            self.assertFalse(any("inherits" in line for line in r.explanation))


class TestLaunchGeometryIsReal(unittest.TestCase):
    """A DispatchResult that reports (0, 0, 0) and no signature looks complete
    and cannot be launched, so the caller re-derives geometry beside the
    dispatcher and the two drift."""

    def test_every_selection_reports_a_launchable_grid_and_signature(self):
        for tokens, _, _, _ in _TUNED_ROUTING:
            for stage in dispatch_moe_plan(_tuned_req(tokens), with_prologue=True):
                with self.subTest(tokens=tokens, candidate=stage.candidate.name):
                    self.assertTrue(all(d > 0 for d in stage.grid), stage.grid)
                    self.assertTrue(all(d > 0 for d in stage.block), stage.block)
                    self.assertTrue(stage.signature)

    def test_grid_tracks_the_selected_row_blocking(self):
        """A wider tile_m means fewer sorted token blocks, hence a shorter y."""
        narrow = dispatch_moe(_tuned_req(512))
        wide = dispatch_moe(_tuned_req(4096))
        self.assertEqual((narrow.spec.tile_m, wide.spec.tile_m), (32, 64))
        self.assertGreater(wide.grid[1], narrow.grid[1])


class TestBlockCountIsABound(unittest.TestCase):
    """The sorted-token-block count reported before routing exists.

    Under-counting here is the worst failure the dispatcher can produce: the
    launch is short some blocks, so the tokens in them get no output, and
    nothing faults. Spreading slots evenly over the experts computes the
    average rather than the bound and does exactly that.
    """

    @staticmethod
    def _blocks(histogram, tile_m):
        return sum(-(-n // tile_m) for n in histogram if n)

    def test_bound_holds_for_lopsided_routing(self):
        rng = random.Random(0)
        for trial in range(200):
            tokens = rng.choice([1, 7, 32, 128, 512, 4096])
            experts = rng.choice([8, 32, 128, 256])
            top_k = rng.choice([1, 2, 4, 8])
            tile_m = rng.choice([16, 32, 64])
            histogram = [0] * experts
            for _ in range(tokens * top_k):
                histogram[rng.randrange(experts)] += 1
            req = _moe(
                num_tokens=tokens,
                num_experts=experts,
                top_k=top_k,
                dtype="fp8_e4m3",
            )
            with self.subTest(trial=trial, tokens=tokens, tile_m=tile_m):
                self.assertGreaterEqual(
                    _num_m_blocks(req, tile_m), self._blocks(histogram, tile_m)
                )

    def test_bound_covers_a_maximally_lopsided_histogram(self):
        """Every slot on one expert is the worst case an even split misses."""
        req = _moe(num_tokens=512, num_experts=128, top_k=8, dtype="fp8_e4m3")
        slots = 512 * 8
        self.assertGreaterEqual(_num_m_blocks(req, 32), self._blocks([slots], 32))

    def test_bound_is_tight_when_tokens_cannot_reach_every_expert(self):
        """A single token touches top_k experts, not all 128 of them."""
        req = _moe(num_tokens=1, num_experts=128, top_k=8, dtype="fp8_e4m3")
        self.assertEqual(_num_m_blocks(req, 16), 8)


class TestExactGridForKnownRouting(unittest.TestCase):
    """The reported grid carries a pre-routing bound, so a caller that has
    routed needs a way to ask for the real geometry rather than launching the
    bound -- the split down kernel dereferences the empty-block marker."""

    def test_exact_grid_replaces_the_bound_on_the_block_axis(self):
        """Which axis carries the blocks is the launch's business -- the
        prologue puts them on x, the GEMMs on y -- so this asserts the
        substitution, not a position."""
        blocks = 189
        for stage in dispatch_moe_plan(_tuned_req(512), with_prologue=True):
            bound = _num_m_blocks(stage.request, stage.spec.tile_m)
            exact = moe_launch_grid(stage, blocks)
            with self.subTest(candidate=stage.candidate.name):
                self.assertEqual(moe_launch_grid(stage, bound), stage.grid)
                self.assertIn(blocks, exact)
                differing = [i for i in range(3) if exact[i] != stage.grid[i]]
                self.assertEqual(len(differing), 1)
                self.assertEqual(stage.grid[differing[0]], bound)

    def test_launch_kind_is_answerable_without_reading_plan_position(self):
        plan = dispatch_moe_plan(_tuned_req(512), with_prologue=True)
        kinds = [moe_launch_kind(stage) for stage in plan]
        self.assertEqual(len(set(kinds)), len(kinds))
        self.assertIn("down", kinds)


class TestActivationPrologue(unittest.TestCase):
    """The gather/rescale prologue as a dispatch candidate.

    It was reachable only by an adapter importing the instance module, which
    left ``tile_m`` -- the row count the activation scale it publishes is
    uniform over, and which must equal the consuming GEMM's -- outside dispatch.
    """

    def _prologue(self, tokens, **kw):
        return dispatch_moe(_tuned_req(tokens, spec_id=PROLOGUE_SPEC_ID, **kw))

    def test_registered_and_buildable(self):
        candidate = MOE_REGISTRY.get(PROLOGUE_SPEC_ID)
        self.assertIsNotNone(candidate.build)
        self.assertIsNotNone(candidate.capability)
        self.assertEqual(candidate.capability.arches, ("gfx950",))

    def test_not_an_answer_to_auto(self):
        """A prologue is not the MoE layer, the same reason stage 2 declines."""
        candidate = MOE_REGISTRY.get(PROLOGUE_SPEC_ID)
        for tokens in (1, 32, 512, 4096):
            with self.subTest(tokens=tokens):
                ok, why = candidate.admits(_tuned_req(tokens))
                self.assertFalse(ok)
                self.assertIn("not the MoE layer", why)
                self.assertNotEqual(
                    dispatch_moe(_tuned_req(tokens)).candidate.name, PROLOGUE_SPEC_ID
                )

    def test_tile_m_follows_the_band_it_will_feed(self):
        """A prologue built at a different tile_m than its GEMM applies the
        wrong row's activation scale to a legal address: wrong, not slow."""
        for tokens, _, _, tile_m in _TUNED_ROUTING:
            with self.subTest(tokens=tokens):
                self.assertEqual(self._prologue(tokens).spec.tile_m, tile_m)
                self.assertEqual(
                    self._prologue(tokens).spec.tile_m,
                    dispatch_moe(_tuned_req(tokens)).spec.tile_m,
                )

    def test_scratch_bound_covers_the_requested_hidden(self):
        spec = self._prologue(32).spec
        self.assertGreaterEqual(spec.max_n_hb * 128, 2048)

    def test_plan_prepends_it_on_request_only(self):
        for tokens, _, launches, _ in _TUNED_ROUTING:
            with self.subTest(tokens=tokens):
                plain = dispatch_moe_plan(_tuned_req(tokens))
                self.assertEqual(len(plain), launches)
                withp = dispatch_moe_plan(_tuned_req(tokens), with_prologue=True)
                self.assertEqual(len(withp), launches + 1)
                self.assertEqual(withp[0].candidate.name, PROLOGUE_SPEC_ID)
                self.assertEqual([r.candidate.name for r in withp[1:]],
                                 [r.candidate.name for r in plain])

    def test_serves_shapes_outside_the_tuned_cohort(self):
        """It has no tuned geometry to overclaim: hidden, topk and the group
        count are runtime arguments."""
        r = self._prologue(128, num_experts=256, intermediate=1536)
        self.assertEqual(r.candidate.name, PROLOGUE_SPEC_ID)
        self.assertTrue(r.signature)

    def test_declines_a_hidden_that_is_not_whole_scale_groups(self):
        """The kernel emits no tail path; a partial group has no rescale
        ratio for the last elements of a vector load."""
        candidate = MOE_REGISTRY.get(PROLOGUE_SPEC_ID)
        ok, _ = candidate.admits(_tuned_req(32, spec_id=PROLOGUE_SPEC_ID, hidden=2048 + 64))
        self.assertFalse(ok)

    def test_has_its_own_compile_identity(self):
        """It must not share a spec_hash with the GEMM it feeds."""
        plan = dispatch_moe_plan(_tuned_req(32), with_prologue=True)
        hashes = {r.kernel_id.spec_hash for r in plan}
        self.assertEqual(len(hashes), len(plan))

    def test_absent_from_the_sweep_space_of_an_auto_request(self):
        self.assertTrue(moe_sweep_space(_tuned_req(32)))
        for spec in moe_sweep_space(_tuned_req(32)):
            self.assertFalse(hasattr(spec, "max_n_hb"))


if __name__ == "__main__":
    unittest.main()
