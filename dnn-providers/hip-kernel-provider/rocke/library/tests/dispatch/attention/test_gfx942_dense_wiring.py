# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the gfx942 ``attention_dense`` dispatcher wiring.

Required by ``library/dispatch/AGENTS.md`` step 4. Covers:
  - the candidate is registered and discoverable, with the right spec_id/algorithm
  - OPT-IN ONLY: ``algorithm="auto"`` never selects it, despite priority 3 outranking
    every other attention candidate
  - ``spec_id`` is an equivalent opt-in door
  - routing on gfx942, and rejection of every out-of-scope request
  - ``dense_persistent``: 'auto' turns the persistent grid on once there is enough
    work; an explicit 'on' is accepted
  - non-persistent gfx942 dense reads ``batch`` / ``seqlen_q`` / ``seqlen_kv`` as
    runtime kernel params, so those fields drop out of ``kernel_name()`` and the
    dispatched signature includes them. The persistent grid still bakes batch.

The priority-3 tests are the load-bearing ones: the arm sorts ahead of every other
candidate, so the opt-in check is the ONLY thing keeping a correctness-first P0 kernel
off the default gfx942 path.
"""

from __future__ import annotations

import unittest

import kernels.common.attention_unified as au
from dispatch.attention import (
    AttentionMaskType,
    AttentionRequest,
    attention_candidates,
    dispatch_attention,
    registered_attention_combos,
)

# gfx942's own spec factory. The package-level ``dense_spec_for_request`` routes
# here by ``req.arch``; importing it from the arch module pins these tests to the
# gfx942 factory and its arch guard without going through that routing.
from dispatch.attention.gfx942 import _dense_spec, dense_spec_for_request
from kernels.common.attention_dense_spec import AttentionDenseSpec
from kernels.gfx942.attention_dense import (
    Gfx942AttentionDenseSpec,
    build_attention_dense,
    supports_attention_dense,
)

_NAME = "attention_gfx942_dense"
_SPEC_ID = "gfx942_attention_dense"


def _req(**kw) -> AttentionRequest:
    base = dict(
        batch=1,
        nhead_q=128,
        nhead_k=8,
        seqlen_q=2048,
        seqlen_k=2048,
        hdim_q=128,
        hdim_v=128,
        arch="gfx942",
        dtype="bf16",
        mask_type=1,
        algorithm="attention_dense",
    )
    base.update(kw)
    return AttentionRequest(**base)


def _candidate():
    return next(c for c in attention_candidates() if c.name == _NAME)


class _Gfx942Arch:
    """Pin _RESOLVED_ATTENTION_ARCH so routing does not depend on the host GPU."""

    def __enter__(self):
        self._old = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = "gfx942"
        return self

    def __exit__(self, *_):
        au._RESOLVED_ATTENTION_ARCH = self._old


class TestGfx942DenseRegistration(unittest.TestCase):
    def test_candidate_is_registered(self):
        self.assertIn(_NAME, [c.name for c in attention_candidates()])

    def test_spec_id_and_algorithm(self):
        c = _candidate()
        self.assertEqual(c.spec_id, _SPEC_ID)
        self.assertEqual(c.algorithm, "attention_dense")

    def test_priority_outranks_every_other_candidate(self):
        """Documents WHY the opt-in gate matters: nothing else holds this arm back."""
        c = _candidate()
        others = [o for o in attention_candidates() if o.name != _NAME]
        self.assertTrue(all(c.priority <= o.priority for o in others))


class TestGfx942DenseOptIn(unittest.TestCase):
    def test_auto_algorithm_never_selects_it(self):
        with _Gfx942Arch():
            ok, why = _candidate().admits(_req(algorithm="auto", spec_id="auto"))
            self.assertFalse(ok, "attention_dense must never be auto-selected")
            self.assertIn("opt-in", why)
            routed = dispatch_attention(_req(algorithm="auto", spec_id="auto"))
            self.assertNotEqual(routed.candidate.name, _NAME)

    def test_spec_id_is_an_equivalent_opt_in(self):
        with _Gfx942Arch():
            ok, why = _candidate().admits(_req(algorithm="auto", spec_id=_SPEC_ID))
            self.assertTrue(ok, why)

    def test_routes_on_explicit_algorithm(self):
        with _Gfx942Arch():
            r = dispatch_attention(_req())
            self.assertEqual(r.candidate.name, _NAME)
            from kernels.gfx942.attention_dense import Gfx942AttentionDenseSpec

            self.assertIsInstance(r.spec, Gfx942AttentionDenseSpec)
            self.assertIn("gfx942", r.spec.kernel_name())
            self.assertNotEqual(r.grid, (0, 0, 0))


class TestGfx942DenseSupportGates(unittest.TestCase):
    """Arch, dtype and feature rejections are the declared ``Capability``'s job;
    only what capability cannot express as data stays in the predicate. Each test
    below asserts which of the two turned the request down, so a gate silently
    migrating between them is a failure rather than a rename."""

    def test_rejects_non_gfx942_arch(self):
        ok, why = _candidate().admits(_req(arch="gfx950"))
        self.assertFalse(ok)
        self.assertIn("capability", why)
        self.assertIn("gfx942", why)

    def test_rejects_unsupported_dtype(self):
        with _Gfx942Arch():
            ok, why = _candidate().admits(_req(dtype="fp8"))
            self.assertFalse(ok)
            self.assertIn("capability", why)
            self.assertIn("fp8", why)

    def test_admits_sliding_window(self):
        with _Gfx942Arch():
            ok, _ = _candidate().admits(_req(sliding_window=64))
            self.assertTrue(ok)

    def test_rejects_sinks(self):
        with _Gfx942Arch():
            ok, why = _candidate().admits(_req(use_sinks=True))
            self.assertFalse(ok)
            self.assertIn("capability", why)
            self.assertIn("sinks", why)

    def test_rejects_a_partial_tile_and_names_the_length(self):
        """_dense_spec sets ragged=True for any self-attention length that is not a
        tile multiple -- most real serving shapes. The kernel must decline, not
        select-then-fail. Capability cannot see this one: it is a property of the
        BUILT spec, so it stays in the predicate. The graph is not ragged, so the
        reason names the length and the tile instead of saying "ragged"."""
        cases = (
            (384, ["seqlen_q 384 is not a multiple of block_m 256"]),
            (1152, ["seqlen_q 1152 is not a multiple of block_m 256"]),
            (
                1000,
                [
                    "seqlen_q 1000 is not a multiple of block_m 256",
                    "seqlen_kv 1000 is not a multiple of block_n 64",
                ],
            ),
        )
        with _Gfx942Arch():
            for seqlen, expected in cases:
                with self.subTest(seqlen=seqlen):
                    ok, why = _candidate().admits(
                        _req(seqlen_q=seqlen, seqlen_k=seqlen)
                    )
                    self.assertFalse(ok)
                    self.assertNotIn("capability", why)
                    self.assertNotIn("ragged", why)
                    for part in expected:
                        self.assertIn(part, why)
            # 1152 is a block_n multiple, so only the query length is named.
            ok, why = _candidate().admits(_req(seqlen_q=1152, seqlen_k=1152))
            self.assertNotIn("seqlen_kv", why)


class TestGfx942BottomRightSafety(unittest.TestCase):
    def test_moving_bottom_right_declines_at_capability(self):
        for mask_type in (AttentionMaskType.BOTTOM_RIGHT_CAUSAL, 2):
            with self.subTest(mask_type=mask_type), _Gfx942Arch():
                ok, why = _candidate().admits(
                    _req(
                        seqlen_q=2048,
                        seqlen_k=4096,
                        mask_type=mask_type,
                        dense_persistent="off",
                    )
                )
                self.assertFalse(ok)
                self.assertIn("capability", why)
                self.assertIn("causal_bottom_right", why)

    def test_direct_factory_rejects_moving_bottom_right(self):
        for mask_type in (AttentionMaskType.BOTTOM_RIGHT_CAUSAL, 2):
            with self.subTest(mask_type=mask_type):
                with self.assertRaisesRegex(ValueError, "causal_bottom_right"):
                    _dense_spec(
                        _req(
                            seqlen_q=2048,
                            seqlen_k=4096,
                            mask_type=mask_type,
                            dense_persistent="off",
                        )
                    )

    def test_concrete_support_rejects_shared_bottom_right_spec(self):
        common = dict(
            batch=1,
            seqlen_q=2048,
            seqlen_kv=4096,
            num_query_heads=128,
            num_kv_heads=8,
            head_size=128,
            causal=True,
            causal_bottom_right=True,
            dtype="bf16",
        )
        spec = AttentionDenseSpec(**common)
        ok, why = supports_attention_dense(spec, arch="gfx942")
        self.assertFalse(ok)
        self.assertIn("causal_bottom_right", why)
        with self.assertRaisesRegex(ValueError, "causal_bottom_right"):
            Gfx942AttentionDenseSpec(**common)

    def test_equal_length_bottom_right_preserves_persistent_policy(self):
        common = dict(
            seqlen_q=8192,
            seqlen_k=8192,
            dense_persistent="auto",
        )
        mask_pairs = (
            (AttentionMaskType.TOP_LEFT_CAUSAL, 2),
            (1, AttentionMaskType.BOTTOM_RIGHT_CAUSAL),
        )
        with _Gfx942Arch():
            for top_left, bottom_right in mask_pairs:
                with self.subTest(top_left=top_left, bottom_right=bottom_right):
                    top_left_spec = _dense_spec(_req(mask_type=top_left, **common))
                    bottom_right_req = _req(mask_type=bottom_right, **common)
                    bottom_right_spec = _dense_spec(bottom_right_req)
                    self.assertEqual(bottom_right_spec, top_left_spec)
                    self.assertFalse(bottom_right_spec.causal_bottom_right)
                    self.assertTrue(bottom_right_spec.persistent)
                    ok, why = _candidate().admits(bottom_right_req)
                    self.assertTrue(ok, why)


class TestGfx942DensePersistent(unittest.TestCase):
    def test_auto_persistent_turns_on_for_large_sq(self):
        """Post-P4 (ledger row 16): 'auto' turns the persistent grid-stride variant
        ON once there is enough work to fill the grid -- the large-Sq prefill
        regime -- and the request is accepted."""
        with _Gfx942Arch():
            req = _req(seqlen_q=8192, seqlen_k=8192, dense_persistent="auto")
            ok, why = _candidate().admits(req)
            self.assertTrue(ok, why)
            self.assertTrue(_dense_spec(req).persistent)

    def test_explicit_persistent_on_is_accepted_and_builds_persistent(self):
        """Post-P4 the persistent variant ships, so an explicit 'on' is accepted
        and yields a genuinely persistent spec -- never silently downgraded to a
        default-grid kernel."""
        with _Gfx942Arch():
            req = _req(dense_persistent="on")
            ok, why = _candidate().admits(req)
            self.assertTrue(ok, why)
            self.assertTrue(_dense_spec(req).persistent)


class TestGfx942DenseWavesPerEu(unittest.TestCase):
    def test_shipped_policy_and_explicit_override(self):
        self.assertEqual(_dense_spec(_req()).waves_per_eu, 2)
        self.assertEqual(
            _dense_spec(_req(hdim_q=64, hdim_v=64)).waves_per_eu,
            4,
        )
        overridden = _dense_spec(_req(dense_waves_per_eu=3))
        self.assertEqual(overridden.waves_per_eu, 3)
        self.assertIn("wpe3", overridden.kernel_name())
        self.assertEqual(
            build_attention_dense(overridden, arch="gfx942").attrs["waves_per_eu"],
            3,
        )

    def test_invalid_override_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dense_waves_per_eu"):
            _dense_spec(_req(dense_waves_per_eu=9))

    def _swept_waves(self, level: str, *, pin: int = 0) -> set[int]:
        return {
            spec.waves_per_eu
            for _candidate, spec in registered_attention_combos(
                _req(dense_waves_per_eu=pin),
                candidate_prefix=_NAME,
                sweep_level=level,
            )
        }

    def test_production_and_full_sweeps_expand_wpe(self):
        self.assertEqual(self._swept_waves("production"), {2, 4})
        self.assertEqual(self._swept_waves("full"), {1, 2, 3, 4})

    def test_explicit_override_pins_sweep(self):
        self.assertEqual(self._swept_waves("production", pin=3), {3})
        self.assertEqual(self._swept_waves("full", pin=3), {3})


class TestGfx942DenseSpecIdentity(unittest.TestCase):
    def test_kernel_name_follows_the_runtime_shape_contract(self):
        """Non-persistent gfx942 dense takes batch and both seqlens as kernel
        params, so one name covers every batch. The persistent grid still bakes
        batch into the symbol, and the dispatched signature matches that split."""
        from kernels.gfx942.attention_dense import attention_dense_signature

        with _Gfx942Arch():
            runtime = [
                dispatch_attention(_req(batch=b, dense_persistent="off")).spec
                for b in (1, 2, 4)
            ]
            self.assertTrue(all(s.runtime_shape for s in runtime))
            self.assertEqual(len({s.kernel_name() for s in runtime}), 1)
            self.assertNotRegex(runtime[0].kernel_name(), r"_b\d+")
            names = [p["name"] for p in attention_dense_signature(runtime[0])]
            self.assertEqual(
                names,
                [
                    "q_ptr",
                    "k_ptr",
                    "v_ptr",
                    "o_ptr",
                    "scale",
                    "batch",
                    "seqlen_q",
                    "seqlen_kv",
                ],
            )

            baked = [
                dispatch_attention(_req(batch=b, dense_persistent="on")).spec
                for b in (1, 2, 4)
            ]
            self.assertTrue(all(not s.runtime_shape for s in baked))
            self.assertEqual(len({s.kernel_name() for s in baked}), 3)
            baked_names = [p["name"] for p in attention_dense_signature(baked[0])]
            self.assertEqual(baked_names, ["q_ptr", "k_ptr", "v_ptr", "o_ptr", "scale"])

    def test_support_implies_the_dispatched_spec_builds(self):
        """The dispatch-level half of the supports/build contract: the spec the
        dispatcher actually selects (persistent auto-on for this large-Sq shape,
        post-P4) is exactly what the builder emits."""
        with _Gfx942Arch():
            req = _req()
            self.assertTrue(_candidate().admits(req)[0])
            spec = _dense_spec(req)
            kd = build_attention_dense(spec, arch="gfx942")
            self.assertEqual(kd.name, dispatch_attention(req).spec.kernel_name())


class TestGfx942SlidingWindow(unittest.TestCase):
    """Sliding-window pass-through and capability tests, mirroring gfx950's suite."""

    def test_sliding_window_zero_by_default(self):
        with _Gfx942Arch():
            spec = _dense_spec(_req())
            self.assertEqual(spec.sliding_window, 0)

    def test_sliding_window_passes_through_to_spec(self):
        with _Gfx942Arch():
            spec = _dense_spec(_req(sliding_window=128))
            self.assertEqual(spec.sliding_window, 128)

    def test_sliding_window_appears_in_kernel_name(self):
        with _Gfx942Arch():
            spec = _dense_spec(_req(sliding_window=256))
            self.assertIn("swa256", spec.kernel_name())

    def test_different_window_sizes(self):
        with _Gfx942Arch():
            for window in (64, 128, 256):
                spec = _dense_spec(_req(sliding_window=window))
                self.assertEqual(spec.sliding_window, window)

    def test_sliding_window_in_supports_features(self):
        self.assertIn("sliding_window", _candidate().capability.supports_features)

    def test_sliding_window_requires_causal(self):
        """sliding_window without causal is rejected by _dense_spec (spec validates it)."""
        with _Gfx942Arch():
            ok, why = _candidate().admits(_req(sliding_window=128, mask_type=0))
            self.assertFalse(ok)
            self.assertNotIn("capability", why)


class TestGfx942DirectFactoryHonoursRequest(unittest.TestCase):
    """The public factory plus ``supports_attention_dense`` is the pair the
    IngestorGenerator profile tools call, without ``Capability`` in front. Every
    request feature the factory cannot carry into the spec must therefore either
    reach the spec (so the predicate can refuse it) or make the factory raise. A
    field it drops turns an unsupported request into a plain dense one."""

    def test_sinks_reach_the_spec_and_the_predicate_refuses(self):
        spec = dense_spec_for_request(_req(use_sinks=True))
        self.assertTrue(spec.use_sinks)
        ok, why = supports_attention_dense(spec, arch="gfx942")
        self.assertFalse(ok)
        self.assertIn("sinks", why)

    def test_fp8_request_raises(self):
        # fp8_fnuz without use_fp8 is an invalid FP8 request elsewhere
        # (validate_explicit_fp8_encoding), so it must not pass as plain dense.
        for kw in (dict(use_fp8=True), dict(fp8_fnuz=True)):
            with self.subTest(**kw):
                with self.assertRaisesRegex(ValueError, "fp8"):
                    dense_spec_for_request(_req(**kw))

    def test_zero_kv_heads_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "nhead_k must be positive"):
            dense_spec_for_request(_req(nhead_k=0))

    def test_mismatched_value_head_size_raises(self):
        with self.assertRaisesRegex(ValueError, "hdim_q == hdim_v"):
            dense_spec_for_request(_req(hdim_q=128, hdim_v=64))

    def test_gfx950_variant_pins_raise(self):
        for kw, field in (
            (dict(dense_tile="bm128"), "dense_tile"),
            (dict(dense_wide_lds_dma="on"), "dense_wide_lds_dma"),
        ):
            with self.subTest(**kw):
                with self.assertRaisesRegex(ValueError, field):
                    dense_spec_for_request(_req(**kw))

    def test_pins_gfx942_already_satisfies_keep_the_spec(self):
        baseline = dense_spec_for_request(_req())
        self.assertFalse(baseline.use_sinks)
        for kw in (dict(dense_tile="default"), dict(dense_wide_lds_dma="off")):
            with self.subTest(**kw):
                self.assertEqual(dense_spec_for_request(_req(**kw)), baseline)


if __name__ == "__main__":
    unittest.main()
