# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Config identity: one canonical id per kernel, replayable from recorded knobs,
the same on every problem, and carried through dispatch results."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from unittest import mock

from dispatch.attention import (
    ATTENTION_EXECUTION_REGISTRY,
    AttentionRequest,
    attention_dispatch_result,
    attention_tuning_spec,
    dispatch_attention,
    iter_registered_attention_combos,
    tuning_spec_with_knobs,
)
from rocke.dispatch.core import PinRefused, spec_identity

_UNIFIED = "gfx950_u2d_narrow_nw2_mw16_t4xb_llvm"
_DENSE = "gfx950_dense_persist"
_GFX942_DENSE = "gfx942_dense"


def _req(**kw) -> AttentionRequest:
    base = dict(
        batch=1,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=1024,
        seqlen_k=1024,
        hdim_q=128,
        hdim_v=128,
        arch="gfx950",
        dtype="bf16",
        mask_type=1,
    )
    base.update(kw)
    return AttentionRequest(**base)


def _candidate(spec_id: str):
    return next(
        c for c in ATTENTION_EXECUTION_REGISTRY.candidates() if c.spec_id == spec_id
    )


class TestCanonicalKnobs(unittest.TestCase):
    def test_knob_at_its_default_is_the_default_spec(self):
        for spec_id, knobs in (
            (_DENSE, {"pv_priority": 1}),
            (_UNIFIED, {"use_register_pv": False}),
        ):
            with self.subTest(spec_id=spec_id):
                req = _req()
                self.assertEqual(
                    tuning_spec_with_knobs(req, spec_id, knobs),
                    attention_tuning_spec(req, spec_id),
                )

    def test_knob_built_ids_replay_by_knobs_and_production_ids_by_id(self):
        """Replay rebuilds from (id, knobs) for any id. A bare id resolves only
        within the production set -- the full space is not searched -- so a
        combination outside it is refused without its knobs."""
        req = _req()
        for spec_id, knobs, in_production in (
            (_DENSE, {"pv_priority": 2}, True),
            (_DENSE, {"pv_priority": 2, "o_store_width": 2}, False),
            (_UNIFIED, {"use_register_pv": True, "waves_per_eu": 2}, True),
        ):
            with self.subTest(knobs=knobs):
                built = tuning_spec_with_knobs(req, spec_id, knobs)
                self.assertEqual(dict(built.knobs), knobs)
                by_knobs = attention_tuning_spec(
                    req, spec_id, built.tuning_id, knobs=dict(built.knobs)
                )
                self.assertEqual(by_knobs, built)
                if in_production:
                    self.assertEqual(
                        attention_tuning_spec(req, spec_id, built.tuning_id), built
                    )
                else:
                    with self.assertRaisesRegex(
                        PinRefused, "must be pinned with its tuning_knobs"
                    ):
                        attention_tuning_spec(req, spec_id, built.tuning_id)

    def test_knobs_that_do_not_reproduce_the_id_are_refused(self):
        req = _req()
        built = tuning_spec_with_knobs(req, _DENSE, {"pv_priority": 2})
        with self.assertRaisesRegex(ValueError, "canonicalize to"):
            attention_tuning_spec(
                req, _DENSE, built.tuning_id, knobs={"pv_priority": 3}
            )

    def test_inert_unified_knobs_are_dropped(self):
        req = _req()
        default = attention_tuning_spec(req, _UNIFIED)
        # Softmax interleave is emitted only on the transposed body.
        self.assertEqual(
            tuning_spec_with_knobs(
                req, _UNIFIED, {"use_softmax_mfma_interleave": True}
            ),
            default,
        )
        # A gated sub-knob without its gate is never read.
        self.assertEqual(
            tuning_spec_with_knobs(req, _UNIFIED, {"sched_barrier_mask": 0x8}),
            default,
        )
        gated = tuning_spec_with_knobs(
            req, _UNIFIED, {"use_sched_barrier": True, "sched_barrier_mask": 0x8}
        )
        self.assertEqual(
            dict(gated.knobs), {"use_sched_barrier": True, "sched_barrier_mask": 0x8}
        )

    def test_inert_dense_knobs_are_dropped(self):
        req = _req()
        grid = "gfx950_dense_grid"
        # The grid body never reads the persistent-only knobs.
        self.assertEqual(
            tuning_spec_with_knobs(
                req, grid, {"lazy_rescale_threshold": 2.0, "lazy_rescale": True}
            ),
            tuning_spec_with_knobs(req, grid, {"lazy_rescale_threshold": 2.0}),
        )
        self.assertEqual(
            tuning_spec_with_knobs(
                req, _DENSE, {"lazy_rescale": False, "lazy_rescale_threshold": 2.0}
            ),
            tuning_spec_with_knobs(req, _DENSE, {"lazy_rescale": False}),
        )

    def test_illegal_knobs_are_refused(self):
        req = _req()
        for spec_id, knobs, reason in (
            (_DENSE, {"seqlen_q": 64}, "not tunable"),
            (_DENSE, {"persistent": False}, "not tunable"),
            (_DENSE, {"wide_lds_dma": True}, "not tunable"),
            (_DENSE, {"block_m": 192}, "block_m"),
            (_DENSE, {"lds_num_buffers": 3}, "not tunable"),
            (_UNIFIED, {"not_a_knob": True}, "not tunable"),
        ):
            with self.subTest(knobs=knobs), self.assertRaisesRegex(ValueError, reason):
                tuning_spec_with_knobs(req, spec_id, knobs)

    def test_equal_knob_values_of_another_type_are_one_config(self):
        """``True == 1`` and ``2 == 2.0`` hash alike, so a request cache keyed
        on them must not see two configurations."""
        req = _req()
        for spec_id, typed, other in (
            (_DENSE, {"o_store_width": 1}, {"o_store_width": True}),
            (_DENSE, {"lazy_rescale_threshold": 2.0}, {"lazy_rescale_threshold": 2}),
            (_UNIFIED, {"use_register_pv": True}, {"use_register_pv": 1}),
            (_UNIFIED, {"waves_per_eu": 2}, {"waves_per_eu": 2.0}),
        ):
            with self.subTest(spec_id=spec_id, knobs=other):
                expected = tuning_spec_with_knobs(req, spec_id, typed)
                self.assertEqual(tuning_spec_with_knobs(req, spec_id, other), expected)
                self.assertEqual(dict(expected.knobs), typed)
        for spec_id, knobs in (
            (_DENSE, {"o_store_width": "1"}),
            (_UNIFIED, {"use_register_pv": 2}),
        ):
            with self.subTest(knobs=knobs), self.assertRaisesRegex(ValueError, "takes"):
                tuning_spec_with_knobs(req, spec_id, knobs)

    def test_known_wrong_knobs_are_refused_on_every_entry_point(self):
        req = _req(arch="gfx942", dtype="fp16")
        tuning = next(
            c
            for c in ATTENTION_EXECUTION_REGISTRY.candidates()
            if c.spec_id.startswith("gfx942_u2d_transposed_x8")
            and c.admits(replace(req, algorithm=c.algorithm, spec_id=c.spec_id))[0]
        )
        with self.assertRaisesRegex(ValueError, "wrong output"):
            tuning_spec_with_knobs(req, tuning.spec_id, {"use_k_hbm_direct": True})


class TestProblemIndependentIdentity(unittest.TestCase):
    def test_same_knobs_same_config_key_on_other_problems(self):
        shapes = (_req(), _req(batch=4, seqlen_q=4096, seqlen_k=4096, nhead_q=64))
        for spec_id, knobs in (
            (_DENSE, {"pv_priority": 2}),
            (_UNIFIED, {"use_register_pv": True}),
        ):
            with self.subTest(spec_id=spec_id):
                a, b = (tuning_spec_with_knobs(r, spec_id, knobs) for r in shapes)
                self.assertEqual(a.config_key, b.config_key)
                self.assertEqual(a.tuning_id, b.tuning_id)
                self.assertNotEqual(spec_identity(a), spec_identity(b))

    def test_identity_ignores_wrapper_metadata(self):
        spec = attention_tuning_spec(_req(), _UNIFIED)
        self.assertEqual(
            spec_identity(spec),
            spec_identity(replace(spec, allow_unsupported=True, candidate_name="x")),
        )

    def test_samplers_dedupe_by_tuning_id(self):
        for prefix in (
            "attention_gfx950_dense_persist",
            "attention_" + _UNIFIED,
        ):
            with self.subTest(prefix=prefix):
                ids = [
                    spec.tuning_id
                    for _c, spec in iter_registered_attention_combos(
                        _req(),
                        candidate_prefix=prefix,
                        sweep_level="full",
                        tuning_sample=12,
                        seed=3,
                    )
                ]
                self.assertEqual(len(ids), len(set(ids)))


class TestDispatchResultsCarryTheConfig(unittest.TestCase):
    def test_results_pin_the_exact_spec(self):
        candidate = _candidate(_DENSE)
        req = _req()
        default = attention_tuning_spec(req, _DENSE)
        tuned = tuning_spec_with_knobs(req, _DENSE, {"pv_priority": 2})
        a = attention_dispatch_result(req, candidate, default)
        b = attention_dispatch_result(req, candidate, tuned)
        self.assertEqual(b.kernel_id.tuning_id, tuned.tuning_id)
        self.assertNotEqual(a.kernel_id.request_hash, b.kernel_id.request_hash)
        self.assertIn(f"tuning_id={tuned.tuning_id}", b.explanation)
        self.assertTrue(any(line.startswith("spec_hash=") for line in b.explanation))
        self.assertEqual(dispatch_attention(b.request).spec, tuned)
        # The stored request survives a JSON round trip.
        restored = AttentionRequest(**json.loads(json.dumps(b.request.normalized())))
        self.assertEqual(dispatch_attention(restored).spec, tuned)

    def test_unpinned_dispatch_keeps_the_callers_request(self):
        req = _req()
        result = dispatch_attention(req)
        self.assertEqual(result.request, req)
        self.assertEqual(result.kernel_id.tuning_id, "")


def _pin(spec_id: str, spec, **kw) -> AttentionRequest:
    candidate = _candidate(spec_id)
    return _req(
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        tuning_id=spec.tuning_id,
        tuning_knobs=spec.knobs,
        **kw,
    )


class TestStoredPinsAcrossReleases(unittest.TestCase):
    """What a long-lived cache (hipDNN) sees when it replays a stored pin."""

    def test_a_changed_default_refuses_the_pin_instead_of_drifting(self):
        from dispatch.attention import gfx950_dense

        variant = gfx950_dense.GFX950_DENSE_VARIANT_BY_SPEC_ID[_DENSE]
        stored = tuning_spec_with_knobs(_req(), _DENSE, {"pv_priority": 2})
        pin = _pin(_DENSE, stored)

        unchanged = gfx950_dense._make_gfx950_attention_dense_candidate(variant)
        self.assertEqual(unchanged.select_spec(pin), stored)

        # The knobs are the same, but the default they are relative to moved:
        # the key moves with it, so the pin is refused rather than rebuilt.
        with mock.patch.object(gfx950_dense, "_DEFAULT_NUM_PERSISTENT", 300):
            upgraded = gfx950_dense._make_gfx950_attention_dense_candidate(variant)
            ok, why = upgraded.admits(pin)
            self.assertFalse(ok)
            self.assertIn("defaults they are relative to", why)
            fresh = upgraded.select_spec(replace(pin, tuning_id="auto"))
            self.assertNotEqual(fresh.config_key, stored.config_key)

    def test_stale_pins_raise_with_the_reason_and_never_fall_back(self):
        stored = tuning_spec_with_knobs(_req(), _DENSE, {"pv_priority": 2})
        cases = {
            "removed variant": (
                replace(_pin(_DENSE, stored), spec_id="gfx950_dense_retired"),
                "no registered candidate has spec_id",
            ),
            "unknown id": (
                replace(
                    _pin(_DENSE, stored),
                    tuning_id="persist_wpe2@" + "0" * 16,
                    tuning_knobs=(),
                ),
                "unknown tuning_id",
            ),
            "knobs do not reproduce the id": (
                replace(_pin(_DENSE, stored), tuning_knobs={"pv_priority": 3}),
                "canonicalize to config_key",
            ),
        }
        for label, (pin, reason) in cases.items():
            with self.subTest(label), self.assertRaises(PinRefused) as raised:
                dispatch_attention(pin)
            self.assertIn(reason, str(raised.exception))
            self.assertIsInstance(raised.exception, ValueError)
            self.assertEqual(raised.exception.tuning_id, pin.tuning_id)

    def test_the_display_stem_is_not_compared(self):
        stored = tuning_spec_with_knobs(_req(), _DENSE, {"pv_priority": 2})
        restemmed = replace(
            _pin(_DENSE, stored), tuning_id=f"renamed_wpe9@{stored.config_key}"
        )
        self.assertEqual(dispatch_attention(restemmed).spec, stored)

    def test_the_default_config_replays_without_a_search(self):
        req = _req(seqlen_q=2048, seqlen_k=2048)
        default = attention_tuning_spec(req, _DENSE)
        self.assertEqual(default.knobs, ())
        from dispatch.attention.dense_rules import DenseSpace

        with mock.patch.object(
            DenseSpace, "find", side_effect=AssertionError("walked")
        ):
            pinned = _pin(_DENSE, default, seqlen_q=2048, seqlen_k=2048)
            self.assertEqual(dispatch_attention(pinned).spec, default)

    def test_knobs_are_validated_when_the_request_is_built(self):
        with self.assertRaisesRegex(TypeError, "JSON scalar"):
            _req(tuning_knobs={"pv_priority": [2]})
        with self.assertRaisesRegex(ValueError, "twice"):
            _req(tuning_knobs=[["pv_priority", 2], ["pv_priority", 3]])
        stored_json = json.loads(json.dumps([["pv_priority", 2], ["o_store_width", 2]]))
        self.assertEqual(
            _req(tuning_knobs=stored_json).tuning_knobs,
            (("o_store_width", 2), ("pv_priority", 2)),
        )


class TestRegistryAndSweepState(unittest.TestCase):
    def test_one_candidate_instance_serves_both_registries(self):
        from dispatch.attention import ATTENTION_ROUTE_REGISTRY

        for candidate in ATTENTION_EXECUTION_REGISTRY.candidates():
            with self.subTest(candidate.name):
                self.assertIs(ATTENTION_ROUTE_REGISTRY.get(candidate.name), candidate)

    def test_interleaved_sweeps_keep_their_own_levels(self):
        prefix = "attention_" + _DENSE

        def sweep(level):
            return iter_registered_attention_combos(
                _req(), candidate_prefix=prefix, sweep_level=level, tuning_sample=4
            )

        alone = {
            level: [s.tuning_id for _c, s in sweep(level)]
            for level in ("production", "full")
        }
        a, b = sweep("production"), sweep("full")
        mixed = {"production": [], "full": []}
        for (_ca, sa), (_cb, sb) in zip(a, b, strict=False):
            mixed["production"].append(sa.tuning_id)
            mixed["full"].append(sb.tuning_id)
        n = min(len(alone["production"]), len(alone["full"]))
        for level in alone:
            self.assertEqual(mixed[level][:n], alone[level][:n])


if __name__ == "__main__":
    unittest.main()
