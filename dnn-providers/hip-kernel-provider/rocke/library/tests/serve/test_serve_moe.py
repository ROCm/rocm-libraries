# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``rocke-serve`` tests for the fused-MoE operator.

The MoE op shares the envelope with attention but carries a different entry
type, so the risks worth testing are the seams: that ``op`` picks the right
entry parser, that an attention-shaped payload sent as ``moe`` is rejected
rather than half-parsed, and that a shape outside the tuned cohort comes back
as a declined plan with a reason instead of an exception.

The measured lane is not exercised here -- it spawns the benchmark harnesses
and needs a GPU. What is exercised is that the runner's spec translation still
describes the kernel the dispatcher chose, which is the part that can silently
drift.
"""

from __future__ import annotations

import unittest

from serve.protocol import (
    MOE_PROBLEM_FIELDS,
    REQUEST_SCHEMA,
    SERVE_OPS,
    MoeShapeEntry,
    ProtocolError,
    ServeRequest,
)


def _problem(**overrides):
    p = {
        "tokens": 32,
        "experts": 128,
        "topk": 8,
        "hidden": 2048,
        "intermediate": 768,
        "group_k": 128,
        "activation": "silu",
        "dtype": "fp8e4m3",
    }
    p.update(overrides)
    return p


def _entry(**overrides):
    problem = _problem(**overrides.pop("problem", {}))
    entry = {
        "moe_request": dict(problem),
        "problem": problem,
        "call_count": 48,
        "active_experts": 111,
        "shape_provenance": "traced:qwen3-30b-a3b-decode",
    }
    entry.update(overrides)
    return entry


def _request(**overrides):
    raw = {
        "schema": REQUEST_SCHEMA,
        "op": "moe",
        "arch": "gfx950",
        "requests": [_entry()],
    }
    raw.update(overrides)
    return raw


class TestEnvelope(unittest.TestCase):
    def test_moe_is_a_served_op(self):
        self.assertIn("moe", SERVE_OPS)
        self.assertIn("attention", SERVE_OPS)

    def test_op_selects_the_moe_entry_type(self):
        req = ServeRequest.from_dict(_request())
        self.assertEqual(req.op, "moe")
        self.assertIsInstance(req.entries[0], MoeShapeEntry)

    def test_op_still_defaults_to_attention(self):
        raw = _request()
        raw.pop("op")
        # An attention entry under the default op: the point is that omitting
        # `op` keeps the old meaning rather than becoming ambiguous.
        raw["requests"] = [
            {
                "attention_request": {"batch": 1},
                "problem": {
                    "num_seqs": 1,
                    "num_query_heads": 8,
                    "num_kv_heads": 1,
                    "head_size": 128,
                    "block_size": 16,
                    "max_seqlen_q": 1,
                    "max_seqlen_k": 128,
                    "dtype": "bf16",
                },
            }
        ]
        self.assertEqual(ServeRequest.from_dict(raw).op, "attention")

    def test_unknown_op_is_refused_with_the_served_list(self):
        with self.assertRaises(ProtocolError) as ctx:
            ServeRequest.from_dict(_request(op="conv"))
        self.assertIn("moe", str(ctx.exception))

    def test_attention_payload_sent_as_moe_is_refused(self):
        raw = _request()
        raw["requests"] = [{"attention_request": {"batch": 1}, "problem": {}}]
        with self.assertRaises(ProtocolError):
            ServeRequest.from_dict(raw)


class TestMoeEntry(unittest.TestCase):
    def test_required_fields_are_named_when_missing(self):
        raw = _request()
        raw["requests"][0]["problem"].pop("hidden")
        with self.assertRaises(ProtocolError) as ctx:
            ServeRequest.from_dict(raw)
        self.assertIn("hidden", str(ctx.exception))

    def test_unknown_problem_fields_are_dropped_not_rejected(self):
        # Forward compatibility: an upstream that learns a new field must not
        # break a rocKE that has not.
        raw = _request()
        raw["requests"][0]["problem"]["future_knob"] = 7
        entry = ServeRequest.from_dict(raw).entries[0]
        self.assertNotIn("future_knob", entry.problem)
        self.assertEqual(set(entry.problem) - set(MOE_PROBLEM_FIELDS), set())

    def test_signature_identifies_the_layer(self):
        entry = ServeRequest.from_dict(_request()).entries[0]
        self.assertEqual(
            entry.signature, "e128k8_h2048_i768_t32_g128_silu_fp8e4m3"
        )

    def test_active_experts_is_observational_and_optional(self):
        raw = _request()
        raw["requests"][0].pop("active_experts")
        self.assertEqual(ServeRequest.from_dict(raw).entries[0].active_experts, 0)


class TestPlanning(unittest.TestCase):
    def test_tuned_cohort_is_servable(self):
        from serve.planner import PLANNERS

        req = ServeRequest.from_dict(_request())
        plan = PLANNERS[req.op](req.entries, arch=req.arch)[0]
        self.assertTrue(plan["servable"])
        # T=32 leaves the fused band. The fused kernel wins only up to ~8
        # tokens; from 9 up the cooperative split measures faster (412.4 vs
        # 429.8 us at this shape), so the band edge was moved down to 8 and
        # this fixture now serves the split.
        self.assertEqual(plan["candidate"], "moe_split_coop_tm16")
        self.assertEqual(plan["weight_layout"], "swizzled")
        self.assertEqual(plan["grid"], [6, 144, 1])
        self.assertEqual(plan["block"], [64, 1, 1])

    def test_out_of_cohort_falls_back_to_the_untuned_candidate(self):
        """Out of cohort is served, but by the kernel that claims no tuning.

        This is the visible consequence of serving from the platform registry
        rather than a serve-local one: the generic fp8 mega-kernel is a real,
        correct candidate at any shape, so a planner backed by that registry
        has an answer here where the old serve-local registry -- which held
        only the banded cohort -- had none.

        What the plan must not do is present it as the tuned answer, so the
        assertion is on *which* candidate: ``mega_fp8`` is the untuned generic,
        and a band id appearing here would mean a crossover measured at
        hidden=2048 had been applied to a shape that never saw it.
        """
        from serve.planner import PLANNERS

        raw = _request()
        raw["requests"][0]["moe_request"]["hidden"] = 4096
        raw["requests"][0]["problem"]["hidden"] = 4096
        req = ServeRequest.from_dict(raw)
        plan = PLANNERS[req.op](req.entries, arch=req.arch)[0]
        self.assertTrue(plan["servable"])
        self.assertEqual(plan["spec_id"], "mega_fp8")

    def test_a_shape_the_family_cannot_run_is_still_declined(self):
        """Declining is still the answer when no candidate admits the request.

        The fallback above widens what is servable; it must not make the
        planner claim everything. A non-CDNA arch has no candidate at all, and
        the plan has to carry the registry's reason rather than raise.
        """
        from serve.planner import PLANNERS

        req = ServeRequest.from_dict(_request(arch="gfx1100"))
        plan = PLANNERS[req.op](req.entries, arch=req.arch)[0]
        self.assertFalse(plan["servable"])
        self.assertTrue(plan["reason"])

    def test_envelope_arch_overrides_the_entry(self):
        from serve.planner import build_moe_request

        raw = _request()
        raw["requests"][0]["moe_request"]["arch"] = "gfx942"
        entry = ServeRequest.from_dict(raw).entries[0]
        self.assertEqual(build_moe_request(entry, arch="gfx950").arch, "gfx950")

    def test_planners_are_registered_for_every_served_op(self):
        from serve.planner import PLANNERS

        self.assertEqual(set(PLANNERS), set(SERVE_OPS))


class TestRunnerSpecTranslation(unittest.TestCase):
    """The measured lane must run the kernel dispatch picked, not a lookalike."""

    def _plan(self):
        from serve.planner import PLANNERS

        req = ServeRequest.from_dict(_request())
        return PLANNERS[req.op](req.entries, arch=req.arch)[0]

    def test_spec_json_matches_the_dispatched_spec(self):
        """Every knob the benchmark will run is the one that was dispatched.

        Asserted against the plan's own spec rather than against the tuning
        record, because the spec is what dispatch actually chose: a band sets
        knobs the record's base does not carry, and comparing to the base would
        pass while the measured kernel differed from the planned one.
        """
        from serve.runner import _spec_json_for

        plan = self._plan()
        fields = _spec_json_for(plan)
        shared = set(fields) & set(plan["spec"])
        self.assertIn("tile_m", shared)
        for key in sorted(shared):
            self.assertEqual(
                fields[key], plan["spec"][key], f"{key} drifted from the dispatcher"
            )

    def test_spec_json_tracks_the_weight_layout(self):
        from serve.runner import _spec_json_for

        plan = self._plan()
        plan["spec"]["swizzle_gu"] = False
        plan["spec"]["swizzle_down"] = False
        fields = _spec_json_for(plan)
        self.assertFalse(fields["swizzle_gu"])
        self.assertFalse(fields["swizzle_down"])

    def test_shape_lookup_covers_the_claimed_cohort(self):
        from serve.runner import moe_shape_name

        self.assertEqual(moe_shape_name(_problem()), "qwen3")

    def test_shape_lookup_refuses_what_it_cannot_generate(self):
        from serve.runner import moe_shape_name

        with self.assertRaises(ValueError):
            moe_shape_name(_problem(hidden=4096))

    def test_routing_dir_embeds_shape_and_seed(self):
        from serve.runner import _moe_routing_dir

        self.assertTrue(str(_moe_routing_dir("qwen3", 128)).endswith("qwen3_e128_seed11939"))


if __name__ == "__main__":
    unittest.main()
