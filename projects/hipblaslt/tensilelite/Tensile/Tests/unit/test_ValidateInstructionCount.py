################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""
Negative + positive tests for InstructionCountRule
(verify_correct_number_of_instructions).

Why this rule lives outside the dataflow-graph framework
--------------------------------------------------------
Instruction count is a structural shape property: "schedule key K has exactly
len(id_map[K]) entries". A dataflow graph cannot express "exactly N nodes of
category X" without ad-hoc counting — which is what the rule already does.
The migration epic (ola/CMSMajorRefactorPlan.md) explicitly carries this rule
as a "keep structural" exception.

Failure mode this rule guards against
-------------------------------------
If `optSchedule[K]` and `id_map[K]` ever drift out of length agreement, every
downstream timeline rule silently uses only min(len_a, len_b) of the entries —
producing false-positive validation. No other rule (including the GR/LR/Pack
constraint passes) cross-checks this length. This rule is the only line of
defence against that drift.
"""

from Tensile.Components.CMSValidator import (
    verify_correct_number_of_instructions,
    ValidationContext,
    InstructionCountRule,
    ValidationConcern,
)
from Tensile.Components.CustomSchedule import ScheduleInfo


def _make_schedule_info(opt_schedule):
    """Minimal ScheduleInfo just for verify_correct_number_of_instructions.

    Only `optSchedule` is read by the rule under test — the other fields are
    plumbing the constructor demands.
    """
    return ScheduleInfo(
        numCodePaths=1,
        numMfma=None,
        optSchedule=opt_schedule,
        syncCode=[],
        nglshift=None,
        nllshift=None,
        nllZeroDscnt=None,
    )


def _make_ctx(id_map):
    """Minimal ValidationContext — only `id_map` matters for this rule."""
    return ValidationContext(kernel={}, id_map=id_map, mfma_code=[])


class TestVerifyCorrectNumberOfInstructions:
    def test_count_match_passes(self):
        sched = _make_schedule_info({"GRA": [[0, 1, 2, 3]]})
        ctx = _make_ctx({"GRA": [object()] * 4})
        ok, msg = verify_correct_number_of_instructions(sched, ctx, 0)
        assert ok, f"Expected pass, got failure: {msg}"
        assert msg == ""

    def test_count_too_few_in_id_map_fails(self):
        # optSchedule says 4 GRAs; id_map only has 3. Without this rule the
        # downstream timeline pass would silently use 3 and miss the drift.
        sched = _make_schedule_info({"GRA": [[0, 1, 2, 3]]})
        ctx = _make_ctx({"GRA": [object()] * 3})
        ok, msg = verify_correct_number_of_instructions(sched, ctx, 0)
        assert not ok
        assert "GRA" in msg
        assert "4 instructions" in msg
        assert "3 instructions are required" in msg

    def test_count_too_many_in_id_map_fails(self):
        # id_map has more entries than the schedule slots them — equally bad.
        sched = _make_schedule_info({"GRB": [[0, 1]]})
        ctx = _make_ctx({"GRB": [object()] * 3})
        ok, msg = verify_correct_number_of_instructions(sched, ctx, 0)
        assert not ok
        assert "GRB" in msg
        assert "2 instructions" in msg
        assert "3 instructions are required" in msg

    def test_first_mismatched_key_is_reported(self):
        # When several keys are present and one mismatches, the rule fails on
        # the first mismatch encountered. We assert on the failing key only —
        # iteration order over a dict in Python 3.7+ is insertion order, so
        # the test is deterministic without depending on internal ordering.
        sched = _make_schedule_info({
            "GRA": [[0, 1, 2]],
            "GRIncA": [[0, 1]],   # mismatched
            "LRA0": [[0, 1, 2, 3]],
        })
        ctx = _make_ctx({
            "GRA": [object()] * 3,
            "GRIncA": [object()] * 4,   # 4 vs 2
            "LRA0": [object()] * 4,
        })
        ok, msg = verify_correct_number_of_instructions(sched, ctx, 0)
        assert not ok
        assert "GRIncA" in msg
        assert "2 instructions" in msg
        assert "4 instructions are required" in msg

    def test_multi_code_path_uses_correct_path(self):
        # When numCodePaths > 1, schedule_get returns the per-path list. Path 0
        # has the right count; path 1 is short — only the path under test is
        # checked.
        opt = {"GRA": [[0, 1, 2], [0, 1]]}
        ctx = _make_ctx({"GRA": [object()] * 3})
        sched = ScheduleInfo(
            numCodePaths=2,
            numMfma=None,
            optSchedule=opt,
            syncCode=[],
            nglshift=None,
            nllshift=None,
            nllZeroDscnt=None,
        )
        ok0, _ = verify_correct_number_of_instructions(sched, ctx, 0)
        assert ok0, "Code path 0 has matching counts and should pass"
        ok1, msg1 = verify_correct_number_of_instructions(sched, ctx, 1)
        assert not ok1
        assert "GRA" in msg1
        assert "2 instructions" in msg1
        assert "3 instructions are required" in msg1


class TestInstructionCountRuleWiring:
    """Pin the rule's registration so a refactor that drops it from
    STRUCTURAL_RULES or changes its concern is caught immediately."""

    def test_rule_concerns_schedule_completeness(self):
        rule = InstructionCountRule()
        assert rule.concerns() == {ValidationConcern.SCHEDULE_COMPLETENESS}

    def test_rule_run_dispatches_to_function(self):
        # The rule wrapper should be a thin pass-through. Verifying both a
        # passing and a failing input exercises the dispatch path end-to-end.
        rule = InstructionCountRule()
        sched = _make_schedule_info({"GRA": [[0, 1]]})
        ok_ctx = _make_ctx({"GRA": [object()] * 2})
        bad_ctx = _make_ctx({"GRA": [object()] * 5})

        ok, _ = rule.run(sched, ok_ctx, 0)
        assert ok

        bad, msg = rule.run(sched, bad_ctx, 0)
        assert not bad
        assert "GRA" in msg

    def test_rule_is_registered_in_structural_rules(self):
        # Defence-in-depth: catch accidental deregistration.
        from Tensile.Components.CMSValidator import STRUCTURAL_RULES
        assert any(isinstance(r, InstructionCountRule) for r in STRUCTURAL_RULES)
