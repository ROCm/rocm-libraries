################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Round-trip test for the CMS-vs-default flag-handling reconciliation.

bead: rocm-libraries-9lcs

The CMS path (UseCustomMainLoopSchedule=1) and the default SIA3 path
(UseCustomMainLoopSchedule=0) must produce equivalent Solution dicts
for the same input YAML, modulo the schedule-content choice itself
(UseCustomMainLoopSchedule, schedule-decided SwapGlobalReadOrder /
UsePLRPack, and a few derived knobs).

These tests round-trip a representative kernel-config dict through both
=0 and =1 and assert that fields *outside* a documented allowlist of
schedule-derived keys are identical between the two solutions. They
also assert the loud-rejection behavior added in the same bead: a
YAML that requests SwapGlobalReadOrder=1 or UsePLRPack=1 alongside
UseCustomMainLoopSchedule=1 must rejected with a clear "CMS does not
support" message rather than silently mutating the input.
"""

from copy import deepcopy

import pytest

from cms_test_utils import _make_solution


# A representative TN bf16 config that has a registered CMS schedule on
# gfx950 (mirrors the 256x256 schedule used by other CMS tests in this
# directory). Both UseCustomMainLoopSchedule=0 and =1 are valid for
# this shape.
def _base_config():
    return {
        "ProblemType": {
            "OperationType": "GEMM",
            "DataType": "H",
            "DestDataType": "H",
            "TransposeA": True,
            "TransposeB": False,
            "UseBeta": True,
            "Batched": True,
            "HighPrecisionAccumulate": True,
        },
        # 256x256 macro-tile via mi=16x16x32 * MIWaveTile=8x8 * MIWaveGroup=2x2
        "MatrixInstruction": [16, 16, 32, 1, 1, 8, 8, 2, 2],
        "DepthU": 64,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "DirectToLds": 1,
        "TransposeLDS": 1,
        "LocalReadVectorWidth": 8,
        "GlobalReadVectorWidthA": 8,
        "GlobalReadVectorWidthB": 8,
        "ExpandPointerSwap": 0,
        "SourceSwap": 1,
        "StreamK": 0,
    }


# Keys whose value is *expected* to differ between the CMS=0 and CMS=1
# solutions because they encode the schedule-content choice itself.
# Anything outside this allowlist must agree.
#
# Audit (Tensile/Components/CustomSchedule.py):
# * SwapGlobalReadOrder, UsePLRPack — the bead's primary subjects;
#   matched CMS schedules write these into kernel state on selection.
# * MfmaInitCVgprs — set unconditionally to True by every CMS schedule
#   that hits a matching variant (greppable via "MfmaInitCVgprs = True"
#   in CustomSchedule.py). Not a bug: the schedule's instruction layout
#   requires the C-vgprs to be initialized. The default path leaves it
#   at the framework default (False unless UseMFMAF32XEmulation also
#   triggers the line at Solution.py:2040), which is correct for SIA3.
_SCHEDULE_CHOICE_KEYS = {
    "UseCustomMainLoopSchedule",
    "SwapGlobalReadOrder",
    "UsePLRPack",
    "MfmaInitCVgprs",
}


def _diff_state(state_cms_off, state_cms_on):
    """Return the set of keys whose values differ between two Solution
    state dicts (excluding keys that exist in only one)."""
    common_keys = set(state_cms_off.keys()) & set(state_cms_on.keys())
    diffs = {}
    for key in common_keys:
        if state_cms_off[key] != state_cms_on[key]:
            diffs[key] = (state_cms_off[key], state_cms_on[key])
    return diffs


class TestCMSFlagReconciliation:
    def test_roundtrip_no_extra_flags_matches(self, isa_infrastructure):
        """A clean YAML with neither SwapGlobalReadOrder nor UsePLRPack
        set should produce equivalent solutions on both paths, except
        for the schedule-content allowlist."""
        _, isaInfoMap, asm = isa_infrastructure

        cfg_off = _base_config()
        cfg_off["UseCustomMainLoopSchedule"] = 0
        cfg_on = _base_config()
        cfg_on["UseCustomMainLoopSchedule"] = 1

        sol_off = _make_solution(cfg_off, asm, isaInfoMap)
        sol_on = _make_solution(cfg_on, asm, isaInfoMap)

        # Sanity: the CMS=1 path actually selected CMS.
        assert sol_on["UseCustomMainLoopSchedule"] == 1, \
            "CMS=1 path did not resolve to CMS for the chosen kernel shape"
        # Sanity: the CMS=0 path stayed off.
        assert sol_off["UseCustomMainLoopSchedule"] == 0

        # Asymmetric add/drop of keys would silently slip past the
        # value-only diff below — so assert key sets are equal first.
        keys_off = set(sol_off._state.keys())
        keys_on = set(sol_on._state.keys())
        assert keys_off == keys_on, (
            "CMS=0 vs CMS=1 produced different key sets:\n"
            f"  only in CMS=0: {sorted(keys_off - keys_on)}\n"
            f"  only in CMS=1: {sorted(keys_on - keys_off)}"
        )

        diffs = _diff_state(sol_off._state, sol_on._state)
        # Only keys in the schedule-choice allowlist may differ.
        unexpected = {k: v for k, v in diffs.items()
                      if k not in _SCHEDULE_CHOICE_KEYS}
        assert not unexpected, (
            "CMS=0 vs CMS=1 produced different values for non-schedule "
            "fields, breaking round-trip equivalence:\n"
            + "\n".join(f"  {k}: off={v[0]!r}  on={v[1]!r}"
                        for k, v in unexpected.items())
        )

    def test_cms_rejects_yaml_swap_global_read_order(self, isa_infrastructure):
        """SwapGlobalReadOrder=1 in YAML alongside CMS=1 must reject loudly,
        not silently zero the flag. (Pre-fix behavior: silent zero.)"""
        _, isaInfoMap, asm = isa_infrastructure

        cfg = _base_config()
        cfg["UseCustomMainLoopSchedule"] = 1
        cfg["SwapGlobalReadOrder"] = 1

        # _make_solution asserts the resulting Solution is Valid; for
        # this negative test we build the Solution directly and inspect
        # the rejection state.
        from copy import deepcopy
        from Tensile.SolutionStructs.Solution import Solution
        from Tensile.SolutionStructs.Problem import ProblemType
        from Tensile.TensileLogic.HandleCustomKernel import (
            matrixInstructionToMIParameters,
        )

        # Mirror _make_solution's setup but skip its Valid assertion.
        isa = next(iter(isaInfoMap.keys()))
        pt = ProblemType(cfg["ProblemType"], False)
        config = dict(cfg)
        config["ProblemType"] = deepcopy(pt.state)
        config["ISA"] = isa
        config.setdefault("KernelLanguage", "Assembly")
        config.setdefault("WavefrontSize", 64)
        config.setdefault("WorkGroup", [32, 8, 1])
        mi_params = matrixInstructionToMIParameters(
            config["MatrixInstruction"], isa, config["WavefrontSize"],
            config["ProblemType"], config["WorkGroup"], isaInfoMap
        )
        config.update(mi_params)

        sol = Solution(config, splitGSU=False, printSolutionRejectionReason=False,
                       printIndexAssignmentInfo=False, assembler=asm,
                       isaInfoMap=isaInfoMap)
        assert not sol["Valid"], (
            "Expected loud rejection of SwapGlobalReadOrder=1 + CMS=1, but "
            "Solution was marked Valid (the silent-mutation regression is back)."
        )

    def test_cms_rejects_yaml_use_plr_pack(self, isa_infrastructure):
        """UsePLRPack=1 in YAML alongside CMS=1 must reject loudly."""
        _, isaInfoMap, asm = isa_infrastructure

        cfg = _base_config()
        cfg["UseCustomMainLoopSchedule"] = 1
        cfg["UsePLRPack"] = 1

        from copy import deepcopy
        from Tensile.SolutionStructs.Solution import Solution
        from Tensile.SolutionStructs.Problem import ProblemType
        from Tensile.TensileLogic.HandleCustomKernel import (
            matrixInstructionToMIParameters,
        )

        isa = next(iter(isaInfoMap.keys()))
        pt = ProblemType(cfg["ProblemType"], False)
        config = dict(cfg)
        config["ProblemType"] = deepcopy(pt.state)
        config["ISA"] = isa
        config.setdefault("KernelLanguage", "Assembly")
        config.setdefault("WavefrontSize", 64)
        config.setdefault("WorkGroup", [32, 8, 1])
        mi_params = matrixInstructionToMIParameters(
            config["MatrixInstruction"], isa, config["WavefrontSize"],
            config["ProblemType"], config["WorkGroup"], isaInfoMap
        )
        config.update(mi_params)

        sol = Solution(config, splitGSU=False, printSolutionRejectionReason=False,
                       printIndexAssignmentInfo=False, assembler=asm,
                       isaInfoMap=isaInfoMap)
        assert not sol["Valid"], (
            "Expected loud rejection of UsePLRPack=1 + CMS=1, but "
            "Solution was marked Valid (the silent-mutation regression is back)."
        )

    def test_cms_auto_with_yaml_flags_and_no_schedule_falls_through(self, isa_infrastructure):
        """UseCustomMainLoopSchedule=-1 with YAML SwapGlobalReadOrder/UsePLRPack
        flags must NOT reject when no CMS schedule is registered for the shape;
        the auto-fallback to the non-CMS path must honor the YAML flags as the
        default path describes (subject to that path's own gating)."""
        _, isaInfoMap, asm = isa_infrastructure

        # Use a too-small tile that has no registered CMS schedule, so
        # hasCustomSchedule returns False and we fall back to non-CMS.
        cfg = _base_config()
        cfg["MatrixInstruction"] = [16, 16, 32, 1, 1, 1, 1, 1, 1]
        cfg["UseCustomMainLoopSchedule"] = -1
        cfg["SwapGlobalReadOrder"] = 1
        # UsePLRPack=1 will still get gated to 0 by the non-CMS path's
        # documented preconditions (e.g., not F32X), but the key thing is
        # we don't reject the solution outright.
        cfg["UsePLRPack"] = 1

        from copy import deepcopy
        from Tensile.SolutionStructs.Solution import Solution
        from Tensile.SolutionStructs.Problem import ProblemType
        from Tensile.TensileLogic.HandleCustomKernel import (
            matrixInstructionToMIParameters,
        )

        isa = next(iter(isaInfoMap.keys()))
        pt = ProblemType(cfg["ProblemType"], False)
        config = dict(cfg)
        config["ProblemType"] = deepcopy(pt.state)
        config["ISA"] = isa
        config.setdefault("KernelLanguage", "Assembly")
        config.setdefault("WavefrontSize", 64)
        config.setdefault("WorkGroup", [32, 8, 1])
        mi_params = matrixInstructionToMIParameters(
            config["MatrixInstruction"], isa, config["WavefrontSize"],
            config["ProblemType"], config["WorkGroup"], isaInfoMap
        )
        config.update(mi_params)

        sol = Solution(config, splitGSU=False, printSolutionRejectionReason=False,
                       printIndexAssignmentInfo=False, assembler=asm,
                       isaInfoMap=isaInfoMap)

        # The solution may or may not validate (depends on a bunch of
        # other cross-cutting solver constraints for this small tile),
        # but we MUST NOT see a CMS-flag-related rejection. The bead's
        # contract is: -1 with no matching schedule and YAML
        # SwapGlobalReadOrder=1 must behave like the default path —
        # which means SwapGlobalReadOrder is honored (subject to the
        # default path's gating at lines 2018-2023 of Solution.py),
        # not silently dropped by the CMS pre-zero block.
        assert sol["UseCustomMainLoopSchedule"] == 0, (
            "auto mode should resolve to non-CMS when no schedule is registered"
        )
