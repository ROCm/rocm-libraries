################################################################################
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""Solution acceptance: TDMSplit + StreamK + PAP on MX multi-wave (gfx1250).

Pins the Solution.py acceptance change that goes with the multi-wave TDMSplit
transient split-increment recompute. Previously ``assignDerivedParameters``
rejected ``TDMSplit + PrefetchAcrossPersistent`` on MX-scaled inputs; that reject
was removed because the multi-wave path no longer persists the split-increment
SGPRs. The single-wave guard (``prod(MIWaveGroup) <= 1`` -> reject) is unchanged.

Both arms are driven config-first through the CPU-only
``config_harness`` (no GPU, no compile):

* Positive: the shipped ``streamk_tdmsplit.yaml`` (StreamK=3, TDMInst=3,
  TDMSplit, PAP=1, multi-wave MX WMMA MatrixInstruction) yields at least one
  VALID solution -- the removed reject no longer fires.
* Negative control: the same config with only the MatrixInstruction dropped to
  single-wave (prod(MIWaveGroup) == 1) yields ZERO valid solutions, and the
  surviving guard fires with the wave-separated rejection reason.
"""

import importlib
import math
import os
from copy import deepcopy

import pytest

import config_harness as _ch

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "..",
    "_codegen",
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "streamk_tdmsplit.yaml",
)

# 9-item MI is [M, N, K, B, BM, WaveTile0, WaveTile1, WaveGroup0, WaveGroup1].
# The shipped config uses WaveGroup=[2,2] (prod == 4, multi-wave). Dropping it to
# WaveGroup=[1,1] (prod == 1) is the single-wave negative control.
_SINGLE_WAVE_MI = [16, 16, 128, 1, 1, 1, 1, 1, 1]

_SolMod = importlib.import_module("Tensile.SolutionStructs.Solution")


def _valid_solutions_from_dict(config):
    """Build valid Solution objects for an in-memory config dict, in-process.

    Mirrors ``config_harness.solutions_from_config`` but (a) accepts an already
    parsed/mutated config dict and (b) runs each fork permutation through
    ``_build_and_validate_solution`` directly (no ParallelMap), so a monkeypatched
    ``reject`` in this process observes the rejection reasons.
    """
    from Tensile.BenchmarkProblems import _build_and_validate_solution
    from Tensile.BenchmarkStructs import BenchmarkProcess, constructForkPermutations
    from Tensile.Common.Types import makeDebugConfig

    assembler, iim = _ch._toolchain_for(_ARCH)
    isa = next(iter(iim.keys()))
    valid = []
    with _ch._isolated_globals_with_isa(iim):
        problemTypeConfig, problemSizeGroupConfig = (
            config["BenchmarkProblems"][0][0],
            config["BenchmarkProblems"][0][1],
        )
        proc = BenchmarkProcess(problemTypeConfig, problemSizeGroupConfig, False)
        step = proc[0]
        perms = list(constructForkPermutations(step.forkParams, step.paramGroups))
        debugConfig = makeDebugConfig(config.get("GlobalParameters", {}))
        for perm in perms:
            solution = {"ProblemType": deepcopy(proc.problemType.state), "ISA": isa}
            solution.update(step.constantParams)
            solution.update(perm)
            obj = _build_and_validate_solution(
                solution, assembler, debugConfig, iim, silent=True
            )
            if obj is not None:
                valid.append(obj)
    return valid


def test_tdmsplit_pap_mx_multiwave_accepts():
    """TDMSplit + StreamK=3 + PAP + MX multi-wave (gfx1250) yields a valid solution."""
    sols = _ch.solutions_from_config(_CONFIG, arch=_ARCH)
    assert len(sols) >= 1, "expected >=1 valid solution; the removed reject fired again"
    # Confirm the valid solution is the intended now-allowed combination.
    accepted = [
        s
        for s in sols
        if s["StreamK"] == 3
        and s["TDMSplit"]
        and s["PrefetchAcrossPersistent"]
        and (s["ProblemType"]["MXBlockA"] or s["ProblemType"]["MXBlockB"])
        and math.prod(s["MIWaveGroup"]) > 1
    ]
    assert accepted, (
        "no valid TDMSplit+StreamK3+PAP MX multi-wave solution found; "
        f"got StreamK/TDMSplit/PAP/MIWaveGroup = "
        f"{[(s['StreamK'], s['TDMSplit'], s['PrefetchAcrossPersistent'], s['MIWaveGroup']) for s in sols]}"
    )


def test_tdmsplit_pap_mx_singlewave_still_rejects(monkeypatch):
    """Negative control: single-wave (prod(MIWaveGroup)==1) TDM+PAP still rejects.

    Only the MatrixInstruction is changed to single-wave; every other parameter
    matches the accepted config, so the zero-valid outcome isolates the surviving
    ``prod(MIWaveGroup) <= 1`` guard (Solution.py).
    """
    config = _ch._load_config(_CONFIG)
    changed = False
    for entry in config["BenchmarkProblems"][0][1]["ForkParameters"]:
        if "MatrixInstruction" in entry:
            entry["MatrixInstruction"] = [list(_SINGLE_WAVE_MI)]
            changed = True
    assert changed, "config no longer forks MatrixInstruction; update the test"

    reasons = []
    real_reject = _SolMod.reject

    def _capturing_reject(state, printReason=True, *args):
        reasons.append(" ".join(str(a) for a in args))
        return real_reject(state, printReason, *args)

    monkeypatch.setattr(_SolMod, "reject", _capturing_reject)
    sols = _valid_solutions_from_dict(config)

    assert len(sols) == 0, "single-wave TDM+PAP unexpectedly produced a valid solution"
    assert any("wave-separated mode" in r for r in reasons), (
        "single-wave rejection did not fire the wave-separated (prod(MIWaveGroup) > 1) "
        f"guard; captured reasons: {[r for r in reasons if r]}"
    )
