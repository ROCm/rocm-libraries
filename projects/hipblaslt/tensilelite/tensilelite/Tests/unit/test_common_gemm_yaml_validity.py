# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validate shipped GEMM inputs, except the measured upstream-base residuals."""
import copy
from pathlib import Path

import pytest
import yaml

from config_harness import _isolated_globals_with_isa, _toolchain_for
from tensilelite.BenchmarkStructs import BenchmarkProcess, constructForkPermutations
from tensilelite.SolutionStructs.Solution import Solution

from tensilelite.SolutionStructs.Validators.MatrixInstruction import (
    matrixInstructionToMIParameters, validateMIParameters,
)

pytestmark = pytest.mark.unit
GEMM = Path(__file__).resolve().parents[1] / "common" / "gemm"
# These inputs also fail this same CPU validator on upstream base 98737a5c5f4.
# Exclude only those measured upstream residuals, not branch-introduced failures;
# every other existing or newly added YAML remains in the derived-parameter sweep.
# The inventory assertion reads an independent record of that base validation.
UPSTREAM_BASE_FAILURES = frozenset({
    "gfx12/persistent_advanced.yaml", "swizzleA.yaml", "swizzleB.yaml",
})
ALL_CONFIGS = sorted(GEMM.rglob("*.yaml"))
CONFIGS = [path for path in ALL_CONFIGS
           if path.relative_to(GEMM).as_posix() not in UPSTREAM_BASE_FAILURES]

# Pass: the validation scope excludes exactly the measured upstream failures.
# Break: hide another YAML failure by enlarging the exclusion set or filter.
def test_base_failure_exclusions_match_measured_upstream_residuals():
    excluded = set(ALL_CONFIGS) - set(CONFIGS)
    measurement = yaml.safe_load((Path(__file__).parent / "test_data" /
                                  "common_gemm_base_failures.json").read_text())
    measured_failures = {record["path"] for record in measurement["failed"]}
    assert {path.relative_to(GEMM).as_posix() for path in excluded} == measured_failures



# Pass: use general activation plus runtime ActivationArgs in ordinary configs.
# Break: put a keyless single ActivationType back in any shipped GEMM input.
@pytest.mark.slow  # about 17 minutes serially over every GEMM test config
@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.relative_to(GEMM).as_posix())
def test_common_gemm_yaml_derived_parameters_have_no_fatal_input_errors(path):
    config = yaml.safe_load(path.read_text())
    arch = config.get("GlobalParameters", {}).get("Architecture")
    if not arch:
        arch = "gfx1250" if "gfx12" in path.parts or "gfx1250" in path.parts else (
            "gfx950" if "gfx950" in path.parts else "gfx1100" if "gfx11" in path.parts else "gfx942")
    assembler, isa_info = _toolchain_for(arch)
    failures = []
    checked = 0
    with _isolated_globals_with_isa(isa_info):
        for problem_index, problem in enumerate(config["BenchmarkProblems"]):
            problem_type, *size_groups = problem
            for group_index, size_group in enumerate(size_groups):
                process = BenchmarkProcess(problem_type, size_group, False)
                for step_index in range(len(process)):
                    step = process[step_index]
                    permutations = constructForkPermutations(step.forkParams, step.paramGroups) or [{}]
                    for permutation_index, permutation in enumerate(permutations):
                        checked += 1
                        candidate = {"ProblemType": copy.deepcopy(process.problemType.state),
                                     "ISA": next(iter(isa_info))}
                        candidate.update(copy.deepcopy(step.constantParams))
                        candidate.update(copy.deepcopy(permutation))
                        candidate["InternalSupportParams"] = copy.deepcopy(step.internalSupportParams)
                        isa = candidate["ISA"]
                        if candidate["WavefrontSize"] == -1:
                            candidate["WavefrontSize"] = 32 if isa_info[isa].archCaps["HasWave32"] else 64
                        matrix = candidate["MatrixInstruction"]
                        if len(matrix) == 9:
                            candidate.update(matrixInstructionToMIParameters(
                                matrix, isa, candidate["WavefrontSize"], process.problemType,
                                candidate["WorkGroup"], isa_info, candidate.get("SourceSwap", False)))
                        elif not matrix:
                            candidate["EnableMatrixInstruction"] = False
                        if not validateMIParameters(candidate, isa_info):
                            continue
                        try:
                            # Unlike the benchmark driver, do not swallow derivation exceptions.
                            # Ordinary Valid=False geometry/architecture pruning is not a fatal input error.
                            Solution(candidate, False, False, False, assembler, isa_info)
                        except RuntimeError as error:
                            failures.append(f"problem {problem_index}, group {group_index}, "
                                            f"step {step_index}, solution {permutation_index}: {error}")
    assert checked, "GEMM input had no solution candidates to validate"
    assert not failures, f"{path.relative_to(GEMM)} has fatal derived-parameter errors:\n" + "\n".join(failures)
