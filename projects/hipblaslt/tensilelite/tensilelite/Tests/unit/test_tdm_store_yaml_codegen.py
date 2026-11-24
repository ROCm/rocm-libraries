# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Codegen check over the tdm_store yamls: every configuration they list is a TDMStoreInst solution of its own, and
every kernel generates and stores D through the TDM block."""
from pathlib import Path

import pytest

import config_harness

pytestmark = pytest.mark.unit
_DIR = Path(__file__).resolve().parents[1] / "common/gemm/gfx12"
_GROUPS = [(config, index) for config in ("tdm_store_gfx1250.yaml", "tdm_store_beta_gfx1250.yaml")
           for index, _ in config_harness.benchmark_problem_fingerprints(_DIR / config)]


@pytest.mark.parametrize("config, index", _GROUPS, ids=["%s:%u" % group for group in _GROUPS])
def test_every_tdm_store_configuration_generates(config, index):
  """Each group sets TDMStoreInst on and lists only the configurations it tests. A benchmark run drops a configuration
  the solution checks reject, merges configurations that give one solution, and drops a kernel whose generation fails,
  on an SGPR overflow for instance, with a warning only, so the yaml tests pass without them. Here every configuration
  of the group gives a TDMStoreInst solution that no other configuration gives, and every kernel generates without the
  error code of such a drop and stores D with tensor_store_from_lds."""
  from tensilelite.BenchmarkProblems import _generateForkedSolutions
  from tensilelite.BenchmarkStructs import BenchmarkProcess, constructForkPermutations
  from tensilelite.Common.Types import DebugConfig, makeDebugConfig
  from tensilelite.create_library.run import generateKernelObjectsFromSolutions
  from tensilelite.KernelWriterAssembly import KernelWriterAssembly

  assembler, iim = config_harness._toolchain_for("gfx1250")
  path = _DIR / config
  yamlConfig = config_harness._load_config(path)
  problem = config_harness._select_benchmark_problem(yamlConfig["BenchmarkProblems"], path, index, None)
  debugConfig = makeDebugConfig(yamlConfig.get("GlobalParameters", {}))
  with config_harness._isolated_globals_with_isa(iim):
    process = BenchmarkProcess(problem[0], problem[1], False)
    step = process[0]
    solutions = []
    for fork in constructForkPermutations(step.forkParams, step.paramGroups):
      forked = _generateForkedSolutions(process.problemType, step.constantParams, [fork], assembler, debugConfig, iim)
      assert len(forked) == 1 and forked[0]["TDMStoreInst"], (config, index, fork)
      solutions += forked
    assert len(set(solutions)) == len(solutions), (config, index, [str(solution) for solution in solutions])
    kernels = generateKernelObjectsFromSolutions(solutions)
    results = config_harness._ch.emit_prepared_kernels(KernelWriterAssembly(assembler, DebugConfig()), kernels,
                                                       splitGSU=False, canonical=False)
  assert kernels and len(results) == len(kernels)
  for name, source, error in results:
    assert error == 0 and "tensor_store_from_lds" in source, (config, index, name, error)
