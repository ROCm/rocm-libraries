# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Codegen check over the tdm_store yamls: every kernel with TDMStoreInst on generates and stores D through the TDM
block."""
from pathlib import Path

import pytest

import config_harness

pytestmark = pytest.mark.unit
_DIR = Path(__file__).resolve().parents[1] / "common/gemm/gfx12"
_GROUPS = [(config, index) for config in ("tdm_store_gfx1250.yaml", "tdm_store_beta_gfx1250.yaml")
           for index, _ in config_harness.benchmark_problem_fingerprints(_DIR / config)]


@pytest.mark.parametrize("config, index", _GROUPS, ids=["%s:%u" % group for group in _GROUPS])
def test_every_tdm_store_kernel_generates(config, index):
  """A kernel whose generation fails, on an SGPR overflow for instance, is dropped from a benchmark run with a warning
  only, so the yaml tests pass without it. Every kernel of the two yamls with TDMStoreInst on generates, without the
  error code of such a drop, and stores D with tensor_store_from_lds. The kernels with it off are left out: the switch
  leaves their code as it is."""
  from tensilelite.Common.Types import DebugConfig
  from tensilelite.create_library.run import generateKernelObjectsFromSolutions
  from tensilelite.KernelWriterAssembly import KernelWriterAssembly

  assembler, iim = config_harness._toolchain_for("gfx1250")
  with config_harness._isolated_globals_with_isa(iim):
    solutions = config_harness._solutions_from_config_unguarded(_DIR / config, assembler, iim, problem_index=index)
    kernels = [kernel for kernel in generateKernelObjectsFromSolutions(solutions) if kernel["TDMStoreInst"]]
    results = config_harness._ch.emit_prepared_kernels(KernelWriterAssembly(assembler, DebugConfig()), kernels,
                                                       splitGSU=False, canonical=False)
  assert len(results) == len(kernels)
  for name, source, error in results:
    assert error == 0 and "tensor_store_from_lds" in source, (config, index, name, error)
