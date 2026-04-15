################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

import os
import pytest
import subprocess
from pathlib import Path

from Tensile import Tensile

# The yaml config is defined inline (rather than as a separate .yaml file) to
# prevent test_config.py's findConfigs() from picking it up as a standalone
# parametrized test.  This config is only meant to be used by the tests below.
#
# Trimmed from Tensile/Tests/common/gemm/fp16_tn.yaml to a single solution
# (one MatrixInstruction) and a single benchmark problem size.
_CONFIG = """\
GlobalParameters:
  ISA: [[9,5,0]]
  MinimumRequiredVersion: 5.0.0
  NumElementsToValidate: 0
  DataInitTypeBeta: 0
  DataInitTypeAlpha: 1
  NewClient: 2
  Device: 0

BenchmarkProblems:
  -
    - # ProblemType
      OperationType: GEMM
      DataType: h
      DestDataType: h
      ComputeDataType: s
      HighPrecisionAccumulate: True
      TransposeA: 1
      TransposeB: 0
      UseBeta: True
      Batched: True
    - # BenchmarkProblemSizeGroup
      InitialSolutionParameters:
      BenchmarkCommonParameters:
        - KernelLanguage: ["Assembly"]
      ForkParameters:
        - MatrixInstruction:
          - [16, 16, 16, 1, 1, 1, 1, 4, 1]
        - PrefetchGlobalRead: [2]
        - PrefetchLocalRead: [1]
        - ClusterLocalRead: [1]
        - DepthU: [32]
        - LocalReadVectorWidth: [8]
        - ScheduleIterAlg: [3]
        - ExpandPointerSwap: [0]
        - TransposeLDS: [1]
        - LdsBlockSizePerPadA: [-1]
        - LdsBlockSizePerPadB: [-1]
        - LdsPadA: [-1]
        - LdsPadB: [-1]
        - 1LDSBuffer: [-1]
        - GlobalSplitU: [1]
        - SourceSwap: [0]
      BenchmarkJoinParameters:
      BenchmarkFinalParameters:
        - ProblemSizes:
          - Exact: [256, 256, 1, 256]
"""

def _get_available_archs() -> list[str]:
    """Get list of available GPU architectures via rocm_agent_enumerator."""
    rocmpath = os.environ.get("TENSILE_ROCM_PATH",
                              os.environ.get("ROCM_PATH", "/opt/rocm"))
    rocm_agent_enum = os.path.join(rocmpath, "bin/rocm_agent_enumerator")
    try:
        output = subprocess.check_output([rocm_agent_enum, "-t", "GPU"])
        return [line.strip() for line in output.decode().splitlines()
                if line.strip() and "gfx000" not in line]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

_HAS_GFX942 = any("gfx950" in arch for arch in _get_available_archs())


def _write_config(path: str) -> None:
    """Write the test config to a yaml file."""
    with open(path, "w") as f:
        f.write(_CONFIG)


def test_compile(tensile_args: list[str], tmp_path: Path) -> None:
    """
    Compile the kernel. This can run on any machine.
    """
    config_path = str(tmp_path / "config.yaml")
    _write_config(config_path)
    output_dir = str(tmp_path / "output")
    Tensile.Tensile([config_path, output_dir, "--build-only", *tensile_args])


@pytest.mark.skipif(not _HAS_GFX942, reason="gfx942 GPU not available")
def test_use_cache(tensile_args: list[str], tmp_path: Path) -> None:
    """
    Compile with --build-only, then run again using --use-cache and see that it doesn't crash.
    """
    config_path = str(tmp_path / "config.yaml")
    _write_config(config_path)
    output_dir = str(tmp_path / "output")

    # First run: compile only
    Tensile.Tensile([config_path, output_dir, "--build-only", *tensile_args])

    # Second run: use cache
    Tensile.Tensile([config_path, output_dir, "--use-cache", *tensile_args])
