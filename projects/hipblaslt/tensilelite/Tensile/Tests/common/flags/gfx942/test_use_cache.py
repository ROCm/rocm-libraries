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
import re
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

_HAS_GFX942 = any("gfx942" in arch for arch in _get_available_archs())


def _write_config(path: str) -> None:
    """Write the test config to a yaml file."""
    with open(path, "w") as f:
        f.write(_CONFIG)


@pytest.mark.skipif(not _HAS_GFX942, reason="gfx942 GPU not available")
def test_compile_and_benchmark(tensile_args: list[str], tmp_path: Path) -> None:
    """Compile and benchmark the kernel in one go."""
    config_path = str(tmp_path / "config.yaml")
    _write_config(config_path)
    output_dir = str(tmp_path / "output")
    Tensile.Tensile([config_path, output_dir, *tensile_args])


@pytest.mark.skipif(not _HAS_GFX942, reason="gfx942 GPU not available")
def test_use_cache(tensile_args: list[str], tmp_path: Path) -> None:
    """Compile and benchmark, then run again using --use-cache and see that it doesn't crash."""
    config_path = str(tmp_path / "config.yaml")
    _write_config(config_path)
    output_dir = str(tmp_path / "output")

    # First run: compile and benchmark
    Tensile.Tensile([config_path, output_dir, *tensile_args])

    # Second run: use cache -- should not crash
    Tensile.Tensile([config_path, output_dir, "--use-cache", *tensile_args])


def _find_caches_dirs(output_dir: str) -> list[Path]:
    """Find all 'caches' directories under the output dir."""
    return list(Path(output_dir).rglob("caches"))


def _write_config_with_depth_u(path: str, depth_u: int) -> None:
    """Write config with a specific DepthU value, regardless of what the base value is."""
    config, n = re.subn(r"DepthU:\s*\[\s*\d+\s*\]", f"DepthU: [{depth_u}]", _CONFIG)
    assert n == 1, f"Expected exactly one DepthU entry in _CONFIG, found {n}"
    assert config != _CONFIG, f"DepthU substitution to {depth_u} produced no change in _CONFIG"
    with open(path, "w") as f:
        f.write(config)


@pytest.mark.skipif(not _HAS_GFX942, reason="gfx942 GPU not available")
def test_use_cache_creates_cache_dir(tensile_args: list[str], tmp_path: Path) -> None:
    """After a compile+benchmark run, a caches/ directory with one entry should exist."""
    config_path = str(tmp_path / "config.yaml")
    _write_config(config_path)
    output_dir = str(tmp_path / "output")

    Tensile.Tensile([config_path, output_dir, *tensile_args])

    caches_dirs = _find_caches_dirs(output_dir)
    assert len(caches_dirs) >= 1, "Expected at least one caches/ directory"
    # Each caches/ dir should have exactly one cache entry
    for caches_dir in caches_dirs:
        entries = [e for e in caches_dir.iterdir() if e.is_dir()]
        assert len(entries) == 1, f"Expected 1 cache entry in {caches_dir}, got {len(entries)}"


@pytest.mark.skipif(not _HAS_GFX942, reason="gfx942 GPU not available")
def test_use_cache_different_params(tensile_args: list[str], tmp_path: Path) -> None:
    """Different parameters should produce separate cache entries, each independently reusable."""
    output_dir = str(tmp_path / "output")

    # Run 1: DepthU=32
    config1 = str(tmp_path / "config1.yaml")
    _write_config(config1)
    Tensile.Tensile([config1, output_dir, *tensile_args])

    # Run 2: DepthU=64 (different params, same output dir)
    config2 = str(tmp_path / "config2.yaml")
    _write_config_with_depth_u(config2, 64)
    Tensile.Tensile([config2, output_dir, *tensile_args])

    # Verify: caches/ directory should have 2 entries
    caches_dirs = _find_caches_dirs(output_dir)
    assert len(caches_dirs) >= 1
    for caches_dir in caches_dirs:
        entries = [e for e in caches_dir.iterdir() if e.is_dir()]
        assert len(entries) == 2, f"Expected 2 cache entries in {caches_dir}, got {len(entries)}"

    # Run 3: --use-cache with original config should reuse first cache (no crash, no new entries)
    Tensile.Tensile([config1, output_dir, "--use-cache", *tensile_args])
    for caches_dir in caches_dirs:
        entries = [e for e in caches_dir.iterdir() if e.is_dir()]
        assert len(entries) == 2, f"Expected still 2 cache entries after reuse, got {len(entries)}"
