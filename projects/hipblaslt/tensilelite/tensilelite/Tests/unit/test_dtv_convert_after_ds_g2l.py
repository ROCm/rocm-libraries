# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DirectToVgprB with ConvertAfterDS: the global-read buffers of B hold converted elements.

An F8 B operand converted to F16 after the global read needs twice the VGPRs per buffer. If
the size is not scaled, the second DirectToVgpr buffer starts inside the first one and the two
prefetch stages overwrite each other's data (wrong results, not a codegen error).
"""
from pathlib import Path
import re

import pytest
import yaml

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit
_CONFIG = Path(__file__).resolve().parents[1] / "common/gemm/dtv.yaml"
_PROBLEM = 20  # F16 A, F8 B, F16 output, DirectToVgprB + ConvertAfterDS
_FORKS = {"GlobalReadVectorWidthA": [4], "GlobalReadVectorWidthB": [4], "VectorWidthB": [4],
          "UseSgprForGRO": [0], "Groups": [[{"MatrixInstruction": [16, 16, 16, 1, 1, 1, 4, 1, 4]}]]}
_OFFSET = re.compile(r"vgprG2LB\+(\d+)(?::vgprG2LB\+(\d+))?\]")


def _config(tmp_path):
    config = yaml.safe_load(_CONFIG.read_text())
    problem = config["BenchmarkProblems"][_PROBLEM]
    for entry in problem[1]["ForkParameters"]:
        key = next(iter(entry))
        if key in _FORKS:
            entry[key] = _FORKS[key]
    config["BenchmarkProblems"] = [problem]
    path = tmp_path / "dtv_f8b.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_second_b_buffer_starts_after_the_first(tmp_path):
    results = emit_kernels_from_config(_config(tmp_path), limit=4, arch="gfx942", canonical=False)
    assert len(results) == 3
    for name, source, error in results:
        assert error == 0, name
        assert re.search(r"\.set vgprG2LB, vgprG2LB_BASE\+0\b", source), name
        start = int(re.search(r"\.set vgprG2LB2, vgprG2LB_BASE\+(\d+)", source).group(1))
        used = [int(last or first) for first, last in _OFFSET.findall(source)]
        assert used, name
        assert max(used) < start, (name, max(used), start)
