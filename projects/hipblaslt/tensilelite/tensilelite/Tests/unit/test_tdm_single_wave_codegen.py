# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Codegen checks for single-wave TDM kernels (HalfPLR tail, clusters)."""
from pathlib import Path
import re

import pytest

from config_harness import assert_assembles, emit_kernels_from_config

pytestmark = pytest.mark.unit
_CONFIG = Path(__file__).resolve().parents[1] / "common/gemm/gfx12/tdm_single_wave_gfx1250.yaml"


def _instructions(source):
    lines = [line.split("//", 1)[0].strip() for line in source.splitlines()]
    return [line for line in lines if line]


@pytest.fixture(scope="module")
def kernels():
    # HalfPLR x ClusterDim x StreamK at DepthU 64.
    results = emit_kernels_from_config(_CONFIG, limit=8, arch="gfx1250", canonical=False,
                                       expected_fork_count=8)
    assert len(results) == 8
    for name, _, error in results:
        assert error == 0, (name, error)
    return results


def test_kernels_assemble(kernels):
    for name, source, _ in kernels:
        assert_assembles(source, name)


def test_halfplr_keeps_global_read_increments(kernels):
    # The tail loop, the stagger removal and the StreamK tile offset read the
    # increments after the unrolled loop, so they must not be cleared in place.
    halfPLR = [(name, source) for name, source, _ in kernels if "_HPLR1_" in source]
    assert halfPLR
    for name, source in halfPLR:
        for line in _instructions(source):
            assert not re.match(r"s_c?mov_b32 s\[sgprGlobalReadIncs[AB]\]", line), (name, line)
        assert "no increment on the last iters" in source, name
