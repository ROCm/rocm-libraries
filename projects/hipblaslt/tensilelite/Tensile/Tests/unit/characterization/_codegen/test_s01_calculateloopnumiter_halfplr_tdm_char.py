################################################################################
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""Characterize rejection of the disabled TDMSplit configuration.

The designed config requests the single-wave HalfPLR and TDMSplit combination.
``Solution.assignDerivedParameters`` rejects every solution because TDMSplit is
currently disabled, so the config cannot reach the corresponding
``KernelWriterAssembly.calculateLoopNumIter`` code. These tests record that
boundary without claiming coverage of unreachable emitter lines.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "s01_calculateloopnumiter_halfplr_tdm.yaml",
)


def test_s01_calculateloopnumiter_halfplr_tdm_is_rejected():
    """The disabled combination produces no kernel."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert results == []


def test_s01_calculateloopnumiter_halfplr_tdm_reports_reason(capsys, monkeypatch):
    """The rejection names TDMSplit instead of silently dropping the config."""
    import Tensile.BenchmarkProblems as benchmark_problems

    def serial_map(function, objects, *_args, **_kwargs):
        return [function(*args) for args in objects]

    monkeypatch.setattr(benchmark_problems, "ParallelMap2", serial_map)
    emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert "reject: TDMSplit is currently disabled" in capsys.readouterr().out
