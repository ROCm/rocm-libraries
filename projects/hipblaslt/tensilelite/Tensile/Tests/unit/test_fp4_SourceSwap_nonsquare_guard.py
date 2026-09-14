# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GPU-less guard for SourceSwap on a non-square F4 MatrixInstruction (32x16).

Drives the full benchmark flow via the ``--cpu-only`` switch on a config whose
only candidates are 32x16 WMMA + SourceSwap:[true] (see
``test_data/fp4_SourceSwap_nonsquare_gfx1250.yaml``). If the effective-vs-physical
MatrixInstM/N contract regresses, every candidate is rejected, the group yields
0 valid solutions, and BenchmarkProblems calls ``printExit`` (SystemExit) -- so
this test fails loudly instead of the drop being silently masked.

--gpu-targets is pinned to gfx1250 (the revision that has the fp4 32x16 WMMA
opcode); gfx1250v0 lacks it and would legitimately reject 32x16, so it must not
drive this guard.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from Tensile import Tensile

_CONFIG = Path(__file__).parent / "test_data" / "fp4_SourceSwap_nonsquare_gfx1250.yaml"


@pytest.fixture(autouse=True)
def _no_stdin(monkeypatch):
    """Fail loudly on any unattended stdin read instead of hanging GPU-less."""

    def _boom(*args, **kwargs):
        raise AssertionError("builtins.input() called on the --cpu-only path")

    monkeypatch.setattr("builtins.input", _boom)


def test_fp4_ss_nonsquare_survives_validation(tensile_args, tmp_path):
    output_dir = tmp_path / "output"
    # Explicit --gpu-targets AFTER *tensile_args so it last-wins over any forwarded
    # default (e.g. tox forwarding gfx1250v0): the 32x16 opcode only exists on gfx1250.
    args = [
        str(_CONFIG),
        str(output_dir),
        *tensile_args,
        "--cpu-only",
        "--gpu-targets",
        "gfx1250",
    ]

    # Regression -> 0 valid solutions -> BenchmarkProblems.printExit -> SystemExit.
    Tensile.Tensile(args)

    # Benchmark data proves the group actually ran with surviving solutions, rather
    # than the run being vacuously skipped (e.g. an arch filter dropping the group),
    # which would let this pass without ever reaching MI validation.
    benchmark_data = list((output_dir / "2_BenchmarkData").glob("*.csv"))
    assert benchmark_data, "no 2_BenchmarkData CSV; SourceSwap 32x16 never benchmarked"
