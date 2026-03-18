# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest
import rrperf
import rrperf.utils as utils


@pytest.mark.slow
def test_run_suite_unit():
    if utils.rocm_gfx().startswith("gfx120"):
        result, rundir = rrperf.run.run_cli(
            suite="unit_gfx120X", rundir="performance_unit"
        )
    else:
        result, rundir = rrperf.run.run_cli(suite="unit", rundir="performance_unit")

    assert result
