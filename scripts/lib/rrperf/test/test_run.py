import pytest
import rrperf


@pytest.mark.slow
def test_run_suite_unit():
    result, rundir = rrperf.run.run_cli(suite="unit", rundir="performance_unit")

    assert result
