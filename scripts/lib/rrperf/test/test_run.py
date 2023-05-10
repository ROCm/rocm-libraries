import pytest
import rrperf


@pytest.mark.slow
def test_run_suite_unit():
    assert rrperf.run.run_cli(suite="unit", rundir="performance_unit")
