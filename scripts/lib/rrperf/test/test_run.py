import pytest
import rrperf


@pytest.mark.slow
def test_run_suite_unit():
    assert rrperf.run.run(suite="unit", rundir="performance_unit")
