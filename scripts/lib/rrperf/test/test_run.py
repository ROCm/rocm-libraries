import rrperf


def test_run_suite_unit():
    assert rrperf.run.run(suite="unit", working_dir="performance_unit")
