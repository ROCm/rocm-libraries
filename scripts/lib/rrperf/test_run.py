import rrperf


def test_run_suite_unit():
    assert rrperf.run.run(suite="unit")
