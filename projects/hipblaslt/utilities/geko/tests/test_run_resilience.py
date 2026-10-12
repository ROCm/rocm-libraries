# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the run-resilience mechanisms: stall detection, shape retirement, and
the consecutive-failure abort."""

import subprocess
import sys
import threading
import time

import joblib
import pytest

from geko.concurrency.runner import Runner, Worker
from geko.concurrency.utils import wait_process_or_stop
from geko.optim.utils import (
    list_optimization_configs,
    record_shape_failure,
    shape_failure_count,
)


def _sleeper(seconds: float) -> subprocess.Popen:
    """A child that lives for `seconds` and never writes to any log."""
    return subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})"])


class TestStallDetection:
    def test_returns_false_when_process_exits_normally(self):
        proc = _sleeper(0.5)
        stalled = wait_process_or_stop(
            proc, threading.Event(), "cfg", poll_interval=0.05, stall_timeout=0
        )
        assert stalled is False
        assert proc.poll() == 0

    def test_kills_worker_whose_log_stops_advancing(self, tmp_path):
        log = tmp_path / "worker-tensilelite.log"
        log.write_text("start\n")
        proc = _sleeper(60)
        t0 = time.monotonic()
        stalled = wait_process_or_stop(
            proc, threading.Event(), "cfg",
            poll_interval=0.05, terminate_timeout=5,
            progress_path=log, stall_timeout=0.4,
        )
        assert stalled is True, "a worker with a frozen log should be reported stalled"
        assert proc.poll() is not None, "the stalled worker must be torn down"
        assert time.monotonic() - t0 < 20

    def test_advancing_log_is_not_killed(self, tmp_path):
        """A slow-but-alive worker must survive: the check is progress, not total time."""
        log = tmp_path / "worker-tensilelite.log"
        log.write_text("start\n")
        proc = _sleeper(1.5)
        stop = threading.Event()

        def touch():
            while not stop.is_set() and proc.poll() is None:
                log.write_text(f"progress {time.time()}\n")
                time.sleep(0.05)

        t = threading.Thread(target=touch, daemon=True); t.start()
        try:
            stalled = wait_process_or_stop(
                proc, threading.Event(), "cfg",
                poll_interval=0.05, progress_path=log, stall_timeout=0.5,
            )
        finally:
            stop.set(); t.join(timeout=2)
        assert stalled is False, "a worker still writing to its log must not be killed"
        assert proc.poll() == 0

    def test_stall_timeout_zero_disables_the_check(self, tmp_path):
        log = tmp_path / "worker-tensilelite.log"
        log.write_text("start\n")
        proc = _sleeper(0.6)
        stalled = wait_process_or_stop(
            proc, threading.Event(), "cfg",
            poll_interval=0.05, progress_path=log, stall_timeout=0,
        )
        assert stalled is False
        assert proc.poll() == 0

    def test_stop_event_terminates_and_is_not_reported_as_stall(self):
        proc = _sleeper(60)
        stop = threading.Event(); stop.set()
        stalled = wait_process_or_stop(
            proc, stop, "cfg", poll_interval=0.05, terminate_timeout=5
        )
        assert stalled is False, "an explicit stop is not a stall"
        assert proc.poll() is not None


class TestShapeFailureTracking:
    def test_count_starts_at_zero_and_survives_reload(self, tmp_path):
        assert shape_failure_count(tmp_path) == 0
        assert record_shape_failure(tmp_path) == 1
        assert record_shape_failure(tmp_path) == 2
        assert shape_failure_count(tmp_path) == 2

    def test_corrupt_failcount_reads_as_zero(self, tmp_path):
        (tmp_path / ".failcount").write_text("not-a-number")
        assert shape_failure_count(tmp_path) == 0

    def test_missing_build_dir_reads_as_zero(self, tmp_path):
        assert shape_failure_count(tmp_path / "does-not-exist") == 0


class TestShapeRetirement:
    @staticmethod
    def _tuning_dir(tmp_path, n=3):
        """n tuning configs. Each needs a MatrixInstruction or it is skipped for
        generating no kernels, which is a separate filter from retirement."""
        for i in range(n):
            (tmp_path / f"cfg_{i}.yaml").write_text(
                "BenchmarkProblems:\n- - {}\n  - MatrixInstruction: [16, 16, 32, 1]\n"
            )
        return tmp_path

    def test_disabled_by_default_retries_every_shape(self, tmp_path):
        d = self._tuning_dir(tmp_path)
        record_shape_failure(d / "build_cfg_1")
        record_shape_failure(d / "build_cfg_1")
        assert len(list_optimization_configs(d)) == 3, (
            "retirement must be opt-in: the default should retry every shape"
        )

    def test_retires_only_shapes_at_or_over_the_threshold(self, tmp_path):
        d = self._tuning_dir(tmp_path)
        record_shape_failure(d / "build_cfg_1")
        kept = list_optimization_configs(d, max_shape_failures=1)
        assert len(kept) == 2
        assert not any(k.endswith("cfg_1.yaml") for k in kept)

    def test_threshold_above_the_count_keeps_the_shape(self, tmp_path):
        d = self._tuning_dir(tmp_path)
        record_shape_failure(d / "build_cfg_1")
        assert len(list_optimization_configs(d, max_shape_failures=2)) == 3


class _FailingWorker(Worker):
    started = 0

    def setup(self):
        type(self).started += 1

    def run(self):
        return False

    def teardown(self):
        pass


class TestConsecutiveFailureAbort:
    @staticmethod
    def _runner(**kwargs):
        _FailingWorker.started = 0
        with joblib.parallel_config(backend="sequential"):
            return Runner(list(range(12)), _FailingWorker, devices=[0], **kwargs)

    def test_off_by_default_so_every_job_runs(self, tmp_path):
        # Benchmarking and search build their own Runner and must not stop early.
        assert self._runner()(tmp_path, silent=True) == []
        assert _FailingWorker.started == 12

    def test_stops_scheduling_at_the_threshold(self, tmp_path):
        runner = self._runner(abort_after_consecutive_failures=3)
        with pytest.raises(SystemExit):
            runner(tmp_path, silent=True)
        assert _FailingWorker.started == 3
