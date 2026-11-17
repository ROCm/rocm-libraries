# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from queue import Queue
from threading import Event

import pytest

from geko.concurrency.runner import Runner, Worker


class _GoodWorker(Worker[int]):
    def setup(self) -> None:
        return None

    def run(self) -> bool:
        return True

    def teardown(self) -> None:
        return None


class _SelectiveWorker(Worker[int]):
    def setup(self) -> None:
        return None

    def run(self) -> bool:
        return self.item % 2 == 0

    def teardown(self) -> None:
        return None


class _SetupErrorWorker(Worker[int]):
    def setup(self) -> None:
        raise RuntimeError("boom")

    def run(self) -> bool:
        return True

    def teardown(self) -> None:
        return None


def test_worker_call_handles_setup_exception_and_reports_failure() -> None:
    q: Queue[tuple[bool, int]] = Queue()
    w = _SetupErrorWorker(item=5, device=0, slot_id=0, stop_event=Event(), output_queue=q)
    w()
    success, item = q.get(timeout=1)
    assert success is False
    assert item == 5


def test_runner_returns_completed_items_only(tmp_path: Path) -> None:
    runner = Runner(
        items=[1, 2, 3, 4],
        worker_impl=_SelectiveWorker,
        devices=[0, 1],
        n_slots=1,
    )
    completed = runner(tmp_path, silent=True)
    assert set(completed) == {2, 4}
    assert (tmp_path / "failed_jobs.log").is_file()


def test_runner_clears_failed_log_when_no_failures(tmp_path: Path) -> None:
    failed_log = tmp_path / "failed_jobs.log"
    failed_log.write_text("stale\n")

    runner = Runner(
        items=[1, 2],
        worker_impl=_GoodWorker,
        devices=[0],
        n_slots=1,
    )
    completed = runner(tmp_path, silent=True)
    assert completed == [1, 2]
    assert not failed_log.exists()


def test_runner_rejects_invalid_n_slots(tmp_path: Path) -> None:
    runner = Runner(items=[1], worker_impl=_GoodWorker, devices=[0], n_slots=0)
    with pytest.raises(ValueError, match="n_slots"):
        runner(tmp_path, silent=True)


def test_runner_rejects_empty_devices(tmp_path: Path) -> None:
    runner = Runner(items=[1], worker_impl=_GoodWorker, devices=[], n_slots=1)
    with pytest.raises(ValueError, match="at least one device"):
        runner(tmp_path, silent=True)


def test_runner_empty_jobs_returns_empty_list(tmp_path: Path) -> None:
    runner = Runner(items=[], worker_impl=_GoodWorker, devices=[0], n_slots=1)
    assert runner(tmp_path, silent=True) == []
