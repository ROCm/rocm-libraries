# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

"""Concurrency runner for executing tasks across multiple devices with load balancing."""

from pathlib import Path
from queue import Queue, Empty
from typing import Sequence, Callable, Any, TypeVar, Generic
from dataclasses import dataclass
from threading import Thread, Event
from geko.concurrency.utils import install_stop_handlers, restore_stop_handlers, parallel_for

import time
import logging
from abc import ABC, abstractmethod
from tqdm import tqdm
logger = logging.getLogger("GEKO")

ItemT = TypeVar("ItemT")


@dataclass
class DeviceState:
    """Tracks currently available capacity and assigned workload for one device."""

    free_slots: int
    workload: float


class Worker(ABC, Generic[ItemT]):
    """Base worker lifecycle for a single scheduled item.

    Subclasses implement setup, run, and teardown. The runner invokes the
    worker in a dedicated thread and receives a `(success, item)` tuple on the
    output queue for progress tracking and failure logging.
    """

    def __init__(
        self,
        item: ItemT,
        device: int,
        slot_id: int,
        stop_event: Event,
        output_queue: Queue[tuple[bool, ItemT]],
    ) -> None:
        self.item = item
        self.device = device
        self.slot_id = slot_id
        self.stop_event = stop_event
        self.output_queue = output_queue
        
    @abstractmethod
    def setup(self) -> None:
        """Prepare any state needed before running the job."""
    
    @abstractmethod
    def run(self) -> bool:
        """Execute the job and return whether it completed successfully."""

    @abstractmethod
    def teardown(self) -> None:
        """Clean up worker state regardless of run outcome."""

    def __call__(self) -> None:
        """Run the full worker lifecycle and publish the final status to the queue."""

        success = False
        try:
            self.setup()
            success = self.run()
        except Exception as e:
            logger.error(f"Worker encountered an error with item={self.item}: {e}")
        finally:
            try:
                self.teardown()
            except Exception as e:
                logger.error(f"Worker encountered an error during teardown with item={self.item}: {e}")
        self.output_queue.put((success, self.item))

class Runner(Generic[ItemT]):
    """Schedule workload-balanced worker threads across multiple devices.

    Jobs are ordered by descending estimated workload, then assigned to the
    currently least-loaded device with a free slot. Progress is driven by worker
    completions reported through a queue.
    """

    def __init__(
        self,
        items: Sequence[ItemT],
        worker_impl: type[Worker[ItemT]],
        devices: Sequence[int],
        n_slots: int = 1,
        estimate_workload_fn: Callable[[ItemT], float] = lambda _item: 0.0,
        job_logger_fn: Callable[[], None] = lambda: None,
    ) -> None:
        """Build a runner for a collection of items.

        Args:
            items: Items to schedule.
            worker_impl: Worker type instantiated once per scheduled item.
            devices: Device identifiers that can run jobs.
            n_slots: Maximum concurrent jobs allowed per device.
            estimate_workload_fn: Callback used to estimate relative job cost.
            job_logger_fn: Optional callback invoked periodically by the scheduler.
        """
        estimate_workload_fn = estimate_workload_fn or (lambda _item: 0.0)
        def _estimate_wkld(item: ItemT) -> tuple[ItemT, float]:
            try:
                return item, estimate_workload_fn(item)
            except Exception as e:
                logger.warning(f"Could not compute workload for '{item}': {e}")
                return item, 0.0
            
        jobs = parallel_for(_estimate_wkld, items)

        jobs.sort(key=lambda x: x[1], reverse=True)
        logger.debug(f"Prepared {len(jobs)} jobs for scheduling")

        if len(jobs) < len(devices):
            devices = devices[: len(jobs)]
        logger.debug(f"Using devices for load balancing: {devices}")

        self.jobs = jobs
        self.worker_impl = worker_impl
        self.devices = devices
        self.n_slots = n_slots
        self.job_logger_fn = job_logger_fn
    
    def __call__(self, workdir: str | Path, silent: bool = False) -> list[ItemT]:
        """Execute all jobs and return the items that completed successfully.

        Args:
            workdir: Directory where runner-owned progress artifacts are written.
            silent: If True, disable progress logging.

        Returns:
            Items whose workers reported success.
        """

        total_jobs = len(self.jobs)
        if total_jobs == 0:
            return []

        if self.n_slots < 1:
            raise ValueError("n_slots must be >= 1")
        if len(self.devices) == 0:
            raise ValueError("Need at least one device")

        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        failed_log_file = workdir / "failed_jobs.log"

        jobs = list(self.jobs)

        _stop_event = Event()
        output_queue: Queue[tuple[bool, ItemT]] = Queue()

        def _runner() -> None:
            """Drive scheduling and slot reclamation until all work is drained."""

            device_states = {
                device_id: DeviceState(free_slots=self.n_slots, workload=0.0)
                for device_id in self.devices
            }

            active_threads: dict[tuple[int, int], Thread] = {}
            assigned_workloads: dict[tuple[int, int], float] = {}

            while len(jobs) > 0 or len(active_threads) > 0:
                # Reap finished workers and release their slots.
                finished = [k for k, thr in active_threads.items() if not thr.is_alive()]
                for key in finished:
                    active_threads[key].join()
                    device, _ = key
                    device_states[device].workload -= assigned_workloads.pop(key, 0.0)
                    device_states[device].free_slots = min(
                        self.n_slots, device_states[device].free_slots + 1
                    )
                    del active_threads[key]

                if _stop_event.is_set() and len(active_threads) == 0:
                    break

                assigned_any = False
                while len(jobs) > 0 and not _stop_event.is_set():
                    eligible_devices = [d for d, st in device_states.items() if st.free_slots > 0]
                    if len(eligible_devices) == 0:
                        break

                    device = min(eligible_devices, key=lambda d: device_states[d].workload)

                    slot_id = next((s for s in range(self.n_slots) if (device, s) not in active_threads), None)
                    if slot_id is None:
                        break

                    item, workload = jobs.pop(0)
                    device_states[device].workload += workload
                    device_states[device].free_slots -= 1

                    logger.debug(
                        f"Assigned item={item} to device={device} slot={slot_id} "
                        f"workload={workload:.3f} new_workload={device_states[device].workload:.3f} "
                        f"free_slots={device_states[device].free_slots}"
                    )

                    worker = self.worker_impl(item, device, slot_id, _stop_event, output_queue)
                    t = Thread(target=worker, args=())
                    t.start()
                    active_threads[(device, slot_id)] = t
                    assigned_workloads[(device, slot_id)] = workload
                    assigned_any = True

                if not assigned_any and len(active_threads) > 0:
                    time.sleep(0.2)
                
                if self.job_logger_fn is not None:
                    self.job_logger_fn()

            for t in active_threads.values():
                t.join()

        prev_handlers = install_stop_handlers(_stop_event)

        completed: list[ItemT] = []
        failed: list[ItemT] = []
        try:
            runner = Thread(target=_runner, daemon=False)
            runner.start()

            n_completed, n_failed = 0, 0
            with tqdm(total=total_jobs, desc="Jobs in progress", disable=silent) as pbar:
                while True:
                    runner.join(timeout=0.05)
                    if _stop_event.is_set():
                        runner.join()
                        break
                    
                    try:
                        res = output_queue.get(timeout=1)
                    except Empty:
                        if not runner.is_alive() and output_queue.empty():
                            runner.join()
                            break
                        continue  # No result yet, loop back to check scheduler
                    
                    success, item = res
                    if success:
                        completed.append(item)
                        n_completed += 1
                    else:
                        failed.append(item)
                        n_failed += 1

                    pbar.update(1)
                    pbar.set_postfix({"n_completed": n_completed, "n_failed": n_failed})

                    if n_failed > 0:
                        with open(failed_log_file, "w") as f:
                            f.write("\n".join(str(i) for i in failed))

                    if not runner.is_alive() and output_queue.empty():
                        runner.join()
                        break

        finally:
            restore_stop_handlers(prev_handlers)

        if _stop_event.is_set():
            logger.warning("Optimization interrupted; terminating workflow")
            raise SystemExit(130) # Exit code = 128 + 2 (SIGINT)
    
        if n_failed > 0:
            logger.warning(f"{n_failed} jobs failed, check '{failed_log_file}' for more information")
        elif failed_log_file.is_file():
            failed_log_file.unlink()
        
        return completed
