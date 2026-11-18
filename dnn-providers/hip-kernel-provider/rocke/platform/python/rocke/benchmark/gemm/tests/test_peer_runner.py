# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Mock-only runner/ABI/lifetime tests; these are not GPU numerical validation."""

from __future__ import annotations

import ctypes
import re
import struct
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from rocke.benchmark.gemm import peer_runner as runner_module
from rocke.instances.common import peer_gemm_pipeline as pipeline_module
from rocke.instances.common.peer_gemm import (
    PeerGemmScheduleSpec,
    PeerGemmSpec,
    scheduled_peer_gemm_cta_work,
)


class FakeRuntime:
    """Execute packed-argument host models only when that device's FIFO drains."""

    def __init__(self, spec):
        self.spec = spec
        self.device = 0
        self.memory = {}
        self.queues = {}
        self.retained = {}
        self.log = []
        self.fail_device = None
        self.fail_launch = None
        self.launch_count = 0
        self.fail_event = None
        self.event_count = 0

    @contextmanager
    def scope(self, device):
        previous = self.device
        self.device = device
        try:
            yield
        finally:
            self.device = previous

    def alloc(self, nbytes):
        pointer = (self.device + 1) * 0x10000000 + (len(self.memory) + 1) * 0x100000
        self.memory[pointer] = (self.device, bytearray(nbytes))
        return pointer

    def free(self, pointer):
        assert self.memory[pointer][0] == self.device
        assert not any(self.queues.values()), "free before all-rank completion"
        self.log.append(("free", self.device))
        del self.memory[pointer]

    def view(self, pointer, words, dtype=np.uint16):
        base = next(
            base
            for base, (_, data) in self.memory.items()
            if base <= pointer < base + len(data)
        )
        data = self.memory[base][1]
        return np.frombuffer(data, dtype=dtype, count=words, offset=pointer - base)

    def memcpy_h2d(self, destination, host, nbytes):
        assert not any(self.queues.values()), "upload before all-rank completion"
        self.view(destination, nbytes, np.uint8)[:] = np.frombuffer(
            bytes(host), np.uint8, count=nbytes
        )

    def memcpy_d2h(self, host, source, nbytes):
        assert not any(self.queues.values()), "readback before all-rank completion"
        ctypes.memmove(host, self.view(source, nbytes, np.uint8).tobytes(), nbytes)

    def load_module(self, blob):
        owner = self.device

        def unload():
            assert not any(self.queues.values()), "unload before all-rank completion"
            self.log.append(("unload", owner))

        return SimpleNamespace(
            get_function=lambda name: SimpleNamespace(
                device=owner, kind="copy" if "tile_copy" in name else "gemm", name=name
            ),
            unload=unload,
        )

    def launch(self, function, grid, block, packed, **kwargs):
        assert function.device == self.device
        assert kwargs == {"stream": 0, "record_event": False}
        self.launch_count += 1
        if self.launch_count == self.fail_launch:
            raise RuntimeError("synthetic enqueue failure")
        self.log.append(("launch", self.device, function.kind))
        match = re.search(
            r"_(source_grouped|tile_interleaved)_q([0-9]+)$", function.name
        )
        if match:
            assert grid == (int(match.group(2)), 1, 1)
        self.queues.setdefault(self.device, []).append(
            (function.kind, packed, function.name)
        )

    def retain_for_stream(self, stream, *owners):
        self.retained.setdefault(self.device, []).append(owners)

    def wait_stream(self, stream):
        self.log.append(("wait", self.device))
        if self.device == self.fail_device:
            raise RuntimeError("synthetic drain failure")
        for kind, packed, name in self.queues.pop(self.device, []):
            if kind == "copy":
                values = struct.unpack("<QQ" + "q" * 14, packed)
                source, target, rows, cols = values[:4]
                # The baseline uses contiguous bounded panels with zero origins.
                assert values[4:8] == (rows, cols, rows, cols)
                assert values[8:12] == (0, 0, 0, 0)
                assert values[12:] == (cols, 1, cols, 1)
                self.view(target, rows * cols)[:] = self.view(source, rows * cols)
            elif kind == "gemm":
                ap, bp, cp = struct.unpack("<QQQ", packed)
                m, n, k, p = (
                    self.spec.rows_per_source,
                    self.spec.cols_per_rank,
                    self.spec.k,
                    self.spec.sources,
                )
                a_addresses = self.view(ap, p, np.uint64)
                c_addresses = self.view(cp, p, np.uint64)
                b = runner_module._decode(
                    self.view(bp, k * n), self.spec.dtype
                ).reshape(k, n)
                match = re.search(
                    r"_(source_grouped|tile_interleaved)_q([0-9]+)$", name
                )
                if match:
                    schedule = PeerGemmScheduleSpec(
                        self.spec, match.group(1), int(match.group(2))
                    )
                    coordinates = [
                        item
                        for cta in range(schedule.cta_count)
                        for item in scheduled_peer_gemm_cta_work(schedule, cta)
                    ]
                else:
                    coordinates = [
                        (source, row, col)
                        for source in range(p)
                        for row in range(m // 16)
                        for col in range(n // 16)
                    ]
                for source, row, col in coordinates:
                    a = runner_module._decode(
                        self.view(int(a_addresses[source]), m * k), self.spec.dtype
                    ).reshape(m, k)
                    rows, columns = slice(row * 16, (row + 1) * 16), slice(
                        col * 16, (col + 1) * 16
                    )
                    values = a[rows] @ b[:, columns]
                    output = self.view(int(c_addresses[source]), m * n).reshape(m, n)
                    output[rows, columns] = runner_module._encode(
                        values, self.spec.dtype
                    )
        self.retained.pop(self.device, None)

    def event(self):
        self.event_count += 1
        if self.event_count == self.fail_event:
            raise RuntimeError("synthetic event allocation failure")
        device = self.device

        def record():
            assert self.device == device
            self.log.append(("event", device))

        def elapsed(other):
            assert other.device == device
            self.log.append(("elapsed", device))
            return 1.0  # Synthetic fixture value, not hardware timing.

        def destroy():
            assert not any(self.queues.values())
            self.log.append(("destroy", device))

        return SimpleNamespace(
            device=device, record=record, elapsed_to=elapsed, destroy=destroy
        )


@pytest.fixture
def fake_environment(monkeypatch):
    def install(spec):
        rt = FakeRuntime(spec)
        monkeypatch.setattr(runner_module, "Runtime", lambda: rt)
        monkeypatch.setattr(runner_module, "device_scope", rt.scope)
        monkeypatch.setattr(pipeline_module, "device_scope", rt.scope)
        monkeypatch.setattr(runner_module, "get_device_count", lambda: 8)
        monkeypatch.setattr(runner_module, "get_device_arch", lambda device: "gfx950")
        monkeypatch.setattr(
            runner_module,
            "compile_kernel",
            lambda *args, **kwargs: SimpleNamespace(hsaco=b"fixture"),
        )

        @contextmanager
        def mappings(paths):
            rt.log.append(("map", paths))
            yield
            assert not any(rt.queues.values()), "unmap before all-rank completion"
            rt.log.append(("unmap", paths))

        monkeypatch.setattr(runner_module, "peer_access_scope", mappings)
        return rt

    return install


@pytest.mark.parametrize("sources", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize(
    "inputs,outputs",
    [
        ("gather", "local"),
        ("gather", "scatter"),
        ("replicated", "local"),
        ("replicated", "scatter"),
    ],
)
def test_runner_mock_reference_guards_fifo_and_teardown(
    fake_environment, sources, dtype, inputs, outputs
):
    spec = PeerGemmSpec(16, 32, 32, sources=sources, dtype=dtype)
    rt = fake_environment(spec)
    with runner_module.PeerGemmRunner(
        spec,
        devices=tuple(reversed(range(sources))),
        arch="gfx950",
        input_placement=inputs,
        output_placement=outputs,
    ) as runner:
        for fused in (True, False):
            check = runner.validate(fused=fused, repeats=3)
            assert check["status"] == "passed"
            assert (
                check["launches_per_rank_per_iteration"]
                == (
                    (
                        1
                        if fused
                        else 1
                        + sources * ((inputs == "gather") + (outputs == "scatter"))
                    ),
                )
                * sources
            )
            assert not any("ms" in key or "timing" in key for key in check)
            assert not rt.queues and not rt.retained
        measured = runner.measure(fused=False)
        assert len(measured.device_stream_ms) == sources
        assert rt.device == 0
    assert not rt.memory
    names = [entry[0] for entry in rt.log]
    assert names.index("unmap") < names.index("unload") < names.index("free")
    assert names.count("destroy") == 2 * sources
    with pytest.raises(RuntimeError, match="not active"):
        runner.enqueue(fused=True)


def test_drain_failure_preserves_live_allocations_modules_and_mappings(
    fake_environment,
):
    spec = PeerGemmSpec(16, 16, 16, sources=2)
    rt = fake_environment(spec)
    runner = runner_module.PeerGemmRunner(spec, devices=(0, 1), arch="gfx950")
    runner.enqueue(fused=False)
    rt.fail_device = 0
    with pytest.raises(RuntimeError, match="failed to drain"):
        runner.close()
    assert runner._work_may_be_live
    assert rt.memory and rt.queues and rt.retained
    assert ("wait", 1) in rt.log
    assert not any(entry[0] in ("free", "unload", "unmap") for entry in rt.log)
    rt.fail_device = None
    runner.close()
    assert not rt.memory and not rt.queues and not rt.retained


def test_enqueue_failure_can_be_drained_without_releasing_peer_owners_early(
    fake_environment,
):
    spec = PeerGemmSpec(16, 16, 16, sources=2)
    rt = fake_environment(spec)
    runner = runner_module.PeerGemmRunner(spec, devices=(0, 1), arch="gfx950")
    rt.fail_launch = 2
    with pytest.raises(RuntimeError, match="enqueue failure"):
        runner.enqueue(fused=False)
    assert rt.retained and rt.queues
    runner.close()
    assert not rt.memory and not rt.queues


def test_measurement_requires_matching_path_correctness(fake_environment):
    spec = PeerGemmSpec(16, 16, 16, sources=1)
    rt = fake_environment(spec)
    with runner_module.PeerGemmRunner(spec, devices=(0,), arch="gfx950") as runner:
        with pytest.raises(RuntimeError, match="validate this path"):
            runner.measure(fused=False)
        runner.validate(fused=True)
        with pytest.raises(RuntimeError, match="validate this path"):
            runner.measure(fused=False)
        assert rt.event_count == 0


def test_runner_detects_corrupted_input_and_output_guards(
    fake_environment, monkeypatch
):
    spec = PeerGemmSpec(16, 16, 16, sources=1)
    rt = fake_environment(spec)
    with runner_module.PeerGemmRunner(spec, devices=(0,), arch="gfx950") as runner:
        original = runner.enqueue

        def corrupt(*, fused):
            counts = original(fused=fused)
            runner.drain()
            rt.view(runner.outputs[0].pointer, 1)[0] = 0
            return counts

        monkeypatch.setattr(runner, "enqueue", corrupt)
        with pytest.raises(AssertionError, match="guards"):
            runner.validate(fused=True)


def test_second_event_creation_failure_cleans_up_first_event(fake_environment):
    spec = PeerGemmSpec(16, 16, 16, sources=1)
    rt = fake_environment(spec)
    with runner_module.PeerGemmRunner(spec, devices=(0,), arch="gfx950") as runner:
        runner.validate(fused=True)
        rt.fail_event = 2
        with pytest.raises(RuntimeError, match="event allocation"):
            runner.measure(fused=True)
        assert not runner._events
        assert [entry[0] for entry in rt.log].count("destroy") == 1


def test_partial_measurement_keeps_events_until_all_rank_drain(fake_environment):
    spec = PeerGemmSpec(16, 16, 16, sources=2)
    rt = fake_environment(spec)
    runner = runner_module.PeerGemmRunner(spec, devices=(0, 1), arch="gfx950")
    runner.validate(fused=True)
    rt.fail_launch = rt.launch_count + 2
    with pytest.raises(RuntimeError, match="enqueue failure"):
        runner.measure(fused=True)
    assert rt.queues and runner._events and runner._work_may_be_live
    runner.close()
    assert not runner._events and not rt.memory
    assert [entry[0] for entry in rt.log].count("destroy") == 4


def test_partial_free_failure_retry_does_not_double_free(fake_environment, monkeypatch):
    spec = PeerGemmSpec(16, 16, 16, sources=1)
    rt = fake_environment(spec)
    runner = runner_module.PeerGemmRunner(spec, devices=(0,), arch="gfx950")
    free = rt.free
    calls = 0

    def fail_once(pointer):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic free failure")
        free(pointer)

    monkeypatch.setattr(rt, "free", fail_once)
    with pytest.raises(RuntimeError, match="free failure"):
        runner.close()
    runner.close()
    assert not rt.memory


def test_staging_guard_corruption_is_detected(fake_environment, monkeypatch):
    spec = PeerGemmSpec(16, 16, 16, sources=1)
    rt = fake_environment(spec)
    with runner_module.PeerGemmRunner(spec, devices=(0,), arch="gfx950") as runner:
        original = runner.enqueue

        def corrupt(*, fused):
            counts = original(fused=fused)
            runner.drain()
            stage = runner.plans[0].local_tables.keepalive[0].owner
            rt.view(stage.pointer, 1)[0] = 0
            return counts

        monkeypatch.setattr(runner, "enqueue", corrupt)
        with pytest.raises(AssertionError, match="staging guards"):
            runner.validate(fused=False)


@pytest.mark.parametrize("order", ["source_grouped", "tile_interleaved"])
@pytest.mark.parametrize("cap", [0, 1, 3])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize(
    "inputs,outputs",
    [
        ("gather", "local"),
        ("gather", "scatter"),
        ("replicated", "local"),
        ("replicated", "scatter"),
    ],
)
def test_scheduled_runner_mock_matched_baseline_and_grid(
    fake_environment, order, cap, dtype, inputs, outputs
):
    spec = PeerGemmSpec(32, 48, 32, sources=4, dtype=dtype)
    schedule = PeerGemmScheduleSpec(spec, order, cap)
    rt = fake_environment(spec)
    with runner_module.PeerGemmRunner(
        spec,
        devices=(0, 1, 2, 3),
        arch="gfx950",
        input_placement=inputs,
        output_placement=outputs,
        schedule=schedule,
    ) as runner:
        assert all(gemm.name == schedule.kernel_name() for gemm, _ in runner.functions)
        for fused in [True, False]:
            check = runner.validate(fused=fused, repeats=2)
            assert check["status"] == "passed" and check["work_order"] == order
            assert (
                check["max_ctas"] == cap
                and check["scheduled_ctas"] == schedule.cta_count
            )
            assert not rt.queues and not rt.retained
    assert not rt.memory


def test_schedule_mismatch_rejects_before_gpu_setup(fake_environment, monkeypatch):
    spec = PeerGemmSpec(32, 48, 32, sources=4)
    fake_environment(spec)
    monkeypatch.setattr(
        runner_module,
        "get_device_count",
        lambda: pytest.fail("GPU setup ran before schedule validation"),
    )
    with pytest.raises(ValueError, match="must match"):
        runner_module.PeerGemmRunner(
            spec,
            devices=(0, 1, 2, 3),
            arch="gfx950",
            schedule=PeerGemmScheduleSpec(PeerGemmSpec(16, 16, 16)),
        )
