# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""CPU-hermetic tests for the PyTorch GPU executor control flow."""

import importlib
import sys
import types
from contextlib import contextmanager
from typing import Any, List

import pytest

from dnn_benchmarking.config.benchmark_config import BenchmarkConfig


class FakeStream:
    def __init__(self, ptr: int) -> None:
        self.cuda_stream = ptr


class FakeCuda:
    def __init__(self) -> None:
        self.device_depth = 0
        self.device_entries: List[Any] = []
        self.stream_entries: List[int] = []
        self.default_stream_devices: List[Any] = []
        self.initialized = False
        self.default_stream_obj = FakeStream(0xCAFE)

    def is_available(self) -> bool:
        return True

    def init(self) -> None:
        assert self.device_depth > 0
        self.initialized = True

    def default_stream(self, device: Any) -> FakeStream:
        assert self.device_depth > 0
        self.default_stream_devices.append(device)
        return self.default_stream_obj

    @contextmanager
    def device(self, device: Any):
        self.device_entries.append(device)
        self.device_depth += 1
        try:
            yield
        finally:
            self.device_depth -= 1

    @contextmanager
    def stream(self, stream: FakeStream):
        assert self.device_depth > 0
        self.stream_entries.append(stream.cuda_stream)
        yield


class FakeHipTimer:
    created_streams: List[int] = []
    start_calls = 0
    stop_calls = 0
    sync_calls = 0

    fake_cuda: FakeCuda

    def __init__(self, stream: int = 0) -> None:
        assert self.fake_cuda.device_depth > 0
        self.stream = stream
        self.created_streams.append(stream)

    @property
    def backend_name(self) -> str:
        return "hip"

    def start(self) -> None:
        assert self.fake_cuda.device_depth > 0
        self.__class__.start_calls += 1

    def stop(self) -> None:
        assert self.fake_cuda.device_depth > 0
        self.__class__.stop_calls += 1

    def elapsed_ms(self) -> float:
        assert self.fake_cuda.device_depth > 0
        return 1.25

    def synchronize_stream(self) -> None:
        assert self.fake_cuda.device_depth > 0
        self.__class__.sync_calls += 1


def _load_executor_module(monkeypatch: pytest.MonkeyPatch, fake_cuda: FakeCuda):
    fake_torch = types.SimpleNamespace(
        Tensor=object,
        cuda=fake_cuda,
        device=lambda device: device,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    module_name = "dnn_benchmarking.execution.pytorch_executor"
    old_module = sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "_is_torch_available", lambda: True)
    monkeypatch.setattr(module, "HipGpuTimer", FakeHipTimer)
    monkeypatch.setattr(
        module, "create_gpu_timer", lambda backend, stream=0: FakeHipTimer(stream)
    )
    monkeypatch.setattr(
        module.pytorch_ops, "get_unsupported_operations", lambda graph: []
    )
    yield module

    sys.modules.pop(module_name, None)
    if old_module is not None:
        sys.modules[module_name] = old_module


@pytest.fixture
def pytorch_executor_module(monkeypatch: pytest.MonkeyPatch):
    fake_cuda = FakeCuda()
    FakeHipTimer.created_streams = []
    FakeHipTimer.start_calls = 0
    FakeHipTimer.stop_calls = 0
    FakeHipTimer.sync_calls = 0
    FakeHipTimer.fake_cuda = fake_cuda

    yield from _load_executor_module(monkeypatch, fake_cuda)


def _make_executor(module: Any, timing_backend: str = "hip"):
    config = BenchmarkConfig(graph_path="test.json", warmup_iters=2, benchmark_iters=2)
    executor = module.PyTorchCudaExecutor(
        graph_json={"nodes": []},
        config=config,
        device="cuda:1",
        timing_backend=timing_backend,
    )
    return executor


def test_prepare_creates_stream_timer_on_requested_device(
    pytorch_executor_module,
) -> None:
    module = pytorch_executor_module
    executor = _make_executor(module, timing_backend="none")

    executor.prepare()

    fake_cuda = module.torch.cuda
    assert fake_cuda.initialized is True
    assert fake_cuda.default_stream_devices == ["cuda:1"]
    assert FakeHipTimer.created_streams == [0xCAFE]
    assert all(device == "cuda:1" for device in fake_cuda.device_entries)


def test_timing_backend_none_uses_stream_sync_only(
    pytorch_executor_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = pytorch_executor_module
    executed = []
    monkeypatch.setattr(
        module.pytorch_ops,
        "execute_graph",
        lambda graph, tensors: executed.append("execute"),
    )
    executor = _make_executor(module, timing_backend="none")
    executor.prepare()

    result = executor.benchmark(tensors={}, graph_name="pytorch_none")

    assert executed == ["execute", "execute"]
    assert result.kernel_timings is None
    assert result.metadata is not None
    assert result.metadata.timing_backend == ""
    assert FakeHipTimer.created_streams == [0xCAFE]
    assert FakeHipTimer.start_calls == 0
    assert FakeHipTimer.stop_calls == 0
    assert FakeHipTimer.sync_calls == 2


def test_timing_backend_hip_collects_kernel_timings(
    pytorch_executor_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = pytorch_executor_module
    monkeypatch.setattr(
        module.pytorch_ops, "execute_graph", lambda graph, tensors: None
    )
    executor = _make_executor(module, timing_backend="hip")
    executor.prepare()

    result = executor.benchmark(tensors={}, graph_name="pytorch_hip")

    assert result.kernel_timings == [1.25, 1.25]
    assert result.metadata is not None
    assert result.metadata.timing_backend == "hip"
    assert FakeHipTimer.created_streams == [0xCAFE, 0xCAFE]
    assert FakeHipTimer.start_calls == 2
    assert FakeHipTimer.stop_calls == 2
    assert FakeHipTimer.sync_calls == 0


def test_warmup_and_execute_once_use_stream_sync(
    pytorch_executor_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = pytorch_executor_module
    executed = []
    monkeypatch.setattr(
        module.pytorch_ops,
        "execute_graph",
        lambda graph, tensors: executed.append("execute"),
    )
    executor = _make_executor(module, timing_backend="none")
    executor.prepare()

    executor.warmup(tensors={})
    executor.execute_once(tensors={})

    assert executed == ["execute", "execute", "execute"]
    assert FakeHipTimer.sync_calls == 2
    assert module.torch.cuda.stream_entries == [0xCAFE, 0xCAFE]
