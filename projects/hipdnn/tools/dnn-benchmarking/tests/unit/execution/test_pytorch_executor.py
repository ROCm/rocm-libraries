# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""CPU-hermetic tests for the PyTorch GPU executor control flow."""

import importlib
import sys
import types
from contextlib import contextmanager
from typing import Any, List

import pytest

from dnn_benchmarking.config.benchmark_config import BenchmarkConfig, TimingBackendName


class FakeStream:
    def __init__(self, ptr: int) -> None:
        self.cuda_stream = ptr
        self.synchronize_calls = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1


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


def _load_executor_module(
    monkeypatch: pytest.MonkeyPatch, fake_cuda: FakeCuda, is_rocm: bool = True
):
    fake_torch = types.ModuleType("torch")
    fake_torch.__path__ = []  # mark as package for torch.nn.functional imports
    fake_torch.Tensor = object
    fake_torch.cuda = fake_cuda
    fake_torch.device = lambda device: device
    fake_nn = types.ModuleType("torch.nn")
    fake_functional = types.ModuleType("torch.nn.functional")
    fake_nn.functional = fake_functional
    fake_torch.nn = fake_nn
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_nn)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", fake_functional)

    module_name = "dnn_benchmarking.execution.pytorch_executor"
    old_module = sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    # The executor resolves the auto/none timing backend from the torch
    # build (ROCm -> HIP events, CUDA -> torch events), so drive that here.
    monkeypatch.setattr(module.torch_support, "gpu_available", lambda: True)
    monkeypatch.setattr(module.torch_support, "is_rocm_build", lambda: is_rocm)
    monkeypatch.setattr(module, "HipGpuTimer", FakeHipTimer)
    monkeypatch.setattr(
        module,
        "create_gpu_timer",
        lambda backend, stream=0, torch_stream=None: FakeHipTimer(stream),
    )
    monkeypatch.setattr(
        module.pytorch_ops, "get_unsupported_operations", lambda graph: []
    )
    yield module

    sys.modules.pop(module_name, None)
    import dnn_benchmarking.execution as execution_pkg

    if old_module is not None:
        sys.modules[module_name] = old_module
        execution_pkg.pytorch_executor = old_module
    elif getattr(execution_pkg, "pytorch_executor", None) is module:
        # Drop the stale package attribute so later imports (and
        # mock.patch dotted-name resolution) see a freshly imported module
        # instead of this fixture's fake-torch variant.
        del execution_pkg.pytorch_executor


@pytest.fixture
def pytorch_executor_module(monkeypatch: pytest.MonkeyPatch):
    fake_cuda = FakeCuda()
    FakeHipTimer.created_streams = []
    FakeHipTimer.start_calls = 0
    FakeHipTimer.stop_calls = 0
    FakeHipTimer.sync_calls = 0
    FakeHipTimer.fake_cuda = fake_cuda

    yield from _load_executor_module(monkeypatch, fake_cuda)


def _make_executor(module: Any, collect_kernel_timing: bool = True):
    config = BenchmarkConfig(graph_path="test.json", warmup_iters=2, benchmark_iters=2)
    executor = module.PyTorchCudaExecutor(
        graph_json={"nodes": []},
        config=config,
        device="cuda:1",
        collect_kernel_timing=collect_kernel_timing,
    )
    return executor


def test_prepare_creates_stream_timer_on_requested_device(
    pytorch_executor_module,
) -> None:
    module = pytorch_executor_module
    executor = _make_executor(module, collect_kernel_timing=False)

    executor.prepare()

    fake_cuda = module.torch.cuda
    assert fake_cuda.initialized is True
    assert fake_cuda.default_stream_devices == ["cuda:1"]
    assert FakeHipTimer.created_streams == [0xCAFE]
    assert all(device == "cuda:1" for device in fake_cuda.device_entries)


def test_no_kernel_timing_uses_stream_sync_only(
    pytorch_executor_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = pytorch_executor_module
    executed = []
    monkeypatch.setattr(
        module.pytorch_ops,
        "execute_graph",
        lambda graph, tensors: executed.append("execute"),
    )
    executor = _make_executor(module, collect_kernel_timing=False)
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


def test_collect_kernel_timing_collects_kernel_timings(
    pytorch_executor_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = pytorch_executor_module
    monkeypatch.setattr(
        module.pytorch_ops, "execute_graph", lambda graph, tensors: None
    )
    executor = _make_executor(module, collect_kernel_timing=True)
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
    executor = _make_executor(module, collect_kernel_timing=False)
    executor.prepare()

    executor.warmup(tensors={})
    executor.execute_once(tensors={})

    assert executed == ["execute", "execute", "execute"]
    assert FakeHipTimer.sync_calls == 2
    assert module.torch.cuda.stream_entries == [0xCAFE, 0xCAFE]


@pytest.fixture
def pytorch_executor_module_no_hip(monkeypatch: pytest.MonkeyPatch):
    """Executor module fixture for a CUDA torch build (no HIP events)."""
    fake_cuda = FakeCuda()
    FakeHipTimer.created_streams = []
    FakeHipTimer.start_calls = 0
    FakeHipTimer.stop_calls = 0
    FakeHipTimer.sync_calls = 0
    FakeHipTimer.fake_cuda = fake_cuda

    yield from _load_executor_module(monkeypatch, fake_cuda, is_rocm=False)


def test_prepare_without_hip_skips_hip_sync_timer(
    pytorch_executor_module_no_hip,
) -> None:
    module = pytorch_executor_module_no_hip
    executor = _make_executor(module, collect_kernel_timing=True)

    executor.prepare()

    assert FakeHipTimer.created_streams == []


def test_no_hip_synchronizes_through_torch_stream(
    pytorch_executor_module_no_hip, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = pytorch_executor_module_no_hip
    executed = []
    monkeypatch.setattr(
        module.pytorch_ops,
        "execute_graph",
        lambda graph, tensors: executed.append("execute"),
    )
    executor = _make_executor(module, collect_kernel_timing=False)
    executor.prepare()

    result = executor.benchmark(tensors={}, graph_name="pytorch_cuda")

    assert executed == ["execute", "execute"]
    assert result.kernel_timings is None
    assert FakeHipTimer.created_streams == []
    assert FakeHipTimer.sync_calls == 0
    assert module.torch.cuda.default_stream_obj.synchronize_calls == 2


def test_auto_on_cuda_build_requests_torch_not_hip(
    pytorch_executor_module_no_hip, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto timing on a CUDA torch build must resolve to torch, never HIP.

    Guards the mixed-host case (CUDA torch with visible ROCm/hipDNN): the
    executor must not record HIP events on a CUDA stream pointer.
    """
    module = pytorch_executor_module_no_hip
    requested = []

    class FakeTorchTimer:
        backend_name = "torch"

        def __init__(self, stream) -> None:
            self.stream = stream

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def elapsed_ms(self) -> float:
            return 0.5

    monkeypatch.setattr(
        module,
        "create_gpu_timer",
        lambda backend, stream=0, torch_stream=None: requested.append(backend)
        or FakeTorchTimer(torch_stream),
    )
    monkeypatch.setattr(
        module.pytorch_ops, "execute_graph", lambda graph, tensors: None
    )
    executor = _make_executor(module, collect_kernel_timing=True)
    executor.prepare()

    result = executor.benchmark(tensors={}, graph_name="pytorch_mixed_host")

    # Resolved to torch despite the factory preferring HIP when available.
    assert requested == [TimingBackendName.TORCH]
    # No HIP sync timer created in prepare() either.
    assert FakeHipTimer.created_streams == []
    assert result.metadata is not None
    assert result.metadata.timing_backend == "torch"
