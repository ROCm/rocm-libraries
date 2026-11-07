# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""A bootstrap that failed after binding the backend re-raises its own error
on retry (the binding cannot run twice), and an early failure stays retryable.
Runs without a device: torch and _C are stand-ins."""

import sys
import types

import pytest

import hipdnn_vllm
from hipdnn_vllm import _bootstrap
from hipdnn_vllm.config import load_config


class _FakeC:
    def __init__(self):
        self.bind_calls = 0

    def set_backend_directory(self, directory):
        self.bind_calls += 1
        if self.bind_calls == 1:
            raise RuntimeError("backend failed to initialize")
        # A second binding after the first resolution is refused, as in _C.
        return False


@pytest.fixture
def fakes(monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True, init=lambda: None)
    fake_c = _FakeC()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "hipdnn_vllm._C", fake_c)
    monkeypatch.setattr(hipdnn_vllm, "_C", fake_c, raising=False)
    monkeypatch.setitem(sys.modules, "hipdnn_vllm.fake_impls", types.ModuleType("hipdnn_vllm.fake_impls"))
    monkeypatch.setattr(hipdnn_vllm, "fake_impls", sys.modules["hipdnn_vllm.fake_impls"], raising=False)
    monkeypatch.setattr(_bootstrap, "_state", None)
    monkeypatch.setattr(_bootstrap, "_failure", None)
    monkeypatch.setattr(_bootstrap, "_bundled_backend_path", lambda: "/opt/x/libhipdnn_backend.so")
    return fake_c


def test_retry_after_bind_failure_reraises_the_original_error(fakes):
    config = load_config({})
    with pytest.raises(RuntimeError, match="backend failed to initialize"):
        _bootstrap.load(config)
    with pytest.raises(_bootstrap.BootstrapError, match="cannot be retried") as retry:
        _bootstrap.load(config)
    assert isinstance(retry.value.__cause__, RuntimeError)
    assert "backend failed to initialize" in str(retry.value)
    assert fakes.bind_calls == 1


def test_failure_before_binding_stays_retryable(fakes, monkeypatch):
    config = load_config({})
    sys.modules["torch"].cuda.is_available = lambda: False
    with pytest.raises(_bootstrap.BootstrapError, match="no HIP device"):
        _bootstrap.load(config)
    assert _bootstrap._failure is None
    assert fakes.bind_calls == 0
