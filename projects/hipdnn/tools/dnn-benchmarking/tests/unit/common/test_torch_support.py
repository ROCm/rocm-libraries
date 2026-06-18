# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for PyTorch availability helpers."""

import builtins
import sys
from types import SimpleNamespace

from dnn_benchmarking.common import torch_support


def test_module_available_false_when_torch_import_fails(monkeypatch) -> None:
    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("blocked torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(builtins, "__import__", blocking_import)

    assert torch_support.module_available() is False


def test_module_available_false_when_torch_import_raises_oserror(monkeypatch) -> None:
    """Broken installs raise OSError (missing shared libs), not ImportError."""
    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise OSError("libcudart.so.12: cannot open shared object file")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(builtins, "__import__", broken_import)

    assert torch_support.module_available() is False


def test_module_available_false_when_torch_import_raises_runtimeerror(
    monkeypatch,
) -> None:
    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise RuntimeError("CUDA driver initialization failed")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(builtins, "__import__", broken_import)

    assert torch_support.module_available() is False


def test_gpu_available_reflects_torch_cuda_state(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert torch_support.gpu_available() is True


def test_gpu_available_false_when_torch_cuda_probe_fails(monkeypatch) -> None:
    def fail() -> bool:
        raise RuntimeError("no device")

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=fail))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert torch_support.gpu_available() is False


def test_is_rocm_build_detects_hip(monkeypatch) -> None:
    fake_torch = SimpleNamespace(version=SimpleNamespace(hip="6.4.0", cuda=None))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert torch_support.is_rocm_build() is True
    assert torch_support.is_cuda_build() is False


def test_is_cuda_build_detects_cuda(monkeypatch) -> None:
    fake_torch = SimpleNamespace(version=SimpleNamespace(hip=None, cuda="12.8"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert torch_support.is_cuda_build() is True
    assert torch_support.is_rocm_build() is False


def test_build_detection_false_for_cpu_torch(monkeypatch) -> None:
    fake_torch = SimpleNamespace(version=SimpleNamespace(hip=None, cuda=None))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert torch_support.is_cuda_build() is False
    assert torch_support.is_rocm_build() is False
