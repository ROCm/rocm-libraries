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
