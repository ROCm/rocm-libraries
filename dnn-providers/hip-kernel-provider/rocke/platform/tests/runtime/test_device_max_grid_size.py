# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``get_device_max_grid_size`` over a mocked ``hipDeviceGetAttribute``.

No GPU and no HIP library: the attribute query is answered by hand, keyed by
the ``hipDeviceAttributeMaxGridDim{X,Y,Z}`` enum values.
"""

from __future__ import annotations

import ctypes

import pytest

from rocke.runtime import hip_module


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(hip_module, "_max_grid_size", {})


def _attrs(answers: dict[int, int], calls: list | None = None):
    """A ``hipDeviceGetAttribute`` stand-in answering ``answers[attr]``."""

    def query(out, attr, device):
        if calls is not None:
            calls.append((attr, device))
        if attr not in answers:
            return 1  # hipErrorInvalidValue
        ctypes.cast(out, ctypes.POINTER(ctypes.c_int)).contents.value = answers[attr]
        return 0

    return query


def test_reads_the_three_grid_dims(monkeypatch):
    monkeypatch.setattr(
        hip_module,
        "_hipDeviceGetAttribute",
        _attrs({29: 2**31 - 1, 30: 65536, 31: 65536}),
    )
    assert hip_module.get_device_max_grid_size() == (2**31 - 1, 65536, 65536)


def test_answer_is_cached_per_device(monkeypatch):
    calls = []
    monkeypatch.setattr(
        hip_module, "_hipDeviceGetAttribute", _attrs({29: 7, 30: 8, 31: 9}, calls)
    )
    assert hip_module.get_device_max_grid_size(1) == (7, 8, 9)
    assert hip_module.get_device_max_grid_size(1) == (7, 8, 9)
    assert calls == [(29, 1), (30, 1), (31, 1)]


def test_unknown_when_hip_refuses_an_axis(monkeypatch):
    monkeypatch.setattr(hip_module, "_hipDeviceGetAttribute", _attrs({29: 1, 30: 1}))
    assert hip_module.get_device_max_grid_size() is None


def test_unknown_when_a_limit_is_not_positive(monkeypatch):
    monkeypatch.setattr(
        hip_module, "_hipDeviceGetAttribute", _attrs({29: 1, 30: 0, 31: 1})
    )
    assert hip_module.get_device_max_grid_size() is None


def test_unknown_without_a_hip_library(monkeypatch):
    def missing(*args):
        raise OSError("libamdhip64 not found")

    monkeypatch.setattr(hip_module, "_hipDeviceGetAttribute", missing)
    assert hip_module.get_device_max_grid_size() is None
