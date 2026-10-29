# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The plan cache keeps declines without keeping the declined call's tensors alive."""

import gc
import weakref

import pytest
import torch

from hipdnn_sdpa import EngineDeclined, _runtime


class _Prob:
    def __init__(self, tensor, key):
        self.tensors = {"q": tensor}
        self.key = key
        self.direction = "fwd"


@pytest.fixture
def declining_runtime(monkeypatch):
    monkeypatch.setattr(_runtime, "_PLANS", {})
    monkeypatch.setattr(_runtime, "_handle", lambda device: object())

    def build(prob, engine, handle):
        raise EngineDeclined(prob.direction, "check_support: no engine")

    monkeypatch.setattr(_runtime, "_build_plan", build)


def _call_and_drop(key):
    """Run one declined call whose tensor is reachable only from this frame."""
    tensor = torch.empty(1024)
    ref = weakref.ref(tensor)
    with pytest.raises(EngineDeclined):
        _runtime.run(_Prob(tensor, key), None)
    return ref


def test_cached_decline_does_not_keep_tensors_alive(declining_runtime):
    ref = _call_and_drop(("shape", 1))
    gc.collect()
    assert ref() is None


def test_cached_decline_is_raised_again_with_the_same_reason(declining_runtime):
    _call_and_drop(("shape", 2))
    with pytest.raises(EngineDeclined) as again:
        _runtime.run(_Prob(torch.empty(1), ("shape", 2)), None)
    assert (again.value.direction, again.value.reason) == (
        "fwd",
        "check_support: no engine",
    )
