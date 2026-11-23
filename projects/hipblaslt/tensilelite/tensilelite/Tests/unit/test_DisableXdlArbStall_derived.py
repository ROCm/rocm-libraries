# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Behavioral tests for the DisableXdlArbStall derivation.

DisableXdlArbStall makes the kernel set SCHED_MODE.DISABLE_XDL_ARB_STALL at entry
so WMMAs issue back to back. It is derived in assignDerivedParameters as on for
every dense kernel, including persistent (StreamK and persistent DataParallel)
ones, and off for sparse kernels. Persistent kernels used to derive False.

These tests execute the derivation rather than reading the source. The emitter
side is test_disableWmmaArbStall.py. An arch whose toolchain is unavailable skips.
"""

import os

import pytest

from tensilelite.ExecutionPolicy import isPersistentDataParallel, isStreamK
from toolchain_test_helpers import shared_isa_info_map_or_skip

pytestmark = pytest.mark.unit

_UNIT = os.path.dirname(__file__)
_DESIGNED = os.path.join(_UNIT, "characterization", "_codegen", "data", "test_data", "_designed")
_COMMON = os.path.join(_UNIT, os.pardir, "common")

_KEY = "DisableXdlArbStall"
_ARCH = "gfx1250"
_STREAMK = os.path.join(_DESIGNED, _ARCH, "streamk.yaml")
_PERSISTENT_DP = os.path.join(_COMMON, "streamk", _ARCH, "data_parallel_cluster_multicast_boundaries.yaml")
_SPARSE = os.path.join(_COMMON, "sparse", _ARCH, "spmm_f16.yaml")

_STATES = {}


def _states(cfg_path):
    """Derived solution states for one config, cached per config."""
    if cfg_path not in _STATES:
        shared_isa_info_map_or_skip(_ARCH, toolchainRequired=False)
        from config_harness import derive_states

        _STATES[cfg_path] = derive_states(cfg_path, arch=_ARCH, limit_solutions=4)
    return _STATES[cfg_path]


def test_enabled_for_streamk():
    states = _states(_STREAMK)
    assert states
    for st in states:
        assert not st["ProblemType"]["Sparse"] and isStreamK(st)
        assert st[_KEY] is True


def test_enabled_for_persistent_data_parallel():
    states = _states(_PERSISTENT_DP)
    assert states
    for st in states:
        assert not st["ProblemType"]["Sparse"] and isPersistentDataParallel(st)
        assert st[_KEY] is True


def test_disabled_for_sparse():
    states = _states(_SPARSE)
    assert states
    for st in states:
        assert st["ProblemType"]["Sparse"]
        assert st[_KEY] is False
