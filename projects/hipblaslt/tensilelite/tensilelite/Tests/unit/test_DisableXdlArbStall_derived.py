# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Behavioral tests for the DisableXdlArbStall derivation.

DisableXdlArbStall makes the kernel set SCHED_MODE.DISABLE_XDL_ARB_STALL at
entry, so WMMAs issue back to back. It is evaluated in assignDerivedParameters:
on for every dense kernel, persistent (StreamK, persistent DataParallel) or
not, and off for sparse kernels. Persistent kernels used to derive False.

These tests execute the derivation (config -> BenchmarkProcess -> Solution ->
assignDerivedParameters -> the closure) rather than reading the source. The
emitter side is test_disableWmmaArbStall.py.

An arch whose toolchain is unavailable skips. A new arch adds rows to CONFIGS.
"""

import os

import pytest

from tensilelite.Common.Architectures import gfxToIsa
from tensilelite.Common.Capabilities import makeIsaInfoMap
from tensilelite.ExecutionPolicy import isPersistent, isPersistentDataParallel, isStreamK

pytestmark = pytest.mark.unit

_UNIT = os.path.dirname(__file__)
_DESIGNED = os.path.join(_UNIT, "characterization", "_codegen", "data", "test_data", "_designed")
_COMMON = os.path.join(_UNIT, os.pardir, "common")

_KEY = "DisableXdlArbStall"

# (arch, config, kind). kind names the property the config is chosen for, and
# the test checks the derived states really have it before checking the flag.
CONFIGS = [
    ("gfx1250", os.path.join(_DESIGNED, "gfx1250", "streamk.yaml"), "streamk"),
    (
        "gfx1250",
        os.path.join(_COMMON, "streamk", "gfx1250", "data_parallel_cluster_multicast_boundaries.yaml"),
        "persistent_dp",
    ),
    ("gfx1250", os.path.join(_COMMON, "sparse", "gfx1250", "spmm_f16.yaml"), "sparse"),
]

_STATES = {}


def _require_toolchain(arch):
    from tensilelite.Toolchain.Validators import validateToolchain

    try:
        cxx = validateToolchain("amdclang++")
    except (ValueError, FileNotFoundError) as e:
        pytest.skip(f"amdclang++ is unavailable: {e}")

    isa = gfxToIsa(arch)
    if not makeIsaInfoMap([isa], cxx)[isa].asmCaps["SupportedISA"]:
        pytest.skip(f"amdclang++ in this environment does not support {arch}")


def _states(arch, cfg_path):
    key = (arch, cfg_path)
    if key not in _STATES:
        _require_toolchain(arch)
        from config_harness import derive_states

        _STATES[key] = derive_states(cfg_path, arch=arch, limit_solutions=4)
    return _STATES[key]


@pytest.mark.parametrize("arch, cfg, kind", CONFIGS, ids=[f"{a}-{k}" for a, _, k in CONFIGS])
def test_derivation(arch, cfg, kind):
    states = _states(arch, cfg)
    assert states, f"expected >=1 derived solution from {cfg}"
    for st in states:
        if kind == "streamk":
            assert not st["ProblemType"]["Sparse"] and isStreamK(st)
            assert st[_KEY] is True
        elif kind == "persistent_dp":
            assert not st["ProblemType"]["Sparse"] and isPersistentDataParallel(st)
            assert st[_KEY] is True
        else:
            assert st["ProblemType"]["Sparse"]
            assert st[_KEY] is False
