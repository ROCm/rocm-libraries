# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Behavioral tests for the EnableESM2TrackValuVsrc derivation.

EnableESM2TrackValuVsrc is the switch that makes expert scheduling mode2 track
VALU source operands on VA_VDST (the src-operand WAR hazard). It is evaluated
in assignDerivedParameters and is unconditional: every derived solution gets
True. It used to be derived from the Sparse problem type.

These tests execute the derivation (config -> BenchmarkProcess -> Solution ->
assignDerivedParameters -> the closure) rather than reading the source. The
source-text companion test_EnableESM2TrackValuVsrc.py stays as the fast,
toolchain-free guard; this file is the executing one.

Nothing below is arch-specific: the archs under test and the dense/sparse config
pair each one derives from are ARCH_CONFIGS, and an arch whose toolchain is
unavailable skips. A new arch adds a row there and needs no new test.
"""

import os

import pytest

from Tensile.Common.Architectures import gfxToIsa
from Tensile.Common.Capabilities import makeIsaInfoMap

pytestmark = pytest.mark.unit

_UNIT = os.path.dirname(__file__)
_DESIGNED = os.path.join(_UNIT, "characterization", "_codegen", "data", "test_data", "_designed")
_COMMON = os.path.join(_UNIT, os.pardir, "common")

_KEY = "EnableESM2TrackValuVsrc"

# One row per arch: (dense config, sparse config). The pair must differ only in
# Sparse, so the flag's independence from it is what the tests measure.
# SIA=4 is the stinkytofu path, i.e. the one the flag is about.
ARCH_CONFIGS = {
    "gfx1250": (
        os.path.join(_DESIGNED, "gfx1250", "streamk.yaml"),
        os.path.join(_COMMON, "sparse", "gfx1250", "spmm_f16_sia4.yaml"),
    ),
}

ARCHS = sorted(ARCH_CONFIGS)

_STATES = {}


def _require_toolchain(arch):
    """Skip where amdclang++ cannot target ``arch``; the derivation would raise."""
    from Tensile.Toolchain.Validators import validateToolchain

    try:
        cxx = validateToolchain("amdclang++")
    except (ValueError, FileNotFoundError) as e:
        pytest.skip(f"amdclang++ is unavailable: {e}")

    isa = gfxToIsa(arch)
    if not makeIsaInfoMap([isa], cxx)[isa].asmCaps["SupportedISA"]:
        pytest.skip(f"amdclang++ in this environment does not support {arch}")


def _states(arch, cfg_path):
    """Derived solution states for one config, cached per (arch, config)."""
    key = (arch, cfg_path)
    if key not in _STATES:
        _require_toolchain(arch)
        # config_harness imports bare: conftest.py puts characterization/_codegen
        # on sys.path (it has no __init__.py).
        from config_harness import derive_states

        # CPU-only derivation, no assembler. 4 solutions is enough to show the
        # value does not vary within a config, and keeps these tests fast.
        _STATES[key] = derive_states(cfg_path, arch=arch, limit_solutions=4)
    return _STATES[key]


def _dense(arch):
    return _states(arch, ARCH_CONFIGS[arch][0])


def _sparse(arch):
    return _states(arch, ARCH_CONFIGS[arch][1])


@pytest.mark.parametrize("arch", ARCHS)
def test_enabled_for_a_dense_solution(arch):
    """A non-sparse kernel gets the stamp."""
    states = _dense(arch)
    assert states, f"expected >=1 derived solution from the {arch} dense config"
    for st in states:
        assert st["ProblemType"]["Sparse"] == 0, "config is meant to be dense"
        assert st[_KEY] is True


@pytest.mark.parametrize("arch", ARCHS)
def test_enabled_for_a_sparse_solution(arch):
    """A sparse kernel gets the stamp too -- the case that used to derive False,
    so this pair is what shows the flag no longer tracks Sparse."""
    states = _sparse(arch)
    assert states, f"expected >=1 derived solution from the {arch} sparse config"
    for st in states:
        assert st["ProblemType"]["Sparse"] != 0, "config is meant to be sparse"
        assert st[_KEY] is True


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
