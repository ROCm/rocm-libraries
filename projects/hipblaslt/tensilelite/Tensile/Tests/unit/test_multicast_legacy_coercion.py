#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Regression tests for the legacy bool -> int Multicast coercion.
#
# Multicast became an int solution parameter (valid values [-1, 0, 1]: -1 auto,
# 0 off, 1 on). But 600+ shipped library-logic artifacts still serialize the
# pre-existing derived value as `Multicast: false` (BOOL). Because msgpack
# serializes bool and int as different wire types, a bool value trips the strict
# create-library type gate (raiseIfTypeMismatches) -> ConfigTypeError, and would
# cause std::bad_cast at runtime.
#
# Fix: on the serialized-state loading path (raiseProblemTypeOnTypeMismatch=
# False), coerce a bool Multicast to the tri-state int (false -> 0 "off",
# true -> 1 "on"; never -1 "auto"), so both the type gate and the emitted
# msgpack see an int. The config/derivation path now emits an int Multicast on
# all branches directly (no post-derivation coercion needed).
#
# Usage:
#   pytest test_multicast_legacy_coercion.py -v
################################################################################

import copy
import importlib

import pytest

pytestmark = pytest.mark.unit

from Tensile.SolutionStructs.Solution import (
    coerceLegacyMulticastType,
    validateParameterTypes,
)

S = importlib.import_module("Tensile.SolutionStructs.Solution")
Solution = S.Solution


# --- pure coercion helper --------------------------------------------------

class TestCoerceHelper:
    def test_false_maps_to_int_zero_off(self):
        state = {"Multicast": False}
        coerceLegacyMulticastType(state)
        assert state["Multicast"] == 0
        assert type(state["Multicast"]) is int
        # Must be "off" (0), never "auto" (-1).
        assert state["Multicast"] != -1

    def test_true_maps_to_int_one_on(self):
        state = {"Multicast": True}
        coerceLegacyMulticastType(state)
        assert state["Multicast"] == 1
        assert type(state["Multicast"]) is int

    def test_int_values_untouched(self):
        for v in (-1, 0, 1):
            state = {"Multicast": v}
            coerceLegacyMulticastType(state)
            assert state["Multicast"] == v
            assert type(state["Multicast"]) is int

    def test_absent_key_is_noop(self):
        state = {"SomethingElse": 5}
        coerceLegacyMulticastType(state)
        assert "Multicast" not in state


# --- coercion clears the strict type gate ----------------------------------

class TestGateIntegration:
    def test_bool_multicast_flagged_before_coercion(self):
        """A bool Multicast is what the create-library gate flags as
        'found bool ... expected int'."""
        records = validateParameterTypes({"Multicast": False})
        assert records, "expected a type-mismatch record for bool Multicast"
        (name, actual, expected), _value, _src = records[0]
        assert name == "Multicast"
        assert actual == "bool"
        assert "int" in expected

    def test_no_mismatch_after_coercion(self):
        """After coercion the same state passes validateParameterTypes clean,
        i.e. raiseIfTypeMismatches() would not fire."""
        for legacy in (False, True):
            state = {"Multicast": legacy}
            coerceLegacyMulticastType(state)
            assert validateParameterTypes(state) == []


# --- end-to-end through the real Solution constructor ----------------------
#
# Mirrors the create-library / library-logic loading path
# (raiseProblemTypeOnTypeMismatch=False), which is where the CI build broke.
# Requires amdclang++ for the ISA-info map (same as the other Solution
# characterization tests); skipped gracefully if the toolchain is absent.

@pytest.fixture(scope="module")
def _toolchain():
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults

    try:
        cxx = validateToolchain("amdclang++")
        bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"amdclang++ toolchain unavailable: {e}")
    isa = gfxToIsa("gfx950")
    iim = makeIsaInfoMap([isa], cxx)
    assembler = makeAssemblyToolchain(cxx, bundler, "default").assembler
    return iim, assembler


@pytest.fixture(scope="module")
def _gp_assigned(_toolchain):
    from Tensile.Common.GlobalParameters import globalParameters, assignGlobalParameters
    from Tensile.Common.ValidParameters import validParameters

    iim, _ = _toolchain
    saved_gp = copy.deepcopy(dict(globalParameters))
    saved_vp = copy.deepcopy(dict(validParameters))
    assignGlobalParameters({}, iim)
    yield
    globalParameters.clear()
    globalParameters.update(saved_gp)
    validParameters.clear()
    validParameters.update(saved_vp)


def _make_solution(iim, assembler, multicast, *, loadingSerializedState):
    """Build & derive a minimal gfx950 MX FP8 solution carrying a bool Multicast.

    ``loadingSerializedState=True`` drives the library-logic/artifact path
    (raiseProblemTypeOnTypeMismatch=False); False drives the config path.
    """
    from Tensile.BenchmarkProblems import matrixInstructionToMIParameters

    isa = list(iim.keys())[0]
    params = {
        "ProblemType": {
            "OperationType": "GEMM",
            "DataType": "F8",
            "DestDataType": "s",
            "ComputeDataType": "s",
            "HighPrecisionAccumulate": True,
            "MXBlockA": 32,
            "MXBlockB": 32,
            "TransposeA": True,
            "TransposeB": False,
            "UseBeta": True,
            "Batched": True,
        },
        "ISA": isa,
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "WorkGroup": [16, 16, 1],
        "WavefrontSize": 64,
        "DepthU": 256,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 3,
        "DirectToLds": 1,
        "StaggerU": 0,
        "StreamK": 3,
        "UseSubtileImpl": True,
        "LocalReadVectorWidth": -1,
        "GlobalReadVectorWidthA": 16,
        "GlobalReadVectorWidthB": 16,
        "TransposeLDS": 1,
        "LdsPadA": -1,
        "LdsPadB": -1,
        "LdsBlockSizePerPadA": -1,
        "LdsBlockSizePerPadB": -1,
        "1LDSBuffer": -1,
        "VectorWidthA": -1,
        "VectorWidthB": -1,
        "StoreVectorWidth": -1,
        "SourceSwap": True,
        "ExpandPointerSwap": True,
        "GlobalSplitU": 1,
        "InnerUnroll": 1,
        "DebugStreamK": 0,
        # The legacy artifact wire type under test: a serialized bool.
        "Multicast": multicast,
    }
    mi = params["MatrixInstruction"]
    params.update(matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], params["ProblemType"], params["WorkGroup"], iim))

    return Solution(
        params, False, False, False, assembler, iim,
        raiseProblemTypeOnTypeMismatch=not loadingSerializedState,
    )


class TestSolutionConstructorPath:
    @pytest.mark.parametrize("legacy,expected_int", [(False, 0), (True, 1)])
    def test_library_logic_path_coerces_to_int(
        self, _toolchain, _gp_assigned, legacy, expected_int
    ):
        """On the serialized-state path a bool Multicast is coerced to the
        tri-state int, and the state passes the strict type gate clean."""
        iim, assembler = _toolchain
        sol = _make_solution(iim, assembler, legacy, loadingSerializedState=True)
        mc = sol._state["Multicast"]
        assert type(mc) is int, f"Multicast must be int on the load path, got {type(mc)}"
        assert mc == expected_int
        assert mc != -1  # coerced off/on, never auto
        # The gate the create-library path runs must see no Multicast mismatch.
        records = validateParameterTypes(sol._state)
        assert all(rec[0][0] != "Multicast" for rec in records), records

    def test_config_path_derives_int(self, _toolchain, _gp_assigned):
        """The config/derivation path now emits an int Multicast on all branches
        (the post-derivation coercion was removed as redundant): a Multicast=False
        input derives to int 0 (the mc==0 force-off branch), not a bool."""
        iim, assembler = _toolchain
        sol = _make_solution(iim, assembler, False, loadingSerializedState=False)
        mc = sol._state["Multicast"]
        assert type(mc) is int, f"config path must derive int, got {type(mc)}"
        assert mc == 0, mc


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
