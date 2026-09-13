# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side custom-kernel plumbing coverage: the pure/near-pure helpers the
custom-kernel path adds across Solution, Naming, BenchmarkProblems, and the
Toolchain assembler.

These live in a direct ``Tests/unit`` module (not under ``characterization/``)
so the coverage lane credits them against the changed source lines.
"""

import pytest

from Tensile.BenchmarkProblems import _hashableProblemTypeKV
from Tensile.Common.DataType import DataType
from Tensile.Common.Utilities import deriveWaveParams
from Tensile.SolutionStructs.Naming import _getName, getKernelFileBase
from Tensile.SolutionStructs.Solution import Solution
from Tensile.Toolchain.Component import Assembler

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Solution._assignCustomKernelParameters
# --------------------------------------------------------------------------- #


def _ck_state(**over):
    problem_type = {
        "ComputeDataType": DataType("s"),
        "DestDataType": DataType("s"),
        "UseBias": False,
        "Gradient": False,
    }
    problem_type.update(over.pop("ProblemType", {}))
    state = {
        "CustomKernel": {"name": "k", "macrotile": [128, 256, 64], "threads": [256, 1, 1]},
        "StreamK": 0,
        "StreamKAtomic": 0,
        "GlobalSplitUAlgorithm": "",
        "ProblemType": problem_type,
        "DirectToLds": 0,
        "MatrixInstruction": [16, 16, 16, 1],
        "WavefrontSize": 64,
    }
    state.update(over)
    return state


def test_assign_custom_kernel_params_basic_derivation():
    state = _ck_state()
    Solution._assignCustomKernelParameters(state)

    assert state["MacroTile0"] == 128
    assert state["MacroTile1"] == 256
    assert state["DepthU"] == 64
    assert state["NumThreads"] == 256
    assert state["NumElementsPerThread"] == (128 * 256) // 256
    assert state["CUOccupancy"] == -1
    assert state["MathClocksUnrolledLoop"] == 0
    assert state["PackedC0IndicesX"] == []
    assert state["ThreadTile0"] == 0 and state["ThreadTile1"] == 0
    assert state["LocalSplitU"] == 1
    assert state["GlobalReadVectorWidthA"] == 1
    assert state["GlobalReadVectorWidthB"] == 1
    assert state["StoreVectorWidth"] == 1
    assert state["_GlobalAccumulation"] is None  # GlobalSplitUAlgorithm == ""


def test_assign_custom_kernel_params_enable_mi_sets_wave_params():
    state = _ck_state()
    Solution._assignCustomKernelParameters(state)

    assert state["EnableMatrixInstruction"] is True
    assert isinstance(state["MIWaveTile"], list)
    assert isinstance(state["MIWaveGroup"], list)


def test_assign_custom_kernel_params_no_mi_zeroes_wave_params():
    state = _ck_state(MatrixInstruction=[])
    Solution._assignCustomKernelParameters(state)

    assert state["EnableMatrixInstruction"] is False
    assert state["MIWaveTile"] == [0, 0]
    assert state["MIWaveGroup"] == [0, 0]


@pytest.mark.parametrize("dtl,expect_a,expect_b", [
    (0, False, False),
    (1, True, True),
    (2, True, False),
    (3, False, True),
])
def test_assign_custom_kernel_params_direct_to_lds(dtl, expect_a, expect_b):
    state = _ck_state(DirectToLds=dtl)
    Solution._assignCustomKernelParameters(state)
    assert state["DirectToLdsA"] is expect_a
    assert state["DirectToLdsB"] is expect_b


def test_assign_custom_kernel_params_streamk_partials_accumulation():
    state = _ck_state(StreamK=2, StreamKAtomic=0)
    Solution._assignCustomKernelParameters(state)
    assert state["_GlobalAccumulation"] == "PartialsBuffer"


def test_assign_custom_kernel_params_single_buffer_accumulation():
    # SingleBuffer only sets accumulation when compute dtype != dest dtype.
    state = _ck_state(
        GlobalSplitUAlgorithm="SingleBuffer",
        ProblemType={"ComputeDataType": DataType("s"), "DestDataType": DataType("h")},
    )
    Solution._assignCustomKernelParameters(state)
    assert state["_GlobalAccumulation"] == "SingleBuffer"


def test_assign_custom_kernel_params_multiple_buffer_accumulation():
    state = _ck_state(GlobalSplitUAlgorithm="MultipleBuffer")
    Solution._assignCustomKernelParameters(state)
    assert state["_GlobalAccumulation"] == "MultipleBuffer"


def test_assign_custom_kernel_params_mbsk_accumulation():
    state = _ck_state(GlobalSplitUAlgorithm="MultipleBufferSingleKernel")
    Solution._assignCustomKernelParameters(state)
    assert state["_GlobalAccumulation"] == "MultipleBufferSingleKernel"


def test_assign_custom_kernel_params_bias_gradient_workspace():
    state = _ck_state(
        ProblemType={
            "ComputeDataType": DataType("s"), "DestDataType": DataType("s"),
            "UseBias": True, "Gradient": True,
        },
    )
    state["CustomKernel"]["workspaceSizePerElemBias"] = 4
    Solution._assignCustomKernelParameters(state)
    assert state["_WorkspaceSizePerElemBias"] == 4


# --------------------------------------------------------------------------- #
# Naming: custom-kernel name short-circuits
# --------------------------------------------------------------------------- #


def test_get_kernel_file_base_custom_mapping_name():
    assert getKernelFileBase(False, {"CustomKernel": {"name": "my_ck"}}) == "my_ck"


def test_get_kernel_file_base_legacy_name():
    assert getKernelFileBase(False, {"CustomKernelName": "legacy_ck"}) == "legacy_ck"


def test_get_kernel_file_base_generated_falls_back_to_legacy():
    # A "generated" CustomKernel mapping is not treated as handwritten, so the
    # legacy CustomKernelName wins.
    kernel = {"CustomKernel": {"name": "gen", "generated": True}, "CustomKernelName": "legacy"}
    assert getKernelFileBase(False, kernel) == "legacy"


def test_get_name_custom_mapping_name():
    assert _getName({"CustomKernel": {"name": "ck_map"}}, frozenset(), False, False) == "ck_map"


def test_get_name_legacy_name():
    assert _getName({"CustomKernelName": "ck_legacy"}, frozenset(), False, False) == "ck_legacy"


# --------------------------------------------------------------------------- #
# BenchmarkProblems._hashableProblemTypeKV
# --------------------------------------------------------------------------- #


def test_hashable_kv_list_becomes_tuple():
    assert _hashableProblemTypeKV("Index", [0, 1, 2]) == ("Index", (0, 1, 2))


def test_hashable_kv_hashable_passthrough():
    assert _hashableProblemTypeKV("DataType", "s") == ("DataType", "s")


def test_hashable_kv_unhashable_uses_repr():
    key, value = _hashableProblemTypeKV("Meta", {"a": 1})
    assert key == "Meta"
    assert value == repr({"a": 1})


# --------------------------------------------------------------------------- #
# Toolchain.Component.Assembler._retargetAssemblySource (direct-unit coverage)
# --------------------------------------------------------------------------- #


def test_retarget_rewrites_mismatched_target(tmp_path):
    src = tmp_path / "k.s"
    src.write_text(
        '\t.amdgcn_target "amdgcn-amd-amdhsa--gfx900:sramecc+:xnack-"\n'
        "\tamdhsa.target: amdgcn-amd-amdhsa--gfx900:sramecc+:xnack-\n"
        "s_endpgm\n"
    )
    Assembler._retargetAssemblySource("gfx942", str(src))
    updated = src.read_text()
    assert '.amdgcn_target "amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-"' in updated
    assert "amdhsa.target: amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-" in updated


def test_retarget_leaves_matching_target_untouched(tmp_path):
    src = tmp_path / "k.s"
    original = '\t.amdgcn_target "amdgcn-amd-amdhsa--gfx942"\ns_endpgm\n'
    src.write_text(original)
    mtime_before = src.stat().st_mtime_ns
    Assembler._retargetAssemblySource("gfx942", str(src))
    assert src.read_text() == original
    assert src.stat().st_mtime_ns == mtime_before  # no rewrite -> no write


def test_retarget_missing_source_does_not_raise():
    # Opportunistic rewrite: an unreadable/missing source is left alone rather
    # than crashing before the real assembler invocation.
    Assembler._retargetAssemblySource("gfx942", "/no/such/file.s")


# --------------------------------------------------------------------------- #
# deriveWaveParams: non-perfect-square wave count exercises the wgM search loop
# --------------------------------------------------------------------------- #


def test_derive_wave_params_non_square_wave_count():
    # num_threads=320, wavefront=64 -> num_waves=5 (not a perfect square), so
    # the wgM-decrement loop runs until wgM divides num_waves (2 -> 1).
    wave_group, wave_tile = deriveWaveParams([16, 16, 16, 1], 320, [256, 256], 64)
    assert wave_group == [1, 5]
    assert wave_tile == [max(1, 256 // (16 * 1)), max(1, 256 // (16 * 5))]


def test_derive_wave_params_square_wave_count():
    # num_threads=256, wavefront=64 -> num_waves=4 (perfect square) -> wgM=2.
    wave_group, _ = deriveWaveParams([16, 16, 16, 1], 256, [256, 256], 64)
    assert wave_group == [2, 2]
