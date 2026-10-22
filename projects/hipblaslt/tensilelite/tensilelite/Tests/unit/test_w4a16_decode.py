# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatch bounds and universal ABI for generated runtime-group decode."""
import pytest
import yaml
from test_w4a16_block_dequant import (toolchain, _emit, _emit_solution, FP16_UNSIGNED,
                                      GFX1151_LOGIC, needs_logic)
from test_w4a16_decode_codegen import solution as decode_solution, _size_bounds

pytestmark = pytest.mark.unit
SCHEDULES = {
    "_W4": [4, 4, 1, 128, 1, True, True],
    "_W4_U1_A4": [4, 1, 4, 128, 1, False, False],
    "_W4_NativePerm": [4, 4, 1, 128, 1, False, True],
}


def metadata(source):
    return yaml.safe_load(source.split(".amdgpu_metadata", 1)[1].split(".end_amdgpu_metadata", 1)[0])["amdhsa.kernels"][0]


@pytest.fixture(scope="module")
def generated_matrix(toolchain):
    from Tensile.CustomKernels import _buildCustomKernelFromMetadata

    kernel, source = _emit(toolchain, blockSize=32, depthU=64, zeroPoint=True,
                           problemType=dict(FP16_UNSIGNED, ScaleBlockSizesA=[32, 64, 128]))
    general = metadata(source)
    return general, _buildCustomKernelFromMetadata(
        "matrix", {"amdhsa.kernels": [general]}, kernel)


@pytest.mark.parametrize("suffix", SCHEDULES)
def test_decode_selection_bounds(toolchain, suffix):
    config = decode_solution(toolchain, decode=SCHEDULES[suffix], zeroPoint=True)
    assert _size_bounds(config) == [
        {"type": "SizeEqual", "index": 2, "value": 1},
        {"type": "SizeGreaterThan", "index": 3, "value": 0},
        {"type": "SizeEqual", "index": 1, "value": 1},
    ]
    assert config["AssertSummationElementMultiple"] == 256
    for key in ("SupportUserGSU", "SupportCustomWGM", "SupportCustomStaggerU"):
        assert not config["InternalSupportParams"][key]


@pytest.mark.parametrize("suffix", SCHEDULES)
def test_decode_universal_arguments_match_matrix_kernel(toolchain, suffix, generated_matrix):
    kernel, source = _emit_solution(toolchain, decode_solution(toolchain, decode=SCHEDULES[suffix], zeroPoint=True))
    decode, general = metadata(source), generated_matrix[0]
    assert decode[".kernarg_segment_size"] == general[".kernarg_segment_size"] == 168
    assert decode[".kernarg_segment_align"] == general[".kernarg_segment_align"]
    assert [(a[".offset"], a[".size"]) for a in decode[".args"]] == [(a[".offset"], a[".size"]) for a in general[".args"]]
    assert decode[".args"][-1][".offset"] == 160
    assert [a[".name"] for a in decode[".args"]] == [a[".name"] for a in general[".args"]]
    assert kernel["CustomKernel"]["args"] == generated_matrix[1]["args"]
    assert kernel["CustomKernel"]["grid"] == ["TilesXYBatchGSU", "One", "One"]
    assert kernel["CustomKernel"]["macrotile"] == [4, 1, 256]


@needs_logic
def test_unsigned_symmetric_logic_preserves_shapes_and_selects_small_n_decode():
    logic_dir = GFX1151_LOGIC / "Equality"
    asymmetric = yaml.safe_load((logic_dir / "gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_Q27B.yaml").read_text())
    symmetric = yaml.safe_load((logic_dir / "gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGU8_Q27B.yaml").read_text())
    original_shapes = {tuple(row[0]) for row in asymmetric["ExactLogic"]}
    assert original_shapes <= {tuple(row[0]) for row in symmetric["ExactLogic"]}
    for shape, selection in symmetric["ExactLogic"]:
        if tuple(shape) not in original_shapes:
            continue
        reference = list(shape)
        if shape[1] in (2, 3, 4):
            reference[1] = 1
        old_index = next(row[1][0] for row in asymmetric["ExactLogic"] if row[0] == reference)
        expected = list(asymmetric["Solutions"][old_index].get("W4A16Decode", []))
        if expected and shape[1] in (2, 3, 4):
            expected[4] = 4
        assert symmetric["Solutions"][selection[0]].get("W4A16Decode", []) == expected
    for shape, selection in symmetric["ExactLogic"]:
        selected = symmetric["Solutions"][selection[0]]
        schedule = selected.get("W4A16Decode", [])
        if schedule and schedule[:2] == [4, 4]:
            expected_prefetch = 4 if shape[1] == 1 and shape[3] % 1024 == 0 else 1
            assert selected.get("W4A16DecodePrefetch", 1) == expected_prefetch
    assert symmetric["ProblemType"]["ScaleZeroPointA"] is False
    assert symmetric["ProblemType"]["Int4EncodingA"] == "UnsignedBias8"
    assert symmetric["ProblemType"]["ScaleBlockSizesA"] == [32, 64, 128]


@pytest.mark.parametrize("suffix", SCHEDULES)
def test_small_n_decode_bounds_and_abi(toolchain, suffix):
    params = list(SCHEDULES[suffix])
    original, source = _emit_solution(toolchain, decode_solution(toolchain, decode=params))
    params[4] = 4
    small, small_source = _emit_solution(toolchain, decode_solution(toolchain, decode=params))
    assert _size_bounds(small) == [
        {"type": "SizeEqual", "index": 2, "value": 1},
        {"type": "SizeGreaterThan", "index": 3, "value": 0},
        {"type": "SizeGreaterThan", "index": 1, "value": 0},
        {"type": "SizeLessThan", "index": 1, "value": 5},
    ]
    assert metadata(source)[".args"] == metadata(small_source)[".args"]
    assert original["CustomKernel"]["macrotile"] == small["CustomKernel"]["macrotile"]


def test_wide_decode_launch_geometry(toolchain):
    config, source = _emit_solution(toolchain, decode_solution(toolchain, decode=[2, 1, 4, 512, 1, True, True]))
    assert config["WorkGroup"] == [16, 1, 32]
    assert metadata(source)[".max_flat_workgroup_size"] == 512
    assert {"type": "SizeEqual", "index": 1, "value": 1} in _size_bounds(config)


@pytest.mark.parametrize("name", ["generated_prefill", "unrelated_kernel", ""])
def test_decode_bounds_do_not_apply_to_other_kernels(name):
    assert _size_bounds({"CustomKernelName": name}) == []


