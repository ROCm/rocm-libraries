# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest
import yaml

from Tensile.KernelWriterW4A16Decode import DecodeConfig, KernelWriterW4A16Decode
from test_w4a16_block_dequant import toolchain, _solution, _emit_solution, FP16_UNSIGNED

pytestmark = pytest.mark.unit


def solution(toolchain, **overrides):
    pt = dict(FP16_UNSIGNED, ScaleBlockSizesA=[32, 64, 128])
    pt.update(overrides.pop("problemType", {}))
    return _solution(toolchain, blockSize=32, depthU=256, problemType=pt,
                     W4A16Decode=overrides.pop("decode", [4, 4, 1, 128, 1, False, True]),
                     **overrides)


@pytest.mark.parametrize("zero_point", [False, True])
@pytest.mark.parametrize("decode", [[4, 4, 1, 128, 1, True, True],
                                    [4, 1, 4, 128, 4, False, False],
                                    [2, 1, 4, 512, 1, True, True]])
def test_decode_generates_universal_abi(toolchain, zero_point, decode):
    from Tensile.Common.Utilities import state
    from Tensile.Contractions import ProblemPredicate

    sol = solution(toolchain, zeroPoint=zero_point, decode=decode)
    kernel, src = _emit_solution(toolchain, sol)
    metadata = yaml.safe_load(src.split(".amdgpu_metadata", 1)[1].split(".end_amdgpu_metadata", 1)[0])["amdhsa.kernels"][0]
    args = {arg[".name"]: arg for arg in metadata[".args"]}
    assert args["groupSize"][".offset"] == (160 if zero_point else 152)
    assert args["groupSize"][".offset"] == args["batchOffsetB"][".offset"] + 8
    assert ("AddressScaleZeroA" in args) == zero_point
    assert kernel["CustomKernel"]["generated"]
    assert kernel["CustomKernel"]["macrotile"] == [decode[3] // 32, 1, 256]
    assert metadata[".max_flat_workgroup_size"] == decode[3]
    assert metadata[".group_segment_fixed_size"] == 0
    assert "v_dot2_f32_f16" in src and "v_wmma" not in src
    assert "ds_bpermute_b32" in src
    assert "s_mov_b64 exec_lo" not in src
    assert "label_Fallback:" in src
    predicates = [state(p) for p in ProblemPredicate.W4A16DecodePredicates(sol)]
    assert {"type": "SizeEqual", "index": 2, "value": 1} in predicates
    bound = {"type": "SizeEqual", "index": 1, "value": 1} if decode[4] == 1 else {
        "type": "SizeLessThan", "index": 1, "value": 5}
    assert bound in predicates


@pytest.mark.parametrize("bad", [[], [4], [4, 4, 1, 128, 1, 0, True],
                                 [8, 4, 1, 128, 1, False, True],
                                 [4, 4, 2, 128, 1, False, True],
                                 [4, 4, 1, 128, 8, False, True]])
def test_decode_rejects_invalid_schedule(bad):
    with pytest.raises(ValueError):
        DecodeConfig.fromSolution({"W4A16Decode": bad, "ProblemType": {"ScaleZeroPointA": False}})


@pytest.mark.parametrize("override", [{"Int4EncodingA": "Signed"}, {"TransposeB": True},
                                      {"GroupedGemm": True}, {"UseScaleAB": ""},
                                      {"ScaleBlockSizesA": []},
                                      {"HighPrecisionAccumulate": False},
                                      {"UseInitialStridesAB": True},
                                      {"UseInitialStridesCD": True},
                                      {"FusedGemmA2A": True},
                                      {"SwizzleTensorA": True},
                                      {"SwizzleTensorB": True},
                                      {"StochasticRounding": True},
                                      {"Activation": True, "ActivationType": "all"}])
def test_decode_rejects_unsupported_problems(toolchain, override):
    with pytest.raises(ValueError, match="W4A16Decode requires"):
        solution(toolchain, problemType=override)


@pytest.mark.parametrize("prefetch", [1, 4])
@pytest.mark.parametrize("zero_point", [False, True])
def test_decode_prefetch_resources_and_waits(toolchain, prefetch, zero_point):
    sol = solution(toolchain, zeroPoint=zero_point, W4A16DecodePrefetch=prefetch)
    kernel, src = _emit_solution(toolchain, sol)
    metadata = yaml.safe_load(src.split(".amdgpu_metadata", 1)[1].split(".end_amdgpu_metadata", 1)[0])["amdhsa.kernels"][0]
    assert metadata[".vgpr_count"] == (128 if prefetch == 4 else 104)
    assert ("_PF4_" in kernel["CustomKernel"]["name"]) == (prefetch == 4)
    if prefetch == 4:
        assert f"s_waitcnt vmcnt({(7 if zero_point else 6) * 3})" in src
        assert "v[116:119]" in src


@pytest.mark.parametrize("overrides", [{"prefetch": 2}, {"prefetch": True},
                                        {"prefetch": 4, "unroll": 1},
                                        {"prefetch": 4, "loadWidth": 2}])
def test_decode_rejects_invalid_prefetch(overrides):
    with pytest.raises(ValueError, match="prefetch"):
        DecodeConfig(**overrides)


def test_decode_scalar_tail_keeps_short_pair_and_long_serial_paths(toolchain):
    _, src = _emit_solution(toolchain, solution(toolchain))
    tail = src.split("label_TailW1:", 1)[1].split("label_Reduce:", 1)[0]
    assert "s_cmp_ge_u32 s7, 4096" in tail
    paired, serial = tail.split("label_ScalarTail:", 1)
    assert paired.count("global_load_b32") == 2
    assert "s_waitcnt vmcnt(3)" in paired
    assert serial.count("global_load_b32") == 1
