# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the skinny MXFP8 GEMM kernels of the tensilelite_codegen bindings."""

import numpy as np
import pytest
import skinny_gemm_support as support
import tensilelite_codegen as codegen
import tensilelite_jit as jit

ELF_MAGIC = b"\x7fELF"

KERNEL_NAMES = [
    (
        {},
        "gemm_BM1_TN1_LANES_K64_WAVES4_UP_ITERS0_PARTIAL_STEPS0_PF2_GROUP1_"
        "NUM_XCDS1_GRID_3D1",
    ),
    (
        {"prefetch": 1, "upIters": 1},
        "gemm_BM1_TN1_LANES_K64_WAVES4_UP_ITERS1_PARTIAL_STEPS0_PF1_GROUP1_"
        "NUM_XCDS1_GRID_3D1",
    ),
    (
        {
            "tn": 4,
            "lanesK": 16,
            "waves": 1,
            "prefetch": 2,
            "upIters": 0,
            "partialSteps": 2,
            "grid3d": False,
        },
        "gemm_BM1_TN4_LANES_K16_WAVES1_UP_ITERS0_PARTIAL_STEPS2_PF2_GROUP1_"
        "NUM_XCDS1_GRID_3D0",
    ),
    (
        {"group": 4, "numXcds": 8},
        "gemm_BM1_TN1_LANES_K64_WAVES4_UP_ITERS0_PARTIAL_STEPS0_PF2_GROUP4_"
        "NUM_XCDS8_GRID_3D1",
    ),
]

INVALID_KNOBS = [
    {"lanesK": 32},
    {"lanesK": 0},
    {"bm": 0},
    {"tn": 0},
    {"waves": 0},
    {"waves": 17},
    {"prefetch": -1},
    {"upIters": -1},
    {"partialSteps": 3},
    {"partialSteps": -1},
    {"group": 0},
    {"numXcds": 0},
]


def test_config_defaults_and_keywords():
    config = codegen.SkinnyGEMMConfig()
    assert (config.bm, config.tn, config.lanesK, config.waves) == (1, 1, 64, 4)
    assert (config.prefetch, config.upIters, config.partialSteps) == (2, 0, 0)
    assert (config.group, config.numXcds, config.grid3d) == (1, 1, True)

    config = codegen.SkinnyGEMMConfig(tn=4, grid3d=False)
    assert (config.tn, config.grid3d, config.bm) == (4, False, 1)
    config.waves = 2
    config.numXcds = 8
    assert (config.waves, config.numXcds) == (2, 8)
    with pytest.raises(TypeError):
        codegen.SkinnyGEMMConfig(1)


@pytest.mark.parametrize(("knobs", "expected"), KERNEL_NAMES)
def test_kernel_name(knobs, expected):
    assert codegen.skinnyGEMMKernelName(codegen.SkinnyGEMMConfig(**knobs)) == expected


def test_emit_skinny_gemm_is_not_implemented_yet():
    with pytest.raises(ValueError, match="not implemented"):
        codegen.emitSkinnyGEMM(codegen.SkinnyGEMMConfig())


@pytest.mark.parametrize("knobs", INVALID_KNOBS)
def test_invalid_knobs(knobs):
    config = codegen.SkinnyGEMMConfig(**knobs)
    with pytest.raises(ValueError, match=next(iter(knobs))):
        codegen.skinnyGEMMKernelName(config)
    with pytest.raises(ValueError, match=next(iter(knobs))):
        codegen.emitSkinnyGEMM(config)


def test_unsupported_arch():
    with pytest.raises(ValueError):
        codegen.emitSplitKReduce("gfx900")


def test_emit_split_k_reduce_assembles():
    asm = codegen.emitSplitKReduce()
    assert ".globl splitk_reduce" in asm
    assert ".kernarg_segment_size: 56" in asm
    assert jit.assembleToHSACO(asm, support.TARGET)[:4] == ELF_MAGIC


def test_e4m3fn_round_trips_every_code():
    codes = np.arange(256, dtype=np.uint8)
    nan = np.isnan(support.decodeE4M3FN(codes))
    assert list(codes[nan]) == [0x7F, 0xFF]
    assert (
        support.encodeE4M3FN(support.decodeE4M3FN(codes))[~nan] == codes[~nan]
    ).all()


def test_e4m3fn_known_values():
    known = {
        1.0: 0x38,
        448.0: 0x7E,
        2.0**-9: 0x01,
        2.0**-10: 0x00,
        3 * 2.0**-10: 0x02,
        464.0: 0x7E,
        465.0: 0x7F,
        -465.0: 0xFF,
        float("inf"): 0x7F,
        float("-inf"): 0xFF,
        float("nan"): 0x7F,
        -1e-30: 0x80,
        0.0: 0x00,
    }
    encoded = support.encodeE4M3FN(np.array(list(known)))
    assert list(encoded) == list(known.values())


def test_e4m3fn_ties_to_even():
    # 1.0625 lies between codes 0x38 (1.0) and 0x39 (1.125); 1.1875 between
    # 0x39 and 0x3A (1.25).
    encoded = support.encodeE4M3FN(np.array([1.0625, 1.1875, -1.0625, -1.1875]))
    assert list(encoded) == [0x38, 0x3A, 0xB8, 0xBA]


def test_pre_swizzle_scales_matches_the_byte_formula():
    rng = np.random.default_rng(0)
    scales = rng.integers(0, 256, size=(64, 16), dtype=np.uint8)
    swizzled = support.preSwizzleScales(scales)
    assert swizzled.size == support.scaleExtent(64, 16 * support.MX_BLOCK)
    for row in range(64):
        for col in range(16):
            tile = (row // 32) * 2 + col // 8
            r, c = row % 32, col % 8
            byte = (c % 4) * 64 + (r % 16) * 4 + ((c // 4) % 2) * 2 + (r // 16) % 2
            assert swizzled[tile * 256 + byte] == scales[row, col]


def test_reduce_reference_sums_in_split_order():
    # In order the sum is 1.0; in reverse order it would be 0.0.
    ws = np.array([1e8, 1.0, -1e8, 1.0], dtype=np.float32)
    assert support.reduceReference(ws, 4, 1, 1)[0, 0] == np.float32(1.0)


@pytest.fixture(scope="module")
def reduceKernel():
    if not support.gpuAvailable():
        pytest.skip("no HIP device or hip-python is not usable")
    hsaco = jit.assembleToHSACO(codegen.emitSplitKReduce(), support.TARGET)
    with support.loadKernel(hsaco, support.REDUCE_KERNEL_NAME) as kernel:
        yield kernel


REDUCE_CASES = [
    (1, 1, 0, 1),
    (5, 3, 4, 2),
    (256, 2, 0, 3),
    (257, 3, 0, 4),
    (300, 4, 12, 5),
    (1000, 2, 4, 7),
    (64, 7, 8, 16),
    (1, 5120, 0, 2),
    (5120, 1, 0, 2),
]


def runReduce(kernel, m, n, ldD, splits, seed):
    rng = np.random.default_rng(seed)
    exponent = rng.integers(-8, 9, size=splits * m * n)
    ws = (rng.standard_normal(splits * m * n) * 2.0**exponent).astype(np.float32)
    dBits, _ = support.makeOutputs(m, n, ldD, splits)
    with support.DeviceBuffer(dBits) as dBuf, support.DeviceBuffer(ws) as wsBuf:
        support.launchReduce(kernel, dBuf, wsBuf, ldD, m, n, splits)
        dBuf.download(dBits)
    return ws, dBits


@pytest.mark.parametrize(("m", "n", "ldDPad", "splits"), REDUCE_CASES)
def test_splitk_reduce(reduceKernel, m, n, ldDPad, splits):
    ldD = support.alignUp(m, 4) + ldDPad
    ws, dBits = runReduce(reduceKernel, m, n, ldD, splits, seed=m * 31 + n)
    expected = support.reduceReference(ws, splits, m, n)
    label = f"m={m} n={n} ldD={ldD} splits={splits}"
    # Bit exact against the truncation of the in-order f32 sum, then through the
    # tolerance based check that the GEMM tests use.
    actual = support.dColumns(dBits, m, n, ldD).T
    np.testing.assert_array_equal(
        actual, support.truncateToBf16(expected), err_msg=label
    )
    support.verifyD(dBits, expected, m, n, ldD, label)


def test_splitk_reduce_truncates_instead_of_rounding(reduceKernel):
    m, n, ldD, splits = 1000, 4, 1000, 2
    ws, dBits = runReduce(reduceKernel, m, n, ldD, splits, seed=7)
    expected = support.reduceReference(ws, splits, m, n)
    nearest = np.rint(expected.view(np.uint32).astype(np.float64) / 65536.0)
    actual = support.dColumns(dBits, m, n, ldD).T
    assert (nearest.astype(np.uint32) != actual).any()
