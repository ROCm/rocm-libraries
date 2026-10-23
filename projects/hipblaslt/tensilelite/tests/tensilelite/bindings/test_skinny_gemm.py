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


def kernelBody(asm, config):
    """The assembly with the kernel name replaced, so that kernels of
    different names compare equal."""
    return asm.replace(codegen.skinnyGEMMKernelName(config), "gemm")


def test_grid3d_ignores_group_and_xcds():
    knobs = {"lanesK": 16, "waves": 1, "prefetch": 1}
    plain = codegen.SkinnyGEMMConfig(**knobs)
    grouped = codegen.SkinnyGEMMConfig(group=4, numXcds=8, **knobs)
    assert plain.grid3d and grouped.grid3d
    assert kernelBody(codegen.emitSkinnyGEMM(plain), plain) == kernelBody(
        codegen.emitSkinnyGEMM(grouped), grouped
    )


@pytest.mark.parametrize("reordering", [{}, {"group": 4, "numXcds": 8}])
def test_persistent_launch_assembles(reordering):
    config = codegen.SkinnyGEMMConfig(
        lanesK=16, waves=1, prefetch=1, grid3d=False, **reordering
    )
    asm = codegen.emitSkinnyGEMM(config)
    assert "workgroup_id_y" not in asm
    assert "workgroup_id_z" not in asm
    assert jit.assembleToHSACO(asm, support.TARGET)[:4] == ELF_MAGIC


def test_emit_skinny_gemm_assembles():
    config = codegen.SkinnyGEMMConfig(lanesK=16, waves=1, prefetch=1)
    asm = codegen.emitSkinnyGEMM(config)
    assert f".globl {codegen.skinnyGEMMKernelName(config)}" in asm
    assert ".kernarg_segment_size: 104" in asm
    assert ".group_segment_fixed_size: 0" in asm
    assert jit.assembleToHSACO(asm, support.TARGET)[:4] == ELF_MAGIC


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


# The skinny GEMM tests of the aster driver.


def resolve(m, n, k, **knobs):
    """The problem of makeProblem with the driver's defaults: every knob from
    pickConfig unless given, tight leading dimensions."""
    resolved = {name: knobs.pop(name, None) for name in support.KNOB_OVERRIDE_NAMES}
    padding = knobs.pop("padding", (0, 0, 0))
    assert not knobs, f"unknown knobs {sorted(knobs)}"
    return support.makeProblem(m, n, k, padding=padding, **resolved)


def skipUnlessRunnable(*problems):
    """Skips the test if there is no GPU."""
    if not support.gpuAvailable():
        pytest.skip("no HIP device or hip-python is not usable")


def run(m, n, k, *, iters=10, seed=42, **knobs):
    """Resolves the problem and, if the emitter supports it, runs and verifies it."""
    p = resolve(m, n, k, **knobs)
    skipUnlessRunnable(p)
    support.runProblem(p, iters, seed)


# (BM, TN, LANES_K, WAVES, PF, M, N, K, SPK).
PARAMS = [
    (1, 4, 16, 1, 1, 1, 256, 1024, 1024),  # Decode M=1, path A, no split.
    (4, 1, 8, 2, 2, 4, 512, 2048, 2048),  # Path A, PF=2, 2 waves, LANES_K=8.
    (4, 2, 8, 2, 1, 4, 512, 2048, 2048),  # PF=1, NI = 4: 3 steady trips and the drain.
    (2, 1, 16, 1, 3, 2, 128, 9216, 9216),  # PF=3, NI = 9: 2 steady trips and the drain.
    (1, 4, 16, 1, 2, 2, 256, 8192, 2048),  # Path A, split-K x4, PF=2.
    (4, 4, 16, 2, 1, 512, 4, 2048, 2048),  # Path B (N skinny).
    (1, 4, 4, 1, 2, 256, 1, 4096, 1024),  # Path B, split-K x4, LANES_K=4.
    (4, 2, 16, 2, 1, 576, 8, 3072, 1024),  # Path B, split-K x3, reduce grid.x=3.
    (2, 2, 16, 2, 1, 8, 256, 6144, 3072),  # 2 waves, split-K x2, grid.y=4.
    (8, 1, 64, 4, 1, 8, 128, 4096, 4096),  # LANES_K=64 full-wave reduce.
    (4, 4, 16, 1, 1, 64, 64, 1024, 1024),  # m == n tie selects path A.
    # Tails: skinny % BM != 0 and/or wide % BN != 0.
    (2, 2, 16, 1, 1, 3, 100, 2048, 2048),  # Path A, skinny and wide tails.
    (4, 1, 8, 2, 2, 7, 131, 2048, 2048),  # Path A, small primes, PF=2.
    (4, 4, 16, 2, 1, 301, 5, 2048, 2048),  # Path B, skinny and wide tails.
    (2, 2, 16, 1, 1, 131, 7, 2048, 2048),  # Path B, small primes.
    (2, 3, 16, 1, 2, 3, 250, 8192, 2048),  # Path A tails with split-K x4, PF=2.
    (2, 2, 16, 1, 1, 301, 5, 4096, 2048),  # Path B tails, split-K x2, reduce grid.x=2.
    (4, 2, 64, 2, 1, 6, 67, 4096, 4096),  # Tails, LANES_K=64, 2 waves.
    (
        2,
        4,
        64,
        1,
        1,
        3,
        9,
        4096,
        4096,
    ),  # Tails with uniform column flags (one column group).
    (4, 2, 16, 2, 1, 1, 100, 2048, 2048),  # m=1 with BM=4, path A.
    (4, 2, 16, 2, 1, 100, 1, 2048, 2048),  # n=1 with BM=4, path B.
    (2, 2, 16, 1, 1, 1, 1, 1024, 1024),  # m == n == 1.
    # Single-dimension tails (BN = 8): only the skinny or only the wide dimension is ragged.
    (2, 2, 16, 1, 1, 3, 64, 2048, 2048),  # Path A, skinny-only tail.
    (2, 2, 16, 1, 1, 4, 100, 2048, 2048),  # Path A, wide-only tail.
    (2, 2, 16, 1, 1, 64, 3, 2048, 2048),  # Path B, skinny-only tail.
    (2, 2, 16, 1, 1, 100, 4, 2048, 2048),  # Path B, wide-only tail.
    # BN = 8, grid.x = 38, NI = 3: the bx/16 rotation term with NI >= 3, wide-only tail.
    (2, 2, 16, 1, 1, 2, 300, 3072, 3072),
    # Non-power-of-two BM and BN: the tile decode divides by non-power-of-two counts.
    (3, 2, 16, 1, 1, 7, 100, 2048, 2048),  # Path A, BM = 3, skinny and wide tails.
    (2, 3, 16, 2, 1, 2, 200, 2048, 2048),  # Path A, BN = 24, wide-only tail.
    (3, 3, 16, 1, 1, 100, 7, 2048, 2048),  # Path B, BM = 3, BN = 12, both tails.
    # PF = 0: no prefetch ring, a separate K loop.
    (1, 4, 16, 1, 0, 1, 256, 1024, 1024),  # PF=0, path A, aligned, no split.
    (4, 4, 16, 2, 0, 512, 4, 2048, 2048),  # PF=0, path B, aligned.
    (2, 2, 16, 1, 0, 3, 100, 2048, 2048),  # PF=0, path A, skinny and wide tails.
    (4, 4, 16, 2, 0, 301, 5, 2048, 2048),  # PF=0, path B, skinny and wide tails.
    (2, 4, 16, 1, 0, 2, 256, 8192, 2048),  # PF=0, path A, split-K x4.
    (4, 2, 64, 2, 0, 6, 67, 4096, 4096),  # PF=0, LANES_K=64 tails, 2 waves.
    (1, 4, 4, 1, 0, 256, 1, 4096, 1024),  # PF=0, path B, split-K x4, LANES_K=4.
    (3, 7, 16, 1, 0, 3, 56, 1024, 1024),  # PF=0 allocates (253 VGPRs), PF=1 does not.
    # Ring depth against the iteration count NI = spk/(64*LANES_K): the steady loop runs
    # NI/PF - 1 times and a drain consumes the last PF iterations without loading.
    (
        1,
        4,
        16,
        1,
        2,
        1,
        256,
        2048,
        2048,
    ),  # PF=2, NI = PF: no steady trip, only the drain.
    (
        2,
        3,
        16,
        1,
        2,
        2,
        192,
        4096,
        4096,
    ),  # PF=2, NI = 2*PF, the largest tile that fits.
    (1, 2, 16, 1, 3, 2, 128, 3072, 3072),  # PF=3, NI = PF.
    (1, 2, 16, 1, 4, 1, 256, 4096, 4096),  # PF=4, NI = PF, the largest tile that fits.
    (1, 2, 16, 1, 4, 1, 128, 8192, 8192),  # PF=4, NI = 2*PF.
    (1, 4, 16, 1, 2, 2, 256, 8192, 4096),  # PF=2, NI = 4, split-K x2.
    (1, 2, 16, 1, 4, 1, 256, 8192, 4096),  # PF=4, NI = PF, split-K x2.
    # Steady-loop trip counts 0, 1 and 2 (NI = PF, 2*PF, 3*PF) with split-K, both paths and
    # more than 16 tiles so the bx/16 term of the K rotation varies.
    (1, 4, 16, 1, 1, 1, 256, 2048, 1024),  # PF=1, NI = PF, split-K x2.
    (2, 2, 16, 1, 1, 301, 5, 6144, 3072),  # PF=1, NI = 3*PF, path B tails, split-K x2.
    (2, 2, 16, 1, 1, 2, 300, 6144, 3072),  # PF=1, NI = 3*PF, path A, split-K x2.
    (
        1,
        5,
        64,
        1,
        1,
        323,
        1,
        8192,
        8192,
    ),  # PF=1, NI = 2*PF, LANES_K=64, path B wide tail.
    (1, 4, 16, 1, 2, 256, 2, 4096, 2048),  # PF=2, NI = PF, path B, split-K x2.
    (
        1,
        4,
        16,
        1,
        2,
        2,
        301,
        12288,
        6144,
    ),  # PF=2, NI = 3*PF, path A wide tail, split-K x2.
    (
        1,
        4,
        64,
        1,
        2,
        257,
        1,
        8192,
        8192,
    ),  # PF=2, NI = PF, LANES_K=64, path B wide tail.
    (2, 1, 16, 1, 3, 128, 2, 6144, 3072),  # PF=3, NI = PF, path B, split-K x2.
    (2, 1, 16, 1, 3, 3, 101, 6144, 6144),  # PF=3, NI = 2*PF, path A tails.
    (1, 2, 16, 1, 3, 200, 2, 9216, 9216),  # PF=3, NI = 3*PF, path B.
    # NI not a multiple of PF: the steady loop leaves a remainder of 1 to 2*PF-1 iterations.
    (1, 4, 16, 1, 2, 1, 256, 3072, 3072),  # PF=2, NI = 3.
    (2, 1, 16, 1, 3, 2, 128, 4096, 4096),  # PF=3, NI = 4.
    (1, 2, 16, 1, 4, 1, 256, 6144, 6144),  # PF=4, NI = 6.
    # K tails (K not a multiple of 64*LANES_K, so the last iteration is a partial
    # step of dwordx4 loads and a few masked dword loads). NIfull is the
    # number of full iterations before the tail.
    (1, 4, 16, 1, 1, 1, 256, 1, 1024),  # K = 1.
    (1, 4, 16, 1, 1, 1, 256, 16, 1024),  # K = 16, one lane group.
    (2, 2, 16, 1, 1, 3, 100, 33, 1024),  # K = 33, a partial MX block.
    (1, 4, 16, 1, 1, 1, 256, 1000, 1024),  # K < KI, NIfull = 0.
    (2, 2, 16, 1, 1, 2, 128, 1023, 1024),  # K = KI - 1.
    (
        2,
        2,
        16,
        1,
        1,
        2,
        128,
        2049,
        2048,
    ),  # K = 2*KI + 1, split-K x2: a one-element last split.
    (1, 4, 64, 1, 1, 1, 256, 4095, 4096),  # LANES_K=64, K = KI - 1.
    (1, 4, 64, 1, 1, 256, 1, 4097, 8192),  # LANES_K=64, K = KI + 1, path B.
    (1, 4, 16, 1, 1, 1, 256, 8191, 8192),  # Odd K, NIfull = 7.
    (1, 4, 4, 1, 1, 1, 256, 1000, 1024),  # LANES_K=4.
    (2, 2, 8, 2, 1, 3, 100, 3000, 3072),  # LANES_K=8, 2 waves.
    (1, 4, 64, 1, 1, 1, 256, 5000, 8192),  # LANES_K=64, NIfull = 1.
    (
        2,
        2,
        16,
        1,
        1,
        2,
        128,
        1008,
        1024,
    ),  # ld == K: the last column ends at the buffer end.
    (2, 2, 16, 1, 1, 128, 2, 2032, 2048),  # ld == K on path B.
    # Lane classes of the tail step. With LANES_K >= 16 the K lanes form subgroups of
    # 8 lanes, so the 16-byte slot a = r // 16 of the iteration, r = K mod KI, is
    # j*32 + u*8 + i: subgroup j, chunk u (uStar, that of the one lane that
    # straddles k) and lane i of the subgroup.
    (2, 2, 16, 1, 1, 3, 100, 7, 1024),  # a = 0: j = 0, uStar = 0, r % 4 = 3.
    (2, 2, 16, 1, 1, 3, 100, 263, 1024),  # a = 16: j = 0, uStar = 2, r % 16 = 7.
    (2, 2, 16, 1, 1, 3, 100, 552, 1024),  # a = 34: j = 1, uStar = 0, r % 16 = 8.
    (2, 2, 16, 1, 1, 3, 100, 1002, 1024),  # a = 62: j = 1, uStar = 3, r % 4 = 2.
    (
        2,
        2,
        16,
        1,
        1,
        3,
        100,
        1015,
        1024,
    ),  # a = 63: j = 1, uStar = 3, r % 4 = 3, 7 bytes.
    (2, 2, 16, 1, 1, 3, 100, 1004, 1024),  # K % 4 = 0, K % 16 = 12.
    (2, 2, 16, 1, 1, 3, 100, 697, 1024),  # a = 43: j = 1, uStar = 1, r % 16 = 9.
    (2, 2, 16, 1, 1, 3, 100, 838, 1024),  # a = 52: j = 1, uStar = 2, r % 16 = 6.
    (2, 2, 16, 1, 1, 3, 100, 256, 1024),  # r = 256: chunks 2+ empty.
    (2, 2, 16, 1, 1, 3, 100, 528, 1024),  # r % 16 = 0: no straddling lane.
    (2, 2, 16, 1, 1, 100, 3, 263, 1024),  # Path B, uStar = 2.
    (2, 2, 16, 1, 1, 100, 3, 1002, 1024),  # Path B, uStar = 3.
    (2, 2, 16, 1, 1, 100, 3, 2064, 2048),  # Path B, r = 16, split-K x2.
    (2, 2, 16, 1, 0, 3, 100, 1002, 1024),  # PF=0, uStar = 3.
    (2, 2, 16, 1, 2, 3, 100, 1002, 1024),  # PF=2, uStar = 3.
    (1, 4, 4, 1, 1, 1, 256, 71, 256),  # LANES_K=4, uStar = 1.
    (1, 4, 4, 1, 1, 1, 256, 255, 256),  # LANES_K=4, r = KI - 1.
    (1, 4, 4, 1, 1, 1, 256, 128, 256),  # LANES_K=4, r = 2 chunks.
    (1, 4, 64, 1, 1, 1, 256, 1035, 4096),  # LANES_K=64, a = 64: j = 2, uStar = 0.
    (1, 4, 64, 1, 1, 1, 256, 3079, 4096),  # LANES_K=64, a = 192: j = 6, uStar = 0.
    (1, 4, 64, 1, 1, 1, 256, 2048, 4096),  # LANES_K=64, r % 16 = 0, subgroups 0-3 full.
    (1, 4, 64, 1, 1, 1, 256, 100, 4096),  # a = 6: j = 0, uStar = 0, chunks 1-3 zeroed.
    (1, 4, 64, 1, 1, 1, 256, 1717, 4096),  # a = 107: j = 3, uStar = 1, r % 16 = 5.
    (1, 4, 64, 1, 1, 1, 256, 2921, 4096),  # a = 182: j = 5, uStar = 2, r % 16 = 9.
    (
        1,
        4,
        64,
        1,
        1,
        1,
        256,
        1536,
        4096,
    ),  # a = 96, r % 16 = 0: subgroups 0-2 full, 3-7 empty.
    # Ring depth 0 to 4 against NIfull in 0 to 3 (and more) with a tail.
    (2, 2, 16, 1, 0, 3, 100, 3500, 4096),  # PF=0, NIfull = 3.
    (2, 2, 16, 1, 0, 3, 100, 500, 1024),  # PF=0, NIfull = 0.
    (2, 2, 16, 1, 1, 3, 100, 2100, 3072),  # PF=1, NIfull = 2.
    (2, 2, 16, 1, 2, 3, 100, 500, 1024),  # PF=2, NIfull = 0.
    (2, 2, 16, 1, 2, 3, 100, 1500, 2048),  # PF=2, NIfull = 1 < PF.
    (2, 2, 16, 1, 2, 3, 100, 2500, 3072),  # PF=2, NIfull = 2 = PF.
    (2, 2, 16, 1, 2, 3, 100, 3100, 4096),  # PF=2, NIfull = 3.
    (1, 4, 16, 1, 2, 1, 256, 7268, 8192),  # PF=2, NIfull = 7.
    (2, 1, 16, 1, 3, 3, 100, 1500, 2048),  # PF=3, NIfull = 1 < PF.
    (2, 1, 16, 1, 3, 3, 100, 2100, 3072),  # PF=3, NIfull = 2 < PF.
    (1, 2, 16, 1, 3, 2, 128, 8292, 9216),  # PF=3, NIfull = 8.
    (1, 2, 16, 1, 4, 1, 256, 1500, 2048),  # PF=4, NIfull = 1.
    (1, 2, 16, 1, 4, 1, 256, 3500, 4096),  # PF=4, NIfull = 3 < PF.
    (1, 2, 16, 1, 4, 1, 256, 6200, 7168),  # PF=4, NIfull = 6.
    # A short last split: it has fewer iterations (possibly none) than the others.
    (
        1,
        4,
        16,
        1,
        1,
        2,
        300,
        5000,
        2048,
    ),  # Split-K x3, the last has 904 elements, NIfull = 0.
    (
        1,
        4,
        16,
        1,
        2,
        2,
        300,
        5200,
        2048,
    ),  # Split-K x3, the last has NIfull = 1 and a tail.
    (2, 2, 16, 1, 1, 301, 5, 4097, 1024),  # Split-K x5, the last has 1 element, path B.
    (
        2,
        2,
        16,
        1,
        2,
        301,
        5,
        7200,
        3072,
    ),  # Short last split with NIfull = 1, PF=2, path B.
    (
        2,
        2,
        16,
        1,
        1,
        3,
        100,
        3072,
        2048,
    ),  # Short last split without partial steps (PARTIAL_STEPS = 0).
    # M/N tails together with K tails.
    (2, 2, 16, 1, 1, 3, 100, 2500, 3072),  # Path A, both tails.
    (4, 4, 16, 2, 1, 301, 5, 1500, 2048),  # Path B, both tails.
    (2, 2, 16, 1, 0, 3, 100, 1500, 2048),  # PF=0, path A, both tails.
    # Scale layout: both extents beyond one 32-row group and ragged, so the row group
    # stride and the zero padded rows of the last row group of both operands matter.
    (
        2,
        2,
        16,
        1,
        1,
        33,
        97,
        2048,
        2048,
    ),  # Path A: the last skinny row group has 1 row.
    (2, 2, 16, 1, 1, 97, 33, 2048, 2048),  # Path B: the same through the other operand.
    (2, 2, 16, 1, 2, 45, 77, 3072, 3072),  # Path A, ring of 2.
    (2, 2, 16, 1, 2, 77, 45, 3072, 3072),  # Path B, ring of 2.
    # K not a multiple of 256 (the scale row group is padded to 256 bytes) with every
    # LANES_K, ring scales and both operands of more than one row group.
    (2, 2, 4, 1, 1, 45, 77, 1300, 1536),  # LANES_K=4, 41 scale columns.
    (2, 2, 8, 1, 1, 77, 45, 1124, 1536),  # LANES_K=8, 36 scale columns.
    (2, 2, 16, 1, 2, 45, 77, 2148, 3072),  # LANES_K=16, ring of 2, 68 scale columns.
    (
        1,
        2,
        64,
        1,
        1,
        77,
        45,
        4196,
        4096,
    ),  # LANES_K=64, split-K x2, 100-element last split.
    # Split-K with ring scales, a partial last split and a K tail.
    (
        2,
        2,
        16,
        1,
        2,
        77,
        45,
        7200,
        3072,
    ),  # Split-K x3, last split NIfull = 1 and a tail.
]


def problemKind(row) -> tuple:
    """(path, tail) of a PARAMS row: the dispatched path and which dimensions
    are not multiples of their block extents ("aligned", "skinny", "wide" or
    "both")."""
    bm, tn, lanesK, waves, _, m, n, _, _ = row
    bn = waves * (64 // lanesK) * tn
    skinnyTail = min(m, n) % bm != 0
    wideTail = max(m, n) % bn != 0
    tail = {
        (False, False): "aligned",
        (True, False): "skinny",
        (False, True): "wide",
        (True, True): "both",
    }[(skinnyTail, wideTail)]
    return ("A" if m <= n else "B", tail)


# Both dispatch paths must cover aligned shapes, a tail in each single
# dimension and tails in both (see the Tails paragraph of gemm_mxfp8_skinny.mlir).
KINDS = {problemKind(row) for row in PARAMS}
for _path in "AB":
    for _tail in ("aligned", "skinny", "wide", "both"):
        assert (_path, _tail) in KINDS, f"need a path {_path} {_tail} shape in PARAMS"

# Padded leading dimensions: (ldA, ldB, ldD) minus their minimal values
# alignUp(k, 4) and alignUp(m, 4); multiples of 4 but not of 16.
PADDING = (4, 12, 4)
PADDED_PARAMS = [
    (2, 2, 16, 1, 1, 16, 64, 2048, 1024),  # Path A, split-K x2.
    (2, 2, 16, 1, 1, 64, 16, 2048, 2048),  # Path B, no split.
    (2, 2, 16, 1, 1, 13, 67, 2048, 1024),  # Path A tails, split-K x2.
    (2, 2, 16, 1, 1, 67, 13, 2048, 2048),  # Path B tails, no split.
    (2, 2, 16, 1, 1, 13, 67, 1999, 1024),  # Odd K, split-K x2 with a short last split.
    (2, 2, 16, 1, 2, 67, 13, 2049, 2048),  # Odd K, path B, PF=2.
]


@pytest.mark.parametrize("BM,TN,LANES_K,WAVES,PF,M,N,K,SPK", PARAMS)
def test_gemm_mxfp8_skinny(BM, TN, LANES_K, WAVES, PF, M, N, K, SPK):
    """D = A^T @ B matches the float64 dequantized reference (bf16-truncated)."""
    run(
        M, N, K, spk=SPK, bm=BM, tn=TN, lanes_k=LANES_K, waves=WAVES, pf=PF, up_scales=0
    )


@pytest.mark.parametrize("BM,TN,LANES_K,WAVES,PF,M,N,K,SPK", PADDED_PARAMS)
def test_gemm_mxfp8_skinny_padded_ld(BM, TN, LANES_K, WAVES, PF, M, N, K, SPK):
    """Padded leading dimensions are honoured and the D padding is untouched."""
    run(
        M,
        N,
        K,
        spk=SPK,
        bm=BM,
        tn=TN,
        lanes_k=LANES_K,
        waves=WAVES,
        pf=PF,
        up_scales=0,
        padding=PADDING,
    )


# (BM, TN, LANES_K, WAVES, PF, M, N, K, SPK, LDD_PAD, BLOCKS, UP): shapes that
# exercise the vector stores of a tile fully inside the output next to the
# masked stores of the tail tiles. ldD = alignUp(M, 4) + LDD_PAD. The
# configurations have a run length (BM on path A, TN on path B) of: (2, 4, 16,
# 1): 2 and 4, BN = 16; (4, 2, 8, 2): 4 and 2, BN = 32; (2, 3, 16, 1): 2 and 3
# (odd), BN = 12; (3, 2, 16, 1): 3 (odd) and 2, BN = 8. BLOCKS None is the 3-D
# launch.
STORE_TAIL_PARAMS = [
    (
        2,
        4,
        16,
        1,
        1,
        2,
        64,
        2048,
        2048,
        0,
        None,
        0,
    ),  # A, inside D, ldD = 4, m = 2: bf16x2 per column.
    (
        2,
        4,
        16,
        1,
        0,
        64,
        2,
        2048,
        2048,
        0,
        None,
        0,
    ),  # B, inside D, ldD = 64: bf16x4 per row.
    (2, 4, 16, 1, 0, 3, 64, 2048, 2048, 0, None, 0),  # A, skinny tail, m = 3, ldD = 4.
    (
        2,
        4,
        16,
        1,
        1,
        4,
        101,
        2048,
        2048,
        0,
        None,
        0,
    ),  # A, wide tail with 5 valid columns, ldD = 4.
    (
        2,
        4,
        16,
        1,
        1,
        5,
        37,
        2048,
        2048,
        0,
        None,
        0,
    ),  # A, tails in both, m = 5, ldD = 8.
    (
        2,
        4,
        16,
        1,
        0,
        5,
        37,
        2048,
        2048,
        4,
        None,
        0,
    ),  # A, tails in both, padded ldD = 12.
    (
        2,
        4,
        16,
        1,
        1,
        6,
        18,
        2048,
        2048,
        0,
        None,
        0,
    ),  # A, m = 6, ldD = 8, wide tail of 2.
    (2, 4, 16, 1, 1, 1, 5, 1024, 1024, 0, None, 0),  # A, m = 1: every tile is a tail.
    (
        2,
        4,
        16,
        1,
        0,
        100,
        3,
        2048,
        2048,
        0,
        None,
        0,
    ),  # B, ldD = 100, skinny and wide tails.
    (2, 4, 16, 1, 1, 101, 3, 2048, 2048, 0, None, 0),  # B, m = 101, ldD = 104.
    (2, 4, 16, 1, 0, 5, 1, 1024, 1024, 0, None, 0),  # B, n = 1: every tile is a tail.
    (
        2,
        4,
        16,
        1,
        1,
        301,
        5,
        2048,
        2048,
        4,
        None,
        0,
    ),  # B, padded ldD = 308, tails in both.
    (4, 2, 8, 2, 1, 4, 64, 1024, 1024, 0, None, 0),  # A, inside D: bf16x4 per column.
    (4, 2, 8, 2, 0, 64, 4, 1024, 1024, 0, None, 0),  # B, inside D: bf16x2 per row.
    (4, 2, 8, 2, 0, 6, 70, 1024, 1024, 0, None, 0),  # A, m = 6, ldD = 8, tails in both.
    (4, 2, 8, 2, 1, 70, 6, 1024, 1024, 0, None, 0),  # B, ldD = 72, tails in both.
    (
        4,
        2,
        8,
        2,
        1,
        33,
        8,
        1024,
        1024,
        4,
        None,
        0,
    ),  # B, last bf16 pair straddles the end of m, ldD = 40.
    (2, 3, 16, 1, 2, 2, 24, 2048, 2048, 0, None, 0),  # A vectorizes, inside D.
    (2, 3, 16, 1, 1, 24, 2, 2048, 2048, 0, None, 0),  # B has an odd run: ushort only.
    (2, 3, 16, 1, 2, 3, 30, 2048, 2048, 0, None, 0),  # A, tails in both, ldD = 4.
    (3, 2, 16, 1, 1, 6, 32, 2048, 2048, 0, None, 0),  # A has an odd run: ushort only.
    (3, 2, 16, 1, 2, 32, 6, 2048, 2048, 0, None, 0),  # B vectorizes, inside D.
    (
        2,
        4,
        16,
        1,
        1,
        5,
        37,
        2100,
        1024,
        0,
        None,
        0,
    ),  # Split-K x3, A, tails, m odd: ws x2 aligned to 4.
    (2, 4, 16, 1, 0, 37, 5, 3000, 1024, 0, None, 0),  # Split-K x3, B, tails, m odd.
    (
        2,
        4,
        16,
        1,
        0,
        3,
        64,
        2100,
        1024,
        0,
        None,
        0,
    ),  # Split-K, A, ws tile inside, m = 3.
    (
        2,
        4,
        16,
        1,
        1,
        49,
        4,
        2100,
        1024,
        0,
        None,
        0,
    ),  # Split-K, B, ws tiles inside, m = 49.
    (4, 2, 8, 2, 1, 6, 70, 2100, 1024, 0, None, 0),  # Split-K, A, short last split.
    (4, 2, 8, 2, 0, 70, 6, 3000, 1024, 0, None, 0),  # Split-K, B.
    (
        3,
        2,
        16,
        1,
        2,
        7,
        32,
        2100,
        1024,
        0,
        None,
        0,
    ),  # Split-K, A, ws run of 3: dwordx2 and dword.
    (
        2,
        3,
        16,
        1,
        2,
        30,
        5,
        3000,
        1024,
        0,
        None,
        0,
    ),  # Split-K, B, ws run of 3: dwordx2 and dword.
    (2, 4, 16, 1, 1, 5, 37, 2048, 2048, 0, 7, 0),  # Persistent launch, tails, ldD = 8.
    (4, 2, 8, 2, 1, 70, 6, 2100, 1024, 0, 3, 0),  # Persistent launch, split-K x3.
    (2, 4, 16, 1, 1, 5, 37, 2048, 2048, 4, None, 1),  # Up-front scales, ldD = 12.
    (4, 2, 8, 2, 1, 70, 6, 1024, 1024, 4, None, 1),  # Up-front scales, B, ldD = 76.
]


@pytest.mark.parametrize(
    "BM,TN,LANES_K,WAVES,PF,M,N,K,SPK,LDD_PAD,BLOCKS,UP", STORE_TAIL_PARAMS
)
def test_gemm_mxfp8_skinny_store_tails(
    BM, TN, LANES_K, WAVES, PF, M, N, K, SPK, LDD_PAD, BLOCKS, UP
):
    """Vector stores of tiles inside the output and masked stores of the tail
    tiles write exactly the valid elements: the sentinel in the ld padding, past
    D and next to a tail survives, and every valid element and workspace slot is
    written."""
    run(
        M,
        N,
        K,
        spk=SPK,
        bm=BM,
        tn=TN,
        lanes_k=LANES_K,
        waves=WAVES,
        pf=PF,
        blocks=BLOCKS,
        up_scales=UP,
        padding=(0, 0, LDD_PAD),
    )


@pytest.mark.parametrize(
    "M,N,VECTOR,MASKED",
    [
        (4, 64, (2, 4), (1, 4)),  # Path A: runs of BM = 2 against BM = 1.
        (64, 4, (2, 4), (2, 3)),  # Path B: runs of TN = 4 against TN = 3.
    ],
    ids=["pathA", "pathB"],
)
def test_vector_and_masked_stores_give_the_same_bf16(M, N, VECTOR, MASKED):
    """A tile with an even run stores D with vectors and one with an odd run with
    ushorts; with one K iteration (no rotation) both accumulate in the same order,
    so the bf16 results of the same inputs are bit-identical."""
    problems = [
        resolve(
            M, N, 1024, spk=1024, bm=bm, tn=tn, lanes_k=16, waves=1, pf=1, up_scales=0
        )
        for bm, tn in (VECTOR, MASKED)
    ]
    skipUnlessRunnable(*problems)
    results = [support.computeD(p, seed=42) for p in problems]
    assert np.array_equal(results[0], results[1])


TIGHT = (0, 0, 0)
# ld = k + 4, k + 12 and ldD = m + 4 for k and m multiples of 4: multiples of 4
# but not of 16.
PAD = (4, 12, 4)

# (BM, TN, LANES_K, WAVES, PF, M, N, K, SPK, PADDING, BLOCKS, UP): the buffers hold
# exactly the required extents: (cols - 1)*ld + alignUp(k, 4) bytes for A and B
# (see makeOperand) and the logical D (see makeOutputs). With TIGHT and k, m
# multiples of 4 the leading dimensions equal k and m; an odd k gets the minimal
# ld = alignUp(k, 4), so the tail step's dword holding the last valid bytes ends
# at the buffer end, followed by no byte. BLOCKS None is the 3-D launch.
EXTENT_PARAMS = [
    # Tight, k % 16 != 0: ldA = ldB = k and ldD = m.
    (2, 2, 16, 1, 1, 4, 100, 1004, 1024, TIGHT, None, 0),  # A, K and wide tails.
    (2, 2, 16, 1, 1, 100, 4, 1004, 1024, TIGHT, None, 0),  # B, K and wide tails.
    (
        2,
        2,
        16,
        1,
        0,
        8,
        52,
        2052,
        2048,
        TIGHT,
        None,
        0,
    ),  # A, PF=0, split-K x2, 4-element last split.
    (2, 2, 16, 1, 2, 52, 8, 3076, 1024, TIGHT, None, 0),  # B, PF=2, split-K x4.
    (1, 4, 4, 1, 1, 4, 256, 1004, 1024, TIGHT, None, 0),  # A, LANES_K=4.
    (2, 2, 8, 2, 1, 100, 8, 1012, 1024, TIGHT, None, 0),  # B, LANES_K=8, 2 waves.
    (1, 4, 64, 1, 1, 4, 256, 4100, 4096, TIGHT, None, 0),  # A, LANES_K=64, split-K x2.
    # Tight with BM/TN tails.
    (5, 2, 16, 1, 1, 4, 68, 1004, 1024, TIGHT, None, 0),  # A, m < BM, wide tail.
    (5, 2, 16, 1, 1, 68, 4, 1004, 1024, TIGHT, None, 0),  # B, n < BM, wide tail.
    (
        2,
        4,
        16,
        1,
        1,
        4,
        12,
        1004,
        1024,
        TIGHT,
        None,
        0,
    ),  # A, wide < BN: one partial tile.
    (2, 3, 16, 1, 2, 10, 4, 1004, 1024, TIGHT, None, 0),  # B, wide < BN, PF=2.
    (
        4,
        2,
        16,
        1,
        1,
        1,
        100,
        1004,
        1024,
        TIGHT,
        None,
        0,
    ),  # A, m = 1 with BM = 4, ldD = 4.
    (4, 2, 16, 1, 0, 100, 1, 1004, 1024, TIGHT, None, 0),  # B, n = 1 with BM = 4, PF=0.
    (2, 2, 16, 1, 1, 4, 300, 2052, 1024, TIGHT, 5, 0),  # A, persistent, split-K x3.
    (2, 2, 16, 1, 2, 300, 8, 1004, 1024, TIGHT, 7, 0),  # B, persistent, PF=2.
    # Tight with up-front scales.
    (2, 2, 16, 1, 1, 4, 36, 2048, 1024, TIGHT, None, 1),  # A, split-K x2.
    (2, 2, 16, 1, 2, 36, 4, 3072, 3072, TIGHT, None, 1),  # B, PF=2.
    (
        2,
        2,
        16,
        1,
        1,
        4,
        100,
        1004,
        1024,
        TIGHT,
        None,
        1,
    ),  # A, up-front scales with a K tail.
    # Odd k with the minimal ld = alignUp(k, 4): the last column ends at the
    # buffer end with 1 to 3 NaN bytes after byte k - 1.
    (2, 2, 16, 1, 1, 3, 100, 1001, 1024, TIGHT, None, 0),  # A, k % 4 = 1, ld = 1004.
    (2, 2, 16, 1, 1, 100, 3, 1002, 1024, TIGHT, None, 0),  # B, k % 4 = 2, ld = 1004.
    (2, 2, 16, 1, 0, 3, 100, 1003, 1024, TIGHT, None, 0),  # A, PF=0, k % 4 = 3.
    (
        2,
        2,
        16,
        1,
        2,
        100,
        3,
        2049,
        2048,
        TIGHT,
        None,
        0,
    ),  # B, PF=2, split-K x2, 1-element last split.
    # Padded: ld and ldD multiples of 4 but not of 16.
    (2, 2, 16, 1, 1, 4, 100, 1004, 1024, PAD, None, 0),  # A, K and wide tails.
    (2, 2, 16, 1, 2, 100, 4, 3076, 1024, PAD, None, 0),  # B, PF=2, split-K x4.
    (2, 2, 16, 1, 0, 3, 100, 1003, 1024, PAD, None, 0),  # A, PF=0, odd k.
    (
        2,
        2,
        16,
        1,
        1,
        36,
        4,
        2048,
        1024,
        PAD,
        None,
        1,
    ),  # B, up-front scales, split-K x2.
    (2, 2, 16, 1, 1, 4, 300, 2052, 1024, PAD, 5, 0),  # A, persistent, split-K x3.
]


@pytest.mark.parametrize(
    "BM,TN,LANES_K,WAVES,PF,M,N,K,SPK,PADDING,BLOCKS,UP", EXTENT_PARAMS
)
def test_gemm_mxfp8_skinny_exact_extents(
    BM, TN, LANES_K, WAVES, PF, M, N, K, SPK, PADDING, BLOCKS, UP
):
    """The kernel reads only the required extents of A and B and writes only the
    logical extent of D, tight or padded, and matches the reference."""
    run(
        M,
        N,
        K,
        spk=SPK,
        bm=BM,
        tn=TN,
        lanes_k=LANES_K,
        waves=WAVES,
        pf=PF,
        blocks=BLOCKS,
        up_scales=UP,
        padding=PADDING,
        iters=1,
    )


# (BM, TN, LANES_K, WAVES, PF, M, N, K, SPK, PADDING, BLOCKS, GROUP, XCDS): spk
# is not a multiple of KI = 64*LANES_K and need not divide k, so splits start and
# end inside iterations (head and tail partial steps). The comments give the
# split boundaries kz*spk modulo 256 (8-block group), 32 (MX block), 16 and 4
# where they matter.
SPK_PARAMS = [
    (
        2,
        2,
        16,
        1,
        1,
        3,
        100,
        3000,
        1300,
        TIGHT,
        None,
        1,
        1,
    ),  # A, spk > KI, 3 splits; boundaries 1300 and 2600 are inside an MX block and a 16-byte part; split 1 has no full iteration.
    (2, 2, 16, 1, 2, 100, 3, 3000, 1300, PAD, None, 1, 1),  # B, PF=2, padded.
    (
        2,
        2,
        16,
        1,
        0,
        3,
        100,
        2500,
        700,
        TIGHT,
        None,
        1,
        1,
    ),  # A, PF=0, spk < KI, 4 splits.
    (2, 2, 16, 1, 1, 100, 3, 2500, 700, PAD, None, 1, 1),  # B.
    (
        2,
        2,
        16,
        1,
        1,
        3,
        100,
        1000,
        100,
        TIGHT,
        None,
        1,
        1,
    ),  # 10 splits inside one iteration: one step with both bounds.
    (
        2,
        2,
        4,
        1,
        1,
        3,
        40,
        300,
        7,
        TIGHT,
        None,
        1,
        1,
    ),  # LANES_K=4, odd spk = 7, 43 splits.
    (
        2,
        2,
        8,
        1,
        1,
        3,
        20,
        100,
        3,
        TIGHT,
        None,
        1,
        1,
    ),  # spk = 3: lo and hi in the same dword.
    (
        1,
        4,
        16,
        1,
        1,
        2,
        10,
        37,
        1,
        TIGHT,
        None,
        1,
        1,
    ),  # spk = 1: one byte per split, lo and hi in one dword.
    (
        2,
        2,
        8,
        1,
        1,
        3,
        100,
        2000,
        333,
        PAD,
        None,
        1,
        1,
    ),  # LANES_K=8, odd spk < KI, 2-element last split.
    (
        2,
        2,
        8,
        2,
        2,
        100,
        3,
        2500,
        777,
        TIGHT,
        None,
        1,
        1,
    ),  # LANES_K=8, 2 waves, odd spk > KI, B, PF=2.
    (1, 4, 4, 1, 2, 2, 256, 1000, 300, TIGHT, None, 1, 1),  # LANES_K=4, spk > KI, PF=2.
    (
        1,
        4,
        4,
        1,
        0,
        256,
        2,
        1100,
        129,
        PAD,
        None,
        1,
        1,
    ),  # LANES_K=4, B, PF=0, odd spk < KI.
    (1, 4, 64, 1, 1, 2, 64, 12000, 5000, TIGHT, None, 1, 1),  # LANES_K=64, spk > KI.
    (
        1,
        4,
        64,
        1,
        1,
        64,
        2,
        4000,
        1301,
        PAD,
        None,
        1,
        1,
    ),  # LANES_K=64, B, odd spk, all splits inside iteration 0.
    (
        2,
        1,
        16,
        1,
        3,
        3,
        100,
        9000,
        4100,
        TIGHT,
        None,
        1,
        1,
    ),  # PF=3, NIfull 4 and 3, boundaries 4100 and 8200 at 4 and 8 mod 16.
    (
        2,
        2,
        16,
        1,
        1,
        2,
        300,
        7000,
        3500,
        TIGHT,
        None,
        1,
        1,
    ),  # 38 x 1 x 2 tiles: rotation modulus 4 against NIfull 3 and 2.
    (2, 2, 16, 1, 2, 300, 2, 7000, 3500, PAD, None, 1, 1),  # B, PF=2, same.
    (
        2,
        2,
        16,
        1,
        1,
        3,
        100,
        4000,
        1312,
        TIGHT,
        None,
        1,
        1,
    ),  # Boundaries on MX blocks inside 8-block groups.
    (
        2,
        2,
        16,
        1,
        1,
        100,
        3,
        3100,
        1040,
        TIGHT,
        None,
        1,
        1,
    ),  # B, boundary 1040 on a 16-byte part inside an MX block, 2080 on an MX block.
    (
        2,
        2,
        16,
        1,
        1,
        3,
        100,
        3000,
        768,
        TIGHT,
        None,
        1,
        1,
    ),  # Boundaries on 8-block groups, not on iterations.
    (
        2,
        2,
        16,
        1,
        1,
        3,
        100,
        5200,
        1025,
        TIGHT,
        None,
        1,
        1,
    ),  # spk = KI + 1: lo walks through the first dword.
    (2, 2, 16, 1, 2, 100, 3, 4000, 1023, PAD, None, 1, 1),  # spk = KI - 1, B, PF=2.
    (
        2,
        2,
        16,
        1,
        1,
        2,
        300,
        3000,
        1300,
        TIGHT,
        7,
        1,
        1,
    ),  # Persistent, 114 tiles over 7 blocks.
    (
        2,
        2,
        16,
        1,
        1,
        11,
        300,
        3000,
        1300,
        PAD,
        5,
        4,
        8,
    ),  # Persistent, grouped and XCD remapped.
    (
        2,
        2,
        16,
        1,
        1,
        11,
        300,
        2500,
        700,
        TIGHT,
        None,
        4,
        8,
    ),  # Reordered, one block per tile.
    (
        1,
        1,
        4,
        1,
        1,
        1,
        1,
        70000,
        1,
        TIGHT,
        None,
        1,
        1,
    ),  # 70000 splits: above HIP's 65535 grid.z, so the 1-D launch.
]


@pytest.mark.parametrize(
    "BM,TN,LANES_K,WAVES,PF,M,N,K,SPK,PADDING,BLOCKS,GROUP,XCDS", SPK_PARAMS
)
def test_gemm_mxfp8_skinny_any_spk(
    BM, TN, LANES_K, WAVES, PF, M, N, K, SPK, PADDING, BLOCKS, GROUP, XCDS
):
    """Any spk >= 1 matches the reference: each split sums exactly its K range."""
    run(
        M,
        N,
        K,
        spk=SPK,
        bm=BM,
        tn=TN,
        lanes_k=LANES_K,
        waves=WAVES,
        pf=PF,
        blocks=BLOCKS,
        group=GROUP,
        xcds=XCDS,
        up_scales=0,
        padding=PADDING,
        iters=1,
    )


# (BM, TN, LANES_K, WAVES, PF, M, N, K, SPK, PADDING, BLOCKS, GROUP, XCDS) for the
# up-front scales on top of SPK_PARAMS: single splits with a K tail and splits of
# whole iterations with a short last split.
UP_SPK_PARAMS = [
    (
        2,
        2,
        16,
        1,
        1,
        3,
        100,
        3000,
        3000,
        TIGHT,
        None,
        1,
        1,
    ),  # One split, spk = k not a multiple of KI: a tail step, UP_ITERS = 3.
    (
        2,
        2,
        16,
        1,
        0,
        100,
        3,
        3000,
        3072,
        PAD,
        None,
        1,
        1,
    ),  # One split, spk % KI == 0 (aligned tail step), B, PF=0.
    (
        2,
        2,
        16,
        1,
        2,
        3,
        100,
        1001,
        1024,
        TIGHT,
        None,
        1,
        1,
    ),  # One split inside one iteration (NIfull = 0), PF=2.
    (
        1,
        2,
        16,
        1,
        1,
        1,
        128,
        7000,
        7168,
        TIGHT,
        None,
        1,
        1,
    ),  # UP_ITERS = 7, the tail selects the last entry.
    (
        2,
        2,
        16,
        1,
        1,
        3,
        100,
        2600,
        1024,
        TIGHT,
        None,
        1,
        1,
    ),  # spk % KI == 0, short last split with NIfull = 0.
    (
        2,
        2,
        16,
        1,
        2,
        100,
        3,
        5200,
        2048,
        PAD,
        None,
        1,
        1,
    ),  # Short last split with NIfull = 1 and a tail, B, PF=2.
    (
        2,
        2,
        8,
        2,
        0,
        100,
        3,
        2600,
        1024,
        PAD,
        None,
        1,
        1,
    ),  # LANES_K=8, 2 waves, PF=0, short last split.
    (
        1,
        4,
        4,
        1,
        1,
        2,
        256,
        1000,
        512,
        TIGHT,
        None,
        1,
        1,
    ),  # LANES_K=4, short last split with a tail.
    (
        1,
        4,
        64,
        1,
        2,
        64,
        2,
        9000,
        4096,
        TIGHT,
        None,
        1,
        1,
    ),  # LANES_K=64, B, PF=2, short last split.
    (
        2,
        2,
        16,
        1,
        1,
        4,
        100,
        4097,
        1024,
        PAD,
        5,
        1,
        1,
    ),  # Persistent, 1-element last split.
    (
        2,
        2,
        16,
        1,
        1,
        11,
        300,
        2600,
        1024,
        TIGHT,
        None,
        4,
        8,
    ),  # Reordered, short last split.
    (
        2,
        2,
        16,
        1,
        1,
        3,
        100,
        3072,
        2048,
        TIGHT,
        None,
        1,
        1,
    ),  # PARTIAL_STEPS = 0, short last split: its second vector entry lies past k, unsteered and never selected.
    (
        2,
        2,
        16,
        1,
        1,
        2,
        300,
        5120,
        3072,
        PAD,
        5,
        1,
        1,
    ),  # PARTIAL_STEPS = 0, persistent, rotation modulus 3 against NIfull 2.
    # LANES_K = 64 with several waves shares the scales through LDS.
    (
        2,
        2,
        64,
        2,
        1,
        3,
        100,
        5000,
        5000,
        TIGHT,
        None,
        1,
        1,
    ),  # Shared, one split with a K tail, UP_ITERS = 2.
    (
        2,
        2,
        64,
        2,
        2,
        100,
        3,
        9000,
        4096,
        PAD,
        None,
        1,
        1,
    ),  # Shared, B, PF=2, short last split.
    (
        1,
        2,
        64,
        4,
        1,
        2,
        70,
        9000,
        3100,
        TIGHT,
        None,
        1,
        1,
    ),  # Shared, 4 waves, splits inside iterations: both partial steps.
    (
        2,
        2,
        64,
        2,
        1,
        4,
        100,
        4097,
        1024,
        PAD,
        5,
        1,
        1,
    ),  # Shared, persistent, 1-element last split.
]


@pytest.mark.parametrize(
    "BM,TN,LANES_K,WAVES,PF,M,N,K,SPK,PADDING,BLOCKS,GROUP,XCDS",
    SPK_PARAMS + UP_SPK_PARAMS,
)
def test_gemm_mxfp8_skinny_up_scales_any_spk(
    BM, TN, LANES_K, WAVES, PF, M, N, K, SPK, PADDING, BLOCKS, GROUP, XCDS
):
    """The scales loaded up front give the reference result for any spk and K:
    the vector starts at the split's first touched iteration and the partial
    steps read it too."""
    run(
        M,
        N,
        K,
        spk=SPK,
        bm=BM,
        tn=TN,
        lanes_k=LANES_K,
        waves=WAVES,
        pf=PF,
        blocks=BLOCKS,
        group=GROUP,
        xcds=XCDS,
        up_scales=1,
        padding=PADDING,
        iters=1,
    )


# (BM, TN, LANES_K, WAVES, PF, M, N, K, SPK, UP) run with TIGHT and with PAD.
IDENTITY_PARAMS = [
    (2, 2, 16, 1, 1, 4, 100, 1004, 1024, 0),  # A, K and wide tails.
    (2, 2, 16, 1, 2, 100, 4, 3076, 1024, 0),  # B, PF=2, split-K x4 (reduce writes D).
    (2, 2, 16, 1, 1, 4, 36, 2048, 1024, 1),  # A, up-front scales, split-K x2.
    (
        2,
        2,
        16,
        1,
        1,
        100,
        4,
        3004,
        1300,
        1,
    ),  # B, up-front scales, spk not a multiple of KI.
    (1, 4, 64, 1, 0, 100, 4, 4100, 4096, 0),  # B, LANES_K=64, PF=0, split-K x2.
]


@pytest.mark.parametrize("BM,TN,LANES_K,WAVES,PF,M,N,K,SPK,UP", IDENTITY_PARAMS)
def test_tight_and_padded_buffers_give_the_same_d(
    BM, TN, LANES_K, WAVES, PF, M, N, K, SPK, UP
):
    """The same problem with tight and with padded leading dimensions gives
    bit-identical D: the kernel does not depend on the padding."""
    problems = [
        resolve(
            M,
            N,
            K,
            spk=SPK,
            bm=BM,
            tn=TN,
            lanes_k=LANES_K,
            waves=WAVES,
            pf=PF,
            up_scales=UP,
            padding=padding,
        )
        for padding in (TIGHT, PAD)
    ]
    skipUnlessRunnable(*problems)
    results = [support.computeD(p, seed=42) for p in problems]
    assert np.array_equal(results[0], results[1])


# (BM, TN, LANES_K, WAVES, PF, M, N, K, SPK, B, GROUP, XCDS) with the scales loaded
# up front, NI = SPK / (64 * LANES_K) and B = None for one block per tile.
UP_PARAMS = [
    (1, 5, 64, 1, 1, 5120, 1, 8192, 8192, None, 1, 1),  # Primary shape, path B, NI = 2.
    (1, 5, 64, 1, 1, 1, 5120, 8192, 8192, None, 1, 1),  # Its path A mirror.
    (1, 4, 16, 1, 1, 1, 256, 1024, 1024, None, 1, 1),  # NI = 1.
    (2, 2, 16, 1, 1, 3, 100, 3072, 3072, None, 1, 1),  # NI = 3, tails.
    (2, 2, 16, 1, 3, 100, 3, 3072, 3072, None, 1, 1),  # NI = 3 = PF, path B tails.
    (
        1,
        4,
        16,
        1,
        2,
        2,
        300,
        8192,
        4096,
        None,
        1,
        1,
    ),  # NI = 4, split-K x2, PF=2, wide tail.
    (2, 2, 16, 1, 0, 3, 100, 4096, 4096, None, 1, 1),  # NI = 4, PF=0, tails.
    (2, 2, 16, 1, 1, 301, 5, 10240, 5120, None, 1, 1),  # NI = 5, split-K x2.
    (1, 2, 8, 2, 0, 2, 64, 2560, 2560, None, 1, 1),  # NI = 5, PF=0, LANES_K=8, 2 waves.
    (1, 2, 16, 1, 2, 1, 256, 6144, 6144, None, 1, 1),  # NI = 6, PF=2.
    (1, 2, 16, 1, 3, 1, 256, 6144, 6144, None, 1, 1),  # NI = 6, PF=3.
    (1, 2, 16, 1, 1, 1, 128, 7168, 7168, None, 1, 1),  # NI = 7.
    (
        1,
        4,
        4,
        1,
        2,
        256,
        1,
        4096,
        2048,
        None,
        1,
        1,
    ),  # NI = 8, LANES_K=4, split-K x2, path B.
    (
        2,
        2,
        64,
        2,
        1,
        6,
        67,
        12288,
        12288,
        None,
        1,
        1,
    ),  # NI = 3, LANES_K=64, 2 waves, tails.
    # LANES_K = 64 with several waves shares the scales through LDS.
    (
        1,
        1,
        64,
        4,
        2,
        5120,
        1,
        8192,
        8192,
        None,
        1,
        1,
    ),  # Primary shape, 4 waves, path B, PF=2.
    (1, 1, 64, 4, 2, 1, 5120, 8192, 8192, None, 1, 1),  # Its path A mirror.
    (
        2,
        2,
        64,
        2,
        1,
        67,
        6,
        12288,
        12288,
        None,
        1,
        1,
    ),  # NI = 3, 2 waves, path B, tails.
    (1, 2, 64, 8, 1, 3, 300, 8192, 4096, None, 1, 1),  # 8 waves, split-K x2, wide tail.
    (
        2,
        2,
        64,
        2,
        1,
        6,
        67,
        12288,
        12288,
        5,
        1,
        1,
    ),  # Persistent, 51 tiles over 5 blocks: the buffer is reused.
    (
        2,
        2,
        16,
        1,
        1,
        11,
        300,
        3072,
        3072,
        None,
        4,
        8,
    ),  # NI = 3, grouped and XCD remapped.
    (
        2,
        2,
        16,
        1,
        1,
        2,
        300,
        3072,
        3072,
        5,
        1,
        1,
    ),  # NI = 3, persistent, 38 tiles: K rotation varies.
    (
        2,
        3,
        16,
        1,
        2,
        3,
        250,
        8192,
        2048,
        10,
        3,
        8,
    ),  # NI = 2, split-K x4, PF=2, persistent, grouped.
    (
        2,
        2,
        16,
        1,
        1,
        45,
        77,
        6144,
        3072,
        None,
        1,
        1,
    ),  # NI = 3, split-K x2, both extents beyond 32 rows.
    (
        2,
        2,
        16,
        1,
        2,
        77,
        45,
        8192,
        4096,
        None,
        1,
        1,
    ),  # NI = 4, split-K x2, PF=2, path B.
]


@pytest.mark.parametrize("BM,TN,LANES_K,WAVES,PF,M,N,K,SPK,B,GROUP,XCDS", UP_PARAMS)
def test_gemm_mxfp8_skinny_up_scales(
    BM, TN, LANES_K, WAVES, PF, M, N, K, SPK, B, GROUP, XCDS
):
    """The scales loaded up front give the reference result
    for every iteration count, ring depth, tail, split and tile order."""
    # The zero scales of masked columns are not observable through D (masked
    # columns are never stored); they are checked by ISA inspection only.
    run(
        M,
        N,
        K,
        spk=SPK,
        bm=BM,
        tn=TN,
        lanes_k=LANES_K,
        waves=WAVES,
        pf=PF,
        blocks=B,
        group=GROUP,
        xcds=XCDS,
        up_scales=1,
    )


# Not from the driver: up-front scales through LDS in shapes that no driver row
# covers. (BM, TN, LANES_K, WAVES, PF, M, N, K, SPK).
SHARED_SCALES_PARAMS = [
    (1, 1, 64, 3, 1, 1, 2000, 8192, 8192),  # R = 3 is not a power of two.
    (1, 1, 64, 3, 1, 2000, 1, 8192, 8192),  # R = 3, path B.
    (1, 1, 64, 4, 0, 1, 5120, 8192, 8192),  # PF=0 with shared scales.
    (1, 1, 64, 2, 4, 1, 1024, 16384, 16384),  # PF=4 with shared scales.
    # UP = 5, NC = 4, WAVES = 16: 80 KiB of LDS, so the ds offsets exceed 65535.
    (2, 2, 64, 16, 1, 4, 2000, 20480, 20480),
]


@pytest.mark.parametrize("BM,TN,LANES_K,WAVES,PF,M,N,K,SPK", SHARED_SCALES_PARAMS)
def test_gemm_mxfp8_skinny_shared_up_scales(BM, TN, LANES_K, WAVES, PF, M, N, K, SPK):
    """The scales shared through LDS give the reference result for block sizes
    with a non-power-of-two role domain, both paths, every ring depth and LDS
    offsets beyond the ds offset field."""
    run(
        M,
        N,
        K,
        spk=SPK,
        bm=BM,
        tn=TN,
        lanes_k=LANES_K,
        waves=WAVES,
        pf=PF,
        up_scales=1,
    )


# (BM, TN, LANES_K, WAVES, PF, M, N, K, SPK, B, GROUP, XCDS) with B = None for one
# block per tile. The tile counts are deliberately not multiples of the XCD
# count or of GROUP.
GROUPED_PARAMS = [
    (
        2,
        2,
        16,
        1,
        1,
        11,
        300,
        2048,
        2048,
        None,
        4,
        8,
    ),  # 6 x 38 tiles, ragged last band.
    (2, 2, 16, 1, 1, 11, 300, 2048, 2048, None, 4, 1),  # Grouping alone.
    (2, 2, 16, 1, 1, 5, 150, 2048, 2048, None, 1, 8),  # Remap alone, 3 x 19 tiles.
    (2, 2, 16, 1, 1, 13, 100, 2048, 2048, None, 3, 8),  # 7 rows in bands of 3.
    (2, 2, 16, 1, 1, 8, 200, 4096, 1024, None, 2, 8),  # Split-K x4.
    (2, 2, 16, 2, 1, 301, 7, 2048, 1024, None, 3, 8),  # Path B tails, split-K x2.
    (4, 2, 16, 1, 1, 3, 200, 2048, 2048, None, 8, 8),  # GROUP above the tile rows.
    (4, 2, 64, 2, 1, 13, 67, 4096, 4096, None, 2, 8),  # LANES_K=64 tails.
    (2, 2, 16, 1, 1, 7, 6, 2048, 2048, None, 4, 8),  # A single tile column.
    (
        2,
        2,
        16,
        1,
        1,
        11,
        300,
        2048,
        2048,
        5,
        4,
        8,
    ),  # Persistent, B not a multiple of 8.
    (2, 2, 16, 1, 1, 11, 300, 2048, 2048, 16, 4, 8),  # Persistent, B a multiple of 8.
    (2, 2, 16, 1, 1, 11, 300, 2048, 2048, 1, 4, 8),  # One block runs every tile.
    (2, 2, 16, 1, 1, 11, 300, 2048, 2048, 512, 4, 8),  # More blocks than tiles.
    (2, 2, 16, 1, 0, 11, 300, 2048, 2048, None, 4, 8),  # PF=0, grouped + XCD remap.
    (2, 3, 16, 1, 2, 3, 250, 8192, 2048, 10, 3, 8),  # Persistent, split-K x4, PF=2.
]


@pytest.mark.parametrize(
    "BM,TN,LANES_K,WAVES,PF,M,N,K,SPK,B,GROUP,XCDS", GROUPED_PARAMS
)
def test_gemm_mxfp8_skinny_reordered(
    BM, TN, LANES_K, WAVES, PF, M, N, K, SPK, B, GROUP, XCDS
):
    """Every tile is computed exactly once under the grouped and XCD-remapped order."""
    run(
        M,
        N,
        K,
        spk=SPK,
        bm=BM,
        tn=TN,
        lanes_k=LANES_K,
        waves=WAVES,
        pf=PF,
        blocks=B,
        group=GROUP,
        xcds=XCDS,
        up_scales=0,
    )


# (BM, TN, LANES_K, WAVES, PF, M, N, K, SPK, B) with B != number of tiles.
PERSISTENT_PARAMS = [
    (2, 2, 16, 1, 1, 2, 300, 3072, 3072, 5),  # 38 tiles, not a multiple of B.
    (2, 2, 16, 1, 1, 2, 300, 3072, 3072, 1),  # One block runs every tile.
    (2, 2, 16, 1, 1, 2, 300, 3072, 3072, 64),  # More blocks than tiles.
    (2, 2, 16, 1, 0, 2, 300, 3072, 3072, 5),  # PF=0, 38 tiles, not a multiple of B.
    (2, 2, 16, 2, 1, 8, 256, 6144, 3072, 13),  # 3-D tile grid: 16 x 4 x 2.
    (2, 3, 16, 1, 2, 3, 250, 8192, 2048, 10),  # Path A tails, split-K x4, PF=2.
    (
        1,
        4,
        16,
        1,
        2,
        2,
        300,
        4096,
        4096,
        5,
    ),  # PF=2, NI = 2*PF, 38 tiles, not a multiple of B.
    (
        1,
        2,
        16,
        1,
        3,
        2,
        300,
        3072,
        3072,
        5,
    ),  # PF=3, NI = PF, 76 tiles, not a multiple of B.
    (2, 2, 16, 1, 1, 301, 5, 4096, 2048, 7),  # Path B tails, split-K x2.
    (4, 2, 64, 2, 1, 6, 67, 4096, 4096, 3),  # Tails, LANES_K=64.
    (2, 2, 16, 1, 1, 2, 8, 4096, 1024, 3),  # tilesX = tilesY = 1, split-K x4.
    (
        2,
        2,
        16,
        1,
        1,
        2,
        300,
        2600,
        1024,
        5,
    ),  # K tail, short last split with NIfull = 0.
    (
        2,
        2,
        16,
        1,
        2,
        2,
        300,
        5200,
        2048,
        7,
    ),  # K tail, short last split with NIfull = 1, PF=2.
    (3, 2, 16, 2, 1, 7, 150, 2048, 1024, 5),  # BM = 3, BN = 16: 10 x 3 x 2 tiles.
    (2, 2, 16, 1, 1, 2, 4200, 1024, 1024, 7),  # 525 tiles: more than 512.
    (3, 3, 16, 1, 1, 7, 6200, 1024, 1024, 50),  # BM = 3, BN = 12: 1551 tiles.
]


@pytest.mark.parametrize("BM,TN,LANES_K,WAVES,PF,M,N,K,SPK,B", PERSISTENT_PARAMS)
def test_gemm_mxfp8_skinny_persistent(BM, TN, LANES_K, WAVES, PF, M, N, K, SPK, B):
    """Every tile is computed exactly once for any number of launched blocks."""
    run(
        M,
        N,
        K,
        spk=SPK,
        bm=BM,
        tn=TN,
        lanes_k=LANES_K,
        waves=WAVES,
        pf=PF,
        blocks=B,
        up_scales=0,
    )


@pytest.mark.parametrize(
    "M,N,K,SPK",
    [
        (2, 300, 4096, 1024),  # Split-K x4 with a tail tile.
        (5120, 1, 8192, 8192),  # Primary shape, path B.
    ],
)
def test_gemm_mxfp8_skinny_one_dimensional_launch(M, N, K, SPK):
    """An explicit block count equal to the tile count selects the 1-D launch
    (GRID_3D = 0), which keeps the tile loop and decode, and matches the
    reference."""
    knobs = support.pickConfig(M, N, SPK)
    tiles = support.countTiles(
        M, N, K, SPK, knobs.bm, support.blockCols(knobs.waves, knobs.lanes_k, knobs.tn)
    )
    assert not support.noKnobProblem(M, N, K, spk=SPK, blocks=tiles).grid3d
    run(M, N, K, spk=SPK, blocks=tiles)


# (BM, TN, LANES_K, WAVES, PF, M, N, K, SPK, UP). BN = WAVES * (64 / LANES_K) * TN
# wide columns per tile, G = 32 / BN tiles per XCD run (1 unless BN is a power of
# two below 32), gx = ceil(wide / BN) blocks along x and the span is 8 * G.
XCD_GROUPED_PARAMS = [
    # LANES_K = 64, WAVES = 4, TN = 1: BN 4, G 8, span 64.
    (1, 1, 64, 4, 1, 768, 1, 4096, 4096, 0),  # B, gx 192 = 3 spans.
    (1, 1, 64, 4, 1, 789, 1, 4096, 4096, 1),  # B, gx 198: tail of 6, partial tile.
    (1, 1, 64, 4, 2, 1, 789, 4096, 4096, 1),  # A, gx 198: tail of 6, partial tile.
    (1, 1, 64, 4, 1, 100, 1, 4096, 4096, 0),  # B, gx 25 < span: identity.
    (2, 1, 64, 4, 1, 789, 5, 8192, 4096, 1),  # B, gx 198, gy 3, split-K x2.
    # LANES_K = 64, WAVES = 1, TN = 2: BN 2, G 16, span 128.
    (1, 2, 64, 1, 1, 300, 1, 4096, 4096, 0),  # B, gx 150: 1 span, tail of 22.
    (1, 2, 64, 1, 1, 1, 512, 4096, 4096, 1),  # A, gx 256 = 2 spans.
    # LANES_K = 16, WAVES = 1, TN = 1: BN 4, G 8, span 64.
    (2, 1, 16, 1, 1, 520, 5, 2048, 1024, 0),  # B, gx 130: 2 spans + 2, gy 3, gz 2.
    # LANES_K = 16, WAVES = 2, TN = 2: BN 16, G 2, span 16.
    (1, 2, 16, 2, 1, 3, 512, 1024, 1024, 0),  # A, gx 32 = 2 spans, gy 3.
    (1, 2, 16, 2, 1, 3, 300, 2048, 1024, 1),  # A, gx 19: tail of 3, gy 3, gz 2.
    (1, 2, 16, 2, 1, 1, 100, 1024, 1024, 0),  # A, gx 7 < span: identity.
    # LANES_K = 8, WAVES = 1, TN = 1: BN 8, G 4, span 32.
    (2, 1, 8, 1, 1, 512, 2, 512, 512, 0),  # B, gx 64 = 2 spans.
    (2, 1, 8, 1, 1, 1000, 2, 1024, 512, 1),  # B, gx 125: tail of 29, gz 2.
    # LANES_K = 4, WAVES = 1, TN = 1: BN 16, G 2, span 16.
    (2, 1, 4, 1, 1, 2, 1024, 256, 256, 0),  # A, gx 64 = 4 spans.
    (2, 1, 4, 1, 1, 2, 1000, 512, 256, 1),  # A, gx 63: tail of 15, partial tile.
    # G = 1 controls, no remap.
    (2, 1, 4, 2, 1, 2, 640, 256, 256, 0),  # LANES_K = 4, WAVES = 2: BN 32.
    (1, 3, 16, 1, 1, 1, 300, 1024, 1024, 0),  # BN 12, not a power of two.
]


@pytest.mark.parametrize("BM,TN,LANES_K,WAVES,PF,M,N,K,SPK,UP", XCD_GROUPED_PARAMS)
def test_gemm_mxfp8_skinny_xcd_grouped_tiles(
    BM, TN, LANES_K, WAVES, PF, M, N, K, SPK, UP
):
    """Every wide tile is computed exactly once when the 3-D launch groups the
    tiles that share scale lines per XCD (a missed tile leaves the D sentinel,
    so the reference check catches it)."""
    run(
        M,
        N,
        K,
        spk=SPK,
        bm=BM,
        tn=TN,
        lanes_k=LANES_K,
        waves=WAVES,
        pf=PF,
        up_scales=UP,
    )


@pytest.mark.parametrize("M,N", [(5120, 1), (1, 5120)])
def test_gemm_mxfp8_skinny_xcd_grouped_default_knobs(M, N):
    """The default knobs of the focus shape (BN 4, gx 1280 = 20 spans of 64)
    compute every tile exactly once."""
    run(M, N, 8192)


def test_default_knobs_run_and_overrides_win():
    """run() without knobs uses pickConfig; an explicit knob replaces its field."""
    problems = [resolve(4, 256, 4096), resolve(4, 256, 4096, tn=1, waves=1)]
    skipUnlessRunnable(*problems)
    for p in problems:
        support.runProblem(p, 10)


@pytest.mark.parametrize(
    "BM,TN,PF,BLOCKS",
    [
        (6, 3, 1, 3),  # Persistent: allocates at 255 VGPRs.
        (3, 6, 1, None),  # 3-D: allocates at 245 VGPRs.
    ],
)
def test_tiles_near_the_register_limit_run(BM, TN, PF, BLOCKS):
    """Tiles near the 256 VGPR limit compile and verify, persistent (BLOCKS) or
    in the 3-D launch."""
    run(
        BM,
        1024,
        4096,
        bm=BM,
        tn=TN,
        lanes_k=16,
        waves=1,
        pf=PF,
        up_scales=0,
        blocks=BLOCKS,
        iters=1,
    )


def test_vocabulary_projection_runs():
    """The vLLM vocabulary projection (n = 151936, above the 65536 that the
    kernel assumed once) runs with the default knobs and matches the reference."""
    run(1, 151936, 5120, iters=1)


# (M, N, K, PF, UP): the last byte of each scale buffer holds a real scale, so a scale
# extent too small by even one byte zeroes the scale of the last row and column.
SCALE_END_PARAMS = [(32, 32, 512, 1, 0), (32, 64, 1000, 2, 0), (32, 32, 2048, 1, 1)]


@pytest.mark.parametrize("M,N,K,PF,UP", SCALE_END_PARAMS)
def test_scales_fill_exactly_their_extent(M, N, K, PF, UP):
    """The host scale buffers are exactly roundUp(rows, 32) * roundUp(ceil(k / 32),
    8) bytes and the kernel, whose buffer resources have the same extents, reads
    the scale in the last byte of each of them."""
    p = support.noKnobProblem(
        M,
        N,
        K,
        bm=2,
        tn=2,
        lanes_k=16,
        waves=1,
        pf=PF,
        up_scales=UP,
        spk=2048 if UP else 1024,
    )
    inputs = support.makeInputs(p, seed=3)
    for rows, scales in ((M, inputs.sclABuf), (N, inputs.sclBBuf)):
        extent = support.roundUp(rows, 32) * support.roundUp(support.ceilDiv(K, 32), 8)
        assert scales.size == extent
        assert scales[-1] != 0
    skipUnlessRunnable(p)
    support.executeProblem(p, inputs, 1)


# (LANES_K, k) whose partial step has 8-block groups at or after k; the group
# after the last one of a row group holds the scales of other rows. The last
# three have a remainder of 256 bytes in the last iteration, so k is a multiple
# of 256 and the next group starts exactly at k. With 4 lanes the remainder is
# below 256 (no group at or after k when k is a multiple of 256), so that case
# needs no boundary row.
SCALE_STEER_PARAMS = [
    (4, 1300),
    (8, 1124),
    (16, 2148),
    (64, 4196),
    (8, 1280),
    (16, 2304),
    (64, 4352),
]


@pytest.mark.parametrize("LANES_K,K", SCALE_STEER_PARAMS)
@pytest.mark.parametrize("UP", [0, 1])
def test_tail_scales_never_read_the_next_row_group(LANES_K, K, UP):
    """With every scale of the row groups after the first set to 255 (+inf), the
    outputs that only involve the first row group of both operands stay finite
    and correct: a scale read of the partial step, ring or up front (where
    @load_scale_vectors steers them), that crossed into the next row group would
    turn the zero padding of the data into 0 * inf = NaN."""
    p = support.noKnobProblem(
        40, 64, K, lanes_k=LANES_K, pf=1, bm=1, tn=2, waves=1, up_scales=UP
    )
    skipUnlessRunnable(p)
    inputs = support.makeInputs(p, seed=42)
    rowGroupBytes = support.roundUp(K, support.SCALE_COLS * support.MX_BLOCK)
    inputs.sclABuf[rowGroupBytes:] = 255
    inputs.sclBBuf[rowGroupBytes:] = 255

    def verifyFirstRowGroup(_inputs, outputs):
        rows = support.SCALE_ROWS
        actual = support.dColumns(outputs[0], p.m, p.n, p.ldD)[:rows, :rows]
        reference = support.truncateToBf16(inputs.expected[:rows, :rows])
        np.testing.assert_allclose(
            support.bf16ToFloat32(actual).T,
            support.bf16ToFloat32(reference),
            rtol=1e-2,
            atol=1e-4 * float(np.abs(inputs.expected).max()),
        )

    support.executeProblem(p, inputs, 1, verifyFn=verifyFirstRowGroup)
