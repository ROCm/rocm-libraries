// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <mxDataGen.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <vector>

/**
 * @brief Returns true if a 4-bit FP4 E2M1 nibble represents zero.
 *
 * FP4 E2M1 values are packed two-per-byte (low nibble first).
 * Both 0x0 (+0) and 0x8 (-0) decode to zero.
 */
static bool isZeroNibble(uint8_t nibble)
{
    // FP4 E2M1: 0x0 = +0.0, 0x8 = -0.0
    return (nibble == 0x0) || (nibble == 0x8);
}

/**
 * @brief Count elements that decode to zero in a packed FP4 buffer.
 */
static size_t countZerosFP4(const uint8_t* packedData, size_t numPackedBytes)
{
    size_t zeros = 0;
    for(size_t i = 0; i < numPackedBytes; ++i)
    {
        uint8_t lo = packedData[i] & 0x0F;
        uint8_t hi = (packedData[i] >> 4) & 0x0F;
        if(isZeroNibble(lo))
            ++zeros;
        if(isZeroNibble(hi))
            ++zeros;
    }
    return zeros;
}

class MXDataGenFP4Test : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, int, bool>>
{
};

/**
 * @brief Verify that generateMXInput produces FP4 data with an acceptable zero frequency.
 *
 * FP4 E2M1 has 16 nibble values, 2 of which are zero (0x0 = +0, 0x8 = -0), giving a
 * naive baseline of 2/16 = 12.5%. MX block scaling slightly elevates this: the block
 * maximum is guaranteed non-zero, pushing small elements toward zero. Empirically the
 * zero frequency converges to ~12.89% for large matrices with bounded [-1, 1] input.
 */
TEST_P(MXDataGenFP4Test, ZeroFrequencyWithinBounds)
{
    auto [rows, cols, mxBlock, isTranspose] = GetParam();

    if(rows == 2048u && cols == 514u && mxBlock == 32 && !isTranspose)
        GTEST_SKIP() << "AIHPBLAS-3506: known segfault for this (rows, cols, mxBlock, isTranspose) "
                        "combination, first observed on gfx950/MI350. generateMXInput has no "
                        "architecture parameter, so this skip is unconditional (all hosts hit the "
                        "same data-shape bug); remove it once the underlying issue is fixed.";

    const uint64_t numElements  = rows * cols;
    const uint64_t numPacked    = (numElements + 1) / 2;
    const size_t   numScales    = ((rows + mxBlock - 1) / mxBlock) * cols;

    std::vector<uint8_t> dataBuffer(numPacked, 0);
    std::vector<uint8_t> scaleBuffer(numScales, 0);

    generateMXInput((hipDataType)HIP_R_4F_E2M1,
                    HIP_R_8F_UE8M0,
                    dataBuffer.data(),
                    scaleBuffer.data(),
                    rows,
                    cols,
                    rows, // stride = rows (column-major)
                    isTranspose,
                    mxBlock,
                    1,
                    true,
                    MXScaleLayout::None,
                    "Bounded",
                    -1.0f,
                    1.0f);

    size_t zeros       = countZerosFP4(dataBuffer.data(), numPacked);
    double zeroPercent = 100.0 * static_cast<double>(zeros) / static_cast<double>(numElements);

    EXPECT_LT(zeroPercent, 13.0)
        << "Zero frequency " << zeroPercent << "% exceeds 13% upper bound for "
        << rows << "x" << cols << " FP4 matrix (transpose=" << isTranspose << ")";

    // Ensure non-trivial data was actually generated (not all zeros)
    EXPECT_GT(numElements - zeros, 0u)
        << "All elements are zero for " << rows << "x" << cols << " FP4 matrix";
}

INSTANTIATE_TEST_SUITE_P(
    FP4ZeroFrequency,
    MXDataGenFP4Test,
    ::testing::Values(
        // rows, cols, mxBlock, isTranspose
        std::make_tuple(128u,  128u,  32, true),
        std::make_tuple(256u,  256u,  32, true),
        std::make_tuple(2048u, 1026u, 32, true),
        std::make_tuple(2048u, 514u,  32, false)
    )
);

/**
 * @brief Regression guard: generateMXInput must be deterministic (fixed seed).
 *
 * Any post-generation overwrite of the MXSA/MXSB buffers (e.g., the general
 * tensor-init loop in initializeCPUInputs) desynchronises the CPU reference
 * from GPU data, causing intermittent single-element validation failures.
 * rows=K (must be mxBlock-aligned), cols=M/N (need not be).
 */
class MXGeneratorDeterminismTest
    : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, int, bool, bool>>
{
};

TEST_P(MXGeneratorDeterminismTest, GeneratorOutputIsDeterministic)
{
    auto [rows, cols, mxBlock, isTranspose, isMatrixA] = GetParam();

    const size_t numPacked = (rows * cols + 1) / 2;
    const size_t numScales = (rows / mxBlock) * cols;

    std::vector<uint8_t> data1(numPacked);
    std::vector<uint8_t> data2(numPacked);
    std::vector<uint8_t> scale1(numScales, 0x00);
    std::vector<uint8_t> scale2(numScales, 0xFF); // sentinel: catches no-write if scale1==scale2 passes

    generateMXInput((hipDataType)HIP_R_4F_E2M1, HIP_R_8F_UE8M0,
                    data1.data(), scale1.data(),
                    rows, cols, rows, isTranspose,
                    mxBlock, 1, isMatrixA,
                    MXScaleLayout::None, "Bounded", -1.f, 1.f);

    generateMXInput((hipDataType)HIP_R_4F_E2M1, HIP_R_8F_UE8M0,
                    data2.data(), scale2.data(),
                    rows, cols, rows, isTranspose,
                    mxBlock, 1, isMatrixA,
                    MXScaleLayout::None, "Bounded", -1.f, 1.f);

    EXPECT_EQ(data1, data2)
        << "FP4 data is non-deterministic";
    EXPECT_EQ(scale1, scale2)
        << "Scale data is non-deterministic; any post-generation overwrite will corrupt validation";

    bool allZero = std::all_of(scale1.begin(), scale1.end(), [](uint8_t b){ return b == 0; });
    bool allOnes = std::all_of(scale1.begin(), scale1.end(), [](uint8_t b){ return b == 0xFF; });
    EXPECT_FALSE(allZero) << "Scale buffer is all-zero — generator did not write";
    EXPECT_FALSE(allOnes) << "Scale buffer is all-0xFF (max UE8M0 value) — generator likely failed; bounded [-1,1] input should produce varied scales";
}

INSTANTIATE_TEST_SUITE_P(
    GeneratorDeterminism,
    MXGeneratorDeterminismTest,
    ::testing::Values(
        // rows=K, cols=M or N  (tensorA.sizes()={K,M}, tensorB.sizes()={K,N})
        std::make_tuple(1024u, 128u, 32, true,  true),  // transposed A
        std::make_tuple(1024u, 128u, 32, false, false), // non-transposed B
        std::make_tuple(1024u, 204u, 32, true,  true),  // M=204, non-32-aligned (was failing)
        std::make_tuple(1024u, 213u, 32, true,  true),  // M=213, non-32-aligned (was failing)
        // 1x128 block scaling (MXBlockA/MXBlockB = 128).
        std::make_tuple(1024u, 128u, 128, true,  true),  // transposed A
        std::make_tuple(1024u, 128u, 128, false, false), // non-transposed B
        std::make_tuple(1024u, 204u, 128, true,  true)   // free dim not block-aligned
    )
);

// ============================================================================
// 1x128 block scaling
//
// Every other case in this file pins mxBlock to 32. These tests pin the
// generator's block granularity itself: with MXBlockA/MXBlockB = 128 one scale
// must cover 128 contiguous K elements, so the generator must emit K/128 scales
// per free-dim column -- not K/32. A generator that silently ignored mxBlock and
// kept 32-element blocks would still produce plausible-looking data, so the
// checks below are on the *granularity*, not just on the values being finite.
//
// Layout for these calls (isTranspose=true, isMatrixA=true): rows=K, cols=M,
// column-major with stride=rows, so element (k, m) lives at data index
// m*rows + k and its scale at scale index m*(rows/mxBlock) + k/mxBlock.
// ============================================================================

/** @brief UE8M0 exponent byte -> float scale (bias 127). */
static float decodeUE8M0(uint8_t byte)
{
    return std::ldexp(1.0f, static_cast<int>(byte) - 127);
}

/**
 * @brief The generator must write exactly (rows/mxBlock)*cols scale bytes.
 *
 * The buffer is sized for the 32-block case (4x too large for mxBlock=128) and
 * pre-filled with a sentinel that bounded [-1,1] input can never produce: 0xAB
 * decodes to 2^44, and scales for data bounded to [-1,1] have exponent <= 127.
 * If the generator ignored mxBlock and kept writing at 32 granularity, the tail
 * would be overwritten and the sentinel check would fail.
 */
TEST(MXDataGenBlock128, ScaleGranularityIsBlock128)
{
    constexpr uint64_t rows    = 1024; // K
    constexpr uint64_t cols    = 128; // M
    constexpr int      mxBlock = 128;

    const size_t numPacked     = (rows * cols + 1) / 2;
    const size_t expectedScales = (rows / mxBlock) * cols;
    const size_t oversized      = (rows / 32) * cols; // what a 32-block generator would write

    ASSERT_EQ(oversized, expectedScales * 4u);

    constexpr uint8_t kSentinel = 0xAB;
    std::vector<uint8_t> data(numPacked, 0);
    std::vector<uint8_t> scale(oversized, kSentinel);

    generateMXInput((hipDataType)HIP_R_4F_E2M1, HIP_R_8F_UE8M0,
                    data.data(), scale.data(),
                    rows, cols, rows, /*isTranspose=*/true,
                    mxBlock, 1, /*isMatrixA=*/true,
                    MXScaleLayout::None, "Bounded", -1.f, 1.f);

    size_t written = 0;
    for(size_t i = 0; i < expectedScales; ++i)
        if(scale[i] != kSentinel)
            ++written;
    EXPECT_GT(written, 0u) << "generator wrote no scales at mxBlock=128";

    for(size_t i = expectedScales; i < oversized; ++i)
        ASSERT_EQ(scale[i], kSentinel)
            << "scale byte " << i << " past the K/128 region was overwritten; the generator "
            << "is emitting scales at a finer granularity than mxBlock=128";
}

/**
 * @brief All 128 elements of an MX block must share one scale.
 *
 * Dequantized reference = fp4_value * blockScale, and FP4 E2M1 has only eight
 * magnitudes. So dividing each reference value by the scale read at 128
 * granularity must land back on that set. If the generator actually applied a
 * distinct scale per 32 elements, three quarters of the elements in each block
 * would be divided by the wrong power of two and fall off the FP4 grid.
 *
 * Parameterized over 32 as well so a mistake in the index mapping above fails
 * the known-good control case too, rather than silently passing at 128.
 */
class MXBlockScaleGranularityTest : public ::testing::TestWithParam<int>
{
};

TEST_P(MXBlockScaleGranularityTest, DequantizedValuesLieOnTheFP4Grid)
{
    const int mxBlock = GetParam();

    constexpr uint64_t rows = 512; // K
    constexpr uint64_t cols = 64; // M

    const size_t numPacked  = (rows * cols + 1) / 2;
    const size_t kBlocks    = rows / mxBlock;
    const size_t numScales  = kBlocks * cols;

    std::vector<uint8_t> data(numPacked, 0);
    std::vector<uint8_t> scale(numScales, 0);

    auto ref = generateMXInput((hipDataType)HIP_R_4F_E2M1, HIP_R_8F_UE8M0,
                               data.data(), scale.data(),
                               rows, cols, rows, /*isTranspose=*/true,
                               mxBlock, 1, /*isMatrixA=*/true,
                               MXScaleLayout::None, "Bounded", -1.f, 1.f);

    ASSERT_EQ(ref.size(), rows * cols);

    // FP4 E2M1 magnitudes.
    const std::vector<float> grid = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

    size_t checked = 0, nonZero = 0;
    for(uint64_t m = 0; m < cols; ++m)
    {
        for(uint64_t k = 0; k < rows; ++k)
        {
            const float s = decodeUE8M0(scale[m * kBlocks + k / mxBlock]);
            ASSERT_GT(s, 0.f) << "zero/denormal scale at m=" << m << " kblock=" << k / mxBlock;

            const float q = std::abs(ref[m * rows + k]) / s;
            const bool  onGrid
                = std::any_of(grid.begin(), grid.end(),
                              [&](float g) { return std::abs(q - g) <= 1e-3f * std::max(1.f, g); });
            ASSERT_TRUE(onGrid)
                << "dequantized value " << ref[m * rows + k] << " / scale " << s << " = " << q
                << " is not an FP4 E2M1 magnitude (m=" << m << ", k=" << k
                << ", mxBlock=" << mxBlock << "); the scale is not shared across the whole block";
            ++checked;
            if(q != 0.f)
                ++nonZero;
        }
    }
    EXPECT_EQ(checked, rows * cols);
    EXPECT_GT(nonZero, 0u) << "all-zero data proves nothing about scale granularity";
}

INSTANTIATE_TEST_SUITE_P(BlockSizes,
                         MXBlockScaleGranularityTest,
                         ::testing::Values(32, 128));

// ============================================================================
// PreSwizzle scale tests
//
// Verify generateMXInput with MXScaleLayout::GFX950 produces scale data
// that is a permutation of the unswizzled (None) layout. The actual
// swizzle parameters (swizzleTileMN=32, tileK=8, subTileK=MiK/mxBlock) are
// hard-coded inside `generateMXInput` -- callers just pick the layout.
// ============================================================================

// Params: {rows, cols, mxBlock, isTranspose, isMatrixA}
class MXPreSwizzleTest
    : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, int, bool, bool>>
{
};

/** @brief Verify the gfx950 swizzle produces a non-trivial permutation of scale data. */
TEST_P(MXPreSwizzleTest, ScaleIsPermutationOfUnswizzled)
{
    auto [rows, cols, mxBlock, isTranspose, isMatrixA] = GetParam();

    const uint64_t numElements  = rows * cols;
    const uint64_t numPacked    = (numElements + 1) / 2;
    const size_t   numScales    = ((rows + mxBlock - 1) / mxBlock) * cols;

    std::vector<uint8_t> dataNoShuf(numPacked, 0);
    std::vector<uint8_t> scaleNoShuf(numScales, 0);
    std::vector<uint8_t> dataShuf(numPacked, 0);
    std::vector<uint8_t> scaleShuf(numScales, 0);

    // Generate without preSwizzle
    generateMXInput((hipDataType)HIP_R_4F_E2M1, HIP_R_8F_UE8M0,
                    dataNoShuf.data(),
                    scaleNoShuf.data(),
                    rows, cols, rows,
                    isTranspose,
                    mxBlock, 1, isMatrixA,
                    MXScaleLayout::None,
                    "Bounded", -1.0f, 1.0f);

    // Generate with preSwizzle
    generateMXInput((hipDataType)HIP_R_4F_E2M1, HIP_R_8F_UE8M0,
                    dataShuf.data(),
                    scaleShuf.data(),
                    rows, cols, rows,
                    isTranspose,
                    mxBlock, 1, isMatrixA,
                    MXScaleLayout::GFX950,
                    "Bounded", -1.0f, 1.0f);

    // The scale buffers must be different
    EXPECT_NE(scaleNoShuf, scaleShuf)
        << "Scale data was not shuffled for " << rows << "x" << cols
        << " (transpose=" << isTranspose << ", isMatrixA=" << isMatrixA << ")";

    // The shuffled scale must be a permutation: same multiset of bytes
    std::vector<uint8_t> sortedNoShuf = scaleNoShuf;
    std::vector<uint8_t> sortedShuf   = scaleShuf;
    std::sort(sortedNoShuf.begin(), sortedNoShuf.end());
    std::sort(sortedShuf.begin(), sortedShuf.end());
    EXPECT_EQ(sortedNoShuf, sortedShuf)
        << "Pre-shuffled scale is not a permutation of the unshuffled scale for "
        << rows << "x" << cols;

    // Data buffer must be identical (preSwizzle only affects scale, not data)
    EXPECT_EQ(dataNoShuf, dataShuf)
        << "Data buffer changed unexpectedly with preSwizzle for "
        << rows << "x" << cols;
}

INSTANTIATE_TEST_SUITE_P(
    FP4PreSwizzle,
    MXPreSwizzleTest,
    ::testing::Values(
        // rows, cols, mxBlock, isTranspose, isMatrixA
        // Test size constraints for preSwizzle {32,8,4} + preTile {8,32}:
        //   rows % 256 == 0  (scaleRows = rows/mxBlock must be divisible by tileK=8)
        //   cols % 32  == 0  (scaleCols must be divisible by swizzleTileMN=32)
        std::make_tuple(256u,  256u,  32, true,  true),   // scale A transposed
        std::make_tuple(256u,  256u,  32, false, false),  // scale B non-transposed
        std::make_tuple(512u,  256u,  32, true,  true),   // larger scale A
        std::make_tuple(256u,  512u,  32, false, false),  // larger scale B
        std::make_tuple(4096u, 16384u, 32, true, true)    // benchmark-scale problem
    )
);

// ============================================================================
// Init-mode coverage
//
// generateMXInput accepts hipblaslt-level init-method strings beyond the
// "Bounded" default exercised above. These tests cover the newly-wired modes
// "zero", "norm_dist", "rand_int", and "uniform_low_precision" for both FP4
// (E2M1) and an FP8 (E4M3) dtype. The returned vector is the dequantized
// reference float (data * per-block scale), which is what callers validate
// against.
// ============================================================================

// Params: {dataType, initMethod}
class MXDataGenModeTest
    : public ::testing::TestWithParam<std::tuple<hipDataType, std::string>>
{
public:
    // OCP FP4 E2M1 max-normal magnitude; "uniform_low_precision" draws data
    // uniformly from [-6, 6], so dequantized values must stay within that range.
    static constexpr float FP4E2M1Max = 6.0f;

    // Run generateMXInput for one (dtype, mode) and return the dequantized
    // reference floats. FP4 packs two elements per byte; FP8 is one byte per element.
    static std::vector<float> runMXMode(hipDataType            dataType,
                                        std::string_view const initMethod,
                                        uint64_t               rows    = 256,
                                        uint64_t               cols    = 256,
                                        int                    mxBlock = 32)
    {
        const uint64_t numElements  = rows * cols;
        const size_t   bytesPerData = (dataType == (hipDataType)HIP_R_4F_E2M1)
                                           ? (numElements + 1) / 2
                                           : numElements;
        const size_t   numScales    = (rows / mxBlock) * cols;

        std::vector<uint8_t> dataBuffer(bytesPerData, 0);
        std::vector<uint8_t> scaleBuffer(numScales, 0);

        return generateMXInput(dataType,
                               HIP_R_8F_UE8M0,
                               dataBuffer.data(),
                               scaleBuffer.data(),
                               rows,
                               cols,
                               rows, // stride = rows (column-major)
                               /*isTranspose=*/true,
                               mxBlock,
                               1,
                               /*isMatrixA=*/true,
                               MXScaleLayout::None,
                               initMethod,
                               -1.0f,
                               1.0f);
    }
};

/** @brief Every wired init mode must generate without throwing and yield finite data. */
TEST_P(MXDataGenModeTest, GeneratesFiniteOutputWithoutThrowing)
{
    auto [dataType, initMethod] = GetParam();

    std::vector<float> ref;
    ASSERT_NO_THROW(ref = runMXMode(dataType, initMethod))
        << "generateMXInput threw for init mode '" << initMethod << "'";
    EXPECT_FALSE(ref.empty()) << "no reference data produced for '" << initMethod << "'";
    for(float v : ref)
        EXPECT_TRUE(std::isfinite(v))
            << "non-finite value " << v << " for init mode '" << initMethod << "'";
}

INSTANTIATE_TEST_SUITE_P(
    InitModes,
    MXDataGenModeTest,
    ::testing::Combine(
        ::testing::Values((hipDataType)HIP_R_4F_E2M1, HIP_R_8F_E4M3),
        ::testing::Values(std::string("zero"),
                          std::string("norm_dist"),
                          std::string("rand_int"),
                          std::string("uniform_low_precision"))
    )
);

/** @brief "zero" mode must produce all-zero dequantized output. */
TEST(MXDataGenInitMode, ZeroFP4IsAllZero)
{
    auto ref = MXDataGenModeTest::runMXMode((hipDataType)HIP_R_4F_E2M1, "zero");
    ASSERT_FALSE(ref.empty());
    for(float v : ref)
        EXPECT_EQ(v, 0.0f) << "zero mode produced non-zero value " << v;
}

TEST(MXDataGenInitMode, ZeroFP8IsAllZero)
{
    auto ref = MXDataGenModeTest::runMXMode(HIP_R_8F_E4M3, "zero");
    ASSERT_FALSE(ref.empty());
    for(float v : ref)
        EXPECT_EQ(v, 0.0f) << "zero mode produced non-zero value " << v;
}

/**
 * @brief "uniform_low_precision" draws from [-6, 6] (full FP4 E2M1 range).
 *
 * Dequantized values = fp4_quantized * UE8M0_scale. With input bounded to
 * [-6, 6] the per-block scale is <= 1 and fp4 magnitudes are <= 6, so the
 * dequantized magnitude can never exceed 6. A tiny epsilon guards float math.
 */
TEST(MXDataGenInitMode, UniformLowPrecisionFP4WithinRange)
{
    auto ref = MXDataGenModeTest::runMXMode((hipDataType)HIP_R_4F_E2M1, "uniform_low_precision");
    ASSERT_FALSE(ref.empty());

    bool anyNonZero = false;
    for(float v : ref)
    {
        EXPECT_TRUE(std::isfinite(v)) << "non-finite uniform_low_precision value " << v;
        EXPECT_LE(std::abs(v), MXDataGenModeTest::FP4E2M1Max + 1e-3f)
            << "uniform_low_precision value " << v << " outside [-6, 6]";
        anyNonZero = anyNonZero || (v != 0.0f);
    }
    EXPECT_TRUE(anyNonZero) << "uniform_low_precision produced all-zero data";
}

// Phase 1: decoupled scale init -- random data with unity scales (--init-a=Random,
// --init-mx-a=One). Scale bytes must all encode 1.0; dequantized data must vary.
TEST(MXDataGenDecoupledScale, BoundedDataWithUnityScales)
{
    constexpr uint64_t rows    = 256;
    constexpr uint64_t cols    = 256;
    constexpr int      mxBlock = 32;

    const size_t numPacked = (rows * cols + 1) / 2;
    const size_t numScales = (rows / mxBlock) * cols;

    std::vector<uint8_t> dataBuffer(numPacked, 0);
    std::vector<uint8_t> scaleBuffer(numScales, 0);

    auto ref = generateMXInput((hipDataType)HIP_R_4F_E2M1,
                               HIP_R_8F_UE8M0,
                               dataBuffer.data(),
                               scaleBuffer.data(),
                               rows,
                               cols,
                               rows,
                               /*isTranspose=*/true,
                               mxBlock,
                               1,
                               /*isMatrixA=*/true,
                               MXScaleLayout::None,
                               "Bounded",
                               -1.0f,
                               1.0f,
                               "Ones");

    ASSERT_FALSE(ref.empty());

    // UE8M0 unity scale byte (E8M0_1 = 127 = 0x7F).
    constexpr uint8_t kUnityScale = 0x7F;
    for(uint8_t s : scaleBuffer)
        EXPECT_EQ(s, kUnityScale) << "expected unity UE8M0 scale byte";

    int meaningful = 0;
    for(float v : ref)
    {
        EXPECT_TRUE(std::isfinite(v));
        if(std::abs(v) > 1e-4f)
            ++meaningful;
    }
    EXPECT_GT(meaningful, 0)
        << "Bounded data with Ones scales should produce representable dequantized values";

    std::set<uint8_t> uniqueData(dataBuffer.begin(), dataBuffer.end());
    EXPECT_GT(uniqueData.size(), 1u)
        << "packed data should not collapse to a single byte pattern";
}

TEST(MXDataGenDecoupledScale, MatchingInitUnchangedFromDefault)
{
    constexpr uint64_t rows    = 128;
    constexpr uint64_t cols    = 128;
    constexpr int      mxBlock = 32;

    const size_t numPacked = (rows * cols + 1) / 2;
    const size_t numScales = (rows / mxBlock) * cols;

    std::vector<uint8_t> dataA(numPacked);
    std::vector<uint8_t> dataB(numPacked);
    std::vector<uint8_t> scaleA(numScales, 0);
    std::vector<uint8_t> scaleB(numScales, 0);

    generateMXInput((hipDataType)HIP_R_4F_E2M1,
                    HIP_R_8F_UE8M0,
                    dataA.data(),
                    scaleA.data(),
                    rows,
                    cols,
                    rows,
                    true,
                    mxBlock,
                    1,
                    true,
                    MXScaleLayout::None,
                    "Bounded",
                    -1.f,
                    1.f);

    generateMXInput((hipDataType)HIP_R_4F_E2M1,
                    HIP_R_8F_UE8M0,
                    dataB.data(),
                    scaleB.data(),
                    rows,
                    cols,
                    rows,
                    true,
                    mxBlock,
                    1,
                    true,
                    MXScaleLayout::None,
                    "Bounded",
                    -1.f,
                    1.f,
                    "Bounded");

    EXPECT_EQ(dataA, dataB);
    EXPECT_EQ(scaleA, scaleB);
}

TEST(MXScaleLayoutArch, MapsArchNameToScaleLayout)
{
    EXPECT_EQ(mxScaleLayoutForArchName("gfx950"), MXScaleLayout::GFX950);
    EXPECT_EQ(mxScaleLayoutForArchName("gfx950:sramecc+:xnack-"), MXScaleLayout::GFX950);
    EXPECT_EQ(mxScaleLayoutForArchName("gfx1250"), MXScaleLayout::GFX1250);
    EXPECT_EQ(mxScaleLayoutForArchName("gfx942"), MXScaleLayout::None);
    EXPECT_EQ(mxScaleLayoutForArchName("gfx90a"), MXScaleLayout::None);
}

TEST(MXScaleRestride, ExpandsKFastRowsInPlace)
{
    // Canonical K-fast layout: 2 free rows x 4 K-blocks, padded to 8 K-blocks per row.
    std::vector<uint8_t> scale(16, 0);
    for(size_t i = 0; i < 8; ++i)
        scale[i] = static_cast<uint8_t>(i + 1);

    restrideMXScaleBufferKFast(scale.data(), /*compactFreeDim=*/2, /*compactKBlocks=*/4,
                                 /*paddedKBlocks=*/8, /*elemBytes=*/1);

    EXPECT_EQ(scale[0], 1);
    EXPECT_EQ(scale[3], 4);
    EXPECT_EQ(scale[4], 0);
    EXPECT_EQ(scale[7], 0);
    EXPECT_EQ(scale[8], 5);
    EXPECT_EQ(scale[11], 8);
    EXPECT_EQ(scale[12], 0);
    EXPECT_EQ(scale[15], 0);
}

TEST(MXScaleLayoutFormat, MapsScalingFormatToLayout)
{
    using SF = hipblaslt_scaling_format;
    EXPECT_EQ(mxScaleLayoutForFormat(SF::Block_32_UE8M0_32_8_EXT, "gfx950"),
              MXScaleLayout::GFX950);
    EXPECT_EQ(mxScaleLayoutForFormat(SF::Block_32_UE8M0, "gfx950"), MXScaleLayout::None);
    EXPECT_EQ(mxScaleLayoutForFormat(SF::Block_32_UE8M0, "gfx1250"), MXScaleLayout::GFX1250);
    EXPECT_EQ(mxScaleLayoutForFormat(SF::Block_32_UE4M3, "gfx942"), MXScaleLayout::None);
}

// ---------------------------------------------------------------------------
// maf2_0 MX data, checked with decoders written independently of mxDataGenerator.
// ---------------------------------------------------------------------------

// OCP FP8 value (E4M3: bias 7, 0x7F/0xFF NaN; E5M2: bias 15, exponent 31 Inf/NaN).
static double decodeOcpFp8(uint8_t code, bool e4m3)
{
    const int    expBits  = e4m3 ? 4 : 5;
    const int    manBits  = e4m3 ? 3 : 2;
    const int    bias     = e4m3 ? 7 : 15;
    const int    exponent = (code >> manBits) & ((1 << expBits) - 1);
    const int    mantissa = code & ((1 << manBits) - 1);
    const double sign     = (code & 0x80) ? -1.0 : 1.0;
    if(e4m3 && (code & 0x7F) == 0x7F)
        return std::nan("");
    if(!e4m3 && exponent == 31)
        return mantissa == 0 ? sign * INFINITY : std::nan("");
    if(exponent == 0)
        return sign * std::ldexp(mantissa, 1 - bias - manBits);
    return sign * std::ldexp((1 << manBits) + mantissa, exponent - bias - manBits);
}

struct MXOut
{
    std::vector<uint8_t> data;
    std::vector<uint8_t> scale;
    std::vector<float>   ref;
};

// A transposed A of rows x cols (K = rows, contiguous): one scale per mxBlock elements.
static MXOut runFp8MX(hipDataType            dataType,
                      hipDataType            scaleType,
                      std::string_view const initMethod,
                      uint32_t               seedIndex = 0,
                      int                    mxBlock   = 32,
                      uint64_t               rows      = 256,
                      uint64_t               cols      = 64,
                      MXScaleLayout          layout    = MXScaleLayout::None)
{
    MXOut out;
    out.data.assign(rows * cols, 0);
    out.scale.assign((rows / mxBlock) * cols * 2, 0); // room for swizzle padding
    out.ref = generateMXInput(dataType,
                              scaleType,
                              out.data.data(),
                              out.scale.data(),
                              rows,
                              cols,
                              rows,
                              /*isTranspose=*/true,
                              mxBlock,
                              1,
                              /*isMatrixA=*/true,
                              layout,
                              initMethod,
                              -1.0f,
                              1.0f,
                              "",
                              seedIndex);
    return out;
}

class MXDataGenMaf2_0 : public ::testing::TestWithParam<hipDataType>
{
};

TEST_P(MXDataGenMaf2_0, E8M0ScalesAreBinomialPowersOfTwoAndDataIsFinite)
{
    const hipDataType dt   = GetParam();
    const bool        e4m3 = dt == HIP_R_8F_E4M3;
    const auto        out  = runFp8MX(dt, HIP_R_8F_UE8M0, "maf2_0");

    std::set<uint8_t> scales(out.scale.begin(), out.scale.begin() + (256 / 32) * 64);
    for(uint8_t s : scales)
    {
        EXPECT_GE(s, 127 - 11) << "scale exponent below 2^-11";
        EXPECT_LE(s, 127 + 10) << "scale exponent above 2^10";
    }
    EXPECT_GT(scales.size(), 5u) << "binomial scales should vary";

    for(size_t i = 0; i < out.data.size(); ++i)
    {
        const double v = decodeOcpFp8(out.data[i], e4m3);
        ASSERT_TRUE(std::isfinite(v)) << "i=" << i << " code=" << int(out.data[i]);
        const double expected = v * std::ldexp(1.0, int(out.scale[i / 32]) - 127);
        EXPECT_EQ(out.ref[i], static_cast<float>(expected)) << "i=" << i;
    }
}

TEST_P(MXDataGenMaf2_0, SeedIndexSeparatesMatricesAndBatches)
{
    const hipDataType dt = GetParam();
    const auto        a0 = runFp8MX(dt, HIP_R_8F_UE8M0, "maf2_0", 0);
    const auto        a0again = runFp8MX(dt, HIP_R_8F_UE8M0, "maf2_0", 0);
    const auto        b0 = runFp8MX(dt, HIP_R_8F_UE8M0, "maf2_0", 1);
    const auto        a1 = runFp8MX(dt, HIP_R_8F_UE8M0, "maf2_0", 3);
    EXPECT_EQ(a0.data, a0again.data);
    EXPECT_EQ(a0.scale, a0again.scale);
    EXPECT_NE(a0.data, b0.data) << "A and B must differ";
    EXPECT_NE(a0.data, a1.data) << "batches must differ";
}

TEST_P(MXDataGenMaf2_0, OtherInitMethodsIgnoreSeedIndex)
{
    const hipDataType dt = GetParam();
    const auto        x  = runFp8MX(dt, HIP_R_8F_UE8M0, "Bounded", 0);
    const auto        y  = runFp8MX(dt, HIP_R_8F_UE8M0, "Bounded", 4);
    EXPECT_EQ(x.data, y.data);
    EXPECT_EQ(x.scale, y.scale);
}

INSTANTIATE_TEST_SUITE_P(Fp8, MXDataGenMaf2_0, ::testing::Values(HIP_R_8F_E4M3, HIP_R_8F_E5M2));

// OCP FP6 value (E2M3: bias 1; E3M2: bias 3); no Inf/NaN encodings.
static double decodeOcpFp6(uint8_t code, bool e2m3)
{
    const int    expBits  = e2m3 ? 2 : 3;
    const int    manBits  = e2m3 ? 3 : 2;
    const int    bias     = e2m3 ? 1 : 3;
    const int    exponent = (code >> manBits) & ((1 << expBits) - 1);
    const int    mantissa = code & ((1 << manBits) - 1);
    const double sign     = (code & 0x20) ? -1.0 : 1.0;
    if(exponent == 0)
        return sign * std::ldexp(mantissa, 1 - bias - manBits);
    return sign * std::ldexp((1 << manBits) + mantissa, exponent - bias - manBits);
}

// Element i of a packed FP6 buffer (6 bits per element, little-endian bit order).
static uint8_t fp6At(std::vector<uint8_t> const& packed, size_t i)
{
    size_t const bit  = i * 6;
    uint16_t     word = packed[bit / 8];
    if(bit / 8 + 1 < packed.size())
        word |= static_cast<uint16_t>(packed[bit / 8 + 1]) << 8;
    return static_cast<uint8_t>((word >> (bit % 8)) & 0x3F);
}

class MXDataGenMaf2_0Fp6 : public ::testing::TestWithParam<hipDataType>
{
};

TEST_P(MXDataGenMaf2_0Fp6, E8M0ScalesAreBinomialPowersOfTwo)
{
    const hipDataType dt   = GetParam();
    const bool        e2m3 = dt == static_cast<hipDataType>(HIP_R_6F_E2M3);
    const auto        out  = runFp8MX(dt, HIP_R_8F_UE8M0, "maf2_0");
    for(size_t i = 0; i < out.ref.size(); ++i)
    {
        const uint8_t s = out.scale[i / 32];
        ASSERT_GE(s, 127 - 11);
        ASSERT_LE(s, 127 + 10);
        const double expected = decodeOcpFp6(fp6At(out.data, i), e2m3) * std::ldexp(1.0, s - 127);
        ASSERT_EQ(out.ref[i], static_cast<float>(expected)) << "i=" << i;
    }
    const auto b = runFp8MX(dt, HIP_R_8F_UE8M0, "maf2_0", 1);
    EXPECT_NE(out.data, b.data) << "A and B must differ";
}

INSTANTIATE_TEST_SUITE_P(Fp6,
                         MXDataGenMaf2_0Fp6,
                         ::testing::Values(static_cast<hipDataType>(HIP_R_6F_E2M3),
                                           static_cast<hipDataType>(HIP_R_6F_E3M2)));

// OCP FP4 E2M1 value (bias 1); element i is the low nibble of byte i/2 for even i.
static double decodeFp4At(std::vector<uint8_t> const& packed, size_t i)
{
    const uint8_t code     = (packed[i / 2] >> ((i % 2) * 4)) & 0xF;
    const int     exponent = (code >> 1) & 0x3;
    const int     mantissa = code & 0x1;
    const double  sign     = (code & 0x8) ? -1.0 : 1.0;
    if(exponent == 0)
        return sign * 0.5 * mantissa;
    return sign * std::ldexp(2 + mantissa, exponent - 2);
}

// Unsigned E4M3 / E5M3 scale values (payload 0x7F / 0xFF is NaN).
static double decodeUe4m3Scale(uint8_t code)
{
    return decodeOcpFp8(code & 0x7F, /*e4m3=*/true);
}

static double decodeUe5m3Scale(uint8_t code)
{
    if(code == 0xFF)
        return std::nan("");
    const int e = (code >> 3) & 0x1F;
    const int m = code & 0x7;
    return e == 0 ? std::ldexp(m, 1 - 15 - 3) : std::ldexp(8 + m, e - 15 - 3);
}

struct Fp4ScaleCase
{
    hipDataType scaleType;
    int         mxBlock;
};

class MXDataGenMaf2_0Fp4 : public ::testing::TestWithParam<Fp4ScaleCase>
{
};

// FP4 maf2_0 blocks take |N(0.34375, 0.08)| scales rounded to the scale format, and data
// uniform over the 16 FP4 codes.
TEST_P(MXDataGenMaf2_0Fp4, NormalScalesAndDecodedReference)
{
    const auto        c   = GetParam();
    const hipDataType f4  = static_cast<hipDataType>(HIP_R_4F_E2M1);
    const auto        out = runFp8MX(f4, c.scaleType, "maf2_0", 0, c.mxBlock);

    auto scaleValue = [&](uint8_t s) {
        if(c.scaleType == HIP_R_8F_E4M3)
            return decodeUe4m3Scale(s);
        if(c.scaleType == static_cast<hipDataType>(HIP_R_8F_E5M3_EXT))
            return decodeUe5m3Scale(s);
        return std::ldexp(1.0, int(s) - 127);
    };

    const size_t blocks = out.ref.size() / c.mxBlock;
    double       sum = 0, quarter = 0;
    for(size_t b = 0; b < blocks; ++b)
    {
        const double v = scaleValue(out.scale[b]);
        ASSERT_TRUE(std::isfinite(v)) << "block " << b;
        sum += v;
        quarter += v == 0.25;
    }
    if(c.scaleType == HIP_R_8F_UE8M0)
        EXPECT_NEAR(quarter / blocks, 0.627, 0.1) << "E8M0 rounds most draws to 0.25";
    else
        EXPECT_NEAR(sum / blocks, 0.3435, 0.015) << "mean scale";

    for(size_t i = 0; i < out.ref.size(); ++i)
    {
        const double expected = decodeFp4At(out.data, i) * scaleValue(out.scale[i / c.mxBlock]);
        ASSERT_EQ(out.ref[i], static_cast<float>(expected)) << "i=" << i;
    }

    const auto b = runFp8MX(f4, c.scaleType, "maf2_0", 1, c.mxBlock);
    EXPECT_NE(out.data, b.data) << "A and B must differ";
}

INSTANTIATE_TEST_SUITE_P(
    Fp4,
    MXDataGenMaf2_0Fp4,
    ::testing::Values(Fp4ScaleCase{HIP_R_8F_UE8M0, 32},
                      Fp4ScaleCase{HIP_R_8F_E4M3, 16},
                      Fp4ScaleCase{static_cast<hipDataType>(HIP_R_8F_E5M3_EXT), 16}));

class MXDataGenMaf2_0UnitScale : public ::testing::TestWithParam<Fp4ScaleCase>
{
};

// The FP4 non-scaled row: every block scale is 1.0 and the data is N(0, 1) rounded to FP4.
TEST_P(MXDataGenMaf2_0UnitScale, UnitScalesAndRoundedNormalData)
{
    const auto        c   = GetParam();
    const hipDataType f4  = static_cast<hipDataType>(HIP_R_4F_E2M1);
    const auto        out = runFp8MX(f4, c.scaleType, "maf2_0_unit_scale", 0, c.mxBlock);

    const uint8_t one = c.scaleType == HIP_R_8F_E4M3                                    ? 0x38
                        : c.scaleType == static_cast<hipDataType>(HIP_R_8F_E5M3_EXT) ? 0x78
                                                                                     : 0x7F;
    const size_t  blocks = out.ref.size() / c.mxBlock;
    for(size_t b = 0; b < blocks; ++b)
        ASSERT_EQ(out.scale[b], one) << "block " << b;

    double zeros = 0, small = 0;
    for(size_t i = 0; i < out.ref.size(); ++i)
    {
        const double v = decodeFp4At(out.data, i);
        ASSERT_EQ(out.ref[i], static_cast<float>(v)) << "i=" << i;
        zeros += v == 0.0;
        small += std::abs(v) <= 1.0;
    }
    // P(|x| < 0.25) = 0.197 and P(|x| < 1.25) = 0.789 for x ~ N(0, 1).
    EXPECT_NEAR(zeros / out.ref.size(), 0.197, 0.02);
    EXPECT_NEAR(small / out.ref.size(), 0.789, 0.02);

    const auto b = runFp8MX(f4, c.scaleType, "maf2_0_unit_scale", 1, c.mxBlock);
    EXPECT_NE(out.data, b.data) << "A and B must differ";
}

INSTANTIATE_TEST_SUITE_P(
    Fp4,
    MXDataGenMaf2_0UnitScale,
    ::testing::Values(Fp4ScaleCase{HIP_R_8F_UE8M0, 32},
                      Fp4ScaleCase{HIP_R_8F_E4M3, 16},
                      Fp4ScaleCase{static_cast<hipDataType>(HIP_R_8F_E5M3_EXT), 16}));

// ---------------------------------------------------------------------------
// maf1_0 MX data: every block scale is 0.5, and the code of stored element n (= i + j * rows) is
// sin (A, seedIndex % 3 == 0) or cos (B) of n + batch * rows * cols (batch = seedIndex / 3),
// rounded once to nearest even at unit scale, so its value is 0.5 * code. Checked against a
// brute-force rounding over every encoding.
// ---------------------------------------------------------------------------

struct Maf1_0Case
{
    hipDataType dataType;
    hipDataType scaleType;
    int         mxBlock;
    uint8_t     halfScale; // 0.5 in the scale format
};

static bool isFp8(hipDataType dt)
{
    return dt == HIP_R_8F_E4M3 || dt == HIP_R_8F_E5M2;
}

static bool isFp4(hipDataType dt)
{
    return dt == static_cast<hipDataType>(HIP_R_4F_E2M1);
}

static uint8_t maf1_0SignBit(hipDataType dt)
{
    return isFp8(dt) ? 0x80 : isFp4(dt) ? 0x8 : 0x20;
}

// Code of element n of a packed data buffer.
static uint8_t maf1_0CodeAt(hipDataType dt, std::vector<uint8_t> const& data, size_t n)
{
    if(isFp8(dt))
        return data[n];
    if(isFp4(dt))
        return (data[n / 2] >> ((n % 2) * 4)) & 0xF;
    return fp6At(data, n);
}

static double maf1_0Decode(hipDataType dt, uint8_t code)
{
    if(isFp8(dt))
        return decodeOcpFp8(code, dt == HIP_R_8F_E4M3);
    if(isFp4(dt))
        return decodeFp4At(std::vector<uint8_t>{code}, 0);
    return decodeOcpFp6(code, dt == static_cast<hipDataType>(HIP_R_6F_E2M3));
}

// The finite code nearest to v (the even one on a tie), carrying the sign of v.
static uint8_t maf1_0NearestEven(hipDataType dt, double v)
{
    uint8_t const signBit = maf1_0SignBit(dt);
    int           best    = 0;
    double        bestErr = INFINITY;
    for(int c = 0; c < signBit; ++c)
    {
        double const d = maf1_0Decode(dt, static_cast<uint8_t>(c));
        if(!std::isfinite(d))
            continue;
        double const err = std::abs(d - std::abs(v));
        if(err < bestErr || (err == bestErr && c % 2 == 0))
        {
            best    = c;
            bestErr = err;
        }
    }
    return static_cast<uint8_t>(best | (std::signbit(v) ? signBit : 0));
}

static double maf1_0Trig(uint32_t seedIndex, size_t n, size_t elements)
{
    double const x = static_cast<double>(n + seedIndex / 3 * elements);
    return seedIndex % 3 == 0 ? std::sin(x) : std::cos(x);
}

class MXDataGenMaf1_0 : public ::testing::TestWithParam<Maf1_0Case>
{
};

TEST_P(MXDataGenMaf1_0, HalfScalesAndRoundedTrigData)
{
    const auto     c        = GetParam();
    const uint64_t rows     = 256;
    const uint64_t cols     = 64;
    const size_t   elements = rows * cols;
    for(uint32_t seedIndex : {0u, 1u, 3u, 4u}) // A, B, batch-1 A, batch-1 B
    {
        SCOPED_TRACE("seedIndex " + std::to_string(seedIndex));
        const auto out = runFp8MX(c.dataType, c.scaleType, "maf1_0", seedIndex, c.mxBlock);
        for(size_t b = 0; b < (rows / c.mxBlock) * cols; ++b)
            ASSERT_EQ(out.scale[b], c.halfScale) << "block " << b;
        ASSERT_EQ(out.ref.size(), elements);
        for(size_t n = 0; n < elements; ++n)
        {
            const uint8_t code = maf1_0CodeAt(c.dataType, out.data, n);
            ASSERT_EQ(code, maf1_0NearestEven(c.dataType, maf1_0Trig(seedIndex, n, elements)))
                << "n=" << n;
            ASSERT_EQ(out.ref[n], static_cast<float>(0.5 * maf1_0Decode(c.dataType, code)))
                << "n=" << n;
        }
    }
}

TEST_P(MXDataGenMaf1_0, MatricesAndBatchesDifferAndRepeat)
{
    const auto c       = GetParam();
    const auto a0      = runFp8MX(c.dataType, c.scaleType, "maf1_0", 0, c.mxBlock);
    const auto a0again = runFp8MX(c.dataType, c.scaleType, "maf1_0", 0, c.mxBlock);
    const auto b0      = runFp8MX(c.dataType, c.scaleType, "maf1_0", 1, c.mxBlock);
    const auto a1      = runFp8MX(c.dataType, c.scaleType, "maf1_0", 3, c.mxBlock);
    const auto b1      = runFp8MX(c.dataType, c.scaleType, "maf1_0", 4, c.mxBlock);
    EXPECT_EQ(a0.data, a0again.data);
    EXPECT_EQ(a0.scale, a0again.scale);
    EXPECT_EQ(a0.ref, a0again.ref);
    EXPECT_NE(a0.data, b0.data) << "A (sin) and B (cos) must differ";
    EXPECT_NE(a0.data, a1.data) << "batches must differ";
    EXPECT_NE(b0.data, b1.data) << "batches must differ";
}

// With K along the stored columns (non-transposed A) the index is still i + j * rows of the
// stored matrix.
TEST_P(MXDataGenMaf1_0, IndexFollowsTheStoredMatrix)
{
    const auto           c    = GetParam();
    const uint64_t       rows = 64;
    const uint64_t       cols = 256;
    std::vector<uint8_t> data(rows * cols, 0);
    std::vector<uint8_t> scale(rows * cols, 0);
    generateMXInput(c.dataType,
                    c.scaleType,
                    data.data(),
                    scale.data(),
                    rows,
                    cols,
                    rows,
                    /*isTranspose=*/false,
                    c.mxBlock,
                    1,
                    /*isMatrixA=*/true,
                    MXScaleLayout::None,
                    "maf1_0",
                    -1.0f,
                    1.0f,
                    "",
                    0);
    for(size_t b = 0; b < (rows / c.mxBlock) * cols; ++b)
        ASSERT_EQ(scale[b], c.halfScale) << "block " << b;
    for(size_t n = 0; n < rows * cols; ++n)
        ASSERT_EQ(maf1_0CodeAt(c.dataType, data, n),
                  maf1_0NearestEven(c.dataType, std::sin(static_cast<double>(n))))
            << "n=" << n;
}

INSTANTIATE_TEST_SUITE_P(Fp8,
                         MXDataGenMaf1_0,
                         ::testing::Values(Maf1_0Case{HIP_R_8F_E4M3, HIP_R_8F_UE8M0, 32, 0x7E},
                                           Maf1_0Case{HIP_R_8F_E5M2, HIP_R_8F_UE8M0, 32, 0x7E},
                                           Maf1_0Case{HIP_R_8F_E4M3, HIP_R_8F_UE8M0, 16, 0x7E}));

INSTANTIATE_TEST_SUITE_P(
    Fp6,
    MXDataGenMaf1_0,
    ::testing::Values(Maf1_0Case{static_cast<hipDataType>(HIP_R_6F_E2M3), HIP_R_8F_UE8M0, 32, 0x7E},
                      Maf1_0Case{
                          static_cast<hipDataType>(HIP_R_6F_E3M2), HIP_R_8F_UE8M0, 16, 0x7E}));

INSTANTIATE_TEST_SUITE_P(
    Fp4,
    MXDataGenMaf1_0,
    ::testing::Values(Maf1_0Case{static_cast<hipDataType>(HIP_R_4F_E2M1), HIP_R_8F_UE8M0, 32, 0x7E},
                      Maf1_0Case{static_cast<hipDataType>(HIP_R_4F_E2M1), HIP_R_8F_E4M3, 16, 0x30},
                      Maf1_0Case{static_cast<hipDataType>(HIP_R_4F_E2M1),
                                 static_cast<hipDataType>(HIP_R_8F_E5M3_EXT),
                                 16,
                                 0x70}));

// Hand-computed codes: E4M3 sin(1) = 0.8415 -> 0.8125 (0x35), sin(2) = 0.9093 -> 0.875 (0x37),
// cos(0) = 1 (0x38); FP4 sin(1) -> 1.0 (0x2), element 1 being the high nibble of byte 0.
TEST(MXDataGenMaf1_0Golden, HandComputedCodes)
{
    const auto a = runFp8MX(HIP_R_8F_E4M3, HIP_R_8F_UE8M0, "maf1_0", 0);
    EXPECT_EQ(a.data[0], 0x00);
    EXPECT_EQ(a.data[1], 0x35);
    EXPECT_EQ(a.data[2], 0x37);
    EXPECT_EQ(a.ref[1], 0.40625f);
    const auto b = runFp8MX(HIP_R_8F_E4M3, HIP_R_8F_UE8M0, "maf1_0", 1);
    EXPECT_EQ(b.data[0], 0x38);
    EXPECT_EQ(b.ref[0], 0.5f);
    const auto f4 = runFp8MX(static_cast<hipDataType>(HIP_R_4F_E2M1), HIP_R_8F_UE8M0, "maf1_0", 0);
    EXPECT_EQ((f4.data[0] >> 4) & 0xF, 0x2);
    EXPECT_EQ(f4.ref[1], 0.5f);
}
