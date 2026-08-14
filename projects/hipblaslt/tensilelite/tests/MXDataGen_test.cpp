// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <mxDataGen.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <span>
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

    const uint64_t numElements = rows * cols;
    const uint64_t numPacked   = (numElements + 1) / 2;
    const size_t   numScales   = ((rows + mxBlock - 1) / mxBlock) * cols;

    std::vector<uint8_t> dataBuffer(numPacked, 0);
    std::vector<uint8_t> scaleBuffer(numScales, 0);

    generateMXInput((hipDataType)HIP_R_4F_E2M1,
                    HIP_R_8F_UE8M0,
                    dataBuffer,
                    scaleBuffer,
                    rows,
                    cols,
                    rows, // stride = rows (column-major)
                    mxBlock,
                    1,
                    MXScaleLayout::None,
                    "Bounded",
                    -1.0f,
                    1.0f);

    size_t zeros       = countZerosFP4(dataBuffer.data(), numPacked);
    double zeroPercent = 100.0 * static_cast<double>(zeros) / static_cast<double>(numElements);

    EXPECT_LT(zeroPercent, 13.0) << "Zero frequency " << zeroPercent
                                 << "% exceeds 13% upper bound for " << rows << "x" << cols
                                 << " FP4 matrix (transpose=" << isTranspose << ")";

    // Ensure non-trivial data was actually generated (not all zeros)
    EXPECT_GT(numElements - zeros, 0u)
        << "All elements are zero for " << rows << "x" << cols << " FP4 matrix";
}

INSTANTIATE_TEST_SUITE_P(FP4ZeroFrequency,
                         MXDataGenFP4Test,
                         ::testing::Values(
                             // rows, cols, mxBlock, isTranspose
                             std::make_tuple(128u, 128u, 32, true),
                             std::make_tuple(256u, 256u, 32, true),
                             std::make_tuple(2048u, 1026u, 32, true),
                             std::make_tuple(2048u, 514u, 32, false)));

/**
 * @brief Regression guard: generateMXInput must be deterministic (fixed seed).
 *
 * Any post-generation overwrite of the MXSA/MXSB buffers (e.g., the general
 * tensor-init loop in initializeCPUInputs) desynchronises the CPU reference
 * from GPU data, causing intermittent single-element validation failures.
 * rows=K (must be mxBlock-aligned), cols=M/N (need not be).
 */
class MXGeneratorDeterminismTest
    : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, int>>
{
};

TEST_P(MXGeneratorDeterminismTest, GeneratorOutputIsDeterministic)
{
    auto [rows, cols, mxBlock] = GetParam();

    const size_t numPacked = (rows * cols + 1) / 2;
    const size_t numScales = (rows / mxBlock) * cols;

    std::vector<uint8_t> data1(numPacked);
    std::vector<uint8_t> data2(numPacked);
    std::vector<uint8_t> scale1(numScales, 0x00);
    std::vector<uint8_t> scale2(numScales,
                                0xFF); // sentinel: catches no-write if scale1==scale2 passes

    generateMXInput((hipDataType)HIP_R_4F_E2M1,
                    HIP_R_8F_UE8M0,
                    data1,
                    scale1,
                    rows,
                    cols,
                    rows,
                    mxBlock,
                    1,
                    MXScaleLayout::None,
                    "Bounded",
                    -1.f,
                    1.f);

    generateMXInput((hipDataType)HIP_R_4F_E2M1,
                    HIP_R_8F_UE8M0,
                    data2,
                    scale2,
                    rows,
                    cols,
                    rows,
                    mxBlock,
                    1,
                    MXScaleLayout::None,
                    "Bounded",
                    -1.f,
                    1.f);

    EXPECT_EQ(data1, data2) << "FP4 data is non-deterministic";
    EXPECT_EQ(scale1, scale2)
        << "Scale data is non-deterministic; any post-generation overwrite will corrupt validation";

    bool allZero = std::all_of(scale1.begin(), scale1.end(), [](uint8_t b) { return b == 0; });
    bool allOnes = std::all_of(scale1.begin(), scale1.end(), [](uint8_t b) { return b == 0xFF; });
    EXPECT_FALSE(allZero) << "Scale buffer is all-zero — generator did not write";
    EXPECT_FALSE(allOnes) << "Scale buffer is all-0xFF (max UE8M0 value) — generator likely "
                             "failed; bounded [-1,1] input should produce varied scales";
}

INSTANTIATE_TEST_SUITE_P(
    GeneratorDeterminism,
    MXGeneratorDeterminismTest,
    ::testing::Values(
        std::make_tuple(1024u, 128u, 32),
        std::make_tuple(1024u, 204u, 32), // non-32-aligned regression
        std::make_tuple(1024u, 213u, 32) // non-32-aligned regression
        ));

class MXDataGenSeedTest : public ::testing::Test
{
protected:
    struct Output
    {
        std::vector<uint8_t> data;
        std::vector<uint8_t> scales;
        std::vector<float>   reference;
    };

    static Output generate(uint32_t seed)
    {
        constexpr uint64_t rows    = 256;
        constexpr uint64_t cols    = 256;
        constexpr int      mxBlock = 32;

        Output output;
        output.data.resize((rows * cols + 1) / 2);
        output.scales.resize((rows / mxBlock) * cols);
        output.reference = generateMXInput((hipDataType)HIP_R_4F_E2M1,
                                           HIP_R_8F_UE8M0,
                                           output.data,
                                           output.scales,
                                           rows,
                                           cols,
                                           rows,
                                           mxBlock,
                                           1,
                                           MXScaleLayout::None,
                                           "rand_int",
                                           -1.0f,
                                           1.0f,
                                           "",
                                           seed);
        return output;
    }
};

TEST_F(MXDataGenSeedTest, SameSeedRepeatsRandomOutput)
{
    auto first  = generate(12345U);
    auto second = generate(12345U);

    EXPECT_EQ(first.data, second.data);
    EXPECT_EQ(first.scales, second.scales);
    EXPECT_EQ(first.reference, second.reference);
}

TEST_F(MXDataGenSeedTest, DifferentSeedsChangeRandomOutput)
{
    auto first  = generate(12345U);
    auto second = generate(67890U);

    EXPECT_NE(first.data, second.data);
    EXPECT_NE(first.reference, second.reference);
}

// ============================================================================
// PreSwizzle scale tests
//
// Verify generateMXInput with MXScaleLayout::GFX950 produces scale data
// that is a permutation of the unswizzled (None) layout. The actual
// swizzle parameters (swizzleTileMN=32, tileK=8, subTileK=MiK/mxBlock) are
// hard-coded inside `generateMXInput` -- callers just pick the layout.
// ============================================================================

// Params: {rows, cols, mxBlock}
class MXPreSwizzleTest : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, int>>
{
};

/** @brief Verify the gfx950 swizzle produces a non-trivial permutation of scale data. */
TEST_P(MXPreSwizzleTest, ScaleIsPermutationOfUnswizzled)
{
    auto [rows, cols, mxBlock] = GetParam();

    const uint64_t numElements = rows * cols;
    const uint64_t numPacked   = (numElements + 1) / 2;
    const size_t   numScales   = ((rows + mxBlock - 1) / mxBlock) * cols;

    std::vector<uint8_t> dataNoShuf(numPacked, 0);
    std::vector<uint8_t> scaleNoShuf(numScales, 0);
    std::vector<uint8_t> dataShuf(numPacked, 0);
    std::vector<uint8_t> scaleShuf(numScales, 0);

    // Generate without preSwizzle
    generateMXInput((hipDataType)HIP_R_4F_E2M1,
                    HIP_R_8F_UE8M0,
                    dataNoShuf,
                    scaleNoShuf,
                    rows,
                    cols,
                    rows,
                    mxBlock,
                    1,
                    MXScaleLayout::None,
                    "Bounded",
                    -1.0f,
                    1.0f);

    // Generate with preSwizzle
    generateMXInput((hipDataType)HIP_R_4F_E2M1,
                    HIP_R_8F_UE8M0,
                    dataShuf,
                    scaleShuf,
                    rows,
                    cols,
                    rows,
                    mxBlock,
                    1,
                    MXScaleLayout::GFX950,
                    "Bounded",
                    -1.0f,
                    1.0f);

    // The scale buffers must be different
    EXPECT_NE(scaleNoShuf, scaleShuf)
        << "Scale data was not shuffled for " << rows << "x" << cols;

    // The shuffled scale must be a permutation: same multiset of bytes
    std::vector<uint8_t> sortedNoShuf = scaleNoShuf;
    std::vector<uint8_t> sortedShuf   = scaleShuf;
    std::sort(sortedNoShuf.begin(), sortedNoShuf.end());
    std::sort(sortedShuf.begin(), sortedShuf.end());
    EXPECT_EQ(sortedNoShuf, sortedShuf)
        << "Pre-shuffled scale is not a permutation of the unshuffled scale for " << rows << "x"
        << cols;

    // Data buffer must be identical (preSwizzle only affects scale, not data)
    EXPECT_EQ(dataNoShuf, dataShuf)
        << "Data buffer changed unexpectedly with preSwizzle for " << rows << "x" << cols;
}

INSTANTIATE_TEST_SUITE_P(
    FP4PreSwizzle,
    MXPreSwizzleTest,
    ::testing::Values(
        // Test size constraints for preSwizzle {32,8,4} + preTile {8,32}:
        //   rows % 256 == 0  (scaleRows = rows/mxBlock must be divisible by tileK=8)
        //   cols % 32  == 0  (scaleCols must be divisible by swizzleTileMN=32)
        std::make_tuple(256u, 256u, 32),
        std::make_tuple(512u, 256u, 32),
        std::make_tuple(256u, 512u, 32),
        std::make_tuple(4096u, 16384u, 32) // benchmark-scale problem
        ));

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
class MXDataGenModeTest : public ::testing::TestWithParam<std::tuple<hipDataType, std::string>>
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
        const uint64_t numElements = rows * cols;
        const size_t   bytesPerData
            = (dataType == (hipDataType)HIP_R_4F_E2M1) ? (numElements + 1) / 2 : numElements;
        const size_t numScales = (rows / mxBlock) * cols;

        std::vector<uint8_t> dataBuffer(bytesPerData, 0);
        std::vector<uint8_t> scaleBuffer(numScales, 0);

        return generateMXInput(dataType,
                               HIP_R_8F_UE8M0,
                               dataBuffer,
                               scaleBuffer,
                               rows,
                               cols,
                               rows, // stride = rows (column-major)
                               mxBlock,
                               1,
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
    ::testing::Combine(::testing::Values((hipDataType)HIP_R_4F_E2M1, HIP_R_8F_E4M3),
                       ::testing::Values(std::string("zero"),
                                         std::string("norm_dist"),
                                         std::string("rand_int"),
                                         std::string("uniform_low_precision"))));

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
                               dataBuffer,
                               scaleBuffer,
                               rows,
                               cols,
                               rows,
                               mxBlock,
                               1,
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
    EXPECT_GT(uniqueData.size(), 1u) << "packed data should not collapse to a single byte pattern";
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
                    dataA,
                    scaleA,
                    rows,
                    cols,
                    rows,
                    mxBlock,
                    1,
                    MXScaleLayout::None,
                    "Bounded",
                    -1.f,
                    1.f);

    generateMXInput((hipDataType)HIP_R_4F_E2M1,
                    HIP_R_8F_UE8M0,
                    dataB,
                    scaleB,
                    rows,
                    cols,
                    rows,
                    mxBlock,
                    1,
                    MXScaleLayout::None,
                    "Bounded",
                    -1.f,
                    1.f,
                    "Bounded");

    EXPECT_EQ(dataA, dataB);
    EXPECT_EQ(scaleA, scaleB);
}

TEST(MXDataGenCapacity, RejectsUndersizedOutputsBeforeWriting)
{
    constexpr uint64_t rows    = 32;
    constexpr uint64_t cols    = 2;
    constexpr int      mxBlock = 32;

    std::vector<uint8_t> data((rows * cols) / 2, 0xa5);
    std::vector<uint8_t> scales((rows / mxBlock) * cols, 0x5a);
    auto const           expectedData   = data;
    auto const           expectedScales = scales;

    EXPECT_THROW(generateMXInput((hipDataType)HIP_R_4F_E2M1,
                                 HIP_R_8F_UE8M0,
                                 std::span<uint8_t>(data).first(data.size() - 1),
                                 scales,
                                 rows,
                                 cols,
                                 rows,
                                 mxBlock,
                                 1),
                 std::invalid_argument);
    EXPECT_EQ(data, expectedData);
    EXPECT_EQ(scales, expectedScales);

    EXPECT_THROW(generateMXInput((hipDataType)HIP_R_4F_E2M1,
                                 HIP_R_8F_UE8M0,
                                 data,
                                 std::span<uint8_t>(scales).first(scales.size() - 1),
                                 rows,
                                 cols,
                                 rows,
                                 mxBlock,
                                 1),
                 std::invalid_argument);
    EXPECT_EQ(data, expectedData);
    EXPECT_EQ(scales, expectedScales);
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

    restrideMXScaleBufferKFast(scale,
                               /*compactFreeDim=*/2,
                               /*compactKBlocks=*/4,
                               /*paddedKBlocks=*/8,
                               /*elemBytes=*/1);

    EXPECT_EQ(scale[0], 1);
    EXPECT_EQ(scale[3], 4);
    EXPECT_EQ(scale[4], 0);
    EXPECT_EQ(scale[7], 0);
    EXPECT_EQ(scale[8], 5);
    EXPECT_EQ(scale[11], 8);
    EXPECT_EQ(scale[12], 0);
    EXPECT_EQ(scale[15], 0);
}

TEST(MXScaleRestride, RejectsInvalidExtents)
{
    std::vector<uint8_t> scale(16, 0);
    EXPECT_THROW(restrideMXScaleBufferKFast(std::span<uint8_t>{}, 2, 4, 8, 1),
                 std::invalid_argument);
    EXPECT_THROW(restrideMXScaleBufferKFast(scale, 2, 4, 8, 0), std::invalid_argument);
    EXPECT_THROW(restrideMXScaleBufferKFast(scale, 2, 8, 4, 1), std::invalid_argument);
    EXPECT_THROW(restrideMXScaleBufferKFast(scale, std::numeric_limits<size_t>::max(), 1, 2, 1),
                 std::overflow_error);
}

TEST(MXScaleRestride, RejectsUndersizedDestinationBeforeWriting)
{
    std::vector<uint8_t> scale(15, 0xa5);
    auto const           expected = scale;

    EXPECT_THROW(restrideMXScaleBufferKFast(scale, 2, 4, 8, 1), std::invalid_argument);
    EXPECT_EQ(scale, expected);
}

TEST(MXGfx1250ScaleBuffer, PartialBlockedAxesStayWithinPaddedCapacity)
{
    struct TestCase
    {
        uint64_t rows;
        uint64_t columns;
        int      blockRows;
        int      blockColumns;
    };
    constexpr std::array cases{
        TestCase{33, 17, 32, 1},
        TestCase{17, 33, 1, 32},
        TestCase{130, 385, 1, 32},
    };

    for(auto const& test : cases)
    {
        size_t const blockSize     = static_cast<size_t>(test.blockRows * test.blockColumns);
        size_t const blockedExtent = test.blockColumns > 1 ? static_cast<size_t>(test.columns)
                                                           : static_cast<size_t>(test.rows);
        size_t const freeExtent    = test.blockColumns > 1 ? static_cast<size_t>(test.rows)
                                                           : static_cast<size_t>(test.columns);
        size_t const blockCount
            = blockedExtent / blockSize + static_cast<size_t>(blockedExtent % blockSize != 0);
        size_t const fastExtent = test.blockColumns > 1 ? freeExtent : blockCount;
        size_t const slowExtent = test.blockColumns > 1 ? blockCount : freeExtent;
        size_t const dimk       = 128 / blockSize;
        size_t const paddedFast = (fastExtent + dimk - 1) / dimk * dimk;
        size_t const scaleBytes = slowExtent * paddedFast;

        size_t const         physicalElements = static_cast<size_t>(test.rows * test.columns);
        std::vector<uint8_t> data((physicalElements + 1) / 2, 0);
        constexpr size_t     guardBytes = 64;
        std::vector<uint8_t> scales(scaleBytes + guardBytes, 0xa5);

        auto reference = generateMXInput((hipDataType)HIP_R_4F_E2M1,
                                         HIP_R_8F_UE8M0,
                                         data,
                                         std::span<uint8_t>(scales).first(scaleBytes),
                                         test.rows,
                                         test.columns,
                                         test.rows,
                                         test.blockRows,
                                         test.blockColumns,
                                         MXScaleLayout::GFX1250,
                                         "Bounded",
                                         -1.0f,
                                         1.0f);

        EXPECT_EQ(reference.size(), physicalElements);
        EXPECT_TRUE(std::all_of(
            scales.begin() + scaleBytes, scales.end(), [](uint8_t value) { return value == 0xa5; }))
            << "GFX1250 scale generation wrote beyond the padded output for " << test.rows << "x"
            << test.columns;
    }
}

TEST(MXScaleLayoutFormat, MapsScalingFormatToLayout)
{
    using SF = hipblaslt_scaling_format;
    EXPECT_EQ(mxScaleLayoutForFormat(SF::Block_32_UE8M0_32_8_EXT, "gfx950"), MXScaleLayout::GFX950);
    EXPECT_EQ(mxScaleLayoutForFormat(SF::Block_32_UE8M0, "gfx950"), MXScaleLayout::None);
    EXPECT_EQ(mxScaleLayoutForFormat(SF::Block_32_UE8M0, "gfx1250"), MXScaleLayout::GFX1250);
    EXPECT_EQ(mxScaleLayoutForFormat(SF::Block_32_UE4M3, "gfx942"), MXScaleLayout::None);
}
