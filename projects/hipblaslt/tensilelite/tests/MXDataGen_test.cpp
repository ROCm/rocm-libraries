// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <mxDataGen.hpp>

#include <cstdint>
#include <vector>

/**
 * @brief Unpack two FP4 nibbles from a packed byte.
 *
 * FP4 E2M1 values are packed two-per-byte (low nibble first).
 * Zero is represented by nibble value 0x0 (+0) or 0x8 (-0).
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

    const uint64_t numElements  = rows * cols;
    const uint64_t numPacked    = (numElements + 1) / 2;
    const size_t   numScales    = ((rows + mxBlock - 1) / mxBlock) * cols;

    std::vector<uint8_t> dataBuffer(numPacked, 0);
    std::vector<uint8_t> scaleBuffer(numScales, 0);

    std::vector<size_t> emptySwizzle;
    std::vector<size_t> emptyTile;

    generateMXInput((hipDataType)HIP_R_4F_E2M1_EXT,
                    dataBuffer.data(),
                    scaleBuffer.data(),
                    rows,
                    cols,
                    rows, // stride = rows (column-major)
                    isTranspose,
                    emptySwizzle,
                    emptyTile,
                    mxBlock,
                    1,
                    true,
                    "Bounded",
                    -1.0f,
                    1.0f);

    size_t zeros       = countZerosFP4(dataBuffer.data(), numPacked);
    double zeroPercent = 100.0 * static_cast<double>(zeros) / static_cast<double>(numElements);

    // Empirically ~12.5–12.9% zeros; naive baseline is 2/16 = 12.5% (2 zero values
    // out of 16 FP4 nibble values), slightly elevated by MX block scaling bias.
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

// ============================================================================
// PreSwizzle scale tests
//
// These tests verify that generateMXInput with preSwizzle parameters produces
// scale data that is a permutation of the unswizzled scale data.  The FP4 MX
// kernels on gfx950 expect scale data in a specific shuffled layout:
//   preSwizzle = {swizzleTileMN=32, tileK=8, subTileK=MiK/mxBlock}
//   preTile    = {tileK=8, swizzleTileMN=32}
// where swizzleTileMN=32 is fixed (2 SIMDs * 16 lanes per wave) and
// subTileK=4 for MiK=128, mxBlock=32.
//
// Constraints on test sizes for preSwizzle {32,8,4} + preTile {8,32}:
//   scaleRows = rows / mxBlock  must be divisible by tileK (8)  → rows % 256 == 0
//   scaleCols = cols             must be divisible by swizzleTileMN (32) → cols % 32 == 0
// ============================================================================

// Params: {rows, cols, mxBlock, isTranspose, isMatrixA}
class MXPreSwizzleTest
    : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, int, bool, bool>>
{
};

/**
 * @brief Verify that generateMXInput with preSwizzle parameters produces scale data
 * that is a permutation of the scale data generated without preSwizzle.
 *
 * Pre-shuffled scale data must contain the same bytes as the unshuffled version
 * (just reordered) so that numerical values are preserved.  A non-trivial shuffle
 * must also have occurred — i.e. the buffers must differ.
 */
TEST_P(MXPreSwizzleTest, ScaleIsPermutationOfUnswizzled)
{
    auto [rows, cols, mxBlock, isTranspose, isMatrixA] = GetParam();

    // preSwizzle parameters for gfx950 FP4 MX kernels:
    //   swizzleTileMN=32 (2 SIMDs*16 lanes), tileK=256/32=8, subTileK=MiK/mxBlock=128/32=4
    const std::vector<size_t> preSwizzle = {32, 8, 4};
    const std::vector<size_t> preTile    = {8, 32};

    const uint64_t numElements  = rows * cols;
    const uint64_t numPacked    = (numElements + 1) / 2;
    const size_t   numScales    = ((rows + mxBlock - 1) / mxBlock) * cols;

    std::vector<uint8_t> dataNoShuf(numPacked, 0);
    std::vector<uint8_t> scaleNoShuf(numScales, 0);
    std::vector<uint8_t> dataShuf(numPacked, 0);
    std::vector<uint8_t> scaleShuf(numScales, 0);

    // Generate without preSwizzle (natural scale order)
    generateMXInput((hipDataType)HIP_R_4F_E2M1_EXT,
                    dataNoShuf.data(),
                    scaleNoShuf.data(),
                    rows, cols, rows,
                    isTranspose,
                    {}, {},
                    mxBlock, 1, isMatrixA,
                    "Bounded", -1.0f, 1.0f);

    // Generate with preSwizzle (shuffled scale order expected by the GPU kernel)
    generateMXInput((hipDataType)HIP_R_4F_E2M1_EXT,
                    dataShuf.data(),
                    scaleShuf.data(),
                    rows, cols, rows,
                    isTranspose,
                    preSwizzle, preTile,
                    mxBlock, 1, isMatrixA,
                    "Bounded", -1.0f, 1.0f);

    // The scale buffers must be different (a non-trivial shuffle occurred)
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
        // rows % 256 == 0, cols % 32 == 0 required for preSwizzle {32,8,4}+preTile{8,32}
        std::make_tuple(256u,  256u,  32, true,  true),   // scale A transposed
        std::make_tuple(256u,  256u,  32, false, false),  // scale B non-transposed
        std::make_tuple(512u,  256u,  32, true,  true),   // larger scale A
        std::make_tuple(256u,  512u,  32, false, false),  // larger scale B
        std::make_tuple(4096u, 16384u, 32, true, true)    // benchmark-scale problem
    )
);
