// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests of the integer_exact MX data generateMxData produces for fp6 (E2M3, E3M2)
// elements, and for fp8 where the two share code, and of the bound that keeps it exact against
// the fp32 host reference.

#include <hipblaslt/hostnumerics/HipblasltDataInitialization.hpp>

#include <gtest/gtest.h>

#include "mx_tile128x4.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace
{
    namespace hn     = roc::hostnumerics;
    namespace client = hipblaslt::hostnumerics;

    std::vector<uint8_t> bytesOf(std::span<const std::byte> storage)
    {
        std::vector<uint8_t> bytes;
        bytes.reserve(storage.size());
        for(std::byte b : storage)
            bytes.push_back(std::to_integer<uint8_t>(b));
        return bytes;
    }

    // 6-bit LSB-first packing, as the kernels read it.
    uint8_t fp6At(std::vector<uint8_t> const& packed, size_t idx)
    {
        size_t const bit  = 6 * idx;
        unsigned     word = packed[bit / 8];
        if(bit / 8 + 1 < packed.size())
            word |= unsigned(packed[bit / 8 + 1]) << 8;
        return uint8_t((word >> (bit % 8)) & 0x3F);
    }

    // The OCP encodings of -2, -1, 0, 1 and 2, written out rather than taken from the generator.
    float decodeFp6(hipDataType type, uint8_t code)
    {
        bool const e3m2 = type == HIP_R_6F_E3M2;
        switch(code)
        {
        case 0x30:
            return -2.f;
        case 0x00:
            return 0.f;
        case 0x10:
            return 2.f;
        case 0x28:
            if(!e3m2)
                return -1.f;
            break;
        case 0x08:
            if(!e3m2)
                return 1.f;
            break;
        case 0x2C:
            if(e3m2)
                return -1.f;
            break;
        case 0x0C:
            if(e3m2)
                return 1.f;
            break;
        }
        ADD_FAILURE() << "unexpected fp6 code 0x" << std::hex << int(code) << std::dec << " for "
                      << (e3m2 ? "E3M2" : "E2M3");
        return 0.f;
    }

    float decodeFp8(hipDataType type, uint8_t code)
    {
        bool const e5m2 = type == HIP_R_8F_E5M2;
        switch(code)
        {
        case 0xC0:
            return -2.f;
        case 0x00:
            return 0.f;
        case 0x40:
            return 2.f;
        case 0xB8:
            if(!e5m2)
                return -1.f;
            break;
        case 0x38:
            if(!e5m2)
                return 1.f;
            break;
        case 0xBC:
            if(e5m2)
                return -1.f;
            break;
        case 0x3C:
            if(e5m2)
                return 1.f;
            break;
        }
        ADD_FAILURE() << "unexpected fp8 code 0x" << std::hex << int(code) << std::dec << " for "
                      << (e5m2 ? "E5M2" : "E4M3");
        return 0.f;
    }

    bool isFp6(hipDataType type)
    {
        return type == HIP_R_6F_E2M3 || type == HIP_R_6F_E3M2;
    }

    // Decodes element `offset` of the packed data, which uses strides {1, leading dimension}.
    float elementAt(hipDataType type, std::vector<uint8_t> const& data, size_t offset)
    {
        return isFp6(type) ? decodeFp6(type, fp6At(data, offset)) : decodeFp8(type, data[offset]);
    }

    float e8m0(uint8_t code)
    {
        return std::ldexp(1.f, int(code) - 127);
    }

    // K along the stored rows (block axis 0) for transposed A and non-transposed B.
    hn::MxTensor generate(hipDataType type,
                          uint64_t    rows,
                          uint64_t    cols,
                          uint64_t    ld,
                          size_t      blockAxis,
                          bool        isMatrixA)
    {
        return client::generateMxData(type,
                                      HIP_R_8F_UE8M0,
                                      hn::Shape{rows, cols},
                                      ld,
                                      blockAxis,
                                      32,
                                      hipblaslt_initialization::integer_exact,
                                      client::mxDefaultSeed,
                                      isMatrixA ? client::MatrixRole::A : client::MatrixRole::B);
    }
} // namespace

// Every dequantized value is an element in {0, 1, 2} (A) or {-2, ..., 2} (B) times a scale of
// 1, 2 or 4, more than one scale is used, and the reference matches the packed bytes and scales
// the GPU reads, in all four A and B layouts.
TEST(HostUnitMXIntegerExact, smoke_Fp6ValuesAreExactWithVariedScales)
{
    const uint64_t rows = 64, cols = 128;
    for(hipDataType type : {HIP_R_6F_E2M3, HIP_R_6F_E3M2})
        for(bool isMatrixA : {true, false})
            for(bool transpose : {false, true})
            {
                // K runs along the stored rows when A is transposed or B is not; each run of 32
                // along K shares a scale, which the other layouts index as (k / 32) * rows + row.
                const bool kMajor = isMatrixA == transpose;
                const auto mx     = generate(type, rows, cols, rows, kMajor ? 0 : 1, isMatrixA);
                const auto data   = bytesOf(mx.data.rawEncodedBackingStorage());
                const auto scale  = bytesOf(mx.scales.rawEncodedBackingStorage());
                bool       sawLargeScale = false, sawUnitScale = false, sawNegative = false;
                for(size_t idx = 0; idx < rows * cols; idx++)
                {
                    const size_t row = idx % rows, k = idx / rows;
                    const float  v = mx.reference.loadAs<float>({row, k});
                    const float  a = std::fabs(v);
                    ASSERT_TRUE(a == 0 || a == 1 || a == 2 || a == 4 || a == 8)
                        << v << " isMatrixA=" << isMatrixA << " transpose=" << transpose;
                    ASSERT_LE(a, client::integerExactMxMaxElement * client::integerExactMxMaxScale);
                    sawLargeScale |= a == 8 || a == 4;
                    sawUnitScale |= a == 1;
                    sawNegative |= v < 0;

                    const size_t s = kMajor ? idx / 32 : (k / 32) * rows + row;
                    ASSERT_EQ(v, elementAt(type, data, idx) * e8m0(scale[s]))
                        << "element " << idx << " isMatrixA=" << isMatrixA
                        << " transpose=" << transpose;
                }
                EXPECT_TRUE(sawLargeScale && sawUnitScale);
                EXPECT_EQ(sawNegative, !isMatrixA);
            }
}

// With K = 33 along the stored rows, each column holds a whole block and a one-element block,
// each with its own scale. The reference must not carry a column's tail scale into the next one.
TEST(HostUnitMXIntegerExact, smoke_PartialKBlockKeepsItsColumnScale)
{
    const size_t rows = 33, cols = 4, blocksPerCol = 2;
    for(hipDataType type : {HIP_R_6F_E2M3, HIP_R_6F_E3M2, HIP_R_8F_E4M3, HIP_R_8F_E5M2})
        for(bool isMatrixA : {true, false})
        {
            // Transposed A and non-transposed B, both K-contiguous.
            const auto mx    = generate(type, rows, cols, rows, 0, isMatrixA);
            const auto data  = bytesOf(mx.data.rawEncodedBackingStorage());
            const auto scale = bytesOf(mx.scales.rawEncodedBackingStorage());
            ASSERT_EQ(scale.size(), cols * blocksPerCol);
            for(size_t idx = 0; idx < rows * cols; idx++)
            {
                const size_t k = idx % rows, col = idx / rows;
                const size_t s = col * blocksPerCol + k / 32;
                ASSERT_EQ(mx.reference.loadAs<float>({k, col}),
                          elementAt(type, data, idx) * e8m0(scale[s]))
                    << "element " << idx << " (k " << k << ", column " << col << ") type "
                    << int(type) << " isMatrixA=" << isMatrixA;
            }
        }
}

// A padded leading dimension moves each column, not the blocks: element (k, col) sits at
// k + col * ld and keeps the scale of its own column's K block, whether or not the padded
// buffer holds a whole number of runs of rows elements. The former generator refused the
// latter shapes; HostNumerics blocks the logical matrix.
TEST(HostUnitMXIntegerExact, smoke_PaddedLeadingDimensionKeepsTheLayout)
{
    struct Shape
    {
        size_t rows, cols, ld;
    };
    for(Shape shape : {Shape{64, 4, 96}, Shape{33, 4, 66}, Shape{64, 3, 96}})
        for(hipDataType type : {HIP_R_6F_E2M3, HIP_R_6F_E3M2, HIP_R_8F_E4M3, HIP_R_8F_E5M2})
            for(bool isMatrixA : {true, false})
            {
                const size_t blocksPerCol = (shape.rows + 31) / 32;
                const auto   mx    = generate(type, shape.rows, shape.cols, shape.ld, 0, isMatrixA);
                const auto   data  = bytesOf(mx.data.rawEncodedBackingStorage());
                const auto   scale = bytesOf(mx.scales.rawEncodedBackingStorage());
                ASSERT_EQ(scale.size(), shape.cols * blocksPerCol);
                for(size_t col = 0; col < shape.cols; col++)
                    for(size_t k = 0; k < shape.rows; k++)
                    {
                        const size_t s = col * blocksPerCol + k / 32;
                        ASSERT_EQ(mx.reference.loadAs<float>({k, col}),
                                  elementAt(type, data, k + col * shape.ld) * e8m0(scale[s]))
                            << "k " << k << " column " << col << " rows " << shape.rows << " ld "
                            << shape.ld << " type " << int(type) << " isMatrixA=" << isMatrixA;
                    }
            }
}

// Zero rows are an empty matrix: generation writes no scales and no reference values. The
// former generator refused this shape because its padded-layout check divided by the row count.
TEST(HostUnitMXIntegerExact, smoke_ZeroRowsGenerateNothing)
{
    for(hipDataType type : {HIP_R_6F_E2M3, HIP_R_6F_E3M2})
        for(bool isMatrixA : {true, false})
        {
            const auto mx = generate(type, 0, 4, 1, 0, isMatrixA);
            EXPECT_EQ(mx.scales.elementCount(), 0u) << "type " << int(type);
            EXPECT_EQ(mx.reference.elementCount(), 0u) << "type " << int(type);
        }
}

// Non-transposed A and transposed B block K along the stored columns. With 33 columns each row
// has a whole block and a one-element block. The former generator read past its packed buffer
// for this shape and refused it; HostNumerics generates it with each tail block's own scale.
TEST(HostUnitMXIntegerExact, smoke_PartialKAlongColumnsKeepsItsRowScale)
{
    const uint64_t rows = 64, cols = 33, kBlocks = 2;
    for(hipDataType type : {HIP_R_6F_E2M3, HIP_R_8F_E4M3})
        for(bool isMatrixA : {true, false})
        {
            const auto mx    = generate(type, rows, cols, rows, 1, isMatrixA);
            const auto data  = bytesOf(mx.data.rawEncodedBackingStorage());
            const auto scale = bytesOf(mx.scales.rawEncodedBackingStorage());
            ASSERT_EQ(scale.size(), kBlocks * rows);
            for(size_t k = 0; k < cols; k++)
                for(size_t row = 0; row < rows; row++)
                {
                    const size_t s = (k / 32) * rows + row;
                    ASSERT_EQ(mx.reference.loadAs<float>({row, k}),
                              elementAt(type, data, row + k * rows) * e8m0(scale[s]))
                        << "row " << row << " k " << k << " type " << int(type)
                        << " isMatrixA=" << isMatrixA;
                }
        }
}

// The 128x4 scale shuffle the MXFP6 kernels read: the reference must match the shuffled scales.
TEST(HostUnitMXIntegerExact, smoke_Fp6Tile128x4ReferencesMatchTheShuffledScales)
{
    const size_t K = 256, mn = 256, kBlocks = K / 32;
    for(hipDataType type : {HIP_R_6F_E2M3, HIP_R_6F_E3M2})
        for(bool isMatrixA : {true, false})
        {
            // Transposed A and non-transposed B, both K-contiguous.
            const auto mx    = generate(type, K, mn, K, 0, isMatrixA);
            const auto data  = bytesOf(mx.data.rawEncodedBackingStorage());
            const auto plan  = hn::amd_gpu_layout::MxScaleStoragePlan::natural({mn, kBlocks});
            const auto scale = bytesOf(client::mxScaleBytesForDevice(
                mx, 0, plan, client::MxScaleStorageLayout::Tile128x4));
            ASSERT_EQ(scale.size(), mn * kBlocks);
            for(size_t idx = 0; idx < K * mn; idx++)
            {
                const size_t k = idx % K, row = idx / K;
                const size_t s = mx_tile128x4::shuffledScaleOffset(row, k / 32, mn);
                ASSERT_EQ(mx.reference.loadAs<float>({k, row}),
                          decodeFp6(type, fp6At(data, idx)) * e8m0(scale[s]))
                    << "element " << idx << " isMatrixA=" << isMatrixA;
            }
        }
}

// The 128x4 scale layout shuffles natural [MN, K blocks] bytes in whole 128 x 4 tiles. It refuses
// a padded (non-natural) plan and shapes that are not whole tiles.
TEST(HostUnitMXIntegerExact, smoke_Tile128x4RefusesOtherPlansAndPartialTiles)
{
    const auto whole = generate(HIP_R_6F_E2M3, 256, 128, 256, 0, true);
    EXPECT_THROW(
        client::mxScaleBytesForDevice(whole,
                                      0,
                                      hn::amd_gpu_layout::MxScaleStoragePlan::tile32x8({128, 8}),
                                      client::MxScaleStorageLayout::Tile128x4),
        std::runtime_error);

    // 64 rows of M and 2 K blocks: neither is a whole tile.
    const auto partial = generate(HIP_R_6F_E2M3, 64, 64, 64, 0, true);
    EXPECT_THROW(
        client::mxScaleBytesForDevice(partial,
                                      0,
                                      hn::amd_gpu_layout::MxScaleStoragePlan::natural({64, 2}),
                                      client::MxScaleStorageLayout::Tile128x4),
        std::invalid_argument);
}

// FP4 has no integer_exact encoding; the generator refuses it instead of writing other data.
TEST(HostUnitMXIntegerExact, smoke_Fp4IsRefused)
{
    EXPECT_THROW(generate(static_cast<hipDataType>(HIP_R_4F_E2M1), 64, 64, 64, 0, true),
                 std::runtime_error);
}

// At the longest K the MXFP6 kernels are registered for, the sum of |A * B| over K, which
// bounds every partial sum of D in any order, stays below 2^24 for the generated data as well as
// by design, so fp32 sums agree with the exact sum whatever order the device uses.
TEST(HostUnitMXIntegerExact, smoke_Fp6PartialSumsStayExactAtK131584)
{
    constexpr int64_t K = 131584;
    static_assert(client::integerExactMxSumBound(K) < client::integerExactMxExactLimit);
    const size_t mn = 16;
    for(hipDataType type : {HIP_R_6F_E2M3, HIP_R_6F_E3M2})
    {
        // Transposed A and non-transposed B, both K-contiguous, as the kernels read them.
        const auto mxA = generate(type, K, mn, K, 0, /*isMatrixA=*/true);
        const auto mxB = generate(type, K, mn, K, 0, /*isMatrixA=*/false);

        auto    iabs      = [](int64_t v) { return v < 0 ? -v : v; };
        int64_t maxAbsSum = 0, maxAbsPartial = 0, maxAbsD = 0;
        for(size_t m = 0; m < mn; m++)
            for(size_t n = 0; n < mn; n++)
            {
                std::vector<float> x(K), y(K);
                for(int64_t k = 0; k < K; k++)
                {
                    x[k] = mxA.reference.loadAs<float>({size_t(k), m});
                    y[k] = mxB.reference.loadAs<float>({size_t(k), n});
                }
                int64_t exact   = 0;
                int64_t absSum  = 0;
                float   forward = 0, reverse = 0;
                for(int64_t k = 0; k < K; k++)
                {
                    const float product = x[k] * y[k];
                    ASSERT_EQ(product, std::trunc(product));
                    exact += int64_t(product);
                    absSum += iabs(int64_t(product));
                    maxAbsPartial = std::max(maxAbsPartial, iabs(exact));
                    forward += product;
                    reverse += x[K - 1 - k] * y[K - 1 - k];
                }
                ASSERT_EQ(double(forward), double(exact)) << "m=" << m << " n=" << n;
                ASSERT_EQ(double(reverse), double(exact)) << "m=" << m << " n=" << n;
                maxAbsSum = std::max(maxAbsSum, absSum);
                maxAbsD   = std::max(maxAbsD, iabs(exact));
            }
        std::cout << (type == HIP_R_6F_E3M2 ? "E3M2" : "E2M3") << " K=" << K
                  << ": max |D| = " << maxAbsD << ", max |partial sum| = " << maxAbsPartial
                  << ", max sum |A * B| = " << maxAbsSum
                  << ", design bound = " << client::integerExactMxSumBound(K)
                  << ", 2^24 = " << client::integerExactMxExactLimit << std::endl;
        EXPECT_LE(maxAbsSum, client::integerExactMxSumBound(K));
        EXPECT_LT(maxAbsSum, client::integerExactMxExactLimit);
        EXPECT_GT(maxAbsD, 0) << "D is zero, so the data cannot detect an error";
    }
}
