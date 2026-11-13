// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests of the integer_exact MX data the clients generate for fp6 (E2M3, E3M2)
// elements, and for fp8 where the two share code, and of the bound that keeps it exact against
// the fp32 host reference.

#include "mxDataGen.hpp"

#include <gtest/gtest.h>

#if HIPBLASLT_ENABLE_MXDATAGENERATOR

#include "mx_tile128x4.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{
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

    size_t fp6Bytes(size_t elements)
    {
        return (elements * 6 + 7) / 8;
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
                std::vector<uint8_t> data(fp6Bytes(rows * cols)), scale(rows * cols / 32 + 64);
                std::vector<float>   ref = generateMXInput(type,
                                                         HIP_R_8U,
                                                         data.data(),
                                                         scale.data(),
                                                         rows,
                                                         cols,
                                                         rows,
                                                         transpose,
                                                         32,
                                                         1,
                                                         isMatrixA,
                                                         MXScaleLayout::None,
                                                         "integer_exact");
                ASSERT_EQ(ref.size(), rows * cols);
                bool sawLargeScale = false, sawUnitScale = false, sawNegative = false;
                for(float v : ref)
                {
                    const float a = std::fabs(v);
                    ASSERT_TRUE(a == 0 || a == 1 || a == 2 || a == 4 || a == 8)
                        << v << " isMatrixA=" << isMatrixA << " transpose=" << transpose;
                    ASSERT_LE(a, kIntegerExactMXMaxElement * kIntegerExactMXMaxScale);
                    sawLargeScale |= a == 8 || a == 4;
                    sawUnitScale |= a == 1;
                    sawNegative |= v < 0;
                }
                EXPECT_TRUE(sawLargeScale && sawUnitScale);
                EXPECT_EQ(sawNegative, !isMatrixA);

                // K runs along the stored rows when A is transposed or B is not; each run of 32
                // along K shares a scale, which the other layouts index as (k / 32) * rows + row.
                const bool kMajor = isMatrixA == transpose;
                for(size_t idx = 0; idx < ref.size(); idx++)
                {
                    const size_t row = idx % rows, k = idx / rows;
                    const size_t s = kMajor ? idx / 32 : (k / 32) * rows + row;
                    const float  expected
                        = decodeFp6(type, fp6At(data, idx)) * std::ldexp(1.f, int(scale[s]) - 127);
                    ASSERT_EQ(ref[idx], expected) << "element " << idx << " isMatrixA=" << isMatrixA
                                                  << " transpose=" << transpose;
                }
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
            const bool           fp6 = type == HIP_R_6F_E2M3 || type == HIP_R_6F_E3M2;
            std::vector<uint8_t> data(fp6 ? fp6Bytes(rows * cols) : rows * cols);
            std::vector<uint8_t> scale(cols * blocksPerCol + 64);
            // Transposed A and non-transposed B, both K-contiguous.
            std::vector<float> ref = generateMXInput(type,
                                                     HIP_R_8U,
                                                     data.data(),
                                                     scale.data(),
                                                     rows,
                                                     cols,
                                                     rows,
                                                     /*isTranspose=*/isMatrixA,
                                                     isMatrixA ? 32 : 1,
                                                     isMatrixA ? 1 : 32,
                                                     isMatrixA,
                                                     MXScaleLayout::None,
                                                     "integer_exact");
            ASSERT_EQ(ref.size(), rows * cols);
            for(size_t idx = 0; idx < ref.size(); idx++)
            {
                const size_t k = idx % rows, col = idx / rows;
                const size_t s = col * blocksPerCol + k / 32;
                const float  element
                    = fp6 ? decodeFp6(type, fp6At(data, idx)) : decodeFp8(type, data[idx]);
                const float expected = element * std::ldexp(1.f, int(scale[s]) - 127);
                ASSERT_EQ(ref[idx], expected) << "element " << idx << " (k " << k << ", column "
                                              << col << ") type " << int(type)
                                              << " isMatrixA=" << isMatrixA;
            }
        }
}

// With a padded leading dimension the reference covers the generator's whole stride * cols
// buffer, as the other initializations' references do. Each element, at its padded offset,
// matches the packed data and the scale the generator maps it to: one per block of 32 within
// each run of rows elements of the buffer.
TEST(HostUnitMXIntegerExact, smoke_PaddedLeadingDimensionKeepsTheLayout)
{
    struct Shape
    {
        size_t rows, cols, stride;
    };
    for(Shape shape : {Shape{64, 4, 96}, Shape{33, 4, 66}})
        for(hipDataType type : {HIP_R_6F_E2M3, HIP_R_6F_E3M2, HIP_R_8F_E4M3, HIP_R_8F_E5M2})
            for(bool isMatrixA : {true, false})
            {
                const size_t count        = shape.stride * shape.cols;
                const size_t blocksPerRun = (shape.rows + 31) / 32;
                const size_t scaleCount   = count / shape.rows * blocksPerRun;
                const bool   fp6          = type == HIP_R_6F_E2M3 || type == HIP_R_6F_E3M2;
                std::vector<uint8_t> data(fp6 ? fp6Bytes(count) : count);
                std::vector<uint8_t> scale(scaleCount + 64);
                // Transposed A and non-transposed B, both K-contiguous.
                std::vector<float> ref = generateMXInput(type,
                                                         HIP_R_8U,
                                                         data.data(),
                                                         scale.data(),
                                                         shape.rows,
                                                         shape.cols,
                                                         shape.stride,
                                                         /*isTranspose=*/isMatrixA,
                                                         isMatrixA ? 32 : 1,
                                                         isMatrixA ? 1 : 32,
                                                         isMatrixA,
                                                         MXScaleLayout::None,
                                                         "integer_exact");
                ASSERT_EQ(ref.size(), count) << "rows " << shape.rows << " stride " << shape.stride;
                for(size_t idx = 0; idx < count; idx++)
                {
                    const size_t s = idx / shape.rows * blocksPerRun + idx % shape.rows / 32;
                    ASSERT_LT(s, scaleCount);
                    const float element
                        = fp6 ? decodeFp6(type, fp6At(data, idx)) : decodeFp8(type, data[idx]);
                    const float expected = element * std::ldexp(1.f, int(scale[s]) - 127);
                    ASSERT_EQ(ref[idx], expected)
                        << "offset " << idx << " (k " << idx % shape.stride << ", column "
                        << idx / shape.stride << ") rows " << shape.rows << " stride "
                        << shape.stride << " type " << int(type) << " isMatrixA=" << isMatrixA;
                }
            }
}

// When stride * cols is not a multiple of rows, the generator has no scales for the elements
// past its last whole run of rows, so integer_exact refuses the shape before writing.
TEST(HostUnitMXIntegerExact, smoke_PaddedLeadingDimensionWithoutWholeRunsIsRefused)
{
    const size_t rows = 64, cols = 3, stride = 96;
    for(hipDataType type : {HIP_R_6F_E2M3, HIP_R_8F_E4M3})
        for(bool isMatrixA : {true, false})
        {
            std::vector<uint8_t> data(stride * cols, 0xA5), scale(stride * cols / 32 + 64, 0xA5);
            EXPECT_THROW(generateMXInput(type,
                                         HIP_R_8U,
                                         data.data(),
                                         scale.data(),
                                         rows,
                                         cols,
                                         stride,
                                         /*isTranspose=*/isMatrixA,
                                         isMatrixA ? 32 : 1,
                                         isMatrixA ? 1 : 32,
                                         isMatrixA,
                                         MXScaleLayout::None,
                                         "integer_exact"),
                         std::runtime_error)
                << "type " << int(type) << " isMatrixA=" << isMatrixA;
            EXPECT_TRUE(std::all_of(data.begin(), data.end(), [](uint8_t b) { return b == 0xA5; }));
            EXPECT_TRUE(
                std::all_of(scale.begin(), scale.end(), [](uint8_t b) { return b == 0xA5; }));
        }
}

// The 128x4 scale shuffle the MXFP6 kernels read: the reference must match the shuffled scales.
TEST(HostUnitMXIntegerExact, smoke_Fp6Tile128x4ReferencesMatchTheShuffledScales)
{
    const size_t K = 256, mn = 256, kBlocks = K / 32;
    for(hipDataType type : {HIP_R_6F_E2M3, HIP_R_6F_E3M2})
        for(bool isMatrixA : {true, false})
        {
            std::vector<uint8_t> data(fp6Bytes(K * mn)), scale(mn * kBlocks);
            std::vector<float>   ref = generateMXInput(type,
                                                     HIP_R_8U,
                                                     data.data(),
                                                     scale.data(),
                                                     K,
                                                     mn,
                                                     K,
                                                     /*isTranspose=*/isMatrixA,
                                                     isMatrixA ? 32 : 1,
                                                     isMatrixA ? 1 : 32,
                                                     isMatrixA,
                                                     MXScaleLayout::Tile128x4,
                                                     "integer_exact");
            ASSERT_EQ(ref.size(), K * mn);
            for(size_t idx = 0; idx < ref.size(); idx++)
            {
                const size_t k = idx % K, row = idx / K;
                const size_t s = mx_tile128x4::shuffledScaleOffset(row, k / 32, mn);
                const float  expected
                    = decodeFp6(type, fp6At(data, idx)) * std::ldexp(1.f, int(scale[s]) - 127);
                ASSERT_EQ(ref[idx], expected) << "element " << idx << " isMatrixA=" << isMatrixA;
            }
        }
}

// FP4 has no integer_exact encoding; the generator refuses it instead of writing other data.
TEST(HostUnitMXIntegerExact, smoke_Fp4IsRefused)
{
    std::vector<uint8_t> data(64 * 64 / 2), scale(64 * 64 / 32);
    EXPECT_THROW(generateMXInput(HIP_R_4F_E2M1,
                                 HIP_R_8U,
                                 data.data(),
                                 scale.data(),
                                 64,
                                 64,
                                 64,
                                 true,
                                 32,
                                 1,
                                 true,
                                 MXScaleLayout::None,
                                 "integer_exact"),
                 std::runtime_error);
}

// At the longest K the MXFP6 kernels are registered for, the sum of |A * B| over K, which
// bounds every partial sum of D in any order, stays below 2^24 for the generated data as well as
// by design, so fp32 sums agree with the exact sum whatever order the device uses.
TEST(HostUnitMXIntegerExact, smoke_Fp6PartialSumsStayExactAtK131584)
{
    constexpr int64_t K = 131584;
    static_assert(integerExactMXSumBound(K) < kIntegerExactMXExactLimit);
    const size_t mn = 16;
    for(hipDataType type : {HIP_R_6F_E2M3, HIP_R_6F_E3M2})
    {
        std::vector<uint8_t> dataA(fp6Bytes(K * mn)), scaleA(K / 32 * mn);
        std::vector<uint8_t> dataB(fp6Bytes(K * mn)), scaleB(K / 32 * mn);
        // Transposed A and non-transposed B, both K-contiguous, as the kernels read them.
        const std::vector<float> a = generateMXInput(type,
                                                     HIP_R_8U,
                                                     dataA.data(),
                                                     scaleA.data(),
                                                     K,
                                                     mn,
                                                     K,
                                                     /*isTranspose=*/true,
                                                     32,
                                                     1,
                                                     /*isMatrixA=*/true,
                                                     MXScaleLayout::None,
                                                     "integer_exact");
        const std::vector<float> b = generateMXInput(type,
                                                     HIP_R_8U,
                                                     dataB.data(),
                                                     scaleB.data(),
                                                     K,
                                                     mn,
                                                     K,
                                                     /*isTranspose=*/false,
                                                     1,
                                                     32,
                                                     /*isMatrixA=*/false,
                                                     MXScaleLayout::None,
                                                     "integer_exact");
        ASSERT_EQ(a.size(), size_t(K) * mn);
        ASSERT_EQ(b.size(), size_t(K) * mn);

        auto    iabs      = [](int64_t v) { return v < 0 ? -v : v; };
        int64_t maxAbsSum = 0, maxAbsPartial = 0, maxAbsD = 0;
        for(size_t m = 0; m < mn; m++)
            for(size_t n = 0; n < mn; n++)
            {
                const float* x       = a.data() + m * K;
                const float* y       = b.data() + n * K;
                int64_t      exact   = 0;
                int64_t      absSum  = 0;
                float        forward = 0, reverse = 0;
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
                  << ", design bound = " << integerExactMXSumBound(K)
                  << ", 2^24 = " << kIntegerExactMXExactLimit << std::endl;
        EXPECT_LE(maxAbsSum, integerExactMXSumBound(K));
        EXPECT_LT(maxAbsSum, kIntegerExactMXExactLimit);
        EXPECT_GT(maxAbsD, 0) << "D is zero, so the data cannot detect an error";
    }
}

#endif
