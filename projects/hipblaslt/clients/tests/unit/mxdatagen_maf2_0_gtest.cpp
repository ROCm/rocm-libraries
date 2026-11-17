// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for the maf2_0 and maf2_0_unit_scale methods of generateMXInput
// (mxDataGen.cpp): which block-scale distribution each scale format gets, and the unit-scale rows.
// The draws are random, so these check the scale bytes exactly and the data statistically. See
// README.md in this directory for the HostUnit/smoke_ naming.

#if HIPBLASLT_ENABLE_MXDATAGENERATOR

#include "mxDataGen.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace
{
    // Unsigned UE4M3 (bias 7) and UE5M3 (bias 15) scales, 3 mantissa bits each.
    double decodeScale(uint8_t code, int exponentBits)
    {
        const int bias     = (1 << (exponentBits - 1)) - 1;
        const int exponent = (code >> 3) & ((1 << exponentBits) - 1);
        const int mantissa = code & 0x7;
        return exponent == 0 ? std::ldexp(mantissa / 8.0, 1 - bias)
                             : std::ldexp(1.0 + mantissa / 8.0, exponent - bias);
    }

    constexpr uint8_t kUnwritten = 0xAA;

    struct Generated
    {
        std::vector<uint8_t> data;
        std::vector<uint8_t> scales; // every scale byte written
        std::vector<float>   ref;
    };

    // A transposed A (K x M stored, K along the rows) of K = 256, M = 64.
    Generated generate(hipDataType dataType,
                       hipDataType scaleType,
                       int         block,
                       char const* method,
                       uint32_t    seedIndex = 0)
    {
        constexpr uint64_t   rows = 256, cols = 64, count = rows * cols;
        const bool           fp4 = dataType == static_cast<hipDataType>(HIP_R_4F_E2M1);
        std::vector<uint8_t> data(fp4 ? count / 2 : count);
        std::vector<uint8_t> scale(count, kUnwritten);
        Generated            g;
        g.ref  = generateMXInput(dataType,
                                scaleType,
                                data.data(),
                                scale.data(),
                                rows,
                                cols,
                                rows,
                                /*isTranspose=*/true,
                                block,
                                1,
                                /*isMatrixA=*/true,
                                MXScaleLayout::None,
                                method,
                                -1.0f,
                                1.0f,
                                "",
                                seedIndex);
        g.data = std::move(data);
        for(uint8_t s : scale)
            if(s != kUnwritten)
                g.scales.push_back(s);
        EXPECT_EQ(g.ref.size(), count);
        EXPECT_GE(g.scales.size(), count / block);
        return g;
    }

    // MAF2.0 E8M0 scales are 2^(k - 11), k ~ Binomial(21, 0.5): biased exponents 116..137 with
    // mean 126.5. A normal N(0.34375, 0.08) scale would round to only 2^-3..2^-1 under E8M0.
    TEST(HostUnitMXDataGenMaf2_0, smoke_fp4_e8m0_scales_are_binomial)
    {
        const auto g
            = generate(static_cast<hipDataType>(HIP_R_4F_E2M1), HIP_R_8F_UE8M0, 32, "maf2_0");
        double            sum = 0;
        std::set<uint8_t> distinct;
        for(uint8_t s : g.scales)
        {
            ASSERT_GE(s, 116);
            ASSERT_LE(s, 137);
            sum += s;
            distinct.insert(s);
        }
        EXPECT_GE(distinct.size(), 8u);
        EXPECT_NEAR(sum / g.scales.size(), 126.5, 0.5);
    }

    // FP4 with a UE4M3 or UE5M3 scale takes |N(0.34375, 0.08)| scales, rounded once. With 3
    // mantissa bits the rounding steps near the mean are 2^-5 and 2^-6, which keeps the standard
    // deviation near 0.08 and spreads the scales over many codes; a constant scale fails both.
    TEST(HostUnitMXDataGenMaf2_0, smoke_fp4_ue4m3_ue5m3_scales_are_normal)
    {
        struct ScaleFormat
        {
            hipDataType type;
            int         exponentBits;
        };
        const ScaleFormat formats[] = {
            {HIP_R_8F_E4M3, 4},
            {static_cast<hipDataType>(HIP_R_8F_E5M3_EXT), 5},
        };
        for(auto const& f : formats)
        {
            SCOPED_TRACE("scale type " + std::to_string(f.type));
            const auto g = generate(static_cast<hipDataType>(HIP_R_4F_E2M1), f.type, 16, "maf2_0");
            double     sum = 0, sumSq = 0;
            std::set<uint8_t> distinct;
            for(uint8_t s : g.scales)
            {
                const double v = decodeScale(s, f.exponentBits);
                ASSERT_LT(v, 1.0) << int(s);
                sum += v;
                sumSq += v * v;
                distinct.insert(s);
            }
            const double mean = sum / g.scales.size();
            EXPECT_NEAR(mean, 0.34375, 0.02);
            EXPECT_NEAR(std::sqrt(sumSq / g.scales.size() - mean * mean), 0.08, 0.015);
            EXPECT_GE(distinct.size(), 8u);
        }
    }

    // maf2_0 salts its seed with seedIndex = abc + 3 * batch: the same index repeats the data and
    // scales exactly, while A of batch 0 (0), B of batch 0 (1) and A of batch 1 (3) all differ.
    TEST(HostUnitMXDataGenMaf2_0, smoke_seed_index_selects_the_stream)
    {
        for(auto const* method : {"maf2_0", "maf2_0_unit_scale"})
        {
            SCOPED_TRACE(method);
            auto run = [&](uint32_t seedIndex) {
                return generate(HIP_R_8F_E4M3, HIP_R_8F_UE8M0, 32, method, seedIndex);
            };
            const auto a0 = run(0), again = run(0), b0 = run(1), a1 = run(3);
            EXPECT_EQ(a0.data, again.data);
            EXPECT_EQ(a0.scales, again.scales);
            EXPECT_NE(a0.data, b0.data);
            EXPECT_NE(a0.data, a1.data);
            EXPECT_NE(b0.data, a1.data);
        }
        // Binomial scales differ too; unit scales are all 1.0 by design.
        auto scales = [](uint32_t seedIndex) {
            return generate(HIP_R_8F_E4M3, HIP_R_8F_UE8M0, 32, "maf2_0", seedIndex).scales;
        };
        EXPECT_NE(scales(0), scales(1));
        EXPECT_NE(scales(0), scales(3));
    }

    // testing_matmul passes mxSeedIndex: abc + 3 * batch, and batch 0's index for every batch of
    // a broadcast (stride 0) matrix, so all its batches regenerate the same data.
    TEST(HostUnitMXDataGenMaf2_0, smoke_seed_index_of_each_matrix)
    {
        EXPECT_EQ(mxSeedIndex(/*isMatrixA=*/true, 0, 4096), 0u);
        EXPECT_EQ(mxSeedIndex(/*isMatrixA=*/false, 0, 4096), 1u);
        EXPECT_EQ(mxSeedIndex(/*isMatrixA=*/true, 2, 4096), 6u);
        EXPECT_EQ(mxSeedIndex(/*isMatrixA=*/false, 2, 4096), 7u);
        EXPECT_EQ(mxSeedIndex(/*isMatrixA=*/true, 2, 0), 0u);
        EXPECT_EQ(mxSeedIndex(/*isMatrixA=*/false, 2, 0), 1u);
    }

    // maf2_0_unit_scale (the MXFP8 and FP4 X=1 rows): every block scale is 1.0 and the data is
    // N(0, 1) rounded to the element type.
    TEST(HostUnitMXDataGenMaf2_0, smoke_unit_scale_fp8_and_fp4)
    {
        struct Case
        {
            hipDataType data;
            hipDataType scale;
            int         block;
            uint8_t     one;
        };
        const Case cases[] = {
            {HIP_R_8F_E4M3, HIP_R_8F_UE8M0, 32, 0x7F},
            {HIP_R_8F_E5M2, HIP_R_8F_UE8M0, 32, 0x7F},
            {static_cast<hipDataType>(HIP_R_4F_E2M1), HIP_R_8F_UE8M0, 32, 0x7F},
            {static_cast<hipDataType>(HIP_R_4F_E2M1), HIP_R_8F_E4M3, 16, 0x38},
            {static_cast<hipDataType>(HIP_R_4F_E2M1),
             static_cast<hipDataType>(HIP_R_8F_E5M3_EXT),
             16,
             0x78},
        };
        for(auto const& c : cases)
        {
            SCOPED_TRACE("data type " + std::to_string(c.data) + " scale type "
                         + std::to_string(c.scale));
            const auto g = generate(c.data, c.scale, c.block, "maf2_0_unit_scale");
            for(uint8_t s : g.scales)
                ASSERT_EQ(s, c.one);
            double sum = 0, sumSq = 0;
            for(float v : g.ref)
            {
                sum += v;
                sumSq += double(v) * v;
            }
            const double mean = sum / g.ref.size();
            EXPECT_NEAR(mean, 0.0, 0.05);
            EXPECT_NEAR(std::sqrt(sumSq / g.ref.size() - mean * mean), 1.0, 0.1);
        }
    }
} // namespace

#endif // HIPBLASLT_ENABLE_MXDATAGENERATOR
