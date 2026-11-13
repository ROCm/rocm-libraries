// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for the maf2_1 method of generateMXInput (mxDataGen.cpp): E8M0 block scales
// are 2^e with e uniform over [-127, 0], UE4M3/UE5M3 scales U[0, 1] rounded to the scale format, and
// the data U[-1, 1] rounded to the element. The draws are random, so these check ranges exactly and
// distributions statistically. See README.md in this directory for the HostUnit/smoke_ naming.

#if HIPBLASLT_ENABLE_MXDATAGENERATOR

#include "mxDataGen.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
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
        std::vector<uint8_t> scale(count, kUnwritten);
        Generated            g;
        g.data.resize(fp4 ? count / 2 : count);
        g.ref = generateMXInput(dataType,
                                scaleType,
                                g.data.data(),
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
        for(uint8_t s : scale)
            if(s != kUnwritten)
                g.scales.push_back(s);
        EXPECT_EQ(g.ref.size(), count);
        EXPECT_GE(g.scales.size(), count / block);
        return g;
    }

    // E8M0 scales are 2^e with e uniform over [-127, 0]: biased exponents 0..127, mean 63.5.
    TEST(HostUnitMXDataGenMaf2_1, smoke_e8m0_scales_are_exponent_uniform)
    {
        for(auto dataType : {HIP_R_8F_E4M3, static_cast<hipDataType>(HIP_R_4F_E2M1)})
        {
            SCOPED_TRACE("data type " + std::to_string(dataType));
            const auto        g   = generate(dataType, HIP_R_8F_UE8M0, 32, "maf2_1");
            double            sum = 0;
            std::set<uint8_t> distinct;
            for(uint8_t s : g.scales)
            {
                ASSERT_LE(s, 127);
                sum += s;
                distinct.insert(s);
            }
            EXPECT_GE(distinct.size(), 100u);
            EXPECT_NEAR(sum / g.scales.size(), 63.5, 6.0);
        }
    }

    // UE4M3 / UE5M3 scales are U[0, 1] rounded to the scale format: values in [0, 1], mean 0.5.
    TEST(HostUnitMXDataGenMaf2_1, smoke_ue4m3_ue5m3_scales_are_value_uniform)
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
            const auto g = generate(static_cast<hipDataType>(HIP_R_4F_E2M1), f.type, 16, "maf2_1");
            double     sum = 0;
            for(uint8_t s : g.scales)
            {
                const double v = decodeScale(s, f.exponentBits);
                ASSERT_LE(v, 1.0) << int(s);
                sum += v;
            }
            EXPECT_NEAR(sum / g.scales.size(), 0.5, 0.03);
        }
    }

    // The reference is the data value times its block scale; with data in [-1, 1], every reference
    // magnitude is at most the largest scale (1). A and B, and maf2_0, use different streams.
    TEST(HostUnitMXDataGenMaf2_1, smoke_data_bounded_and_streams_differ)
    {
        const auto a = generate(HIP_R_8F_E4M3, HIP_R_8F_UE8M0, 32, "maf2_1", 0);
        for(float v : a.ref)
            ASSERT_LE(std::fabs(v), 1.0f);
        const auto b = generate(HIP_R_8F_E4M3, HIP_R_8F_UE8M0, 32, "maf2_1", 1);
        const auto n = generate(HIP_R_8F_E4M3, HIP_R_8F_UE8M0, 32, "maf2_0", 0);
        EXPECT_NE(a.data, b.data);
        EXPECT_NE(a.scales, b.scales);
        EXPECT_NE(a.scales, n.scales);
        const auto again = generate(HIP_R_8F_E4M3, HIP_R_8F_UE8M0, 32, "maf2_1", 0);
        EXPECT_EQ(a.data, again.data);
        EXPECT_EQ(a.scales, again.scales);
    }
} // namespace

#endif // HIPBLASLT_ENABLE_MXDATAGENERATOR
