// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for maf2_0 and maf2_0_unit_scale block-scaled (MX) data from
// generateMxData: which block-scale distribution each scale format gets, and the unit-scale rows.
// The draws are random, so these check the scale bytes exactly and the data statistically. See
// README.md in this directory for the HostUnit/smoke_ naming.

#include <hipblaslt/hostnumerics/HipblasltDataInitialization.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    namespace hn     = roc::hostnumerics;
    namespace client = hipblaslt::hostnumerics;

    // Unsigned UE4M3 (bias 7) and UE5M3 (bias 15) scales, 3 mantissa bits each.
    double decodeScale(uint8_t code, int exponentBits)
    {
        const int bias     = (1 << (exponentBits - 1)) - 1;
        const int exponent = (code >> 3) & ((1 << exponentBits) - 1);
        const int mantissa = code & 0x7;
        return exponent == 0 ? std::ldexp(mantissa / 8.0, 1 - bias)
                             : std::ldexp(1.0 + mantissa / 8.0, exponent - bias);
    }

    struct Generated
    {
        std::vector<uint8_t> data;
        std::vector<uint8_t> scales;
        std::vector<float>   ref;
    };

    std::vector<uint8_t> bytesOf(hn::Tensor const& tensor)
    {
        std::vector<uint8_t> bytes;
        for(std::byte b : tensor.rawEncodedBackingStorage())
            bytes.push_back(std::to_integer<uint8_t>(b));
        return bytes;
    }

    // A transposed A (K x M stored, K along the rows) of K = 256, M = 64, in batch `batch`.
    Generated generate(hipDataType              dataType,
                       hipDataType              scaleType,
                       int                      block,
                       hipblaslt_initialization method,
                       client::MatrixRole       role  = client::MatrixRole::A,
                       size_t                   batch = 0)
    {
        constexpr uint64_t rows = 256, cols = 64;
        const auto         mx = client::generateMxData(dataType,
                                               scaleType,
                                               hn::Shape{rows, cols},
                                               rows,
                                               /*blockAxis=*/0,
                                               block,
                                               method,
                                               client::mxDefaultSeed,
                                               role,
                                               batch);
        Generated          g;
        g.data   = bytesOf(mx.data);
        g.scales = bytesOf(mx.scales);
        for(uint64_t j = 0; j < cols; j++)
            for(uint64_t i = 0; i < rows; i++)
                g.ref.push_back(mx.reference.loadAs<float>({i, j}));
        EXPECT_EQ(g.scales.size(), cols * rows / block);
        return g;
    }

    // MAF2.0 E8M0 scales are 2^(k - 11), k ~ Binomial(21, 0.5): biased exponents 116..137 with
    // mean 126.5. A normal N(0.34375, 0.08) scale would round to only 2^-3..2^-1 under E8M0.
    TEST(HostUnitMXDataGenMaf2_0, smoke_fp4_e8m0_scales_are_binomial)
    {
        const auto        g   = generate(static_cast<hipDataType>(HIP_R_4F_E2M1),
                                HIP_R_8F_UE8M0,
                                32,
                                hipblaslt_initialization::maf2_0);
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
            const auto        g   = generate(static_cast<hipDataType>(HIP_R_4F_E2M1),
                                    f.type,
                                    16,
                                    hipblaslt_initialization::maf2_0);
            double            sum = 0, sumSq = 0;
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

    // The data and scale streams depend on the matrix role and batch: the same matrix repeats
    // exactly, while A of batch 0, B of batch 0 and A of batch 1 all differ.
    TEST(HostUnitMXDataGenMaf2_0, smoke_role_and_batch_select_the_stream)
    {
        for(auto method :
            {hipblaslt_initialization::maf2_0, hipblaslt_initialization::maf2_0_unit_scale})
        {
            SCOPED_TRACE(hipblaslt_initialization2string(method));
            auto run = [&](client::MatrixRole role, size_t batch) {
                return generate(HIP_R_8F_E4M3, HIP_R_8F_UE8M0, 32, method, role, batch);
            };
            const auto a0    = run(client::MatrixRole::A, 0);
            const auto again = run(client::MatrixRole::A, 0);
            const auto b0    = run(client::MatrixRole::B, 0);
            const auto a1    = run(client::MatrixRole::A, 1);
            EXPECT_EQ(a0.data, again.data);
            EXPECT_EQ(a0.scales, again.scales);
            EXPECT_NE(a0.data, b0.data);
            EXPECT_NE(a0.data, a1.data);
            EXPECT_NE(b0.data, a1.data);
        }
        // Binomial scales differ too; unit scales are all 1.0 by design.
        auto scales = [](client::MatrixRole role, size_t batch) {
            return generate(HIP_R_8F_E4M3,
                            HIP_R_8F_UE8M0,
                            32,
                            hipblaslt_initialization::maf2_0,
                            role,
                            batch)
                .scales;
        };
        EXPECT_NE(scales(client::MatrixRole::A, 0), scales(client::MatrixRole::B, 0));
        EXPECT_NE(scales(client::MatrixRole::A, 0), scales(client::MatrixRole::A, 1));
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
            const auto g
                = generate(c.data, c.scale, c.block, hipblaslt_initialization::maf2_0_unit_scale);
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

    // maf2_0_unit_scale covers block-scaled FP8 and FP4 only; FP6 is refused before writing.
    TEST(HostUnitMXDataGenMaf2_0, smoke_unit_scale_refuses_fp6)
    {
        for(hipDataType type : {HIP_R_6F_E2M3, HIP_R_6F_E3M2})
            EXPECT_THROW(
                generate(type, HIP_R_8F_UE8M0, 32, hipblaslt_initialization::maf2_0_unit_scale),
                std::runtime_error)
                << "type " << int(type);
    }
} // namespace
