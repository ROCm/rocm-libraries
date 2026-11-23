// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for the maf1_0 method of generateMXInput (mxDataGen.cpp). maf1_0 is
// closed-form, so every element is checked exactly: element (i, j) of the stored matrix of
// batch b must hold the code nearest (ties to even) to sin(idx) for A or cos(idx) for B, with
// idx = i + j * rows + b * rows * cols, and every block scale must be 0.5. The expected codes come
// from a brute-force search over the element encodings written here, independent of
// mxDataGenerator. See README.md in this directory for the HostUnit/smoke_ naming.

#if HIPBLASLT_ENABLE_MXDATAGENERATOR

#include "mxDataGen.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace
{
    // OCP E4M3: bias 7, 3 mantissa bits, no infinities, S.1111.111 is NaN.
    double decodeE4M3(uint8_t code)
    {
        const int    exponent = (code >> 3) & 0xF;
        const int    mantissa = code & 0x7;
        const double value    = exponent == 0 ? std::ldexp(mantissa / 8.0, -6)
                                              : std::ldexp(1.0 + mantissa / 8.0, exponent - 7);
        return (code & 0x80) ? -value : value;
    }

    // OCP E2M1: bias 1, 1 mantissa bit.
    double decodeE2M1(uint8_t code)
    {
        const int    exponent = (code >> 1) & 0x3;
        const int    mantissa = code & 0x1;
        const double value
            = exponent == 0 ? mantissa * 0.5 : std::ldexp(1.0 + mantissa / 2.0, exponent - 1);
        return (code & 0x8) ? -value : value;
    }

    // Nearest finite code to v, ties to the even code (mantissa LSB 0), keeping the sign of v.
    template <typename Decode>
    uint8_t nearestCode(double v, Decode decode, uint8_t signBit, uint8_t maxMagnitudeCode)
    {
        const double magnitude = std::fabs(v);
        uint8_t      best      = 0;
        for(int c = 1; c <= maxMagnitudeCode; c++)
        {
            const double d     = std::fabs(decode(uint8_t(c)) - magnitude);
            const double dBest = std::fabs(decode(best) - magnitude);
            if(d < dBest || (d == dBest && (c & 1) == 0))
                best = uint8_t(c);
        }
        return std::signbit(v) ? uint8_t(best | signBit) : best;
    }

    double trig(bool isMatrixA, uint64_t idx)
    {
        return isMatrixA ? std::sin(double(idx)) : std::cos(double(idx));
    }

    struct Layout
    {
        bool isMatrixA;
        bool isTranspose;
        // Stored matrix (column-major, dense).
        uint64_t rows;
        uint64_t cols;
    };

    std::string describe(Layout const& l, uint32_t batch)
    {
        return std::string(l.isMatrixA ? "A" : "B") + (l.isTranspose ? "T" : "N") + " "
               + std::to_string(l.rows) + "x" + std::to_string(l.cols) + " batch "
               + std::to_string(batch);
    }

    // Both orientations of A and B: K along the stored rows (AT, BN) and K along the stored
    // columns (AN, BT), with M or N not a multiple of the block.
    const std::vector<Layout> kLayouts = {
        {true, false, 40, 64}, // AN: M x K
        {true, true, 64, 40}, // AT: K x M
        {false, false, 64, 24}, // BN: K x N
        {false, true, 24, 64}, // BT: N x K
    };

    TEST(HostUnitMXDataGenMaf1_0, smoke_fp8_e4m3_values_and_scales)
    {
        constexpr int block = 32;
        for(auto const& l : kLayouts)
        {
            for(uint32_t batch : {0u, 1u, 5u})
            {
                SCOPED_TRACE(describe(l, batch));
                const size_t         count = l.rows * l.cols;
                std::vector<uint8_t> data(count, 0xAA);
                std::vector<uint8_t> scale(count, 0xAA);
                const uint32_t       seedIndex = (l.isMatrixA ? 0u : 1u) + 3u * batch;
                const auto           ref       = generateMXInput(HIP_R_8F_E4M3,
                                                 HIP_R_8F_UE8M0,
                                                 data.data(),
                                                 scale.data(),
                                                 l.rows,
                                                 l.cols,
                                                 l.rows,
                                                 l.isTranspose,
                                                 l.isMatrixA ? block : 1,
                                                 l.isMatrixA ? 1 : block,
                                                 l.isMatrixA,
                                                 MXScaleLayout::None,
                                                 "maf1_0",
                                                 -1.0f,
                                                 1.0f,
                                                 "",
                                                 seedIndex);
                ASSERT_EQ(ref.size(), count);
                for(uint64_t j = 0; j < l.cols; j++)
                    for(uint64_t i = 0; i < l.rows; i++)
                    {
                        const uint64_t idx = i + j * l.rows + batch * count;
                        const uint8_t  expected
                            = nearestCode(trig(l.isMatrixA, idx), decodeE4M3, 0x80, 0x7E);
                        const size_t o = i + j * l.rows;
                        ASSERT_EQ(data[o], expected) << "i " << i << " j " << j;
                        ASSERT_EQ(ref[o], float(0.5 * decodeE4M3(expected)))
                            << "i " << i << " j " << j;
                    }
                // Every block scale written is E8M0 0.5; at least one per block of data.
                size_t written = 0;
                for(uint8_t s : scale)
                {
                    if(s == 0xAA)
                        continue;
                    ASSERT_EQ(s, 0x7E);
                    written++;
                }
                EXPECT_GE(written, count / block);
            }
        }
    }

    // FP4 packs two elements per byte, the first in the low nibble. Scale formats: E8M0 block 32,
    // UE4M3 block 16 and UE5M3 block 16 (scaleA/B 3, 6 and 8), whose 0.5 encodings are 0x7E,
    // 0x30 and 0x70.
    TEST(HostUnitMXDataGenMaf1_0, smoke_fp4_values_and_scales)
    {
        struct ScaleFormat
        {
            hipDataType type;
            int         block;
            uint8_t     half;
        };
        const ScaleFormat formats[] = {
            {HIP_R_8F_UE8M0, 32, 0x7E},
            {HIP_R_8F_E4M3, 16, 0x30},
            {static_cast<hipDataType>(HIP_R_8F_E5M3_EXT), 16, 0x70},
        };
        for(auto const& f : formats)
        {
            for(auto const& l : kLayouts)
            {
                for(uint32_t batch : {0u, 2u})
                {
                    SCOPED_TRACE(describe(l, batch) + " scale type " + std::to_string(f.type));
                    const size_t         count = l.rows * l.cols;
                    std::vector<uint8_t> data(count / 2, 0xAA);
                    std::vector<uint8_t> scale(count, 0xAA);
                    const uint32_t       seedIndex = (l.isMatrixA ? 0u : 1u) + 3u * batch;
                    const auto ref = generateMXInput(static_cast<hipDataType>(HIP_R_4F_E2M1),
                                                     f.type,
                                                     data.data(),
                                                     scale.data(),
                                                     l.rows,
                                                     l.cols,
                                                     l.rows,
                                                     l.isTranspose,
                                                     l.isMatrixA ? f.block : 1,
                                                     l.isMatrixA ? 1 : f.block,
                                                     l.isMatrixA,
                                                     MXScaleLayout::None,
                                                     "maf1_0",
                                                     -1.0f,
                                                     1.0f,
                                                     "",
                                                     seedIndex);
                    ASSERT_EQ(ref.size(), count);
                    for(uint64_t j = 0; j < l.cols; j++)
                        for(uint64_t i = 0; i < l.rows; i++)
                        {
                            const uint64_t idx = i + j * l.rows + batch * count;
                            const uint8_t  expected
                                = nearestCode(trig(l.isMatrixA, idx), decodeE2M1, 0x8, 0x7);
                            const size_t  o    = i + j * l.rows;
                            const uint8_t code = (data[o / 2] >> (4 * (o % 2))) & 0xF;
                            ASSERT_EQ(code, expected) << "i " << i << " j " << j;
                            ASSERT_EQ(ref[o], float(0.5 * decodeE2M1(expected)))
                                << "i " << i << " j " << j;
                        }
                    size_t written = 0;
                    for(uint8_t s : scale)
                    {
                        if(s == 0xAA)
                            continue;
                        ASSERT_EQ(s, f.half);
                        written++;
                    }
                    EXPECT_GE(written, count / f.block);
                }
            }
        }
    }

    // The gfx1250 scale swizzle only reorders scale bytes, and maf1_0 data does not depend on
    // the scale layout.
    TEST(HostUnitMXDataGenMaf1_0, smoke_fp8_gfx1250_layout_keeps_data)
    {
        for(auto const& l : kLayouts)
        {
            SCOPED_TRACE(describe(l, 0));
            const size_t         count = l.rows * l.cols;
            std::vector<uint8_t> dataNone(count), dataSwizzled(count);
            std::vector<uint8_t> scaleNone(count, 0xAA), scaleSwizzled(4 * count, 0xAA);
            for(auto layout : {MXScaleLayout::None, MXScaleLayout::GFX1250})
            {
                const bool swizzled = layout == MXScaleLayout::GFX1250;
                generateMXInput(HIP_R_8F_E4M3,
                                HIP_R_8F_UE8M0,
                                swizzled ? dataSwizzled.data() : dataNone.data(),
                                swizzled ? scaleSwizzled.data() : scaleNone.data(),
                                l.rows,
                                l.cols,
                                l.rows,
                                l.isTranspose,
                                l.isMatrixA ? 32 : 1,
                                l.isMatrixA ? 1 : 32,
                                l.isMatrixA,
                                layout,
                                "maf1_0",
                                -1.0f,
                                1.0f,
                                "",
                                l.isMatrixA ? 0u : 1u);
            }
            EXPECT_EQ(dataNone, dataSwizzled);
            for(uint8_t s : scaleSwizzled)
                if(s != 0xAA)
                    ASSERT_TRUE(s == 0x7E || s == 0x00) << int(s); // the swizzle may zero-pad
        }
    }
} // namespace

#endif // HIPBLASLT_ENABLE_MXDATAGENERATOR
