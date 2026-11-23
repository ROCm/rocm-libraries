// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for maf1_0 block-scaled (MX) data from generateMxData. maf1_0 is
// closed-form, so every element is checked exactly: element (i, j) of the stored matrix of
// batch b must hold the code nearest (ties to even) to sin(idx) for A or cos(idx) for B, with
// idx = i + j * rows + b * rows * cols, and every block scale must be 0.5. The expected codes come
// from a brute-force search over the element encodings written here, independent of
// HostNumerics. See README.md in this directory for the HostUnit/smoke_ naming.

#include <hipblaslt/hostnumerics/HipblasltDataInitialization.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace
{
    namespace hn     = roc::hostnumerics;
    namespace client = hipblaslt::hostnumerics;

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

    // K runs along the stored rows (block axis 0) for transposed A and non-transposed B.
    size_t blockAxis(Layout const& l)
    {
        return l.isMatrixA == l.isTranspose ? 0 : 1;
    }

    hn::MxTensor generate(
        hipDataType dataType, hipDataType scaleType, Layout const& l, int block, uint32_t batch)
    {
        return client::generateMxData(dataType,
                                      scaleType,
                                      hn::Shape{l.rows, l.cols},
                                      l.rows,
                                      blockAxis(l),
                                      block,
                                      hipblaslt_initialization::maf1_0,
                                      client::mxDefaultSeed,
                                      l.isMatrixA ? client::MatrixRole::A : client::MatrixRole::B,
                                      batch);
    }

    size_t expectedScaleCount(Layout const& l, int block)
    {
        const uint64_t k    = blockAxis(l) == 0 ? l.rows : l.cols;
        const uint64_t free = blockAxis(l) == 0 ? l.cols : l.rows;
        return free * ((k + block - 1) / block);
    }

    void expectEveryScaleByte(hn::MxTensor const& mx, size_t count, uint8_t expected)
    {
        const auto bytes = mx.scales.rawEncodedBackingStorage();
        ASSERT_EQ(bytes.size(), count);
        for(std::byte s : bytes)
            ASSERT_EQ(std::to_integer<uint8_t>(s), expected);
    }

    TEST(HostUnitMXDataGenMaf1_0, smoke_fp8_e4m3_values_and_scales)
    {
        constexpr int block = 32;
        for(auto const& l : kLayouts)
        {
            for(uint32_t batch : {0u, 1u, 5u})
            {
                SCOPED_TRACE(describe(l, batch));
                const size_t count = l.rows * l.cols;
                const auto   mx    = generate(HIP_R_8F_E4M3, HIP_R_8F_UE8M0, l, block, batch);
                const auto   data  = mx.data.rawEncodedBackingStorage();
                ASSERT_EQ(data.size(), count);
                for(uint64_t j = 0; j < l.cols; j++)
                    for(uint64_t i = 0; i < l.rows; i++)
                    {
                        const uint64_t idx = i + j * l.rows + batch * count;
                        const uint8_t  expected
                            = nearestCode(trig(l.isMatrixA, idx), decodeE4M3, 0x80, 0x7E);
                        ASSERT_EQ(std::to_integer<uint8_t>(data[i + j * l.rows]), expected)
                            << "i " << i << " j " << j;
                        ASSERT_EQ(mx.reference.loadAs<float>({i, j}),
                                  float(0.5 * decodeE4M3(expected)))
                            << "i " << i << " j " << j;
                    }
                // Every block scale is E8M0 0.5.
                expectEveryScaleByte(mx, expectedScaleCount(l, block), 0x7E);
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
                    const size_t count = l.rows * l.cols;
                    const auto   mx    = generate(
                        static_cast<hipDataType>(HIP_R_4F_E2M1), f.type, l, f.block, batch);
                    const auto data = mx.data.rawEncodedBackingStorage();
                    ASSERT_EQ(data.size(), count / 2);
                    for(uint64_t j = 0; j < l.cols; j++)
                        for(uint64_t i = 0; i < l.rows; i++)
                        {
                            const uint64_t idx = i + j * l.rows + batch * count;
                            const uint8_t  expected
                                = nearestCode(trig(l.isMatrixA, idx), decodeE2M1, 0x8, 0x7);
                            const size_t  o = i + j * l.rows;
                            const uint8_t code
                                = (std::to_integer<uint8_t>(data[o / 2]) >> (4 * (o % 2))) & 0xF;
                            ASSERT_EQ(code, expected) << "i " << i << " j " << j;
                            ASSERT_EQ(mx.reference.loadAs<float>({i, j}),
                                      float(0.5 * decodeE2M1(expected)))
                                << "i " << i << " j " << j;
                        }
                    expectEveryScaleByte(mx, expectedScaleCount(l, f.block), f.half);
                }
            }
        }
    }

    // The gfx1250 scale tiling only reorders scale bytes and pads them with zeros; maf1_0 data
    // does not depend on the scale layout, which generateMxData does not take.
    TEST(HostUnitMXDataGenMaf1_0, smoke_fp8_gfx1250_layout_keeps_scales)
    {
        constexpr size_t block = 32;
        for(auto const& l : kLayouts)
        {
            SCOPED_TRACE(describe(l, 0));
            const auto   mx      = generate(HIP_R_8F_E4M3, HIP_R_8F_UE8M0, l, block, 0);
            const size_t axis    = blockAxis(l);
            const size_t free    = axis == 0 ? l.cols : l.rows;
            const size_t kBlocks = ((axis == 0 ? l.rows : l.cols) + block - 1) / block;
            // The natural scales are [free, K blocks] when K runs along the stored rows and
            // [K blocks, free] otherwise; gfx1250 tiles the fast dimension.
            const std::array<size_t, 2> natural = axis == 0 ? std::array<size_t, 2>{free, kBlocks}
                                                            : std::array<size_t, 2>{kBlocks, free};
            const auto                  plan
                = hn::amd_gpu_layout::MxScaleStoragePlan::fastDimensionTiles(natural, 128 / block);
            const auto device = client::mxScaleBytesForDevice(
                mx, axis, plan, client::MxScaleStorageLayout::FastDimensionTiles);
            ASSERT_EQ(device.size(), plan.physicalByteCount());
            size_t written = 0;
            for(std::byte s : device)
            {
                const uint8_t b = std::to_integer<uint8_t>(s);
                ASSERT_TRUE(b == 0x7E || b == 0x00) << int(b);
                written += b == 0x7E;
            }
            EXPECT_EQ(written, expectedScaleCount(l, block));
        }
    }
} // namespace
