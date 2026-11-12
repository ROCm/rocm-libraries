// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Maf1_0: every block scale 0.5, each element sin(idx) or cos(idx) rounded to the nearest
// even element code at a scale of one, where idx is the element's logical index.

#include <mxDataGenerator/DataGenerator.hpp>

#include <gtest/gtest.h>

#include <omp.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

namespace
{
    using namespace DGen;

    template <typename DT>
    class Maf1_0Test : public ::testing::Test
    {
    };

    using Maf1_0Types = ::testing::Types<ocp_e4m3_mxfp8,
                                         ocp_e5m2_mxfp8,
                                         ocp_e2m3_mxfp6,
                                         ocp_e3m2_mxfp6,
                                         ocp_e2m1_mxfp4,
                                         ocp_e2m1_mxfp4_e4m3,
                                         ocp_e2m1_mxfp4_e5m3>;
    TYPED_TEST_SUITE(Maf1_0Test, Maf1_0Types);

    DataGeneratorOptions maf1_0Options(index_t blockSize, Maf1_0 mode = {})
    {
        DataGeneratorOptions opts;
        opts.initMode     = DataInitMode(mode);
        opts.blockScaling = blockSize;
        return opts;
    }

    // Encoding of 0.5 in each scale format.
    template <typename DT>
    uint8_t halfScaleByte()
    {
        return scale_info_t<DT>::mantissaBits == 0   ? 0x7E // E8M0
               : scale_info_t<DT>::exponentBits == 4 ? 0x30 // UE4M3
                                                     : 0x70; // UE5M3
    }

    template <typename DT>
    uint8_t scaleOneByte()
    {
        uint8_t scale[1] = {0};
        uint8_t data[1]  = {0};
        setOne<DT>(scale, data, 0, 0, false);
        return scale[0];
    }

    template <typename DT>
    double decode(uint8_t scale, uint8_t code)
    {
        return toDouble<DT>(&scale, &code, 0, 0);
    }

    // Element code of x at a scale of one, by brute force over every code with x's sign:
    // nearest finite magnitude, ties to the even code.
    template <typename DT>
    uint8_t rneCode(double x)
    {
        const uint8_t one = scaleOneByte<DT>();
        const int     bits
            = DT::dataInfo.signBits + DT::dataInfo.exponentBits + DT::dataInfo.mantissaBits;
        const uint8_t signBit = static_cast<uint8_t>(1u << (bits - 1));
        int           best    = -1;
        double        bestD   = std::numeric_limits<double>::infinity();
        for(int c = 0; c < (1 << bits); ++c)
        {
            const uint8_t code = static_cast<uint8_t>(c);
            if(((code & signBit) != 0) != std::signbit(x))
                continue;
            const double v = decode<DT>(one, code);
            if(!std::isfinite(v))
                continue;
            const double d = std::abs(v - x);
            if(d < bestD || (d == bestD && (code & 1) == 0))
            {
                best  = c;
                bestD = d;
            }
        }
        return static_cast<uint8_t>(best);
    }

    // Checks every element of dgen (stored over arraySize memory offsets) against
    // 0.5 * RNE(f(idx(offset))); idx < 0 marks a padding offset, expected to be +0.
    template <typename DT>
    void expectTrig(DataGenerator<DT> const&               dgen,
                    index_t                                arraySize,
                    bool                                   useCos,
                    std::function<int64_t(index_t)> const& idxOf)
    {
        const uint8_t half   = halfScaleByte<DT>();
        const auto    scales = dgen.getScaleBytes();
        ASSERT_EQ(scales.size(), static_cast<size_t>(dgen.numScaleBlocks()));
        for(size_t s = 0; s < scales.size(); ++s)
            ASSERT_EQ(scales[s], half) << "s=" << s;

        const auto ref = dgen.getReferenceDouble();
        ASSERT_EQ(ref.size(), static_cast<size_t>(arraySize));
        std::vector<uint8_t> codes(arraySize);
        for(index_t i = 0; i < arraySize; ++i)
        {
            const int64_t idx     = idxOf(i);
            const double  x       = static_cast<double>(idx);
            codes[i]              = idx < 0 ? 0 : rneCode<DT>(useCos ? std::cos(x) : std::sin(x));
            const double expected = 0.5 * decode<DT>(scaleOneByte<DT>(), codes[i]);
            ASSERT_EQ(ref[i], expected) << "i=" << i << " idx=" << idx;
            ASSERT_EQ(std::signbit(ref[i]), std::signbit(expected)) << "i=" << i;
            ASSERT_EQ(ref[i], decode<DT>(half, codes[i])) << "i=" << i;
        }
        if constexpr(DT::dataInfo.exponentBits + DT::dataInfo.mantissaBits == 7)
        {
            EXPECT_EQ(dgen.getDataBytes(), codes);
        }
    }

    TEST(Maf1_0Mode, ToStringAndEquality)
    {
        EXPECT_EQ(toString(DataInitMode(Maf1_0{})),
                  "DataInitMode(Maf1_0(sin, scale 0.500000, offset 0))");
        Maf1_0 m;
        m.useCos       = true;
        m.indexOffset  = 12;
        m.indexStrides = {40, 1};
        EXPECT_EQ(m.toString(), "Maf1_0(cos, scale 0.500000, offset 12, strides 40 1)");
        EXPECT_TRUE(dataInitModesEqual(Maf1_0{}, Maf1_0{0.5, false, 0, {}}));
        EXPECT_FALSE(dataInitModesEqual(Maf1_0{}, Maf1_0{0.5, true, 0, {}}));
        EXPECT_FALSE(dataInitModesEqual(Maf1_0{}, Maf1_0{0.5, false, 1, {}}));
        EXPECT_FALSE(dataInitModesEqual(Maf1_0{}, Maf1_0{0.5, false, 0, {1}}));
        EXPECT_FALSE(isRandomLikeDataInitMode(Maf1_0{}));
    }

    TEST(Maf1_0Mode, UnscaledTypesThrow)
    {
        EXPECT_THROW(DataGenerator<f32>().generate({16}, {1}, maf1_0Options(1)),
                     std::invalid_argument);
        EXPECT_THROW(DataGenerator<fp16>().generate({16}, {1}, maf1_0Options(1)),
                     std::invalid_argument);
        EXPECT_THROW(DataGenerator<bf16>().generate({16}, {1}, maf1_0Options(1)),
                     std::invalid_argument);
    }

    TEST(Maf1_0Mode, InvalidOptionsThrow)
    {
        Maf1_0 badStrides;
        badStrides.indexStrides = {1};
        EXPECT_THROW(DataGenerator<ocp_e4m3_mxfp8>().generate(
                         {32, 2}, {1, 32}, maf1_0Options(32, badStrides)),
                     std::invalid_argument);
        for(double s : {0.0, -0.5, std::numeric_limits<double>::infinity()})
            EXPECT_THROW(DataGenerator<ocp_e4m3_mxfp8>().generate(
                             {32}, {1}, maf1_0Options(32, Maf1_0{s, false, 0, {}})),
                         std::invalid_argument);
    }

    // Hand-computed codes: sin(1) = 0.8415 -> E4M3 0.8125 (0x35), FP4 1.0 (0x2);
    // sin(2) = 0.9093 -> E4M3 0.9375 (0x37); cos(0) = 1 -> E4M3 0x38.
    TEST(Maf1_0Mode, HandComputedGoldens)
    {
        DataGenerator<ocp_e4m3_mxfp8> e4m3;
        e4m3.generate({32}, {1}, maf1_0Options(32));
        const auto b = e4m3.getDataBytes();
        EXPECT_EQ(b[0], 0x00);
        EXPECT_EQ(b[1], 0x35);
        EXPECT_EQ(b[2], 0x37);
        EXPECT_EQ(e4m3.getScaleBytes(), std::vector<uint8_t>{0x7E});
        EXPECT_EQ(e4m3.getReferenceDouble()[1], 0.40625);

        e4m3.generate({32}, {1}, maf1_0Options(32, Maf1_0{0.5, true, 0, {}}));
        EXPECT_EQ(e4m3.getDataBytes()[0], 0x38);
        EXPECT_EQ(e4m3.getReferenceDouble()[0], 0.5);

        DataGenerator<ocp_e2m1_mxfp4> fp4;
        fp4.generate({32}, {1}, maf1_0Options(32));
        EXPECT_EQ(fp4.getDataBytes()[0] >> 4, 0x2); // element 1 in the high nibble
        EXPECT_EQ(fp4.getReferenceDouble()[1], 0.5);
    }

    // Dense column-major matrix with a tail block (40 = 32 + 8): idx = i + j * rows.
    TYPED_TEST(Maf1_0Test, DenseSinAndCos)
    {
        for(bool useCos : {false, true})
        {
            for(index_t bs : {16, 32})
            {
                DataGenerator<TypeParam> dgen;
                dgen.generate({40, 37}, {1, 40}, maf1_0Options(bs, Maf1_0{0.5, useCos, 0, {}}));
                expectTrig(dgen, 40 * 37, useCos, [](index_t i) { return int64_t(i); });
            }
        }
    }

    // Batch offset in indexOffset, 3-D dense: idx = offset + i + 8 j + 40 b.
    TYPED_TEST(Maf1_0Test, IndexOffsetAndBatch)
    {
        const uint64_t offsets[] = {1000, (uint64_t(1) << 33) + 5};
        for(uint64_t off : offsets)
        {
            DataGenerator<TypeParam> dgen;
            dgen.generate({8, 5, 3}, {1, 8, 40}, maf1_0Options(16, Maf1_0{0.5, true, off, {}}));
            expectTrig(dgen, 120, true, [&](index_t i) { return int64_t(off + i); });
        }
    }

    // Dimension 1 contiguous (sizes {rows, cols}, strides {cols, 1}). The default dense index
    // runs over dimension 0 first; explicit indexStrides can select the memory order instead.
    TYPED_TEST(Maf1_0Test, TransposedLayoutAndIndexStrides)
    {
        const index_t rows = 37, cols = 40;
        {
            DataGenerator<TypeParam> dgen;
            dgen.generate({rows, cols}, {cols, 1}, maf1_0Options(32));
            expectTrig(dgen, rows * cols, false, [&](index_t off) {
                const index_t r = off / cols, c = off % cols;
                return int64_t(r + c * rows);
            });
        }
        {
            Maf1_0 m;
            m.indexStrides = {cols, 1};
            DataGenerator<TypeParam> dgen;
            dgen.generate({rows, cols}, {cols, 1}, maf1_0Options(32, m));
            expectTrig(dgen, rows * cols, false, [](index_t off) { return int64_t(off); });
        }
        {
            // Explicit strides on a column-major layout: idx = 2 i + 1000 j.
            Maf1_0 m;
            m.indexStrides = {2, 1000};
            DataGenerator<TypeParam> dgen;
            dgen.generate({40, 6}, {1, 40}, maf1_0Options(16, m));
            expectTrig(dgen, 240, false, [](index_t off) {
                return int64_t(2 * (off % 40) + 1000 * (off / 40));
            });
        }
    }

    // Leading dimension 12 > rows 6: padding cells are +0, elements use the logical index.
    // The scale-block layout splits memory into runs of rows elements, so it only covers the
    // whole buffer when ld is a multiple of rows.
    TYPED_TEST(Maf1_0Test, PaddedLeadingDimension)
    {
        DataGenerator<TypeParam> dgen;
        dgen.generate({6, 5}, {1, 12}, maf1_0Options(2));
        expectTrig(dgen, 12 * 5, false, [](index_t off) {
            const index_t i = off % 12, j = off / 12;
            return i < 6 ? int64_t(i + 6 * j) : int64_t(-1);
        });
    }

    // Size-1 dimensions sharing a stride with another dimension.
    TYPED_TEST(Maf1_0Test, SizeOneDimensions)
    {
        DataGenerator<TypeParam> dgen;
        dgen.generate({1, 48, 1}, {1, 1, 48}, maf1_0Options(16));
        expectTrig(dgen, 48, false, [](index_t off) { return int64_t(off); });

        // A size-1 contiguous dimension under a strided one: offsets 1, 2 are padding.
        DataGenerator<TypeParam> strided;
        strided.generate({1, 16}, {1, 3}, maf1_0Options(1));
        expectTrig(strided, 48, false, [](index_t off) {
            return off % 3 == 0 ? int64_t(off / 3) : int64_t(-1);
        });
    }

    TYPED_TEST(Maf1_0Test, IndependentOfThreadCount)
    {
        const int            saved = omp_get_max_threads();
        std::vector<uint8_t> data0, scale0;
        for(int threads : {1, 7, 32})
        {
            omp_set_num_threads(threads);
            DataGenerator<TypeParam> dgen;
            dgen.setSeed(1234 + threads);
            dgen.generate({40, 37}, {1, 40}, maf1_0Options(16, Maf1_0{0.5, true, 77, {}}));
            if(data0.empty())
            {
                data0  = dgen.getDataBytes();
                scale0 = dgen.getScaleBytes();
            }
            EXPECT_EQ(dgen.getDataBytes(), data0) << "threads=" << threads;
            EXPECT_EQ(dgen.getScaleBytes(), scale0) << "threads=" << threads;
        }
        omp_set_num_threads(saved);
    }

    TYPED_TEST(Maf1_0Test, IgnoresRangeAndSpecialValueOptions)
    {
        auto run = [](DataGeneratorOptions const& o) {
            DataGenerator<TypeParam> dgen;
            dgen.generate({64, 3}, {1, 64}, o);
            return std::make_pair(dgen.getDataBytes(), dgen.getScaleBytes());
        };
        const auto base    = run(maf1_0Options(32));
        auto       opts    = maf1_0Options(32);
        opts.min           = -0.01;
        opts.max           = 0.01;
        opts.includeNaN    = true;
        opts.includeInf    = true;
        opts.forceDenorm   = true;
        opts.scaleInitMode = DataInitMode(Ones{});
        EXPECT_EQ(run(opts), base);
    }

    TEST(Maf1_0Mode, OtherScaleValues)
    {
        DataGenerator<ocp_e4m3_mxfp8> e8;
        e8.generate({32}, {1}, maf1_0Options(32, Maf1_0{2.0, false, 0, {}}));
        EXPECT_EQ(e8.getScaleBytes(), std::vector<uint8_t>{0x80});
        DataGenerator<ocp_e2m1_mxfp4_e4m3> e4;
        e4.generate({16}, {1}, maf1_0Options(16, Maf1_0{1e6, false, 0, {}}));
        EXPECT_EQ(e4.getScaleBytes(), std::vector<uint8_t>{0x7E}); // saturates to 448
    }
} // namespace
