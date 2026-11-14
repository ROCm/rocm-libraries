// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Maf2_1: data x ~ U[-1, 1] rounded to the element format at a scale of one; E8M0 block scales
// 2^e with e uniform over [-127, 0], other scale formats U[0, 1] rounded to the scale format;
// drawn per block in the mxDataGenerator thread layout.

#include <mxDataGenerator/DataGenerator.hpp>

#include <gtest/gtest.h>

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

namespace
{
    using namespace DGen;

    template <typename DT>
    class Maf2_1Test : public ::testing::Test
    {
    };

    using Maf2_1Types = ::testing::Types<ocp_e4m3_mxfp8,
                                         ocp_e5m2_mxfp8,
                                         ocp_e2m3_mxfp6,
                                         ocp_e3m2_mxfp6,
                                         ocp_e2m1_mxfp4,
                                         ocp_e2m1_mxfp4_e4m3,
                                         ocp_e2m1_mxfp4_e5m3>;
    TYPED_TEST_SUITE(Maf2_1Test, Maf2_1Types);

    DataGeneratorOptions maf2_1Options(index_t blockSize)
    {
        DataGeneratorOptions opts;
        opts.initMode     = DataInitMode(Maf2_1{});
        opts.blockScaling = blockSize;
        return opts;
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
    uint8_t signBitOf()
    {
        return static_cast<uint8_t>(1u << (DT::dataInfo.exponentBits + DT::dataInfo.mantissaBits));
    }

    template <typename DT>
    double decode(uint8_t scale, uint8_t code)
    {
        return toDouble<DT>(&scale, &code, 0, 0);
    }

    // Chi-square of observed against expected counts over the same keys, pooling neighbouring
    // bins until each pooled expected count is at least 5.
    template <typename K>
    double pooledChi2(std::map<K, double> const& expected, std::map<K, double>& observed)
    {
        double chi2 = 0, eAcc = 0, oAcc = 0;
        for(auto const& [key, e] : expected)
        {
            eAcc += e;
            oAcc += observed[key];
            if(eAcc >= 5)
            {
                chi2 += (oAcc - eAcc) * (oAcc - eAcc) / eAcc;
                eAcc = oAcc = 0;
            }
        }
        return chi2;
    }

    // Length of [lo, hi) ∩ [a, b).
    double overlap(double lo, double hi, double a, double b)
    {
        return std::max(0.0, std::min(hi, b) - std::max(lo, a));
    }

    TEST(Maf2_1Mode, ToStringAndEquality)
    {
        EXPECT_EQ(toString(DataInitMode(Maf2_1{})),
                  "DataInitMode(Maf2_1(data U[-1.000000, 1.000000], scale U[0.000000, 1.000000]))");
        EXPECT_TRUE(dataInitModesEqual(Maf2_1{}, Maf2_1{-1.0, 1.0, 0.0, 1.0}));
        EXPECT_FALSE(dataInitModesEqual(Maf2_1{}, Maf2_0{}));
        EXPECT_FALSE(dataInitModesEqual(Maf2_1{}, Maf2_1{-4.0, 4.0}));
        EXPECT_FALSE(isRandomLikeDataInitMode(Maf2_1{}));
    }

    // E8M0 scales are drawn over the exponents round(log2(lo)) .. round(log2(hi)), with lo raised
    // to the smallest scale (2^-127) and both ends clamped to the format.
    TEST(Maf2_1Mode, PowerOfTwoScaleBytes)
    {
        const auto scales = sortedFiniteScales<ocp_e4m3_mxfp8>();
        const auto unit   = maf2_1PowerOfTwoScaleBytes(0.0, 1.0, scales);
        ASSERT_EQ(unit.size(), 128u);
        for(int e = -127; e <= 0; ++e)
            EXPECT_EQ(unit[e + 127], e + 127) << "e=" << e;

        // log2(0.3) = -1.74 -> -2, log2(5) = 2.32 -> 2.
        EXPECT_EQ(maf2_1PowerOfTwoScaleBytes(0.3, 5.0, scales),
                  (std::vector<uint8_t>{125, 126, 127, 128, 129}));
        // Saturates at 2^127 (byte 254).
        const auto wide = maf2_1PowerOfTwoScaleBytes(0.0, 1e300, scales);
        EXPECT_EQ(wide.size(), 255u);
        EXPECT_EQ(wide.back(), 254);
    }

    // FP4 E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 are codes 0..7; ties go to the even code.
    TEST(Maf2_1Mode, Fp4ElementCodeRoundsToNearestEvenAndSaturates)
    {
        const uint8_t one        = scaleOneByte<ocp_e2m1_mxfp4>();
        const auto    magnitudes = sortedNonNegativeCodes<ocp_e2m1_mxfp4>(one, 0x8);
        ASSERT_EQ(magnitudes.size(), 8u);
        auto code = [&](double x) { return maf2_0ElementCode(x, magnitudes, 0x8); };
        EXPECT_EQ(code(0.1), 0x0);
        EXPECT_EQ(code(-0.1), 0x8); // -0
        EXPECT_EQ(code(0.25), 0x0); // tie 0 / 0.5 -> 0
        EXPECT_EQ(code(0.3), 0x1);
        EXPECT_EQ(code(0.75), 0x2); // tie 0.5 / 1 -> 1
        EXPECT_EQ(code(-0.9), 0xA);
        EXPECT_EQ(code(5.0), 0x6); // tie 4 / 6 -> 4
        EXPECT_EQ(code(100.0), 0x7); // saturates at 6
    }

    TEST(Maf2_1Mode, UnscaledTypesThrow)
    {
        EXPECT_THROW(DataGenerator<f32>().generate({16}, {1}, maf2_1Options(1)),
                     std::invalid_argument);
        EXPECT_THROW(DataGenerator<fp16>().generate({16}, {1}, maf2_1Options(1)),
                     std::invalid_argument);
        EXPECT_THROW(DataGenerator<bf16>().generate({16}, {1}, maf2_1Options(1)),
                     std::invalid_argument);
    }

    TEST(Maf2_1Mode, InvalidRangesThrow)
    {
        auto run = [](Maf2_1 mode) {
            DataGeneratorOptions opts;
            opts.initMode     = DataInitMode(mode);
            opts.blockScaling = 32;
            DataGenerator<ocp_e4m3_mxfp8>().generate({32}, {1}, opts);
        };
        EXPECT_THROW(run(Maf2_1{1.0, -1.0}), std::invalid_argument);
        EXPECT_THROW(run(Maf2_1{-1.0, INFINITY}), std::invalid_argument);
        EXPECT_THROW(run(Maf2_1{-1.0, 1.0, -0.5, 1.0}), std::invalid_argument);
        EXPECT_THROW(run(Maf2_1{-1.0, 1.0, 1.0, 1.0}), std::invalid_argument);
    }

    // Leading dimension 7 is not a multiple of rows 6, so the 35-element buffer is not a whole
    // number of scale-block runs; the tail would stay unwritten, so the layout is rejected.
    TEST(Maf2_1Mode, PartialRowLayoutThrows)
    {
        EXPECT_THROW(DataGenerator<ocp_e4m3_mxfp8>().generate({6, 5}, {1, 7}, maf2_1Options(2)),
                     std::invalid_argument);
    }

    TYPED_TEST(Maf2_1Test, DeterministicPerSeed)
    {
        auto gen = [](uint32_t seed) {
            DataGenerator<TypeParam> dgen;
            dgen.setSeed(seed);
            dgen.generate({64, 48}, {1, 64}, maf2_1Options(32));
            return std::make_pair(dgen.getDataBytes(), dgen.getScaleBytes());
        };
        const auto a = gen(1713573849);
        const auto b = gen(1713573849);
        const auto c = gen(1713573850);
        EXPECT_EQ(a.first, b.first);
        EXPECT_EQ(a.second, b.second);
        EXPECT_NE(a.first, c.first);
    }

    // The bytes follow the mxDataGenerator thread layout exactly: T = min(32, max threads),
    // thread t uses mt19937(seed + t) over a static contiguous range of blocks, and each block
    // draws its scale, then its elements in order. Covers tail blocks
    // (40 = 2*16 + 8).
    TYPED_TEST(Maf2_1Test, MatchesSerialThreadLayout)
    {
        const int saved = omp_get_max_threads();
        for(int threads : {1, 7, 32})
        {
            omp_set_num_threads(threads);
            const uint32_t           seed = 1234;
            const index_t            bs   = 16;
            DataGenerator<TypeParam> dgen;
            dgen.setSeed(seed);
            dgen.generate({40, 37}, {1, 40}, maf2_1Options(bs));

            const int      T          = std::min(32, threads);
            const auto     numBlocks  = dgen.numScaleBlocks();
            const uint8_t  one        = scaleOneByte<TypeParam>();
            const uint8_t  signBit    = signBitOf<TypeParam>();
            const auto     magnitudes = sortedNonNegativeCodes<TypeParam>(one, signBit);
            const auto     table      = sortedFiniteScales<TypeParam>();
            const auto     pow2       = maf2_1PowerOfTwoScaleBytes(0.0, 1.0, table);
            constexpr bool isE8M0     = scale_info_t<TypeParam>::mantissaBits == 0;

            std::vector<uint8_t> scales(numBlocks);
            std::vector<uint8_t> data(40 * 37);
            const index_t        q = numBlocks / T, r = numBlocks % T;
            for(int t = 0; t < T; ++t)
            {
                std::mt19937                           g(seed + t);
                std::uniform_real_distribution<double> dataDist(-1.0, 1.0);
                std::uniform_real_distribution<double> scaleDist(0.0, 1.0);
                std::uniform_int_distribution<size_t>  pow2Dist(0, pow2.size() - 1);
                const index_t                          b0 = t * q + std::min<index_t>(t, r);
                const index_t b1 = b0 + q + (static_cast<index_t>(t) < r ? 1 : 0);
                for(index_t s = b0; s < b1; ++s)
                {
                    scales[s]
                        = isE8M0 ? pow2[pow2Dist(g)] : nearestScaleByteRne(table, scaleDist(g));
                    for(index_t i = 0; i < dgen.scaleBlockElementCount(s); ++i)
                        data[dgen.dataIndex(s, i)]
                            = maf2_0ElementCode(dataDist(g), magnitudes, signBit);
                }
            }

            EXPECT_EQ(dgen.getScaleBytes(), scales) << "threads=" << threads;
            const auto ref = dgen.getReferenceDouble();
            ASSERT_EQ(ref.size(), data.size());
            for(index_t i = 0; i < static_cast<index_t>(data.size()); ++i)
            {
                const double expected
                    = decode<TypeParam>(scales[dgen.scaleIndexForData(i)], data[i]);
                ASSERT_EQ(ref[i], expected) << "threads=" << threads << " i=" << i;
                ASSERT_EQ(std::signbit(ref[i]), std::signbit(expected)) << "i=" << i;
            }
            if constexpr(TypeParam::dataInfo.exponentBits + TypeParam::dataInfo.mantissaBits == 7)
            {
                EXPECT_EQ(dgen.getDataBytes(), data) << "threads=" << threads;
            }
        }
        omp_set_num_threads(saved);
    }

    // Each scale byte's probability: for E8M0, 1/128 for each of 2^-127 .. 2^0; otherwise the
    // length of its rounding interval within [0, 1] (intervals end at the midpoints between
    // neighbouring scale values; ties have probability zero).
    TYPED_TEST(Maf2_1Test, ScaleBytesFollowUniform)
    {
        const index_t            blocks = 1 << 17;
        DataGenerator<TypeParam> dgen;
        dgen.generate({blocks}, {1}, maf2_1Options(1));
        const auto scales = dgen.getScaleBytes();
        const auto table  = sortedFiniteScales<TypeParam>();

        std::map<uint8_t, double> expected;
        if constexpr(scale_info_t<TypeParam>::mantissaBits == 0)
        {
            for(auto b : maf2_1PowerOfTwoScaleBytes(0.0, 1.0, table))
                expected[b] += double(blocks) / 128;
        }
        else
        {
            for(size_t k = 0; k < table.size(); ++k)
            {
                const double lo = k == 0 ? 0.0 : 0.5 * (table[k - 1].first + table[k].first);
                const double hi = k + 1 == table.size()
                                      ? INFINITY
                                      : 0.5 * (table[k].first + table[k + 1].first);
                const double p  = overlap(lo, hi, 0.0, 1.0);
                if(p > 0)
                    expected[table[k].second] += p * blocks;
            }
        }

        std::map<uint8_t, double> observed;
        for(auto s : scales)
        {
            ASSERT_TRUE(expected.count(s)) << "unexpected scale byte " << int(s);
            observed[s] += 1;
        }
        // At most 128 pooled bins; the 1e-6 upper quantile for 127 degrees of freedom is < 230.
        EXPECT_LT(pooledChi2(expected, observed), 230.0);
    }

    // Each element value's probability is the length of its rounding interval within [-1, 1],
    // over 2. Codes that decode to the same value share a bin (the FP4 and FP6 decoders return
    // +0 for -0; FP8 keeps -0 apart). The unscaled value is the reference divided by its block
    // scale, which is exact (a few significant bits each); zero-scale blocks are skipped.
    TYPED_TEST(Maf2_1Test, DataValuesFollowRoundedUniform)
    {
        const index_t            elements = 1 << 18;
        DataGenerator<TypeParam> dgen;
        dgen.generate({elements}, {1}, maf2_1Options(32));
        const auto refScaled = dgen.getReferenceDouble();
        const auto scaleB    = dgen.getScaleBytes();

        const uint8_t one        = scaleOneByte<TypeParam>();
        uint8_t       oneCode[1] = {0}, oneScale[1] = {0};
        setOne<TypeParam>(oneScale, oneCode, 0, 0, false);
        std::vector<double> ref;
        for(index_t i = 0; i < elements; ++i)
        {
            const double scale = decode<TypeParam>(scaleB[dgen.scaleIndexForData(i)], oneCode[0]);
            if(scale != 0)
                ref.push_back(refScaled[i] / scale);
        }
        const double n = static_cast<double>(ref.size());
        ASSERT_GT(n, 0.99 * elements);

        const uint8_t signBit    = signBitOf<TypeParam>();
        const auto    magnitudes = sortedNonNegativeCodes<TypeParam>(one, signBit);
        using Key                = std::pair<double, bool>; // value, sign bit
        std::map<Key, double> expected;
        for(size_t k = 0; k < magnitudes.size(); ++k)
        {
            const double m  = magnitudes[k].first;
            const double lo = k == 0 ? 0.0 : 0.5 * (magnitudes[k - 1].first + m);
            const double hi
                = k + 1 == magnitudes.size() ? INFINITY : 0.5 * (m + magnitudes[k + 1].first);
            const double p = overlap(lo, hi, 0.0, 1.0) / 2 * n;
            if(p == 0)
                continue;
            const double pos = decode<TypeParam>(one, magnitudes[k].second);
            const double neg
                = decode<TypeParam>(one, static_cast<uint8_t>(magnitudes[k].second | signBit));
            expected[{pos, static_cast<bool>(std::signbit(pos))}] += p;
            expected[{neg, static_cast<bool>(std::signbit(neg))}] += p;
        }

        std::map<Key, double> observed;
        for(double v : ref)
        {
            const Key key{v, static_cast<bool>(std::signbit(v))};
            ASSERT_TRUE(expected.count(key)) << "unexpected value " << v;
            observed[key] += 1;
        }
        // At most ~130 pooled bins (E5M2); the 1e-6 upper quantile for 130 dof is < 235.
        EXPECT_LT(pooledChi2(expected, observed), 235.0) << "bins=" << expected.size();
        for(double v : ref)
            ASSERT_LE(std::abs(v), 1.0);
    }

    TYPED_TEST(Maf2_1Test, ReferenceFloatMatchesBytes)
    {
        DataGenerator<TypeParam> dgen;
        dgen.generate({128, 16}, {1, 128}, maf2_1Options(32));
        const auto refD = dgen.getReferenceDouble();
        const auto refF = dgen.getReferenceFloat();
        ASSERT_EQ(refD.size(), refF.size());
        for(size_t i = 0; i < refD.size(); ++i)
            EXPECT_EQ(refF[i], static_cast<float>(refD[i])) << "i=" << i;
    }

    TYPED_TEST(Maf2_1Test, IgnoresRangeAndSpecialValueOptions)
    {
        auto run = [](DataGeneratorOptions opts) {
            DataGenerator<TypeParam> dgen;
            dgen.generate({64, 8}, {1, 64}, opts);
            return std::make_pair(dgen.getDataBytes(), dgen.getScaleBytes());
        };
        const auto base    = run(maf2_1Options(32));
        auto       opts    = maf2_1Options(32);
        opts.min           = -0.5;
        opts.max           = 0.5;
        opts.includeNaN    = true;
        opts.includeInf    = true;
        opts.forceDenorm   = true;
        opts.clampToF32    = true;
        opts.scaleInitMode = Ones{};
        EXPECT_EQ(run(opts), base);
    }
} // namespace
