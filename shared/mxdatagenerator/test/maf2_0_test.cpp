// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Maf2_0: block scale 2^(Binomial(2n+1, 0.5) - (n+1) + center), data uniform over the
// finite encodings, drawn per block in the mxDataGenerator thread layout.

#include <mxDataGenerator/DataGenerator.hpp>

#include <gtest/gtest.h>

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <vector>

namespace
{
    using namespace DGen;

    template <typename DT>
    class Maf2_0Test : public ::testing::Test
    {
    };

    using Maf2_0Types = ::testing::Types<ocp_e4m3_mxfp8,
                                         ocp_e5m2_mxfp8,
                                         ocp_e2m3_mxfp6,
                                         ocp_e3m2_mxfp6,
                                         ocp_e2m1_mxfp4,
                                         ocp_e2m1_mxfp4_e4m3,
                                         ocp_e2m1_mxfp4_e5m3>;
    TYPED_TEST_SUITE(Maf2_0Test, Maf2_0Types);

    DataGeneratorOptions maf2_0Options(index_t blockSize, int n = 10, int center = 0)
    {
        DataGeneratorOptions opts;
        opts.initMode     = DataInitMode(Maf2_0{n, center});
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

    // Finite data encodings of DT, in ascending byte order.
    template <typename DT>
    std::vector<uint8_t> finiteCodes()
    {
        const uint8_t one = scaleOneByte<DT>();
        const int     bits
            = DT::dataInfo.signBits + DT::dataInfo.exponentBits + DT::dataInfo.mantissaBits;
        std::vector<uint8_t> codes;
        for(int c = 0; c < (1 << bits); ++c)
        {
            const uint8_t data = static_cast<uint8_t>(c);
            if(std::isfinite(toDouble<DT>(&one, &data, 0, 0)))
                codes.push_back(data);
        }
        return codes;
    }

    template <typename DT>
    double decode(uint8_t scale, uint8_t code)
    {
        return toDouble<DT>(&scale, &code, 0, 0);
    }

    TEST(Maf2_0Mode, ToStringAndEquality)
    {
        EXPECT_EQ(toString(DataInitMode(Maf2_0{})), "DataInitMode(Maf2_0(10, 0))");
        EXPECT_TRUE(dataInitModesEqual(Maf2_0{10, 0}, Maf2_0{}));
        EXPECT_FALSE(dataInitModesEqual(Maf2_0{10, 0}, Maf2_0{3, 2}));
        EXPECT_FALSE(isRandomLikeDataInitMode(Maf2_0{}));
    }

    TEST(Maf2_0Mode, CodeTableSizes)
    {
        EXPECT_EQ(finiteCodes<ocp_e4m3_mxfp8>().size(), 254u); // all but 0x7F, 0xFF
        EXPECT_EQ(finiteCodes<ocp_e5m2_mxfp8>().size(), 248u); // all but exponent all-ones
        EXPECT_EQ(finiteCodes<ocp_e2m3_mxfp6>().size(), 64u);
        EXPECT_EQ(finiteCodes<ocp_e3m2_mxfp6>().size(), 64u);
        EXPECT_EQ(finiteCodes<ocp_e2m1_mxfp4>().size(), 16u);
    }

    TEST(Maf2_0Mode, BinomialHalfMatchesPopcount)
    {
        std::mt19937 a(7), b(7);
        for(int trials : {0, 1, 21, 31, 32, 33, 64, 65})
        {
            for(int rep = 0; rep < 100; ++rep)
            {
                int expected = 0;
                int t        = trials;
                for(; t >= 32; t -= 32)
                    expected += __builtin_popcount(static_cast<uint32_t>(b()));
                if(t > 0)
                    expected += __builtin_popcount(static_cast<uint32_t>(b()) & ((1u << t) - 1u));
                const int k = binomialHalf(a, trials);
                EXPECT_EQ(k, expected);
                EXPECT_GE(k, 0);
                EXPECT_LE(k, trials);
            }
        }
    }

    TEST(Maf2_0Mode, E8M0ScaleTableIsExactPowersOfTwo)
    {
        const auto table = maf2_0ScaleBytes<ocp_e4m3_mxfp8>(10, 0);
        ASSERT_EQ(table.size(), 22u);
        for(int k = 0; k <= 21; ++k)
            EXPECT_EQ(table[k], 127 + k - 11) << "k=" << k;

        // Saturation at the top of E8M0 (2^127, byte 254).
        const auto high = maf2_0ScaleBytes<ocp_e4m3_mxfp8>(10, 130);
        EXPECT_EQ(high.back(), 254);
    }

    TEST(Maf2_0Mode, E4M3ScaleTableSaturatesAndUnderflowsLikeRne)
    {
        // 2^e for e = k - 11. E4M3: min subnormal 2^-9 (0x01), max 448 (0x7E).
        const auto table = maf2_0ScaleBytes<ocp_e2m1_mxfp4_e4m3>(10, 0);
        ASSERT_EQ(table.size(), 22u);
        EXPECT_EQ(table[0], 0x00); // 2^-11 -> 0
        EXPECT_EQ(table[1], 0x00); // 2^-10, a tie between 0 and 2^-9 -> even (0)
        EXPECT_EQ(table[2], 0x01); // 2^-9
        EXPECT_EQ(table[3], 0x02); // 2^-8
        EXPECT_EQ(table[4], 0x04); // 2^-7
        for(int k = 5; k <= 19; ++k) // 2^-6 .. 2^8: normal, mantissa 0
            EXPECT_EQ(table[k], static_cast<uint8_t>((k - 11 + 7) << 3)) << "k=" << k;
        EXPECT_EQ(table[20], 0x7E); // 2^9 -> 448
        EXPECT_EQ(table[21], 0x7E); // 2^10 -> 448
        for(int k = 0; k <= 21; ++k)
            EXPECT_FALSE(isScaleNaN<ScaleInfo<ScaleType::E4M3>>(table[k]));
    }

    TEST(Maf2_0Mode, UnscaledTypesThrow)
    {
        EXPECT_THROW(DataGenerator<f32>().generate({16}, {1}, maf2_0Options(1)),
                     std::invalid_argument);
        EXPECT_THROW(DataGenerator<fp16>().generate({16}, {1}, maf2_0Options(1)),
                     std::invalid_argument);
        EXPECT_THROW(DataGenerator<bf16>().generate({16}, {1}, maf2_0Options(1)),
                     std::invalid_argument);
    }

    TEST(Maf2_0Mode, NegativeNThrows)
    {
        EXPECT_THROW(DataGenerator<ocp_e4m3_mxfp8>().generate({32}, {1}, maf2_0Options(32, -1)),
                     std::invalid_argument);
    }

    TYPED_TEST(Maf2_0Test, DeterministicPerSeed)
    {
        auto gen = [](uint32_t seed) {
            DataGenerator<TypeParam> dgen;
            dgen.setSeed(seed);
            dgen.generate({64, 48}, {1, 64}, maf2_0Options(32));
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
    // draws its scale, then its elements in order. Covers tail blocks (40 = 2*16 + 8).
    TYPED_TEST(Maf2_0Test, MatchesSerialThreadLayout)
    {
        const int saved = omp_get_max_threads();
        for(int threads : {1, 7, 32})
        {
            omp_set_num_threads(threads);
            const uint32_t           seed = 1234;
            const index_t            bs   = 16;
            DataGenerator<TypeParam> dgen;
            dgen.setSeed(seed);
            dgen.generate({40, 37}, {1, 40}, maf2_0Options(bs));

            const int                             T         = std::min(32, threads);
            const auto                            numBlocks = dgen.numScaleBlocks();
            const auto                            table     = maf2_0ScaleBytes<TypeParam>(10, 0);
            const auto                            codes     = finiteCodes<TypeParam>();
            std::uniform_int_distribution<size_t> codeDist(0, codes.size() - 1);

            std::vector<uint8_t> scales(numBlocks);
            std::vector<uint8_t> data(40 * 37);
            const index_t        q = numBlocks / T, r = numBlocks % T;
            for(int t = 0; t < T; ++t)
            {
                std::mt19937  g(seed + t);
                const index_t b0 = t * q + std::min<index_t>(t, r);
                const index_t b1 = b0 + q + (static_cast<index_t>(t) < r ? 1 : 0);
                for(index_t s = b0; s < b1; ++s)
                {
                    scales[s] = table[binomialHalf(g, 21)];
                    for(index_t i = 0; i < dgen.scaleBlockElementCount(s); ++i)
                        data[dgen.dataIndex(s, i)] = codes[codeDist(g)];
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

    TYPED_TEST(Maf2_0Test, ScaleExponentsFollowBinomial)
    {
        // 2^16 blocks of one element each; compare the drawn k against Binomial(21, 0.5).
        const index_t            blocks = 1 << 16;
        DataGenerator<TypeParam> dgen;
        dgen.generate({blocks}, {1}, maf2_0Options(1));
        const auto scales = dgen.getScaleBytes();
        const auto table  = maf2_0ScaleBytes<TypeParam>(10, 0);

        // Map each scale byte back to its k (bytes shared by several k, e.g. E4M3 saturation,
        // are pooled into one bin).
        std::map<uint8_t, double> expected;
        for(int k = 0; k <= 21; ++k)
            expected[table[k]]
                += std::ldexp(std::tgamma(22.0) / (std::tgamma(k + 1.0) * std::tgamma(22.0 - k)),
                              -21)
                   * blocks;
        std::map<uint8_t, double> observed;
        for(auto s : scales)
        {
            ASSERT_TRUE(expected.count(s)) << "unexpected scale byte " << int(s);
            observed[s] += 1;
        }

        // Pool bins with expected count < 5 from both tails into their neighbours.
        double chi2 = 0, eAcc = 0, oAcc = 0;
        int    dof = -1;
        for(auto& [byte, e] : expected)
        {
            eAcc += e;
            oAcc += observed[byte];
            if(eAcc >= 5)
            {
                chi2 += (oAcc - eAcc) * (oAcc - eAcc) / eAcc;
                ++dof;
                eAcc = oAcc = 0;
            }
        }
        // 99.9999th percentile of chi-square with <= 21 degrees of freedom is below 75.
        EXPECT_LT(chi2, 75.0) << "dof=" << dof;
    }

    TYPED_TEST(Maf2_0Test, DataCodesAreUniformOverFiniteEncodings)
    {
        DataGenerator<TypeParam> dgen;
        dgen.generate({1 << 18}, {1}, maf2_0Options(32));
        const auto ref    = dgen.getReferenceDouble();
        const auto scales = dgen.getScaleBytes();
        const auto codes  = finiteCodes<TypeParam>();

        // Bin elements by their unit-scale value; codes that decode to the same value (the FP4
        // and FP6 decoders return +0 for -0) share a bin. Zero-scale blocks carry no code.
        const uint8_t                             one = scaleOneByte<TypeParam>();
        std::map<std::pair<double, bool>, double> expectedShare, observed;
        for(auto c : codes)
        {
            const double v = decode<TypeParam>(one, c);
            expectedShare[{v, std::signbit(v)}] += 1.0 / codes.size();
        }

        double total = 0;
        for(index_t i = 0; i < static_cast<index_t>(ref.size()); ++i)
        {
            ASSERT_TRUE(std::isfinite(ref[i])) << "i=" << i;
            const double v
                = getScaleValue<scale_info_t<TypeParam>>(scales[dgen.scaleIndexForData(i)]);
            if(v == 0.0)
                continue;
            const double unit = ref[i] / v;
            const auto   key  = std::make_pair(unit, static_cast<bool>(std::signbit(unit)));
            ASSERT_TRUE(expectedShare.count(key)) << "i=" << i << " value " << unit;
            observed[key] += 1;
            total += 1;
        }

        double chi2 = 0;
        for(auto& [key, share] : expectedShare)
        {
            const double e = share * total;
            chi2 += (observed[key] - e) * (observed[key] - e) / e;
            EXPECT_GT(observed[key], 0.0) << "value " << key.first << " never drawn";
        }
        // dof < 254; mean ~253, sd ~22.5; a 1e-6 bound is below 400.
        EXPECT_LT(chi2, 400.0) << "bins=" << expectedShare.size();
    }

    TYPED_TEST(Maf2_0Test, ReferenceFloatMatchesBytes)
    {
        DataGenerator<TypeParam> dgen;
        dgen.generate({128, 16}, {1, 128}, maf2_0Options(32));
        const auto refD = dgen.getReferenceDouble();
        const auto refF = dgen.getReferenceFloat();
        ASSERT_EQ(refD.size(), refF.size());
        for(size_t i = 0; i < refD.size(); ++i)
            EXPECT_EQ(refF[i], static_cast<float>(refD[i])) << "i=" << i;
    }

    TYPED_TEST(Maf2_0Test, IgnoresRangeAndSpecialValueOptions)
    {
        auto run = [](DataGeneratorOptions opts) {
            DataGenerator<TypeParam> dgen;
            dgen.generate({64, 8}, {1, 64}, opts);
            return std::make_pair(dgen.getDataBytes(), dgen.getScaleBytes());
        };
        const auto base    = run(maf2_0Options(32));
        auto       opts    = maf2_0Options(32);
        opts.min           = -0.5;
        opts.max           = 0.5;
        opts.includeNaN    = true;
        opts.includeInf    = true;
        opts.forceDenorm   = true;
        opts.clampToF32    = true;
        opts.scaleInitMode = Ones{};
        EXPECT_EQ(run(opts), base);
    }

    TEST(Maf2_0Mode, CenterShiftsScaleExponent)
    {
        DataGenerator<ocp_e4m3_mxfp8> dgen;
        dgen.generate({1 << 16}, {1}, maf2_0Options(1, 3, 2));
        const auto scales = dgen.getScaleBytes();
        double     sum    = 0;
        for(auto s : scales)
        {
            const int e = int(s) - 127;
            EXPECT_GE(e, -4 + 2); // k - (n+1) + center, k in [0, 7]
            EXPECT_LE(e, 3 + 2);
            sum += e;
        }
        EXPECT_NEAR(sum / scales.size(), 2 - 0.5, 0.02);
    }

    // ---- normal block scales (the FP4 rows): |N(0.34375, 0.08)| rounded once to the scale ----

    DataGeneratorOptions maf2_0NormalOptions(index_t blockSize)
    {
        DataGeneratorOptions opts;
        Maf2_0               mode;
        mode.normalScale  = true;
        opts.initMode     = DataInitMode(mode);
        opts.blockScaling = blockSize;
        return opts;
    }

    template <typename DT>
    class Maf2_0NormalScaleTest : public ::testing::Test
    {
    };

    using Maf2_0Fp4Types
        = ::testing::Types<ocp_e2m1_mxfp4, ocp_e2m1_mxfp4_e4m3, ocp_e2m1_mxfp4_e5m3>;
    TYPED_TEST_SUITE(Maf2_0NormalScaleTest, Maf2_0Fp4Types);

    TEST(Maf2_0Mode, NormalScaleToString)
    {
        Maf2_0 mode;
        mode.normalScale = true;
        EXPECT_EQ(mode.toString(), "Maf2_0(normal 0.343750, 0.080000)");
        EXPECT_FALSE(dataInitModesEqual(mode, Maf2_0{}));
    }

    // mblas-bench float_to_ue8m0: |x|, nearest power of two with the midpoint at 1.5 * 2^e,
    // ties to the even biased exponent, saturate at 0xFE, zero and float subnormals -> 0x00.
    TEST(Maf2_0Mode, NormalScaleE8M0MatchesMblasConversion)
    {
        const auto scales = sortedFiniteScales<ocp_e2m1_mxfp4>();
        auto byte = [&](double x) { return maf2_0NormalScaleByte<ocp_e2m1_mxfp4>(x, scales); };
        EXPECT_EQ(byte(0.25), 125);
        EXPECT_EQ(byte(0.3), 125);
        EXPECT_EQ(byte(0.37), 125);
        EXPECT_EQ(byte(0.38), 126);
        EXPECT_EQ(byte(0.375), 126); // 1.5 * 2^-2: biased 125 is odd -> up to 126
        EXPECT_EQ(byte(0.75), 126); // 1.5 * 2^-1: biased 126 is even -> stays
        EXPECT_EQ(byte(-0.3), 125);
        EXPECT_EQ(byte(0.0), 0x00);
        EXPECT_EQ(byte(1e-39), 0x00);
        EXPECT_EQ(byte(1e300), 0xFE);
    }

    TEST(Maf2_0Mode, NormalScaleE4M3AndE5M3RoundToNearestEven)
    {
        const auto s4 = sortedFiniteScales<ocp_e2m1_mxfp4_e4m3>();
        const auto s5 = sortedFiniteScales<ocp_e2m1_mxfp4_e5m3>();
        auto       b4 = [&](double x) { return maf2_0NormalScaleByte<ocp_e2m1_mxfp4_e4m3>(x, s4); };
        auto       b5 = [&](double x) { return maf2_0NormalScaleByte<ocp_e2m1_mxfp4_e5m3>(x, s5); };
        EXPECT_EQ(b4(0.34375), 0x2B); // 1.375 * 2^-2, exact
        EXPECT_EQ(b4(-0.34375), 0x2B);
        EXPECT_EQ(b4(0.359375), 0x2C); // tie between 0x2B and 0x2C -> even (0x2C)
        EXPECT_EQ(b4(0.328125), 0x2A); // tie between 0x2A and 0x2B -> even (0x2A)
        EXPECT_EQ(b4(1000.0), 0x7E); // saturates at 448
        EXPECT_EQ(b4(0.0), 0x00);
        EXPECT_EQ(b5(0.34375), 0x6B); // E5M3 bias 15: exponent 13, mantissa 3
        EXPECT_EQ(b5(-0.34375), 0x6B);
        EXPECT_EQ(b5(0.0), 0x00);
    }

    TYPED_TEST(Maf2_0NormalScaleTest, MatchesSerialThreadLayout)
    {
        const int saved = omp_get_max_threads();
        for(int threads : {1, 7, 32})
        {
            omp_set_num_threads(threads);
            const uint32_t           seed = 99;
            DataGenerator<TypeParam> dgen;
            dgen.setSeed(seed);
            dgen.generate({40, 37}, {1, 40}, maf2_0NormalOptions(16));

            const int                             T         = std::min(32, threads);
            const auto                            numBlocks = dgen.numScaleBlocks();
            const auto                            table     = sortedFiniteScales<TypeParam>();
            const auto                            codes     = finiteCodes<TypeParam>();
            std::uniform_int_distribution<size_t> codeDist(0, codes.size() - 1);

            std::vector<uint8_t> scales(numBlocks);
            std::vector<uint8_t> data(40 * 37);
            const index_t        q = numBlocks / T, r = numBlocks % T;
            for(int t = 0; t < T; ++t)
            {
                std::mt19937                     g(seed + t);
                std::normal_distribution<double> nd(0.34375, 0.08);
                const index_t                    b0 = t * q + std::min<index_t>(t, r);
                const index_t b1 = b0 + q + (static_cast<index_t>(t) < r ? 1 : 0);
                for(index_t s = b0; s < b1; ++s)
                {
                    scales[s] = maf2_0NormalScaleByte<TypeParam>(nd(g), table);
                    for(index_t i = 0; i < dgen.scaleBlockElementCount(s); ++i)
                        data[dgen.dataIndex(s, i)] = codes[codeDist(g)];
                }
            }

            EXPECT_EQ(dgen.getScaleBytes(), scales) << "threads=" << threads;
            const auto ref = dgen.getReferenceDouble();
            for(index_t i = 0; i < static_cast<index_t>(data.size()); ++i)
                ASSERT_EQ(ref[i], decode<TypeParam>(scales[dgen.scaleIndexForData(i)], data[i]))
                    << "threads=" << threads << " i=" << i;
        }
        omp_set_num_threads(saved);
    }

    // Each scale byte's probability is P(|x| in its rounding interval), x ~ N(0.34375, 0.08).
    // The intervals end at the arithmetic midpoints between neighbouring scale values, which is
    // also where E8M0 rounds (1.5 * 2^e); ties have probability zero.
    TYPED_TEST(Maf2_0NormalScaleTest, ScaleBytesFollowRoundedNormal)
    {
        const index_t            blocks = 1 << 17;
        DataGenerator<TypeParam> dgen;
        dgen.generate({blocks}, {1}, maf2_0NormalOptions(1));
        const auto scales = dgen.getScaleBytes();
        const auto table  = sortedFiniteScales<TypeParam>();

        auto cdf
            = [](double v) { return 0.5 * std::erfc(-(v - 0.34375) / (0.08 * std::sqrt(2.0))); };
        auto pAbsIn = [&](double lo, double hi) { // P(lo <= |x| < hi) for lo >= 0
            return (cdf(hi) - cdf(lo)) + (cdf(-lo) - cdf(-hi));
        };
        std::map<uint8_t, double> expected;
        for(size_t k = 0; k < table.size(); ++k)
        {
            const double lo = k == 0 ? 0.0 : 0.5 * (table[k - 1].first + table[k].first);
            const double hi
                = k + 1 == table.size() ? INFINITY : 0.5 * (table[k].first + table[k + 1].first);
            expected[table[k].second] += pAbsIn(lo, hi) * blocks;
        }

        std::map<uint8_t, double> observed;
        for(auto s : scales)
        {
            ASSERT_TRUE(expected.count(s)) << "unexpected scale byte " << int(s);
            observed[s] += 1;
        }
        double chi2 = 0, eAcc = 0, oAcc = 0;
        int    bins = 0;
        for(auto& [byte, e] : expected)
        {
            eAcc += e;
            oAcc += observed[byte];
            if(eAcc >= 5)
            {
                chi2 += (oAcc - eAcc) * (oAcc - eAcc) / eAcc;
                ++bins;
                eAcc = oAcc = 0;
            }
        }
        // At most ~60 pooled bins; the 1e-6 upper quantile for 60 degrees of freedom is < 130.
        EXPECT_LT(chi2, 130.0) << "bins=" << bins;
    }

    // ---- unit scales with N(0, 1) data (the FP4 non-scaled row) ----

    DataGeneratorOptions maf2_0UnitScaleOptions(index_t blockSize)
    {
        DataGeneratorOptions opts;
        Maf2_0               mode;
        mode.unitScaleNormalData = true;
        opts.initMode            = DataInitMode(mode);
        opts.blockScaling        = blockSize;
        return opts;
    }

    template <typename DT>
    class Maf2_0UnitScaleTest : public ::testing::Test
    {
    };

    TYPED_TEST_SUITE(Maf2_0UnitScaleTest, Maf2_0Fp4Types);

    TEST(Maf2_0Mode, UnitScaleToString)
    {
        Maf2_0 mode;
        mode.unitScaleNormalData = true;
        EXPECT_EQ(mode.toString(), "Maf2_0(unit scale, N(0, 1))");
        EXPECT_FALSE(dataInitModesEqual(mode, Maf2_0{}));
    }

    // FP4 E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 are codes 0..7; ties go to the even code.
    TEST(Maf2_0Mode, UnitScaleFp4RoundsToNearestEvenAndSaturates)
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
        EXPECT_EQ(code(2.5), 0x4); // tie 2 / 3 -> 2
        EXPECT_EQ(code(5.0), 0x6); // tie 4 / 6 -> 4
        EXPECT_EQ(code(5.1), 0x7);
        EXPECT_EQ(code(100.0), 0x7); // saturates at 6
        EXPECT_EQ(code(-100.0), 0xF);
    }

    TYPED_TEST(Maf2_0UnitScaleTest, MatchesSerialThreadLayout)
    {
        const int saved = omp_get_max_threads();
        for(int threads : {1, 7, 32})
        {
            omp_set_num_threads(threads);
            const uint32_t           seed = 7;
            DataGenerator<TypeParam> dgen;
            dgen.setSeed(seed);
            dgen.generate({40, 37}, {1, 40}, maf2_0UnitScaleOptions(16));

            const uint8_t one        = scaleOneByte<TypeParam>();
            const auto    magnitudes = sortedNonNegativeCodes<TypeParam>(one, 0x8);
            const int     T          = std::min(32, threads);
            const auto    numBlocks  = dgen.numScaleBlocks();

            std::vector<uint8_t> data(40 * 37);
            const index_t        q = numBlocks / T, r = numBlocks % T;
            for(int t = 0; t < T; ++t)
            {
                std::mt19937                     g(seed + t);
                std::normal_distribution<double> nd(0.0, 1.0);
                const index_t                    b0 = t * q + std::min<index_t>(t, r);
                const index_t b1 = b0 + q + (static_cast<index_t>(t) < r ? 1 : 0);
                for(index_t s = b0; s < b1; ++s)
                    for(index_t i = 0; i < dgen.scaleBlockElementCount(s); ++i)
                        data[dgen.dataIndex(s, i)] = maf2_0ElementCode(nd(g), magnitudes, 0x8);
            }

            for(auto s : dgen.getScaleBytes())
                ASSERT_EQ(s, one) << "threads=" << threads;
            const auto ref = dgen.getReferenceDouble();
            for(index_t i = 0; i < static_cast<index_t>(data.size()); ++i)
                ASSERT_EQ(ref[i], decode<TypeParam>(one, data[i]))
                    << "threads=" << threads << " i=" << i;
        }
        omp_set_num_threads(saved);
    }

    // Each FP4 value's probability is P(x in its rounding interval) for x ~ N(0, 1).
    TYPED_TEST(Maf2_0UnitScaleTest, ValuesFollowRoundedNormal)
    {
        const index_t            n = 1 << 18;
        DataGenerator<TypeParam> dgen;
        dgen.generate({n}, {1}, maf2_0UnitScaleOptions(16));
        const auto ref = dgen.getReferenceDouble();

        const std::vector<double> mags = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
        auto cdf = [](double v) { return 0.5 * std::erfc(-v / std::sqrt(2.0)); };
        std::map<double, double> expected; // signed value -> expected count (0 pools both signs)
        for(size_t k = 0; k < mags.size(); ++k)
        {
            const double lo = k == 0 ? 0.0 : 0.5 * (mags[k - 1] + mags[k]);
            const double hi = k + 1 == mags.size() ? INFINITY : 0.5 * (mags[k] + mags[k + 1]);
            const double p  = cdf(hi) - cdf(lo);
            if(mags[k] == 0)
                expected[0.0] += 2 * p * n;
            else
            {
                expected[mags[k]] += p * n;
                expected[-mags[k]] += p * n;
            }
        }
        std::map<double, double> observed;
        for(double v : ref)
        {
            ASSERT_TRUE(expected.count(v)) << "unexpected value " << v;
            observed[v] += 1;
        }
        double chi2 = 0, eAcc = 0, oAcc = 0;
        for(auto& [v, e] : expected)
        {
            eAcc += e;
            oAcc += observed[v];
            if(eAcc >= 5)
            {
                chi2 += (oAcc - eAcc) * (oAcc - eAcc) / eAcc;
                eAcc = oAcc = 0;
            }
        }
        // At most 14 pooled bins; the 1e-6 upper quantile for 14 degrees of freedom is < 60.
        EXPECT_LT(chi2, 60.0);
    }
} // namespace
