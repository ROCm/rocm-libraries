// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "harness/input-init/PhiloxDropout.hpp"

using namespace hipdnn_integration_tests::philox;

// The three Philox4x32-10 known-answer vectors of Random123 (kat_vectors): they pin the cipher.
TEST(TestPhiloxDropout, Philox4x32MatchesTheRandom123KnownAnswers)
{
    EXPECT_EQ(philox4x32({0U, 0U, 0U, 0U}, 0U, 0U),
              (Words{0x6627e8d5U, 0xe169c58dU, 0xbc57ac4cU, 0x9b00dbd8U}));
    EXPECT_EQ(
        philox4x32({0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU}, 0xffffffffU, 0xffffffffU),
        (Words{0x408f276dU, 0x41c83b0eU, 0xa20bc7c6U, 0x6d5451fdU}));
    EXPECT_EQ(
        philox4x32({0x243f6a88U, 0x85a308d3U, 0x13198a2eU, 0x03707344U}, 0xa4093822U, 0x299f31d0U),
        (Words{0xd16cfe09U, 0x94fdccebU, 0x5001e420U, 0x24126ea1U}));
}

// The seed's low word is key 0 and its high word key 1; the counter's low word is c0 and its high
// word c1: swapping either changes the stream.
TEST(TestPhiloxDropout, TheSeedAndTheCounterAreSplitLowWordFirst)
{
    const uint64_t seed = 0x0123456789ABCDEFULL;
    const uint64_t counter = 0x00000007000000AAULL;
    EXPECT_EQ(randomsAt(seed, counter), philox4x32({0xAAU, 7U, 0U, 0U}, 0x89ABCDEFU, 0x01234567U));
    EXPECT_NE(randomsAt(seed, counter), randomsAt(0xCDEF012389AB4567ULL, counter));
    EXPECT_NE(randomsAt(seed, counter), randomsAt(seed, 0x000000AA00000007ULL));
}

TEST(TestPhiloxDropout, ThresholdIsTheScaledProbabilityClampedToInt32)
{
    EXPECT_EQ(dropoutThreshold(0.5F), 0);
    EXPECT_EQ(dropoutThreshold(0.25F), -1073741824);
    EXPECT_EQ(dropoutThreshold(0.75F), 1073741824);
    EXPECT_EQ(dropoutThreshold(0.0F), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(dropoutThreshold(1.0F), std::numeric_limits<int32_t>::max());
    EXPECT_FLOAT_EQ(dropoutScale(0.5F), 2.0F);
}

// The mask's layout: counter = offset + ((b * Hq + h) * Sq + i) * ceil(Skv / 4) + j / 4, so the
// four keys of a group share one counter and shifting the offset by a row stride shifts the rows.
TEST(TestPhiloxDropout, TheMaskFollowsTheCounterLayout)
{
    const MaskExtent extent{2, 3, 5, 10}; // Skv = 10: three counters per row, the last half used
    const uint64_t seed = 0xFEEDFACE12345678ULL;
    const float p = 0.3F;
    std::vector<float> mask(static_cast<size_t>(extent.elements()));
    fillKeepMask(mask.data(), seed, 1000U, extent, p);

    const int32_t threshold = dropoutThreshold(p);
    size_t at = 0;
    for(int64_t b = 0; b < extent.batch; ++b)
    {
        for(int64_t h = 0; h < extent.heads; ++h)
        {
            for(int64_t i = 0; i < extent.seqQ; ++i)
            {
                for(int64_t j = 0; j < extent.seqKv; ++j, ++at)
                {
                    ASSERT_EQ(mask[at] == 1.0F, keeps(seed, 1000U, extent, b, h, i, j, threshold))
                        << b << "," << h << "," << i << "," << j;
                    ASSERT_TRUE(mask[at] == 0.0F || mask[at] == 1.0F);
                }
            }
        }
    }

    // A counter step of ceil(Skv / 4) = 3 moves every row down by one.
    std::vector<float> shifted(mask.size());
    fillKeepMask(shifted.data(), seed, 1003U, extent, p);
    for(int64_t plane = 0; plane < extent.batch * extent.heads; ++plane)
    {
        for(int64_t i = 0; i + 1 < extent.seqQ; ++i)
        {
            for(int64_t j = 0; j < extent.seqKv; ++j)
            {
                const auto base = static_cast<size_t>((plane * extent.seqQ + i) * extent.seqKv + j);
                ASSERT_EQ(shifted[base], mask[base + static_cast<size_t>(extent.seqKv)])
                    << plane << "," << i << "," << j;
            }
        }
    }
}

TEST(TestPhiloxDropout, TheKeepRateIsOneMinusTheProbability)
{
    const MaskExtent extent{2, 4, 64, 129};
    std::vector<float> mask(static_cast<size_t>(extent.elements()));
    for(const float p : {0.1F, 0.5F, 0.9F})
    {
        fillKeepMask(mask.data(), 42U, 0U, extent, p);
        double kept = 0;
        for(const float value : mask)
        {
            kept += static_cast<double>(value);
        }
        const auto n = static_cast<double>(mask.size());
        EXPECT_NEAR(kept / n, 1.0 - static_cast<double>(p), 6.0 * std::sqrt(0.25 / n))
            << static_cast<double>(p);
    }
}

TEST(TestPhiloxDropout, TheMaskDependsOnTheSeedAndTheOffset)
{
    const MaskExtent extent{1, 2, 16, 33};
    std::vector<float> a(static_cast<size_t>(extent.elements()));
    std::vector<float> b(a.size());
    std::vector<float> c(a.size());
    fillKeepMask(a.data(), 1U, 0U, extent, 0.5F);
    fillKeepMask(b.data(), 2U, 0U, extent, 0.5F);
    fillKeepMask(c.data(), 1U, 1000U, extent, 0.5F);
    EXPECT_NE(a, b);
    EXPECT_NE(a, c);
    std::vector<float> again(a.size());
    fillKeepMask(again.data(), 1U, 0U, extent, 0.5F);
    EXPECT_EQ(a, again);
}

// The plane product is 64-bit: an offset past 2^32 carries into the counter's high word.
TEST(TestPhiloxDropout, ACounterPast32BitsCarriesIntoTheHighWord)
{
    const MaskExtent extent{1, 1, 1, 4};
    const uint64_t offset = 0xFFFFFFFFULL; // counter 2^32 - 1 then 2^32 for the next row's group
    const int32_t threshold = dropoutThreshold(0.5F);
    EXPECT_EQ(keeps(7U, offset, extent, 0, 0, 0, 2, threshold),
              static_cast<int32_t>(philox4x32({0xFFFFFFFFU, 0U, 0U, 0U}, 7U, 0U)[2]) > threshold);
    EXPECT_EQ(keeps(7U, offset + 1, extent, 0, 0, 0, 2, threshold),
              static_cast<int32_t>(philox4x32({0U, 1U, 0U, 0U}, 7U, 0U)[2]) > threshold);
}
