// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// SliceAlignment::split and hints: the rule that rounds the share of a split dimension so that
// every slice starts on 256 bytes. Pure arithmetic, no GPU.

#include <Tensile/SliceAlignment.hpp>

#include <gtest/gtest.h>

#include <string>

using namespace TensileLite::SliceAlignment;

namespace
{
    const Matrix fp16D{"D", 1, 16}; // M split: D's stride along M is 1
}

TEST(SliceAlignment, AlignedHalfIsKept)
{
    auto r = split(4096, 2, {fp16D}, Mode::Auto);
    EXPECT_EQ(r.share, 2048);
    EXPECT_TRUE(r.misaligned.empty());
}

TEST(SliceAlignment, SmallMAutoStaysPlainAndReportsD)
{
    auto r = split(200, 2, {fp16D}, Mode::Auto);
    EXPECT_EQ(r.share, 100);
    ASSERT_EQ(r.misaligned.size(), 1u);
    EXPECT_STREQ(r.misaligned[0].name, "D");
    EXPECT_EQ(r.misaligned[0].partition, 1u);
    EXPECT_EQ(r.misaligned[0].byteOffset, 200);
}

TEST(SliceAlignment, ForceRoundsAndOffKeepsPlain)
{
    auto f = split(200, 2, {fp16D}, Mode::Force);
    EXPECT_EQ(f.share, 128);
    EXPECT_TRUE(f.misaligned.empty());
    auto o = split(200, 2, {fp16D}, Mode::Off);
    EXPECT_EQ(o.share, 100);
    EXPECT_EQ(o.misaligned.size(), 1u);
}

TEST(SliceAlignment, AutoRoundsWhenImbalanceIsUnderOnePercent)
{
    // 255 over 2: plain 127 -> 128, the last slice gets 127 (0.78% less).
    EXPECT_EQ(split(255, 2, {fp16D}, Mode::Auto).share, 128);
}

TEST(SliceAlignment, NSplitWithTwoMatrices)
{
    // B: ldb 128 (step 128), D: ldd 2 (step 2), both fp16.
    std::vector<Matrix> cut{{"B", 128, 16}, {"D", 2, 16}};
    auto                r = split(1000000, 2, cut, Mode::Auto);
    EXPECT_EQ(r.share, 500032);
    EXPECT_TRUE(r.misaligned.empty());

    auto o = split(1000000, 2, cut, Mode::Off);
    EXPECT_EQ(o.share, 500000);
    ASSERT_EQ(o.misaligned.size(), 1u);
    EXPECT_STREQ(o.misaligned[0].name, "D");
    EXPECT_EQ(o.misaligned[0].byteOffset, 2000000);
}

TEST(SliceAlignment, Fp4UsesBits)
{
    // 4 bits: 512 elements per 256 bytes.
    auto r = split(4096, 2, {{"D", 1, 4}}, Mode::Auto);
    EXPECT_EQ(r.share, 2048);
    EXPECT_TRUE(r.misaligned.empty());
    EXPECT_EQ(r.granularity, 512);
}

TEST(SliceAlignment, ThreeWay)
{
    // 3 * 2048 = 6144 over 3 is 2048 each.
    auto r = split(6144, 3, {fp16D}, Mode::Auto);
    EXPECT_EQ(r.share, 2048);
    EXPECT_TRUE(r.misaligned.empty());
    // 3000 over 3: plain 1000 -> 1024, last = 952 (7% less): stays plain, partition 1 misaligned.
    auto p = split(3000, 3, {fp16D}, Mode::Auto);
    EXPECT_EQ(p.share, 1000);
    ASSERT_EQ(p.misaligned.size(), 1u);
    EXPECT_EQ(p.misaligned[0].byteOffset, 2000);
}

TEST(SliceAlignment, MorePartsThanTotalGivesZero)
{
    EXPECT_EQ(split(1, 2, {fp16D}, Mode::Auto).share, 0);
}

TEST(SliceAlignment, ParseMode)
{
    Mode m = Mode::Off;
    EXPECT_TRUE(parseMode("auto", m));
    EXPECT_EQ(m, Mode::Auto);
    EXPECT_TRUE(parseMode("off", m));
    EXPECT_EQ(m, Mode::Off);
    EXPECT_TRUE(parseMode("force", m));
    EXPECT_EQ(m, Mode::Force);
    EXPECT_FALSE(parseMode("on", m));
    EXPECT_FALSE(parseMode("", m));
}

TEST(SliceAlignment, HintNamesTheMatrixAndOffset)
{
    auto r = split(200, 2, {fp16D}, Mode::Auto);
    auto h = hints(r, Dim::M, 2);
    ASSERT_EQ(h.size(), 1u);
    EXPECT_NE(h[0].find("slice of D"), std::string::npos);
    EXPECT_NE(h[0].find("200 bytes"), std::string::npos);
    EXPECT_NE(h[0].find("multiple of 256"), std::string::npos);
    EXPECT_TRUE(hints(split(4096, 2, {fp16D}, Mode::Auto), Dim::M, 2).empty());
}

TEST(SliceAlignment, HintForAContiguousMatrixPadsTheSplitDimension)
{
    // N split with a transposed fp16 B: B's step along N is 1, so only the share moves the start
    // of its slice and padding ldb does not help. The hint must ask for N to be padded.
    auto r = split(1000, 2, {{"B", 1, 16}, {"D", 4096, 16}}, Mode::Auto);
    auto h = hints(r, Dim::N, 2);
    ASSERT_EQ(h.size(), 1u);
    EXPECT_NE(h[0].find("slice of B"), std::string::npos);
    EXPECT_NE(h[0].find("pad N to a multiple of 256"), std::string::npos);
    EXPECT_EQ(h[0].find("leading dimension"), std::string::npos);
}

TEST(SliceAlignment, HintForAStridedMatrixPadsItsLeadingDimension)
{
    // N split with a non-transposed fp16 B and ldb = 100 (200 bytes): padding ldb to 256 bytes
    // aligns every slice whatever the share.
    auto r = split(1000, 2, {{"B", 100, 16}}, Mode::Off);
    auto h = hints(r, Dim::N, 2);
    ASSERT_EQ(h.size(), 1u);
    EXPECT_NE(h[0].find("pad the leading dimension of B to a multiple of 256 bytes"),
              std::string::npos);
}
