// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/layout.hpp>

#include <gtest/gtest.h>

using namespace rocm_ck;

// ============================================================================
// layout_name
// ============================================================================

TEST(Layout, MapsEnumValuesToExpectedStrings)
{
    EXPECT_STREQ(layout_name(Layout::Contiguous), "Contiguous");
    EXPECT_STREQ(layout_name(Layout::Row), "Row");
    EXPECT_STREQ(layout_name(Layout::Col), "Col");
    EXPECT_STREQ(layout_name(Layout::Auto), "Auto");
}

// ============================================================================
// is_valid_layout_for_rank
// ============================================================================

TEST(Layout, AllowsContiguousOnlyForRank1)
{
    constexpr bool r1 = is_valid_layout_for_rank(Layout::Contiguous, 1);
    constexpr bool r2 = is_valid_layout_for_rank(Layout::Contiguous, 2);
    constexpr bool r0 = is_valid_layout_for_rank(Layout::Contiguous, 0);
    EXPECT_TRUE(r1);
    EXPECT_FALSE(r2);
    EXPECT_FALSE(r0);
}

TEST(Layout, AllowsRowAndColOnlyForRank2)
{
    constexpr bool row_r1 = is_valid_layout_for_rank(Layout::Row, 1);
    constexpr bool row_r2 = is_valid_layout_for_rank(Layout::Row, 2);
    constexpr bool col_r1 = is_valid_layout_for_rank(Layout::Col, 1);
    constexpr bool col_r2 = is_valid_layout_for_rank(Layout::Col, 2);

    EXPECT_FALSE(row_r1);
    EXPECT_TRUE(row_r2);
    EXPECT_FALSE(col_r1);
    EXPECT_TRUE(col_r2);
}

TEST(Layout, RejectsAutoForAllRanks)
{
    constexpr bool auto_r0 = is_valid_layout_for_rank(Layout::Auto, 0);
    constexpr bool auto_r1 = is_valid_layout_for_rank(Layout::Auto, 1);
    constexpr bool auto_r2 = is_valid_layout_for_rank(Layout::Auto, 2);
    EXPECT_FALSE(auto_r0);
    EXPECT_FALSE(auto_r1);
    EXPECT_FALSE(auto_r2);
}
