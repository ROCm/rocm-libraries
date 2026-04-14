// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck_tile/ops/direct_convolution/utils/transpose_lds_layout.hpp"

#include <set>
#include <tuple>

using namespace ck_tile::direct_conv;

// Dgrad config: M=4, K=4, B=16
using TrLayout = TransposeLDSLayout<4, 4, 16>;

TEST(TransposeLDSLayout, RowRange)
{
    for(int lane = 0; lane < 64; lane++)
    {
        int r = TrLayout::row(lane);
        EXPECT_GE(r, 0) << "lane=" << lane;
        EXPECT_LT(r, 4) << "lane=" << lane;  // K=4
    }
}

TEST(TransposeLDSLayout, ColMultipleOf4)
{
    for(int lane = 0; lane < 64; lane++)
    {
        int c = TrLayout::col(lane);
        EXPECT_EQ(c % 4, 0) << "lane=" << lane;
        EXPECT_GE(c, 0) << "lane=" << lane;
        EXPECT_LT(c, 4) << "lane=" << lane;  // M=4
    }
}

TEST(TransposeLDSLayout, BatchRange)
{
    for(int lane = 0; lane < 64; lane++)
    {
        int b = TrLayout::batch(lane);
        EXPECT_GE(b, 0) << "lane=" << lane;
        EXPECT_LT(b, 16) << "lane=" << lane;  // B=16
    }
}

TEST(TransposeLDSLayout, CompleteMapping)
{
    // All 64 lanes should cover the complete M x K x B element space
    // M=4 but col returns multiples of 4, so col/4 is always 0 for M=4
    // Effective unique values: (row, col/4, batch) = (K, M/4, B) = (4, 1, 16) = 64 elements
    std::set<std::tuple<int, int, int>> elements;
    for(int lane = 0; lane < 64; lane++)
    {
        auto t = std::make_tuple(TrLayout::row(lane), TrLayout::col(lane), TrLayout::batch(lane));
        elements.insert(t);
    }
    EXPECT_EQ(elements.size(), 64u);
}
