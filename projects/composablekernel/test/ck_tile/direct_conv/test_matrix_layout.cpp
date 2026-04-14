// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck_tile/ops/direct_convolution/utils/matrix_layout.hpp"

#include <set>
#include <tuple>

using namespace ck_tile::direct_conv;

// The kernel uses MatrixLayout<4,4,16,__half> for operands
using OpLayout = MatrixLayout<4, 4, 16, __half>;
// and MatrixLayout<4,4,16,float> for results
using ResLayout = MatrixLayout<4, 4, 16, float>;

TEST(MatrixLayout, ItemsPerRegister_Half)
{
    EXPECT_EQ(OpLayout::items_per_register(), 2);
}

TEST(MatrixLayout, ItemsPerRegister_Float)
{
    EXPECT_EQ(ResLayout::items_per_register(), 1);
}

TEST(MatrixLayout, OuterModM)
{
    for(int lane = 0; lane < 64; lane++)
    {
        EXPECT_EQ(OpLayout::outer(lane), lane % 4) << "lane=" << lane;
    }
}

TEST(MatrixLayout, BatchModB)
{
    for(int lane = 0; lane < 64; lane++)
    {
        EXPECT_EQ(OpLayout::batch(lane), (lane / 4) % 16) << "lane=" << lane;
    }
}

TEST(MatrixLayout, InnerStriding)
{
    // K_L = K / (64 / (M * B)) = 4 / (64 / (4 * 16)) = 4 / 1 = 4
    // inner(lane, idx) = idx * items_per_register() + (lane / (M * B)) * K_L
    // For half: items_per_register = 2
    // M * B = 64, so lane / 64 = 0 for all lanes 0-63
    // inner(lane, 0) = 0, inner(lane, 1) = 2
    for(int lane = 0; lane < 64; lane++)
    {
        EXPECT_EQ(OpLayout::inner(lane, 0), 0) << "lane=" << lane;
        EXPECT_EQ(OpLayout::inner(lane, 1), 2) << "lane=" << lane;
    }
}

TEST(MatrixLayout, UniqueTriples)
{
    // Each lane should produce a unique (outer, inner, batch) triple (for idx=0)
    std::set<std::tuple<int, int, int>> triples;
    for(int lane = 0; lane < 64; lane++)
    {
        auto t = std::make_tuple(OpLayout::outer(lane), OpLayout::inner(lane, 0), OpLayout::batch(lane));
        triples.insert(t);
    }
    EXPECT_EQ(triples.size(), 64u);
}
