// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <vector>
#include <tuple>
#include "ck/utility/functional3.hpp"
#include "ck/utility/sequence.hpp"

using namespace ck;

// ============================================================================
// static_ford Tests
// ============================================================================

// Test basic static_ford construction and iteration
TEST(StaticFord, BasicConstruction2D)
{
    std::vector<std::pair<index_t, index_t>> visited;

    static_ford<Sequence<2, 3>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[Number<0>{}];
        constexpr index_t j = multi_id[Number<1>{}];
        visited.emplace_back(i, j);
    });

    ASSERT_EQ(visited.size(), 6u);
    EXPECT_EQ(visited[0], std::make_pair(0, 0));
    EXPECT_EQ(visited[1], std::make_pair(0, 1));
    EXPECT_EQ(visited[2], std::make_pair(0, 2));
    EXPECT_EQ(visited[3], std::make_pair(1, 0));
    EXPECT_EQ(visited[4], std::make_pair(1, 1));
    EXPECT_EQ(visited[5], std::make_pair(1, 2));
}

TEST(StaticFord, ReversedOrder2D)
{
    std::vector<std::pair<index_t, index_t>> visited;

    // Order (1, 0) means dim 1 is outer, dim 0 is inner (column-major)
    static_ford<Sequence<2, 3>, Sequence<1, 0>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[Number<0>{}];
        constexpr index_t j = multi_id[Number<1>{}];
        visited.emplace_back(i, j);
    });

    ASSERT_EQ(visited.size(), 6u);
    EXPECT_EQ(visited[0], std::make_pair(0, 0));
    EXPECT_EQ(visited[1], std::make_pair(1, 0));
    EXPECT_EQ(visited[2], std::make_pair(0, 1));
    EXPECT_EQ(visited[3], std::make_pair(1, 1));
    EXPECT_EQ(visited[4], std::make_pair(0, 2));
    EXPECT_EQ(visited[5], std::make_pair(1, 2));
}

TEST(StaticFord, BasicConstruction3D)
{
    std::vector<std::tuple<index_t, index_t, index_t>> visited;

    static_ford<Sequence<2, 2, 2>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[Number<0>{}];
        constexpr index_t j = multi_id[Number<1>{}];
        constexpr index_t k = multi_id[Number<2>{}];
        visited.emplace_back(i, j, k);
    });

    ASSERT_EQ(visited.size(), 8u);
    EXPECT_EQ(visited[0], std::make_tuple(0, 0, 0));
    EXPECT_EQ(visited[1], std::make_tuple(0, 0, 1));
    EXPECT_EQ(visited[2], std::make_tuple(0, 1, 0));
    EXPECT_EQ(visited[3], std::make_tuple(0, 1, 1));
    EXPECT_EQ(visited[4], std::make_tuple(1, 0, 0));
    EXPECT_EQ(visited[5], std::make_tuple(1, 0, 1));
    EXPECT_EQ(visited[6], std::make_tuple(1, 1, 0));
    EXPECT_EQ(visited[7], std::make_tuple(1, 1, 1));
}

TEST(StaticFord, CustomOrder3D)
{
    std::vector<std::tuple<index_t, index_t, index_t>> visited;

    // Order (2, 0, 1) means: dim 2 outer, dim 0 middle, dim 1 inner
    static_ford<Sequence<2, 2, 2>, Sequence<2, 0, 1>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[Number<0>{}];
        constexpr index_t j = multi_id[Number<1>{}];
        constexpr index_t k = multi_id[Number<2>{}];
        visited.emplace_back(i, j, k);
    });

    ASSERT_EQ(visited.size(), 8u);
    // With order (2, 0, 1): k varies slowest, then i, then j fastest
    EXPECT_EQ(visited[0], std::make_tuple(0, 0, 0));
    EXPECT_EQ(visited[1], std::make_tuple(0, 1, 0));
    EXPECT_EQ(visited[2], std::make_tuple(1, 0, 0));
    EXPECT_EQ(visited[3], std::make_tuple(1, 1, 0));
    EXPECT_EQ(visited[4], std::make_tuple(0, 0, 1));
    EXPECT_EQ(visited[5], std::make_tuple(0, 1, 1));
    EXPECT_EQ(visited[6], std::make_tuple(1, 0, 1));
    EXPECT_EQ(visited[7], std::make_tuple(1, 1, 1));
}

TEST(StaticFord, SingleDimension)
{
    std::vector<index_t> visited;

    static_ford<Sequence<5>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[Number<0>{}];
        visited.push_back(i);
    });

    ASSERT_EQ(visited.size(), 5u);
    EXPECT_EQ(visited[0], 0);
    EXPECT_EQ(visited[1], 1);
    EXPECT_EQ(visited[2], 2);
    EXPECT_EQ(visited[3], 3);
    EXPECT_EQ(visited[4], 4);
}

TEST(StaticFord, LargerDimensions)
{
    std::vector<std::pair<index_t, index_t>> visited;

    static_ford<Sequence<3, 4>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[Number<0>{}];
        constexpr index_t j = multi_id[Number<1>{}];
        visited.emplace_back(i, j);
    });

    ASSERT_EQ(visited.size(), 12u);
    // Verify first and last
    EXPECT_EQ(visited.front(), std::make_pair(0, 0));
    EXPECT_EQ(visited.back(), std::make_pair(2, 3));
}

TEST(StaticFord, CompileTimeMultiIndex)
{
    // Test that multi_id is truly compile-time by using it in constexpr context
    static_ford<Sequence<2, 2>>{}([](auto multi_id) {
        // These should all be compile-time constants
        constexpr index_t i    = multi_id[Number<0>{}];
        constexpr index_t j    = multi_id[Number<1>{}];
        constexpr index_t size = decltype(multi_id)::Size();
        static_assert(size == 2, "Multi-index should have 2 elements");
        (void)i;
        (void)j;
    });
    SUCCEED();
}

// ============================================================================
// ford Tests
// ============================================================================

TEST(Ford, BasicConstruction2D)
{
    std::vector<std::pair<index_t, index_t>> visited;

    ford<Sequence<2, 3>>{}([&](auto multi_id) {
        index_t i = multi_id[Number<0>{}];
        index_t j = multi_id[Number<1>{}];
        visited.emplace_back(i, j);
    });

    ASSERT_EQ(visited.size(), 6u);
    EXPECT_EQ(visited[0], std::make_pair(0, 0));
    EXPECT_EQ(visited[1], std::make_pair(0, 1));
    EXPECT_EQ(visited[2], std::make_pair(0, 2));
    EXPECT_EQ(visited[3], std::make_pair(1, 0));
    EXPECT_EQ(visited[4], std::make_pair(1, 1));
    EXPECT_EQ(visited[5], std::make_pair(1, 2));
}

TEST(Ford, ReversedOrder2D)
{
    std::vector<std::pair<index_t, index_t>> visited;

    ford<Sequence<2, 3>, Sequence<1, 0>>{}([&](auto multi_id) {
        index_t i = multi_id[Number<0>{}];
        index_t j = multi_id[Number<1>{}];
        visited.emplace_back(i, j);
    });

    ASSERT_EQ(visited.size(), 6u);
    EXPECT_EQ(visited[0], std::make_pair(0, 0));
    EXPECT_EQ(visited[1], std::make_pair(1, 0));
    EXPECT_EQ(visited[2], std::make_pair(0, 1));
    EXPECT_EQ(visited[3], std::make_pair(1, 1));
    EXPECT_EQ(visited[4], std::make_pair(0, 2));
    EXPECT_EQ(visited[5], std::make_pair(1, 2));
}

TEST(Ford, BasicConstruction3D)
{
    std::vector<std::tuple<index_t, index_t, index_t>> visited;

    ford<Sequence<2, 2, 2>>{}([&](auto multi_id) {
        index_t i = multi_id[Number<0>{}];
        index_t j = multi_id[Number<1>{}];
        index_t k = multi_id[Number<2>{}];
        visited.emplace_back(i, j, k);
    });

    ASSERT_EQ(visited.size(), 8u);
    EXPECT_EQ(visited[0], std::make_tuple(0, 0, 0));
    EXPECT_EQ(visited[1], std::make_tuple(0, 0, 1));
    EXPECT_EQ(visited[2], std::make_tuple(0, 1, 0));
    EXPECT_EQ(visited[3], std::make_tuple(0, 1, 1));
    EXPECT_EQ(visited[4], std::make_tuple(1, 0, 0));
    EXPECT_EQ(visited[5], std::make_tuple(1, 0, 1));
    EXPECT_EQ(visited[6], std::make_tuple(1, 1, 0));
    EXPECT_EQ(visited[7], std::make_tuple(1, 1, 1));
}

TEST(Ford, CustomOrder3D)
{
    std::vector<std::tuple<index_t, index_t, index_t>> visited;

    ford<Sequence<2, 2, 2>, Sequence<2, 0, 1>>{}([&](auto multi_id) {
        index_t i = multi_id[Number<0>{}];
        index_t j = multi_id[Number<1>{}];
        index_t k = multi_id[Number<2>{}];
        visited.emplace_back(i, j, k);
    });

    ASSERT_EQ(visited.size(), 8u);
    EXPECT_EQ(visited[0], std::make_tuple(0, 0, 0));
    EXPECT_EQ(visited[1], std::make_tuple(0, 1, 0));
    EXPECT_EQ(visited[2], std::make_tuple(1, 0, 0));
    EXPECT_EQ(visited[3], std::make_tuple(1, 1, 0));
    EXPECT_EQ(visited[4], std::make_tuple(0, 0, 1));
    EXPECT_EQ(visited[5], std::make_tuple(0, 1, 1));
    EXPECT_EQ(visited[6], std::make_tuple(1, 0, 1));
    EXPECT_EQ(visited[7], std::make_tuple(1, 1, 1));
}

TEST(Ford, SingleDimension)
{
    std::vector<index_t> visited;

    ford<Sequence<5>>{}([&](auto multi_id) {
        index_t i = multi_id[Number<0>{}];
        visited.push_back(i);
    });

    ASSERT_EQ(visited.size(), 5u);
    EXPECT_EQ(visited[0], 0);
    EXPECT_EQ(visited[1], 1);
    EXPECT_EQ(visited[2], 2);
    EXPECT_EQ(visited[3], 3);
    EXPECT_EQ(visited[4], 4);
}

// ============================================================================
// Consistency Tests (static_ford vs ford should produce same sequence)
// ============================================================================

TEST(FordConsistency, StaticFordMatchesFord2D)
{
    std::vector<std::pair<index_t, index_t>> static_visited;
    std::vector<std::pair<index_t, index_t>> runtime_visited;

    static_ford<Sequence<3, 4>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[Number<0>{}];
        constexpr index_t j = multi_id[Number<1>{}];
        static_visited.emplace_back(i, j);
    });

    ford<Sequence<3, 4>>{}([&](auto multi_id) {
        index_t i = multi_id[Number<0>{}];
        index_t j = multi_id[Number<1>{}];
        runtime_visited.emplace_back(i, j);
    });

    ASSERT_EQ(static_visited.size(), runtime_visited.size());
    for(size_t idx = 0; idx < static_visited.size(); ++idx)
    {
        EXPECT_EQ(static_visited[idx], runtime_visited[idx])
            << "Mismatch at index " << idx;
    }
}

TEST(FordConsistency, StaticFordMatchesFord3DWithOrder)
{
    std::vector<std::tuple<index_t, index_t, index_t>> static_visited;
    std::vector<std::tuple<index_t, index_t, index_t>> runtime_visited;

    static_ford<Sequence<2, 3, 2>, Sequence<1, 2, 0>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[Number<0>{}];
        constexpr index_t j = multi_id[Number<1>{}];
        constexpr index_t k = multi_id[Number<2>{}];
        static_visited.emplace_back(i, j, k);
    });

    ford<Sequence<2, 3, 2>, Sequence<1, 2, 0>>{}([&](auto multi_id) {
        index_t i = multi_id[Number<0>{}];
        index_t j = multi_id[Number<1>{}];
        index_t k = multi_id[Number<2>{}];
        runtime_visited.emplace_back(i, j, k);
    });

    ASSERT_EQ(static_visited.size(), runtime_visited.size());
    for(size_t idx = 0; idx < static_visited.size(); ++idx)
    {
        EXPECT_EQ(static_visited[idx], runtime_visited[idx])
            << "Mismatch at index " << idx;
    }
}

// ============================================================================
// ford_base Tests
// ============================================================================

TEST(FordBase, TotalSizeComputation)
{
    // 2 * 3 = 6
    using Ford2D = static_ford<Sequence<2, 3>>;
    EXPECT_EQ(Ford2D::Base::TotalSize, 6);

    // 2 * 3 * 4 = 24
    using Ford3D = static_ford<Sequence<2, 3, 4>>;
    EXPECT_EQ(Ford3D::Base::TotalSize, 24);

    // 5
    using Ford1D = static_ford<Sequence<5>>;
    EXPECT_EQ(Ford1D::Base::TotalSize, 5);
}

TEST(FordBase, NDimComputation)
{
    using Ford1D = static_ford<Sequence<5>>;
    EXPECT_EQ(Ford1D::Base::NDim, 1);

    using Ford2D = static_ford<Sequence<2, 3>>;
    EXPECT_EQ(Ford2D::Base::NDim, 2);

    using Ford3D = static_ford<Sequence<2, 3, 4>>;
    EXPECT_EQ(Ford3D::Base::NDim, 3);

    using Ford4D = static_ford<Sequence<2, 3, 4, 5>>;
    EXPECT_EQ(Ford4D::Base::NDim, 4);
}

// ============================================================================
// index_decomposer Tests
// ============================================================================

TEST(IndexDecomposer, DecomposeLinearIndex2D)
{
    // For Sequence<2, 3> (row-major): strides = {3, 1}
    // linear 0: (0/3)%2=0, (0/1)%3=0 -> (0,0)
    // linear 1: (1/3)%2=0, (1/1)%3=1 -> (0,1)
    // linear 5: (5/3)%2=1, (5/1)%3=2 -> (1,2)
    using Decomposer = detail::index_decomposer<Sequence<2, 3>, Sequence<0, 1>>;

    using Idx0 = typename Decomposer::template decompose<0>;
    EXPECT_EQ(Idx0::At(Number<0>{}), 0);
    EXPECT_EQ(Idx0::At(Number<1>{}), 0);

    using Idx1 = typename Decomposer::template decompose<1>;
    EXPECT_EQ(Idx1::At(Number<0>{}), 0);
    EXPECT_EQ(Idx1::At(Number<1>{}), 1);

    using Idx5 = typename Decomposer::template decompose<5>;
    EXPECT_EQ(Idx5::At(Number<0>{}), 1);
    EXPECT_EQ(Idx5::At(Number<1>{}), 2);
}

TEST(IndexDecomposer, DecomposeLinearIndex3D)
{
    // For Sequence<2, 3, 4>: strides = {12, 4, 1}
    using Decomposer = detail::index_decomposer<Sequence<2, 3, 4>, Sequence<0, 1, 2>>;

    using Idx0 = typename Decomposer::template decompose<0>;
    EXPECT_EQ(Idx0::At(Number<0>{}), 0);
    EXPECT_EQ(Idx0::At(Number<1>{}), 0);
    EXPECT_EQ(Idx0::At(Number<2>{}), 0);

    // linear 13 = 1*12 + 0*4 + 1 -> (1, 0, 1)
    using Idx13 = typename Decomposer::template decompose<13>;
    EXPECT_EQ(Idx13::At(Number<0>{}), 1);
    EXPECT_EQ(Idx13::At(Number<1>{}), 0);
    EXPECT_EQ(Idx13::At(Number<2>{}), 1);

    // linear 23 = 1*12 + 2*4 + 3 -> (1, 2, 3)
    using Idx23 = typename Decomposer::template decompose<23>;
    EXPECT_EQ(Idx23::At(Number<0>{}), 1);
    EXPECT_EQ(Idx23::At(Number<1>{}), 2);
    EXPECT_EQ(Idx23::At(Number<2>{}), 3);
}

TEST(IndexDecomposer, StrideComputation)
{
    // For Sequence<2, 3, 4>: strides = {3*4=12, 4, 1}
    using Decomposer = detail::index_decomposer<Sequence<2, 3, 4>, Sequence<0, 1, 2>>;

    EXPECT_EQ(Decomposer::strides[0], 12);
    EXPECT_EQ(Decomposer::strides[1], 4);
    EXPECT_EQ(Decomposer::strides[2], 1);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(FordEdgeCases, DimensionWithSizeOne)
{
    std::vector<std::tuple<index_t, index_t, index_t>> visited;

    // A dimension with size 1 should just contribute one iteration level
    static_ford<Sequence<2, 1, 3>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[Number<0>{}];
        constexpr index_t j = multi_id[Number<1>{}];
        constexpr index_t k = multi_id[Number<2>{}];
        visited.emplace_back(i, j, k);
    });

    ASSERT_EQ(visited.size(), 6u); // 2 * 1 * 3 = 6
    // j should always be 0
    for(const auto& v : visited)
    {
        EXPECT_EQ(std::get<1>(v), 0);
    }
}

TEST(FordEdgeCases, HigherDimensions4D)
{
    index_t count = 0;

    static_ford<Sequence<2, 2, 2, 2>>{}([&](auto multi_id) {
        (void)multi_id;
        ++count;
    });

    EXPECT_EQ(count, 16); // 2^4 = 16
}

TEST(FordEdgeCases, VariousSizes)
{
    index_t count = 0;

    static_ford<Sequence<3, 5, 2>>{}([&](auto multi_id) {
        (void)multi_id;
        ++count;
    });

    EXPECT_EQ(count, 30); // 3 * 5 * 2 = 30
}
