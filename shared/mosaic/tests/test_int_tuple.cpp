// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <mosaic/int_tuple.hpp>

using namespace mosaic;

TEST(IntTuple, Rank) {
    // Rank 1 tuples
    // rank(4) == 1
    auto r1_scalar = Int<4>{};
    EXPECT_EQ(rank(r1_scalar), Int<1>{});

    // rank((4)) == 1
    auto r1_tuple = std::make_tuple(Int<4>{});
    EXPECT_EQ(rank(r1_tuple), Int<1>{});

    // rank(((4))) == 1
    auto r1_htuple = std::make_tuple(r1_tuple);
    EXPECT_EQ(rank(r1_htuple), Int<1>{});

    // rank((4, 8)) == 2
    auto r2_tuple = std::make_tuple(Int<4>{}, Int<8>{});
    EXPECT_EQ(rank(r2_tuple), Int<2>{});

    // rank(((4, 6), (3, (2, 2), 8))) == 2
    auto r2_htuple =
        std::make_tuple(std::make_tuple(Int<4>{}, Int<6>{}),
                        std::make_tuple(Int<3>{}, std::make_tuple(Int<2>{}, Int<2>{}), Int<8>{}));
    EXPECT_EQ(rank(r2_htuple), Int<2>{});

    EXPECT_EQ(rank<0>(r2_htuple), Int<2>{});  // rank((4, 6)) == 2
    EXPECT_EQ(rank<1>(r2_htuple), Int<3>{});  // rank((3, (2, 2), 8)) == 3
    // Extra parens needed: the preprocessor doesn't understand C++ template syntax,
    // so the comma in `rank<1, 0>` looks like a macro arg separator without them.
    EXPECT_EQ((rank<1, 0>(r2_htuple)), Int<1>{});  // rank(3) == 1
}

TEST(IntTuple, Depth) {
    // Depth 0 tuples
    // depth(4) == 0
    auto d0_scalar = Int<4>{};
    EXPECT_EQ(depth(d0_scalar), Int<0>{});

    // depth((4)) == 1
    auto d1_tuple = std::make_tuple(Int<4>{});
    EXPECT_EQ(depth(d1_tuple), Int<1>{});

    // depth(((4))) == 2
    auto d2_htuple = std::make_tuple(d1_tuple);
    EXPECT_EQ(depth(d2_htuple), Int<2>{});

    // // depth(((4, 6), (3, (2, 2), 8))) == 3
    auto r2_htuple =
        std::make_tuple(std::make_tuple(Int<4>{}, Int<6>{}),
                        std::make_tuple(Int<3>{}, std::make_tuple(Int<2>{}, Int<2>{}), Int<8>{}));
    EXPECT_EQ(depth(r2_htuple), Int<3>{});

    EXPECT_EQ(depth<0>(r2_htuple), Int<1>{});       // depth((4, 6)) == 1
    EXPECT_EQ(depth<1>(r2_htuple), Int<2>{});       // depth((3, (2, 2), 8)) == 2
    EXPECT_EQ((depth<1, 0>(r2_htuple)), Int<0>{});  // depth(3) == 0
}

TEST(IntTuple, MaxPreservesCompileTimeAcrossDifferentInts) {
    auto m = max(Int<3>{}, Int<7>{});
    static_assert(std::is_same_v<decltype(m), Int<7>>);
    EXPECT_EQ(m, Int<7>{});
}

TEST(IntTuple, DepthOfNonUniformTuplePreservesCompileTime) {
    auto t = std::make_tuple(Int<5>{}, std::make_tuple(Int<6>{}, Int<7>{}));
    auto d = depth(t);
    static_assert(std::is_same_v<decltype(d), Int<2>>);
    EXPECT_EQ(d, Int<2>{});
}

TEST(IntTuple, Product) {
    // product(((5), (6, 7))) == 210
    auto t = std::make_tuple(Int<5>{}, std::make_tuple(Int<6>{}, Int<7>{}));
    auto p = product(t);
    static_assert(std::is_same_v<decltype(p), Int<210>>);
    EXPECT_EQ(p, Int<210>{});

    // product((6, 7)) == 42
    auto p1 = product<1>(t);
    static_assert(std::is_same_v<decltype(p1), Int<42>>);
    EXPECT_EQ(p1, Int<42>{});

    // product(7) == 7
    auto p2 = product<1, 1>(t);
    static_assert(std::is_same_v<decltype(p2), Int<7>>);
    EXPECT_EQ(p2, Int<7>{});
}

TEST(IntTuple, Size) {
    // size(((5), (6, 7))) == 210
    auto t = std::make_tuple(Int<5>{}, std::make_tuple(Int<6>{}, Int<7>{}));
    auto s = size(t);
    static_assert(std::is_same_v<decltype(s), Int<210>>);
    EXPECT_EQ(s, Int<210>{});

    // size((6, 7)) == 42
    auto s1 = size<1>(t);
    static_assert(std::is_same_v<decltype(s1), Int<42>>);
    EXPECT_EQ(s1, Int<42>{});

    // size(7) == 7
    auto s2 = size<1, 1>(t);
    static_assert(std::is_same_v<decltype(s2), Int<7>>);
    EXPECT_EQ(s2, Int<7>{});
}

// ===================================================================
// Variadic min, find, remove — helpers for complement's sort-by-stride.
// ===================================================================

TEST(IntTuple, MinVariadic) {
    static_assert(std::is_same_v<decltype(min(Int<3>{}, Int<5>{}, Int<2>{})), Int<2>>);
    static_assert(std::is_same_v<decltype(min(Int<7>{}, Int<7>{}, Int<7>{})), Int<7>>);
    auto t = std::make_tuple(Int<3>{}, Int<5>{}, Int<2>{}, Int<8>{});
    static_assert(std::is_same_v<decltype(min(t)), Int<2>>);
}

TEST(IntTuple, FindStatic) {
    auto t = std::make_tuple(Int<5>{}, Int<3>{}, Int<8>{}, Int<3>{});
    static_assert(std::is_same_v<decltype(find(t, Int<5>{})), Int<0>>);
    static_assert(std::is_same_v<decltype(find(t, Int<3>{})), Int<1>>);  // first match
    static_assert(std::is_same_v<decltype(find(t, Int<8>{})), Int<2>>);
}

TEST(IntTuple, FindNotPresent) {
    auto t = std::make_tuple(Int<5>{}, Int<3>{});
    static_assert(std::is_same_v<decltype(find(t, Int<9>{})), Int<-1>>);
}

TEST(IntTuple, RemoveAt) {
    auto t = std::make_tuple(Int<5>{}, Int<3>{}, Int<8>{});
    auto r0 = remove<0>(t);
    auto r1 = remove<1>(t);
    auto r2 = remove<2>(t);
    static_assert(std::is_same_v<decltype(r0), std::tuple<Int<3>, Int<8>>>);
    static_assert(std::is_same_v<decltype(r1), std::tuple<Int<5>, Int<8>>>);
    static_assert(std::is_same_v<decltype(r2), std::tuple<Int<5>, Int<3>>>);
}

TEST(IntTuple, RemoveLastLeavesEmpty) {
    auto t = std::make_tuple(Int<5>{});
    auto r = remove<0>(t);
    static_assert(std::is_same_v<decltype(r), std::tuple<>>);
}

// ===================================================================
// repeat<N>(x) and zip2_by — building blocks for the divide family.
// ===================================================================

TEST(IntTuple, RepeatScalar) {
    // N == 1 returns the value itself (scalar), not a 1-tuple.
    auto r = repeat<1>(Int<5>{});
    static_assert(std::is_same_v<decltype(r), Int<5>>);
}

TEST(IntTuple, RepeatTuple) {
    auto r = repeat<3>(Int<7>{});
    static_assert(std::is_same_v<decltype(r), std::tuple<Int<7>, Int<7>, Int<7>>>);
}

TEST(IntTuple, Zip2ByMatchingRanks) {
    // t = ((A, a), (B, b)), guide = (X, Y) — both rank 2.
    // Expected: ((A, B), (a, b))
    auto t =
        std::make_tuple(std::make_tuple(Int<1>{}, Int<10>{}), std::make_tuple(Int<2>{}, Int<20>{}));
    auto guide = std::make_tuple(Int<0>{}, Int<0>{});  // any scalars
    auto z = zip2_by(t, guide);
    static_assert(
        std::is_same_v<decltype(z),
                       std::tuple<std::tuple<Int<1>, Int<2>>, std::tuple<Int<10>, Int<20>>>>);
}

TEST(IntTuple, Zip2ByTrailingModes) {
    // t rank 3, guide rank 2 → trailing t[2] appended to second output.
    // t = ((A, a), (B, b), C), guide = (X, Y)
    // Expected: ((A, B), (a, b, C))
    auto t = std::make_tuple(std::make_tuple(Int<1>{}, Int<10>{}),
                             std::make_tuple(Int<2>{}, Int<20>{}), Int<99>{});
    auto guide = std::make_tuple(Int<0>{}, Int<0>{});
    auto z = zip2_by(t, guide);
    static_assert(std::is_same_v<decltype(z), std::tuple<std::tuple<Int<1>, Int<2>>,
                                                         std::tuple<Int<10>, Int<20>, Int<99>>>>);
}

TEST(IntTuple, Zip2ByScalarGuide) {
    // Scalar guide + rank-2 t: returns t unchanged.
    auto t = std::make_tuple(Int<5>{}, Int<7>{});
    auto z = zip2_by(t, Int<0>{});
    static_assert(std::is_same_v<decltype(z), decltype(t)>);
}
