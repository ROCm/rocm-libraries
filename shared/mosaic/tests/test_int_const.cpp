// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstddef>
#include <mosaic/int_const.hpp>
#include <mosaic/int_tuple.hpp>  // for binary min
#include <type_traits>

using namespace mosaic;

// Compile-time tests
static_assert(std::is_same_v<decltype(Int<4>{} + Int<8>{}), Int<12>>);
static_assert(std::is_same_v<decltype(Int<4>{} * Int<8>{}), Int<32>>);
static_assert((Int<4>{} + Int<8>{}).value == 12);
static_assert((Int<4>{} * Int<8>{}).value == 32);
static_assert((Int<8>{} / Int<4>{}).value == 2);
static_assert((Int<8>{} % Int<3>{}).value == 2);
static_assert((Int<8>{} - Int<3>{}).value == 5);

TEST(IntConst, MixedStaticRuntimeArithmetic) {
    int x = 10;
    EXPECT_EQ(Int<4>{} + x, 14);
    EXPECT_EQ(x + Int<4>{}, 14);
    EXPECT_EQ(Int<4>{} * x, 40);
    EXPECT_EQ(x - Int<3>{}, 7);
}

TEST(IntConst, MixedRespectsRuntimeType) {
    // Runtime type wins — the static Int<v> casts to T (size_t here),
    // so no int-promotion footgun.
    std::size_t big = 5'000'000'000ull;
    auto result = Int<4>{} * big;
    EXPECT_EQ(result, std::size_t{20'000'000'000ull});
    static_assert(std::is_same_v<decltype(result), std::size_t>);
}

TEST(IntConst, DivmodStatic) {
    auto dm = divmod(Int<7>{}, Int<3>{});
    static_assert(dm.quot.value == 2);
    static_assert(dm.rem.value == 1);
}

TEST(IntConst, DivmodRuntime) {
    auto dm = divmod(std::size_t{13}, std::size_t{5});
    EXPECT_EQ(dm.quot, std::size_t{2});
    EXPECT_EQ(dm.rem, std::size_t{3});
}

TEST(IntConst, DivmodMixed) {
    // Int<v> with runtime
    auto dm1 = divmod(Int<13>{}, std::size_t{5});
    EXPECT_EQ(dm1.quot, std::size_t{2});
    EXPECT_EQ(dm1.rem, std::size_t{3});

    // runtime with Int<v>
    auto dm2 = divmod(std::size_t{13}, Int<5>{});
    EXPECT_EQ(dm2.quot, std::size_t{2});
    EXPECT_EQ(dm2.rem, std::size_t{3});
}

// ===================================================================
// Numeric helpers used by composition: abs, signum, ceil_div, min.
// ===================================================================

TEST(IntConst, AbsStatic) {
    static_assert(std::is_same_v<decltype(abs(Int<5>{})), Int<5>>);
    static_assert(std::is_same_v<decltype(abs(Int<-5>{})), Int<5>>);
    static_assert(std::is_same_v<decltype(abs(Int<0>{})), Int<0>>);
}

TEST(IntConst, AbsRuntime) {
    EXPECT_EQ(abs(int(-7)), 7);
    EXPECT_EQ(abs(int(7)), 7);
    EXPECT_EQ(abs(int(0)), 0);
}

TEST(IntConst, SignumStatic) {
    static_assert(std::is_same_v<decltype(signum(Int<5>{})), Int<1>>);
    static_assert(std::is_same_v<decltype(signum(Int<-5>{})), Int<-1>>);
    static_assert(std::is_same_v<decltype(signum(Int<0>{})), Int<0>>);
}

TEST(IntConst, SignumRuntime) {
    EXPECT_EQ(signum(int(7)), 1);
    EXPECT_EQ(signum(int(-7)), -1);
    EXPECT_EQ(signum(int(0)), 0);
}

TEST(IntConst, CeilDivStatic) {
    static_assert(std::is_same_v<decltype(ceil_div(Int<7>{}, Int<3>{})), Int<3>>);
    static_assert(std::is_same_v<decltype(ceil_div(Int<6>{}, Int<3>{})), Int<2>>);
    static_assert(std::is_same_v<decltype(ceil_div(Int<0>{}, Int<3>{})), Int<0>>);
    static_assert(std::is_same_v<decltype(ceil_div(Int<1>{}, Int<3>{})), Int<1>>);
    static_assert(std::is_same_v<decltype(ceil_div(Int<12>{}, Int<4>{})), Int<3>>);
}

TEST(IntConst, CeilDivRuntime) {
    EXPECT_EQ(ceil_div(int(7), int(3)), 3);
    EXPECT_EQ(ceil_div(int(6), int(3)), 2);
    EXPECT_EQ(ceil_div(int(0), int(3)), 0);
    EXPECT_EQ(ceil_div(int(12), int(4)), 3);
}

TEST(IntConst, CeilDivMixed) {
    EXPECT_EQ(ceil_div(Int<7>{}, int(3)), 3);
    EXPECT_EQ(ceil_div(int(7), Int<3>{}), 3);
}

TEST(IntConst, MinStatic) {
    static_assert(std::is_same_v<decltype(min(Int<3>{}, Int<5>{})), Int<3>>);
    static_assert(std::is_same_v<decltype(min(Int<5>{}, Int<3>{})), Int<3>>);
    static_assert(std::is_same_v<decltype(min(Int<5>{}, Int<5>{})), Int<5>>);
}

TEST(IntConst, MinRuntime) {
    EXPECT_EQ(min(int(3), int(5)), 3);
    EXPECT_EQ(min(int(5), int(3)), 3);
}

TEST(IntConst, MinMixed) {
    // Mixed static/runtime degrades to runtime — can't statically resolve.
    EXPECT_EQ(min(Int<3>{}, int(5)), 3);
    EXPECT_EQ(min(int(3), Int<5>{}), 3);
}
