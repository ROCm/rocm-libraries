// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/core.hpp"

#include <gtest/gtest.h>

namespace {

using namespace ck_tile;

// Checks get_index, _get_index, get_forward_step, get_backward_step and get_step_between of a
// curve against a table of expected indices, one row per access.
template <typename Curve, index_t NDim>
struct curve_checker
{
    using Row = index_t[NDim];

    template <index_t I, index_t... Ds>
    static constexpr bool index_ok(const Row* expected, sequence<Ds...>)
    {
        constexpr auto idx   = Curve::get_index(number<I>{});
        constexpr auto value = Curve::_get_index(number<I>{});
        return ((idx[number<Ds>{}] == expected[I][Ds]) && ...) &&
               ((value[Ds] == expected[I][Ds]) && ...);
    }

    template <index_t I, index_t... Ds>
    static constexpr bool steps_ok(const Row* expected, sequence<Ds...>)
    {
        constexpr auto fwd    = Curve::get_forward_step(number<I>{});
        constexpr auto bwd    = Curve::get_backward_step(number<I + 1>{});
        constexpr auto from_0 = Curve::get_step_between(number<0>{}, number<I + 1>{});
        constexpr auto to_0   = Curve::get_step_between(number<I + 1>{}, number<0>{});
        const auto step_ok    = [&](index_t d, index_t s) {
            return s == expected[I + 1][d] - expected[I][d];
        };
        return (step_ok(Ds, fwd[number<Ds>{}]) && ...) &&
               (step_ok(Ds, -bwd[number<Ds>{}]) && ...) &&
               ((from_0[number<Ds>{}] == expected[I + 1][Ds] - expected[0][Ds]) && ...) &&
               ((to_0[number<Ds>{}] == expected[0][Ds] - expected[I + 1][Ds]) && ...);
    }

    template <index_t... Is>
    static constexpr bool all_indices_ok(const Row* expected, sequence<Is...>)
    {
        return (index_ok<Is>(expected, make_index_sequence<NDim>{}) && ...);
    }

    template <index_t... Is>
    static constexpr bool all_steps_ok(const Row* expected, sequence<Is...>)
    {
        return (steps_ok<Is>(expected, make_index_sequence<NDim>{}) && ...);
    }

    template <index_t NumAccess>
    static constexpr bool check(const index_t (&expected)[NumAccess][NDim])
    {
        return Curve::get_num_of_access() == NumAccess &&
               all_indices_ok(expected, make_index_sequence<NumAccess>{}) &&
               all_steps_ok(expected, make_index_sequence<NumAccess - 1>{});
    }
};

// 2D, row-major access order, 4 scalars per access along dim 1
using Curve2d = space_filling_curve<sequence<4, 8>, sequence<0, 1>, sequence<1, 4>, false>;
constexpr index_t Curve2dExpected[8][2] = {
    {0, 0}, {0, 4}, {1, 0}, {1, 4}, {2, 0}, {2, 4}, {3, 0}, {3, 4}};

// Same curve with snake curving: dim 1 runs backward on odd rows
using Curve2dSnake = space_filling_curve<sequence<4, 8>, sequence<0, 1>, sequence<1, 4>, true>;
constexpr index_t Curve2dSnakeExpected[8][2] = {
    {0, 0}, {0, 4}, {1, 4}, {1, 0}, {2, 0}, {2, 4}, {3, 4}, {3, 0}};

// 3D snake with a non-identity access order (dim 2 slowest, dim 1 fastest), 2 scalars on dim 2
using Curve3dSnake =
    space_filling_curve<sequence<2, 3, 4>, sequence<2, 0, 1>, sequence<1, 1, 2>, true>;
constexpr index_t Curve3dSnakeExpected[12][3] = {{0, 0, 0},
                                                 {0, 1, 0},
                                                 {0, 2, 0},
                                                 {1, 2, 0},
                                                 {1, 1, 0},
                                                 {1, 0, 0},
                                                 {1, 0, 2},
                                                 {1, 1, 2},
                                                 {1, 2, 2},
                                                 {0, 2, 2},
                                                 {0, 1, 2},
                                                 {0, 0, 2}};

// 5D snake with unit dims and a whole dim per access, as in a GEMM tile window
using Curve5dSnake                           = space_filling_curve<sequence<1, 4, 1, 2, 8>,
                                                                   sequence<0, 1, 2, 3, 4>,
                                                                   sequence<1, 1, 1, 1, 8>,
                                                                   true>;
constexpr index_t Curve5dSnakeExpected[8][5] = {{0, 0, 0, 0, 0},
                                                {0, 0, 0, 1, 0},
                                                {0, 1, 0, 1, 0},
                                                {0, 1, 0, 0, 0},
                                                {0, 2, 0, 0, 0},
                                                {0, 2, 0, 1, 0},
                                                {0, 3, 0, 1, 0},
                                                {0, 3, 0, 0, 0}};

using Check2d      = curve_checker<Curve2d, 2>;
using Check2dSnake = curve_checker<Curve2dSnake, 2>;
using Check3dSnake = curve_checker<Curve3dSnake, 3>;
using Check5dSnake = curve_checker<Curve5dSnake, 5>;

static_assert(Check2d::check(Curve2dExpected));
static_assert(Check2dSnake::check(Curve2dSnakeExpected));
static_assert(Check3dSnake::check(Curve3dSnakeExpected));
static_assert(Check5dSnake::check(Curve5dSnakeExpected));

TEST(SpaceFillingCurve, Index2d) { EXPECT_TRUE(Check2d::check(Curve2dExpected)); }

TEST(SpaceFillingCurve, Index2dSnake) { EXPECT_TRUE(Check2dSnake::check(Curve2dSnakeExpected)); }

TEST(SpaceFillingCurve, Index3dSnakeReordered)
{
    EXPECT_TRUE(Check3dSnake::check(Curve3dSnakeExpected));
}

TEST(SpaceFillingCurve, Index5dSnake) { EXPECT_TRUE(Check5dSnake::check(Curve5dSnakeExpected)); }

} // namespace
