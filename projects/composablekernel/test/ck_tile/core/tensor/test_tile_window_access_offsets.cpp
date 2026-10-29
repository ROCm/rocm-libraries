// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/core.hpp"

#include <gtest/gtest.h>

namespace {

using namespace ck_tile;

// A global-memory tile window of DataType over an M x N row-major view, vector size Vec
template <typename DataType, index_t M, index_t N, index_t Vec, typename Encoding>
struct window_case
{
    using View    = decltype(make_naive_tensor_view<address_space_enum::global>(
        static_cast<DataType*>(nullptr),
        make_tuple(number<M>{}, number<N>{}),
        make_tuple(number<N>{}, number<1>{}),
        number<Vec>{},
        number<1>{}));
    using Dstr    = decltype(make_static_tile_distribution(Encoding{}));
    using Lengths = tuple<number<M>, number<N>>;
    using Window  = tile_window_with_static_distribution<View, Lengths, Dstr, 1>;
    using Traits  = typename Window::Traits;
};

// get_access_d_offsets<IAccess>()[J] must equal the thread-buffer offset of element J of the
// access, from the Ys index of the access and the Ys-to-D descriptor, in units of PackedSize
template <typename Window, index_t IAccess, index_t J>
constexpr bool element_ok()
{
    using Traits                = typename Window::Traits;
    constexpr index_t P         = Traits::PackedSize;
    constexpr auto idx_ys_start = Traits::SFC_Ys::get_index(number<IAccess>{});
    constexpr auto idx_ys       = generate_tuple(
        [&](auto jj) {
            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + number<J * P>{})
                                                  : idx_ys_start[jj];
        },
        number<Window::Base::NDimY>{});
    constexpr index_t expected =
        typename Window::Base::TileDstr{}.get_ys_to_d_descriptor().calculate_offset(idx_ys) / P;
    return Window::template get_access_d_offsets<IAccess>()[J] == expected;
}

template <typename Window, index_t IAccess, index_t... Js>
constexpr bool access_ok(sequence<Js...>)
{
    return (element_ok<Window, IAccess, Js>() && ...);
}

template <typename Window, index_t... As>
constexpr bool window_ok(sequence<As...>)
{
    using Traits = typename Window::Traits;
    return (access_ok<Window, As>(
                make_index_sequence<Traits::ScalarPerVector / Traits::PackedSize>{}) &&
            ...);
}

template <typename Case>
constexpr bool check()
{
    return window_ok<typename Case::Window>(make_index_sequence<Case::Traits::NumAccess>{});
}

// (1) 2 Ys, vector on the last Y: Y = (2, 8), P0 = 16 x 4
using E1 = tile_distribution_encoding<sequence<>,
                                      tuple<sequence<2, 16>, sequence<4, 8>>,
                                      tuple<sequence<1, 2>>,
                                      tuple<sequence<1, 0>>,
                                      sequence<1, 2>,
                                      sequence<0, 1>>;
// (2) 3 Ys: Y = (2, 4, 8), P0 = 8 x 4
using E2 = tile_distribution_encoding<sequence<>,
                                      tuple<sequence<2, 4, 8>, sequence<4, 8>>,
                                      tuple<sequence<1, 2>>,
                                      tuple<sequence<2, 0>>,
                                      sequence<1, 1, 2>,
                                      sequence<0, 1, 1>>;
// (3) replication R = 2 on P0, 2 P dims (warp, lane)
using E3 = tile_distribution_encoding<sequence<2>,
                                      tuple<sequence<2, 2, 8>, sequence<4, 16>>,
                                      tuple<sequence<0, 1>, sequence<1, 2>>,
                                      tuple<sequence<0, 0>, sequence<1, 0>>,
                                      sequence<1, 2>,
                                      sequence<2, 1>>;
// (4) Ys in reversed X order, vector on the first Y (Y0 on X1, Y1 on X0)
using E4 = tile_distribution_encoding<sequence<>,
                                      tuple<sequence<4, 8>, sequence<4, 8>>,
                                      tuple<sequence<1, 2>>,
                                      tuple<sequence<1, 0>>,
                                      sequence<2, 1>,
                                      sequence<1, 0>>;
// (5) one Y of length 16, one access per thread row
using E5 = tile_distribution_encoding<sequence<>,
                                      tuple<sequence<16>, sequence<4, 16>>,
                                      tuple<sequence<1, 2>>,
                                      tuple<sequence<0, 0>>,
                                      sequence<2>,
                                      sequence<1>>;

template <typename T, index_t Vec>
constexpr bool check_all_encodings()
{
    return check<window_case<T, 32, 32, Vec, E1>>() && check<window_case<T, 64, 32, Vec, E2>>() &&
           check<window_case<T, 32, 64, Vec, E3>>() && check<window_case<T, 32, 32, Vec, E4>>() &&
           check<window_case<T, 16, 64, Vec, E5>>();
}

// PackedSize 1 (fp16, fp32, fp8) and 2 (pk_int4_t, pk_fp4_t)
static_assert(check_all_encodings<half_t, 8>());
static_assert(check_all_encodings<float, 4>());
static_assert(check_all_encodings<fp8_t, 8>());
static_assert(check_all_encodings<pk_int4_t, 8>());
static_assert(check_all_encodings<pk_fp4_t, 8>());

TEST(TileWindowAccessOffsets, Fp16) { EXPECT_TRUE((check_all_encodings<half_t, 8>())); }

TEST(TileWindowAccessOffsets, Fp32) { EXPECT_TRUE((check_all_encodings<float, 4>())); }

TEST(TileWindowAccessOffsets, Fp8) { EXPECT_TRUE((check_all_encodings<fp8_t, 8>())); }

TEST(TileWindowAccessOffsets, PkInt4) { EXPECT_TRUE((check_all_encodings<pk_int4_t, 8>())); }

TEST(TileWindowAccessOffsets, PkFp4) { EXPECT_TRUE((check_all_encodings<pk_fp4_t, 8>())); }

} // namespace
