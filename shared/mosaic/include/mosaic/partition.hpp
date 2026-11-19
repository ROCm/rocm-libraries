// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "mosaic/algebra.hpp"
#include "mosaic/layout.hpp"
#include "mosaic/slice.hpp"
#include "mosaic/tensor.hpp"

namespace mosaic {

// =====================================================================
// X — "this tiler mode is not in the source tensor" placeholder.
// =====================================================================

struct X {};

// =====================================================================
// Step<Ts...> — mode-mask tag for local_tile / local_partition.
// Each Ts is either an Int<N> (keep that tiler/coord position) or X
// (drop that tiler/coord position).
// =====================================================================

template <class... Ts>
struct Step {
    using types = std::tuple<Ts...>;
    static constexpr std::size_t rank = sizeof...(Ts);
};

namespace detail {

// is_X_v<T> — true if T is mosaic::X.
template <class T>
struct is_X : std::false_type {};
template <>
struct is_X<X> : std::true_type {};
template <class T>
inline constexpr bool is_X_v = is_X<T>::value;

// Variadic "drop or keep" helper. Returns a 0- or 1-element tuple.
template <bool Drop, class T>
MOSAIC_HOST_DEVICE constexpr auto maybe_keep(T const& v) {
    if constexpr (Drop) {
        return std::tuple<>{};
    } else {
        return std::make_tuple(v);
    }
}

// filter_step<StepT>(tuple) — build a new tuple keeping only entries
// at positions where StepT's corresponding type is NOT X.
template <class StepT, class TupleT, std::size_t... I>
MOSAIC_HOST_DEVICE constexpr auto filter_step_impl(TupleT const& t, std::index_sequence<I...>) {
    return std::tuple_cat(
        maybe_keep<is_X_v<std::tuple_element_t<I, typename StepT::types>>>(std::get<I>(t))...);
}

template <class StepT, class TupleT>
MOSAIC_HOST_DEVICE constexpr auto filter_step(TupleT const& t) {
    static_assert(std::tuple_size_v<TupleT> == StepT::rank,
                  "local_tile: Step rank must match tiler / coord rank");
    return filter_step_impl<StepT>(t, std::make_index_sequence<StepT::rank>{});
}

// Build a tuple of N copies of `_` (Underscore). Used to keep all tile
// modes during the slice that follows flat_divide.
template <std::size_t N>
MOSAIC_HOST_DEVICE constexpr auto repeat_underscore() {
    return []<std::size_t... I>(std::index_sequence<I...>) {
        return std::make_tuple(((void)I, _)...);
    }(std::make_index_sequence<N>{});
}

}  // namespace detail

// =====================================================================
// local_tile(tensor, tiler, coord, step)
// ---------------------------------------------------------------------
// Extract this workgroup's view of `tensor` at `coord` after applying `tiler`,
// projected onto the source's modes via `step`.
//
// Algorithm:
//   1. Filter `tiler` and `coord` by `step` — drop entries at X positions.
//      Produces a sub-tiler and sub-coord whose rank matches `tensor`.
//   2. `flat_divide(tensor.layout(), sub_tiler)` — split each source mode
//      by the corresponding sub-tiler mode. Result layout has rank
//      2*source_rank: (tile_modes..., rest_modes...).
//   3. Construct a Tensor over the divided layout using `tensor`'s engine.
//   4. Build the full slice coord as
//        (_, _, ..., _, sub_coord_0, sub_coord_1, ...)
//      where the first `sub_rank` entries are `_` (keep tile modes) and
//      the trailing entries come from sub_coord (slice rest modes, or
//      keep with `_` if the caller passed `_` there).
//   5. Apply the slice — returns a sub-tensor of shape
//        (tile_modes..., kept_rest_modes...).
//
// The standard kernel call sites:
//   gA_wg = local_tile(gA, wg_tiler, make_coord(wg_m, _, _),     Step<_1,X,_1>{});
//   gB_wg = local_tile(gB, wg_tiler, make_coord(_, wg_n, _),     Step<X,_1,_1>{});
//   gD_wg = local_tile(gD, wg_tiler, make_coord(wg_m, wg_n, _), Step<_1,_1,X>{});
//
// For A: result shape (BLK_M, BLK_K, k_tiles)  — K mode kept for K-loop
// For B: result shape (BLK_K, BLK_N, k_tiles) or (BLK_N, BLK_K, k_tiles)
//                                              depending on source layout
// For D: result shape (BLK_M, BLK_N)           — both M and N sliced
//
// =====================================================================

template <class Engine, class L, class Tiler, class Coord, class StepT>
MOSAIC_HOST_DEVICE constexpr auto local_tile(Tensor<Engine, L> const& tensor, Tiler tiler,
                                             Coord coord, StepT) {
    auto sub_tiler = detail::filter_step<StepT>(tiler);
    auto sub_coord = detail::filter_step<StepT>(coord);

    // flat_divide → layout with shape (tile_modes..., rest_modes...).
    auto divided_layout = flat_divide(tensor.layout(), sub_tiler);
    auto divided_tensor = make_tensor(tensor.engine(), divided_layout);

    // The number of tile modes equals sub_tiler's rank — keep them all
    // with `_`, then append sub_coord for the rest modes.
    constexpr std::size_t tile_rank = std::tuple_size_v<decltype(sub_tiler)>;
    auto full_coord = std::tuple_cat(detail::repeat_underscore<tile_rank>(), sub_coord);

    return divided_tensor(full_coord);
}

}  // namespace mosaic
