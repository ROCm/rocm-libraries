// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "mosaic/int_tuple.hpp"

// TODO: CuTe's implementation of these functions may be more efficient.
//       Worth taking a closer look at some point.
// TODO: The hierarchial cases might not fully work here.

namespace mosaic {

// ===================================================================
// crd2idx — Coordinate to Index
//
// Three overloads following:
//   1. (int, int, int):     coord * stride  (single multiply)
//   2. (tuple, tuple, tuple): recursive per-mode sum (inner product)
//   3. (int, tuple, tuple):  divmod decomposition of scalar index
// ===================================================================

// Forward declarations — all crd2idx overloads must be visible to each other
// for the recursive calls in Cases 2/3 (mixed scalar/tuple per-mode dispatch).
template <class Coord, class Shape, class Stride>
    requires(!is_tuple_v<Coord> && !is_tuple_v<Shape> && !is_tuple_v<Stride>)
MOSAIC_HOST_DEVICE constexpr auto crd2idx(Coord const& coord, Shape const& shape,
                                          Stride const& stride);

template <class... CArgs, class... SArgs, class... DArgs>
    requires(sizeof...(CArgs) == sizeof...(SArgs) && sizeof...(SArgs) == sizeof...(DArgs))
MOSAIC_HOST_DEVICE constexpr auto crd2idx(std::tuple<CArgs...> const& coord,
                                          std::tuple<SArgs...> const& shape,
                                          std::tuple<DArgs...> const& stride);

template <class Coord, class... SArgs, class... DArgs>
    requires(!is_tuple_v<Coord> && sizeof...(SArgs) == sizeof...(DArgs) && sizeof...(SArgs) >= 2)
MOSAIC_HOST_DEVICE constexpr auto crd2idx(Coord const& coord, std::tuple<SArgs...> const& shape,
                                          std::tuple<DArgs...> const& stride);

template <class Coord, class S0, class D0>
    requires(!is_tuple_v<Coord>)
MOSAIC_HOST_DEVICE constexpr auto crd2idx(Coord const& coord, std::tuple<S0> const& shape,
                                          std::tuple<D0> const& stride);

// Case 1: scalar coord, scalar shape, scalar stride → coord * stride
template <class Coord, class Shape, class Stride>
    requires(!is_tuple_v<Coord> && !is_tuple_v<Shape> && !is_tuple_v<Stride>)
MOSAIC_HOST_DEVICE constexpr auto crd2idx(Coord const& coord, Shape const& /*shape*/,
                                          Stride const& stride) {
    return coord * stride;
}

// Case 2: tuple coord, tuple shape, tuple stride → sum across modes
template <class... CArgs, class... SArgs, class... DArgs>
    requires(sizeof...(CArgs) == sizeof...(SArgs) && sizeof...(SArgs) == sizeof...(DArgs))
MOSAIC_HOST_DEVICE constexpr auto crd2idx(std::tuple<CArgs...> const& coord,
                                          std::tuple<SArgs...> const& shape,
                                          std::tuple<DArgs...> const& stride) {
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
        return (crd2idx(std::get<I>(coord), std::get<I>(shape), std::get<I>(stride)) + ...);
    }(std::make_index_sequence<sizeof...(CArgs)>{});
}

// Case 3: scalar coord (flat index), tuple shape, tuple stride
// Divmod decomposition: distribute the scalar across modes.
//
// For shape (s0, s1, ..., sN) and stride (d0, d1, ..., dN):
//   c0 = coord % product(s0)   → recurse into mode 0
//   rest = coord / product(s0) → recurse into modes 1..N
//
// This is the colexicographic decomposition: leftmost mode varies fastest.
template <class Coord, class... SArgs, class... DArgs>
    requires(!is_tuple_v<Coord> && sizeof...(SArgs) == sizeof...(DArgs) && sizeof...(SArgs) >= 2)
MOSAIC_HOST_DEVICE constexpr auto crd2idx(Coord const& coord, std::tuple<SArgs...> const& shape,
                                          std::tuple<DArgs...> const& stride) {
    // Decompose into first mode and rest
    auto const& s0 = std::get<0>(shape);
    auto const& d0 = std::get<0>(stride);
    auto s0_size = product(s0);

    auto dm = divmod(coord, s0_size);

    // First mode contribution
    auto idx0 = crd2idx(dm.rem, s0, d0);

    // Remaining modes: build rest tuple
    if constexpr (sizeof...(SArgs) == 2) {
        // Only one remaining mode — no tuple wrapping
        auto idx_rest = crd2idx(dm.quot, std::get<1>(shape), std::get<1>(stride));
        return idx0 + idx_rest;
    } else {
        // Multiple remaining modes — build tail tuples
        auto shape_rest = [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::make_tuple(std::get<I + 1>(shape)...);
        }(std::make_index_sequence<sizeof...(SArgs) - 1>{});

        auto stride_rest = [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::make_tuple(std::get<I + 1>(stride)...);
        }(std::make_index_sequence<sizeof...(DArgs) - 1>{});

        return idx0 + crd2idx(dm.quot, shape_rest, stride_rest);
    }
}

// Single-mode tuple: unwrap and delegate
template <class Coord, class S0, class D0>
    requires(!is_tuple_v<Coord>)
MOSAIC_HOST_DEVICE constexpr auto crd2idx(Coord const& coord, std::tuple<S0> const& shape,
                                          std::tuple<D0> const& stride) {
    return crd2idx(coord, std::get<0>(shape), std::get<0>(stride));
}

// ===================================================================
// crd2idx (2-arg) — Coordinate to Index, the bijection partner of idx2crd
//
// Forms a true bijection between coordinates of Shape and integers in
// [0, size(Shape)) using the natural compact column-major stride
// (colexicographic enumeration: leftmost mode varies fastest).
//
// Round-trip invariant for all i in [0, size(shape)):
//   crd2idx(idx2crd(i, shape), shape) == i
//
// For arbitrary (non-compact) strides, use the 3-arg form which
// computes a layout offset rather than a bijective index.
// ===================================================================

// Scalar coord: trivial bijection (the integer IS the flat index)
template <class Coord, class Shape>
    requires(!is_tuple_v<Coord>)
MOSAIC_HOST_DEVICE constexpr auto crd2idx(Coord const& coord, Shape const& /*shape*/) {
    return coord;
}

// Tuple coord + tuple shape: delegate to the 3-arg form using the
// compact column-major stride implied by the shape.
template <class... CArgs, class Shape>
    requires(is_tuple_v<Shape>)
MOSAIC_HOST_DEVICE constexpr auto crd2idx(std::tuple<CArgs...> const& coord, Shape const& shape) {
    return crd2idx(coord, shape, compact_col_major(shape));
}

// ===================================================================
// idx2crd — Index to Coordinate (colexicographic decomposition)
//
// Converts a flat integer index into a hierarchical coordinate tuple.
// Leftmost mode varies fastest (colexicographic).
// ===================================================================

// Scalar shape: coordinate is just the index (mod shape for safety)
template <class Idx, class Shape>
    requires(!is_tuple_v<Idx> && !is_tuple_v<Shape>)
MOSAIC_HOST_DEVICE constexpr auto idx2crd(Idx const& idx, Shape const& shape) {
    return idx % shape;
}

// Tuple shape: decompose index across modes left-to-right
template <class Idx, class... SArgs>
    requires(!is_tuple_v<Idx>)
MOSAIC_HOST_DEVICE constexpr auto idx2crd(Idx const& idx, std::tuple<SArgs...> const& shape) {
    if constexpr (sizeof...(SArgs) == 1) {
        return std::make_tuple(idx2crd(idx, std::get<0>(shape)));
    } else {
        auto s0_size = product(std::get<0>(shape));
        auto dm = divmod(idx, s0_size);

        auto c0 = idx2crd(dm.rem, std::get<0>(shape));

        auto shape_rest = [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::make_tuple(std::get<I + 1>(shape)...);
        }(std::make_index_sequence<sizeof...(SArgs) - 1>{});

        auto c_rest = idx2crd(dm.quot, shape_rest);

        return std::tuple_cat(std::make_tuple(c0), c_rest);
    }
}

// ===================================================================
// compact stride generators — compute strides for dense packings
// ===================================================================

// Forward declarations: the detail helpers below recurse through these
// base-stride overloads so nested sub-shapes get a congruent (same-nesting)
// stride rather than a flat scalar.
template <class Shape, class Base>
MOSAIC_HOST_DEVICE constexpr auto compact_col_major(Shape const& shape, Base const& base);
template <class Shape, class Base>
MOSAIC_HOST_DEVICE constexpr auto compact_row_major(Shape const& shape, Base const& base);

namespace detail {

// Scalar base accumulator for col-major mode I: base * prod(shape[0..I-1]).
template <std::size_t I, class Shape, class Base>
MOSAIC_HOST_DEVICE constexpr auto col_major_base_at(Shape const& shape, Base const& base) {
    if constexpr (I == 0) {
        return base;
    } else {
        return col_major_base_at<I - 1>(shape, base) * product(std::get<I - 1>(shape));
    }
}

// Stride for col-major mode I. When the mode is itself a tuple, recurse so the
// emitted stride stays congruent with the (nested) shape.
template <std::size_t I, class Shape, class Base>
MOSAIC_HOST_DEVICE constexpr auto col_major_stride_at(Shape const& shape, Base const& base) {
    auto b = col_major_base_at<I>(shape, base);
    using S = std::decay_t<decltype(std::get<I>(shape))>;
    if constexpr (is_tuple_v<S>) {
        return compact_col_major(std::get<I>(shape), b);
    } else {
        return b;
    }
}

// Scalar base accumulator for row-major mode I: base * prod(shape[I+1..end]).
template <std::size_t I, class Shape, class Base>
MOSAIC_HOST_DEVICE constexpr auto row_major_base_at(Shape const& shape, Base const& base) {
    constexpr int R = rank_v<Shape>;
    if constexpr (I == R - 1) {
        return base;
    } else {
        return row_major_base_at<I + 1>(shape, base) * product(std::get<I + 1>(shape));
    }
}

// Stride for row-major mode I. Recurses into tuple modes for congruence.
template <std::size_t I, class Shape, class Base>
MOSAIC_HOST_DEVICE constexpr auto row_major_stride_at(Shape const& shape, Base const& base) {
    auto b = row_major_base_at<I>(shape, base);
    using S = std::decay_t<decltype(std::get<I>(shape))>;
    if constexpr (is_tuple_v<S>) {
        return compact_row_major(std::get<I>(shape), b);
    } else {
        return b;
    }
}

}  // namespace detail

// compact_col_major: strides = (1, s0, s0*s1, s0*s1*s2, ...)
template <class Shape>
MOSAIC_HOST_DEVICE constexpr auto compact_col_major(Shape const& shape) {
    if constexpr (is_tuple_v<Shape>) {
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::make_tuple(detail::col_major_stride_at<I>(shape, C<1>{})...);
        }(std::make_index_sequence<rank_v<Shape>>{});
    } else {
        return C<1>{};
    }
}

// Base-stride overload: strides = (base, base*s0, base*s0*s1, ...).
// Scalar shape returns just `base`. Used by complement to scale the tail
// rest_stride past the just-covered range.
template <class Shape, class Base>
MOSAIC_HOST_DEVICE constexpr auto compact_col_major(Shape const& shape, Base const& base) {
    if constexpr (is_tuple_v<Shape>) {
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::make_tuple(detail::col_major_stride_at<I>(shape, base)...);
        }(std::make_index_sequence<rank_v<Shape>>{});
    } else {
        return base;
    }
}

// compact_row_major: strides = (..., s_{n-1}, 1)
template <class Shape>
MOSAIC_HOST_DEVICE constexpr auto compact_row_major(Shape const& shape) {
    if constexpr (is_tuple_v<Shape>) {
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::make_tuple(detail::row_major_stride_at<I>(shape, C<1>{})...);
        }(std::make_index_sequence<rank_v<Shape>>{});
    } else {
        return C<1>{};
    }
}

// Base-stride overload (mirrors the col-major one): rightmost leaf varies
// fastest, scaled by `base`. Needed by the nested recursion in
// detail::row_major_stride_at.
template <class Shape, class Base>
MOSAIC_HOST_DEVICE constexpr auto compact_row_major(Shape const& shape, Base const& base) {
    if constexpr (is_tuple_v<Shape>) {
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::make_tuple(detail::row_major_stride_at<I>(shape, base)...);
        }(std::make_index_sequence<rank_v<Shape>>{});
    } else {
        return base;
    }
}

}  // namespace mosaic
