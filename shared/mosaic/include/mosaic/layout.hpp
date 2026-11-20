// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>

#include "mosaic/slice.hpp"
#include "mosaic/stride.hpp"

namespace mosaic {

template <class... Ts>
using Shape = std::tuple<Ts...>;

template <class... Ts>
using Stride = std::tuple<Ts...>;

template <class... Ts>
using Coord = std::tuple<Ts...>;

template <class... Ts>
MOSAIC_HOST_DEVICE constexpr auto make_shape(Ts... ts) {
    return Shape<Ts...>{ts...};
}

template <class... Ts>
MOSAIC_HOST_DEVICE constexpr auto make_stride(Ts... ts) {
    return Stride<Ts...>{ts...};
}

template <class... Ts>
MOSAIC_HOST_DEVICE constexpr auto make_coord(Ts... ts) {
    return Coord<Ts...>{ts...};
}

// ===================================================================
// Layout<Shape, Stride> — the central layout type
//
// A layout maps coordinates to offsets via the inner product
// offset = sum(coord_i * stride_i).
//
// Two evaluation paths:
//   L(c0, c1, ...) — coordinate access: pure inner product (zero div/mod)
//   L(flat_idx)    — flat index: idx2crd + crd2idx
// ===================================================================

template <class Shape, class Stride = decltype(compact_col_major(std::declval<Shape>()))>
struct Layout {
    Shape shape_;
    Stride stride_;

    MOSAIC_HOST_DEVICE constexpr Layout() = default;

    MOSAIC_HOST_DEVICE constexpr Layout(Shape const& shape, Stride const& stride)
        : shape_(shape), stride_(stride) {}

    // Shape-only constructor: default to column-major stride (matches CuTe).
    // Column-major means leftmost coordinate varies fastest:
    //   shape (M, N) → stride (1, M)
    MOSAIC_HOST_DEVICE constexpr explicit Layout(Shape const& shape)
        : shape_(shape), stride_(compact_col_major(shape)) {}

    // ---------------------------------------------------------------
    // Accessors
    // ---------------------------------------------------------------

    MOSAIC_HOST_DEVICE constexpr auto const& shape() const {
        return shape_;
    }
    MOSAIC_HOST_DEVICE constexpr auto const& stride() const {
        return stride_;
    }

    // Total number of elements in the domain
    MOSAIC_HOST_DEVICE constexpr auto size() const {
        return product(shape_);
    }

    // rank: number of top-level modes
    static constexpr int rank = rank_v<Shape>;

    // Access shape/stride of a specific mode
    template <int I>
    MOSAIC_HOST_DEVICE constexpr auto shape_at() const {
        if constexpr (is_tuple_v<Shape>) {
            return std::get<I>(shape_);
        } else {
            static_assert(I == 0);
            return shape_;
        }
    }

    template <int I>
    MOSAIC_HOST_DEVICE constexpr auto stride_at() const {
        if constexpr (is_tuple_v<Stride>) {
            return std::get<I>(stride_);
        } else {
            static_assert(I == 0);
            return stride_;
        }
    }

    // ---------------------------------------------------------------
    // L(row, col) = row * stride[0] + col * stride[1]
    // Pure inner product — zero div/mod operations.
    // ---------------------------------------------------------------

    // Variadic coordinates (2+ args). Build a coord tuple and dispatch via
    // the tuple overload so any `_` placeholders trigger slicing.
    template <class C0, class C1, class... Cs>
    MOSAIC_HOST_DEVICE constexpr auto operator()(C0 c0, C1 c1, Cs... cs) const {
        return (*this)(make_coord(c0, c1, cs...));
    }

    // Single coordinate. Underscore alone returns the layout unchanged
    // (degenerate slice); any integral acts as a flat index.
    template <class Idx>
        requires(is_integral_v<Idx> || is_underscore_v<Idx>)
    MOSAIC_HOST_DEVICE constexpr auto operator()(Idx idx) const {
        if constexpr (is_underscore_v<Idx>) {
            return *this;
        } else {
            return crd2idx(idx, shape_, stride_);
        }
    }

    // Tuple coordinate. If the coord contains any `_`, slice; otherwise
    // evaluate as an offset.
    template <class... CArgs>
    MOSAIC_HOST_DEVICE constexpr auto operator()(std::tuple<CArgs...> const& coord) const {
        if constexpr (has_underscore_v<std::tuple<CArgs...>>) {
            return slice(coord, *this);
        } else {
            return crd2idx(coord, shape_, stride_);
        }
    }

    // No iterator. Matches CuTe — explicit index loops are the device-friendly
    // pattern, since end() computed from a runtime size_t doesn't unroll the way
    // a literal `for (int i = 0; i < size(L); ++i)` loop does.
};

// ===================================================================
// make_layout — factory functions
// ===================================================================

// Detect Layout instances so the (Shape, Stride) overload doesn't
// collide with the variadic (Layout, Layout, ...) overload.
template <class T>
struct is_layout : std::false_type {};

template <class S, class D>
struct is_layout<Layout<S, D>> : std::true_type {};

template <class T>
inline constexpr bool is_layout_v = is_layout<std::decay_t<T>>::value;

// Shape + Stride. Disabled when both args are Layout (then the variadic
// hierarchical overload below should win).
template <class Shape, class Stride>
    requires(!is_layout_v<Shape> && !is_layout_v<Stride>)
MOSAIC_HOST_DEVICE constexpr auto make_layout(Shape const& shape, Stride const& stride) {
    return Layout<Shape, Stride>{shape, stride};
}

// Shape only — defaults to compact column-major stride (matches CuTe).
// For shape (M, N), the resulting stride is (1, M): leftmost coord varies fastest.
template <class Shape>
MOSAIC_HOST_DEVICE constexpr auto make_layout(Shape const& shape) {
    auto stride = compact_col_major(shape);
    return Layout<Shape, decltype(stride)>{shape, stride};
}

// 1-Layout overload: wrap into a rank-1 hierarchical layout.
// make_layout(L) → Layout<tuple<L_Shape>, tuple<L_Stride>>. Mode 0 contains L's
// full shape/stride (potentially nested). Matches CuTe's behavior — without this
// overload, the (Shape) overload would match and treat the Layout as a Shape, which
// would silently produce garbage.
template <class S, class D>
MOSAIC_HOST_DEVICE constexpr auto make_layout(Layout<S, D> const& l) {
    return Layout<std::tuple<S>, std::tuple<D>>{std::make_tuple(l.shape()),
                                                std::make_tuple(l.stride())};
}

// Concatenation: make_layout(L0, L1, ...) → Layout with hierarchical shape/stride.
// Each input layout becomes one top-level mode: the result has rank == sizeof...(ls),
// where mode i has shape == ls_i.shape() and stride == ls_i.stride() (themselves
// possibly tuples). Variadic in the number of layouts; deduces matched (Ss, Ds) pairs
// from the parameter pattern.
template <class... Ss, class... Ds>
    requires(sizeof...(Ss) >= 2)
MOSAIC_HOST_DEVICE constexpr auto make_layout(Layout<Ss, Ds> const&... ls) {
    auto shape = std::make_tuple(ls.shape()...);
    auto stride = std::make_tuple(ls.stride()...);
    return Layout<decltype(shape), decltype(stride)>{shape, stride};
}

// ===================================================================
// Free-function accessors for Layout
//
// Members exist too (l.shape(), l.stride(), l.size()), but the free
// functions match CuTe's algorithm-oriented vocabulary: generic algorithms
// can write shape(x), stride(x), size(x) uniformly across IntTuples,
// Layouts, and (eventually) Tensors.
//
// rank(L) returns Int<N> rather than int — same shape as size(L) and
// cosize(L), so they all compose in compile-time arithmetic chains.
// ===================================================================

template <class S, class D>
MOSAIC_HOST_DEVICE constexpr auto const& shape(Layout<S, D> const& l) {
    return l.shape();
}

template <class S, class D>
MOSAIC_HOST_DEVICE constexpr auto const& stride(Layout<S, D> const& l) {
    return l.stride();
}

template <class S, class D>
MOSAIC_HOST_DEVICE constexpr auto size(Layout<S, D> const& l) {
    return l.size();
}

// size<I>(L): product of the I-th top-level mode (sub-shape's leaves).
template <int I, class S, class D>
MOSAIC_HOST_DEVICE constexpr auto size(Layout<S, D> const& l) {
    return product(get<I>(l.shape()));
}

// rank(L): Int<N> for symmetry with size, cosize. Composes in compile-time math.
template <class S, class D>
MOSAIC_HOST_DEVICE constexpr auto rank(Layout<S, D> const& /*l*/) {
    if constexpr (is_tuple_v<S>) {
        return Int<int(std::tuple_size_v<S>)>{};
    } else {
        return Int<1>{};
    }
}

// ===================================================================
// Utility: cosize — maximum offset + 1 (the codomain size)
// ===================================================================

namespace detail {

template <class Shape, class Stride>
MOSAIC_HOST_DEVICE constexpr auto cosize_impl(Shape const& shape, Stride const& stride) {
    if constexpr (is_tuple_v<Shape>) {
        // Sum of (shape_i - 1) * stride_i across all modes, + 1
        auto result = [&]<std::size_t... I>(std::index_sequence<I...>) {
            return ((cosize_impl(std::get<I>(shape), std::get<I>(stride)) - 1) + ... + 0);
        }(std::make_index_sequence<rank_v<Shape>>{});
        return result + 1;
    } else {
        // (shape - 1) * stride + 1 for positive strides
        return (shape - 1) * stride + 1;
    }
}

}  // namespace detail

template <class S, class D>
MOSAIC_HOST_DEVICE constexpr auto cosize(Layout<S, D> const& layout) {
    return detail::cosize_impl(layout.shape(), layout.stride());
}

// ===================================================================
// idx2crd for Layout — convert flat index to coordinate
// ===================================================================

template <class S, class D, class Idx>
MOSAIC_HOST_DEVICE constexpr auto idx2crd(Idx const& idx, Layout<S, D> const& layout) {
    return idx2crd(idx, layout.shape());
}

// ===================================================================
// slice for Layout — apply slice() pairwise to shape and stride.
//
// slice(make_coord(_, k), L) → sub-layout for column k.
// The inner-tuple slice() (from slice.hpp) does the heavy lifting.
// ===================================================================

template <class Coord, class S, class D>
MOSAIC_HOST_DEVICE constexpr auto slice(Coord const& coord, Layout<S, D> const& layout) {
    return make_layout(slice(coord, layout.shape()), slice(coord, layout.stride()));
}

// ===================================================================
// Printing — host-only. Format: (shape):(stride), recursive into nested tuples.
//
// Example output:
//   (4,8):(1,4)              — flat 2D layout
//   ((2,2),(3,3)):((1,4),(2,8)) — hierarchical 2-level layout
// ===================================================================

namespace detail {

// Print an IntTuple-shaped value: scalars print directly, tuples wrap in (..).
template <class T>
void print_tuple(std::ostream& os, T const& t) {
    if constexpr (is_tuple_v<T>) {
        os << "(";
        std::apply(
            [&os](auto const&... args) {
                std::size_t i = 0;
                ((os << (i++ == 0 ? "" : ","), print_tuple(os, args)), ...);
            },
            t);
        os << ")";
    } else {
        os << t;
    }
}

}  // namespace detail

template <class S, class D>
std::ostream& operator<<(std::ostream& os, Layout<S, D> const& l) {
    detail::print_tuple(os, l.shape());
    os << ":";
    detail::print_tuple(os, l.stride());
    return os;
}

// CuTe-style free function: print(layout) → stdout. Defers to operator<<.
template <class S, class D>
void print(Layout<S, D> const& l) {
    // Inline the operator<< logic so we don't need <iostream> for std::cout
    // here — caller decides how to flush. Most users will want std::cout << l.
    std::cerr << l;
}

}  // namespace mosaic
