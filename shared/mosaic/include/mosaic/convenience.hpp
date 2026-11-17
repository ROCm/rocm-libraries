// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "mosaic/algebra.hpp"

namespace mosaic {

// ===================================================================
// Convenience factories — Mosaic-style API that produces Layout<S,D>
// ===================================================================

// ---------------------------------------------------------------------------
// row_major(M, N, ...) — rightmost dimension varies fastest
//   row_major(M, N) → Layout<Shape<M,N>, Stride<N, 1>>
//   row_major(_4, _8) → sizeof == 0 (fully static)
// ---------------------------------------------------------------------------

template <class... Dims>
    requires(sizeof...(Dims) >= 1)
MOSAIC_HOST_DEVICE constexpr auto row_major(Dims... dims) {
    auto shape = make_shape(dims...);
    auto stride = compact_row_major(shape);
    return Layout<decltype(shape), decltype(stride)>{shape, stride};
}

// ---------------------------------------------------------------------------
// column_major(M, N, ...) — leftmost dimension varies fastest
//   column_major(M, N) → Layout<Shape<M,N>, Stride<1, M>>
//   column_major(_4, _8) → sizeof == 0 (fully static)
// ---------------------------------------------------------------------------

template <class... Dims>
    requires(sizeof...(Dims) >= 1)
MOSAIC_HOST_DEVICE constexpr auto column_major(Dims... dims) {
    auto shape = make_shape(dims...);
    auto stride = compact_col_major(shape);
    return Layout<decltype(shape), decltype(stride)>{shape, stride};
}

// ---------------------------------------------------------------------------
// strided<S0, S1, ...>(M, N, ...) — explicit compile-time strides
//   strided<N, 1>(M, N)  → Layout<Shape<M,N>, Stride<C<N>, C<1>>>
//   strided<8, 1>(M, N)  → stride-8 in first dim, contiguous in second
// ---------------------------------------------------------------------------

template <auto... Strides>
struct strided_t {
    template <class... Dims>
        requires(sizeof...(Dims) == sizeof...(Strides))
    MOSAIC_HOST_DEVICE constexpr auto operator()(Dims... dims) const {
        auto shape = make_shape(dims...);
        auto stride = make_stride(C<Strides>{}...);
        return Layout<decltype(shape), decltype(stride)>{shape, stride};
    }
};

template <auto... Strides>
inline constexpr strided_t<Strides...> strided{};

// ---------------------------------------------------------------------------
// make_hierarchical(outer, ..., inner) — Mosaic v1 exact enumeration
//
// Produces a Layout with the IDENTICAL flat-index mapping as Mosaic v1's
// make_layout(outer, ..., inner). Arguments are outermost-first.
//
// The key difference from blocked_product: the flat index is decomposed
// level-by-level (innermost first), with each level's ordering determining
// the dimension traversal order within that level:
//   - row_major: rightmost dim first  (dim R-1, R-2, ..., 0)
//   - column_major: leftmost dim first (dim 0, 1, ..., R-1)
//
// The result is a standard Layout<Shape, Stride> — same O(rank) flat-index
// evaluation, same O(1) coordinate access. No per-evaluation level traversal.
//
// Example:
//   make_hierarchical(row_major(2,2), column_major(4,4))
//   produces shape (4,4,2,2) : stride (8,1,4,32)
//   which matches Mosaic v1's make_layout(rm(2,2), cm(4,4)) exactly.
// ---------------------------------------------------------------------------

namespace detail_hier {

// Describe the dim order for a layout's ordering.
// Returns indices in the order they should be decomposed.
// row_major(M,N): decompose dim 1 first, then dim 0 → {1, 0}
// col_major(M,N): decompose dim 0 first, then dim 1 → {0, 1}
//
// We detect ordering by checking stride pattern:
//   col-major: stride[0] < stride[1] (leftmost has smallest stride)
//   row-major: stride[0] > stride[1] (rightmost has smallest stride)

template <class S, class D>
constexpr bool is_col_major_ordering(const Layout<S, D>& L) {
    if constexpr (rank_v<S> == 1) {
        return true;
    } else {
        // Compare first and last stride
        auto d0 = std::size_t(L.template stride_at<0>());
        auto d_last = std::size_t(std::get<rank_v<S> - 1>(L.stride()));
        return d0 <= d_last;
    }
}

}  // namespace detail_hier

// 2-level hierarchical layout (the common GEMM case)
template <class S0, class D0, class S1, class D1>
    requires(rank_v<S0> == rank_v<S1>)
constexpr auto make_hierarchical(Layout<S0, D0> const& outer, Layout<S1, D1> const& inner) {
    constexpr int R = rank_v<S0>;
    static_assert(R == 2, "make_hierarchical currently supports rank-2 levels");

    // Global extents per dimension
    auto ext0 =
        std::size_t(outer.template shape_at<0>()) * std::size_t(inner.template shape_at<0>());
    auto ext1 =
        std::size_t(outer.template shape_at<1>()) * std::size_t(inner.template shape_at<1>());

    // Row-major base strides for the global coordinate
    // dim 0 stride = ext1, dim 1 stride = 1
    std::size_t base_stride[2] = {ext1, std::size_t(1)};

    // Determine dim orders for each level
    bool inner_cm = detail_hier::is_col_major_ordering(inner);
    bool outer_cm = detail_hier::is_col_major_ordering(outer);

    // Inner level's shapes per dim
    auto is0 = std::size_t(inner.template shape_at<0>());
    auto is1 = std::size_t(inner.template shape_at<1>());
    // Outer level's shapes per dim
    auto os0 = std::size_t(outer.template shape_at<0>());
    auto os1 = std::size_t(outer.template shape_at<1>());

    // Build modes: innermost level first, in its ordering's dim order
    // Then outermost level, in its ordering's dim order
    //
    // For each mode: stride = base_stride[dim] * cumulative_inner_extent[dim]
    // cumulative_inner_extent starts at 1 and gets multiplied by each level's shape per dim

    std::size_t cum_inner[2] = {1, 1};

    // Inner level modes
    std::size_t shape_arr[4], stride_arr[4];
    int mode = 0;

    if (inner_cm) {
        // col-major: dim 0 first, then dim 1
        shape_arr[mode] = is0;
        stride_arr[mode] = base_stride[0] * cum_inner[0];
        mode++;
        shape_arr[mode] = is1;
        stride_arr[mode] = base_stride[1] * cum_inner[1];
        mode++;
    } else {
        // row-major: dim 1 first, then dim 0
        shape_arr[mode] = is1;
        stride_arr[mode] = base_stride[1] * cum_inner[1];
        mode++;
        shape_arr[mode] = is0;
        stride_arr[mode] = base_stride[0] * cum_inner[0];
        mode++;
    }

    cum_inner[0] *= is0;
    cum_inner[1] *= is1;

    // Outer level modes
    if (outer_cm) {
        shape_arr[mode] = os0;
        stride_arr[mode] = base_stride[0] * cum_inner[0];
        mode++;
        shape_arr[mode] = os1;
        stride_arr[mode] = base_stride[1] * cum_inner[1];
        mode++;
    } else {
        shape_arr[mode] = os1;
        stride_arr[mode] = base_stride[1] * cum_inner[1];
        mode++;
        shape_arr[mode] = os0;
        stride_arr[mode] = base_stride[0] * cum_inner[0];
        mode++;
    }

    return make_layout(make_shape(shape_arr[0], shape_arr[1], shape_arr[2], shape_arr[3]),
                       make_stride(stride_arr[0], stride_arr[1], stride_arr[2], stride_arr[3]));
}

// 3-level hierarchical layout
template <class S0, class D0, class S1, class D1, class S2, class D2>
    requires(rank_v<S0> == rank_v<S1> && rank_v<S1> == rank_v<S2>)
constexpr auto make_hierarchical(Layout<S0, D0> const& outer, Layout<S1, D1> const& middle,
                                 Layout<S2, D2> const& inner) {
    constexpr int R = rank_v<S0>;
    static_assert(R == 2, "make_hierarchical currently supports rank-2 levels");

    auto ext0 = std::size_t(outer.template shape_at<0>()) *
                std::size_t(middle.template shape_at<0>()) *
                std::size_t(inner.template shape_at<0>());
    auto ext1 = std::size_t(outer.template shape_at<1>()) *
                std::size_t(middle.template shape_at<1>()) *
                std::size_t(inner.template shape_at<1>());

    std::size_t base_stride[2] = {ext1, std::size_t(1)};

    bool inner_cm = detail_hier::is_col_major_ordering(inner);
    bool middle_cm = detail_hier::is_col_major_ordering(middle);
    bool outer_cm = detail_hier::is_col_major_ordering(outer);

    auto is0 = std::size_t(inner.template shape_at<0>());
    auto is1 = std::size_t(inner.template shape_at<1>());
    auto ms0 = std::size_t(middle.template shape_at<0>());
    auto ms1 = std::size_t(middle.template shape_at<1>());
    auto os0 = std::size_t(outer.template shape_at<0>());
    auto os1 = std::size_t(outer.template shape_at<1>());

    std::size_t cum_inner[2] = {1, 1};
    std::size_t shape_arr[6], stride_arr[6];
    int mode = 0;

    // Inner level
    auto add_level = [&](bool cm, std::size_t s0, std::size_t s1) {
        if (cm) {
            shape_arr[mode] = s0;
            stride_arr[mode] = base_stride[0] * cum_inner[0];
            mode++;
            shape_arr[mode] = s1;
            stride_arr[mode] = base_stride[1] * cum_inner[1];
            mode++;
        } else {
            shape_arr[mode] = s1;
            stride_arr[mode] = base_stride[1] * cum_inner[1];
            mode++;
            shape_arr[mode] = s0;
            stride_arr[mode] = base_stride[0] * cum_inner[0];
            mode++;
        }
        cum_inner[0] *= s0;
        cum_inner[1] *= s1;
    };

    add_level(inner_cm, is0, is1);
    add_level(middle_cm, ms0, ms1);
    add_level(outer_cm, os0, os1);

    return make_layout(make_shape(shape_arr[0], shape_arr[1], shape_arr[2], shape_arr[3],
                                  shape_arr[4], shape_arr[5]),
                       make_stride(stride_arr[0], stride_arr[1], stride_arr[2], stride_arr[3],
                                   stride_arr[4], stride_arr[5]));
}

}  // namespace mosaic
