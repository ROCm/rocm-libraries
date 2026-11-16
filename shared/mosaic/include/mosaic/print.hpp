// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdio>

#include "mosaic/composed.hpp"
#include "mosaic/layout.hpp"

namespace mosaic {

// ===================================================================
// Debug printing for layouts
// ===================================================================

namespace detail {

// Print a scalar value
template <class T>
void print_value(T const& val) {
    if constexpr (is_c_v<T>) {
        std::printf("%lld", static_cast<long long>(T::value));
    } else {
        std::printf("%lld", static_cast<long long>(val));
    }
}

// Print a tuple recursively
template <class T>
void print_tuple(T const& t) {
    if constexpr (is_tuple_v<T>) {
        std::printf("(");
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((I > 0 ? (std::printf(","), 0) : 0, print_tuple(std::get<I>(t))), ...);
        }(std::make_index_sequence<rank_v<T>>{});
        std::printf(")");
    } else {
        print_value(t);
    }
}

}  // namespace detail

// Print a Layout as "shape : stride"
template <class S, class D>
void print_layout(Layout<S, D> const& layout) {
    detail::print_tuple(layout.shape());
    std::printf(":");
    detail::print_tuple(layout.stride());
}

// Print a Layout with a label
template <class S, class D>
void print_layout(const char* label, Layout<S, D> const& layout) {
    std::printf("%s = ", label);
    print_layout(layout);
    std::printf("  (size=%lld)\n", static_cast<long long>(layout.size()));
}

// Print the full mapping table: index → offset
template <class S, class D>
void print_layout_table(Layout<S, D> const& layout) {
    auto sz = static_cast<std::size_t>(layout.size());
    for (std::size_t i = 0; i < sz; ++i) {
        auto offset = static_cast<std::size_t>(layout(i));
        std::printf("  %3zu → %3zu\n", i, offset);
    }
}

// Print a 2D layout as a grid (for rank-2 layouts)
template <class S, class D>
void print_layout_2d(Layout<S, D> const& layout) {
    if constexpr (is_tuple_v<S> && rank_v<S> == 2) {
        auto rows = static_cast<std::size_t>(layout.template shape_at<0>());
        auto cols = static_cast<std::size_t>(layout.template shape_at<1>());

        // Header
        std::printf("    ");
        for (std::size_t c = 0; c < cols; ++c) std::printf("%4zu", c);
        std::printf("\n    ");
        for (std::size_t c = 0; c < cols; ++c) std::printf("----");
        std::printf("\n");

        // Body
        for (std::size_t r = 0; r < rows; ++r) {
            std::printf("%3zu|", r);
            for (std::size_t c = 0; c < cols; ++c) {
                auto offset = static_cast<std::size_t>(layout(r, c));
                std::printf("%4zu", offset);
            }
            std::printf("\n");
        }
    } else {
        std::printf("(print_layout_2d: layout is not rank 2)\n");
    }
}

// Print a FunctionLayout
template <class F>
void print_layout(const char* label, FunctionLayout<F> const& layout) {
    std::printf("%s = FunctionLayout(size=%zu)\n", label, layout.size());
}

// Print a ComposedLayout
template <class A, class B>
void print_layout(const char* label, ComposedLayout<A, B> const& layout) {
    std::printf("%s = ComposedLayout(size=%zu)\n", label, layout.size());
}

}  // namespace mosaic
