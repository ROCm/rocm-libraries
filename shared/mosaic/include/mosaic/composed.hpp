// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <cstddef>

#include "mosaic/layout.hpp"

namespace mosaic {

// ===================================================================
// LayoutLike concept
//
// Any type with size() and operator()(idx). Satisfied by:
//   - Layout<Shape, Stride>  (affine, zero-overhead)
//   - FunctionLayout<F>      (wraps any callable)
//   - ComposedLayout<A, B>   (composition of two layouts)
// ===================================================================

template <class L>
concept LayoutLike = requires(L const& l, std::size_t idx) {
    { l.size() } -> std::convertible_to<std::size_t>;
    { l(idx) };
};

// ===================================================================
// FunctionLayout<F> — wraps any callable as a layout
//
// For non-linear orderings (gilbert, snake, chiplet_transform, etc.)
// that can't be expressed as (Shape, Stride) pairs.
//
// Usage:
//   auto gilbert = make_function_layout(gilbert_fn, M * N);
//   auto composed = compose(inner_tile, gilbert);
// ===================================================================

template <class F>
struct FunctionLayout {
    F fn_;
    std::size_t size_;

    MOSAIC_HOST_DEVICE constexpr FunctionLayout(F fn, std::size_t sz) : fn_(fn), size_(sz) {}

    MOSAIC_HOST_DEVICE constexpr auto operator()(std::size_t idx) const {
        return fn_(idx);
    }

    MOSAIC_HOST_DEVICE constexpr std::size_t size() const {
        return size_;
    }
};

template <class F>
MOSAIC_HOST_DEVICE constexpr auto make_function_layout(F&& fn, std::size_t sz) {
    return FunctionLayout<std::decay_t<F>>{std::forward<F>(fn), sz};
}

// With static size
template <class F, auto N>
MOSAIC_HOST_DEVICE constexpr auto make_function_layout(F&& fn, C<N>) {
    return FunctionLayout<std::decay_t<F>>{std::forward<F>(fn), N};
}

// ===================================================================
// ComposedLayout<Outer, Inner>
//
// Composes two LayoutLike types: result(idx) = outer(inner(idx))
// Used when algebraic composition isn't possible (e.g., composing
// a FunctionLayout with a Layout).
// ===================================================================

template <class Outer, class Inner>
struct ComposedLayout {
    Outer outer_;
    Inner inner_;

    MOSAIC_HOST_DEVICE constexpr ComposedLayout(Outer const& outer, Inner const& inner)
        : outer_(outer), inner_(inner) {}

    MOSAIC_HOST_DEVICE constexpr auto operator()(std::size_t idx) const {
        return outer_(static_cast<std::size_t>(inner_(idx)));
    }

    // Variadic coordinate forwarding: result(c0, c1, ...) = outer(inner(c0, c1, ...)).
    // Lets a ComposedLayout wrapping a multi-mode Layout be called with native
    // multi-dim coords (e.g. swizzled_arena_tensor(die, offset)).
    template <class C0, class C1, class... Cs>
    MOSAIC_HOST_DEVICE constexpr auto operator()(C0 c0, C1 c1, Cs... cs) const {
        return outer_(static_cast<std::size_t>(inner_(c0, c1, cs...)));
    }

    template <class... CArgs>
    MOSAIC_HOST_DEVICE constexpr auto operator()(std::tuple<CArgs...> const& coord) const {
        return outer_(static_cast<std::size_t>(inner_(coord)));
    }

    MOSAIC_HOST_DEVICE constexpr std::size_t size() const {
        return static_cast<std::size_t>(inner_.size());
    }

    MOSAIC_HOST_DEVICE constexpr Outer const& outer() const {
        return outer_;
    }
    MOSAIC_HOST_DEVICE constexpr Inner const& inner() const {
        return inner_;
    }
};

// compose() — generic composition of two LayoutLike types
// Returns a ComposedLayout (runtime composition, no algebraic simplification).
// For algebraic composition of Layout<S,D> types, use composition() from algebra.hpp.
template <class Outer, class Inner>
MOSAIC_HOST_DEVICE constexpr auto compose(Outer const& outer, Inner const& inner) {
    return ComposedLayout<Outer, Inner>{outer, inner};
}

// cosize(ComposedLayout) — for our swizzle ∘ base patterns, the outer
// (Swizzle / RotateSwizzle) is a bijection over its block, so cosize of
// the composition equals cosize of the inner layout. Lets kernels do
// `__shared__ T smem[cosize(swizzled_layout)]` without unwrapping by
// hand. ADL via the layout argument.
template <class Outer, class Inner>
MOSAIC_HOST_DEVICE constexpr auto cosize(ComposedLayout<Outer, Inner> const& cl) {
    return cosize(cl.inner());
}

}  // namespace mosaic
