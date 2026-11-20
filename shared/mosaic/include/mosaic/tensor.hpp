// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <tuple>
#include <type_traits>

#include "mosaic/composed.hpp"
#include "mosaic/layout.hpp"

namespace mosaic {

// Trait: true only for the affine Layout<Shape, Stride>.
// Used to disambiguate make_tensor(engine, Layout) from make_tensor(engine, LayoutLike).
template <class T>
struct is_affine_layout : std::false_type {};
template <class S, class D>
struct is_affine_layout<Layout<S, D>> : std::true_type {};
template <class T>
inline constexpr bool is_affine_layout_v = is_affine_layout<T>::value;

// ===================================================================
// Tensor<Engine, Layout> — accessor ∘ layout
//
// A tensor is the composition of an accessor (engine) with a layout.
// The layout maps a coordinate to an offset; the engine is offset
// and dereferenced to yield the value:
//
//   T(c) = (e ∘ L)(c) = e[L(c)]
//
// Anything supporting `e[d]` works.
// Raw pointers (T*) satisfy this directly.
// ===================================================================

template <class Engine, class Layout>
struct Tensor {
    [[no_unique_address]] Engine engine_;
    [[no_unique_address]] Layout layout_;

    using value_type = std::remove_reference_t<decltype(std::declval<Engine>()[0])>;

    MOSAIC_HOST_DEVICE constexpr Tensor() = default;

    MOSAIC_HOST_DEVICE constexpr Tensor(Engine const& engine, Layout const& layout)
        : engine_(engine), layout_(layout) {}

    // ---------------------------------------------------------------
    // Accessors
    // ---------------------------------------------------------------

    MOSAIC_HOST_DEVICE constexpr auto const& engine() const {
        return engine_;
    }
    MOSAIC_HOST_DEVICE constexpr auto const& layout() const {
        return layout_;
    }
    MOSAIC_HOST_DEVICE constexpr auto const& data() const {
        return engine_;
    }
    MOSAIC_HOST_DEVICE constexpr auto size() const {
        return layout_.size();
    }

    // ---------------------------------------------------------------
    // Evaluation: T(c) = engine[layout(c)]
    //
    // Mirrors Layout's three operator() overloads:
    //   t(i, j, k)            — variadic coordinates
    //   t(flat_idx)           — single integral (subsumed by variadic)
    //   t(make_coord(i, j))   — pre-built tuple coordinate
    // ---------------------------------------------------------------

    // Variadic coordinates. If any arg is `_`, dispatch to slicing; otherwise
    // compute an integer offset and dereference the engine.
    template <class C0, class... Cs>
    MOSAIC_HOST_DEVICE constexpr decltype(auto) operator()(C0 c0, Cs... cs) {
        if constexpr (is_underscore_v<C0> || (is_underscore_v<Cs> || ...)) {
            return slice_tensor(make_coord(c0, cs...));
        } else {
            return engine_[layout_(c0, cs...)];
        }
    }
    template <class C0, class... Cs>
    MOSAIC_HOST_DEVICE constexpr decltype(auto) operator()(C0 c0, Cs... cs) const {
        if constexpr (is_underscore_v<C0> || (is_underscore_v<Cs> || ...)) {
            return slice_tensor(make_coord(c0, cs...));
        } else {
            return engine_[layout_(c0, cs...)];
        }
    }

    // Tuple coordinate. Same dispatch as above, with the same const/non-const
    // pair so `t(coord) = x` works for value-type engines.
    template <class... CArgs>
    MOSAIC_HOST_DEVICE constexpr decltype(auto) operator()(std::tuple<CArgs...> const& coord) {
        if constexpr (has_underscore_v<std::tuple<CArgs...>>) {
            return slice_tensor(coord);
        } else {
            return engine_[layout_(coord)];
        }
    }
    template <class... CArgs>
    MOSAIC_HOST_DEVICE constexpr decltype(auto) operator()(
        std::tuple<CArgs...> const& coord) const {
        if constexpr (has_underscore_v<std::tuple<CArgs...>>) {
            return slice_tensor(coord);
        } else {
            return engine_[layout_(coord)];
        }
    }

   private:
    // Helper: build a sub-Tensor with the engine shifted to absorb the bound
    // coords' offsets. Underscore IS Int<0>, so crd2idx naturally treats `_`
    // as 0 — the bound coords contribute their offsets, the underscored ones
    // contribute nothing.
    template <class Coord>
    MOSAIC_HOST_DEVICE constexpr auto slice_tensor(Coord const& coord) const {
        auto base = crd2idx(coord, layout_.shape(), layout_.stride());
        auto sub_layout = slice(coord, layout_);
        return Tensor<Engine, decltype(sub_layout)>{engine_ + base, sub_layout};
    }

   public:
    // operator[] — paper §2.5 convenience: e[d] ≡ *(e + d)
    template <class Coord>
    MOSAIC_HOST_DEVICE constexpr decltype(auto) operator[](Coord const& coord) const {
        return engine_[layout_(coord)];
    }
};

// ===================================================================
// make_tensor — factory functions
// ===================================================================

// Engine + Layout
template <class Engine, class Shape, class Stride>
MOSAIC_HOST_DEVICE constexpr auto make_tensor(Engine&& engine,
                                              Layout<Shape, Stride> const& layout) {
    using E = std::decay_t<Engine>;
    E e = engine;  // force array-to-pointer / function-to-pointer decay
    return Tensor<E, Layout<Shape, Stride>>{e, layout};
}

// Engine + Shape (default row-major stride)
template <class Engine, class Shape>
MOSAIC_HOST_DEVICE constexpr auto make_tensor(Engine&& engine, Shape const& shape) {
    using E = std::decay_t<Engine>;
    E e = engine;  // force array-to-pointer / function-to-pointer decay
    auto layout = make_layout(shape);
    return Tensor<E, decltype(layout)>{e, layout};
}

// Engine + Shape + Stride
template <class Engine, class Shape, class Stride>
MOSAIC_HOST_DEVICE constexpr auto make_tensor(Engine&& engine, Shape const& shape,
                                              Stride const& stride) {
    using E = std::decay_t<Engine>;
    E e = engine;  // force array-to-pointer / function-to-pointer decay
    auto layout = make_layout(shape, stride);
    return Tensor<E, decltype(layout)>{e, layout};
}

// Engine + arbitrary LayoutLike (ComposedLayout, FunctionLayout, ...).
// Excludes the affine Layout<S,D> to keep the dedicated overload above unambiguous.
template <class Engine, class L>
    requires LayoutLike<L> && (!is_affine_layout_v<std::decay_t<L>>)
MOSAIC_HOST_DEVICE constexpr auto make_tensor(Engine&& engine, L const& layout) {
    using E = std::decay_t<Engine>;
    E e = engine;
    return Tensor<E, std::decay_t<L>>{e, layout};
}

}  // namespace mosaic
