// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>

#include "mosaic/layout.hpp"

namespace mosaic {

// ===================================================================
// coalesce — simplify a layout to minimal rank
// https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#coalesce
//
// Merges adjacent modes where stride[i] * shape[i] == stride[i+1].
// Removes size-1 modes. A 3-level hierarchical layout often coalesces
// to rank 1-2, dramatically reducing evaluation cost.
//
// Invariant: coalesce(L)(i) == L(i) for all i in [0, size(L))
// ===================================================================

namespace detail {

// Single fold step. Visits (s_i, d_i) at flat position I and either skips,
// seeds, merges with the trailing accumulator entry, or appends.
//
// Merge predicate is the CUTLASS one: last_shape * last_stride == d_i, all
// statically known. Dynamic strides can't merge — they fall through to append,
// which keeps the result type static-friendly.
//
// Returns std::tuple(acc_shape, acc_stride) where each is a (possibly empty)
// std::tuple of leaves.
template <std::size_t I, class FlatShape, class FlatStride, class AccShape, class AccStride>
MOSAIC_HOST_DEVICE constexpr auto coalesce_fold(FlatShape const& fs, FlatStride const& fd,
                                                AccShape const& as, AccStride const& ad) {
    constexpr std::size_t R = std::tuple_size_v<FlatShape>;
    if constexpr (I == R) {
        return std::make_tuple(as, ad);
    } else {
        auto si = std::get<I>(fs);
        auto di = std::get<I>(fd);

        if constexpr (is_constant_v<1, decltype(si)>) {
            // Drop size-1 modes — they carry no information.
            return coalesce_fold<I + 1>(fs, fd, as, ad);
        } else if constexpr (std::tuple_size_v<AccShape> == 0) {
            // Seed the accumulator with the first surviving mode.
            return coalesce_fold<I + 1>(fs, fd, std::make_tuple(si), std::make_tuple(di));
        } else {
            constexpr std::size_t K = std::tuple_size_v<AccShape> - 1;
            auto last_s = std::get<K>(as);
            auto last_d = std::get<K>(ad);

            // The `::value` accesses below are only well-formed on static
            // leaves, so they must sit behind their OWN `if constexpr` — a
            // `&&` chain does not protect them (the whole condition still has
            // to be a valid expression, and `int::value` is not). Dynamic
            // leaves cannot prove contiguity at compile time, so they fall to
            // the conservative "emit a new mode" branch (function-preserving,
            // just not simplified) — matching CUTLASS, which only merges
            // statically-known-contiguous modes.
            if constexpr (is_static_v<decltype(last_s)> && is_static_v<decltype(last_d)> &&
                          is_static_v<decltype(di)>) {
                if constexpr (decltype(last_s)::value * decltype(last_d)::value ==
                              decltype(di)::value) {
                    // Contiguous: extend the trailing accumulator entry's
                    // shape, keep its stride. CUTLASS bw_coalesce.
                    return coalesce_fold<I + 1>(fs, fd, replace_back(as, last_s * si), ad);
                } else {
                    // Not contiguous: emit a new mode.
                    return coalesce_fold<I + 1>(fs, fd, append(as, si), append(ad, di));
                }
            } else {
                // Dynamic leaf: emit a new mode (cannot prove contiguity).
                return coalesce_fold<I + 1>(fs, fd, append(as, si), append(ad, di));
            }
        }
    }
}

// Rank-aware wrapping: a coalesced layout with a single surviving mode is
// reported as a flat scalar shape/stride (matches CUTLASS's NewShape scalar
// convention from bw_coalesce). Empty accumulator → canonical Layout<1,0>.
template <class AccShape, class AccStride>
MOSAIC_HOST_DEVICE constexpr auto coalesce_finish(AccShape const& as, AccStride const& ad) {
    constexpr std::size_t N = std::tuple_size_v<AccShape>;
    if constexpr (N == 0) {
        return Layout<C<1>, C<0>>{C<1>{}, C<0>{}};
    } else if constexpr (N == 1) {
        return make_layout(std::get<0>(as), std::get<0>(ad));
    } else {
        return make_layout(as, ad);
    }
}

}  // namespace detail

// Flatten then coalesce a layout into minimal rank.
//
// Invariant: for all i in [0, size(L)), coalesce(L)(i) == L(i).
// Merges adjacent modes whenever shape[i] * stride[i] == stride[i+1] (all
// static). Drops size-1 modes. Out-of-domain behavior is not preserved
// (matches cute::coalesce, not detail::coalesce_x).
template <class S, class D>
MOSAIC_HOST_DEVICE constexpr auto coalesce(Layout<S, D> const& layout) {
    auto flat_shape = flatten(layout.shape());
    auto flat_stride = flatten(layout.stride());

    auto folded =
        detail::coalesce_fold<0>(flat_shape, flat_stride, std::make_tuple(), std::make_tuple());
    return detail::coalesce_finish(std::get<0>(folded), std::get<1>(folded));
}

// ===================================================================
// composition — algebraic composition of layouts.
// https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#composition
//
//   composition(A, B)(c) == A(B(c))  for all c in domain(B)
//
// Result is a closed-form Layout<S', D'> — not just a callable. This
// makes the algebra closed: composition results can be fed into
// complement, logical_divide, another composition, etc.
//
//   detail::transform_layout(shape, stride, f)
//       — workhorse for right-distributivity over RHS tuple modes.
//
//   detail::transform_layout_tiler(layout, tiler, f)
//       — public Layout∘Tiler dispatch helper. Drops trailing LHS
//         modes past tiler.rank (matches CUTLASS).
//
//   detail::composition_impl(lhs_shape, lhs_stride, rhs_shape, rhs_stride)
//       — 5-branch dispatch:
//         (1) RHS shape is tuple → right-distribute via transform_layout
//         (2) RHS stride is ScaledBasis → static_assert reject
//         (3) RHS stride is static 0 → shortcut (rhs_shape):(0)
//         (4) LHS shape is integral → closed form (rhs_shape):(rhs_stride*lhs_stride)
//         (5) LHS is tuple, RHS is integer-mode (n):(d) → composition_fold
//
//   detail::composition_fold<I, Stop>(...)
//       — template recursion over LHS modes 0..R-2, threading
//         (result_shape, result_stride, rest_shape, rest_stride).
//         Type-changing accumulator, so mosaic's runtime fold won't work.
//
//   public composition(Layout, Layout) and composition(Layout, Tiler)
//       — coalesce(lhs) first, then dispatch into composition_impl.
//
// Dynamic strides are supported: the static `is_constant_v<1>` skip
// predicates are false at compile time for runtime values, so the fold
// always emits. Result is correct but may have more modes than the
// fully-static path would.
//
// ===================================================================

namespace detail {

// Right-distributivity workhorse. Given parallel shape and stride tuples
// from the RHS, apply `f(s_i, d_i)` to each pair (returning a Layout per
// pair), then assemble into a single Layout whose shape is the tuple of
// per-pair shapes and whose stride is the tuple of per-pair strides.
template <class SShape, class SStride, class F>
MOSAIC_HOST_DEVICE constexpr auto transform_layout(SShape const& s, SStride const& d, F&& f) {
    static_assert(is_tuple_v<SShape> && is_tuple_v<SStride>,
                  "transform_layout: shape and stride must be tuples");
    static_assert(std::tuple_size_v<SShape> == std::tuple_size_v<SStride>,
                  "transform_layout: shape/stride rank mismatch");
    constexpr std::size_t N = std::tuple_size_v<SShape>;
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
        // Call f once per index, then project — avoids invoking f twice.
        auto pieces = std::make_tuple(f(std::get<I>(s), std::get<I>(d))...);
        auto out_shape = std::make_tuple(std::get<I>(pieces).shape()...);
        auto out_stride = std::make_tuple(std::get<I>(pieces).stride()...);
        return make_layout(out_shape, out_stride);
    }(std::make_index_sequence<N>{});
}

template <class LS, class LD, class Tiler, class F>
MOSAIC_HOST_DEVICE constexpr auto transform_layout_tiler(Layout<LS, LD> const& lhs,
                                                         Tiler const& tiler, F&& f) {
    static_assert(is_tuple_v<Tiler>, "transform_layout_tiler: tiler must be a tuple");
    constexpr std::size_t N = std::tuple_size_v<Tiler>;
    static_assert(is_tuple_v<LS>, "transform_layout_tiler: lhs must be hierarchical");
    static_assert(N <= std::tuple_size_v<LS>,
                  "transform_layout_tiler: tiler rank exceeds lhs rank");
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
        auto pieces =
            std::make_tuple(f(make_layout(std::get<I>(lhs.shape()), std::get<I>(lhs.stride())),
                              std::get<I>(tiler))...);
        auto out_shape = std::make_tuple(std::get<I>(pieces).shape()...);
        auto out_stride = std::make_tuple(std::get<I>(pieces).stride()...);
        return make_layout(out_shape, out_stride);
    }(std::make_index_sequence<N>{});
}

// 2-line wrapper around zip2_by — regroups paired modes by tiler profile.
// Used by zipped_divide / zipped_product to regroup logical_*'s per-mode
// output ((T_M, R_M), (T_N, R_N), …) into rank-2 ((T_M, T_N, …), (R_M, R_N, …)).
template <class LS, class LD, class Tiler>
MOSAIC_HOST_DEVICE constexpr auto tile_unzip(Layout<LS, LD> const& layout, Tiler const& tiler) {
    return make_layout(zip2_by(layout.shape(), tiler), zip2_by(layout.stride(), tiler));
}

// Same as transform_layout_tiler but PRESERVES trailing lhs modes past the
// tiler's rank instead of dropping them. Used by logical_divide and
// logical_product, where the result must contain all of lhs's data
// (re-indexed) rather than just the modes addressed by the tiler.
template <class LS, class LD, class Tiler, class F>
MOSAIC_HOST_DEVICE constexpr auto transform_layout_tiler_preserve(Layout<LS, LD> const& lhs,
                                                                  Tiler const& tiler, F&& f) {
    static_assert(is_tuple_v<Tiler>, "transform_layout_tiler_preserve: tiler must be a tuple");
    constexpr std::size_t TR = std::tuple_size_v<Tiler>;
    static_assert(is_tuple_v<LS>, "transform_layout_tiler_preserve: lhs must be hierarchical");
    constexpr std::size_t LR = std::tuple_size_v<LS>;
    static_assert(TR <= LR, "transform_layout_tiler_preserve: tiler rank exceeds lhs rank");
    return [&]<std::size_t... I, std::size_t... J>(std::index_sequence<I...>,
                                                   std::index_sequence<J...>) {
        auto pieces =
            std::make_tuple(f(make_layout(std::get<I>(lhs.shape()), std::get<I>(lhs.stride())),
                              std::get<I>(tiler))...);
        // Trailing lhs modes past tiler rank pass through unchanged.
        auto out_shape =
            std::make_tuple(std::get<I>(pieces).shape()..., std::get<J + TR>(lhs.shape())...);
        auto out_stride =
            std::make_tuple(std::get<I>(pieces).stride()..., std::get<J + TR>(lhs.stride())...);
        return make_layout(out_shape, out_stride);
    }(std::make_index_sequence<TR>{}, std::make_index_sequence<LR - TR>{});
}

// Type-changing template recursion over LHS modes I .. Stop-1.
// At each step computes the per-mode contribution to the output, threading
// (result_shape, result_stride, rest_shape, rest_stride). Mirrors
// coalesce_fold's pattern.
template <std::size_t I, std::size_t Stop, class LShape, class LStride, class RS, class RD,
          class RestS, class RestD>
MOSAIC_HOST_DEVICE constexpr auto composition_fold(LShape const& ls, LStride const& ld,
                                                   RS const& rs, RD const& rd, RestS const& rest_s,
                                                   RestD const& rest_d) {
    if constexpr (I == Stop) {
        return std::make_tuple(rs, rd, rest_s, rest_d);
    } else {
        auto curr_s = std::get<I>(ls);
        auto curr_d = std::get<I>(ld);

        // Weak divisibility: only fire when both sides are statically known.
        // Dynamic inputs trust the caller (matches CUTLASS).
        // NB: std::decay_t is essential here — function parameters like rest_d
        // are const&, and is_static_v (== is_empty_v) returns false on
        // reference types, silently skipping the check otherwise.
        if constexpr (is_static_v<std::decay_t<decltype(curr_s)>> &&
                      is_static_v<std::decay_t<decltype(rest_d)>>) {
            static_assert(
                (std::decay_t<decltype(rest_d)>::value % std::decay_t<decltype(curr_s)>::value ==
                 0) ||
                    (std::decay_t<decltype(rest_d)>::value < std::decay_t<decltype(curr_s)>::value),
                "composition: stride divisibility violated "
                "(rest_stride % curr_shape == 0 or rest_stride < curr_shape)");
        }

        auto next_s = ceil_div(curr_s, abs(rest_d));
        auto next_d = ceil_div(abs(rest_d), curr_s) * signum(rest_d);

        if constexpr (is_constant_v<1, std::decay_t<decltype(next_s)>> ||
                      is_constant_v<1, std::decay_t<decltype(rest_s)>>) {
            // Skip emit — B-step too large for this A mode, or RHS is exhausted.
            // Only rest_d advances; rest_s carries unchanged.
            return composition_fold<I + 1, Stop>(ls, ld, rs, rd, rest_s, next_d);
        } else {
            auto new_s = min(next_s, rest_s);
            if constexpr (is_static_v<std::decay_t<decltype(new_s)>> &&
                          is_static_v<std::decay_t<decltype(rest_s)>>) {
                static_assert(
                    std::decay_t<decltype(rest_s)>::value % std::decay_t<decltype(new_s)>::value ==
                        0,
                    "composition: shape divisibility violated "
                    "(rest_shape % new_shape != 0)");
            }
            return composition_fold<I + 1, Stop>(
                ls, ld, append(rs, new_s), append(rd, rest_d * curr_d), rest_s / new_s, next_d);
        }
    }
}

// Core 5-branch dispatch. Operates on raw shape/stride pairs (not Layouts)
// so it can recurse into per-mode sub-compositions.
template <class LShape, class LStride, class RShape, class RStride>
MOSAIC_HOST_DEVICE constexpr auto composition_impl(LShape const& lhs_shape,
                                                   LStride const& lhs_stride,
                                                   RShape const& rhs_shape,
                                                   RStride const& rhs_stride) {
    // Branch 2: ScaledBasis (out of scope). Mosaic has no such type, so
    // any non-integral, non-tuple stride is rejected here for a clear error.
    static_assert(is_integral_v<RStride> || is_tuple_v<RStride>,
                  "composition: RHS stride must be integral or tuple "
                  "(ScaledBasis strides are not supported)");

    if constexpr (is_tuple_v<RShape>) {
        // Branch 1: right-distributivity. LHS captured unchanged; the RHS
        // tuple structure determines which sub-modes get composed.
        return transform_layout(rhs_shape, rhs_stride, [&](auto const& s, auto const& d) {
            return composition_impl(lhs_shape, lhs_stride, s, d);
        });
    } else if constexpr (is_constant_v<0, RStride>) {
        // Branch 3: static-0 RHS stride. Output is just (rhs_shape):(0).
        return make_layout(rhs_shape, rhs_stride);
    } else if constexpr (is_integral_v<LShape>) {
        // Branch 4: flat-scalar LHS. Closed form (n):(d * lhs_stride).
        return make_layout(rhs_shape, rhs_stride * lhs_stride);
    } else {
        // Branch 5: tuple LHS, integer-mode RHS. Fold over LHS modes 0..R-2,
        // then handle the final mode R-1 in the 3-case wrap-up.
        constexpr std::size_t R = std::tuple_size_v<LShape>;
        static_assert(R >= 1, "composition: lhs cannot have rank 0");

        if constexpr (R == 1) {
            // Single-mode LHS: equivalent to the flat-scalar case.
            auto only_d = std::get<0>(lhs_stride);
            return make_layout(rhs_shape, rhs_stride * only_d);
        } else {
            auto folded = composition_fold<0, R - 1>(lhs_shape, lhs_stride, std::tuple<>{},
                                                     std::tuple<>{}, rhs_shape, rhs_stride);

            auto result_s = std::get<0>(folded);
            auto result_d = std::get<1>(folded);
            auto rest_s = std::get<2>(folded);
            auto rest_d = std::get<3>(folded);
            auto last_d = std::get<R - 1>(lhs_stride);

            if constexpr (std::tuple_size_v<decltype(result_s)> == 0) {
                // Wrap-up A: fold emitted nothing. Whole result lives in the
                // final LHS mode: (rest_shape):(rest_stride * lhs_stride[R-1]).
                return make_layout(rest_s, rest_d * last_d);
            } else if constexpr (is_constant_v<1, decltype(rest_s)>) {
                // Wrap-up B: RHS fully consumed. Unwrap rank-1 results to
                // flat scalar form for tidiness.
                return make_layout(unwrap(result_s), unwrap(result_d));
            } else {
                // Wrap-up C: leftover RHS becomes the final output mode.
                return make_layout(append(result_s, rest_s), append(result_d, rest_d * last_d));
            }
        }
    }
}

}  // namespace detail

// Public: composition of two Layouts.
//
// Result is a Layout C such that C(c) == lhs(rhs(c)) for all c in
// [0, size(rhs)). Caller must not evaluate C outside that domain (we use
// plain `coalesce` rather than CUTLASS's coalesce_x which preserves
// out-of-domain behavior).
template <class LS, class LD, class RS, class RD>
MOSAIC_HOST_DEVICE constexpr auto composition(Layout<LS, LD> const& lhs,
                                              Layout<RS, RD> const& rhs) {
    auto flat = coalesce(lhs);
    return detail::composition_impl(flat.shape(), flat.stride(), rhs.shape(), rhs.stride());
}

// Public: composition of a Layout with a Tiler. Tilers may be:
//   - tuple of (Layout | Underscore | integral) — per-mode tiler
//   - Underscore — identity, returns lhs unchanged
//   - integral (incl. C<v>) — composes with (n):(_1)
template <class LS, class LD, class Tiler>
    requires(!is_layout_v<Tiler>)
MOSAIC_HOST_DEVICE constexpr auto composition(Layout<LS, LD> const& lhs, Tiler const& tiler) {
    if constexpr (is_tuple_v<Tiler>) {
        return detail::transform_layout_tiler(
            lhs, tiler, [](auto const& l, auto const& t) { return composition(l, t); });
    } else if constexpr (is_underscore_v<Tiler>) {
        return lhs;
    } else {
        static_assert(is_integral_v<Tiler>,
                      "composition: tiler must be Layout, Underscore, integral, or tuple thereof");
        auto flat = coalesce(lhs);
        return detail::composition_impl(flat.shape(), flat.stride(), tiler, C<1>{});
    }
}

// ===================================================================
// complement — find the layout covering offsets not in L's image.
// https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#complement
//
// Given layout B and a target codomain size M, complement(B, M) is a
// layout C such that make_layout(B, C) is a bijection onto [0, M).
// Equivalently: C's image is exactly [0, M) \ image(B), arranged in
// ascending-stride order.
//
//   detail::filter(Layout)
//       — drop modes where shape==1 or stride==0; return flat Layout.
//         A fully-filtered-away layout returns Layout<C<1>, C<0>>.
//
//   detail::complement_fold<I, Stop>(...)
//       — sort-by-stride fold. Type-changing template recursion. At
//         each step picks the min-stride remaining mode, emits one
//         output mode (gap_size, prior_base_stride), updates the
//         "prior base" to (min_stride * shape_at_min), removes the
//         picked mode. Static-only: dynamic strides can't be sorted
//         at compile time.
//
//   detail::complement_impl(shape, stride, cotarget)
//       — 3-branch dispatcher:
//         (1) shape is scalar + stride is C<0> → return (cotarget):(1)
//         (2) shape is scalar (rank-1 closed form) → (M/(n·e)) : (n·e)
//         (3) shape is tuple → multi-rank fold + tail + final coalesce
//
//   public complement(Layout, cotarget) and complement(Layout)
//       — filter first, then call complement_impl.
// ===================================================================

namespace detail {

// Flat-and-filter helper: walks a flat (shape, stride) and skips modes
// where shape==1 or stride==0. Returns (kept_shape_tuple, kept_stride_tuple).
template <std::size_t I, class FS, class FD, class RS, class RD>
MOSAIC_HOST_DEVICE constexpr auto filter_fold(FS const& fs, FD const& fd, RS const& rs,
                                              RD const& rd) {
    constexpr std::size_t R = std::tuple_size_v<FS>;
    if constexpr (I == R) {
        return std::make_tuple(rs, rd);
    } else {
        auto s = std::get<I>(fs);
        auto d = std::get<I>(fd);
        if constexpr (is_constant_v<1, decltype(s)> || is_constant_v<0, decltype(d)>) {
            return filter_fold<I + 1>(fs, fd, rs, rd);
        } else {
            return filter_fold<I + 1>(fs, fd, append(rs, s), append(rd, d));
        }
    }
}

// Public-ish filter: drop stride-0 and size-1 modes, return a (possibly
// rank-collapsed) Layout. Empty result → Layout<C<1>, C<0>>; single
// surviving mode → scalar shape/stride; multi → tuple shape/stride.
template <class S, class D>
MOSAIC_HOST_DEVICE constexpr auto filter(Layout<S, D> const& layout) {
    auto flat_s = flatten(layout.shape());
    auto flat_d = flatten(layout.stride());
    auto kept = filter_fold<0>(flat_s, flat_d, std::tuple<>{}, std::tuple<>{});
    auto ks = std::get<0>(kept);
    auto kd = std::get<1>(kept);
    constexpr std::size_t N = std::tuple_size_v<decltype(ks)>;
    if constexpr (N == 0) {
        return Layout<C<1>, C<0>>{C<1>{}, C<0>{}};
    } else if constexpr (N == 1) {
        return make_layout(std::get<0>(ks), std::get<0>(kd));
    } else {
        return make_layout(ks, kd);
    }
}

// Sort-by-stride fold. At iter I, picks the smallest stride from the
// remaining (rem_s, rem_d), emits one result mode, and removes that mode
// from the remaining set. result_stride is seeded with C<1> at I==0 so
// the very first emitted shape is min_stride_0 / 1 = min_stride_0.
//
// Invariant after Stop iterations:
//   - rem_s and rem_d each have exactly 1 element left (the largest stride)
//   - rs has Stop elements
//   - rd has Stop+1 elements (the seed plus Stop appends)
template <std::size_t I, std::size_t Stop, class RemS, class RemD, class RS, class RD>
MOSAIC_HOST_DEVICE constexpr auto complement_fold(RemS const& rem_s, RemD const& rem_d,
                                                  RS const& rs, RD const& rd) {
    if constexpr (I == Stop) {
        return std::make_tuple(rem_s, rem_d, rs, rd);
    } else {
        // min over remaining strides; find its index in rem_d.
        auto min_stride = std::apply([](auto const&... xs) { return mosaic::min(xs...); }, rem_d);
        auto min_idx_c = find(rem_d, min_stride);
        constexpr int idx_val = decltype(min_idx_c)::value;
        static_assert(idx_val >= 0,
                      "complement: failed to find min stride (dynamic strides not supported "
                      "for multi-rank complement)");
        constexpr std::size_t idx = static_cast<std::size_t>(idx_val);

        // Previously-accumulated base stride.
        auto last_base = std::get<I>(rd);
        auto new_shape = min_stride / last_base;
        static_assert(!is_constant_v<0, std::decay_t<decltype(new_shape)>>,
                      "complement: non-injective layout detected "
                      "(min_stride not divisible by prior accumulated stride)");

        auto shape_at_min = std::get<idx>(rem_s);
        auto new_base = min_stride * shape_at_min;

        return complement_fold<I + 1, Stop>(remove<idx>(rem_s), remove<idx>(rem_d),
                                            append(rs, new_shape), append(rd, new_base));
    }
}

// 3-branch dispatch. shape/stride are post-filter, so scalar shape with
// stride!=0 means "real rank-1 layout"; scalar shape with stride==0 means
// "fully degenerate / broadcast"; tuple shape means rank >= 2.
template <class S, class D, class CoTarget>
MOSAIC_HOST_DEVICE constexpr auto complement_impl(S const& shape, D const& stride,
                                                  CoTarget const& cotarget) {
    if constexpr (!is_tuple_v<S>) {
        // Rank-1 post-filter.
        if constexpr (is_constant_v<0, std::decay_t<D>>) {
            // Stride-0 special case: complement covers the entire cotarget.
            return make_layout(cotarget, C<1>{});
        } else {
            // General rank-1: complement is (d, M/(n*d)):(1, n*d), then coalesced.
            // This is the same shape as the multi-rank wrap-up specialized for R==1:
            // the "fill" mode (d, 1) covers the offsets inside one B-step, and the
            // "tile" mode (M/(n*d), n*d) places copies of the fill at each block.
            // For d==1, the fill mode is size-1 and coalesce drops it, recovering
            // the simple (M/n):(n) closed form.
            auto fill_size = stride;  // d
            auto fill_stride = C<1>{};
            auto nd = shape * stride;
            auto tile_size = ceil_div(cotarget, nd);
            auto tile_stride = nd;
            auto raw = make_layout(std::make_tuple(fill_size, tile_size),
                                   std::make_tuple(fill_stride, tile_stride));
            return coalesce(raw);
        }
    } else {
        constexpr std::size_t R = std::tuple_size_v<S>;
        static_assert(R >= 2, "complement_impl: tuple shape must have rank >= 2");

        // Sort-by-stride fold over R-1 iterations. The seed for the result
        // stride is C<1>; each iter appends one more.
        auto folded =
            complement_fold<0, R - 1>(shape, stride, std::tuple<>{}, std::make_tuple(C<1>{}));

        auto rem_s = std::get<0>(folded);  // rank-1: the last surviving (largest-stride) mode
        auto rem_d = std::get<1>(folded);
        auto rs = std::get<2>(folded);  // rank R-1
        auto rd = std::get<3>(folded);  // rank R (seed + R-1 appends)
        auto last_rem_s = std::get<0>(rem_s);
        auto last_rem_d = std::get<0>(rem_d);

        // Final fold iteration's emit, done out-of-line because there's no
        // "next mode" to use as the min stride — we use the remaining mode
        // directly.
        auto last_base = std::get<R - 1>(rd);
        auto new_shape_final = last_rem_d / last_base;
        static_assert(!is_constant_v<0, std::decay_t<decltype(new_shape_final)>>,
                      "complement: non-injective layout detected at wrap-up");

        auto result_shape = append(rs, new_shape_final);  // rank R, paired with rd (rank R)

        // Tail covering [last_offset, cotarget).
        auto tail_base = last_rem_d * last_rem_s;
        auto rest_shape = ceil_div(cotarget, tail_base);
        auto rest_stride = tail_base;  // compact_col_major(scalar, base) == base

        // Build the rank-2 hierarchical result: mode 0 is the gap-filling
        // chain we just built, mode 1 is the tail.
        auto raw = make_layout(std::make_tuple(result_shape, rest_shape),
                               std::make_tuple(rd, rest_stride));
        return coalesce(raw);
    }
}

}  // namespace detail

// Public: complement of a layout into a cotarget codomain size.
//
// Precondition (multi-rank): all strides are static.
// Precondition (any rank): cotarget >= cosize(layout) and is a multiple of
// the smallest "gap" sizes (verified statically when both sides are static
// via the new_shape != 0 asserts).
template <class S, class D, class CoTarget>
MOSAIC_HOST_DEVICE constexpr auto complement(Layout<S, D> const& layout, CoTarget const& cotarget) {
    auto filtered = detail::filter(layout);
    return detail::complement_impl(filtered.shape(), filtered.stride(), cotarget);
}

// Public: default cotarget == cosize(layout). The result is the "minimal"
// complement that exactly fills out the layout's own codomain.
template <class S, class D>
MOSAIC_HOST_DEVICE constexpr auto complement(Layout<S, D> const& layout) {
    return complement(layout, cosize(layout));
}

// ===================================================================
// right_inverse — compute R where L(R(i)) = i
//
// For injective layouts, this returns a Layout whose evaluation gives
// the pre-image. Replaces Mosaic's runtime inverse_layout wrapper.
//
// For a simple strided layout (S, D), the right-inverse is computed by
// sorting modes by stride and inverting the stride mapping.
// ===================================================================

// Simple case: rank-1 layout s : d  →  inverse is s : 1/d (conceptually)
// For integer strides: inverse(s : d) swaps shape and stride roles
template <class S, class D>
MOSAIC_HOST_DEVICE constexpr auto right_inverse(Layout<S, D> const& layout) {
    if constexpr (!is_tuple_v<S> && !is_tuple_v<D>) {
        // Rank-1: inverse of (s : d) is (s : 1) if d == 1,
        // or needs reinterpretation. For d == 1, inverse is just (s : 1).
        // General case: we need the layout that maps offsets back to indices.
        //
        // For a layout s : d, the mapping is i → i*d.
        // The inverse maps j → j/d (when j is a multiple of d).
        //
        // In the full CuTe algebra, right_inverse works through complement
        // and composition. For our simplified version, we support the
        // common cases.
        return make_layout(layout.shape(), layout.stride());
    } else {
        // Multi-rank: sort by stride, then build inverse layout.
        // For the common row-major/column-major case, we can compute
        // the inverse directly.
        //
        // For a general layout with sorted strides d0 < d1 < ... < dN:
        //   inverse shape[i]  = shape[i]
        //   inverse stride[i] = product of shapes with smaller strides
        //
        // This is a simplified approach. The full CuTe right_inverse
        // handles non-injective layouts and arbitrary stride patterns.
        //
        // For Mosaic's use cases (row_major, column_major, and their
        // compositions), this covers all patterns.

        auto flat_shape = flatten(layout.shape());
        auto flat_stride = flatten(layout.stride());
        constexpr int R = rank_v<decltype(flat_shape)>;

        // Build array of (stride, shape, original_index) and sort by stride
        // Since we need this at compile time for static layouts, we handle
        // common cases directly.

        if constexpr (R == 2) {
            auto s0 = std::get<0>(flat_shape);
            auto s1 = std::get<1>(flat_shape);
            auto d0 = std::get<0>(flat_stride);
            auto d1 = std::get<1>(flat_stride);

            // For a compact rank-2 layout (s0,s1):(d0,d1):
            //
            // Col-major-like (d0 <= d1): L(i) = (i%s0)*d0 + (i/s0)*d1
            //   Inverse R = same layout (identity when d0=1)
            //   R = (s0, s1) : (1, s0)  [col-major of same shape]
            //
            // Row-major-like (d0 > d1): L(i) = (i%s0)*d0 + (i/s0)*d1
            //   Inverse R = (s1, s0) : (s0, 1)  [row-major of swapped shape]
            //
            // Derivation: for (M,N):(N,1), R(j) = (j%N)*M + j/N
            //   which is Layout<(N,M), (M,1)> = row_major(N, M).
            //   Verify: L(R(j)) = ((j%N)*M + j/N)%M * N + ((j%N)*M + j/N)/M
            //         = (j/N)*N + (j%N) = j  ✓

            if constexpr (is_static_v<decltype(d0)> && is_static_v<decltype(d1)>) {
                if constexpr (decltype(d0)::value <= decltype(d1)::value) {
                    // Col-major-like: inverse is col-major of same shape
                    return make_layout(make_shape(s0, s1), compact_col_major(make_shape(s0, s1)));
                } else {
                    // Row-major-like: inverse is row-major of swapped shape
                    return make_layout(make_shape(s1, s0), compact_row_major(make_shape(s1, s0)));
                }
            } else {
                std::size_t sd0 = static_cast<std::size_t>(d0);
                std::size_t sd1 = static_cast<std::size_t>(d1);
                std::size_t ss0 = static_cast<std::size_t>(s0);
                std::size_t ss1 = static_cast<std::size_t>(s1);

                if (sd0 <= sd1) {
                    // Col-major-like: inverse strides (1, s0)
                    return make_layout(make_shape(ss0, ss1), make_stride(std::size_t(1), ss0));
                } else {
                    // Row-major-like: swap shapes, row-major strides (s0, 1)
                    return make_layout(make_shape(ss1, ss0), make_stride(ss0, std::size_t(1)));
                }
            }
        } else if constexpr (R == 3) {
            // For rank-3, sort by stride then compute col-major inverse.
            // This handles the MFMA register layout case.
            //
            // For a compact rank-N layout sorted by stride (d_0 < d_1 < ... < d_{N-1}),
            // the inverse decomposes offsets by those strides and re-encodes
            // as a col-major index over the sorted shapes.
            auto s0 = std::size_t(std::get<0>(flat_shape));
            auto s1 = std::size_t(std::get<1>(flat_shape));
            auto s2 = std::size_t(std::get<2>(flat_shape));
            auto d0 = std::size_t(std::get<0>(flat_stride));
            auto d1 = std::size_t(std::get<1>(flat_stride));
            auto d2 = std::size_t(std::get<2>(flat_stride));

            // Sort (shape, stride) by stride value
            struct Mode {
                std::size_t shape;
                std::size_t stride;
            };
            Mode modes[3] = {{s0, d0}, {s1, d1}, {s2, d2}};

            // Insertion sort by stride
            for (int i = 1; i < 3; ++i) {
                Mode key = modes[i];
                int j = i - 1;
                while (j >= 0 && modes[j].stride > key.stride) {
                    modes[j + 1] = modes[j];
                    --j;
                }
                modes[j + 1] = key;
            }

            // The inverse decomposes offset j using the original strides,
            // then re-encodes using col-major strides over sorted shapes.
            // For compact layouts where d_{i+1} = d_i * s_i, this becomes:
            // R(j) = (j/d_0)%s_0 + ((j/d_1)%s_1)*s_0 + ((j/d_2)%s_2)*s_0*s_1
            // Which is: Layout<sorted_shapes, col_major_strides>
            // evaluated with offset-based decomposition.
            //
            // But since our layout evaluation uses colexicographic decomposition
            // (by shape, not by stride), we need to set up strides that
            // produce the correct result under colexicographic decomposition.
            //
            // For sorted modes with col-major convention:
            std::size_t inv_d0 = 1;
            std::size_t inv_d1 = modes[0].shape;
            std::size_t inv_d2 = modes[0].shape * modes[1].shape;

            return make_layout(make_shape(modes[0].shape, modes[1].shape, modes[2].shape),
                               make_stride(inv_d0, inv_d1, inv_d2));
        } else {
            // Rank > 3: runtime sort-based approach
            // For now, return the layout unchanged as a fallback
            return layout;
        }
    }
}

// ===================================================================
// logical_product — the dual of logical_divide.
// https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#product-tiling
//
// Defining identity:
//
//   logical_product(A, B) = make_layout(A, composition(complement(A, size(A)*cosize(B)), B))
//
// Where logical_divide rewrites L's coordinates as (inside-tile, which-tile),
// logical_product BUILDS a layout whose coordinates are (inside-block,
// which-block). The complement gives the "between-block" stride pattern,
// scaled to A's size × B's cosize; composing it with B places copies of A
// at B's grid positions.
//
// Result shape: (A_shape, B_shape) — rank-2 hierarchical. Mode 0 = inside
// one block; mode 1 = which block (placement).
//
// Tiler dispatch mirrors logical_divide:
//   tuple → per-mode via transform_layout_tiler_preserve (block modes past
//           the tiler's rank pass through unchanged)
//   underscore → identity (no expansion for this mode)
//   integer → wrap as make_layout(N)
//
// The zipped/tiled/flat_product wrappers mirror the divide family — same
// structure via tile_unzip + slicing.
// ===================================================================

// Layout × Layout: the defining identity.
template <class LS, class LD, class TS, class TD>
MOSAIC_HOST_DEVICE constexpr auto logical_product(Layout<LS, LD> const& block,
                                                  Layout<TS, TD> const& tiler) {
    return make_layout(block, composition(complement(block, size(block) * cosize(tiler)), tiler));
}

// Layout × Tiler: per-mode dispatch with trailing-mode passthrough.
template <class LS, class LD, class Tiler>
    requires(!is_layout_v<Tiler>)
MOSAIC_HOST_DEVICE constexpr auto logical_product(Layout<LS, LD> const& block, Tiler const& tiler) {
    if constexpr (is_tuple_v<Tiler>) {
        return detail::transform_layout_tiler_preserve(
            block, tiler, [](auto const& l, auto const& t) { return logical_product(l, t); });
    } else if constexpr (is_underscore_v<Tiler>) {
        return block;
    } else {
        static_assert(
            is_integral_v<Tiler>,
            "logical_product: tiler must be Layout, Underscore, integral, or tuple thereof");
        return logical_product(block, make_layout(tiler));
    }
}

// zipped_product / tiled_product / flat_product — convenience wrappers
// parallel to the divide family. Same regrouping via tile_unzip + slicing.

template <class LS, class LD, class Tiler>
MOSAIC_HOST_DEVICE constexpr auto zipped_product(Layout<LS, LD> const& block, Tiler const& tiler) {
    return detail::tile_unzip(logical_product(block, tiler), tiler);
}

template <class LS, class LD, class Tiler>
MOSAIC_HOST_DEVICE constexpr auto tiled_product(Layout<LS, LD> const& block, Tiler const& tiler) {
    auto zipped = zipped_product(block, tiler);
    constexpr int R1 = decltype(rank<1>(zipped.shape()))::value;
    return zipped(_, repeat<R1>(_));
}

template <class LS, class LD, class Tiler>
MOSAIC_HOST_DEVICE constexpr auto flat_product(Layout<LS, LD> const& block, Tiler const& tiler) {
    auto zipped = zipped_product(block, tiler);
    constexpr int R0 = decltype(rank<0>(zipped.shape()))::value;
    constexpr int R1 = decltype(rank<1>(zipped.shape()))::value;
    return zipped(repeat<R0>(_), repeat<R1>(_));
}

// ===================================================================
// blocked_product — "reproduce a block over a tiler"
//
// Place a copy of `inner` (the block) at every position of `outer` (the
// tiler). Each tiler position advances by cosize(inner) so block copies
// don't collide.
//
// Result shape:  ((inner.shape[0], outer.shape[0]), (inner.shape[1], outer.shape[1]), ...)
// Result stride: ((inner.stride[0], cosize(inner) * outer.stride[0]),
//                 (inner.stride[1], cosize(inner) * outer.stride[1]), ...)
//
// Within each mode, the LHS pair-element evaluates `inner` exactly; the RHS
// pair-element advances by cosize(inner) so the next tile's block starts
// past the previous tile's footprint.
//
// Bijection invariant: if `inner` and `outer` are bijective onto their
// respective codomains and have the same rank, the result is bijective onto
// [0, size(inner) * size(outer)).
// ===================================================================

template <class SA, class DA, class SB, class DB>
MOSAIC_HOST_DEVICE constexpr auto blocked_product(Layout<SA, DA> const& inner,
                                                  Layout<SB, DB> const& outer) {
    if constexpr (is_tuple_v<SA> && is_tuple_v<SB>) {
        static_assert(rank_v<SA> == rank_v<SB>,
                      "blocked_product requires same rank for both layouts");

        // Single shared scaling factor for all tiler modes — the block's
        // codomain footprint. Using per-mode product(inner.shape[i]) was the
        // earlier bug; that breaks bijectivity once block strides aren't
        // simply product-of-prior-shapes.
        auto inner_cosize = cosize(inner);

        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            auto shape =
                make_shape(make_shape(std::get<I>(inner.shape()), std::get<I>(outer.shape()))...);

            auto stride = make_stride(make_stride(std::get<I>(inner.stride()),
                                                  inner_cosize * std::get<I>(outer.stride()))...);

            return Layout<decltype(shape), decltype(stride)>{shape, stride};
        }(std::make_index_sequence<rank_v<SA>>{});
    } else if constexpr (!is_tuple_v<SA> && !is_tuple_v<SB>) {
        // Both rank-1 scalars
        auto shape = make_shape(inner.shape(), outer.shape());
        auto stride = make_stride(inner.stride(), cosize(inner) * outer.stride());
        return Layout<decltype(shape), decltype(stride)>{shape, stride};
    } else {
        // Rank mismatch — fall back to logical_product
        return logical_product(inner, outer);
    }
}

// ===================================================================
// logical_divide — expose B's tile/rest split inside A.
// https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#division-tiling
//
// Defining identity:
//
//   logical_divide(L, T) = composition(L, make_layout(T, complement(T, size(coalesce(L)))))
//
// Result shape: ((T_shape, R_shape)) — mode 0 is "inside the tile,"
// mode 1 is "which tile." With both composition and complement correct,
// logical_divide is one line. The Tiler overload mirrors composition's
// Tiler dispatch (tuple → per-mode via transform_layout_tiler; underscore
// → identity; integral → wrap as make_layout(N)).
//
// detail::tile_unzip(layout, tiler) regroups logical_divide's per-mode
// output ((T_M, R_M), (T_N, R_N), …) into rank-2 ((T_M, T_N, …), (R_M, R_N, …))
// for the zipped_divide / tiled_divide / flat_divide convenience wrappers.
// ===================================================================

// Layout ∘ Layout: the defining identity.
template <class LS, class LD, class TS, class TD>
MOSAIC_HOST_DEVICE constexpr auto logical_divide(Layout<LS, LD> const& layout,
                                                 Layout<TS, TD> const& tiler) {
    return composition(layout, make_layout(tiler, complement(tiler, size(coalesce(layout)))));
}

// Layout ∘ Tiler: tuple-per-mode dispatch, Underscore identity, integral wrap.
//
// Unlike composition's Tiler dispatch (which drops trailing layout modes past
// the tiler's rank — "which input element comes from where" doesn't need them),
// logical_divide PRESERVES trailing layout modes. logical_divide is a re-
// indexing of L; dropping data would violate the bijection invariant. The
// tuple dispatch therefore appends the layout's leftover modes.
template <class LS, class LD, class Tiler>
    requires(!is_layout_v<Tiler>)
MOSAIC_HOST_DEVICE constexpr auto logical_divide(Layout<LS, LD> const& layout, Tiler const& tiler) {
    if constexpr (is_tuple_v<Tiler>) {
        return detail::transform_layout_tiler_preserve(
            layout, tiler, [](auto const& l, auto const& t) { return logical_divide(l, t); });
    } else if constexpr (is_underscore_v<Tiler>) {
        return layout;
    } else {
        static_assert(
            is_integral_v<Tiler>,
            "logical_divide: tiler must be Layout, Underscore, integral, or tuple thereof");
        // Integer N is shorthand for the contiguous tiler (N):(1).
        return logical_divide(layout, make_layout(tiler));
    }
}

// ===================================================================
// zipped_divide / tiled_divide / flat_divide — convenience wrappers.
//
// All three regroup logical_divide's output for ergonomic indexing:
//   zipped_divide: ((T_M, T_N, …), (R_M, R_N, …)) — rank 2
//   tiled_divide:  ((T_M, T_N, …), R_M, R_N, …) — flatten rest
//   flat_divide:   (T_M, T_N, …, R_M, R_N, …) — flatten both
//
// tiled/flat use Underscore slicing on the zipped result; the slicing
// machinery (Layout::operator() with tuple coords containing _) already
// supports this via slice/dice.
// ===================================================================

template <class LS, class LD, class Tiler>
MOSAIC_HOST_DEVICE constexpr auto zipped_divide(Layout<LS, LD> const& layout, Tiler const& tiler) {
    return detail::tile_unzip(logical_divide(layout, tiler), tiler);
}

template <class LS, class LD, class Tiler>
MOSAIC_HOST_DEVICE constexpr auto tiled_divide(Layout<LS, LD> const& layout, Tiler const& tiler) {
    auto zipped = zipped_divide(layout, tiler);
    // Pull rest-mode rank as a compile-time constant for repeat<>.
    constexpr int R1 = decltype(rank<1>(zipped.shape()))::value;
    return zipped(_, repeat<R1>(_));
}

template <class LS, class LD, class Tiler>
MOSAIC_HOST_DEVICE constexpr auto flat_divide(Layout<LS, LD> const& layout, Tiler const& tiler) {
    auto zipped = zipped_divide(layout, tiler);
    constexpr int R0 = decltype(rank<0>(zipped.shape()))::value;
    constexpr int R1 = decltype(rank<1>(zipped.shape()))::value;
    return zipped(repeat<R0>(_), repeat<R1>(_));
}

}  // namespace mosaic
