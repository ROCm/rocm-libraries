// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstddef>
#include <mosaic/algebra.hpp>
#include <mosaic/layout.hpp>
#include <set>

using namespace mosaic;

// ===================================================================
// blocked_product — "reproduce a block over a tiler"
//
// Semantic: every element of `tiler` is a position; place a copy of
// `block` at each position. Result is a layout of size(block) * size(tiler)
// that is a bijection onto [0, size(result)).
// ===================================================================

// Bijectivity test: blocked_product over rank-2 row_major × col_major must be a
// bijection onto [0, 16). This is the canonical "tile a block over a tiler" pattern.
TEST(BlockedProduct, FlatRank2_IsBijective) {
    auto block = make_layout(make_shape(2, 2), make_stride(2, 1));
    auto tiler = make_layout(make_shape(2, 2), make_stride(1, 2));

    auto result = blocked_product(block, tiler);

    std::set<std::size_t> seen;
    for (int i = 0; i < size(result); ++i) {
        auto off = static_cast<std::size_t>(result(i));
        EXPECT_LT(off, static_cast<std::size_t>(size(result))) << "offset out of range at i=" << i;
        EXPECT_TRUE(seen.insert(off).second) << "duplicate offset " << off << " at i=" << i;
    }
}

TEST(BlockedProduct, FlatRank2_Sizes) {
    auto block = make_layout(make_shape(2, 2), make_stride(2, 1));
    auto tiler = make_layout(make_shape(2, 2), make_stride(1, 2));
    auto result = blocked_product(block, tiler);

    EXPECT_EQ(static_cast<int>(size(result)),
              static_cast<int>(size(block)) * static_cast<int>(size(tiler)));
}

// Worked example: known offset values for each (block_coord, tiler_coord) pair.
// Tile the row-major block at each col-major tiler position, advancing by cosize(block)=4.
// ===================================================================
// coalesce — collapse adjacent contiguous modes; drop size-1 modes.
//
// Invariant: for all i in [0, size(L)), coalesce(L)(i) == L(i).
// ===================================================================

namespace {

// Bijection-style invariant check: coalesce preserves the layout function
// over its domain. Walk every flat index and compare offsets.
template <class L, class C>
void expect_coalesce_preserves(L const& L_in, C const& C_out) {
    EXPECT_EQ(static_cast<int>(size(L_in)), static_cast<int>(size(C_out)));
    for (int i = 0; i < static_cast<int>(size(L_in)); ++i) {
        EXPECT_EQ(static_cast<int>(L_in(i)), static_cast<int>(C_out(i)))
            << "coalesce changed function at i=" << i;
    }
}

}  // namespace

// Two adjacent contiguous static modes merge to one.
TEST(Coalesce, MergeAdjacentStatic) {
    auto L = make_layout(make_shape(Int<4>{}, Int<3>{}), make_stride(Int<1>{}, Int<4>{}));
    auto C = coalesce(L);
    expect_coalesce_preserves(L, C);
    EXPECT_EQ(static_cast<int>(rank(C)), 1);
    EXPECT_EQ(static_cast<int>(size(C)), 12);
}

// Non-contiguous strides: do NOT merge.
TEST(Coalesce, NoMergeNonContiguous) {
    auto L = make_layout(make_shape(Int<4>{}, Int<3>{}), make_stride(Int<1>{}, Int<8>{}));
    auto C = coalesce(L);
    expect_coalesce_preserves(L, C);
    EXPECT_EQ(static_cast<int>(rank(C)), 2);
}

// Size-1 modes are dropped.
TEST(Coalesce, DropSize1Middle) {
    auto L = make_layout(make_shape(Int<4>{}, Int<1>{}, Int<3>{}),
                         make_stride(Int<1>{}, Int<7>{}, Int<4>{}));
    auto C = coalesce(L);
    expect_coalesce_preserves(L, C);
    EXPECT_EQ(static_cast<int>(rank(C)), 1);
    EXPECT_EQ(static_cast<int>(size(C)), 12);
}

// All size-1 → canonical Layout<C<1>, C<0>>.
TEST(Coalesce, AllSize1) {
    auto L = make_layout(make_shape(Int<1>{}, Int<1>{}), make_stride(Int<5>{}, Int<7>{}));
    auto C = coalesce(L);
    EXPECT_EQ(static_cast<int>(size(C)), 1);
    EXPECT_EQ(static_cast<int>(C(0)), 0);
}

// Rank-5 layout where every adjacent pair is contiguous → collapses to rank 1.
// This is the case the old rank-4 hand-unrolled implementation fell through on.
TEST(Coalesce, Rank5AllContiguous) {
    auto L = make_layout(make_shape(Int<2>{}, Int<3>{}, Int<2>{}, Int<2>{}, Int<5>{}),
                         make_stride(Int<1>{}, Int<2>{}, Int<6>{}, Int<12>{}, Int<24>{}));
    auto C = coalesce(L);
    expect_coalesce_preserves(L, C);
    EXPECT_EQ(static_cast<int>(rank(C)), 1);
    EXPECT_EQ(static_cast<int>(size(C)), 2 * 3 * 2 * 2 * 5);
}

// Rank-6 mix with one non-contiguous break and a trailing size-1.
// (2,3):(1,2)              → contiguous (1*2==2) → merge to 6:1
// then break: 4:100        → append (1*6 != 100) → acc = (6,4):(1,100)
// then (2,5):(400,800)     → contiguous with prior (100*4==400, 100*8==800)
//                            → both merge into the 4-mode → acc = (6,40):(1,100)
// trailing 1:999           → dropped (size-1)
// expected: rank 2, shapes (6, 40), strides (1, 100).
TEST(Coalesce, Rank6Mixed) {
    auto L = make_layout(
        make_shape(Int<2>{}, Int<3>{}, Int<4>{}, Int<2>{}, Int<5>{}, Int<1>{}),
        make_stride(Int<1>{}, Int<2>{}, Int<100>{}, Int<400>{}, Int<800>{}, Int<999>{}));
    auto C = coalesce(L);
    expect_coalesce_preserves(L, C);
    EXPECT_EQ(static_cast<int>(rank(C)), 2);
    EXPECT_EQ(static_cast<int>(size(C)), 2 * 3 * 4 * 2 * 5);
}

// Adjacent modes whose stride doesn't match: cannot merge across the gap,
// even though earlier and later pairs are individually contiguous.
TEST(Coalesce, MixedMergeGap) {
    // (2,3):(1,2)  → merge to 6:1
    // 4:50         → break (stride 50 != 6*1) → append
    // 2:200        → merge with prior (50*4 == 200) → acc = (6,8):(1,50)
    auto L = make_layout(make_shape(Int<2>{}, Int<3>{}, Int<4>{}, Int<2>{}),
                         make_stride(Int<1>{}, Int<2>{}, Int<50>{}, Int<200>{}));
    auto C = coalesce(L);
    expect_coalesce_preserves(L, C);
    EXPECT_EQ(static_cast<int>(rank(C)), 2);
}

// Hierarchical input flattens before coalescing.
TEST(Coalesce, HierarchicalInputFlatten) {
    // ((2,3), 4) : ((1,2), 6)  — every mode contiguous → 24:1 after flatten+coalesce
    auto L = make_layout(make_shape(make_shape(Int<2>{}, Int<3>{}), Int<4>{}),
                         make_stride(make_stride(Int<1>{}, Int<2>{}), Int<6>{}));
    auto C = coalesce(L);
    expect_coalesce_preserves(L, C);
    EXPECT_EQ(static_cast<int>(rank(C)), 1);
    EXPECT_EQ(static_cast<int>(size(C)), 24);
}

// ---------- Dynamic (runtime-int) leaves ----------
//
// Regression: coalesce_fold's merge predicate read `decltype(leaf)::value`.
// A `&&` chain does NOT protect that access — the whole condition must still
// be a well-formed expression, and `int::value` is not — so ANY dynamic leaf
// was a hard compile error. The mosaic test suite only ever used static Int<>
// layouts, so this path was never exercised. The tests below pass plain `int`
// shapes/strides, which would fail to COMPILE under the old code.
//
// Semantic contract for dynamic leaves: contiguity cannot be proven at compile
// time, so no merge happens — every surviving mode is emitted verbatim. The
// layout FUNCTION is still preserved (that's what expect_coalesce_preserves
// checks); only the simplification is skipped. This matches CUTLASS, which
// merges only statically-known-contiguous modes.

// Two dynamic modes that ARE numerically contiguous (1*4 == 4) must still not
// merge — dynamic strides can't be proven contiguous → rank stays 2.
TEST(Coalesce, DynamicContiguousDoesNotMerge) {
    auto L = make_layout(make_shape(4, 3), make_stride(1, 4));
    auto C = coalesce(L);
    expect_coalesce_preserves(L, C);
    EXPECT_EQ(static_cast<int>(rank(C)), 2);
    EXPECT_EQ(static_cast<int>(size(C)), 12);
}

// Dynamic non-contiguous modes: also rank 2, function preserved.
TEST(Coalesce, DynamicNonContiguous) {
    auto L = make_layout(make_shape(4, 3), make_stride(1, 8));
    auto C = coalesce(L);
    expect_coalesce_preserves(L, C);
    EXPECT_EQ(static_cast<int>(rank(C)), 2);
}

// Dynamic size-1 modes are still dropped: the drop path keys off
// is_constant_v<1> on the SHAPE, which a runtime `1` does not satisfy, so a
// dynamic 1 is NOT dropped and survives as its own mode. Pin that behavior.
TEST(Coalesce, DynamicSize1NotDropped) {
    auto L = make_layout(make_shape(4, 1, 3), make_stride(1, 7, 4));
    auto C = coalesce(L);
    expect_coalesce_preserves(L, C);
    // 4, (dynamic)1, 3 all emitted verbatim — no static merge, no dynamic drop.
    EXPECT_EQ(static_cast<int>(rank(C)), 3);
    EXPECT_EQ(static_cast<int>(size(C)), 12);
}

// A STATIC size-1 mixed among dynamic modes IS dropped (is_constant_v<1> holds
// on Int<1>), while the surrounding dynamic modes are emitted un-merged.
TEST(Coalesce, StaticSize1DroppedAmongDynamic) {
    auto L = make_layout(make_shape(4, Int<1>{}, 3), make_stride(1, 7, 4));
    auto C = coalesce(L);
    expect_coalesce_preserves(L, C);
    EXPECT_EQ(static_cast<int>(rank(C)), 2);
    EXPECT_EQ(static_cast<int>(size(C)), 12);
}

// Mixed static/dynamic where the static pair is contiguous: the two static
// modes merge, the trailing dynamic mode is appended un-merged.
// (2,3):(1,2) static → merge to 6:1; then dynamic 4:6 → append (dynamic).
TEST(Coalesce, MixedStaticMergeThenDynamicAppend) {
    auto L = make_layout(make_shape(Int<2>{}, Int<3>{}, 4), make_stride(Int<1>{}, Int<2>{}, 6));
    auto C = coalesce(L);
    expect_coalesce_preserves(L, C);
    // static (2,3) → 6:1, then dynamic 4:6 appended → rank 2.
    EXPECT_EQ(static_cast<int>(rank(C)), 2);
    EXPECT_EQ(static_cast<int>(size(C)), 24);
}

// A dynamic leaf in the accumulator (last_s/last_d dynamic) followed by a
// static leaf: exercises the `is_static_v<last_s>` guard being false. Old code
// would have tried `decltype(last_s)::value` on an int → compile error.
TEST(Coalesce, DynamicAccumulatorThenStatic) {
    auto L = make_layout(make_shape(4, Int<3>{}), make_stride(1, Int<4>{}));
    auto C = coalesce(L);
    expect_coalesce_preserves(L, C);
    EXPECT_EQ(static_cast<int>(rank(C)), 2);
    EXPECT_EQ(static_cast<int>(size(C)), 12);
}

TEST(BlockedProduct, FlatRank2_KnownOffsets) {
    auto block = make_layout(make_shape(2, 2), make_stride(2, 1));  // (0,1,2,3) row-major
    auto tiler = make_layout(make_shape(2, 2), make_stride(1, 2));  // (0,1,2,3) col-major
    auto L = blocked_product(block, tiler);

    // L coord shape: ((b0, t0), (b1, t1))
    // Offset = b0*2 + t0*4 + b1*1 + t1*8
    EXPECT_EQ(L(std::make_tuple(std::make_tuple(0, 0), std::make_tuple(0, 0))), 0);
    EXPECT_EQ(L(std::make_tuple(std::make_tuple(1, 0), std::make_tuple(1, 0))),
              3);  // block(1,1)=3 in tile (0,0)
    EXPECT_EQ(L(std::make_tuple(std::make_tuple(0, 1), std::make_tuple(0, 0))),
              4);  // tile (1,0) starts at offset 4
    EXPECT_EQ(L(std::make_tuple(std::make_tuple(0, 0), std::make_tuple(0, 1))),
              8);  // tile (0,1) starts at offset 8
    EXPECT_EQ(L(std::make_tuple(std::make_tuple(1, 1), std::make_tuple(1, 1))),
              15);  // last position
}

// ===================================================================
// composition — algebraic layout composition.
//
// Defining invariant: composition(A, B)(c) == A(B(c)) for all c in
// domain(B). This is the load-bearing property; the helper below
// asserts it pointwise.
// ===================================================================

namespace {

// Property check: walk every coord in domain(B) and verify the composed
// layout matches A∘B as a function. Single biggest lever for catching
// fold off-by-ones, sign bugs, and wrap-up errors.
template <class A, class B, class AB>
void expect_composition_preserves(A const& A_in, B const& B_in, AB const& AB_out) {
    // Domain compatibility check
    EXPECT_EQ(static_cast<int>(size(B_in)), static_cast<int>(size(AB_out)));
    // Functional composition
    for (int c = 0; c < static_cast<int>(size(B_in)); ++c) {
        auto expected = static_cast<int>(A_in(static_cast<int>(B_in(c))));
        auto actual = static_cast<int>(AB_out(c));
        EXPECT_EQ(actual, expected) << "composition(A,B)(" << c << ") != A(B(" << c << "))";
    }
}

}  // namespace

// ---------- Branch 4: flat-scalar LHS, closed-form ----------

// A = (8):(1), B = (4):(2) → expected (4):(2).
TEST(Composition, FlatA_UnitStride_FlatB) {
    auto A = make_layout(Int<8>{}, Int<1>{});
    auto B = make_layout(Int<4>{}, Int<2>{});
    auto AB = composition(A, B);
    expect_composition_preserves(A, B, AB);
    EXPECT_EQ(static_cast<int>(size(AB)), 4);
    EXPECT_EQ(static_cast<int>(AB(0)), 0);
    EXPECT_EQ(static_cast<int>(AB(1)), 2);
    EXPECT_EQ(static_cast<int>(AB(2)), 4);
    EXPECT_EQ(static_cast<int>(AB(3)), 6);
}

// A = (8):(3), B = (4):(2) → expected (4):(6).
TEST(Composition, FlatA_NonUnitStride_FlatB) {
    auto A = make_layout(Int<8>{}, Int<3>{});
    auto B = make_layout(Int<4>{}, Int<2>{});
    auto AB = composition(A, B);
    expect_composition_preserves(A, B, AB);
    EXPECT_EQ(static_cast<int>(AB(0)), 0);
    EXPECT_EQ(static_cast<int>(AB(1)), 6);
    EXPECT_EQ(static_cast<int>(AB(2)), 12);
    EXPECT_EQ(static_cast<int>(AB(3)), 18);
}

// ---------- Branch 3: RHS stride-0 shortcut ----------

TEST(Composition, RhsStrideZero) {
    auto A = make_layout(Int<8>{}, Int<1>{});
    auto B = make_layout(Int<4>{}, Int<0>{});  // broadcast: all-zero output
    auto AB = composition(A, B);
    expect_composition_preserves(A, B, AB);
    EXPECT_EQ(static_cast<int>(AB(0)), 0);
    EXPECT_EQ(static_cast<int>(AB(3)), 0);
}

// ---------- Branch 5: multi-mode LHS, cross-boundary RHS ----------

// A = (4,3):(1,4), B = (6):(2) — RHS spans LHS mode 0 partially and into
// mode 1. Worked above: result should be ((2,3)):((2,4)).
TEST(Composition, MultiModeA_FlatB_SpansBoundary) {
    auto A = make_layout(make_shape(Int<4>{}, Int<3>{}), make_stride(Int<1>{}, Int<4>{}));
    auto B = make_layout(Int<6>{}, Int<2>{});
    auto AB = composition(A, B);
    expect_composition_preserves(A, B, AB);
    EXPECT_EQ(static_cast<int>(size(AB)), 6);
}

// Wrap-up B (rest_s==1, unwrap): A = (4,3):(1,4), B = (4):(1).
// Fold consumes RHS fully in mode 0; wrap-up returns the unwrapped result.
TEST(Composition, MultiModeA_FlatB_Unwrap) {
    auto A = make_layout(make_shape(Int<4>{}, Int<3>{}), make_stride(Int<1>{}, Int<4>{}));
    auto B = make_layout(Int<4>{}, Int<1>{});
    auto AB = composition(A, B);
    expect_composition_preserves(A, B, AB);
    EXPECT_EQ(static_cast<int>(size(AB)), 4);
    EXPECT_EQ(static_cast<int>(AB(0)), 0);
    EXPECT_EQ(static_cast<int>(AB(3)), 3);
}

// Skip case: RHS stride larger than first LHS mode → fold takes the
// next_s==1 branch on mode 0, then emits in mode 1.
TEST(Composition, MultiModeA_FlatB_SkipFirstMode) {
    auto A = make_layout(make_shape(Int<4>{}, Int<3>{}), make_stride(Int<1>{}, Int<4>{}));
    auto B = make_layout(Int<3>{}, Int<4>{});  // B-steps: 0, 4, 8
    auto AB = composition(A, B);
    expect_composition_preserves(A, B, AB);
    EXPECT_EQ(static_cast<int>(AB(0)), 0);
    EXPECT_EQ(static_cast<int>(AB(1)), 4);
    EXPECT_EQ(static_cast<int>(AB(2)), 8);
}

// Wrap-up A (empty result): RHS consumed entirely in the wrap-up's final
// mode. A=(4,3):(1,4), B=(12):(1) — B covers all of A; fold skips, then
// wrap-up A fires.
TEST(Composition, MultiModeA_FlatB_WrapupEmpty) {
    auto A = make_layout(make_shape(Int<4>{}, Int<3>{}), make_stride(Int<1>{}, Int<4>{}));
    auto B = make_layout(Int<12>{}, Int<1>{});
    auto AB = composition(A, B);
    expect_composition_preserves(A, B, AB);
    EXPECT_EQ(static_cast<int>(size(AB)), 12);
}

// ---------- Branch 1: tuple RHS, right-distributivity ----------

// A = (8):(1), B = ((2, 4)):((1, 2)). The result inherits B's tuple shape.
TEST(Composition, FlatA_TupleB) {
    auto A = make_layout(Int<8>{}, Int<1>{});
    auto B = make_layout(make_shape(Int<2>{}, Int<4>{}), make_stride(Int<1>{}, Int<2>{}));
    auto AB = composition(A, B);
    expect_composition_preserves(A, B, AB);
    EXPECT_EQ(static_cast<int>(size(AB)), 8);
}

// ---------- 2D LHS × 2D RHS ----------

// A = (8, 4):(4, 1) (row-major), B = (4, 2):(1, 4). Verify per-coord.
TEST(Composition, MultiModeA_MultiModeB) {
    auto A = make_layout(make_shape(Int<8>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{}));
    auto B = make_layout(make_shape(Int<4>{}, Int<2>{}), make_stride(Int<1>{}, Int<4>{}));
    auto AB = composition(A, B);
    expect_composition_preserves(A, B, AB);
}

// ---------- Tile RHS via the Tiler overload ----------

// A = (8, 4):(4, 1), Tile<(2):(2), (2):(1)>. Per-mode composition.
TEST(Composition, TileTiler) {
    auto A = make_layout(make_shape(Int<8>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{}));
    auto tile = std::make_tuple(make_layout(Int<2>{}, Int<2>{}), make_layout(Int<2>{}, Int<1>{}));
    auto AT = composition(A, tile);
    // Per-mode invariant: shape rank matches tile rank.
    EXPECT_EQ(static_cast<int>(rank(AT)), 2);
}

// Integer tiler: composition(A, Int<n>{}) treats it as (n):(_1).
TEST(Composition, IntegerTiler) {
    auto A = make_layout(Int<8>{}, Int<1>{});
    auto AT = composition(A, Int<4>{});
    EXPECT_EQ(static_cast<int>(size(AT)), 4);
    EXPECT_EQ(static_cast<int>(AT(0)), 0);
    EXPECT_EQ(static_cast<int>(AT(3)), 3);
}

// Underscore tiler: identity.
TEST(Composition, UnderscoreTiler) {
    auto A = make_layout(make_shape(Int<8>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{}));
    auto AT = composition(A, _);
    EXPECT_EQ(static_cast<int>(size(AT)), static_cast<int>(size(A)));
    for (int c = 0; c < static_cast<int>(size(A)); ++c) {
        EXPECT_EQ(static_cast<int>(AT(c)), static_cast<int>(A(c)));
    }
}

// ---------- Permutation tiler (gather every-other) ----------
// A = (8):(1), B = (4):(2) — tile gathers even offsets {0,2,4,6}.
TEST(Composition, PermutationTilerGathersEvens) {
    auto A = make_layout(Int<8>{}, Int<1>{});
    auto B = make_layout(Int<4>{}, Int<2>{});
    auto AB = composition(A, B);
    expect_composition_preserves(A, B, AB);
}

// ---------- Non-injective LHS (stride-0 mode) ----------
// A = (4, 4):(1, 0) — A(c0, c1) = c0; second mode is "broadcast." Still
// composes correctly because composition only uses A's strides, not its
// injectivity. The property check is the verification.
TEST(Composition, NonInjectiveLhs) {
    auto A = make_layout(make_shape(Int<4>{}, Int<4>{}), make_stride(Int<1>{}, Int<0>{}));
    auto B = make_layout(Int<4>{}, Int<1>{});
    auto AB = composition(A, B);
    expect_composition_preserves(A, B, AB);
}

// ---------- Static-ness propagation ----------
// All-static input should yield all-static result. Confirmed by
// is_static_v on the shape tuple.
TEST(Composition, StaticnessPropagates) {
    auto A = make_layout(make_shape(Int<8>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{}));
    auto B = make_layout(Int<4>{}, Int<2>{});
    auto AB = composition(A, B);
    static_assert(is_static_v<std::decay_t<decltype(AB.shape())>>,
                  "composition of all-static layouts should yield static shape");
    static_assert(is_static_v<std::decay_t<decltype(AB.stride())>>,
                  "composition of all-static layouts should yield static stride");
}

// composition() calls coalesce() internally — verify the dynamic (runtime-int)
// path is reachable through the public composition API without a compile error
// and with the function preserved. Under the old coalesce, `int::value` inside
// the merge predicate made this a hard compile failure.
TEST(Composition, DynamicOperandsCoalesceInternally) {
    auto A = make_layout(make_shape(8, 4), make_stride(4, 1));
    auto B = make_layout(4, 2);
    auto AB = composition(A, B);
    expect_composition_preserves(A, B, AB);
}

// Dynamic operands whose internal coalesce would merge if static: the composed
// function must still be correct even though no static simplification occurs.
TEST(Composition, DynamicUnitStrideFlatB) {
    auto A = make_layout(8, 1);
    auto B = make_layout(4, 2);
    auto AB = composition(A, B);
    expect_composition_preserves(A, B, AB);
    EXPECT_EQ(static_cast<int>(size(AB)), 4);
}

// ===================================================================
// complement — bijection-style invariant.
//
// Defining property: make_layout(B, complement(B, M)) is a bijection
// onto [0, M). The helper below walks every offset and verifies cover
// + no-overlap.
// ===================================================================

namespace {

// Property check: walk every coord of the rank-2 layout (B, C) and
// confirm we hit every offset in [0, M) exactly once.
template <class B, class C>
void expect_complement_bijection(B const& B_in, C const& C_out, int M) {
    auto BC = make_layout(B_in, C_out);  // hierarchical rank-2: (B, C)
    EXPECT_EQ(static_cast<int>(size(BC)), M);

    std::set<int> seen;
    for (int i = 0; i < static_cast<int>(size(BC)); ++i) {
        int off = static_cast<int>(BC(i));
        EXPECT_GE(off, 0) << "offset out of range at i=" << i;
        EXPECT_LT(off, M) << "offset out of range at i=" << i;
        EXPECT_TRUE(seen.insert(off).second) << "duplicate offset " << off << " at i=" << i;
    }
    EXPECT_EQ(static_cast<int>(seen.size()), M);
}

}  // namespace

// ---------- Rank-1: closed-form check ----------

// B = (4):(1), M = 12 → expected (3):(4).
TEST(Complement, Rank1_UnitStride) {
    auto B = make_layout(Int<4>{}, Int<1>{});
    auto C = complement(B, Int<12>{});
    expect_complement_bijection(B, C, 12);
    // Sanity on shape/stride.
    EXPECT_EQ(static_cast<int>(size(C)), 3);
    EXPECT_EQ(static_cast<int>(C(0)), 0);
    EXPECT_EQ(static_cast<int>(C(1)), 4);
    EXPECT_EQ(static_cast<int>(C(2)), 8);
}

// B = (4):(2), M = 16 → expected (2, 2):(1, 8). B covers {0,2,4,6}; complement
// must fill the within-stride gaps AND the across-block tile placements.
// Note: the simple (M/(n·d)):(n·d) formula in plan doc §2.2 is wrong for d>1
// — the general formula has both a fill mode and a tile mode.
TEST(Complement, Rank1_NonUnitStride) {
    auto B = make_layout(Int<4>{}, Int<2>{});
    auto C = complement(B, Int<16>{});
    expect_complement_bijection(B, C, 16);
    EXPECT_EQ(static_cast<int>(size(C)), 4);  // 2 * 2 = 4 offsets in complement
}

// Default cotarget == cosize(B). B = (4):(2), cosize = 7. The complement
// then has to fill the gaps within the layout's own footprint — the result
// covers (d=2) interior offsets, giving size 2 after coalesce.
TEST(Complement, Rank1_DefaultCotarget) {
    auto B = make_layout(Int<4>{}, Int<2>{});
    auto C = complement(B);
    // cosize(B) = (4-1)*2 + 1 = 7. Algorithm gives (2, 1):(1, 8) → coalesce
    // drops the size-1 tile mode, leaving (2):(1).
    EXPECT_EQ(static_cast<int>(size(C)), 2);
}

// ---------- Rank-2: sort-by-stride fold ----------

// B = (2, 2):(1, 8), M = 16. B covers {0,1,8,9}. Complement should cover
// the remaining 12 offsets {2,3,4,5,6,7,10,11,12,13,14,15}.
TEST(Complement, Rank2_StrideGap) {
    auto B = make_layout(make_shape(Int<2>{}, Int<2>{}), make_stride(Int<1>{}, Int<8>{}));
    auto C = complement(B, Int<16>{});
    expect_complement_bijection(B, C, 16);
}

// B = (4, 2):(2, 1), M = 16. B covers {0,1,2,3,4,5,6,7} (compact-ish, but
// with crossing strides — exercises the sort step picking stride 1 first).
TEST(Complement, Rank2_CrossingStrides) {
    auto B = make_layout(make_shape(Int<4>{}, Int<2>{}), make_stride(Int<2>{}, Int<1>{}));
    auto C = complement(B, Int<16>{});
    expect_complement_bijection(B, C, 16);
}

// B = (2, 3):(1, 6), M = 18. Mode 0 covers {0,1}, mode 1 strides {0,6,12}.
// B covers {0,1, 6,7, 12,13}. Complement covers the other 12 offsets.
// (M must be a multiple of the algorithm's tail_base = max_stride * max_shape =
// 6 * 3 = 18, otherwise the divisibility precondition fails.)
TEST(Complement, Rank2_StrideGapWide) {
    auto B = make_layout(make_shape(Int<2>{}, Int<3>{}), make_stride(Int<1>{}, Int<6>{}));
    auto C = complement(B, Int<18>{});
    expect_complement_bijection(B, C, 18);
}

// ---------- Rank-3 ----------

// B = (2, 2, 2):(1, 4, 16), M = 32. Strides leave gaps at every level.
TEST(Complement, Rank3_AllGaps) {
    auto B = make_layout(make_shape(Int<2>{}, Int<2>{}, Int<2>{}),
                         make_stride(Int<1>{}, Int<4>{}, Int<16>{}));
    auto C = complement(B, Int<32>{});
    expect_complement_bijection(B, C, 32);
}

// ---------- Edge cases ----------

// Stride-0 layout: complement covers the whole cotarget.
// B = (4):(0), M = 8. Filter drops the stride-0 mode entirely → degenerate.
// Complement returns make_layout(M, 1) = (8):(1).
TEST(Complement, Rank1_Stride0) {
    auto B = make_layout(Int<4>{}, Int<0>{});
    auto C = complement(B, Int<8>{});
    EXPECT_EQ(static_cast<int>(size(C)), 8);
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(static_cast<int>(C(i)), i);
    }
}

// All size-1: filter drops everything → same degenerate path.
TEST(Complement, AllSize1) {
    auto B = make_layout(make_shape(Int<1>{}, Int<1>{}), make_stride(Int<5>{}, Int<7>{}));
    auto C = complement(B, Int<10>{});
    EXPECT_EQ(static_cast<int>(size(C)), 10);
}

// Hierarchical input flattens through filter.
TEST(Complement, HierarchicalInput) {
    // ((2, 1), 3) : ((1, 99), 6) → after filter: (2, 3):(1, 6) → rank-2 fold
    auto B = make_layout(make_shape(make_shape(Int<2>{}, Int<1>{}), Int<3>{}),
                         make_stride(make_stride(Int<1>{}, Int<99>{}), Int<6>{}));
    auto C = complement(B, Int<18>{});
    // After filter, equivalent to the Rank2_StrideGapWide case.
    expect_complement_bijection(B, C, 18);
}

// ---------- Static-ness propagation ----------
TEST(Complement, StaticnessPropagates) {
    auto B = make_layout(make_shape(Int<2>{}, Int<2>{}), make_stride(Int<1>{}, Int<8>{}));
    auto C = complement(B, Int<16>{});
    static_assert(is_static_v<std::decay_t<decltype(C.shape())>>,
                  "complement of all-static layout should yield static shape");
    static_assert(is_static_v<std::decay_t<decltype(C.stride())>>,
                  "complement of all-static layout should yield static stride");
}

// ---------- Filter standalone ----------
TEST(Complement, FilterDropsStride0AndSize1) {
    // Verify the detail::filter helper directly through complement's behavior.
    // (1, 3, 1):(99, 4, 88) → filter to (3):(4) → complement(B, 12) = (3):(4) inverse.
    auto B = make_layout(make_shape(Int<1>{}, Int<3>{}, Int<1>{}),
                         make_stride(Int<99>{}, Int<4>{}, Int<88>{}));
    auto C = complement(B, Int<12>{});
    // After filter, equivalent to B = (3):(4), cosize 12 → C = (4):(1).
    EXPECT_EQ(static_cast<int>(size(C)), 4);
    expect_complement_bijection(B, C, 12);
}

// ===================================================================
// logical_divide, zipped/tiled/flat_divide.
//
// Defining identity (CUTLASS): logical_divide(L, T) reduces to one
// composition call. Property: walking logical_divide(L, T) in colex
// order produces the same set of offsets as walking L — just re-indexed
// via the (tile, rest) split.
// ===================================================================

namespace {

// The load-bearing invariant: logical_divide preserves L's image as a
// set of offsets. Walks every coord of L and LD, collects the offsets,
// compares the sets.
template <class L, class LD>
void expect_logical_divide_preserves(L const& L_in, LD const& LD_out) {
    EXPECT_EQ(static_cast<int>(size(L_in)), static_cast<int>(size(LD_out)));
    std::set<int> orig, divided;
    for (int i = 0; i < static_cast<int>(size(L_in)); ++i) {
        orig.insert(static_cast<int>(L_in(i)));
    }
    for (int i = 0; i < static_cast<int>(size(LD_out)); ++i) {
        divided.insert(static_cast<int>(LD_out(i)));
    }
    EXPECT_EQ(orig, divided);
}

}  // namespace

// ---------- Worked examples from layout_algebra_plan.md §2.3 ----------

// A = (12):(1), T = 4 → expected ((4, 3)):((1, 4)).
TEST(LogicalDivide, FlatAUnitStride_IntegerTiler) {
    auto A = make_layout(Int<12>{}, Int<1>{});
    auto LD = logical_divide(A, Int<4>{});
    expect_logical_divide_preserves(A, LD);
    EXPECT_EQ(static_cast<int>(size(LD)), 12);
    // Mode 0 = tile (size 4), mode 1 = rest (size 3).
    EXPECT_EQ(static_cast<int>(rank<0>(LD.shape())), 1);
    EXPECT_EQ(static_cast<int>(rank<1>(LD.shape())), 1);
}

// A = (8):(1), T = (4):(2) — permutation tiler gathering even offsets.
TEST(LogicalDivide, FlatA_PermutationTiler) {
    auto A = make_layout(Int<8>{}, Int<1>{});
    auto T = make_layout(Int<4>{}, Int<2>{});
    auto LD = logical_divide(A, T);
    expect_logical_divide_preserves(A, LD);
    EXPECT_EQ(static_cast<int>(size(LD)), 8);
}

// A = (8, 6):(6, 1), T = (Int<2>, Int<3>) tuple tiler.
TEST(LogicalDivide, MultiModeA_TupleTiler) {
    auto A = make_layout(make_shape(Int<8>{}, Int<6>{}), make_stride(Int<6>{}, Int<1>{}));
    auto T = std::make_tuple(Int<2>{}, Int<3>{});
    auto LD = logical_divide(A, T);
    expect_logical_divide_preserves(A, LD);
}

// ---------- Tiler dispatch cases ----------

// Underscore tiler → identity. logical_divide(L, _) == L.
TEST(LogicalDivide, UnderscoreTiler) {
    auto A = make_layout(make_shape(Int<8>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{}));
    auto LD = logical_divide(A, _);
    EXPECT_EQ(static_cast<int>(size(LD)), static_cast<int>(size(A)));
    for (int i = 0; i < static_cast<int>(size(A)); ++i) {
        EXPECT_EQ(static_cast<int>(LD(i)), static_cast<int>(A(i)));
    }
}

// Integer tiler on multi-mode layout: only first mode is tiled.
TEST(LogicalDivide, IntegerTilerOnMultiMode) {
    auto A = make_layout(make_shape(Int<8>{}, Int<4>{}), make_stride(Int<1>{}, Int<8>{}));
    // logical_divide(A, 4) wraps 4 as make_layout(4) and composes.
    // Operates on coalesce(A) which is rank-1 contiguous (8,4):(1,8) →
    // doesn't coalesce since 1*8==8 — wait, 8*1 = 8 ✓, so it coalesces to
    // 32:1. Then complement etc.
    auto LD = logical_divide(A, Int<4>{});
    expect_logical_divide_preserves(A, LD);
}

// ---------- Divide-family wrappers ----------

// zipped_divide regroups ((T_M, R_M), (T_N, R_N)) → ((T_M, T_N), (R_M, R_N))
TEST(ZippedDivide, RegroupsTileAndRest) {
    auto A = make_layout(make_shape(Int<8>{}, Int<6>{}), make_stride(Int<6>{}, Int<1>{}));
    auto T = std::make_tuple(Int<2>{}, Int<3>{});
    auto ZD = zipped_divide(A, T);
    expect_logical_divide_preserves(A, ZD);
    // Rank-2 result: mode 0 is the per-mode tiles, mode 1 is the per-mode rests.
    EXPECT_EQ(static_cast<int>(rank(ZD)), 2);
    EXPECT_EQ(static_cast<int>(rank<0>(ZD.shape())), 2);  // (T_M, T_N)
    EXPECT_EQ(static_cast<int>(rank<1>(ZD.shape())), 2);  // (R_M, R_N)
}

// tiled_divide flattens the rest mode: ((T_M, T_N), R_M, R_N).
TEST(TiledDivide, FlattenedRest) {
    auto A = make_layout(make_shape(Int<8>{}, Int<6>{}), make_stride(Int<6>{}, Int<1>{}));
    auto T = std::make_tuple(Int<2>{}, Int<3>{});
    auto TD = tiled_divide(A, T);
    expect_logical_divide_preserves(A, TD);
    // Rank 3: tile-tuple + 2 flat rest modes.
    EXPECT_EQ(static_cast<int>(rank(TD)), 3);
    EXPECT_EQ(static_cast<int>(rank<0>(TD.shape())), 2);  // (T_M, T_N)
}

// flat_divide flattens both modes: (T_M, T_N, R_M, R_N).
TEST(FlatDivide, FullyFlat) {
    auto A = make_layout(make_shape(Int<8>{}, Int<6>{}), make_stride(Int<6>{}, Int<1>{}));
    auto T = std::make_tuple(Int<2>{}, Int<3>{});
    auto FD = flat_divide(A, T);
    expect_logical_divide_preserves(A, FD);
    // Rank 4: T_M, T_N, R_M, R_N as flat modes.
    EXPECT_EQ(static_cast<int>(rank(FD)), 4);
}

// ---------- Edge cases ----------

// Trailing layout modes past tiler rank: zipped_divide appends them to rest.
TEST(ZippedDivide, TilerRankLessThanLayout) {
    // A rank 3, T rank 2 → mode 2 of A passes through into the rest.
    auto A = make_layout(make_shape(Int<4>{}, Int<6>{}, Int<2>{}),
                         make_stride(Int<1>{}, Int<8>{}, Int<48>{}));
    auto T = std::make_tuple(Int<2>{}, Int<3>{});
    auto ZD = zipped_divide(A, T);
    expect_logical_divide_preserves(A, ZD);
    // Mode 1 (rest) should now have rank 3: (R_M, R_N, mode_2_of_A).
    EXPECT_EQ(static_cast<int>(rank<0>(ZD.shape())), 2);
    EXPECT_EQ(static_cast<int>(rank<1>(ZD.shape())), 3);
}

// Layout that coalesces before division: hierarchical input flattens.
TEST(LogicalDivide, CoalescesBeforeDivide) {
    // ((2, 4), 3):((1, 2), 8) — mode 0 coalesces to 8:1.
    auto A = make_layout(make_shape(make_shape(Int<2>{}, Int<4>{}), Int<3>{}),
                         make_stride(make_stride(Int<1>{}, Int<2>{}), Int<8>{}));
    auto LD = logical_divide(A, Int<4>{});
    expect_logical_divide_preserves(A, LD);
}

// Note on static-ness: unlike Composition (which often produces rank-1
// scalar shape/stride results), logical_divide always produces a hierarchical
// tuple result. mosaic's is_static_v == is_empty_v doesn't propagate through
// std::tuple (EBO isn't guaranteed by the standard), so a tuple of empties
// isn't itself empty. The leaves ARE static (compile-time C<v> constants);
// the bijection property test above is the load-bearing verification.

// ===================================================================
// logical_product, zipped/tiled/flat_product.
//
// Dual of logical_divide. Defining identity:
//   logical_product(A, B) = make_layout(A, composition(complement(A, size(A)·cosize(B)), B))
//
// Result shape: (A_shape, B_shape) — rank 2. Mode 0 = inside the block (= A),
// mode 1 = block placement (= B's coords via the complement).
//
// Bijection property: the result lays out size(A)·size(B) elements bijectively
// onto a range determined by the complement's reach.
// ===================================================================

namespace {

// Bijection check: walk every coord of the product and verify all offsets
// are unique. Note: logical_product does NOT guarantee bijection in general
// — only size is guaranteed (size(A)*size(T)). Bijection holds for the
// rank-1 cases (FlatA_FlatB, NonUnitStride, IntegerTiler) and for
// MultiModeBoth where A and T jointly factor cleanly. Per-mode tuple tilers
// applied to non-compact A can produce overlaps — for those, use
// blocked_product instead. We only invoke this helper on fixtures we expect
// to be bijective.
template <class LP>
void expect_product_bijective(LP const& LP_out) {
    int n = static_cast<int>(size(LP_out));
    std::set<int> seen;
    for (int i = 0; i < n; ++i) {
        int off = static_cast<int>(LP_out(i));
        EXPECT_GE(off, 0);
        EXPECT_TRUE(seen.insert(off).second) << "duplicate offset " << off << " at i=" << i;
    }
    EXPECT_EQ(static_cast<int>(seen.size()), n);
}

// Size-only check: logical_product's @post promises size(result) == size(A) * size(T).
template <class A, class T, class LP>
void expect_product_size(A const& A_in, T const& T_in, LP const& LP_out) {
    EXPECT_EQ(static_cast<int>(size(LP_out)),
              static_cast<int>(size(A_in)) * static_cast<int>(size(T_in)));
}

}  // namespace

// ---------- Rank-1 Layout × Rank-1 Layout ----------

// A = (2):(1), B = (3):(1). Place 2-element blocks at 3 grid positions.
// Result shape: (2, 3), bijective over [0, 6).
TEST(LogicalProduct, FlatA_FlatB) {
    auto A = make_layout(Int<2>{}, Int<1>{});
    auto B = make_layout(Int<3>{}, Int<1>{});
    auto LP = logical_product(A, B);
    EXPECT_EQ(static_cast<int>(size(LP)), 6);
    expect_product_bijective(LP);
}

// A = (2):(2), B = (2):(1). A covers {0, 2}; complement fills the gap (1, 3)
// and B places 2 copies. Precondition: cotarget = size(A)*cosize(B) = 4 must
// be a multiple of A's n*d = 4 ✓.
TEST(LogicalProduct, NonUnitStride) {
    auto A = make_layout(Int<2>{}, Int<2>{});
    auto B = make_layout(Int<2>{}, Int<1>{});
    auto LP = logical_product(A, B);
    EXPECT_EQ(static_cast<int>(size(LP)), 4);
    expect_product_bijective(LP);
}

// ---------- Multi-mode block × multi-mode tiler ----------

// A = (2, 2):(1, 8), B = (2, 2):(1, 2). Both rank 2.
TEST(LogicalProduct, MultiModeBoth) {
    auto A = make_layout(make_shape(Int<2>{}, Int<2>{}), make_stride(Int<1>{}, Int<8>{}));
    auto B = make_layout(make_shape(Int<2>{}, Int<2>{}), make_stride(Int<1>{}, Int<2>{}));
    auto LP = logical_product(A, B);
    EXPECT_EQ(static_cast<int>(size(LP)), 16);
    expect_product_bijective(LP);
}

// ---------- Tiler dispatch cases ----------

// Underscore tiler → identity: logical_product(A, _) == A.
TEST(LogicalProduct, UnderscoreTiler) {
    auto A = make_layout(make_shape(Int<2>{}, Int<3>{}), make_stride(Int<1>{}, Int<2>{}));
    auto LP = logical_product(A, _);
    EXPECT_EQ(static_cast<int>(size(LP)), static_cast<int>(size(A)));
    for (int i = 0; i < static_cast<int>(size(A)); ++i) {
        EXPECT_EQ(static_cast<int>(LP(i)), static_cast<int>(A(i)));
    }
}

// Integer tiler: shorthand for (N):(1) — place N copies of A side by side.
TEST(LogicalProduct, IntegerTiler) {
    auto A = make_layout(Int<2>{}, Int<1>{});
    auto LP = logical_product(A, Int<3>{});
    EXPECT_EQ(static_cast<int>(size(LP)), 6);
    expect_product_bijective(LP);
}

// Tuple tiler: per-mode product. Size invariant: 2*2 * 2*2 = 16. Per-mode
// bijection isn't guaranteed when A's modes have different strides — the
// per-mode complements can interleave and produce overlapping coordinates.
// blocked_product is the rank-merging version that handles this cleanly;
// logical_product just satisfies the size @post.
TEST(LogicalProduct, TupleTiler) {
    auto A = make_layout(make_shape(Int<2>{}, Int<2>{}), make_stride(Int<1>{}, Int<2>{}));
    auto T = std::make_tuple(Int<2>{}, Int<2>{});
    auto LP = logical_product(A, T);
    expect_product_size(A, std::make_tuple(Int<2>{}, Int<2>{}), LP);
}

// ---------- Product-family wrappers ----------

// zipped_product regroups to rank-2: ((BLK modes), (placement modes)).
// Tests structural rank only; bijection isn't a logical_product invariant
// (see TupleTiler note above).
TEST(ZippedProduct, RegroupsBlockAndPlacement) {
    auto A = make_layout(make_shape(Int<2>{}, Int<2>{}), make_stride(Int<1>{}, Int<2>{}));
    auto T = std::make_tuple(Int<2>{}, Int<2>{});
    auto ZP = zipped_product(A, T);
    EXPECT_EQ(static_cast<int>(rank(ZP)), 2);
    EXPECT_EQ(static_cast<int>(rank<0>(ZP.shape())), 2);  // (BLK_M, BLK_N)
    EXPECT_EQ(static_cast<int>(rank<1>(ZP.shape())), 2);  // (P_M, P_N)
    expect_product_size(A, T, ZP);
}

// tiled_product flattens placement: ((BLK_M, BLK_N), P_M, P_N).
TEST(TiledProduct, FlattenedPlacement) {
    auto A = make_layout(make_shape(Int<2>{}, Int<2>{}), make_stride(Int<1>{}, Int<2>{}));
    auto T = std::make_tuple(Int<2>{}, Int<2>{});
    auto TP = tiled_product(A, T);
    EXPECT_EQ(static_cast<int>(rank(TP)), 3);
    expect_product_size(A, T, TP);
}

// flat_product fully flattens: (BLK_M, BLK_N, P_M, P_N).
TEST(FlatProduct, FullyFlat) {
    auto A = make_layout(make_shape(Int<2>{}, Int<2>{}), make_stride(Int<1>{}, Int<2>{}));
    auto T = std::make_tuple(Int<2>{}, Int<2>{});
    auto FP = flat_product(A, T);
    EXPECT_EQ(static_cast<int>(rank(FP)), 4);
    expect_product_size(A, T, FP);
}

// ---------- Trailing-mode passthrough ----------

// Block has rank 3, tiler rank 2 → mode 2 of block passes through unchanged
// (same semantics as logical_divide). Size check only — bijection isn't
// guaranteed for per-mode tuple-tiler products on non-compact blocks.
TEST(LogicalProduct, BlockRankGreaterThanTiler) {
    auto A = make_layout(make_shape(Int<2>{}, Int<2>{}, Int<3>{}),
                         make_stride(Int<1>{}, Int<2>{}, Int<24>{}));
    auto T = std::make_tuple(Int<2>{}, Int<2>{});
    auto LP = logical_product(A, T);
    // Mode 2 (size 3) passes through, so result size = 2*2 * 2*2 * 3 = 48.
    EXPECT_EQ(static_cast<int>(size(LP)), 48);
}
