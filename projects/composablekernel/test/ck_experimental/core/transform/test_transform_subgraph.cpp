// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/// @file
/// @brief Sub-graph splice tests for V4.
///
/// Mirrors `test_v3_subgraph.cpp` for the V4 framework. Verifies that
/// `make_subgraph(g_inner, read(...), write(...))` returns a `SubgraphNode<V>`
/// that `insertNode` dispatches into `spliceInto`, which flattens the inner
/// graph into the outer at constexpr time -- pool slots merged, edge ids
/// remapped, transforms appended.
///
/// IMPORTANT (per V3's test contract): NTTP-equality between a spliced
/// graph and a flat-equivalent graph is TOO STRONG. Inner internal slot
/// ids are relocated to a fresh range in the outer (outer_offset + inner_id),
/// which differs from the dense slot numbering a hand-written flat graph
/// produces. Structural fields like `num_edges` / `pool_used` therefore
/// legitimately differ. Tests instead verify:
///   - Behavioural equivalence: `mapCoord` produces equal output for spliced
///     vs flat-equivalent on the same input coords.
///   - NTTP-equality between two textually-identical splices.
///   - Structural match on counts that should be invariant: `num_transforms`,
///     per-transform routing arity, etc.
///
/// V4 differences from V3:
///   - Uses `SubgraphNode<V>` (folded SubGraphSplice marker into a single
///     bundled-decl type).
///   - `mapCoord<G, GB>(out, in)` takes bindings as NTTP per Fix 7.
///   - Anchors / per-Impl Topology are V4 mechanisms (not V3's slot
///     propagation), so internal-edge anchor preservation is tested via
///     a MERGE-bearing inner.

#include <gtest/gtest.h>

#include "ck_experimental/core/transform/v4_experimental.hpp"

#include <climits>
#include <cstdint>

namespace v4 = ck_tile::core::transform::v4;
using ck_tile::index_t;


/// S0 (smoke): inner has one OFFSET. Splice into outer with no extra
/// transforms. Spliced graph behaves equal to a flat-equivalent.
TEST(V4SubGraph, SimpleOneTransformInner)
{
    using namespace v4;

    constexpr int32_t kDelta = 3;

    // Inner: I -> OFFSET(+kDelta) -> O
    constexpr index_t I = 0, O = 1;
    constexpr auto g_inner = make_transform_graph(
        outputs(read(O)),
        make_offset(kDelta, read(I), write(O)),
        inputs(dims(10), write(I)));

    // Outer (spliced): U -> [splice: U->I, O->Y] -> Y
    constexpr index_t U = 0, Y = 1;
    constexpr auto g_spliced = make_transform_graph(
        outputs(read(Y)),
        make_subgraph(g_inner, read(U), write(Y)),
        inputs(dims(10), write(U)));

    // Outer (flat-equivalent): U -> OFFSET(+kDelta) -> Y
    constexpr auto g_flat = make_transform_graph(
        outputs(read(Y)),
        make_offset(kDelta, read(U), write(Y)),
        inputs(dims(10), write(U)));

    // Structural invariant: spliced absorbs inner's transform count.
    EXPECT_EQ(g_spliced.num_transforms, g_inner.num_transforms);
    EXPECT_EQ(g_spliced.num_transforms, g_flat.num_transforms);

    // Behavioural equivalence (NOT NTTP-equality -- inner internal slot
    // ids are relocated, so structural fields differ legitimately).
    constexpr auto gb_spliced = make_graph_bindings<g_spliced>();
    constexpr auto gb_flat    = make_graph_bindings<g_flat>();

    constexpr int32_t coords[] = {
        INT32_MIN, INT32_MIN + kDelta, -10, -1, 0, 1, 5, 10,
        INT32_MAX - kDelta, INT32_MAX
    };
    for(int32_t c : coords) {
        int32_t in[1]  = {c};
        int32_t out_spliced[1] = {};
        int32_t out_flat[1]    = {};
        v4::mapCoord<g_spliced>(out_spliced, in, gb_spliced);
        v4::mapCoord<g_flat>(out_flat, in, gb_flat);

        // Modular int32 add via uint to avoid signed UB at boundaries.
        const int32_t expected =
            static_cast<int32_t>(static_cast<uint32_t>(c) +
                                 static_cast<uint32_t>(kDelta));
        EXPECT_EQ(out_spliced[0], expected) << "spliced broke at coord=" << c;
        EXPECT_EQ(out_flat[0],    expected) << "flat broke at coord="    << c;
        EXPECT_EQ(out_spliced[0], out_flat[0])
            << "spliced != flat at coord=" << c;
    }
}


/// S1 (slot remap with functional independence): inner has two parallel
/// OFFSETs (2-in / 2-out). Splicing must route each input to the right
/// output -- varying U0 must only affect Y0, varying U1 must only affect
/// Y1. Catches off-by-one bugs in remapInnerEdgeId that S0 cannot.
TEST(V4SubGraph, TwoIndependentPathsRemap)
{
    using namespace v4;

    constexpr int32_t kDelta0 = 5;
    constexpr int32_t kDelta1 = 7;

    constexpr index_t I0_in = 0, I1_in = 1, O0_in = 2, O1_in = 3;
    constexpr auto g_inner = make_transform_graph(
        outputs(read(O0_in, O1_in)),
        make_offset(kDelta0, read(I0_in), write(O0_in)),
        make_offset(kDelta1, read(I1_in), write(O1_in)),
        inputs(dims(10, 10), write(I0_in, I1_in)));

    constexpr index_t U0 = 0, U1 = 1, Y0 = 2, Y1 = 3;
    constexpr auto g_spliced = make_transform_graph(
        outputs(read(Y0, Y1)),
        make_subgraph(g_inner, read(U0, U1), write(Y0, Y1)),
        inputs(dims(10, 10), write(U0, U1)));

    constexpr auto g_flat = make_transform_graph(
        outputs(read(Y0, Y1)),
        make_offset(kDelta0, read(U0), write(Y0)),
        make_offset(kDelta1, read(U1), write(Y1)),
        inputs(dims(10, 10), write(U0, U1)));

    EXPECT_EQ(g_spliced.num_transforms, g_flat.num_transforms);

    // Functional independence: 10x10 coord pairs verified through both graphs.
    constexpr auto gb_spliced = make_graph_bindings<g_spliced>();
    constexpr auto gb_flat    = make_graph_bindings<g_flat>();

    constexpr int32_t coords[] = {
        INT32_MIN, INT32_MIN + kDelta0, -10, -1, 0, 1, 5, 10,
        INT32_MAX - kDelta1, INT32_MAX
    };
    for(int32_t a : coords) {
        for(int32_t b : coords) {
            int32_t in[2]  = {a, b};
            int32_t out_spliced[2] = {};
            int32_t out_flat[2]    = {};
            v4::mapCoord<g_spliced>(out_spliced, in, gb_spliced);
            v4::mapCoord<g_flat>(out_flat, in, gb_flat);

            const int32_t exp0 =
                static_cast<int32_t>(static_cast<uint32_t>(a) +
                                     static_cast<uint32_t>(kDelta0));
            const int32_t exp1 =
                static_cast<int32_t>(static_cast<uint32_t>(b) +
                                     static_cast<uint32_t>(kDelta1));
            EXPECT_EQ(out_spliced[0], exp0) << "Y0 off at u=(" << a << "," << b << ")";
            EXPECT_EQ(out_spliced[1], exp1) << "Y1 off at u=(" << a << "," << b << ")";
            EXPECT_EQ(out_flat[0],    exp0);
            EXPECT_EQ(out_flat[1],    exp1);
        }
    }
}


/// NTTP equality across two TEXTUALLY-identical splices. The same inner
/// graph spliced into two outers (built from the same source text) MUST
/// produce identical TransformGraph values. (NTTP-equality across a
/// spliced-vs-flat pair is too strong -- see file header.)
TEST(V4SubGraph, EquivalentSplicesNTTPEqual)
{
    using namespace v4;

    constexpr int32_t kDelta = 4;

    constexpr index_t I = 0, O = 1;
    constexpr auto g_inner = make_transform_graph(
        outputs(read(O)),
        make_offset(kDelta, read(I), write(O)),
        inputs(dims(8), write(I)));

    constexpr index_t U = 0, Y = 1;
    constexpr auto g_a = make_transform_graph(
        outputs(read(Y)),
        make_subgraph(g_inner, read(U), write(Y)),
        inputs(dims(8), write(U)));
    constexpr auto g_b = make_transform_graph(
        outputs(read(Y)),
        make_subgraph(g_inner, read(U), write(Y)),
        inputs(dims(8), write(U)));

    EXPECT_TRUE(g_a == g_b)
        << "Two textually-identical splice expressions must produce "
           "identical NTTP graph values";
}


/// S2 (synthetic input-length anchors): inner contains MERGE+EMBED whose
/// Topology declares `Anchors::from_array<"component_lengths">` on internal
/// edges. spliceInto must copy those anchors with remapped ids. Verified
/// behaviourally: spliced graph computes the same offset as a flat-equivalent
/// MERGE+EMBED chain on every input coord. (NTTP-equality not asserted --
/// see file header.)
TEST(V4SubGraph, MergeEmbedInnerWithFromArrayAnchors)
{
    using namespace v4;

    constexpr index_t kComp0    = 8;
    constexpr index_t kComp1    = 8;
    constexpr index_t kInputLen = kComp0 * kComp1;
    constexpr index_t kStride0  = kComp1 + 1;

    // Inner: MERGED -> MERGE(c0, c1) -> (C0, C1) -> EMBED -> OFF
    constexpr index_t MERGED = 0, C0 = 1, C1 = 2, OFF = 3;
    constexpr auto g_inner = make_transform_graph(
        outputs(read(OFF)),
        make_embed(dims(kComp0, kComp1), strides(kStride0, 1), read(C0, C1), write(OFF)),
        make_merge(dims(kComp0, kComp1), read(MERGED), write(C0, C1)),
        inputs(dims(kInputLen), write(MERGED)));

    constexpr index_t U = 0, Y = 1;
    constexpr auto g_spliced = make_transform_graph(
        outputs(read(Y)),
        make_subgraph(g_inner, read(U), write(Y)),
        inputs(dims(kInputLen), write(U)));

    // Flat: same MERGE+EMBED chain in outer's namespace.
    constexpr index_t fU = 0, fC0 = 1, fC1 = 2, fY = 3;
    constexpr auto g_flat = make_transform_graph(
        outputs(read(fY)),
        make_embed(dims(kComp0, kComp1), strides(kStride0, 1), read(fC0, fC1), write(fY)),
        make_merge(dims(kComp0, kComp1), read(fU), write(fC0, fC1)),
        inputs(dims(kInputLen), write(fU)));

    // Structural invariant: same transform count.
    EXPECT_EQ(g_spliced.num_transforms, g_flat.num_transforms);

    // Numeric: full sweep of valid coords through MERGE+EMBED.
    constexpr auto gb_spliced = make_graph_bindings<g_spliced>();
    constexpr auto gb_flat    = make_graph_bindings<g_flat>();

    for(int32_t c = 0; c < kInputLen; ++c) {
        int32_t in[1]  = {c};
        int32_t out_spliced[1] = {};
        int32_t out_flat[1]    = {};
        v4::mapCoord<g_spliced>(out_spliced, in, gb_spliced);
        v4::mapCoord<g_flat>(out_flat, in, gb_flat);

        const int32_t expected =
            (c / kComp1) * kStride0 + (c % kComp1);
        EXPECT_EQ(out_spliced[0], expected) << "spliced off at c=" << c;
        EXPECT_EQ(out_flat[0],    expected) << "flat off at c="    << c;
    }
}


/// S3 (3-level nested splice): outer splices middle, middle splices innermost.
/// Verifies that nested-splice slot relocation works recursively (each level
/// adds outer_internal_offset worth of fresh slot space). Easy to off-by-one.
/// Outer terminus is EMBED (not OFFSET) -- exercises a different transform
/// class through the splice frontier per cpp_tester recommendation.
///
/// Innermost: U -> OFFSET(+1) -> MID -> OFFSET(+2) -> OUT  (2 transforms)
/// Middle:    U -> [splice(innermost)] -> A -> OFFSET(+4) -> OUT
/// Outer:     U -> [splice(middle)]    -> A -> EMBED(stride 1) -> OFF
///
/// Reference chain: u -> +1 -> +2 -> +4 -> *1 = u + 7.
TEST(V4SubGraph, NestedSpliceBasic3LevelChain)
{
    using namespace v4;

    constexpr index_t I_U = 0, I_MID = 1, I_OUT = 2;
    constexpr auto g_innermost = make_transform_graph(
        outputs(read(I_OUT)),
        make_offset(int32_t{1}, read(I_U), write(I_MID)),
        make_offset(int32_t{2}, read(I_MID), write(I_OUT)),
        inputs(dims(5), write(I_U)));

    constexpr index_t M_U = 0, M_A = 1, M_OUT = 2;
    constexpr auto g_mid = make_transform_graph(
        outputs(read(M_OUT)),
        make_subgraph(g_innermost, read(M_U), write(M_A)),
        make_offset(int32_t{4}, read(M_A), write(M_OUT)),
        inputs(dims(5), write(M_U)));

    // Outer terminus: EMBED with stride 1 (acts as identity on a 1-D coord
    // input). Different transform class than the OFFSET chain inside.
    constexpr index_t O_U = 0, O_A = 1, O_OFF = 2;
    constexpr auto g_outer = make_transform_graph(
        outputs(read(O_OFF)),
        make_subgraph(g_mid, read(O_U), write(O_A)),
        make_embed(dims(15), strides(1), read(O_A), write(O_OFF)),
        inputs(dims(5), write(O_U)));

    // Flattening invariant: 2 (innermost) + 1 (middle OFFSET) + 1 (outer EMBED) = 4.
    EXPECT_EQ(g_outer.num_transforms, 4)
        << "Nested flattening did not expand to 4 transforms (got "
        << g_outer.num_transforms << ").";

    // Per-transform routing-arity invariants: the 3 inlined OFFSETs each have
    // ndim_input=1 ndim_output=1; EMBED has ndim_input=1 ndim_output=1.
    for(uint8_t t = 0; t < g_outer.num_transforms; ++t) {
        EXPECT_EQ(g_outer.transforms[t].ndim_input, 1)
            << "transforms[" << static_cast<int>(t) << "].ndim_input";
        EXPECT_EQ(g_outer.transforms[t].ndim_output, 1)
            << "transforms[" << static_cast<int>(t) << "].ndim_output";
    }

    // Behavioural reference: 4-transform chain inlined.
    constexpr index_t R_U = 0, R_MID = 1, R_A0 = 2, R_A = 3, R_OFF = 4;
    constexpr auto g_ref = make_transform_graph(
        outputs(read(R_OFF)),
        make_offset(int32_t{1}, read(R_U), write(R_MID)),
        make_offset(int32_t{2}, read(R_MID), write(R_A0)),
        make_offset(int32_t{4}, read(R_A0), write(R_A)),
        make_embed(dims(15), strides(1), read(R_A), write(R_OFF)),
        inputs(dims(5), write(R_U)));

    constexpr auto gb_outer = make_graph_bindings<g_outer>();
    constexpr auto gb_ref   = make_graph_bindings<g_ref>();

    constexpr int32_t coords[] = {
        INT32_MIN, INT32_MIN + 7, -10, -1, 0, 1, 2, 3, 4, INT32_MAX - 7, INT32_MAX
    };
    for(int32_t u : coords) {
        int32_t in[1]  = {u};
        int32_t out_outer[1] = {};
        int32_t out_ref[1]   = {};
        v4::mapCoord<g_outer>(out_outer, in, gb_outer);
        v4::mapCoord<g_ref>(out_ref, in, gb_ref);

        // Modular int32 add via uint to avoid signed UB at boundaries.
        const int32_t expected =
            static_cast<int32_t>(static_cast<uint32_t>(u) + 7U);
        EXPECT_EQ(out_outer[0], expected) << "outer at u=" << u;
        EXPECT_EQ(out_ref[0],   expected) << "ref at u="   << u;
        EXPECT_EQ(out_outer[0], out_ref[0]) << "outer != ref at u=" << u;
    }
}


/// Slot-collision avoidance under 3-level nesting -- denser variant per
/// cpp_tester recommendation. Innermost picks slot id 1 (= middle's A,
/// = outer's first inner slot id). Outer chains 4 OFFSETs after the splice
/// (using slot ids 0..5 densely) so any aliasing bug produces a *different*
/// wrong sum, not just a possible same-sum coincidence.
TEST(V4SubGraph, NestedSpliceSlotCollisionAvoidance)
{
    using namespace v4;

    // Innermost: slot id 1 is internal MID.
    constexpr index_t I_U = 0, I_MID = 1, I_OUT = 2;
    constexpr auto g_innermost = make_transform_graph(
        outputs(read(I_OUT)),
        make_offset(int32_t{7}, read(I_U), write(I_MID)),
        make_offset(int32_t{0}, read(I_MID), write(I_OUT)),
        inputs(dims(5), write(I_U)));

    // Middle: A is slot id 1 -- same numeric value as innermost's MID.
    constexpr index_t M_U = 0, M_A = 1, M_OUT = 2;
    constexpr auto g_mid = make_transform_graph(
        outputs(read(M_OUT)),
        make_subgraph(g_innermost, read(M_U), write(M_A)),
        make_offset(int32_t{3}, read(M_A), write(M_OUT)),
        inputs(dims(5), write(M_U)));

    // Outer: dense 0..5 with 4 chained OFFSETs after the splice.
    // u -> [innermost->A1=u+7] -> A2=A1+3=u+10 -> A3=A2+1=u+11 ->
    // A4=A3+2=u+13 -> A5=A4+5=u+18 -> OFF=A5+10=u+28.
    constexpr index_t O_U = 0, O_A2 = 2, O_A3 = 3,
                      O_A4 = 4, O_A5 = 5, O_OFF = 6;
    constexpr auto g_outer = make_transform_graph(
        outputs(read(O_OFF)),
        make_subgraph(g_mid, read(O_U), write(O_A2)),
        make_offset(int32_t{1}, read(O_A2), write(O_A3)),
        make_offset(int32_t{2}, read(O_A3), write(O_A4)),
        make_offset(int32_t{5}, read(O_A4), write(O_A5)),
        make_offset(int32_t{10}, read(O_A5), write(O_OFF)),
        inputs(dims(5), write(O_U)));

    // INT32 boundary sweep matches the rest of the file.
    constexpr auto gb_outer = make_graph_bindings<g_outer>();
    constexpr int32_t coords[] = {
        INT32_MIN, INT32_MIN + 28, -10, -1, 0, 1, 2, 3, 4, INT32_MAX - 28, INT32_MAX
    };
    for(int32_t u : coords) {
        int32_t in[1]  = {u};
        int32_t out[1] = {};
        v4::mapCoord<g_outer>(out, in, gb_outer);
        const int32_t expected =
            static_cast<int32_t>(static_cast<uint32_t>(u) + 28U);
        EXPECT_EQ(out[0], expected)
            << "NestedSpliceSlotCollisionAvoidance broke at u=" << u
            << " -- innermost internal slot likely aliased a middle/outer slot.";
    }
}


/// Branching mid level: g_mid contains TWO splices (g_innermost_a and
/// g_innermost_b) interleaved. Outer wraps g_mid. Verifies splice expansion
/// of g_mid completes BEFORE g_outer's splice processing sees a flat list,
/// and that the two innermosts get disjoint relocation ranges.
TEST(V4SubGraph, NestedSpliceBranchingMidLevel)
{
    using namespace v4;

    constexpr index_t IA_U = 0, IA_OUT = 1;
    constexpr auto g_innermost_a = make_transform_graph(
        outputs(read(IA_OUT)),
        make_offset(int32_t{1}, read(IA_U), write(IA_OUT)),
        inputs(dims(5), write(IA_U)));

    constexpr index_t IB_U = 0, IB_OUT = 1;
    constexpr auto g_innermost_b = make_transform_graph(
        outputs(read(IB_OUT)),
        make_offset(int32_t{7}, read(IB_U), write(IB_OUT)),
        inputs(dims(5), write(IB_U)));

    // Mid: two parallel splices feed an EMBED with strides (1, 8).
    constexpr index_t M_U_A = 0, M_U_B = 1, M_A = 2, M_B = 3, M_OFF = 4;
    constexpr auto g_mid = make_transform_graph(
        outputs(read(M_OFF)),
        make_embed(dims(8, 8), strides(1, 8), read(M_A, M_B), write(M_OFF)),
        make_subgraph(g_innermost_a, read(M_U_A), write(M_A)),
        make_subgraph(g_innermost_b, read(M_U_B), write(M_B)),
        inputs(dims(5, 5), write(M_U_A, M_U_B)));

    // Outer wraps mid in a passthrough EMBED on the offset.
    constexpr index_t O_U_A = 0, O_U_B = 1, O_OFF_IN = 2, O_OFF = 3;
    constexpr auto g_outer = make_transform_graph(
        outputs(read(O_OFF)),
        make_embed(dims(64), strides(1), read(O_OFF_IN), write(O_OFF)),
        make_subgraph(g_mid, read(O_U_A, O_U_B), write(O_OFF_IN)),
        inputs(dims(5, 5), write(O_U_A, O_U_B)));

    // Flattening: 2 (innermosts) + 1 (mid embed) + 1 (outer embed) = 4.
    EXPECT_EQ(g_outer.num_transforms, 4)
        << "Branching mid splice did not flatten to 4 transforms.";

    // Reference: same DAG inlined.
    constexpr index_t R_U_A = 0, R_U_B = 1, R_A = 2, R_B = 3,
                      R_OFF_IN = 4, R_OFF = 5;
    constexpr auto g_ref = make_transform_graph(
        outputs(read(R_OFF)),
        make_embed(dims(64), strides(1), read(R_OFF_IN), write(R_OFF)),
        make_embed(dims(8, 8), strides(1, 8), read(R_A, R_B), write(R_OFF_IN)),
        make_offset(int32_t{1}, read(R_U_A), write(R_A)),
        make_offset(int32_t{7}, read(R_U_B), write(R_B)),
        inputs(dims(5, 5), write(R_U_A, R_U_B)));

    constexpr auto gb_outer = make_graph_bindings<g_outer>();
    constexpr auto gb_ref   = make_graph_bindings<g_ref>();

    for(int32_t ua = 0; ua < 5; ++ua) {
        for(int32_t ub = 0; ub < 5; ++ub) {
            int32_t in[2]  = {ua, ub};
            int32_t out_outer[1] = {};
            int32_t out_ref[1]   = {};
            v4::mapCoord<g_outer>(out_outer, in, gb_outer);
            v4::mapCoord<g_ref>(out_ref, in, gb_ref);
            EXPECT_EQ(out_outer[0], out_ref[0])
                << "Branching mid mismatch at (" << ua << "," << ub << ")";
        }
    }
}


/// Mixed bindings at every level: each nesting level has BOTH a SubgraphNode
/// and a regular non-spliced TransformNode. Confirms insertNode correctly
/// interleaves splice expansion with regular-transform insertion at multiple
/// levels.
TEST(V4SubGraph, NestedSpliceMixedWithBindingsAtEveryLevel)
{
    using namespace v4;

    constexpr index_t I_U = 0, I_OUT = 1;
    constexpr auto g_innermost = make_transform_graph(
        outputs(read(I_OUT)),
        make_offset(int32_t{2}, read(I_U), write(I_OUT)),
        inputs(dims(5), write(I_U)));

    // Mid: splice + regular OFFSET.
    constexpr index_t M_U = 0, M_A = 1, M_OUT = 2;
    constexpr auto g_mid = make_transform_graph(
        outputs(read(M_OUT)),
        make_subgraph(g_innermost, read(M_U), write(M_A)),
        make_offset(int32_t{3}, read(M_A), write(M_OUT)),
        inputs(dims(5), write(M_U)));

    // Outer: splice + regular EMBED.
    constexpr index_t O_U = 0, O_A = 1, O_OFF = 2;
    constexpr auto g_outer = make_transform_graph(
        outputs(read(O_OFF)),
        make_subgraph(g_mid, read(O_U), write(O_A)),
        make_embed(dims(15), strides(1), read(O_A), write(O_OFF)),
        inputs(dims(5), write(O_U)));

    EXPECT_EQ(g_outer.num_transforms, 3)
        << "Mixed nested flattening did not expand to 3 transforms.";

    constexpr index_t R_U = 0, R_A0 = 1, R_A = 2, R_OFF = 3;
    constexpr auto g_ref = make_transform_graph(
        outputs(read(R_OFF)),
        make_offset(int32_t{2}, read(R_U), write(R_A0)),
        make_offset(int32_t{3}, read(R_A0), write(R_A)),
        make_embed(dims(15), strides(1), read(R_A), write(R_OFF)),
        inputs(dims(5), write(R_U)));

    constexpr auto gb_outer = make_graph_bindings<g_outer>();
    constexpr auto gb_ref   = make_graph_bindings<g_ref>();

    for(int32_t u = 0; u < 5; ++u) {
        int32_t in[1]  = {u};
        int32_t out_outer[1] = {};
        int32_t out_ref[1]   = {};
        v4::mapCoord<g_outer>(out_outer, in, gb_outer);
        v4::mapCoord<g_ref>(out_ref, in, gb_ref);
        EXPECT_EQ(out_outer[0], out_ref[0])
            << "Mixed nested mismatch at u=" << u;
    }
}


/// Synthetic output anchor in a nested context: innermost is UNMERGE-as-
/// terminus declaring an output length via `outputs(dims(N), read(...))`.
/// The anchor is consumed at the mid-level's EMBED input. Verifies the
/// anchor cascade resolves through 3 levels of splice via behavioural
/// equivalence to a fully-inlined reference. (V4 lacks V3's
/// `output_lengths` field for a structural leak check; equivalence on
/// the full coord cube is the practical check.)
TEST(V4SubGraph, NestedSpliceWithSyntheticOutputAnchor)
{
    using namespace v4;

    // Innermost: UNMERGE(4, 4) -> KC, anchored synthetically to length 16.
    constexpr index_t I_K1 = 0, I_K2 = 1, I_KC = 2;
    constexpr auto g_innermost = make_transform_graph(
        outputs(dims(16), read(I_KC)),
        make_unmerge(dims(int32_t{4}, int32_t{4}), read(I_K1, I_K2), write(I_KC)),
        inputs(dims(4, 4), write(I_K1, I_K2)));

    // Mid: passes innermost's output through, then EMBEDs to a single offset.
    constexpr index_t M_K1 = 0, M_K2 = 1, M_KC = 2, M_OFF = 3;
    constexpr auto g_mid = make_transform_graph(
        outputs(read(M_OFF)),
        make_embed(dims(16), strides(1), read(M_KC), write(M_OFF)),
        make_subgraph(g_innermost, read(M_K1, M_K2), write(M_KC)),
        inputs(dims(4, 4), write(M_K1, M_K2)));

    // Outer: simple passthrough EMBED on mid's offset output.
    constexpr index_t O_K1 = 0, O_K2 = 1, O_OFF_IN = 2, O_OFF = 3;
    constexpr auto g_outer = make_transform_graph(
        outputs(read(O_OFF)),
        make_embed(dims(16), strides(1), read(O_OFF_IN), write(O_OFF)),
        make_subgraph(g_mid, read(O_K1, O_K2), write(O_OFF_IN)),
        inputs(dims(4, 4), write(O_K1, O_K2)));

    // Reference: UNMERGE+EMBED+EMBED inlined.
    constexpr index_t R_K1 = 0, R_K2 = 1, R_KC = 2,
                      R_OFF_IN = 3, R_OFF = 4;
    constexpr auto g_ref = make_transform_graph(
        outputs(read(R_OFF)),
        make_embed(dims(16), strides(1), read(R_OFF_IN), write(R_OFF)),
        make_embed(dims(16), strides(1), read(R_KC), write(R_OFF_IN)),
        make_unmerge(dims(int32_t{4}, int32_t{4}), read(R_K1, R_K2), write(R_KC)),
        inputs(dims(4, 4), write(R_K1, R_K2)));

    constexpr auto gb_outer = make_graph_bindings<g_outer>();
    constexpr auto gb_ref   = make_graph_bindings<g_ref>();

    for(int32_t k1 = 0; k1 < 4; ++k1) {
        for(int32_t k2 = 0; k2 < 4; ++k2) {
            int32_t in[2]  = {k1, k2};
            int32_t out_outer[1] = {};
            int32_t out_ref[1]   = {};
            v4::mapCoord<g_outer>(out_outer, in, gb_outer);
            v4::mapCoord<g_ref>(out_ref, in, gb_ref);
            EXPECT_EQ(out_outer[0], out_ref[0])
                << "Anchor cascade at depth 3 at (" << k1 << "," << k2 << ")";
        }
    }
}


/// Master-graph equivalence under nested splice -- the largest single
/// confidence multiplier. Partitions a 9-transform master across 3 nesting
/// levels (innermost: PAD+MERGE; mid: SLICE+OFFSET wrapping innermost;
/// outer: FREEZE+BROADCAST+UNMERGE+XOR+EMBED wrapping mid) and verifies the
/// flattened result matches a fully-inlined master across a 4D coord cube.
TEST(V4SubGraph, NestedSpliceMasterGraphEquivalence)
{
    using namespace v4;

    constexpr index_t USER_M_PAD     = 8;
    constexpr index_t LEFT_PAD       = 2;
    constexpr index_t RIGHT_PAD      = 2;
    constexpr index_t USER_K_FULL    = 4;
    constexpr index_t USER_K_OFF     = 4;
    constexpr index_t USER_BIAS      = 2;
    constexpr index_t SLICE_BEGIN    = 1;
    constexpr index_t SLICE_END      = USER_K_FULL + 1;
    constexpr index_t OFFSET_SHIFT   = 3;
    constexpr index_t FROZEN_BATCH   = 1;
    constexpr index_t M0_LEN         = 2;
    constexpr index_t M1_LEN         = 2;
    constexpr index_t K_COMBINED_LEN = USER_K_FULL * USER_K_OFF;
    constexpr index_t STRIDE_M0      = K_COMBINED_LEN * M1_LEN;
    constexpr index_t STRIDE_M1      = K_COMBINED_LEN;
    constexpr index_t STRIDE_KC      = 1;
    constexpr index_t STRIDE_NF      = USER_M_PAD * USER_K_FULL * USER_K_OFF;

    // Inlined master (all 9 transforms).
    constexpr index_t M_U_M_padded = 0, M_U_K_full = 1, M_U_K_off = 2,
                      M_U_BIAS = 3, M_M_unpadded = 4, M_K_sliced = 5,
                      M_K_offset = 6, M_K_combined = 7, M_M0 = 8, M_M1 = 9,
                      M_M0_x = 10, M_M1_x = 11, M_N_FROZEN = 12,
                      M_M_ADDR = 13;
    constexpr auto g_master = make_transform_graph(
        outputs(read(M_M_ADDR)),
        make_embed(dims(M0_LEN, M1_LEN, K_COMBINED_LEN, 1),
                             strides(STRIDE_M0, STRIDE_M1, STRIDE_KC, STRIDE_NF), read(M_M0_x, M_M1_x, M_K_combined, M_N_FROZEN), write(M_M_ADDR)),
        make_xor(read(M_M0, M_M1), write(M_M0_x, M_M1_x)),
        make_merge(dims(M0_LEN, M1_LEN), read(M_M_unpadded), write(M_M0, M_M1)),
        make_pad(int32_t{LEFT_PAD}, int32_t{RIGHT_PAD}, read(M_U_M_padded), write(M_M_unpadded)),
        make_unmerge(dims(int32_t{USER_K_FULL},
                                          int32_t{USER_K_OFF}), read(M_K_sliced, M_K_offset), write(M_K_combined)),
        make_slice(int32_t{SLICE_BEGIN}, int32_t{SLICE_END}, read(M_U_K_full), write(M_K_sliced)),
        make_offset(int32_t{OFFSET_SHIFT}, read(M_U_K_off), write(M_K_offset)),
        make_freeze(int32_t{FROZEN_BATCH}, read(), write(M_N_FROZEN)),
        make_broadcast(read(M_U_BIAS), write()),
        inputs(dims(USER_M_PAD, USER_K_FULL, USER_K_OFF, USER_BIAS),
               write(M_U_M_padded, M_U_K_full, M_U_K_off, M_U_BIAS)));

    // Innermost: PAD + MERGE (M_padded -> M0, M1).
    constexpr index_t II_U_M = 0, II_M_unpad = 1, II_M0 = 2, II_M1 = 3;
    constexpr auto g_innermost = make_transform_graph(
        outputs(read(II_M0, II_M1)),
        make_merge(dims(M0_LEN, M1_LEN), read(II_M_unpad), write(II_M0, II_M1)),
        make_pad(int32_t{LEFT_PAD}, int32_t{RIGHT_PAD}, read(II_U_M), write(II_M_unpad)),
        inputs(dims(USER_M_PAD), write(II_U_M)));

    // Mid: SLICE + OFFSET wrapping innermost.
    constexpr index_t MD_U_M = 0, MD_U_K_full = 1, MD_U_K_off = 2,
                      MD_M0 = 3, MD_M1 = 4, MD_K_sl = 5, MD_K_of = 6;
    constexpr auto g_mid = make_transform_graph(
        outputs(read(MD_M0, MD_M1, MD_K_sl, MD_K_of)),
        make_slice(int32_t{SLICE_BEGIN}, int32_t{SLICE_END}, read(MD_U_K_full), write(MD_K_sl)),
        make_offset(int32_t{OFFSET_SHIFT}, read(MD_U_K_off), write(MD_K_of)),
        make_subgraph(g_innermost, read(MD_U_M), write(MD_M0, MD_M1)),
        inputs(dims(USER_M_PAD, USER_K_FULL, USER_K_OFF),
               write(MD_U_M, MD_U_K_full, MD_U_K_off)));

    // Outer: FREEZE + BROADCAST + UNMERGE + XOR + EMBED wrapping mid.
    constexpr index_t S_U_M = 0, S_U_K_full = 1, S_U_K_off = 2, S_U_BIAS = 3,
                      S_M0 = 4, S_M1 = 5, S_K_sl = 6, S_K_of = 7,
                      S_K_combined = 8, S_M0_x = 9, S_M1_x = 10,
                      S_N_FROZEN = 11, S_M_ADDR = 12;
    constexpr auto g_outer = make_transform_graph(
        outputs(read(S_M_ADDR)),
        make_embed(dims(M0_LEN, M1_LEN, K_COMBINED_LEN, 1),
                             strides(STRIDE_M0, STRIDE_M1, STRIDE_KC, STRIDE_NF), read(S_M0_x, S_M1_x, S_K_combined, S_N_FROZEN), write(S_M_ADDR)),
        make_xor(read(S_M0, S_M1), write(S_M0_x, S_M1_x)),
        make_unmerge(dims(int32_t{USER_K_FULL},
                                          int32_t{USER_K_OFF}), read(S_K_sl, S_K_of), write(S_K_combined)),
        make_freeze(int32_t{FROZEN_BATCH}, read(), write(S_N_FROZEN)),
        make_broadcast(read(S_U_BIAS), write()),
        make_subgraph(g_mid, read(S_U_M, S_U_K_full, S_U_K_off), write(S_M0, S_M1, S_K_sl, S_K_of)),
        inputs(dims(USER_M_PAD, USER_K_FULL, USER_K_OFF, USER_BIAS),
               write(S_U_M, S_U_K_full, S_U_K_off, S_U_BIAS)));

    // Flattening: nested partition must yield 9 transforms.
    EXPECT_EQ(g_outer.num_transforms, 9)
        << "Nested partition did not flatten to the full 9-transform master "
           "(got " << g_outer.num_transforms << ").";

    // 4D coord cube. The XOR / FREEZE / BROADCAST transforms make this a
    // genuine cross-Impl correctness check, not just OFFSET arithmetic.
    constexpr auto gb_master = make_graph_bindings<g_master>();
    constexpr auto gb_outer  = make_graph_bindings<g_outer>();

    int compared = 0;
    for(int32_t m = 0; m < USER_M_PAD; ++m) {
        for(int32_t k = 0; k < USER_K_FULL; ++k) {
            for(int32_t koff = 0; koff < USER_K_OFF; ++koff) {
                for(int32_t bias = 0; bias < USER_BIAS; ++bias) {
                    int32_t in[4] = {m, k, koff, bias};
                    int32_t out_master[1] = {};
                    int32_t out_outer[1]  = {};
                    v4::mapCoord<g_master>(out_master, in, gb_master);
                    v4::mapCoord<g_outer>(out_outer, in, gb_outer);
                    EXPECT_EQ(out_master[0], out_outer[0])
                        << "Master mismatch at (m=" << m
                        << ", k=" << k << ", koff=" << koff
                        << ", bias=" << bias << ")";
                    ++compared;
                }
            }
        }
    }
    EXPECT_EQ(compared, USER_M_PAD * USER_K_FULL * USER_K_OFF * USER_BIAS);
}


/// NTTP-inequality between a nested-spliced graph and an equivalent fully-
/// inlined graph. Locks the rule that they're NOT NTTP-equal because slot id
/// ranges differ after relocation cascade. Documents the rule -- prevents a
/// future canonicalisation change from silently breaking ICF / template-cache
/// reuse assumptions.
TEST(V4SubGraph, NestedSpliceNTTPInequality)
{
    using namespace v4;

    // Innermost has an INTERNAL slot (I_MID) so relocation forces a distinct
    // outer slot id.
    constexpr index_t I_U = 0, I_MID = 1, I_OUT = 2;
    constexpr auto g_innermost = make_transform_graph(
        outputs(read(I_OUT)),
        make_offset(int32_t{1}, read(I_U), write(I_MID)),
        make_offset(int32_t{2}, read(I_MID), write(I_OUT)),
        inputs(dims(5), write(I_U)));

    constexpr index_t S_U = 0, S_OFF = 1;
    constexpr auto g_nested = make_transform_graph(
        outputs(read(S_OFF)),
        make_subgraph(g_innermost, read(S_U), write(S_OFF)),
        inputs(dims(5), write(S_U)));

    constexpr index_t F_U = 0, F_MID = 1, F_OFF = 2;
    constexpr auto g_inlined = make_transform_graph(
        outputs(read(F_OFF)),
        make_offset(int32_t{1}, read(F_U), write(F_MID)),
        make_offset(int32_t{2}, read(F_MID), write(F_OFF)),
        inputs(dims(5), write(F_U)));

    EXPECT_FALSE(g_nested == g_inlined)
        << "Spliced and inlined graphs SHOULD be NTTP-distinct (slot ids "
           "differ after relocation). If this assertion fires, the framework "
           "has started canonicalising spliced graphs -- audit ICF / cache "
           "assumptions before changing this test.";
}


/// 4-level nested splice. Confirms the relocation cascade is unbounded and
/// not silently capped at depth 3.
TEST(V4SubGraph, NestedSplice4LevelChain)
{
    using namespace v4;

    constexpr index_t L0_U = 0, L0_OUT = 1;
    constexpr auto g_l0 = make_transform_graph(
        outputs(read(L0_OUT)),
        make_offset(int32_t{1}, read(L0_U), write(L0_OUT)),
        inputs(dims(5), write(L0_U)));

    constexpr index_t L1_U = 0, L1_A = 1, L1_OUT = 2;
    constexpr auto g_l1 = make_transform_graph(
        outputs(read(L1_OUT)),
        make_subgraph(g_l0, read(L1_U), write(L1_A)),
        make_offset(int32_t{2}, read(L1_A), write(L1_OUT)),
        inputs(dims(5), write(L1_U)));

    constexpr index_t L2_U = 0, L2_A = 1, L2_OUT = 2;
    constexpr auto g_l2 = make_transform_graph(
        outputs(read(L2_OUT)),
        make_subgraph(g_l1, read(L2_U), write(L2_A)),
        make_offset(int32_t{4}, read(L2_A), write(L2_OUT)),
        inputs(dims(5), write(L2_U)));

    constexpr index_t L3_U = 0, L3_A = 1, L3_OUT = 2;
    constexpr auto g_l3 = make_transform_graph(
        outputs(read(L3_OUT)),
        make_subgraph(g_l2, read(L3_U), write(L3_A)),
        make_offset(int32_t{8}, read(L3_A), write(L3_OUT)),
        inputs(dims(5), write(L3_U)));

    // Flattening: 1 + 1 + 1 + 1 = 4 OFFSETs total (deltas 1, 2, 4, 8 -> +15).
    EXPECT_EQ(g_l3.num_transforms, 4)
        << "4-level nest did not flatten to 4 transforms.";

    constexpr auto gb_l3 = make_graph_bindings<g_l3>();

    constexpr int32_t coords[] = {
        INT32_MIN, INT32_MIN + 15, -1, 0, 1, 4, INT32_MAX - 15, INT32_MAX
    };
    for(int32_t u : coords) {
        int32_t in[1]  = {u};
        int32_t out[1] = {};
        v4::mapCoord<g_l3>(out, in, gb_l3);
        const int32_t expected =
            static_cast<int32_t>(static_cast<uint32_t>(u) + 15U);
        EXPECT_EQ(out[0], expected)
            << "4-level nest broke at u=" << u;
    }
}


// ============================================================================
// S4: placeholder graph as subgraph
// ============================================================================
//
// V4's binding-id namespace is GLOBAL across spliced graphs (verified by
// inspection of spliceInto: BINDING-kind pool payloads are copied verbatim,
// only EDGE_LENGTH payloads are remapped). Inner placeholder<0> + outer
// placeholder<0> resolve to the SAME runtime value at rb.values[0]. This is
// the intentional kernel-arg-sharing contract -- mirrors V3 exactly.
//
// All S4 tests use the RUNTIME-ARG mapCoord overload `mapCoord<g>(out, in, gb)`
// (NOT the NTTP `mapCoord<g, gb>(out, in)` form) because the bindings are
// runtime values.

/// S4-T1: Disjoint binding ids. Inner uses placeholder<0>, outer uses
/// placeholder<1>. Both resolve through rb.values[] without aliasing.
TEST(V4SubGraph, PlaceholderInnerWithDisjointBindingIdsSplices)
{
    using namespace v4;

    // Inner: OFFSET shift comes from placeholder<0>.
    constexpr index_t I_U = 0, I_OUT = 1;
    constexpr auto g_inner = make_transform_graph(
        outputs(read(I_OUT)),
        make_offset(placeholder<0>{}, read(I_U), write(I_OUT)),
        inputs(dims(5), write(I_U)));

    // Outer: EMBED stride comes from placeholder<1> (DISJOINT from inner).
    constexpr index_t U = 0, A = 1, OFF = 2;
    constexpr auto g_outer = make_transform_graph(
        outputs(read(OFF)),
        make_subgraph(g_inner, read(U), write(A)),
        make_embed(dims(20), strides(placeholder<1>{}), read(A), write(OFF)),
        inputs(dims(5), write(U)));

    constexpr auto g_ref = make_transform_graph(
        outputs(read(OFF)),
        make_offset(placeholder<0>{}, read(U), write(A)),
        make_embed(dims(20), strides(placeholder<1>{}), read(A), write(OFF)),
        inputs(dims(5), write(U)));

    constexpr int32_t kShift  = 7;
    constexpr int32_t kStride = 3;

    const auto gb_outer = make_graph_bindings<g_outer>(kShift, kStride);
    const auto gb_ref   = make_graph_bindings<g_ref>(kShift, kStride);

    // Post-splice num_bindings invariant: max(used ids) + 1 = 2.
    EXPECT_EQ(g_outer.num_bindings, 2)
        << "Disjoint splice should yield num_bindings=2 (got "
        << g_outer.num_bindings << ").";

    for(int32_t u = 0; u < 5; ++u) {
        int32_t in[1]  = {u};
        int32_t out_outer[1] = {};
        int32_t out_ref[1]   = {};
        v4::mapCoord<g_outer>(out_outer, in, gb_outer);
        v4::mapCoord<g_ref>(out_ref,     in, gb_ref);
        EXPECT_EQ(out_outer[0], out_ref[0])
            << "Disjoint binding-id splice mismatch at u=" << u;
    }
}


/// S4-T2: Binding-id COLLISION shares the value (intentional contract; not
/// a bug). Inner placeholder<0> for OFFSET shift; outer placeholder<0> for
/// EMBED stride. After splice, BOTH read rb.values[0] -- the same runtime
/// value drives both transforms. This is the V3-style kernel-arg-sharing
/// mechanism. If a future framework change auto-remaps inner ids, this test
/// fires and forces an explicit decision.
TEST(V4SubGraph, PlaceholderBindingIdCollisionSharesValue)
{
    using namespace v4;

    constexpr index_t I_U = 0, I_OUT = 1;
    constexpr auto g_inner = make_transform_graph(
        outputs(read(I_OUT)),
        make_offset(placeholder<0>{}, read(I_U), write(I_OUT)),
        inputs(dims(5), write(I_U)));

    // Outer: EMBED stride ALSO uses placeholder<0> -- collides with inner.
    constexpr index_t U = 0, A = 1, OFF = 2;
    constexpr auto g_outer = make_transform_graph(
        outputs(read(OFF)),
        make_subgraph(g_inner, read(U), write(A)),
        make_embed(dims(20), strides(placeholder<0>{}), read(A), write(OFF)),
        inputs(dims(5), write(U)));

    constexpr int32_t kSharedValue = 4;
    const auto gb = make_graph_bindings<g_outer>(kSharedValue);

    // Sharing means num_bindings stays 1, not 2.
    EXPECT_EQ(g_outer.num_bindings, 1)
        << "Binding-id collision should yield num_bindings=1 (sharing), "
           "got " << g_outer.num_bindings << ".";

    // Behaviour: u -> u + kSharedValue -> (u + kSharedValue) * kSharedValue.
    for(int32_t u = 0; u < 5; ++u) {
        int32_t in[1]  = {u};
        int32_t out[1] = {};
        v4::mapCoord<g_outer>(out, in, gb);
        const int32_t expected = (u + kSharedValue) * kSharedValue;
        EXPECT_EQ(out[0], expected)
            << "Binding-id collision should share the value at u=" << u
            << ". If this fires, auto-remap may have landed -- check the "
               "binding-id namespace contract before updating.";
    }
}


/// S4-T3: Multiple distinct placeholders inside ONE inner sub-graph.
/// Catches Schema bit_cast / payload corruption beyond id=0 (the prior 2
/// tests use only id=0 in the inner; a future Schema-layout change could
/// shrink a placeholder field and silently corrupt id>=1).
TEST(V4SubGraph, MultiplePlaceholdersInSingleSubgraph)
{
    using namespace v4;

    // Inner: 2 OFFSETs in series, each driven by a distinct binding id.
    constexpr index_t I_U = 0, I_MID = 1, I_OUT = 2;
    constexpr auto g_inner = make_transform_graph(
        outputs(read(I_OUT)),
        make_offset(placeholder<0>{}, read(I_U), write(I_MID)),
        make_offset(placeholder<1>{}, read(I_MID), write(I_OUT)),
        inputs(dims(5), write(I_U)));

    constexpr index_t U = 0, A = 1, OFF = 2;
    constexpr auto g_outer = make_transform_graph(
        outputs(read(OFF)),
        make_subgraph(g_inner, read(U), write(A)),
        make_embed(dims(30), strides(1), read(A), write(OFF)),
        inputs(dims(5), write(U)));

    constexpr int32_t kShiftA = 3;
    constexpr int32_t kShiftB = 5;
    const auto gb = make_graph_bindings<g_outer>(kShiftA, kShiftB);

    EXPECT_EQ(g_outer.num_bindings, 2)
        << "Inner uses ids 0 and 1; outer should report num_bindings=2.";

    for(int32_t u = 0; u < 5; ++u) {
        int32_t in[1]  = {u};
        int32_t out[1] = {};
        v4::mapCoord<g_outer>(out, in, gb);
        const int32_t expected = u + kShiftA + kShiftB;
        EXPECT_EQ(out[0], expected)
            << "Multi-placeholder splice broke at u=" << u
            << " -- the second binding id may have been dropped.";
    }
}


/// S4-T4: Outer is literal-only; inner has the only placeholder. The merged
/// graph's binding arity must be computed from the FULL graph (inner
/// included), not just from outer-supplied transforms.
TEST(V4SubGraph, LiteralOnlyOuterAroundPlaceholderInner)
{
    using namespace v4;

    constexpr index_t I_U = 0, I_OUT = 1;
    constexpr auto g_inner = make_transform_graph(
        outputs(read(I_OUT)),
        make_offset(placeholder<0>{}, read(I_U), write(I_OUT)),
        inputs(dims(5), write(I_U)));

    // Outer transforms are pure literal -- no placeholder in any of them.
    constexpr index_t U = 0, A = 1, OFF = 2;
    constexpr auto g_outer = make_transform_graph(
        outputs(read(OFF)),
        make_subgraph(g_inner, read(U), write(A)),
        make_embed(dims(15), strides(2), read(A), write(OFF)),
        inputs(dims(5), write(U)));

    EXPECT_EQ(g_outer.num_bindings, 1)
        << "Outer is literal-only but inner has placeholder<0>; merged "
           "num_bindings should reflect the inner.";

    constexpr int32_t kShift = 6;
    const auto gb = make_graph_bindings<g_outer>(kShift);

    for(int32_t u = 0; u < 5; ++u) {
        int32_t in[1]  = {u};
        int32_t out[1] = {};
        v4::mapCoord<g_outer>(out, in, gb);
        const int32_t expected = (u + kShift) * 2;
        EXPECT_EQ(out[0], expected)
            << "Literal-outer / placeholder-inner mismatch at u=" << u;
    }
}


/// S4-T5: Placeholder appears in inner's `inputs(dims(placeholder<X>))`
/// length anchor (not just per-Impl Schema fields). Verifies the
/// length-anchor route works through splice.
TEST(V4SubGraph, PlaceholderInLengthAnchorSlot)
{
    using namespace v4;

    // Inner: input length is placeholder<0>; OFFSET shift is literal 2.
    constexpr index_t I_U = 0, I_OUT = 1;
    constexpr auto g_inner = make_transform_graph(
        outputs(read(I_OUT)),
        make_offset(int32_t{2}, read(I_U), write(I_OUT)),
        inputs(dims(placeholder<0>{}), write(I_U)));

    // Outer: also uses placeholder<0> in its inputs(dims(...)) anchor --
    // collision is fine (sharing). EMBED uses literal stride.
    constexpr index_t U = 0, A = 1, OFF = 2;
    constexpr auto g_outer = make_transform_graph(
        outputs(read(OFF)),
        make_subgraph(g_inner, read(U), write(A)),
        make_embed(dims(20), strides(1), read(A), write(OFF)),
        inputs(dims(placeholder<0>{}), write(U)));

    constexpr int32_t kInputLength = 8;
    const auto gb = make_graph_bindings<g_outer>(kInputLength);

    // Behaviour: u -> u + 2 -> (u + 2) * 1 = u + 2.
    for(int32_t u = 0; u < 5; ++u) {
        int32_t in[1]  = {u};
        int32_t out[1] = {};
        v4::mapCoord<g_outer>(out, in, gb);
        EXPECT_EQ(out[0], u + 2)
            << "Placeholder-as-length-anchor splice broke at u=" << u;
    }
}


/// S4-T6: NTTP distinctness for disjoint vs. collision binding-id
/// arrangements. The same graph TEXT with different placeholder ids in the
/// outer EMBED produces NTTP-distinct values. Locks against accidental
/// canonicalisation that would silently merge them.
TEST(V4SubGraph, NTTPDistinctnessForDisjointVsCollisionBindingId)
{
    using namespace v4;

    constexpr index_t I_U = 0, I_OUT = 1;
    constexpr auto g_inner = make_transform_graph(
        outputs(read(I_OUT)),
        make_offset(placeholder<0>{}, read(I_U), write(I_OUT)),
        inputs(dims(5), write(I_U)));

    constexpr index_t U = 0, A = 1, OFF = 2;

    // Variant A: outer EMBED stride = placeholder<1> (DISJOINT from inner).
    constexpr auto g_disjoint = make_transform_graph(
        outputs(read(OFF)),
        make_subgraph(g_inner, read(U), write(A)),
        make_embed(dims(20), strides(placeholder<1>{}), read(A), write(OFF)),
        inputs(dims(5), write(U)));

    // Variant B: outer EMBED stride = placeholder<0> (COLLIDES with inner).
    constexpr auto g_collision = make_transform_graph(
        outputs(read(OFF)),
        make_subgraph(g_inner, read(U), write(A)),
        make_embed(dims(20), strides(placeholder<0>{}), read(A), write(OFF)),
        inputs(dims(5), write(U)));

    EXPECT_FALSE(g_disjoint == g_collision)
        << "Disjoint-id and collision-id splice variants SHOULD be "
           "NTTP-distinct (they reference different binding ids in their "
           "Pool). If this fires, the framework has started canonicalising "
           "placeholder ids -- audit kernel-arg sharing assumptions.";

    EXPECT_NE(g_disjoint.num_bindings, g_collision.num_bindings)
        << "Disjoint should have num_bindings=2; collision should have "
           "num_bindings=1.";
}


/// S4-T7: Cross-Impl binding-id sharing. Inner OFFSET uses placeholder<0>;
/// outer EMBED stride ALSO uses placeholder<0> -- the SAME id reaches a
/// DIFFERENT Impl. Verifies the binding plumbing works across distinct
/// Impl boundaries, not just within OFFSET.
TEST(V4SubGraph, CrossImplBindingIdSharing)
{
    using namespace v4;

    // Inner uses placeholder<0> for OFFSET shift.
    constexpr index_t I_U = 0, I_OUT = 1;
    constexpr auto g_inner = make_transform_graph(
        outputs(read(I_OUT)),
        make_offset(placeholder<0>{}, read(I_U), write(I_OUT)),
        inputs(dims(5), write(I_U)));

    // Outer uses placeholder<0> for EMBED stride. After splice, both inner's
    // OFFSET (in inner Impl context) and outer's EMBED (in outer Impl
    // context) read the same rb.values[0].
    constexpr index_t U = 0, A = 1, OFF = 2;
    constexpr auto g_outer = make_transform_graph(
        outputs(read(OFF)),
        make_subgraph(g_inner, read(U), write(A)),
        make_embed(dims(20), strides(placeholder<0>{}), read(A), write(OFF)),
        inputs(dims(5), write(U)));

    constexpr int32_t kSharedValue = 5;
    const auto gb = make_graph_bindings<g_outer>(kSharedValue);

    // Behaviour: u -> u + 5 -> (u + 5) * 5.
    for(int32_t u = 0; u < 5; ++u) {
        int32_t in[1]  = {u};
        int32_t out[1] = {};
        v4::mapCoord<g_outer>(out, in, gb);
        const int32_t expected = (u + kSharedValue) * kSharedValue;
        EXPECT_EQ(out[0], expected)
            << "Cross-Impl binding sharing broke at u=" << u
            << " -- OFFSET (inner Impl) and EMBED (outer Impl) should both "
               "see rb.values[0]=" << kSharedValue;
    }
}


/// S4-T8: Well-formed bindings arity post-splice. `RB<g_outer>` is sized
/// to `g_outer.num_bindings` (per-graph alias); this test confirms the
/// merged arity is computed correctly so the user-facing per-graph alias
/// doesn't under- or over-allocate after a splice.
TEST(V4SubGraph, WellFormedBindingsArityProducesCorrectBehavior)
{
    using namespace v4;

    // Inner uses ids 0, 2 (with a gap at 1) to exercise non-contiguous use.
    constexpr index_t I_U = 0, I_MID = 1, I_OUT = 2;
    constexpr auto g_inner = make_transform_graph(
        outputs(read(I_OUT)),
        make_offset(placeholder<0>{}, read(I_U), write(I_MID)),
        make_offset(placeholder<2>{}, read(I_MID), write(I_OUT)),
        inputs(dims(5), write(I_U)));

    // Outer uses id 1 (filling the gap left by inner) for EMBED stride.
    constexpr index_t U = 0, A = 1, OFF = 2;
    constexpr auto g_outer = make_transform_graph(
        outputs(read(OFF)),
        make_subgraph(g_inner, read(U), write(A)),
        make_embed(dims(30), strides(placeholder<1>{}), read(A), write(OFF)),
        inputs(dims(5), write(U)));

    // Merged arity should be max(0, 1, 2) + 1 = 3.
    EXPECT_EQ(g_outer.num_bindings, 3)
        << "Merged num_bindings should be 3 (max id 2 + 1), got "
        << g_outer.num_bindings;

    constexpr int32_t kShift0 = 1;
    constexpr int32_t kShift2 = 4;
    constexpr int32_t kStride1 = 2;

    // make_graph_bindings binds args in placeholder-id order (3 ids here).
    const auto gb = make_graph_bindings<g_outer>(kShift0, kStride1, kShift2);

    // Behaviour: u -> u + 1 -> u + 1 + 4 -> (u + 5) * 2.
    for(int32_t u = 0; u < 5; ++u) {
        int32_t in[1]  = {u};
        int32_t out[1] = {};
        v4::mapCoord<g_outer>(out, in, gb);
        const int32_t expected = (u + kShift0 + kShift2) * kStride1;
        EXPECT_EQ(out[0], expected)
            << "Well-formed bindings arity broke at u=" << u;
    }
}


/// Spliced DERIVED edge length resolves. Earlier splice tests only checked
/// mapCoord offsets; this observes the resolved length of a transform-derived
/// edge directly via detail::resolveEdgeLengths (the Phase B resolver that
/// make_graph_bindings runs internally; edge lengths are no longer cached in
/// the bindings object). EMBED's OUTPUT length is DERIVED
/// (span = 1 + sum_i (L_i - 1) * stride_i); if spliceInto fails to carry the
/// inner transform's t_output_anchors / derived-mask, the spliced output edge
/// length stays unresolved (0) while the flat-equivalent resolves correctly.
TEST(V4SubGraph, SplicedDerivedEdgeLengthResolves)
{
    using namespace v4;

    constexpr int32_t L = 4, S = 8;
    constexpr uint32_t kExpectedSpan = 1u + static_cast<uint32_t>((L - 1) * S); // 25

    // Inner: I -> EMBED(dims(L), strides(S)) -> O ; O length is DERIVED.
    constexpr index_t I = 0, O = 1;
    constexpr auto g_inner = make_transform_graph(
        outputs(read(O)),
        make_embed(dims(L), strides(S), read(I), write(O)),
        inputs(dims(L), write(I)));

    // Spliced: U -> [inner: U->I, O->Y] -> Y.
    constexpr index_t U = 0, Y = 1;
    constexpr auto g_spliced = make_transform_graph(
        outputs(read(Y)),
        make_subgraph(g_inner, read(U), write(Y)),
        inputs(dims(L), write(U)));

    // Flat-equivalent.
    constexpr auto g_flat = make_transform_graph(
        outputs(read(Y)),
        make_embed(dims(L), strides(S), read(U), write(Y)),
        inputs(dims(L), write(U)));

    // Edge lengths are Phase B intermediate data, no longer cached in the
    // bindings object, so resolve them directly via the same resolver
    // make_graph_bindings uses. Both graphs are literal (num_bindings == 0).
    const RB<g_flat>    rb_flat{};
    const RB<g_spliced> rb_spliced{};
    const auto edges_flat    = detail::resolveEdgeLengths<g_flat>(rb_flat);
    const auto edges_spliced = detail::resolveEdgeLengths<g_spliced>(rb_spliced);

    EXPECT_EQ(static_cast<uint32_t>(edges_flat[Y]), kExpectedSpan)
        << "flat EMBED derived output length wrong";
    EXPECT_EQ(static_cast<uint32_t>(edges_spliced[Y]), kExpectedSpan)
        << "spliced EMBED derived output length unresolved "
           "(spliceInto must carry t_output_anchors + derived mask)";
}


// ---------------------------------------------------------------------------
// Relocated header smokes (runtime form)
//
// These were namespace-scope static_asserts in v4_experimental.hpp. They emit
// no code, so evaluating them at header scope only taxed every TU with the
// underlying trait / adjust_precision instantiations. Moved here as runtime
// EXPECT checks: coverage is preserved in one compiled gate test, and a
// failure reports through gtest instead of breaking the build for all tests.
// ---------------------------------------------------------------------------

/// Precision re-encode (detail::adjust_precision). Exercises the <=32-bit
/// pruned path at P32 ONLY -- a Precision64 check here would pull the 64-bit
/// as_value family into this TU; P64 round-trip correctness is covered by the
/// dedicated P64 test. A signed value sign-extends, an unsigned value
/// zero-extends, and non-VALUE slots pass through unchanged -- static_cast
/// (not a raw byte copy) is what makes the narrow negative survive the widen.
TEST(V4Precision, AdjustPrecisionReencode)
{
    using namespace v4;

    EXPECT_EQ(detail::adjust_precision<Precision32>(Slot::from_value(int8_t{-5}))
                  .as_value<int32_t>(),
              int32_t{-5})
        << "adjust_precision must sign-extend a signed VALUE to target width";

    EXPECT_EQ(detail::adjust_precision<Precision32>(Slot::from_value(uint8_t{200}))
                  .as_value<uint32_t>(),
              uint32_t{200})
        << "adjust_precision must zero-extend an unsigned VALUE to target width";

    EXPECT_TRUE(detail::adjust_precision<Precision32>(Slot::from_binding_id(IndexT{3}))
                    .is_binding_id())
        << "adjust_precision must pass non-VALUE slots through unchanged";
}

/// Schema-framework member-pointer NTTP infrastructure (MemberPtrList,
/// member_array_traits, member_scalar_traits, is_array_member_v,
/// member_ptr_class_t, index_of_member_v). The trait expressions are
/// constant-evaluated; the EXPECTs report a mismatch at runtime rather than
/// failing the build.
namespace {
struct SchemaTraitProbe
{
    int32_t a;
    int32_t arr[8];

    using members = v4::detail::MemberPtrList<&SchemaTraitProbe::a,
                                              &SchemaTraitProbe::arr>;
};
} // namespace

TEST(V4Schema, MemberPtrTraits)
{
    using namespace v4::detail;

    EXPECT_TRUE(std::has_unique_object_representations_v<SchemaTraitProbe>)
        << "SchemaTraitProbe must have no padding holes for NTTP bit-equality";

    EXPECT_EQ((member_array_traits<&SchemaTraitProbe::arr>::count), 8)
        << "member_array_traits: array extent deduction must yield 8";
    EXPECT_TRUE((std::is_same_v<member_array_traits<&SchemaTraitProbe::arr>::element_type,
                                int32_t>))
        << "member_array_traits: element_type must yield int32_t";
    EXPECT_TRUE((std::is_same_v<member_scalar_traits<&SchemaTraitProbe::a>::type,
                                int32_t>))
        << "member_scalar_traits: type must yield int32_t";
    EXPECT_TRUE((is_array_member_v<&SchemaTraitProbe::arr>))
        << "is_array_member_v: array member must be detected";
    EXPECT_FALSE((is_array_member_v<&SchemaTraitProbe::a>))
        << "is_array_member_v: scalar member must not be detected";
    EXPECT_TRUE((std::is_same_v<member_ptr_class_t<decltype(&SchemaTraitProbe::a)>,
                                SchemaTraitProbe>))
        << "member_ptr_class_t: scalar member must yield owning class";
    EXPECT_TRUE((std::is_same_v<member_ptr_class_t<decltype(&SchemaTraitProbe::arr)>,
                                SchemaTraitProbe>))
        << "member_ptr_class_t: array member must yield owning class";
    EXPECT_EQ((SchemaTraitProbe::members::count), 2)
        << "MemberPtrList::count must report the number of pointers";
    EXPECT_EQ((index_of_member_v<&SchemaTraitProbe::a, SchemaTraitProbe::members>), 0)
        << "index_of_member_v: scalar member must resolve to position 0";
    EXPECT_EQ((index_of_member_v<&SchemaTraitProbe::arr, SchemaTraitProbe::members>), 1)
        << "index_of_member_v: array member must resolve to position 1";
}
