// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/// @file
/// @brief Shared header for the V4 cross-TU NTTP equality test.
///
/// V4-specific contribution vs V3 cross-TU: builds a SPLICED graph
/// (transform(g_inner, ...) producing a SubgraphNode dispatched into
/// spliceInto). The pool merge + edge-id remap + base_offset rewrite paths
/// must produce bit-identical TransformGraph values across distinct TUs
/// or the linker collapse via ODR fails.
///
/// Three TUs (cross_tu_v4_a.cpp, cross_tu_v4_b.cpp, cross_tu_v4_c_alt.cpp)
/// each define a function returning the address of `kCanaryFor<g>` where g
/// is either the standard sample SPLICED graph (a, b) or a deliberately
/// different alt SPLICED graph (c). The gtest in cross_tu_v4_main.cpp
/// asserts:
///
///   1. tu_a_canary() == tu_b_canary()  -- same builder text -> same NTTP
///                                          -> linker collapses to one symbol
///   2. tu_a_canary() != tu_c_canary()  -- different builder -> different NTTP
///                                          -> distinct symbols (proves the
///                                          mechanism has discriminating power)
///
/// Per the Itanium C++ ABI, NTTP class-type values are mangled member-wise;
/// equal NTTPs produce identical mangled symbol names which the linker
/// collapses (per [basic.def.odr] for inline variable templates).
///
/// Canary is G-dependent (per V3 pattern) plus V4-specific extensions:
/// includes `pool_used` and a fingerprint of `topo_order[]` to discriminate
/// spliced graphs that happen to share num_transforms/num_edges. Defeats
/// lld's identical-code-folding -- bare constants would silently merge.

#pragma once

#include "ck_experimental/core/transform/v4_experimental.hpp"

#include <array>

namespace cross_tu_v4 {

namespace v4 = ck_tile::core::transform::v4;

/// Inner graph for the SAMPLE splice. Same text consumed by tu_a and tu_b.
constexpr auto buildSampleInner()
{
    using namespace v4;
    constexpr ck_tile::index_t I_U = 0, I_OUT = 1;
    return make_transform_graph(
        outputs(read(I_OUT)),
        make_offset(int32_t{3}, read(I_U), write(I_OUT)),
        inputs(dims(8), write(I_U)));
}

/// Sample SPLICED outer graph -- buildSampleInner() spliced under an EMBED.
/// tu_a and tu_b both build this; their NTTP values MUST be byte-equal.
constexpr auto buildSampleSplicedGraph()
{
    using namespace v4;
    constexpr auto g_inner = buildSampleInner();
    constexpr ck_tile::index_t U = 0, A = 1, OFF = 2;
    return make_transform_graph(
        outputs(read(OFF)),
        make_embed(dims(8), strides(1), read(A), write(OFF)),
        make_subgraph(g_inner, read(U), write(A)),
        inputs(dims(8), write(U)));
}

/// Inner graph for the ALT splice -- different OFFSET shift than sample.
/// Drives a distinct NTTP value for the spliced outer.
constexpr auto buildAltInner()
{
    using namespace v4;
    constexpr ck_tile::index_t I_U = 0, I_OUT = 1;
    return make_transform_graph(
        outputs(read(I_OUT)),
        make_offset(int32_t{7}, read(I_U), write(I_OUT)),
        inputs(dims(8), write(I_U)));
}

/// Alt SPLICED outer graph -- structurally same shape as sample but the
/// inner OFFSET shift differs. NTTP value must distinguish.
constexpr auto buildAltSplicedGraph()
{
    using namespace v4;
    constexpr auto g_inner = buildAltInner();
    constexpr ck_tile::index_t U = 0, A = 1, OFF = 2;
    return make_transform_graph(
        outputs(read(OFF)),
        make_embed(dims(8), strides(1), read(A), write(OFF)),
        make_subgraph(g_inner, read(U), write(A)),
        inputs(dims(8), write(U)));
}

/// G-dependent canary. Defeats lld's identical-code-folding: a bare `int 0`
/// would be ICF'd across distinct G specializations and addresses would
/// compare equal even when NTTPs differ. Encoding G's structural fingerprint
/// forces distinct values per distinct G.
///
/// The fingerprint covers BOTH metadata counts AND a sample of the actual
/// graph payload bytes (per cpp_tester FR3: a regression that drifted
/// `pool.payloads`, `pool.kinds`, `edge_anchors`, or `transforms.base_offset`
/// across TUs would silently pass an EXPECT_EQ if the canary covered only
/// metadata).
template <auto G>
inline constexpr std::array<int, 9> kCanaryFor = {
    static_cast<int>(G.num_transforms),
    static_cast<int>(G.num_edges),
    static_cast<int>(G.pool_used),
    static_cast<int>(G.input_edge_ids.count()),
    static_cast<int>(G.output_edge_ids.count()),
    // Rolling hashes use int64_t accumulators (constexpr-eval refuses int
    // overflow even when values would coincidentally fit) and narrow via
    // mask to int31 for storage. Discriminating power is preserved -- the
    // bottom 31 bits of distinct rolling hashes still differ with very high
    // probability for the small graph sizes in this test.
    static_cast<int>([](){
        long long s = 0;
        for(int i = 0; i < static_cast<int>(G.num_transforms); ++i) {
            s = s * 31LL + static_cast<long long>(G.topo_order[i]);
        }
        return s & 0x7FFFFFFFLL;
    }()),
    static_cast<int>([](){
        long long s = 0;
        for(int i = 0; i < static_cast<int>(G.pool_used); ++i) {
            s = s * 31LL + static_cast<long long>(G.pool[i].payload);
            s = s * 31LL + static_cast<long long>(G.pool[i].kind());
        }
        return s & 0x7FFFFFFFLL;
    }()),
    static_cast<int>([](){
        long long s = 0;
        // Per-edge transform anchors (indexed by edge id).
        for(int i = 0; i < static_cast<int>(G.num_edges); ++i) {
            s = s * 31LL + static_cast<long long>(G.t_input_edge_anchors[i].kind());
            s = s * 31LL + static_cast<long long>(G.t_input_edge_anchors[i].payload);
            s = s * 31LL + static_cast<long long>(G.t_output_edge_anchors[i].kind());
            s = s * 31LL + static_cast<long long>(G.t_output_edge_anchors[i].payload);
        }
        // Boundary anchors (indexed by boundary position).
        const int n_in  = static_cast<int>(G.input_edge_ids.count());
        const int n_out = static_cast<int>(G.output_edge_ids.count());
        for(int i = 0; i < n_in; ++i) {
            s = s * 31LL + static_cast<long long>(G.input_edge_anchors[i].kind());
            s = s * 31LL + static_cast<long long>(G.input_edge_anchors[i].payload);
        }
        for(int i = 0; i < n_out; ++i) {
            s = s * 31LL + static_cast<long long>(G.output_edge_anchors[i].kind());
            s = s * 31LL + static_cast<long long>(G.output_edge_anchors[i].payload);
        }
        return s & 0x7FFFFFFFLL;
    }()),
    static_cast<int>([](){
        long long s = 0;
        for(int i = 0; i < static_cast<int>(G.num_transforms); ++i) {
            s = s * 31LL + static_cast<long long>(G.transforms[i].base_offset);
        }
        return s & 0x7FFFFFFFLL;
    }()),
};

const int* tuACanary();
const int* tuBCanary();
const int* tuCCanary();

} // namespace cross_tu_v4
