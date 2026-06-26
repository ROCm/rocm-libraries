// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/// @file
/// @brief V4 negative-compile test -- graphValidationErrorSpliceTransformOverflow.
///
/// MUST FAIL to compile. The inner graph has MAX_TRANSFORMS_V4 OFFSETs
/// already (=12). Splicing it into an outer that adds even one more
/// transform pushes total > MAX_TRANSFORMS_V4 and the splice-budget check
/// in spliceInto() fires.

#include "ck_experimental/core/transform/v4_experimental.hpp"

namespace v4 = ck_tile::core::transform::v4;
using ck_tile::index_t;

constexpr auto bad = []{
    using namespace v4;

    // Inner: chain of 12 OFFSETs (= MAX_TRANSFORMS_V4). Edge ids 0..12.
    constexpr auto g_inner = make_transform_graph(
        outputs(read(12)),
        make_offset(int32_t{1}, read(11), write(12)),
        make_offset(int32_t{1}, read(10), write(11)),
        make_offset(int32_t{1}, read(9), write(10)),
        make_offset(int32_t{1}, read(8), write(9)),
        make_offset(int32_t{1}, read(7), write(8)),
        make_offset(int32_t{1}, read(6), write(7)),
        make_offset(int32_t{1}, read(5), write(6)),
        make_offset(int32_t{1}, read(4), write(5)),
        make_offset(int32_t{1}, read(3), write(4)),
        make_offset(int32_t{1}, read(2), write(3)),
        make_offset(int32_t{1}, read(1), write(2)),
        make_offset(int32_t{1}, read(0), write(1)),
        inputs(dims(5), write(0)));

    // Outer: ONE EMBED on top of the splice -> 12 + 1 = 13 transforms.
    constexpr index_t U = 0, A = 1, OFF = 2;
    return make_transform_graph(
        outputs(read(OFF)),
        make_embed(dims(20), strides(1), read(A), write(OFF)),
        make_subgraph(g_inner, read(U), write(A)),
        inputs(dims(5), write(U)));
}();

int main() { (void)bad; return 0; }
