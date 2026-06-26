// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/// @file
/// @brief V4 negative-compile test -- transformErrorWriteCountArityMismatch.
///
/// MUST FAIL to compile. The inner graph has ndim_output=1 (one boundary
/// output edge I_OUT). Splicing it with write(A, B) supplies two outer
/// edges, which the `consteval transform(g_inner, read, write)` overload's
/// arity check rejects.

#include "ck_experimental/core/transform/v4_experimental.hpp"

namespace v4 = ck_tile::core::transform::v4;
using ck_tile::index_t;

constexpr auto bad = []{
    using namespace v4;

    constexpr index_t I_U = 0, I_OUT = 1;
    constexpr auto g_inner = make_transform_graph(
        outputs(read(I_OUT)),
        make_offset(int32_t{3}, read(I_U), write(I_OUT)),
        inputs(dims(5), write(I_U)));

    constexpr index_t U = 0, A = 1, B = 2, OFF = 3;
    return make_transform_graph(
        outputs(read(OFF)),
        make_embed(dims(5, 5), strides(5, 1), read(A, B), write(OFF)),
        // Inner has 1 output; supplying 2 write() args trips the stub.
        make_subgraph(g_inner, read(U), write(A, B)),
        inputs(dims(5), write(U)));
}();

int main() { (void)bad; return 0; }
