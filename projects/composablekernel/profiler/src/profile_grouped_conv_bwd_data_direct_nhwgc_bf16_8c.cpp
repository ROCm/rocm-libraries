// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "profiler/direct_conv_instance_registry.hpp"
#include "profiler/direct_conv_profiler_bridge.hpp"

namespace ck_tile::builder::profiling {

std::vector<AlgFuncPtr<SIGNATURE_NHWGC_BF16_BWD_DATA>>
get_bwd_data_direct_instances_nhwgc_bf16_8c()
{
    constexpr auto SIGNATURE = SIGNATURE_NHWGC_BF16_BWD_DATA;
    std::vector<AlgFuncPtr<SIGNATURE>> algs;
    auto run_alg = [&](auto fn) { algs.push_back(fn); };

#include "../../experimental/grouped_convolution_tile_instances/instances/backward_data_direct/grouped_convolution_backward_data_tile_nhwgc_bf16_calls_8c.inc"

    return algs;
}

} // namespace ck_tile::builder::profiling
