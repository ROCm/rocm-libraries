// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <tuple>
#include <string>

#include "ck_tile/builder/testing/conv/args.hpp"
#include "ck_tile/builder/testing/conv/fwd.hpp"
#include "ck_tile/builder/testing/conv/bwd_data.hpp"
#include "ck_tile/builder/conv_signature_concepts.hpp"

// Suppress warnings from hipconv-ported kernel code
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_fwd_4c_fp16_kernel.hpp"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_bwd_data_4c_fp16_kernel.hpp"
#pragma clang diagnostic pop

namespace ck_tile::builder::profiling {

namespace ckt = ck_tile::builder::test;

/// Bridge function that adapts the profiler's run_alg callback signature to the
/// direct convolution kernel wrapper structs. One instantiation per config index.
///
/// For forward convolution (ConvDirectionIsForward):
///   inputs.input  = NHWGC input data
///   inputs.weight = GKYXC weights
///   outputs.output = NHWGK output
///
/// For backward data (ConvDirectionIsBackwardData):
///   inputs.output = NHWGK output gradient (read by kernel)
///   inputs.weight = GKYXC weights (read by kernel)
///   outputs.input = NHWGC input gradient (written by kernel)
template <auto SIGNATURE, int ConfigIdx>
std::tuple<bool, float, std::string>
run_direct_conv_4c_fp16(const ckt::Args<SIGNATURE>& args,
                        const ckt::Inputs<SIGNATURE>& inputs,
                        const ckt::Outputs<SIGNATURE>& outputs,
                        const ck_tile::stream_config& s_conf)
{
    const auto param = args.to_ck_tile_conv_param();

    if constexpr(ck_tile::builder::ConvDirectionIsForward<SIGNATURE>)
    {
        using Kernel = direct_conv::DirectConvForward4CFp16Kernel<ConfigIdx>;
        Kernel conv;

        GroupedConvFwdHostArgs<> host_args(
            param, inputs.input, inputs.weight, {}, outputs.output, 1);
        auto kargs = Kernel::MakeKernelArgs(host_args);

        if(!Kernel::IsSupportedArgument(kargs))
            return {false, 0.0f, conv.GetInstanceString()};

        auto callable = [&](const ck_tile::stream_config& sc) {
            direct_conv::grouped_4c::launch(ConfigIdx,
                                            kargs.lp,
                                            kargs.par,
                                            kargs.in_ptr,
                                            kargs.wei_ptr,
                                            kargs.out_ptr,
                                            nullptr,
                                            sc.stream_id_);
        };

        float avg_time = ck_tile::launch_kernel(s_conf, callable);
        return {true, avg_time, conv.GetInstanceString()};
    }
    else if constexpr(ck_tile::builder::ConvDirectionIsBackwardData<SIGNATURE>)
    {
        using Kernel = direct_conv::DirectConvBwdData4CFp16Kernel<ConfigIdx>;
        Kernel conv;

        GroupedConvBwdDataHostArgs host_args(
            param, outputs.input, inputs.weight, {}, inputs.output, 1);
        auto kargs = Kernel::MakeKernelArgs(host_args);

        if(!Kernel::IsSupportedArgument(kargs))
            return {false, 0.0f, conv.GetInstanceString()};

        auto callable = [&](const ck_tile::stream_config& sc) {
            direct_conv::grouped_4c::launch(ConfigIdx,
                                            kargs.lp,
                                            kargs.par,
                                            kargs.in_ptr,
                                            kargs.wei_ptr,
                                            kargs.out_ptr,
                                            nullptr,
                                            sc.stream_id_);
        };

        float avg_time = ck_tile::launch_kernel(s_conf, callable);
        return {true, avg_time, conv.GetInstanceString()};
    }
    else
    {
        return {false, 0.0f, "direct_conv_4c_fp16_unsupported_direction"};
    }
}

} // namespace ck_tile::builder::profiling
