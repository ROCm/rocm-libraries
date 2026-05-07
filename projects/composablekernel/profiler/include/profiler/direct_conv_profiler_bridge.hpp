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
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_fwd_4c_fp16_hip_conv_kernel.hpp"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_bwd_data_4c_fp16_hip_conv_kernel.hpp"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_fwd_4c_fp16_tile_conv_kernel.hpp"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_bwd_data_4c_fp16_tile_conv_kernel.hpp"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_fwd_16c_fp16_hip_conv_kernel.hpp"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_bwd_data_16c_fp16_hip_conv_kernel.hpp"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_fwd_16c_fp16_tile_conv_kernel.hpp"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_bwd_data_16c_fp16_tile_conv_kernel.hpp"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_fwd_8c_fp16_hip_conv_kernel.hpp"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_bwd_data_8c_fp16_hip_conv_kernel.hpp"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_fwd_8c_fp16_tile_conv_kernel.hpp"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_bwd_data_8c_fp16_tile_conv_kernel.hpp"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_fwd_32c_fp16_hip_conv_kernel.hpp"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_bwd_data_32c_fp16_hip_conv_kernel.hpp"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_fwd_32c_fp16_tile_conv_kernel.hpp"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_bwd_data_32c_fp16_tile_conv_kernel.hpp"
#pragma clang diagnostic pop

namespace ck_tile::builder::profiling {

namespace ckt = ck_tile::builder::test;

/// Generic bridge function that adapts the profiler's run_alg callback signature
/// to any direct convolution kernel wrapper struct.
///
/// The Kernel type must provide:
///   - MakeKernelArgs(host_args) — convert from GroupedConvHostArgs to kernel args
///   - Run(kargs, s_conf) — returns tuple<bool, float, string>
///   - GetInstanceString() — returns a name string
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
template <typename Kernel, auto SIGNATURE>
std::tuple<bool, float, std::string>
run_direct_conv(const ckt::Args<SIGNATURE>& args,
                const ckt::Inputs<SIGNATURE>& inputs,
                const ckt::Outputs<SIGNATURE>& outputs,
                const ck_tile::stream_config& s_conf)
{
    const auto param = args.to_ck_tile_conv_param();
    Kernel conv;

    if constexpr(ck_tile::builder::ConvDirectionIsForward<SIGNATURE>)
    {
        GroupedConvFwdHostArgs<> host_args(
            param, inputs.input, inputs.weight, {}, outputs.output, 1);
        auto kargs = Kernel::MakeKernelArgs(host_args);
        return conv.Run(kargs, s_conf);
    }
    else if constexpr(ck_tile::builder::ConvDirectionIsBackwardData<SIGNATURE>)
    {
        GroupedConvBwdDataHostArgs host_args(
            param, outputs.input, inputs.weight, {}, inputs.output, 1);
        auto kargs = Kernel::MakeKernelArgs(host_args);
        return conv.Run(kargs, s_conf);
    }
    else
    {
        return {false, 0.0f, conv.GetInstanceString()};
    }
}

} // namespace ck_tile::builder::profiling
