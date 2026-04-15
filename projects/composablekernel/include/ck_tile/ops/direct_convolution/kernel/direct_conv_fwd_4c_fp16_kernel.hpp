// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <tuple>

#include "ck_tile/ops/direct_convolution/kernel/grouped_4c_fp16_kernel.hpp"
#include "ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace ck_tile::direct_conv {

/// Wrapper struct that presents the grouped_4c Fprop kernel (at a specific config index)
/// with the same public API as the im2col-based GroupedConvolutionForwardKernel.
///
/// This enables integration into the CK profiler and builder testing infrastructure
/// which expects kernels to provide GetName(), IsSupportedArgument(), MakeKernelArgs(), etc.
template <int ConfigIdx>
struct DirectConvForward4CFp16Kernel
{
    struct KernelArgs
    {
        Conv2dParams par;
        LaunchParams lp;
        const void* in_ptr;
        const void* wei_ptr;
        void* out_ptr;
    };

    static std::string GetName()
    {
        return "direct_conv_grouped_4c_fp16_fwd_" + std::to_string(ConfigIdx);
    }

    static std::string GetTypeString() { return GetName(); }

    std::string GetInstanceString() const { return GetName(); }

    /// Convert shared host-side conv arguments to kernel-specific arguments.
    static KernelArgs MakeKernelArgs(const GroupedConvFwdHostArgs<>& host_args)
    {
        Conv2dParams par;
        par.direction  = Direction::Fprop;
        par.n          = static_cast<int>(host_args.N_);
        par.h          = static_cast<int>(host_args.input_spatial_lengths_[0]);
        par.w          = static_cast<int>(host_args.input_spatial_lengths_[1]);
        par.c          = static_cast<int>(host_args.G_ * host_args.C_);
        par.k          = static_cast<int>(host_args.G_ * host_args.K_);
        par.kh         = static_cast<int>(host_args.filter_spatial_lengths_[0]);
        par.kw         = static_cast<int>(host_args.filter_spatial_lengths_[1]);
        par.pad_h      = static_cast<int>(host_args.input_left_pads_[0]);
        par.pad_w      = static_cast<int>(host_args.input_left_pads_[1]);
        par.stride_h   = static_cast<int>(host_args.conv_filter_strides_[0]);
        par.stride_w   = static_cast<int>(host_args.conv_filter_strides_[1]);
        par.dilation_h = static_cast<int>(host_args.conv_filter_dilations_[0]);
        par.dilation_w = static_cast<int>(host_args.conv_filter_dilations_[1]);
        par.groups     = static_cast<int>(host_args.G_);
        par.in_type    = DataType::fp16;
        par.wei_type   = DataType::fp16;
        par.out_type   = DataType::fp16;
        par.order      = TensorOrder::NHWC;
        par.compute_output_size();

        auto lp = grouped_4c::get_launch_params(ConfigIdx, par);

        return {par, lp, host_args.in_ptr, host_args.wei_ptr, host_args.out_ptr};
    }

    static bool IsSupportedArgument(const KernelArgs& kargs)
    {
        auto variant = grouped_4c::make_variant();
        return variant.is_applicable(kargs.par) &&
               variant.config_is_compatible(kargs.par, ConfigIdx);
    }

    static dim3 GridSize(const KernelArgs& kargs) { return kargs.lp.grid; }

    static dim3 BlockSize()
    {
        return dim3(static_cast<unsigned>(grouped_4c::configs[ConfigIdx].block_size()));
    }

    static constexpr ck_tile::index_t GetSmemSize() { return 0; }
};

/// Launch the forward direct convolution kernel via the profiler / testing infrastructure.
///
/// This function follows the same pattern as the im2col detail::run() but uses
/// ck_tile::launch_kernel() with a lambda that calls grouped_4c::launch() internally.
template <int ConfigIdx>
std::tuple<bool, float, std::string>
run(DirectConvForward4CFp16Kernel<ConfigIdx>& conv,
    const ck_tile::conv::ConvParam& param,
    const void* in_ptr,
    const void* wei_ptr,
    void* out_ptr,
    const ck_tile::stream_config& s_conf)
{
    using Kernel = DirectConvForward4CFp16Kernel<ConfigIdx>;

    GroupedConvFwdHostArgs<> host_args(param, in_ptr, wei_ptr, {}, out_ptr, 1);
    auto kargs = Kernel::MakeKernelArgs(host_args);

    if(!Kernel::IsSupportedArgument(kargs))
        return {false, 0.0f, conv.GetInstanceString()};

    auto callable = [&](const ck_tile::stream_config& sc) {
        grouped_4c::launch(ConfigIdx,
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

} // namespace ck_tile::direct_conv
