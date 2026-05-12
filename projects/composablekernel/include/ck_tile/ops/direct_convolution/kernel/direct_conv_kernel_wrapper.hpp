// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <tuple>

#include "ck_tile/ops/direct_convolution/utils/common.hpp"
#include "ck_tile/ops/direct_convolution/utils/conv_params.hpp"
#include "ck_tile/ops/direct_convolution/utils/launch_params.hpp"
#include "ck_tile/ops/grouped_convolution/utils/grouped_conv_host_args.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace ck_tile::direct_conv {

/// CRTP base class for all direct convolution kernel wrappers.
///
/// Derived must provide:
///   using V = <VariantAccessor>;  // has: configs, get_launch_params, launch, make_variant
///   static constexpr bool kIsFprop = true/false;
///   static std::string GetNamePrefix();  // e.g. "direct_tile_conv_fp16_fwd_"
///
template <typename Derived, int ConfigIdx>
struct DirectConvKernel
{
    struct KernelArgs
    {
        Conv2dParams par;
        LaunchParams lp;
        const void* in_ptr;
        const void* wei_ptr;
        void* out_ptr;
    };

    std::string GetName() const
    {
        return Derived::GetNamePrefix() + Derived::V::configs[ConfigIdx].GetName();
    }

    std::string GetTypeString() const { return GetName(); }

    std::string GetInstanceString() const { return GetName(); }

    /// Convert shared host-side Fprop arguments to kernel-specific arguments.
    static KernelArgs MakeKernelArgs(const GroupedConvFwdHostArgs<>& host_args)
        requires(Derived::kIsFprop)
    {
        auto par       = PopulateConv2dParams(host_args);
        par.direction  = Direction::Fprop;
        par.compute_output_size();

        auto lp = Derived::V::get_launch_params(ConfigIdx, par);
        return {par, lp, host_args.in_ptr, host_args.wei_ptr, host_args.out_ptr};
    }

    /// Convert shared host-side Dgrad arguments to kernel-specific arguments.
    ///
    /// Note the pointer swapping: GroupedConvBwdDataHostArgs uses:
    ///   in_ptr  -> input gradient (written by kernel)
    ///   out_ptr -> output gradient (read by kernel)
    /// Our kernel launch() expects:
    ///   in_ptr  -> data to read (= output gradient = host_args.out_ptr)
    ///   out_ptr -> data to write (= input gradient = host_args.in_ptr)
    static KernelArgs MakeKernelArgs(const GroupedConvBwdDataHostArgs& host_args)
        requires(!Derived::kIsFprop)
    {
        auto par       = PopulateConv2dParams(host_args);
        par.direction  = Direction::Dgrad;
        par.compute_output_size();

        auto lp = Derived::V::get_launch_params(ConfigIdx, par);
        // Swap pointers: kernel reads output gradient, writes input gradient
        return {par, lp, host_args.out_ptr, host_args.wei_ptr, host_args.in_ptr};
    }

    static bool IsSupportedArgument(const KernelArgs& kargs)
    {
        auto variant = Derived::V::make_variant();
        return variant.is_applicable(kargs.par) &&
               variant.config_is_compatible(kargs.par, ConfigIdx);
    }

    static dim3 GridSize(const KernelArgs& kargs) { return kargs.lp.grid; }

    static dim3 BlockSize()
    {
        return dim3(static_cast<unsigned>(Derived::V::configs[ConfigIdx].block_size()));
    }

    static constexpr ck_tile::index_t GetSmemSize() { return 0; }

    /// Run the kernel with timing via stream_config.
    /// Returns {is_supported, avg_time_ms, instance_name}.
    std::tuple<bool, float, std::string>
    Run(const KernelArgs& kargs, const ck_tile::stream_config& s_conf) const
    {
        if(!IsSupportedArgument(kargs))
            return {false, 0.0f, GetInstanceString()};

        auto callable = [&](const ck_tile::stream_config& sc) {
            Derived::V::launch(ConfigIdx,
                               kargs.lp,
                               kargs.par,
                               kargs.in_ptr,
                               kargs.wei_ptr,
                               kargs.out_ptr,
                               nullptr,
                               sc.stream_id_);
        };

        float avg_time = ck_tile::launch_kernel(s_conf, callable);
        return {true, avg_time, GetInstanceString()};
    }

private:
    template <typename HostArgs>
    static Conv2dParams PopulateConv2dParams(const HostArgs& host_args)
    {
        Conv2dParams par;
        par.n          = static_cast<int>(host_args.N_);
        par.h          = static_cast<int>(host_args.input_spatial_lengths_[0]);
        par.w          = static_cast<int>(host_args.input_spatial_lengths_[1]);
        par.c_tot      = static_cast<int>(host_args.G_ * host_args.C_);
        par.k_tot      = static_cast<int>(host_args.G_ * host_args.K_);
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
        return par;
    }
};

} // namespace ck_tile::direct_conv
