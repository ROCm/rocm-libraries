// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <tuple>

#include "ck_tile/ops/direct_convolution/kernel/grouped_32c_fp16_tile_conv_impl_v2.hpp"
#include "ck_tile/ops/direct_convolution/utils/common.hpp"
#include "ck_tile/ops/grouped_convolution/utils/grouped_conv_host_args.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace ck_tile::direct_conv {

template <Version V>
struct VersionTraits32c;

template <>
struct VersionTraits32c<Version::v2>
{
    static constexpr auto& configs = grouped_32c_tile::v2::configs;
    static constexpr auto get_launch_params = &grouped_32c_tile::v2::get_launch_params;
    static constexpr auto launch = &grouped_32c_tile::v2::launch;
    static constexpr auto make_variant = &grouped_32c_tile::v2::make_variant;
};

/// Wrapper struct that presents the grouped_32c Fprop tile conv kernel (at a specific config index)
/// with the same public API as the im2col-based GroupedConvolutionForwardKernel.
template <int ConfigIdx, Version Ver = Version::v2>
struct DirectTileConvForward32CFp16Kernel
{

    using V = VersionTraits32c<Ver>;

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
        return "direct_tile_conv_32c_fp16_fwd_" + V::configs[ConfigIdx].GetName();
    }

    std::string GetTypeString() const { return GetName(); }

    std::string GetInstanceString() const { return GetName(); }

    static KernelArgs MakeKernelArgs(const GroupedConvFwdHostArgs<>& host_args)
    {
        Conv2dParams par;
        par.direction  = Direction::Fprop;
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
        par.compute_output_size();

        auto lp = V::get_launch_params(ConfigIdx, par);
        return {par, lp, host_args.in_ptr, host_args.wei_ptr, host_args.out_ptr};
    }

    static bool IsSupportedArgument(const KernelArgs& kargs)
    {
        auto variant = V::make_variant();
        return variant.is_applicable(kargs.par) &&
               variant.config_is_compatible(kargs.par, ConfigIdx);
    }

    static dim3 GridSize(const KernelArgs& kargs) { return kargs.lp.grid; }

    static dim3 BlockSize()
    {
        return dim3(static_cast<unsigned>(V::configs[ConfigIdx].block_size()));
    }

    static constexpr ck_tile::index_t GetSmemSize() { return 0; }

    std::tuple<bool, float, std::string>
    Run(const KernelArgs& kargs, const ck_tile::stream_config& s_conf) const
    {
        if(!IsSupportedArgument(kargs))
            return {false, 0.0f, GetInstanceString()};

        auto callable = [&](const ck_tile::stream_config& sc) {
            V::launch(ConfigIdx,
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
};

} // namespace ck_tile::direct_conv
