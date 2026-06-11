// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// CRTP base class
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_kernel_wrapper.hpp"

// 8-channel kernel impls
#include "ck_tile/ops/direct_convolution/kernel/impl/grouped_8c_tile_conv_impl_v2.hpp"

namespace ck_tile::direct_conv {

// ============================================================================
// Variant accessor structs — 8c
// ============================================================================

template <Version V, DataType DT = DataType::fp16>
struct TileConvVariant8c;

template <DataType DT>
struct TileConvVariant8c<Version::v2, DT>
{
    static bool is_applicable(const Conv2dParams& par)
    {
        return grouped_8c_tile::v2::is_applicable<DT>(par);
    }

    template <auto Cfg>
    static bool is_config_compatible(const Conv2dParams& par)
    {
        return grouped_8c_tile::v2::is_valid_config<DT>(par, Cfg);
    }

    template <auto Cfg>
    static LaunchParams get_launch_params(const Conv2dParams& par)
    {
        return grouped_8c_tile::v2::get_launch_params<Cfg>(par);
    }

    template <auto Cfg>
    static void launch_kernel(const LaunchParams& lp,
                              const Conv2dParams& par,
                              const void* in,
                              const void* wei,
                              void* out,
                              hipStream_t stream)
    {
        grouped_8c_tile::v2::launch_kernel<Cfg, DT>(lp, par, in, wei, out, stream);
    }
};

// ============================================================================
// Concrete kernel wrappers — 8c
// ============================================================================

template <auto Cfg, Version Ver = Version::v2, DataType DT = DataType::fp16>
struct DirectTileConvForward8CKernel
    : DirectConvKernel<DirectTileConvForward8CKernel<Cfg, Ver, DT>, Cfg>
{
    using V                             = TileConvVariant8c<Ver, DT>;
    static constexpr bool kIsFprop      = true;
    static constexpr DataType kDataType = DT;
    static std::string GetNamePrefix()
    {
        if constexpr(DT == DataType::bf16)
            return "direct_tile_conv_bf16_fwd_";
        else
            return "direct_tile_conv_fp16_fwd_";
    }
};

template <auto Cfg, Version Ver = Version::v2, DataType DT = DataType::fp16>
struct DirectTileConvBwdData8CKernel
    : DirectConvKernel<DirectTileConvBwdData8CKernel<Cfg, Ver, DT>, Cfg>
{
    using V                             = TileConvVariant8c<Ver, DT>;
    static constexpr bool kIsFprop      = false;
    static constexpr DataType kDataType = DT;
    static std::string GetNamePrefix()
    {
        if constexpr(DT == DataType::bf16)
            return "direct_tile_conv_bf16_bwd_data_";
        else
            return "direct_tile_conv_fp16_bwd_data_";
    }
};

} // namespace ck_tile::direct_conv
