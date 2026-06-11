// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// CRTP base class
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_kernel_wrapper.hpp"

// Non-grouped (standard) 32-channel kernel impl
#include "ck_tile/ops/direct_convolution/kernel/impl/conv_32c_tile_impl_v3.hpp"

namespace ck_tile::direct_conv {

// ============================================================================
// Variant accessor struct — 32c dense (non-grouped)
// ============================================================================

// Non-grouped (standard) conv — 32c MFMA with C-reduction (v3 cross-wave LDS reduction)
template <Version V, DataType DT = DataType::fp16>
struct TileConvVariant32cDense;

template <DataType DT>
struct TileConvVariant32cDense<Version::v3, DT>
{
    static bool is_applicable(const Conv2dParams& par)
    {
        return conv_32c_tile::v3::is_applicable<DT>(par);
    }

    template <auto Cfg>
    static bool is_config_compatible(const Conv2dParams& par)
    {
        return conv_32c_tile::v3::is_valid_config<DT>(par, Cfg);
    }

    template <auto Cfg>
    static LaunchParams get_launch_params(const Conv2dParams& par)
    {
        return conv_32c_tile::v3::get_launch_params<Cfg>(par);
    }

    template <auto Cfg>
    static void launch_kernel(const LaunchParams& lp,
                              const Conv2dParams& par,
                              const void* in,
                              const void* wei,
                              void* out,
                              hipStream_t stream)
    {
        conv_32c_tile::v3::launch_kernel<Cfg, DT>(lp, par, in, wei, out, stream);
    }
};

// ============================================================================
// Concrete kernel wrappers — 32c dense (non-grouped)
// ============================================================================

template <auto Cfg, Version Ver = Version::v3, DataType DT = DataType::fp16>
struct DirectTileConvForward32CDenseKernel
    : DirectConvKernel<DirectTileConvForward32CDenseKernel<Cfg, Ver, DT>, Cfg>
{
    using V                             = TileConvVariant32cDense<Ver, DT>;
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

template <auto Cfg, Version Ver = Version::v3, DataType DT = DataType::fp16>
struct DirectTileConvBwdData32CDenseKernel
    : DirectConvKernel<DirectTileConvBwdData32CDenseKernel<Cfg, Ver, DT>, Cfg>
{
    using V                             = TileConvVariant32cDense<Ver, DT>;
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
