// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// CRTP base class
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_kernel_wrapper.hpp"

// 32-channel kernel impls
#include "ck_tile/ops/direct_convolution/kernel/impl/grouped_32c_tile_conv_impl_v2.hpp"
#include "ck_tile/ops/direct_convolution/kernel/impl/grouped_32c_fp16_hip_conv_impl.hpp"

namespace ck_tile::direct_conv {

// ============================================================================
// Variant accessor structs — 32c
// ============================================================================

template <Version V, DataType DT = DataType::fp16>
struct TileConvVariant32c;

template <DataType DT>
struct TileConvVariant32c<Version::v2, DT>
{
    static bool is_applicable(const Conv2dParams& par)
    { return grouped_32c_tile::v2::is_applicable<DT>(par); }

    template <auto Cfg>
    static bool is_config_compatible(const Conv2dParams& par)
    { return grouped_32c_tile::v2::is_valid_config<DT>(par, Cfg); }

    template <auto Cfg>
    static LaunchParams get_launch_params(const Conv2dParams& par)
    { return grouped_32c_tile::v2::get_launch_params<Cfg>(par); }

    template <auto Cfg>
    static void launch_kernel(const LaunchParams& lp, const Conv2dParams& par,
                              const void* in, const void* wei, void* out, hipStream_t stream)
    { grouped_32c_tile::v2::launch_kernel<Cfg, DT>(lp, par, in, wei, out, stream); }
};

struct HipConvVariant32c
{
    static bool is_applicable(const Conv2dParams& par)
    { return ck_tile::direct_hip_conv::grouped_32c::is_applicable(par); }

    template <auto Cfg>
    static bool is_config_compatible(const Conv2dParams& par)
    { return ck_tile::direct_hip_conv::grouped_32c::is_valid_config(par, Cfg); }

    template <auto Cfg>
    static LaunchParams get_launch_params(const Conv2dParams& par)
    { return ck_tile::direct_hip_conv::grouped_32c::get_launch_params<Cfg>(par); }

    template <auto Cfg>
    static void launch_kernel(const LaunchParams& lp, const Conv2dParams& par,
                              const void* in, const void* wei, void* out, hipStream_t stream)
    { ck_tile::direct_hip_conv::grouped_32c::launch_kernel<Cfg>(lp, par, in, wei, out, stream); }
};

// ============================================================================
// Concrete kernel wrappers — 32c
// ============================================================================

template <auto Cfg, Version Ver = Version::v2, DataType DT = DataType::fp16>
struct DirectTileConvForward32CKernel
    : DirectConvKernel<DirectTileConvForward32CKernel<Cfg, Ver, DT>, Cfg>
{
    using V = TileConvVariant32c<Ver, DT>;
    static constexpr bool kIsFprop = true;
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
struct DirectTileConvBwdData32CKernel
    : DirectConvKernel<DirectTileConvBwdData32CKernel<Cfg, Ver, DT>, Cfg>
{
    using V = TileConvVariant32c<Ver, DT>;
    static constexpr bool kIsFprop = false;
    static constexpr DataType kDataType = DT;
    static std::string GetNamePrefix()
    {
        if constexpr(DT == DataType::bf16)
            return "direct_tile_conv_bf16_bwd_data_";
        else
            return "direct_tile_conv_fp16_bwd_data_";
    }
};

template <auto Cfg>
struct DirectHipConvForward32CFp16Kernel
    : DirectConvKernel<DirectHipConvForward32CFp16Kernel<Cfg>, Cfg>
{
    using V = HipConvVariant32c;
    static constexpr bool kIsFprop = true;
    static constexpr DataType kDataType = DataType::fp16;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_fwd_"; }
};

template <auto Cfg>
struct DirectHipConvBwdData32CFp16Kernel
    : DirectConvKernel<DirectHipConvBwdData32CFp16Kernel<Cfg>, Cfg>
{
    using V = HipConvVariant32c;
    static constexpr bool kIsFprop = false;
    static constexpr DataType kDataType = DataType::fp16;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_bwd_data_"; }
};

} // namespace ck_tile::direct_conv
