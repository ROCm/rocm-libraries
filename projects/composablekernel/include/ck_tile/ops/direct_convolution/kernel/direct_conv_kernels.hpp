// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// CRTP base class
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_kernel_wrapper.hpp"

// CK Tile conv impl headers
#include "ck_tile/ops/direct_convolution/kernel/grouped_4c_tile_conv_impl_v3.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_8c_tile_conv_impl_v2.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_16c_tile_conv_impl_v2.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_32c_tile_conv_impl_v2.hpp"
#include "ck_tile/ops/direct_convolution/kernel/conv_32c_tile_impl_v3.hpp"

// HIP conv impl headers
#include "ck_tile/ops/direct_convolution/kernel/grouped_4c_fp16_hip_conv_impl.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_8c_fp16_hip_conv_impl.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_16c_fp16_hip_conv_impl.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_32c_fp16_hip_conv_impl.hpp"

namespace ck_tile::direct_conv {

// ============================================================================
// Variant accessor structs — TileConv
// ============================================================================

template <Version V, DataType DT = DataType::fp16>
struct TileConvVariant4c;

template <DataType DT>
struct TileConvVariant4c<Version::v3, DT>
{
    static constexpr auto& configs_map = grouped_4c_tile::v3::KernelConfigurations<DT>::configs_map;

    static bool is_applicable(const Conv2dParams& par)
    { return grouped_4c_tile::v3::is_applicable<DT>(par); }

    template <auto Cfg>
    static bool is_config_compatible(const Conv2dParams& par)
    { return grouped_4c_tile::v3::is_valid_config<DT>(par, Cfg); }

    template <auto Cfg>
    static LaunchParams get_launch_params(const Conv2dParams& par)
    { return grouped_4c_tile::v3::get_launch_params<Cfg>(par); }

    template <auto Cfg>
    static void launch_kernel(const LaunchParams& lp, const Conv2dParams& par,
                              const void* in, const void* wei, void* out, hipStream_t stream)
    { grouped_4c_tile::v3::launch_kernel<Cfg, DT>(lp, par, in, wei, out, stream); }
};

template <Version V, DataType DT = DataType::fp16>
struct TileConvVariant8c;

template <DataType DT>
struct TileConvVariant8c<Version::v2, DT>
{
    static constexpr auto& configs_map = grouped_8c_tile::v2::KernelConfigurations<DT>::configs_map;

    static bool is_applicable(const Conv2dParams& par)
    { return grouped_8c_tile::v2::is_applicable<DT>(par); }

    template <auto Cfg>
    static bool is_config_compatible(const Conv2dParams& par)
    { return grouped_8c_tile::v2::is_valid_config<DT>(par, Cfg); }

    template <auto Cfg>
    static LaunchParams get_launch_params(const Conv2dParams& par)
    { return grouped_8c_tile::v2::get_launch_params<Cfg>(par); }

    template <auto Cfg>
    static void launch_kernel(const LaunchParams& lp, const Conv2dParams& par,
                              const void* in, const void* wei, void* out, hipStream_t stream)
    { grouped_8c_tile::v2::launch_kernel<Cfg, DT>(lp, par, in, wei, out, stream); }
};

template <Version V, DataType DT = DataType::fp16>
struct TileConvVariant16c;

template <DataType DT>
struct TileConvVariant16c<Version::v2, DT>
{
    static constexpr auto& configs_map = grouped_16c_tile::v2::KernelConfigurations<DT>::configs_map;

    static bool is_applicable(const Conv2dParams& par)
    { return grouped_16c_tile::v2::is_applicable<DT>(par); }

    template <auto Cfg>
    static bool is_config_compatible(const Conv2dParams& par)
    { return grouped_16c_tile::v2::is_valid_config<DT>(par, Cfg); }

    template <auto Cfg>
    static LaunchParams get_launch_params(const Conv2dParams& par)
    { return grouped_16c_tile::v2::get_launch_params<Cfg>(par); }

    template <auto Cfg>
    static void launch_kernel(const LaunchParams& lp, const Conv2dParams& par,
                              const void* in, const void* wei, void* out, hipStream_t stream)
    { grouped_16c_tile::v2::launch_kernel<Cfg, DT>(lp, par, in, wei, out, stream); }
};

template <Version V, DataType DT = DataType::fp16>
struct TileConvVariant32c;

template <DataType DT>
struct TileConvVariant32c<Version::v2, DT>
{
    static constexpr auto& configs_map = grouped_32c_tile::v2::KernelConfigurations<DT>::configs_map;

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

// Non-grouped (standard) conv — 32c MFMA with C-reduction (v3 cross-wave LDS reduction)
template <Version V, DataType DT = DataType::fp16>
struct TileConvVariant32cDense;

template <DataType DT>
struct TileConvVariant32cDense<Version::v3, DT>
{
    static constexpr auto& configs_map = conv_32c_tile::v3::KernelConfigurations<DT>::configs_map;

    static bool is_applicable(const Conv2dParams& par)
    { return conv_32c_tile::v3::is_applicable<DT>(par); }

    template <auto Cfg>
    static bool is_config_compatible(const Conv2dParams& par)
    { return conv_32c_tile::v3::is_valid_config<DT>(par, Cfg); }

    template <auto Cfg>
    static LaunchParams get_launch_params(const Conv2dParams& par)
    { return conv_32c_tile::v3::get_launch_params<Cfg>(par); }

    template <auto Cfg>
    static void launch_kernel(const LaunchParams& lp, const Conv2dParams& par,
                              const void* in, const void* wei, void* out, hipStream_t stream)
    { conv_32c_tile::v3::launch_kernel<Cfg, DT>(lp, par, in, wei, out, stream); }
};

// ============================================================================
// Variant accessor structs — HipConv
// ============================================================================

struct HipConvVariant4c
{
    static constexpr auto& configs_map = ck_tile::direct_hip_conv::grouped_4c::configs_map;

    static bool is_applicable(const Conv2dParams& par)
    { return ck_tile::direct_hip_conv::grouped_4c::is_applicable(par); }

    template <auto Cfg>
    static bool is_config_compatible(const Conv2dParams& par)
    { return ck_tile::direct_hip_conv::grouped_4c::is_valid_config(par, Cfg); }

    template <auto Cfg>
    static LaunchParams get_launch_params(const Conv2dParams& par)
    { return ck_tile::direct_hip_conv::grouped_4c::get_launch_params<Cfg>(par); }

    template <auto Cfg>
    static void launch_kernel(const LaunchParams& lp, const Conv2dParams& par,
                              const void* in, const void* wei, void* out, hipStream_t stream)
    { ck_tile::direct_hip_conv::grouped_4c::launch_kernel<Cfg>(lp, par, in, wei, out, stream); }
};

struct HipConvVariant8c
{
    static constexpr auto& configs_map = ck_tile::direct_hip_conv::grouped_8c::configs_map;

    static bool is_applicable(const Conv2dParams& par)
    { return ck_tile::direct_hip_conv::grouped_8c::is_applicable(par); }

    template <auto Cfg>
    static bool is_config_compatible(const Conv2dParams& par)
    { return ck_tile::direct_hip_conv::grouped_8c::is_valid_config(par, Cfg); }

    template <auto Cfg>
    static LaunchParams get_launch_params(const Conv2dParams& par)
    { return ck_tile::direct_hip_conv::grouped_8c::get_launch_params<Cfg>(par); }

    template <auto Cfg>
    static void launch_kernel(const LaunchParams& lp, const Conv2dParams& par,
                              const void* in, const void* wei, void* out, hipStream_t stream)
    { ck_tile::direct_hip_conv::grouped_8c::launch_kernel<Cfg>(lp, par, in, wei, out, stream); }
};

struct HipConvVariant16c
{
    static constexpr auto& configs_map = ck_tile::direct_hip_conv::grouped_16c::configs_map;

    static bool is_applicable(const Conv2dParams& par)
    { return ck_tile::direct_hip_conv::grouped_16c::is_applicable(par); }

    template <auto Cfg>
    static bool is_config_compatible(const Conv2dParams& par)
    { return ck_tile::direct_hip_conv::grouped_16c::is_valid_config(par, Cfg); }

    template <auto Cfg>
    static LaunchParams get_launch_params(const Conv2dParams& par)
    { return ck_tile::direct_hip_conv::grouped_16c::get_launch_params<Cfg>(par); }

    template <auto Cfg>
    static void launch_kernel(const LaunchParams& lp, const Conv2dParams& par,
                              const void* in, const void* wei, void* out, hipStream_t stream)
    { ck_tile::direct_hip_conv::grouped_16c::launch_kernel<Cfg>(lp, par, in, wei, out, stream); }
};

struct HipConvVariant32c
{
    static constexpr auto& configs_map = ck_tile::direct_hip_conv::grouped_32c::configs_map;

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
// Concrete kernel wrappers — TileConv
// ============================================================================

// 4c TileConv
template <int ConfigIdx, Version Ver = Version::v3, DataType DT = DataType::fp16>
struct DirectTileConvForward4CKernel
    : DirectConvKernel<DirectTileConvForward4CKernel<ConfigIdx, Ver, DT>,
                       TileConvVariant4c<Ver, DT>::configs_map.get(ConfigIdx)>
{
    using V = TileConvVariant4c<Ver, DT>;
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

template <int ConfigIdx, Version Ver = Version::v3, DataType DT = DataType::fp16>
struct DirectTileConvBwdData4CKernel
    : DirectConvKernel<DirectTileConvBwdData4CKernel<ConfigIdx, Ver, DT>,
                       TileConvVariant4c<Ver, DT>::configs_map.get(ConfigIdx)>
{
    using V = TileConvVariant4c<Ver, DT>;
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

// 8c TileConv
template <int ConfigIdx, Version Ver = Version::v2, DataType DT = DataType::fp16>
struct DirectTileConvForward8CKernel
    : DirectConvKernel<DirectTileConvForward8CKernel<ConfigIdx, Ver, DT>,
                       TileConvVariant8c<Ver, DT>::configs_map.get(ConfigIdx)>
{
    using V = TileConvVariant8c<Ver, DT>;
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

template <int ConfigIdx, Version Ver = Version::v2, DataType DT = DataType::fp16>
struct DirectTileConvBwdData8CKernel
    : DirectConvKernel<DirectTileConvBwdData8CKernel<ConfigIdx, Ver, DT>,
                       TileConvVariant8c<Ver, DT>::configs_map.get(ConfigIdx)>
{
    using V = TileConvVariant8c<Ver, DT>;
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

// 16c TileConv
template <int ConfigIdx, Version Ver = Version::v2, DataType DT = DataType::fp16>
struct DirectTileConvForward16CKernel
    : DirectConvKernel<DirectTileConvForward16CKernel<ConfigIdx, Ver, DT>,
                       TileConvVariant16c<Ver, DT>::configs_map.get(ConfigIdx)>
{
    using V = TileConvVariant16c<Ver, DT>;
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

template <int ConfigIdx, Version Ver = Version::v2, DataType DT = DataType::fp16>
struct DirectTileConvBwdData16CKernel
    : DirectConvKernel<DirectTileConvBwdData16CKernel<ConfigIdx, Ver, DT>,
                       TileConvVariant16c<Ver, DT>::configs_map.get(ConfigIdx)>
{
    using V = TileConvVariant16c<Ver, DT>;
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

// 32c TileConv
template <int ConfigIdx, Version Ver = Version::v2, DataType DT = DataType::fp16>
struct DirectTileConvForward32CKernel
    : DirectConvKernel<DirectTileConvForward32CKernel<ConfigIdx, Ver, DT>,
                       TileConvVariant32c<Ver, DT>::configs_map.get(ConfigIdx)>
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

template <int ConfigIdx, Version Ver = Version::v2, DataType DT = DataType::fp16>
struct DirectTileConvBwdData32CKernel
    : DirectConvKernel<DirectTileConvBwdData32CKernel<ConfigIdx, Ver, DT>,
                       TileConvVariant32c<Ver, DT>::configs_map.get(ConfigIdx)>
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

// Non-grouped (standard) conv 32c TileConv
template <int ConfigIdx, Version Ver = Version::v3, DataType DT = DataType::fp16>
struct DirectTileConvForward32CDenseKernel
    : DirectConvKernel<DirectTileConvForward32CDenseKernel<ConfigIdx, Ver, DT>,
                       TileConvVariant32cDense<Ver, DT>::configs_map.get(ConfigIdx)>
{
    using V = TileConvVariant32cDense<Ver, DT>;
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

template <int ConfigIdx, Version Ver = Version::v3, DataType DT = DataType::fp16>
struct DirectTileConvBwdData32CDenseKernel
    : DirectConvKernel<DirectTileConvBwdData32CDenseKernel<ConfigIdx, Ver, DT>,
                       TileConvVariant32cDense<Ver, DT>::configs_map.get(ConfigIdx)>
{
    using V = TileConvVariant32cDense<Ver, DT>;
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

// ============================================================================
// Concrete kernel wrappers — HipConv
// ============================================================================

// 4c HipConv
template <int ConfigIdx>
struct DirectHipConvForward4CFp16Kernel
    : DirectConvKernel<DirectHipConvForward4CFp16Kernel<ConfigIdx>,
                       HipConvVariant4c::configs_map.get(ConfigIdx)>
{
    using V = HipConvVariant4c;
    static constexpr bool kIsFprop = true;
    static constexpr DataType kDataType = DataType::fp16;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_fwd_"; }
};

template <int ConfigIdx>
struct DirectHipConvBwdData4CFp16Kernel
    : DirectConvKernel<DirectHipConvBwdData4CFp16Kernel<ConfigIdx>,
                       HipConvVariant4c::configs_map.get(ConfigIdx)>
{
    using V = HipConvVariant4c;
    static constexpr bool kIsFprop = false;
    static constexpr DataType kDataType = DataType::fp16;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_bwd_data_"; }
};

// 8c HipConv
template <int ConfigIdx>
struct DirectHipConvForward8CFp16Kernel
    : DirectConvKernel<DirectHipConvForward8CFp16Kernel<ConfigIdx>,
                       HipConvVariant8c::configs_map.get(ConfigIdx)>
{
    using V = HipConvVariant8c;
    static constexpr bool kIsFprop = true;
    static constexpr DataType kDataType = DataType::fp16;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_fwd_"; }
};

template <int ConfigIdx>
struct DirectHipConvBwdData8CFp16Kernel
    : DirectConvKernel<DirectHipConvBwdData8CFp16Kernel<ConfigIdx>,
                       HipConvVariant8c::configs_map.get(ConfigIdx)>
{
    using V = HipConvVariant8c;
    static constexpr bool kIsFprop = false;
    static constexpr DataType kDataType = DataType::fp16;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_bwd_data_"; }
};

// 16c HipConv
template <int ConfigIdx>
struct DirectHipConvForward16CFp16Kernel
    : DirectConvKernel<DirectHipConvForward16CFp16Kernel<ConfigIdx>,
                       HipConvVariant16c::configs_map.get(ConfigIdx)>
{
    using V = HipConvVariant16c;
    static constexpr bool kIsFprop = true;
    static constexpr DataType kDataType = DataType::fp16;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_fwd_"; }
};

template <int ConfigIdx>
struct DirectHipConvBwdData16CFp16Kernel
    : DirectConvKernel<DirectHipConvBwdData16CFp16Kernel<ConfigIdx>,
                       HipConvVariant16c::configs_map.get(ConfigIdx)>
{
    using V = HipConvVariant16c;
    static constexpr bool kIsFprop = false;
    static constexpr DataType kDataType = DataType::fp16;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_bwd_data_"; }
};

// 32c HipConv
template <int ConfigIdx>
struct DirectHipConvForward32CFp16Kernel
    : DirectConvKernel<DirectHipConvForward32CFp16Kernel<ConfigIdx>,
                       HipConvVariant32c::configs_map.get(ConfigIdx)>
{
    using V = HipConvVariant32c;
    static constexpr bool kIsFprop = true;
    static constexpr DataType kDataType = DataType::fp16;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_fwd_"; }
};

template <int ConfigIdx>
struct DirectHipConvBwdData32CFp16Kernel
    : DirectConvKernel<DirectHipConvBwdData32CFp16Kernel<ConfigIdx>,
                       HipConvVariant32c::configs_map.get(ConfigIdx)>
{
    using V = HipConvVariant32c;
    static constexpr bool kIsFprop = false;
    static constexpr DataType kDataType = DataType::fp16;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_bwd_data_"; }
};

} // namespace ck_tile::direct_conv
