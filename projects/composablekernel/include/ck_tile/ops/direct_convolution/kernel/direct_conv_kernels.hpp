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
#include "ck_tile/ops/direct_convolution/kernel/conv_32c_tile_impl_v1.hpp"

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
    static constexpr auto& configs         = grouped_4c_tile::v3::KernelConfigurations<DT>::configs;
    static constexpr auto  get_launch_params = &grouped_4c_tile::v3::get_launch_params<DT>;
    static constexpr auto  launch           = &grouped_4c_tile::v3::launch<DT>;
    static constexpr auto  make_variant     = &grouped_4c_tile::v3::make_variant<DT>;
};

template <Version V, DataType DT = DataType::fp16>
struct TileConvVariant8c;

template <DataType DT>
struct TileConvVariant8c<Version::v2, DT>
{
    static constexpr auto& configs         = grouped_8c_tile::v2::KernelConfigurations<DT>::configs;
    static constexpr auto  get_launch_params = &grouped_8c_tile::v2::get_launch_params<DT>;
    static constexpr auto  launch           = &grouped_8c_tile::v2::launch<DT>;
    static constexpr auto  make_variant     = &grouped_8c_tile::v2::make_variant<DT>;
};

template <Version V, DataType DT = DataType::fp16>
struct TileConvVariant16c;

template <DataType DT>
struct TileConvVariant16c<Version::v2, DT>
{
    static constexpr auto& configs         = grouped_16c_tile::v2::KernelConfigurations<DT>::configs;
    static constexpr auto  get_launch_params = &grouped_16c_tile::v2::get_launch_params<DT>;
    static constexpr auto  launch           = &grouped_16c_tile::v2::launch<DT>;
    static constexpr auto  make_variant     = &grouped_16c_tile::v2::make_variant<DT>;
};

template <Version V, DataType DT = DataType::fp16>
struct TileConvVariant32c;

template <DataType DT>
struct TileConvVariant32c<Version::v2, DT>
{
    static constexpr auto& configs         = grouped_32c_tile::v2::KernelConfigurations<DT>::configs;
    static constexpr auto  get_launch_params = &grouped_32c_tile::v2::get_launch_params<DT>;
    static constexpr auto  launch           = &grouped_32c_tile::v2::launch<DT>;
    static constexpr auto  make_variant     = &grouped_32c_tile::v2::make_variant<DT>;
};

// Non-grouped (standard) conv — 32c MFMA with C-reduction
template <Version V, DataType DT = DataType::fp16>
struct TileConvVariant32cDense;

template <DataType DT>
struct TileConvVariant32cDense<Version::v1, DT>
{
    static constexpr auto& configs         = conv_32c_tile::v1::KernelConfigurations<DT>::configs;
    static constexpr auto  get_launch_params = &conv_32c_tile::v1::get_launch_params<DT>;
    static constexpr auto  launch           = &conv_32c_tile::v1::launch<DT>;
    static constexpr auto  make_variant     = &conv_32c_tile::v1::make_variant<DT>;
};

// ============================================================================
// Variant accessor structs — HipConv
// ============================================================================

struct HipConvVariant4c
{
    static constexpr auto& configs         = ck_tile::direct_hip_conv::grouped_4c::configs;
    static constexpr auto  get_launch_params = &ck_tile::direct_hip_conv::grouped_4c::get_launch_params;
    static constexpr auto  launch           = &ck_tile::direct_hip_conv::grouped_4c::launch;
    static constexpr auto  make_variant     = &ck_tile::direct_hip_conv::grouped_4c::make_variant;
};

struct HipConvVariant8c
{
    static constexpr auto& configs         = ck_tile::direct_hip_conv::grouped_8c::configs;
    static constexpr auto  get_launch_params = &ck_tile::direct_hip_conv::grouped_8c::get_launch_params;
    static constexpr auto  launch           = &ck_tile::direct_hip_conv::grouped_8c::launch;
    static constexpr auto  make_variant     = &ck_tile::direct_hip_conv::grouped_8c::make_variant;
};

struct HipConvVariant16c
{
    static constexpr auto& configs         = ck_tile::direct_hip_conv::grouped_16c::configs;
    static constexpr auto  get_launch_params = &ck_tile::direct_hip_conv::grouped_16c::get_launch_params;
    static constexpr auto  launch           = &ck_tile::direct_hip_conv::grouped_16c::launch;
    static constexpr auto  make_variant     = &ck_tile::direct_hip_conv::grouped_16c::make_variant;
};

struct HipConvVariant32c
{
    static constexpr auto& configs         = ck_tile::direct_hip_conv::grouped_32c::configs;
    static constexpr auto  get_launch_params = &ck_tile::direct_hip_conv::grouped_32c::get_launch_params;
    static constexpr auto  launch           = &ck_tile::direct_hip_conv::grouped_32c::launch;
    static constexpr auto  make_variant     = &ck_tile::direct_hip_conv::grouped_32c::make_variant;
};

// ============================================================================
// Concrete kernel wrappers — TileConv
// ============================================================================

// 4c TileConv
template <int ConfigIdx, Version Ver = Version::v3, DataType DT = DataType::fp16>
struct DirectTileConvForward4CKernel
    : DirectConvKernel<DirectTileConvForward4CKernel<ConfigIdx, Ver, DT>, ConfigIdx>
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
    : DirectConvKernel<DirectTileConvBwdData4CKernel<ConfigIdx, Ver, DT>, ConfigIdx>
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
    : DirectConvKernel<DirectTileConvForward8CKernel<ConfigIdx, Ver, DT>, ConfigIdx>
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
    : DirectConvKernel<DirectTileConvBwdData8CKernel<ConfigIdx, Ver, DT>, ConfigIdx>
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
    : DirectConvKernel<DirectTileConvForward16CKernel<ConfigIdx, Ver, DT>, ConfigIdx>
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
    : DirectConvKernel<DirectTileConvBwdData16CKernel<ConfigIdx, Ver, DT>, ConfigIdx>
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
    : DirectConvKernel<DirectTileConvForward32CKernel<ConfigIdx, Ver, DT>, ConfigIdx>
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
    : DirectConvKernel<DirectTileConvBwdData32CKernel<ConfigIdx, Ver, DT>, ConfigIdx>
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
template <int ConfigIdx, Version Ver = Version::v1, DataType DT = DataType::fp16>
struct DirectTileConvForward32CDenseKernel
    : DirectConvKernel<DirectTileConvForward32CDenseKernel<ConfigIdx, Ver, DT>, ConfigIdx>
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

template <int ConfigIdx, Version Ver = Version::v1, DataType DT = DataType::fp16>
struct DirectTileConvBwdData32CDenseKernel
    : DirectConvKernel<DirectTileConvBwdData32CDenseKernel<ConfigIdx, Ver, DT>, ConfigIdx>
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
    : DirectConvKernel<DirectHipConvForward4CFp16Kernel<ConfigIdx>, ConfigIdx>
{
    using V = HipConvVariant4c;
    static constexpr bool kIsFprop = true;
    static constexpr DataType kDataType = DataType::fp16;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_fwd_"; }
};

template <int ConfigIdx>
struct DirectHipConvBwdData4CFp16Kernel
    : DirectConvKernel<DirectHipConvBwdData4CFp16Kernel<ConfigIdx>, ConfigIdx>
{
    using V = HipConvVariant4c;
    static constexpr bool kIsFprop = false;
    static constexpr DataType kDataType = DataType::fp16;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_bwd_data_"; }
};

// 8c HipConv
template <int ConfigIdx>
struct DirectHipConvForward8CFp16Kernel
    : DirectConvKernel<DirectHipConvForward8CFp16Kernel<ConfigIdx>, ConfigIdx>
{
    using V = HipConvVariant8c;
    static constexpr bool kIsFprop = true;
    static constexpr DataType kDataType = DataType::fp16;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_fwd_"; }
};

template <int ConfigIdx>
struct DirectHipConvBwdData8CFp16Kernel
    : DirectConvKernel<DirectHipConvBwdData8CFp16Kernel<ConfigIdx>, ConfigIdx>
{
    using V = HipConvVariant8c;
    static constexpr bool kIsFprop = false;
    static constexpr DataType kDataType = DataType::fp16;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_bwd_data_"; }
};

// 16c HipConv
template <int ConfigIdx>
struct DirectHipConvForward16CFp16Kernel
    : DirectConvKernel<DirectHipConvForward16CFp16Kernel<ConfigIdx>, ConfigIdx>
{
    using V = HipConvVariant16c;
    static constexpr bool kIsFprop = true;
    static constexpr DataType kDataType = DataType::fp16;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_fwd_"; }
};

template <int ConfigIdx>
struct DirectHipConvBwdData16CFp16Kernel
    : DirectConvKernel<DirectHipConvBwdData16CFp16Kernel<ConfigIdx>, ConfigIdx>
{
    using V = HipConvVariant16c;
    static constexpr bool kIsFprop = false;
    static constexpr DataType kDataType = DataType::fp16;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_bwd_data_"; }
};

// 32c HipConv
template <int ConfigIdx>
struct DirectHipConvForward32CFp16Kernel
    : DirectConvKernel<DirectHipConvForward32CFp16Kernel<ConfigIdx>, ConfigIdx>
{
    using V = HipConvVariant32c;
    static constexpr bool kIsFprop = true;
    static constexpr DataType kDataType = DataType::fp16;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_fwd_"; }
};

template <int ConfigIdx>
struct DirectHipConvBwdData32CFp16Kernel
    : DirectConvKernel<DirectHipConvBwdData32CFp16Kernel<ConfigIdx>, ConfigIdx>
{
    using V = HipConvVariant32c;
    static constexpr bool kIsFprop = false;
    static constexpr DataType kDataType = DataType::fp16;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_bwd_data_"; }
};

} // namespace ck_tile::direct_conv
