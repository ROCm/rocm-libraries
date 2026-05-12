// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// CRTP base class
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_kernel_wrapper.hpp"

// CK Tile conv impl headers
#include "ck_tile/ops/direct_convolution/kernel/grouped_4c_fp16_tile_conv_impl_v3.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_8c_fp16_tile_conv_impl_v2.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_16c_fp16_tile_conv_impl_v2.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_32c_fp16_tile_conv_impl_v2.hpp"

// HIP conv impl headers
#include "ck_tile/ops/direct_convolution/kernel/grouped_4c_fp16_hip_conv_impl.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_8c_fp16_hip_conv_impl.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_16c_fp16_hip_conv_impl.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_32c_fp16_hip_conv_impl.hpp"

namespace ck_tile::direct_conv {

// ============================================================================
// Variant accessor structs — TileConv
// ============================================================================

template <Version V>
struct TileConvVariant4c;

template <>
struct TileConvVariant4c<Version::v3>
{
    static constexpr auto& configs         = grouped_4c_tile::v3::configs;
    static constexpr auto  get_launch_params = &grouped_4c_tile::v3::get_launch_params;
    static constexpr auto  launch           = &grouped_4c_tile::v3::launch;
    static constexpr auto  make_variant     = &grouped_4c_tile::v3::make_variant;
};

template <Version V>
struct TileConvVariant8c;

template <>
struct TileConvVariant8c<Version::v2>
{
    static constexpr auto& configs         = grouped_8c_tile::v2::configs;
    static constexpr auto  get_launch_params = &grouped_8c_tile::v2::get_launch_params;
    static constexpr auto  launch           = &grouped_8c_tile::v2::launch;
    static constexpr auto  make_variant     = &grouped_8c_tile::v2::make_variant;
};

template <Version V>
struct TileConvVariant16c;

template <>
struct TileConvVariant16c<Version::v2>
{
    static constexpr auto& configs         = grouped_16c_tile::v2::configs;
    static constexpr auto  get_launch_params = &grouped_16c_tile::v2::get_launch_params;
    static constexpr auto  launch           = &grouped_16c_tile::v2::launch;
    static constexpr auto  make_variant     = &grouped_16c_tile::v2::make_variant;
};

template <Version V>
struct TileConvVariant32c;

template <>
struct TileConvVariant32c<Version::v2>
{
    static constexpr auto& configs         = grouped_32c_tile::v2::configs;
    static constexpr auto  get_launch_params = &grouped_32c_tile::v2::get_launch_params;
    static constexpr auto  launch           = &grouped_32c_tile::v2::launch;
    static constexpr auto  make_variant     = &grouped_32c_tile::v2::make_variant;
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
template <int ConfigIdx, Version Ver = Version::v3>
struct DirectTileConvForward4CFp16Kernel
    : DirectConvKernel<DirectTileConvForward4CFp16Kernel<ConfigIdx, Ver>, ConfigIdx>
{
    using V = TileConvVariant4c<Ver>;
    static constexpr bool kIsFprop = true;
    static std::string GetNamePrefix() { return "direct_tile_conv_fp16_fwd_"; }
};

template <int ConfigIdx, Version Ver = Version::v3>
struct DirectTileConvBwdData4CFp16Kernel
    : DirectConvKernel<DirectTileConvBwdData4CFp16Kernel<ConfigIdx, Ver>, ConfigIdx>
{
    using V = TileConvVariant4c<Ver>;
    static constexpr bool kIsFprop = false;
    static std::string GetNamePrefix() { return "direct_tile_conv_fp16_bwd_data_"; }
};

// 8c TileConv
template <int ConfigIdx, Version Ver = Version::v2>
struct DirectTileConvForward8CFp16Kernel
    : DirectConvKernel<DirectTileConvForward8CFp16Kernel<ConfigIdx, Ver>, ConfigIdx>
{
    using V = TileConvVariant8c<Ver>;
    static constexpr bool kIsFprop = true;
    static std::string GetNamePrefix() { return "direct_tile_conv_fp16_fwd_"; }
};

template <int ConfigIdx, Version Ver = Version::v2>
struct DirectTileConvBwdData8CFp16Kernel
    : DirectConvKernel<DirectTileConvBwdData8CFp16Kernel<ConfigIdx, Ver>, ConfigIdx>
{
    using V = TileConvVariant8c<Ver>;
    static constexpr bool kIsFprop = false;
    static std::string GetNamePrefix() { return "direct_tile_conv_fp16_bwd_data_"; }
};

// 16c TileConv
template <int ConfigIdx, Version Ver = Version::v2>
struct DirectTileConvForward16CFp16Kernel
    : DirectConvKernel<DirectTileConvForward16CFp16Kernel<ConfigIdx, Ver>, ConfigIdx>
{
    using V = TileConvVariant16c<Ver>;
    static constexpr bool kIsFprop = true;
    static std::string GetNamePrefix() { return "direct_tile_conv_fp16_fwd_"; }
};

template <int ConfigIdx, Version Ver = Version::v2>
struct DirectTileConvBwdData16CFp16Kernel
    : DirectConvKernel<DirectTileConvBwdData16CFp16Kernel<ConfigIdx, Ver>, ConfigIdx>
{
    using V = TileConvVariant16c<Ver>;
    static constexpr bool kIsFprop = false;
    static std::string GetNamePrefix() { return "direct_tile_conv_fp16_bwd_data_"; }
};

// 32c TileConv
template <int ConfigIdx, Version Ver = Version::v2>
struct DirectTileConvForward32CFp16Kernel
    : DirectConvKernel<DirectTileConvForward32CFp16Kernel<ConfigIdx, Ver>, ConfigIdx>
{
    using V = TileConvVariant32c<Ver>;
    static constexpr bool kIsFprop = true;
    static std::string GetNamePrefix() { return "direct_tile_conv_fp16_fwd_"; }
};

template <int ConfigIdx, Version Ver = Version::v2>
struct DirectTileConvBwdData32CFp16Kernel
    : DirectConvKernel<DirectTileConvBwdData32CFp16Kernel<ConfigIdx, Ver>, ConfigIdx>
{
    using V = TileConvVariant32c<Ver>;
    static constexpr bool kIsFprop = false;
    static std::string GetNamePrefix() { return "direct_tile_conv_fp16_bwd_data_"; }
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
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_fwd_"; }
};

template <int ConfigIdx>
struct DirectHipConvBwdData4CFp16Kernel
    : DirectConvKernel<DirectHipConvBwdData4CFp16Kernel<ConfigIdx>, ConfigIdx>
{
    using V = HipConvVariant4c;
    static constexpr bool kIsFprop = false;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_bwd_data_"; }
};

// 8c HipConv
template <int ConfigIdx>
struct DirectHipConvForward8CFp16Kernel
    : DirectConvKernel<DirectHipConvForward8CFp16Kernel<ConfigIdx>, ConfigIdx>
{
    using V = HipConvVariant8c;
    static constexpr bool kIsFprop = true;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_fwd_"; }
};

template <int ConfigIdx>
struct DirectHipConvBwdData8CFp16Kernel
    : DirectConvKernel<DirectHipConvBwdData8CFp16Kernel<ConfigIdx>, ConfigIdx>
{
    using V = HipConvVariant8c;
    static constexpr bool kIsFprop = false;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_bwd_data_"; }
};

// 16c HipConv
template <int ConfigIdx>
struct DirectHipConvForward16CFp16Kernel
    : DirectConvKernel<DirectHipConvForward16CFp16Kernel<ConfigIdx>, ConfigIdx>
{
    using V = HipConvVariant16c;
    static constexpr bool kIsFprop = true;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_fwd_"; }
};

template <int ConfigIdx>
struct DirectHipConvBwdData16CFp16Kernel
    : DirectConvKernel<DirectHipConvBwdData16CFp16Kernel<ConfigIdx>, ConfigIdx>
{
    using V = HipConvVariant16c;
    static constexpr bool kIsFprop = false;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_bwd_data_"; }
};

// 32c HipConv
template <int ConfigIdx>
struct DirectHipConvForward32CFp16Kernel
    : DirectConvKernel<DirectHipConvForward32CFp16Kernel<ConfigIdx>, ConfigIdx>
{
    using V = HipConvVariant32c;
    static constexpr bool kIsFprop = true;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_fwd_"; }
};

template <int ConfigIdx>
struct DirectHipConvBwdData32CFp16Kernel
    : DirectConvKernel<DirectHipConvBwdData32CFp16Kernel<ConfigIdx>, ConfigIdx>
{
    using V = HipConvVariant32c;
    static constexpr bool kIsFprop = false;
    static std::string GetNamePrefix() { return "direct_hip_conv_fp16_bwd_data_"; }
};

} // namespace ck_tile::direct_conv
