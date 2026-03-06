// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include <miopen/ck_builder/instance_data/wmma_v3_large_tensor.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/common_aliases.hpp>
#include <array>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace factories {
namespace grouped_conv_fwd {

using namespace instance;

// DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_V3_Large_Tensor is UNSUPPORTED.
// All factory functions return empty arrays.

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_wmma_large_tensor_f16_generic_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // UNSUPPORTED: DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_V3_Large_Tensor
    // Original CK source has 1 instance (F16, blocksize=64)
    return std::array<int, 0>{};
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_wmma_large_tensor_f16_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // UNSUPPORTED: DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_V3_Large_Tensor
    // Original CK source has 5 instances (F16, blocksizes 256/128)
    return std::array<int, 0>{};
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_wmma_large_tensor_bf16_generic_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // UNSUPPORTED: DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_V3_Large_Tensor
    // Original CK source has 1 instance (BF16, blocksize=64)
    return std::array<int, 0>{};
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_wmma_large_tensor_bf16_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // UNSUPPORTED: DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_V3_Large_Tensor
    // Original CK source has 5 instances (BF16, blocksizes 256/128)
    return std::array<int, 0>{};
}

} // namespace grouped_conv_fwd
} // namespace factories
} // namespace ck_builder
} // namespace conv
} // namespace miopen
