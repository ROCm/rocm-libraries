// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include <miopen/ck_builder/instance_data/xdl_large_tensor.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/common_aliases.hpp>
#include <array>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace factories {
namespace grouped_conv_fwd {

using namespace instance;

// ========================================================================================
// BF16 large tensor instances (3 instances)
// ========================================================================================
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_large_tensor_bf16_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    return std::array{
        // clang-format off
        // generic instance
        DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,16,1,4},1,BF16,BF16),

        DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,128,32,8,8,32,32,4,2,{4,64,1},{1,0,2},{1,0,2},2,2,8,true,{4,64,1},{1,0,2},{1,0,2},2,2,8,true,1,1,{1,32,1,8},2,BF16,BF16),
        DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,128,32,8,8,32,32,4,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,BF16,BF16)
        // clang-format on
    };
}

// ========================================================================================
// F16 large tensor instances (3 instances)
// ========================================================================================
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_large_tensor_f16_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    return std::array{
        // clang-format off
        // generic instance
        DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,16,1,4},1,F16,F16),

        DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,128,32,8,8,32,32,4,2,{4,64,1},{1,0,2},{1,0,2},2,2,8,true,{4,64,1},{1,0,2},{1,0,2},2,2,8,true,1,1,{1,32,1,8},2,F16,F16),
        DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,128,32,8,8,32,32,4,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F16,F16)
        // clang-format on
    };
}

// ========================================================================================
// F32 large tensor instances (2 instances)
// ========================================================================================
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_large_tensor_f32_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    return std::array{
        // clang-format off
        // generic instance
        DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,16,4,4,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,8,1,8},1,F32,F32),

        DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,128,16,4,4,32,32,4,2,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,16,1,16},4,F32,F32)
        // clang-format on
    };
}

// ========================================================================================
// F32 TF32 large tensor instances (2 instances)
// ========================================================================================
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_large_tensor_f32_tf32_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: TF32 compute type is not yet supported in ckb::DataType enum
    // return std::array{
    //     // clang-format off
    //     // generic instance
    //     DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,16,4,4,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,8,1,8},1,TF32,TF32),
    //
    //     DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,128,16,4,4,32,32,4,2,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,16,1,16},4,TF32,TF32)
    //     // clang-format on
    // };
    return std::array<XdlLargeTensorInstance, 0>{};
}

// ========================================================================================
// INT8 large tensor instances (2 instances)
// ========================================================================================
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_large_tensor_int8_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    return std::array{
        // clang-format off
        // generic instance
        DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,16,1,4},1,I8,I8),

        DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,128,32,8,8,32,32,4,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,I8,I8)
        // clang-format on
    };
}

} // namespace grouped_conv_fwd
} // namespace factories
} // namespace ck_builder
} // namespace conv
} // namespace miopen
