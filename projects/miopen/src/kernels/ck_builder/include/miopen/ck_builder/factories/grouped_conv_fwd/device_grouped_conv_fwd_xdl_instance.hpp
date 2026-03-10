// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include <miopen/ck_builder/instance_data/xdl.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/common_aliases.hpp>
#include <array>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace factories {
namespace grouped_conv_fwd {

using namespace instance;

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_bf16_generic_instances(
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
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,16,1,4},1,BF16,BF16)
        // clang-format on
    };
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_bf16_instances(
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
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,16,1,4},1,BF16,BF16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},1,BF16,BF16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,32,1,8},8,BF16,BF16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,128,32,8,8,32,32,4,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,BF16,BF16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,256,32,8,8,32,32,2,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,BF16,BF16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,128,32,8,8,32,32,4,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,BF16,BF16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,BF16,BF16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,64,32,8,8,32,32,2,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,BF16,BF16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,128,32,8,8,32,32,2,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,BF16,BF16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,BF16,BF16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,64,32,8,8,32,32,2,1,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,BF16,BF16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,128,32,8,8,32,32,1,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,BF16,BF16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,32,32,8,8,32,32,2,1,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,BF16,BF16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,128,32,8,8,32,32,1,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,BF16,BF16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,BF16,BF16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,32,64,32,8,8,32,32,1,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,BF16,BF16)
        // clang-format on
    };
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_bf16_16x16_instances(
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
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,32,8,8,16,16,2,2,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,{4,64,1},{1,0,2},{1,0,2},2,2,8,true,1,1,{1,32,1,4},1,BF16,BF16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,32,8,8,16,16,2,2,{4,64,1},{1,0,2},{1,0,2},2,2,8,true,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,32,1,4},2,BF16,BF16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,32,8,8,16,16,2,2,{4,64,1},{1,0,2},{1,0,2},2,4,8,true,{4,64,1},{1,0,2},{1,0,2},2,4,8,true,1,1,{1,32,1,4},4,BF16,BF16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,32,8,8,16,16,2,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,BF16,BF16)
        // clang-format on
    };
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_f16_generic_instances(
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
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,16,1,4},1,F16,F16)
        // clang-format on
    };
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_f16_instances(
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
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,16,1,4},1,F16,F16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},1,F16,F16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,32,1,8},8,F16,F16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,128,32,8,8,32,32,4,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F16,F16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,256,32,8,8,32,32,2,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F16,F16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,128,32,8,8,32,32,4,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,F16,F16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F16,F16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,64,32,8,8,32,32,2,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,F16,F16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,128,32,8,8,32,32,2,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,F16,F16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,F16,F16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,64,32,8,8,32,32,2,1,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F16,F16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,128,32,8,8,32,32,1,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F16,F16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,32,32,8,8,32,32,2,1,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,F16,F16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,128,32,8,8,32,32,1,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,F16,F16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,F16,F16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,32,64,32,8,8,32,32,1,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,F16,F16)
        // clang-format on
    };
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_f16_nchw_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: NCHW instances with src_vector_dim=1 not yet supported by ck_tile builder
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,128,32,8,8,32,32,1,2,{4,64,1},{0,2,1},{0,2,1},1,1,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},1,F16,F16),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,{4,16,1},{0,2,1},{0,2,1},1,4,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,8,1,8},1,F16,F16),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,64,32,8,8,16,16,4,2,{4,64,1},{0,2,1},{0,2,1},1,2,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},1,F16,F16),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,{4,16,1},{0,2,1},{0,2,1},1,1,8,true,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,8,1,8},1,F16,F16),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,128,32,8,8,32,32,4,2,{4,64,1},{0,2,1},{0,2,1},1,4,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},4,F16,F16),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,256,32,8,8,32,32,2,4,{4,64,1},{0,2,1},{0,2,1},1,2,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},4,F16,F16),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{0,2,1},{0,2,1},1,2,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},4,F16,F16),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,64,32,8,8,32,32,2,1,{4,64,1},{0,2,1},{0,2,1},1,2,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},4,F16,F16),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,128,32,8,8,32,32,1,2,{4,64,1},{0,2,1},{0,2,1},1,1,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},4,F16,F16),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,{4,16,1},{0,2,1},{0,2,1},1,4,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,8,1,8},4,F16,F16),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,32,32,8,8,32,32,2,1,{4,32,1},{0,2,1},{0,2,1},1,4,8,true,{4,32,1},{1,0,2},{1,0,2},2,4,8,true,1,1,{1,16,1,8},4,F16,F16),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,64,32,8,8,16,16,4,2,{4,64,1},{0,2,1},{0,2,1},1,2,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},4,F16,F16)
    //     // clang-format on
    // };
    return std::array<XdlInstance, 0>{};
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_f16_16x16_instances(
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
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,32,8,8,16,16,2,2,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,{4,64,1},{1,0,2},{1,0,2},2,2,8,true,1,1,{1,32,1,4},1,F16,F16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,32,8,8,16,16,2,2,{4,64,1},{1,0,2},{1,0,2},2,2,8,true,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,32,1,4},2,F16,F16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,32,8,8,16,16,2,2,{4,64,1},{1,0,2},{1,0,2},2,4,8,true,{4,64,1},{1,0,2},{1,0,2},2,4,8,true,1,1,{1,32,1,4},4,F16,F16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,32,8,8,16,16,2,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,F16,F16)
        // clang-format on
    };
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_f32_generic_instances(
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
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,16,4,4,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,8,1,8},1,F32,F32)
        // clang-format on
    };
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_f32_tf32_generic_instances(
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
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,16,4,4,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,8,1,8},1,TF32,TF32)
    //     // clang-format on
    // };
    return std::array<XdlInstance, 0>{};
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_f32_instances(
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
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,16,4,4,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,8,1,8},1,F32,F32),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,16,4,4,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,8,1,8},1,F32,F32),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,16,4,4,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,1,4,true,{4,64,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,16,1,16},4,F32,F32),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,128,16,4,4,32,32,4,2,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,16,1,16},4,F32,F32),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,256,16,4,4,32,32,2,4,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,16,1,16},4,F32,F32),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,128,16,4,4,32,32,4,2,{4,32,1},{1,0,2},{1,0,2},2,4,4,true,{4,32,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,8,1,16},4,F32,F32),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,16,4,4,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,16,1,16},4,F32,F32),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,64,16,4,4,32,32,2,2,{4,32,1},{1,0,2},{1,0,2},2,4,4,true,{4,32,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,16,1,8},4,F32,F32),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,128,16,4,4,32,32,2,2,{4,32,1},{1,0,2},{1,0,2},2,4,4,true,{4,32,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,8,1,16},4,F32,F32),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,16,4,4,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,8,1,8},4,F32,F32),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,64,16,4,4,32,32,2,1,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,16,1,16},4,F32,F32),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,128,16,4,4,32,32,1,2,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,16,1,16},4,F32,F32),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,32,16,4,4,32,32,2,1,{4,32,1},{1,0,2},{1,0,2},2,4,4,true,{4,32,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,16,1,8},4,F32,F32),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,128,16,4,4,32,32,1,2,{4,32,1},{1,0,2},{1,0,2},2,4,4,true,{4,32,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,8,1,16},4,F32,F32),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,16,4,4,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,8,1,8},4,F32,F32),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,32,64,16,4,4,32,32,1,2,{4,16,1},{1,0,2},{1,0,2},2,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,8,1,8},4,F32,F32),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,192,16,4,4,32,32,2,3,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,16,1,16},4,F32,F32)
        // clang-format on
    };
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_f32_tf32_instances(
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
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,16,4,4,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,8,1,8},1,TF32,TF32),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,16,4,4,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,8,1,8},1,TF32,TF32),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,16,4,4,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,1,4,true,{4,64,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,16,1,16},4,TF32,TF32),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,128,16,4,4,32,32,4,2,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,16,1,16},4,TF32,TF32),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,256,16,4,4,32,32,2,4,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,16,1,16},4,TF32,TF32),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,128,16,4,4,32,32,4,2,{4,32,1},{1,0,2},{1,0,2},2,4,4,true,{4,32,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,8,1,16},4,TF32,TF32),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,16,4,4,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,16,1,16},4,TF32,TF32),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,64,16,4,4,32,32,2,2,{4,32,1},{1,0,2},{1,0,2},2,4,4,true,{4,32,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,16,1,8},4,TF32,TF32),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,128,16,4,4,32,32,2,2,{4,32,1},{1,0,2},{1,0,2},2,4,4,true,{4,32,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,8,1,16},4,TF32,TF32),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,16,4,4,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,8,1,8},4,TF32,TF32),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,64,16,4,4,32,32,2,1,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,16,1,16},4,TF32,TF32),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,128,16,4,4,32,32,1,2,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,16,1,16},4,TF32,TF32),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,32,16,4,4,32,32,2,1,{4,32,1},{1,0,2},{1,0,2},2,4,4,true,{4,32,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,16,1,8},4,TF32,TF32),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,128,16,4,4,32,32,1,2,{4,32,1},{1,0,2},{1,0,2},2,4,4,true,{4,32,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,8,1,16},4,TF32,TF32),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,16,4,4,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,8,1,8},4,TF32,TF32),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,32,64,16,4,4,32,32,1,2,{4,16,1},{1,0,2},{1,0,2},2,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,8,1,8},4,TF32,TF32),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,192,16,4,4,32,32,2,3,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,{4,64,1},{1,0,2},{1,0,2},2,4,4,true,1,1,{1,16,1,16},4,TF32,TF32)
    //     // clang-format on
    // };
    return std::array<XdlInstance, 0>{};
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_f32_16x16_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: F32 16x16 tile instances fail ValidABlockTransfer in ck_tile builder
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,32,8,8,16,16,2,2,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,{4,64,1},{1,0,2},{1,0,2},2,2,8,true,1,1,{1,32,1,4},1,F32,F32),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,32,8,8,16,16,2,2,{4,64,1},{1,0,2},{1,0,2},2,2,8,true,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,32,1,4},2,F32,F32),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,32,8,8,16,16,2,2,{4,64,1},{1,0,2},{1,0,2},2,4,8,true,{4,64,1},{1,0,2},{1,0,2},2,4,8,true,1,1,{1,32,1,4},4,F32,F32)
    //     // clang-format on
    // };
    return std::array<XdlInstance, 0>{};
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_f32_tf32_16x16_instances(
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
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,32,8,8,16,16,2,2,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,{4,64,1},{1,0,2},{1,0,2},2,2,8,true,1,1,{1,32,1,4},1,TF32,TF32),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,32,8,8,16,16,2,2,{4,64,1},{1,0,2},{1,0,2},2,2,8,true,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,32,1,4},2,TF32,TF32),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,32,8,8,16,16,2,2,{4,64,1},{1,0,2},{1,0,2},2,4,8,true,{4,64,1},{1,0,2},{1,0,2},2,4,8,true,1,1,{1,32,1,4},4,TF32,TF32)
    //     // clang-format on
    // };
    return std::array<XdlInstance, 0>{};
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_int8_generic_instances(
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
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,16,1,4},1,I8,I8)
        // clang-format on
    };
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_int8_instances(
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
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,16,1,4},1,I8,I8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},1,I8,I8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,32,1,8},8,I8,I8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,128,32,8,8,32,32,4,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,I8,I8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,256,32,8,8,32,32,2,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,I8,I8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,128,32,8,8,32,32,4,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,I8,I8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,I8,I8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,64,32,8,8,32,32,2,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,I8,I8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,128,32,8,8,32,32,2,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,I8,I8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,I8,I8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,64,32,8,8,32,32,2,1,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,I8,I8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,128,32,8,8,32,32,1,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,I8,I8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,32,32,8,8,32,32,2,1,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,I8,I8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,128,32,8,8,32,32,1,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,I8,I8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,I8,I8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,32,64,32,8,8,32,32,1,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,I8,I8)
        // clang-format on
    };
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_f16_comp_f8_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
#ifdef CK_ENABLE_FP8
    return std::array{
        // clang-format off
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,16,1,4},1,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},1,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,32,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,128,32,8,8,32,32,4,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,256,32,8,8,32,32,2,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,128,32,8,8,32,32,4,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,64,32,8,8,32,32,2,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,128,32,8,8,32,32,2,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,64,32,8,8,32,32,2,1,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,128,32,8,8,32,32,1,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,32,32,8,8,32,32,2,1,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,128,32,8,8,32,32,1,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,32,64,32,8,8,32,32,1,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,F8,F8)
        // clang-format on
    };
#else
    return std::array<XdlInstance, 0>{};
#endif
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_f8_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
#ifdef CK_ENABLE_FP8
    return std::array{
        // clang-format off
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,16,1,4},1,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},1,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,32,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,128,32,8,8,32,32,4,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,256,32,8,8,32,32,2,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,128,32,8,8,32,32,4,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,64,32,8,8,32,32,2,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,128,32,8,8,32,32,2,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,64,32,8,8,32,32,2,1,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,128,32,8,8,32,32,1,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,32,32,8,8,32,32,2,1,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,128,32,8,8,32,32,1,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,32,64,32,8,8,32,32,1,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,F8,F8)
        // clang-format on
    };
#else
    return std::array<XdlInstance, 0>{};
#endif
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_bf8_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
#ifdef CK_ENABLE_BF8
    return std::array{
        // clang-format off
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,16,1,4},1,BF8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},1,BF8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,32,1,8},8,BF8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,128,32,8,8,32,32,4,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,BF8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,256,32,8,8,32,32,2,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,BF8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,128,32,8,8,32,32,4,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,BF8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,BF8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,64,32,8,8,32,32,2,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,BF8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,128,32,8,8,32,32,2,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,BF8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,BF8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,64,32,8,8,32,32,2,1,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,BF8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,128,32,8,8,32,32,1,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,BF8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,32,32,8,8,32,32,2,1,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,BF8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,128,32,8,8,32,32,1,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,BF8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,BF8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,32,64,32,8,8,32,32,1,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,BF8,BF8)
        // clang-format on
    };
#else
    return std::array<XdlInstance, 0>{};
#endif
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_f8_bf8_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
#if defined(CK_ENABLE_FP8) && defined(CK_ENABLE_BF8)
    return std::array{
        // clang-format off
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,16,1,4},1,F8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},1,F8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,32,1,8},8,F8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,128,32,8,8,32,32,4,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,256,32,8,8,32,32,2,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,128,32,8,8,32,32,4,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,F8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,64,32,8,8,32,32,2,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,F8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,128,32,8,8,32,32,2,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,F8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,F8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,64,32,8,8,32,32,2,1,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,128,32,8,8,32,32,1,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,32,32,8,8,32,32,2,1,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,F8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,128,32,8,8,32,32,1,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,F8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,F8,BF8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,BF8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,32,64,32,8,8,32,32,1,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,F8,BF8)
        // clang-format on
    };
#else
    return std::array<XdlInstance, 0>{};
#endif
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_bf8_f8_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
#if defined(CK_ENABLE_FP8) && defined(CK_ENABLE_BF8)
    return std::array{
        // clang-format off
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,16,1,4},1,BF8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},1,BF8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,32,1,8},8,BF8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,128,32,8,8,32,32,4,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,BF8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,256,32,8,8,32,32,2,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,BF8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,128,32,8,8,32,32,4,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,BF8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,BF8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,64,32,8,8,32,32,2,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,BF8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,128,32,8,8,32,32,2,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,BF8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,BF8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,64,32,8,8,32,32,2,1,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,BF8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,128,32,8,8,32,32,1,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,BF8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,32,32,8,8,32,32,2,1,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,BF8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,128,32,8,8,32,32,1,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,BF8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,BF8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF8,F8,F32,F8,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,32,64,32,8,8,32,32,1,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,BF8,F8)
        // clang-format on
    };
#else
    return std::array<XdlInstance, 0>{};
#endif
}

} // namespace grouped_conv_fwd
} // namespace factories
} // namespace ck_builder
} // namespace conv
} // namespace miopen
