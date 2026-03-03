// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include <miopen/ck_builder/instance_data/wmma.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/common_aliases.hpp>
#include <array>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace factories {
namespace grouped_conv_fwd {

using namespace instance;

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_wmma_f16_instances(
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
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,64,32,8,16,16,2,2,{4,32,1},{1,0,2},{1,0,2},2,1,8,true,{4,32,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,32,1,4},1),
        // blocksize=256
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,16,16,4,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,256,32,8,16,16,2,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,64,32,8,16,16,8,1,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,64,8,16,16,4,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8),
        // blocksize=128
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,64,32,8,16,16,2,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,64,64,8,16,16,2,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,128,32,8,16,16,2,4,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,128,64,8,16,16,2,4,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,64,32,8,16,16,4,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,64,64,8,16,16,4,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,256,32,8,16,16,1,8,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,256,32,32,8,16,16,8,1,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8),
        // blocksize=64
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,32,64,32,8,16,16,1,4,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,2},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,16,16,2,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,2},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,32,32,64,8,16,16,1,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,2},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,32,128,32,8,16,16,1,8,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,2},8),
        // blocksize=32
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,32,16,64,32,8,16,16,1,4,{2,16,1},{1,0,2},{1,0,2},2,8,8,true,{2,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,2},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,32,64,16,32,8,16,16,4,1,{2,16,1},{1,0,2},{1,0,2},2,8,8,true,{2,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,2},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,32,32,32,32,8,16,16,2,2,{2,16,1},{1,0,2},{1,0,2},2,8,8,true,{2,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,2},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,32,16,16,32,8,16,16,1,1,{2,16,1},{1,0,2},{1,0,2},2,8,8,true,{2,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,2},8)
        // clang-format on
    };
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_wmma_i8_instances(
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
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,64,64,16,16,16,2,2,{4,32,1},{1,0,2},{1,0,2},2,1,16,true,{4,32,1},{1,0,2},{1,0,2},2,1,16,true,1,1,{1,32,1,4},1),
        // blocksize=256
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,64,16,16,16,4,2,{4,64,1},{1,0,2},{1,0,2},2,16,16,true,{4,64,1},{1,0,2},{1,0,2},2,16,16,true,1,1,{1,32,1,8},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,256,64,16,16,16,2,4,{4,64,1},{1,0,2},{1,0,2},2,16,16,true,{4,64,1},{1,0,2},{1,0,2},2,16,16,true,1,1,{1,32,1,8},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,64,64,16,16,16,8,1,{4,64,1},{1,0,2},{1,0,2},2,16,16,true,{4,64,1},{1,0,2},{1,0,2},2,16,16,true,1,1,{1,32,1,8},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,128,16,16,16,4,2,{4,64,1},{1,0,2},{1,0,2},2,16,16,true,{4,64,1},{1,0,2},{1,0,2},2,16,16,true,1,1,{1,32,1,8},8),
        // blocksize=128
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,64,64,16,16,16,2,2,{4,32,1},{1,0,2},{1,0,2},2,16,16,true,{4,32,1},{1,0,2},{1,0,2},2,16,16,true,1,1,{1,32,1,4},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,64,128,16,16,16,2,2,{4,32,1},{1,0,2},{1,0,2},2,16,16,true,{4,32,1},{1,0,2},{1,0,2},2,16,16,true,1,1,{1,32,1,4},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,128,64,16,16,16,2,4,{4,32,1},{1,0,2},{1,0,2},2,16,16,true,{4,32,1},{1,0,2},{1,0,2},2,16,16,true,1,1,{1,32,1,4},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,128,128,16,16,16,2,4,{4,32,1},{1,0,2},{1,0,2},2,16,16,true,{4,32,1},{1,0,2},{1,0,2},2,16,16,true,1,1,{1,32,1,4},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,64,64,16,16,16,4,2,{4,32,1},{1,0,2},{1,0,2},2,16,16,true,{4,32,1},{1,0,2},{1,0,2},2,16,16,true,1,1,{1,32,1,4},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,64,128,16,16,16,4,2,{4,32,1},{1,0,2},{1,0,2},2,16,16,true,{4,32,1},{1,0,2},{1,0,2},2,16,16,true,1,1,{1,32,1,4},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,256,64,16,16,16,1,8,{4,32,1},{1,0,2},{1,0,2},2,16,16,true,{4,32,1},{1,0,2},{1,0,2},2,16,16,true,1,1,{1,32,1,4},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,256,32,64,16,16,16,8,1,{4,32,1},{1,0,2},{1,0,2},2,16,16,true,{4,32,1},{1,0,2},{1,0,2},2,16,16,true,1,1,{1,32,1,4},8),
        // blocksize=64
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,32,64,64,16,16,16,1,4,{4,16,1},{1,0,2},{1,0,2},2,16,16,true,{4,16,1},{1,0,2},{1,0,2},2,16,16,true,1,1,{1,32,1,2},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,64,16,16,16,2,2,{4,16,1},{1,0,2},{1,0,2},2,16,16,true,{4,16,1},{1,0,2},{1,0,2},2,16,16,true,1,1,{1,32,1,2},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,32,32,128,16,16,16,1,2,{4,16,1},{1,0,2},{1,0,2},2,16,16,true,{4,16,1},{1,0,2},{1,0,2},2,16,16,true,1,1,{1,32,1,2},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,32,128,64,16,16,16,1,8,{4,16,1},{1,0,2},{1,0,2},2,16,16,true,{4,16,1},{1,0,2},{1,0,2},2,16,16,true,1,1,{1,32,1,2},8),
        // blocksize=32
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,32,16,64,64,16,16,16,1,4,{2,16,1},{1,0,2},{1,0,2},2,16,16,true,{2,16,1},{1,0,2},{1,0,2},2,16,16,true,1,1,{1,16,1,2},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,32,64,16,64,16,16,16,4,1,{2,16,1},{1,0,2},{1,0,2},2,16,16,true,{2,16,1},{1,0,2},{1,0,2},2,16,16,true,1,1,{1,16,1,2},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,32,32,32,64,16,16,16,2,2,{2,16,1},{1,0,2},{1,0,2},2,16,16,true,{2,16,1},{1,0,2},{1,0,2},2,16,16,true,1,1,{1,16,1,2},8),
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,32,16,16,64,16,16,16,1,1,{2,16,1},{1,0,2},{1,0,2},2,16,16,true,{2,16,1},{1,0,2},{1,0,2},2,16,16,true,1,1,{1,16,1,2},8)
        // clang-format on
    };
}

} // namespace grouped_conv_fwd
} // namespace factories
} // namespace ck_builder
} // namespace conv
} // namespace miopen
