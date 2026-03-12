// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include <miopen/ck_builder/instance_data/xdl_v3.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/common_aliases.hpp>
#include <array>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace factories {
namespace grouped_conv_fwd {

using namespace instance;

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_bf16_direct_load_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    ckb::PipelineScheduler pipeline_scheduler,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,32,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,BF16,BF16,pipeline_scheduler,PipeV1,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,32,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,BF16,BF16,pipeline_scheduler,PipeV4,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,64,64,8,8,16,16,1,2,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,BF16,BF16,pipeline_scheduler,PipeV1,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,32,64,8,8,32,32,2,1,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,64,1,4},8,BF16,BF16,pipeline_scheduler,PipeV1,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,128,64,8,8,32,32,1,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,BF16,BF16,pipeline_scheduler,PipeV1,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,64,8,8,16,16,2,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,4},8,BF16,BF16,pipeline_scheduler,PipeV1,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,64,8,8,16,16,2,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,4},8,BF16,BF16,pipeline_scheduler,PipeV4,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,128,8,8,16,16,1,1,{16,4,1},{1,0,2},{1,0,2},2,8,8,false,{16,4,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,4},4,BF16,BF16,pipeline_scheduler,PipeV4,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,64,8,8,16,16,1,1,{8,8,1},{1,0,2},{1,0,2},2,8,8,false,{8,8,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,4},4,BF16,BF16,pipeline_scheduler,PipeV1,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,32,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},2,BF16,BF16,pipeline_scheduler,PipeV4,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,16,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},2,BF16,BF16,pipeline_scheduler,PipeV4,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,64,8,8,16,16,2,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,4},2,BF16,BF16,pipeline_scheduler,PipeV1,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,64,2,2,16,16,2,2,{32,8,1},{1,0,2},{1,0,2},2,2,2,false,{32,8,1},{1,0,2},{1,0,2},2,2,2,false,1,1,{1,32,1,4},4,BF16,BF16,pipeline_scheduler,PipeV1,1,true)
        // clang-format on
    };
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_f16_direct_load_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    ckb::PipelineScheduler pipeline_scheduler,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,32,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,F16,F16,pipeline_scheduler,PipeV1,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,32,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,F16,F16,pipeline_scheduler,PipeV4,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,64,64,8,8,16,16,1,2,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,F16,F16,pipeline_scheduler,PipeV1,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,32,64,8,8,32,32,2,1,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,64,1,4},8,F16,F16,pipeline_scheduler,PipeV1,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,128,64,8,8,32,32,1,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,F16,F16,pipeline_scheduler,PipeV1,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,64,8,8,16,16,2,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,4},8,F16,F16,pipeline_scheduler,PipeV1,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,64,8,8,16,16,2,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,4},8,F16,F16,pipeline_scheduler,PipeV4,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,128,8,8,16,16,1,1,{16,4,1},{1,0,2},{1,0,2},2,8,8,false,{16,4,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,4},4,F16,F16,pipeline_scheduler,PipeV4,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,64,8,8,16,16,1,1,{8,8,1},{1,0,2},{1,0,2},2,8,8,false,{8,8,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,4},4,F16,F16,pipeline_scheduler,PipeV1,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,32,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},2,F16,F16,pipeline_scheduler,PipeV4,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,16,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},2,F16,F16,pipeline_scheduler,PipeV4,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,64,8,8,16,16,2,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,4},2,F16,F16,pipeline_scheduler,PipeV1,1,true),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,64,2,2,16,16,2,2,{32,8,1},{1,0,2},{1,0,2},2,2,2,false,{32,8,1},{1,0,2},{1,0,2},2,2,2,false,1,1,{1,32,1,4},4,F16,F16,pipeline_scheduler,PipeV1,1,true)
        // clang-format on
    };
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_bf16_mem_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    ckb::PipelineScheduler pipeline_scheduler,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    return std::array{
        // clang-format off
        // Latency friendly
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,16,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},2,BF16,BF16,pipeline_scheduler,PipeV1),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,128,8,8,16,16,1,1,{16,4,1},{1,0,2},{1,0,2},2,8,8,false,{16,4,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,4},4,BF16,BF16,pipeline_scheduler,PipeV1),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,64,8,8,16,16,1,1,{8,8,1},{1,0,2},{1,0,2},2,8,8,false,{8,8,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,4},4,BF16,BF16,pipeline_scheduler,PipeV1),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,32,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,BF16,BF16,pipeline_scheduler,PipeV1),
        // Memory friendly
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,32,64,8,8,32,32,2,1,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},4,BF16,BF16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,16,64,8,8,16,16,4,1,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},2,BF16,BF16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,32,64,8,8,32,32,2,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,BF16,BF16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,16,64,8,8,16,16,4,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},2,BF16,BF16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,32,64,8,8,32,32,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,BF16,BF16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,16,64,8,8,16,16,2,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},2,BF16,BF16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,16,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},2,BF16,BF16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,128,8,8,16,16,1,1,{16,4,1},{1,0,2},{1,0,2},2,8,8,false,{16,4,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,4},4,BF16,BF16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,64,8,8,16,16,1,1,{8,8,1},{1,0,2},{1,0,2},2,8,8,false,{8,8,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,4},4,BF16,BF16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,32,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,BF16,BF16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,64,64,8,8,16,16,1,2,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,BF16,BF16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,64,64,8,8,32,32,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},8,BF16,BF16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,128,64,8,8,16,16,1,4,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,BF16,BF16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,128,64,8,8,32,32,1,2,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},8,BF16,BF16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,16,256,64,8,8,16,16,1,4,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,16},4,BF16,BF16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,32,256,64,8,8,32,32,1,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,16},8,BF16,BF16,pipeline_scheduler,PipeV2)
        // clang-format on
    };
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_f16_mem_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    ckb::PipelineScheduler pipeline_scheduler,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,16,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},2,F16,F16,pipeline_scheduler,PipeV1),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,128,8,8,16,16,1,1,{16,4,1},{1,0,2},{1,0,2},2,8,8,false,{16,4,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,4},4,F16,F16,pipeline_scheduler,PipeV1),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,64,8,8,16,16,1,1,{8,8,1},{1,0,2},{1,0,2},2,8,8,false,{8,8,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,4},4,F16,F16,pipeline_scheduler,PipeV1),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,32,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,F16,F16,pipeline_scheduler,PipeV1),
        // Memory friendly
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,32,64,8,8,32,32,2,1,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},4,F16,F16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,16,64,8,8,16,16,4,1,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},2,F16,F16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,32,64,8,8,32,32,2,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,F16,F16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,16,64,8,8,16,16,4,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},2,F16,F16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,32,64,8,8,32,32,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,F16,F16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,16,64,8,8,16,16,2,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},2,F16,F16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,16,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},2,F16,F16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,128,8,8,16,16,1,1,{16,4,1},{1,0,2},{1,0,2},2,8,8,false,{16,4,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,4},4,F16,F16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,64,8,8,16,16,1,1,{8,8,1},{1,0,2},{1,0,2},2,8,8,false,{8,8,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,4},4,F16,F16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,32,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,F16,F16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,64,64,8,8,16,16,1,2,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,F16,F16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,64,64,8,8,32,32,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},8,F16,F16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,128,64,8,8,16,16,1,4,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,F16,F16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,128,64,8,8,32,32,1,2,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},8,F16,F16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,16,256,64,8,8,16,16,1,4,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,16},4,F16,F16,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,32,256,64,8,8,32,32,1,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,16},8,F16,F16,pipeline_scheduler,PipeV2)
        // clang-format on
    };
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_f32_mem_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    ckb::PipelineScheduler pipeline_scheduler,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,16,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},2,F32,F32,pipeline_scheduler,PipeV1),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,128,8,8,16,16,1,1,{16,4,1},{1,0,2},{1,0,2},2,4,4,false,{16,4,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,4},4,F32,F32,pipeline_scheduler,PipeV1),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,64,8,8,16,16,1,1,{8,8,1},{1,0,2},{1,0,2},2,4,4,false,{8,8,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,4},4,F32,F32,pipeline_scheduler,PipeV1),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,32,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},4,F32,F32,pipeline_scheduler,PipeV1),
        // Memory friendly
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,32,64,8,8,32,32,2,1,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},4,F32,F32,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,16,64,8,8,16,16,4,1,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},2,F32,F32,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,32,64,8,8,32,32,1,1,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},4,F32,F32,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,16,64,8,8,16,16,2,1,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},2,F32,F32,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,16,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},2,F32,F32,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,128,8,8,16,16,1,1,{16,4,1},{1,0,2},{1,0,2},2,4,4,false,{16,4,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,4},4,F32,F32,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,64,8,8,16,16,1,1,{8,8,1},{1,0,2},{1,0,2},2,4,4,false,{8,8,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,4},4,F32,F32,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,32,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},4,F32,F32,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,64,64,8,8,16,16,1,2,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},4,F32,F32,pipeline_scheduler,PipeV2),
        // TODO: c_block_transfer_scalar_per_vector=8 exceeds F32 max of 4 (ValidCBlockTransfer fails)
        // DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,64,64,8,8,32,32,1,1,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},8,F32,F32,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,128,64,8,8,16,16,1,4,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},4,F32,F32,pipeline_scheduler,PipeV2),
        // TODO: c_block_transfer_scalar_per_vector=8 exceeds F32 max of 4 (ValidCBlockTransfer fails)
        // DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,128,64,8,8,32,32,1,2,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},8,F32,F32,pipeline_scheduler,PipeV2)
        // clang-format on
    };
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_f32_tf32_mem_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    ckb::PipelineScheduler pipeline_scheduler,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: TF32 compute type is not yet supported in ckb::DataType enum
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,16,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},2,TF32,TF32,pipeline_scheduler,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,128,8,8,16,16,1,1,{16,4,1},{1,0,2},{1,0,2},2,4,4,false,{16,4,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,4},4,TF32,TF32,pipeline_scheduler,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,64,8,8,16,16,1,1,{8,8,1},{1,0,2},{1,0,2},2,4,4,false,{8,8,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,4},4,TF32,TF32,pipeline_scheduler,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,32,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},4,TF32,TF32,pipeline_scheduler,PipeV1),
    //     // Memory friendly
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,32,64,8,8,32,32,2,1,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},4,TF32,TF32,pipeline_scheduler,PipeV2),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,16,64,8,8,16,16,4,1,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},2,TF32,TF32,pipeline_scheduler,PipeV2),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,32,64,8,8,32,32,1,1,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},4,TF32,TF32,pipeline_scheduler,PipeV2),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,16,64,8,8,16,16,2,1,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},2,TF32,TF32,pipeline_scheduler,PipeV2),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,16,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},2,TF32,TF32,pipeline_scheduler,PipeV2),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,128,8,8,16,16,1,1,{16,4,1},{1,0,2},{1,0,2},2,4,4,false,{16,4,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,4},4,TF32,TF32,pipeline_scheduler,PipeV2),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,64,8,8,16,16,1,1,{8,8,1},{1,0,2},{1,0,2},2,4,4,false,{8,8,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,4},4,TF32,TF32,pipeline_scheduler,PipeV2),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,32,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},4,TF32,TF32,pipeline_scheduler,PipeV2),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,64,64,8,8,16,16,1,2,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},4,TF32,TF32,pipeline_scheduler,PipeV2),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,64,64,8,8,32,32,1,1,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},8,TF32,TF32,pipeline_scheduler,PipeV2),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,128,64,8,8,16,16,1,4,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},4,TF32,TF32,pipeline_scheduler,PipeV2),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,128,64,8,8,32,32,1,2,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,{8,16,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,16,1,8},8,TF32,TF32,pipeline_scheduler,PipeV2)
    //     // clang-format on
    // };
    return std::array<XdlV3Instance, 0>{};
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_int8_mem_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    ckb::PipelineScheduler pipeline_scheduler,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,16,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},2,I8,I8,pipeline_scheduler,PipeV1),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,128,8,8,16,16,1,1,{16,4,1},{1,0,2},{1,0,2},2,8,8,false,{16,4,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,4},4,I8,I8,pipeline_scheduler,PipeV1),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,64,8,8,16,16,1,1,{8,8,1},{1,0,2},{1,0,2},2,8,8,false,{8,8,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,4},4,I8,I8,pipeline_scheduler,PipeV1),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,32,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,I8,I8,pipeline_scheduler,PipeV1),
        // Memory friendly
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,32,64,8,8,32,32,2,1,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},4,I8,I8,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,16,64,8,8,16,16,4,1,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},2,I8,I8,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,32,64,8,8,32,32,2,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,I8,I8,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,16,64,8,8,16,16,4,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},2,I8,I8,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,32,64,8,8,32,32,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,I8,I8,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,16,64,8,8,16,16,2,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},2,I8,I8,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,16,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},2,I8,I8,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,128,8,8,16,16,1,1,{16,4,1},{1,0,2},{1,0,2},2,8,8,false,{16,4,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,4},4,I8,I8,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,16,16,64,8,8,16,16,1,1,{8,8,1},{1,0,2},{1,0,2},2,8,8,false,{8,8,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,4},4,I8,I8,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,32,64,8,8,16,16,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,I8,I8,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,64,64,8,8,16,16,1,2,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,I8,I8,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,64,64,8,8,32,32,1,1,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},8,I8,I8,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,16,128,64,8,8,16,16,1,4,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},4,I8,I8,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,128,64,8,8,32,32,1,2,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},8,I8,I8,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,16,256,64,8,8,16,16,1,4,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,16},4,I8,I8,pipeline_scheduler,PipeV2),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,32,256,64,8,8,32,32,1,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,16},8,I8,I8,pipeline_scheduler,PipeV2)
        // clang-format on
    };
}

} // namespace grouped_conv_fwd
} // namespace factories
} // namespace ck_builder
} // namespace conv
} // namespace miopen
