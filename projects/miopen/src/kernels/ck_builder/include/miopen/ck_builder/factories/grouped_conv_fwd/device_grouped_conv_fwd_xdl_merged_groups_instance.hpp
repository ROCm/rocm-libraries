// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include <miopen/ck_builder/instance_data/xdl.hpp>
#include <miopen/ck_builder/instance_data/xdl_v3.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/common_aliases.hpp>
#include <array>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace factories {
namespace grouped_conv_fwd {

using namespace instance;

// ========================================================================================
// BF16 merged groups instances (non-V3, 3 instances with NumGroupsToMerge = 8/16/32)
// ========================================================================================
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_merged_groups_bf16_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: merged_groups instances with src_vector_dim=1 not yet supported by ck_tile builder
    // return std::array{
    //     // clang-format off
    //     // Instances with NumGroupsPerBatch > 1 (non-V3)
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,16,16,4,4,16,16,4,1,{4,16,1},{0,2,1},{0,2,1},1,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,16,1,4},1,BF16,BF16,ckb::PipelineScheduler::DEFAULT,8),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,16,16,4,4,16,16,4,1,{4,16,1},{0,2,1},{0,2,1},1,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,16,1,4},1,BF16,BF16,ckb::PipelineScheduler::DEFAULT,16),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,16,16,4,4,16,16,4,1,{4,16,1},{0,2,1},{0,2,1},1,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,16,1,4},1,BF16,BF16,ckb::PipelineScheduler::DEFAULT,32)
    //     // clang-format on
    // };
    return std::array<XdlInstance, 0>{};
}

// BF16 merged groups V3 instance
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_merged_groups_bf16_v3_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: V3 merged_groups instances not yet supported by ck_tile builder
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,32,32,8,8,32,32,2,1,{4,64,1},{1,0,2},{1,0,2},2,4,8,true,{4,32,1},{1,0,2},{1,0,2},2,4,8,true,1,1,{1,32,1,4},4,BF16,BF16,Interwave,PipeV1,false,8)
    //     // clang-format on
    // };
    return std::array<XdlV3Instance, 0>{};
}

// ========================================================================================
// BF16 merged groups 2x instances (gfx950, non-V3 + V3)
// ========================================================================================
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_merged_groups_bf16_instances_2x(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: merged_groups instances with mixed src_vector_dim not yet supported by ck_tile builder
    // return std::array{
    //     // clang-format off
    //     // Instances with NumGroupsPerBatch > 1 (non-V3, double rate)
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,16,32,8,4,16,16,4,1,{4,16,1},{0,2,1},{0,2,1},1,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,16,1,4},1,BF16,BF16,ckb::PipelineScheduler::DEFAULT,8),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,16,32,8,4,16,16,4,1,{4,16,1},{0,2,1},{0,2,1},1,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,16,1,4},1,BF16,BF16,ckb::PipelineScheduler::DEFAULT,16),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,16,32,8,4,16,16,4,1,{4,16,1},{0,2,1},{0,2,1},1,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,16,1,4},1,BF16,BF16,ckb::PipelineScheduler::DEFAULT,32),
    //
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,32,8,8,16,16,2,2,{4,64,1},{1,0,2},{1,0,2},2,4,8,true,{4,64,1},{1,0,2},{1,0,2},2,4,8,true,1,1,{1,32,1,4},4,BF16,BF16,ckb::PipelineScheduler::DEFAULT,8),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,64,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,4,8,true,{4,64,1},{1,0,2},{1,0,2},2,4,8,true,1,1,{1,32,1,4},4,BF16,BF16,ckb::PipelineScheduler::DEFAULT,8),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,64,32,8,8,32,32,4,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,BF16,BF16,ckb::PipelineScheduler::DEFAULT,8)
    //     // clang-format on
    // };
    return std::array<XdlInstance, 0>{};
}

// BF16 merged groups 2x V3 instance
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_merged_groups_bf16_v3_instances_2x(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: V3 merged_groups instances not yet supported by ck_tile builder
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,32,32,8,8,32,32,2,1,{4,64,1},{1,0,2},{1,0,2},2,4,8,true,{4,32,1},{1,0,2},{1,0,2},2,4,8,true,1,1,{1,32,1,4},4,BF16,BF16,Interwave,PipeV1,false,8)
    //     // clang-format on
    // };
    return std::array<XdlV3Instance, 0>{};
}

// ========================================================================================
// F16 merged groups instances (non-V3, 3 instances with NumGroupsToMerge = 8/16/32)
// ========================================================================================
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_merged_groups_f16_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: merged_groups instances with src_vector_dim=1 not yet supported by ck_tile builder
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,16,16,4,4,16,16,4,1,{4,16,1},{0,2,1},{0,2,1},1,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,16,1,4},1,F16,F16,ckb::PipelineScheduler::DEFAULT,8),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,16,16,4,4,16,16,4,1,{4,16,1},{0,2,1},{0,2,1},1,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,16,1,4},1,F16,F16,ckb::PipelineScheduler::DEFAULT,16),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,16,16,4,4,16,16,4,1,{4,16,1},{0,2,1},{0,2,1},1,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,16,1,4},1,F16,F16,ckb::PipelineScheduler::DEFAULT,32)
    //     // clang-format on
    // };
    return std::array<XdlInstance, 0>{};
}

// F16 merged groups V3 instance
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_merged_groups_f16_v3_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: V3 merged_groups instances not yet supported by ck_tile builder
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,32,32,8,8,32,32,2,1,{4,64,1},{1,0,2},{1,0,2},2,4,8,true,{4,32,1},{1,0,2},{1,0,2},2,4,8,true,1,1,{1,32,1,4},4,F16,F16,Interwave,PipeV1,false,8)
    //     // clang-format on
    // };
    return std::array<XdlV3Instance, 0>{};
}

// ========================================================================================
// F16 merged groups 2x instances (gfx950, non-V3 + V3)
// ========================================================================================
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_merged_groups_f16_instances_2x(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: merged_groups instances with mixed src_vector_dim not yet supported by ck_tile builder
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,16,32,8,4,16,16,4,1,{4,16,1},{0,2,1},{0,2,1},1,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,16,1,4},1,F16,F16,ckb::PipelineScheduler::DEFAULT,8),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,16,32,8,4,16,16,4,1,{4,16,1},{0,2,1},{0,2,1},1,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,16,1,4},1,F16,F16,ckb::PipelineScheduler::DEFAULT,16),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,16,32,8,4,16,16,4,1,{4,16,1},{0,2,1},{0,2,1},1,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,16,1,4},1,F16,F16,ckb::PipelineScheduler::DEFAULT,32),
    //
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,32,8,8,16,16,2,2,{4,64,1},{1,0,2},{1,0,2},2,4,8,true,{4,64,1},{1,0,2},{1,0,2},2,4,8,true,1,1,{1,32,1,4},4,F16,F16,ckb::PipelineScheduler::DEFAULT,8),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,64,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,4,8,true,{4,64,1},{1,0,2},{1,0,2},2,4,8,true,1,1,{1,32,1,4},4,F16,F16,ckb::PipelineScheduler::DEFAULT,8),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,64,32,8,8,32,32,4,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,F16,F16,ckb::PipelineScheduler::DEFAULT,8)
    //     // clang-format on
    // };
    return std::array<XdlInstance, 0>{};
}

// F16 merged groups 2x V3 instance
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_merged_groups_f16_v3_instances_2x(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: V3 merged_groups instances not yet supported by ck_tile builder
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,32,32,8,8,32,32,2,1,{4,64,1},{1,0,2},{1,0,2},2,4,8,true,{4,32,1},{1,0,2},{1,0,2},2,4,8,true,1,1,{1,32,1,4},4,F16,F16,Interwave,PipeV1,false,8)
    //     // clang-format on
    // };
    return std::array<XdlV3Instance, 0>{};
}

// ========================================================================================
// F32 merged groups instances (non-V3 only, 3 instances with NumGroupsToMerge = 8/16/32)
// ========================================================================================
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_merged_groups_f32_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: merged_groups instances with src_vector_dim=1 not yet supported by ck_tile builder
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,16,16,4,4,16,16,4,1,{4,16,1},{0,2,1},{0,2,1},1,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,16,1,4},1,F32,F32,ckb::PipelineScheduler::DEFAULT,8),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,16,16,4,4,16,16,4,1,{4,16,1},{0,2,1},{0,2,1},1,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,16,1,4},1,F32,F32,ckb::PipelineScheduler::DEFAULT,16),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,16,16,4,4,16,16,4,1,{4,16,1},{0,2,1},{0,2,1},1,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,16,1,4},1,F32,F32,ckb::PipelineScheduler::DEFAULT,32)
    //     // clang-format on
    // };
    return std::array<XdlInstance, 0>{};
}

// ========================================================================================
// F32 TF32 merged groups instances (non-V3 only)
// Note: CK source uses _k_per_block which is 32 on gfx950, 16 otherwise.
// We use 16 here to match the default non-gfx950 behavior.
// TODO: handle gfx950 k_per_block=32 variant
// ========================================================================================
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_merged_groups_f32_tf32_instances(
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
    // Also: merged_groups instances with src_vector_dim=1 not yet supported by ck_tile builder
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,16,16,4,4,16,16,4,1,{4,16,1},{0,2,1},{0,2,1},1,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,16,1,4},1,TF32,TF32,ckb::PipelineScheduler::DEFAULT,8),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,16,16,4,4,16,16,4,1,{4,16,1},{0,2,1},{0,2,1},1,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,16,1,4},1,TF32,TF32,ckb::PipelineScheduler::DEFAULT,16),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,16,16,4,4,16,16,4,1,{4,16,1},{0,2,1},{0,2,1},1,4,4,true,{4,16,1},{1,0,2},{1,0,2},2,1,4,true,1,1,{1,16,1,4},1,TF32,TF32,ckb::PipelineScheduler::DEFAULT,32)
    //     // clang-format on
    // };
    return std::array<XdlInstance, 0>{};
}

// ========================================================================================
// INT8 merged groups instances (non-V3 only, 3 instances with NumGroupsToMerge = 8/16/32)
// ========================================================================================
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_merged_groups_int8_instances(
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
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,32,64,32,8,8,32,32,1,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},1,I8,I8,ckb::PipelineScheduler::DEFAULT,8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,32,64,32,8,8,32,32,1,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},1,I8,I8,ckb::PipelineScheduler::DEFAULT,16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,32,64,32,8,8,32,32,1,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},1,I8,I8,ckb::PipelineScheduler::DEFAULT,32)
        // clang-format on
    };
}

} // namespace grouped_conv_fwd
} // namespace factories
} // namespace ck_builder
} // namespace conv
} // namespace miopen
