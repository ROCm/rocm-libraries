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

// double rate mfma instances on gfx950
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_bf16_comp_instances_2x(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: V3 comp instances not yet supported by ck_tile builder (ValidCBlockTransfer fails)
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,64,16,16,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,BF16,BF16,Interwave,PipeV1)
    //     // clang-format on
    // };
    return std::array<XdlV3Instance, 0>{};
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_bf16_comp_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: V3 comp instances not yet supported by ck_tile builder (ValidCBlockTransfer fails)
    // return std::array{
    //     // clang-format off
    //     // Compute friendly
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,256,32,8,8,32,32,4,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,BF16,BF16,Intrawave,PipeV4),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,64,8,8,32,32,2,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,BF16,BF16,Intrawave,PipeV4),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,BF16,BF16,Intrawave,PipeV4),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,256,32,8,8,32,32,4,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,BF16,BF16,Intrawave,PipeV3),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,256,32,8,8,32,32,4,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,BF16,BF16,Intrawave,PipeV5),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,256,32,8,8,32,32,2,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,BF16,BF16,Interwave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,128,32,8,8,32,32,4,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,BF16,BF16,Interwave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,64,8,8,32,32,2,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,BF16,BF16,Interwave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,64,64,8,8,32,32,2,1,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,BF16,BF16,Intrawave,PipeV3),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,128,64,8,8,32,32,1,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,BF16,BF16,Intrawave,PipeV3),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,64,64,8,8,32,32,1,1,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,BF16,BF16,Intrawave,PipeV3)
    //     // clang-format on
    // };
    return std::array<XdlV3Instance, 0>{};
}

// instances not working on gfx950
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_bf16_comp_instances_part2(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: V3 comp instances not yet supported by ck_tile builder (ValidCBlockTransfer fails)
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,256,32,8,8,16,16,8,8,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,1,2,{1,32,1,8},8,BF16,BF16,Intrawave,PipeV3),
    //     // AGPR Spill when use permuted lds layout. so, use padding for these two.
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,224,256,64,8,8,16,16,7,8,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,2,{1,32,1,8},8,BF16,BF16,Intrawave,PipeV3),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,224,64,8,8,16,16,8,7,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,2,1,{1,64,1,4},8,BF16,BF16,Intrawave,PipeV3),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,64,8,8,32,32,2,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,BF16,BF16,Intrawave,PipeV3),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,64,8,8,32,32,2,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,BF16,BF16,Intrawave,PipeV5)
    //     // clang-format on
    // };
    return std::array<XdlV3Instance, 0>{};
}

// double rate mfma instances on gfx950
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_f16_comp_instances_2x(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: V3 comp instances not yet supported by ck_tile builder (ValidCBlockTransfer fails)
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,64,16,16,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F16,F16,Interwave,PipeV1)
    //     // clang-format on
    // };
    return std::array<XdlV3Instance, 0>{};
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_f16_comp_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: V3 comp instances not yet supported by ck_tile builder (ValidCBlockTransfer fails)
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,64,8,8,32,32,2,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,F16,F16,Intrawave,PipeV4)
    //     // clang-format on
    // };
    return std::array<XdlV3Instance, 0>{};
}

// instances not working on gfx950
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_f16_comp_instances_part2(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: V3 comp instances not yet supported by ck_tile builder (ValidCBlockTransfer fails)
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,256,32,8,8,32,32,4,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,F16,F16,Intrawave,PipeV4),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,F16,F16,Intrawave,PipeV4),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,256,32,8,8,32,32,4,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,F16,F16,Intrawave,PipeV3),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,256,32,8,8,32,32,4,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,F16,F16,Intrawave,PipeV5),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,256,32,8,8,16,16,8,8,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,1,2,{1,32,1,8},8,F16,F16,Intrawave,PipeV3),
    //     // AGPR Spill when use permuted lds layout. so, use padding for these two.
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,224,256,64,8,8,16,16,7,8,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,2,{1,32,1,8},8,F16,F16,Intrawave,PipeV3),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,224,64,8,8,16,16,8,7,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,2,1,{1,64,1,4},8,F16,F16,Intrawave,PipeV3),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,64,8,8,32,32,2,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,F16,F16,Intrawave,PipeV3),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,64,8,8,32,32,2,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,F16,F16,Intrawave,PipeV5),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,256,32,8,8,32,32,2,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,F16,F16,Interwave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,128,32,8,8,32,32,4,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,F16,F16,Interwave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F16,F16,F32,F16,ds_types,F16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,64,8,8,32,32,2,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,F16,F16,Interwave,PipeV1)
    //     // clang-format on
    // };
    return std::array<XdlV3Instance, 0>{};
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_f32_comp_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: V3 comp instances not yet supported by ck_tile builder (ValidCBlockTransfer fails)
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,4,4,false,{4,64,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,32,1,8},8,F32,F32,Intrawave,PipeV4),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,64,8,8,32,32,2,2,{8,32,1},{1,0,2},{1,0,2},2,4,4,false,{8,32,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,32,1,8},8,F32,F32,Intrawave,PipeV3),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,64,8,8,32,32,2,2,{8,32,1},{1,0,2},{1,0,2},2,4,4,false,{8,32,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,32,1,8},8,F32,F32,Intrawave,PipeV5),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,64,8,8,32,32,2,2,{8,32,1},{1,0,2},{1,0,2},2,4,4,false,{8,32,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,32,1,8},8,F32,F32,Interwave,PipeV1)
    //     // clang-format on
    // };
    return std::array<XdlV3Instance, 0>{};
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_f32_tf32_comp_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: V3 comp instances not yet supported by ck_tile builder (ValidCBlockTransfer fails)
    // Also: TF32 compute type is not yet supported in ckb::DataType enum
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,4,4,false,{4,64,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,32,1,8},8,TF32,TF32,Intrawave,PipeV4),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,64,8,8,32,32,2,2,{8,32,1},{1,0,2},{1,0,2},2,4,4,false,{8,32,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,32,1,8},8,TF32,TF32,Intrawave,PipeV3),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,64,8,8,32,32,2,2,{8,32,1},{1,0,2},{1,0,2},2,4,4,false,{8,32,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,32,1,8},8,TF32,TF32,Intrawave,PipeV5),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F32,F32,F32,F32,ds_types,F32,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,64,8,8,32,32,2,2,{8,32,1},{1,0,2},{1,0,2},2,4,4,false,{8,32,1},{1,0,2},{1,0,2},2,4,4,false,1,1,{1,32,1,8},8,TF32,TF32,Interwave,PipeV1)
    //     // clang-format on
    // };
    return std::array<XdlV3Instance, 0>{};
}

// double rate mfma instances on gfx950
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_int8_comp_instances_2x(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: V3 comp instances not yet supported by ck_tile builder (ValidCBlockTransfer fails)
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,128,32,32,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,16,16,true,{4,64,1},{1,0,2},{1,0,2},2,16,16,true,1,1,{1,64,1,4},16,I8,I8,Interwave,PipeV1)
    //     // clang-format on
    // };
    return std::array<XdlV3Instance, 0>{};
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_int8_comp_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: V3 comp instances not yet supported by ck_tile builder (ValidCBlockTransfer fails)
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,I8,I8,Intrawave,PipeV4)
    //     // clang-format on
    // };
    return std::array<XdlV3Instance, 0>{};
}

// instances not working on gfx950
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_int8_comp_instances_part2(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: V3 comp instances not yet supported by ck_tile builder (ValidCBlockTransfer fails)
    // return std::array{
    //     // clang-format off
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,256,32,8,8,32,32,4,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,I8,I8,Intrawave,PipeV4),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,256,32,8,8,32,32,4,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,I8,I8,Intrawave,PipeV3),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,256,32,8,8,32,32,4,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,I8,I8,Intrawave,PipeV5),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,256,32,8,8,16,16,8,8,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,1,2,{1,32,1,8},8,I8,I8,Intrawave,PipeV3),
    //     // AGPR Spill when use permuted lds layout. so, use padding for these two.
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,64,8,8,32,32,2,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,I8,I8,Intrawave,PipeV3),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,256,32,8,8,32,32,2,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,I8,I8,Interwave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,128,32,8,8,32,32,4,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,{4,64,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,I8,I8,Interwave,PipeV1),
    //     DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,I8,I8,I32,I8,ds_types,I8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,64,8,8,32,32,2,2,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,I8,I8,Interwave,PipeV1)
    //     // clang-format on
    // };
    return std::array<XdlV3Instance, 0>{};
}

} // namespace grouped_conv_fwd
} // namespace factories
} // namespace ck_builder
} // namespace conv
} // namespace miopen
