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

template <std::size_t NumDTensor = 1>
constexpr auto device_grouped_conv_fwd_xdl_binary_outelementop_f8_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    ckb::ElementwiseOperation output_op,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {F32})
{
#ifdef CK_ENABLE_FP8
    return std::array{
        // clang-format off
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F32,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,16,1,4},1,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F32,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},1,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F32,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,{4,64,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,32,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F32,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,256,128,32,8,8,32,32,4,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F32,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,256,32,8,8,32,32,2,4,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F32,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,128,32,8,8,32,32,4,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F32,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,128,32,8,8,32,32,2,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F32,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,64,32,8,8,32,32,2,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F32,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,64,128,32,8,8,32,32,2,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F32,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F32,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,128,64,32,8,8,32,32,2,1,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F32,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,256,64,128,32,8,8,32,32,1,2,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,{4,64,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F32,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,128,32,32,8,8,32,32,2,1,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,32,1,4},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F32,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,128,32,128,32,8,8,32,32,1,2,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,{4,32,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,8},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F32,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,F8,F8),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,F8,F8,F32,F32,ds_types,F8,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,32,64,32,8,8,32,32,1,2,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},8,F8,F8)
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
