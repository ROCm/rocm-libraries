// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include <miopen/ck_builder/instance_data/wmma_v3.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/common_aliases.hpp>
#include <array>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace factories {
namespace grouped_conv_fwd {

using namespace instance;

// DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3 is not supported by CK Builder.
// This factory preserves the function signature but returns an empty array.
//
// Original CK source:
//   device_grouped_conv_fwd_wmma_cshufflev3_wave_transfer_instances
//   Template class: DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3
//   8 instances total
//
// Original template parameters:
//   NDimSpatial, ALayout, BLayout, DsLayout, ELayout,
//   ConvSpec, GemmSpec, ABCDataTypes,
//   DsDataTypes = Tuple<>, OutElementOp = PassThrough
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_wmma_cshufflev3_wave_transfer_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    ckb::GemmSpecialization gemm_spec,
    ckb::DataType abc_data_type,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op                   = ckb::ElementwiseOperation::PASS_THROUGH)
{
    // TODO: DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3 is not supported by CK Builder.
    // Original instances (8 total, all using BlockGemmPipelineScheduler::Intrawave,
    // BlockGemmPipelineVersion::v1, false): return std::array{
    //     // clang-format off
    //     // generic instance
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,abc_data_type,abc_data_type,F32,abc_data_type,ds_types,abc_data_type,PassThrough,PassThrough,output_op,conv_spec,gemm_spec,256,128,128,64,8,8,16,16,2,4,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,64,1,4},8,Intrawave,PipeV1,false),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,abc_data_type,abc_data_type,F32,abc_data_type,ds_types,abc_data_type,PassThrough,PassThrough,output_op,conv_spec,gemm_spec,128,64,256,32,8,8,16,16,4,4,{4,32,1},{1,0,2},{1,0,2},2,8,8,false,{4,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},8,Intrawave,PipeV1,false),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,abc_data_type,abc_data_type,F32,abc_data_type,ds_types,abc_data_type,PassThrough,PassThrough,output_op,conv_spec,gemm_spec,128,64,192,64,8,8,16,16,2,6,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,4},8,Intrawave,PipeV1,false),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,abc_data_type,abc_data_type,F32,abc_data_type,ds_types,abc_data_type,PassThrough,PassThrough,output_op,conv_spec,gemm_spec,128,128,64,64,8,8,16,16,2,4,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,64,1,2},8,Intrawave,PipeV1,false),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,abc_data_type,abc_data_type,F32,abc_data_type,ds_types,abc_data_type,PassThrough,PassThrough,output_op,conv_spec,gemm_spec,256,128,256,64,8,8,16,16,4,4,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,8},8,Intrawave,PipeV1,false),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,abc_data_type,abc_data_type,F32,abc_data_type,ds_types,abc_data_type,PassThrough,PassThrough,output_op,conv_spec,gemm_spec,256,128,192,64,8,8,16,16,2,6,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,{8,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,64,1,4},8,Intrawave,PipeV1,false),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,abc_data_type,abc_data_type,F32,abc_data_type,ds_types,abc_data_type,PassThrough,PassThrough,output_op,conv_spec,gemm_spec,128,64,128,64,8,8,16,16,4,2,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,{8,16,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,16,1,8},8,Intrawave,PipeV1,false),
    //     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,abc_data_type,abc_data_type,F32,abc_data_type,ds_types,abc_data_type,PassThrough,PassThrough,output_op,conv_spec,gemm_spec,128,64,192,32,8,8,16,16,2,6,{4,32,1},{1,0,2},{1,0,2},2,8,8,false,{4,32,1},{1,0,2},{1,0,2},2,8,8,false,1,1,{1,32,1,4},8,Intrawave,PipeV1,false)
    //     // clang-format on
    // };
    return std::array<int, 0>{};
}

} // namespace grouped_conv_fwd
} // namespace factories
} // namespace ck_builder
} // namespace conv
} // namespace miopen
