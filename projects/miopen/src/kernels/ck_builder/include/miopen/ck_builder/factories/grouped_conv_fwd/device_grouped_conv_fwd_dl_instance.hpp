// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include <miopen/ck_builder/instance_data/dl.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/common_aliases.hpp>
#include <array>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace factories {
namespace grouped_conv_fwd {

using namespace instance;

// DL factory functions: 7 template parameters become function parameters
// Original CK template: device_grouped_conv2d_fwd_dl_f16_instances<InLayout, WeiLayout, DsLayout,
// OutLayout, DsDatatype, CDEElementOp, ConvSpec> Note: DL factories do NOT have NDimSpatial as a
// template parameter (hardcoded to 2)
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv2d_fwd_dl_f16_instances(
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    const std::array<ckb::DataType, NumDTensor>& ds_types,
    ckb::ElementwiseOperation cde_element_op,
    ckb::ConvSpecialization conv_spec)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<NumDTensor>(2,F16,F16,ds_types,F16,F32,input_layout,weight_layout,ds_layouts,output_layout,PassThrough,PassThrough,cde_element_op,conv_spec,GemmMNKPadding,8,16,4,2,1,1,2,1,{4,2},{1,1},{2,1,2,1},{1,1,8,1},{1,2,0,3},{1,2,0,3},{1,1,1,1},{1,2,0,3},{1,1,1,1},{1,1,1,1},{2,1,4,1},{1,2,0,3},{1,2,0,3},{1,1,1,1},{1,2,0,3},{1,1,1,1},{0,1,2,3,4,5},5,1),
        DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<NumDTensor>(2,F16,F16,ds_types,F16,F32,input_layout,weight_layout,ds_layouts,output_layout,PassThrough,PassThrough,cde_element_op,conv_spec,GemmMNKPadding,256,128,128,16,1,4,4,1,{8,2},{8,2},{8,1,1,1},{2,1,128,1},{1,2,0,3},{1,2,0,3},{1,1,1,1},{1,2,0,3},{1,1,1,1},{8,1,1,1},{2,1,128,1},{1,2,0,3},{1,2,0,3},{1,1,1,1},{1,2,0,3},{1,1,1,1},{0,1,2,3,4,5},5,1),
        DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<NumDTensor>(2,F16,F16,ds_types,F16,F32,input_layout,weight_layout,ds_layouts,output_layout,PassThrough,PassThrough,cde_element_op,conv_spec,GemmMNKPadding,256,128,128,16,2,4,4,1,{8,2},{8,2},{8,1,1,2},{2,1,128,1},{1,2,0,3},{1,2,0,3},{4,1,1,2},{1,2,0,3},{1,1,1,2},{8,1,1,2},{2,1,128,1},{1,2,0,3},{1,2,0,3},{4,1,1,2},{1,2,0,3},{1,1,1,2},{0,1,2,3,4,5},5,4)
        // clang-format on
    };
}

template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv2d_fwd_dl_f32_instances(
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    const std::array<ckb::DataType, NumDTensor>& ds_types,
    ckb::ElementwiseOperation cde_element_op,
    ckb::ConvSpecialization conv_spec)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<NumDTensor>(2,F32,F32,ds_types,F32,F32,input_layout,weight_layout,ds_layouts,output_layout,PassThrough,PassThrough,cde_element_op,conv_spec,GemmMNKPadding,8,16,4,2,1,1,2,1,{4,2},{1,1},{2,1,2,1},{1,1,8,1},{1,2,0,3},{1,2,0,3},{1,1,1,1},{1,2,0,3},{1,1,1,1},{1,1,1,1},{2,1,4,1},{1,2,0,3},{1,2,0,3},{1,1,1,1},{1,2,0,3},{1,1,1,1},{0,1,2,3,4,5},5,1),
        DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<NumDTensor>(2,F32,F32,ds_types,F32,F32,input_layout,weight_layout,ds_layouts,output_layout,PassThrough,PassThrough,cde_element_op,conv_spec,GemmMNKPadding,256,128,128,16,1,4,4,1,{8,2},{8,2},{8,1,1,1},{2,1,128,1},{1,2,0,3},{1,2,0,3},{1,1,1,1},{1,2,0,3},{1,1,1,1},{8,1,1,1},{2,1,128,1},{1,2,0,3},{1,2,0,3},{1,1,1,1},{1,2,0,3},{1,1,1,1},{0,1,2,3,4,5},5,1),
        DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<NumDTensor>(2,F32,F32,ds_types,F32,F32,input_layout,weight_layout,ds_layouts,output_layout,PassThrough,PassThrough,cde_element_op,conv_spec,GemmMNKPadding,256,128,128,16,1,4,4,1,{8,2},{8,2},{8,1,1,1},{2,1,128,1},{1,2,0,3},{1,2,0,3},{4,1,1,1},{1,2,0,3},{1,1,1,1},{8,1,1,1},{2,1,128,1},{1,2,0,3},{1,2,0,3},{4,1,1,1},{1,2,0,3},{1,1,1,1},{0,1,2,3,4,5},5,4)
        // clang-format on
    };
}

} // namespace grouped_conv_fwd
} // namespace factories
} // namespace ck_builder
} // namespace conv
} // namespace miopen
