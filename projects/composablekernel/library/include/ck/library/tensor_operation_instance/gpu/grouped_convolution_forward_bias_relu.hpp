// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_abd.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

#ifdef CK_USE_XDL
#include "grouped_convolution_forward_bias_relu_xdl.inc"
#endif

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <ck::index_t NumDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename DLayouts,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename DDataTypes,
          typename AComputeType,
          typename BComputeType>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
    NumDimSpatial,
    InLayout,
    WeiLayout,
    DLayouts,
    OutLayout,
    InDataType,
    WeiDataType,
    DDataTypes,
    OutDataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::AddRelu,
    AComputeType,
    BComputeType>>
{
    using DeviceOp =
        DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                        InLayout,
                                        WeiLayout,
                                        DLayouts,
                                        OutLayout,
                                        InDataType,
                                        WeiDataType,
                                        DDataTypes,
                                        OutDataType,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::AddRelu,
                                        AComputeType,
                                        BComputeType>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_USE_XDL
        // layout NHWGC/GKYXC/NHWGK
        if constexpr(NumDimSpatial == 2 && is_same_v<InLayout, NHWGC> &&
                     is_same_v<WeiLayout, GKYXC> && is_same_v<OutLayout, NHWGK>)
        {
#ifdef CK_ENABLE_BF16
            if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                         is_same_v<WeiDataType, ck::bhalf_t> &&
                         is_same_v<OutDataType, ck::bhalf_t> &&
                         is_same_v<AComputeType, ck::bhalf_t> &&
                         is_same_v<BComputeType, ck::bhalf_t>)
            {
                add_device_grouped_conv2d_fwd_bias_relu_xdl_nhwgc_gkyxc_nhwgk_bf16_instances(
                    op_ptrs);
                add_device_grouped_conv2d_fwd_bias_relu_xdl_nhwgc_gkyxc_nhwgk_bf16_16x16_instances(
                    op_ptrs);
                add_device_grouped_conv2d_fwd_bias_relu_xdl_large_tensor_nhwgc_gkyxc_nhwgk_bf16_instances(
                    op_ptrs);
                add_device_grouped_conv2d_fwd_bias_relu_xdl_merged_groups_nhwgc_gkyxc_nhwgk_bf16_instances(
                    op_ptrs);
                add_device_grouped_conv2d_fwd_bias_relu_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_instances(
                    op_ptrs);
                add_device_grouped_conv2d_fwd_bias_relu_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_2x_instances(
                    op_ptrs);
                add_device_grouped_conv2d_fwd_bias_relu_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_part2_instances(
                    op_ptrs);
                add_device_grouped_conv2d_fwd_bias_relu_xdl_nhwgc_gkyxc_nhwgk_bf16_mem_intra_instances(
                    op_ptrs);
                add_device_grouped_conv2d_fwd_bias_relu_xdl_nhwgc_gkyxc_nhwgk_bf16_mem_inter_instances(
                    op_ptrs);
            }
#endif
        }
        // layout NDHWGC/GKZYXC/NDHWGK
        if constexpr(NumDimSpatial == 3 && is_same_v<InLayout, NDHWGC> &&
                     is_same_v<WeiLayout, GKZYXC> && is_same_v<OutLayout, NDHWGK>)
        {
#ifdef CK_ENABLE_BF16
            if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                         is_same_v<WeiDataType, ck::bhalf_t> &&
                         is_same_v<OutDataType, ck::bhalf_t> &&
                         is_same_v<AComputeType, ck::bhalf_t> &&
                         is_same_v<BComputeType, ck::bhalf_t>)
            {
                add_device_grouped_conv3d_fwd_bias_relu_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_bias_relu_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_16x16_instances(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_bias_relu_xdl_large_tensor_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_bias_relu_xdl_merged_groups_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_bias_relu_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_comp_instances(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_bias_relu_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_mem_intra_instances(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_bias_relu_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_mem_inter_instances(
                    op_ptrs);
            }
#endif
        }
#endif // CK_USE_XDL

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
