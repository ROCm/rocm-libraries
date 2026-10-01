// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The WMMA filename marker selects RDNA-only compilation for these scalar wave32 kernels.
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_depthwise_bf16.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_depthwise_row_strip_bf16.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_depthwise_grouped_row_strip_bf16.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_ENABLE_BF16
void add_device_grouped_conv2d_bwd_weight_depthwise_nhwgc_gkyxc_nhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           NHWGC,
                                                           GKYXC,
                                                           NHWGK,
                                                           BF16,
                                                           BF16,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances)
{
    instances.emplace_back(std::make_unique<DeviceGroupedConvBwdWeightDepthwiseBf16>());
}

void add_device_grouped_conv2d_bwd_weight_depthwise_row_strip_nhwgc_gkyxc_nhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           NHWGC,
                                                           GKYXC,
                                                           NHWGK,
                                                           BF16,
                                                           BF16,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances)
{
    instances.emplace_back(std::make_unique<DeviceGroupedConvBwdWeightDepthwiseRowStripBf16>());
}

void add_device_grouped_conv2d_bwd_weight_depthwise_grouped_row_strip_nhwgc_gkyxc_nhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           NHWGC,
                                                           GKYXC,
                                                           NHWGK,
                                                           BF16,
                                                           BF16,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances)
{
    instances.emplace_back(
        std::make_unique<DeviceGroupedConvBwdWeightDepthwiseGroupedRowStripBf16>());
}
#endif

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
