// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_nchw_pointwise_bf16_wmma.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_ENABLE_BF16
void add_device_grouped_conv2d_bwd_weight_nchw_pointwise_bf16_wmma_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           NGCHW,
                                                           GKCYX,
                                                           NGKHW,
                                                           BF16,
                                                           BF16,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances)
{
    instances.emplace_back(std::make_unique<DeviceGroupedConvBwdWeightNchwPointwiseBf16Wmma>());
}
#endif

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
