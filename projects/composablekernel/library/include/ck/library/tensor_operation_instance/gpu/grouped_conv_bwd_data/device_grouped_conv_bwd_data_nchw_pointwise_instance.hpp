// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_nchw_pointwise.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using DeviceNCHWPointwiseBwdData =
    DeviceGroupedConvBwdDataMultipleD<2,
                                      tensor_layout::convolution::NGKHW,
                                      tensor_layout::convolution::GKCYX,
                                      Tuple<>,
                                      tensor_layout::convolution::NGCHW,
                                      float,
                                      float,
                                      Tuple<>,
                                      float,
                                      element_wise::PassThrough,
                                      element_wise::PassThrough,
                                      element_wise::PassThrough>;

// Header-defined instances also remain reachable in MIOpen-only builds, which omit
// the unrelated native-layout XDL archive. Architecture/geometry checks live in the operator.
inline void add_device_grouped_conv2d_bwd_data_nchw_pointwise_f32_instances(
    std::vector<std::unique_ptr<DeviceNCHWPointwiseBwdData>>& instances)
{
#if defined(CK_USE_GFX1250) && defined(CK_ENABLE_FP32)
    using Instances = std::tuple<DeviceGroupedConvBwdDataNCHWPointwise<256, 4>,
                                 DeviceGroupedConvBwdDataNCHWPointwise<256, 8>>;
    add_device_operation_instances(instances, Instances{});
#else
    (void)instances;
#endif
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
