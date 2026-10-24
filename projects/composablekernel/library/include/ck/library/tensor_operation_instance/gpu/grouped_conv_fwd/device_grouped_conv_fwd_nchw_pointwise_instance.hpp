// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_abd.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#if defined(CK_USE_GFX1250) && defined(CK_ENABLE_FP16)
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_nchw_pointwise.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#endif

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using DeviceNCHWPointwiseFwd = DeviceGroupedConvFwdMultipleABD<2,
                                                               tensor_layout::convolution::NGCHW,
                                                               tensor_layout::convolution::GKCYX,
                                                               Tuple<>,
                                                               tensor_layout::convolution::NGKHW,
                                                               half_t,
                                                               half_t,
                                                               Tuple<>,
                                                               half_t,
                                                               element_wise::PassThrough,
                                                               element_wise::PassThrough,
                                                               element_wise::PassThrough>;

inline void add_device_grouped_conv2d_fwd_nchw_pointwise_f16_instances(
    std::vector<std::unique_ptr<DeviceNCHWPointwiseFwd>>& instances)
{
#if defined(CK_USE_GFX1250) && defined(CK_ENABLE_FP16)
    add_device_operation_instances(instances, std::tuple<DeviceGroupedConvFwdNCHWPointwise>{});
#else
    (void)instances;
#endif
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
