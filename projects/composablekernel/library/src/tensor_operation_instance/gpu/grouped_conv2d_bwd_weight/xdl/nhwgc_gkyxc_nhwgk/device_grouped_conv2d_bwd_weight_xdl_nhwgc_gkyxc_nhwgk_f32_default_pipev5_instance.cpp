// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_v3_xdl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// Compilation parameters for in[n, hi, wi, g, c] * wei[g, k, y, x, c] = out[n, ho, wo, g, k]
void add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_f32_default_pipev5_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           NHWGC,
                                                           GKYXC,
                                                           NHWGK,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances)
{
    add_device_operation_instances(instances,
                                   device_grouped_conv_bwd_weight_v3_xdl_c_shuffle_f32_instances<
                                       2,
                                       NHWGC,
                                       GKYXC,
                                       NHWGK,
                                       ConvBwdWeightDefault,
                                       BlockGemmPipelineScheduler::Intrawave,
                                       BlockGemmPipelineVersion::v5>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
