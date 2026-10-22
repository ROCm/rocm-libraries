// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_native_output_instance.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#if defined(CK_USE_XDL) && defined(CK_ENABLE_FP32)
void add_device_grouped_conv2d_fwd_native_output_nhwgc_gkyxc_ngkhw_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NGKHW,
                                                                float,
                                                                float,
                                                                Empty_Tuple,
                                                                float,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances)
{
    if(!ck::is_gfx125_supported())
        return;
    using namespace native_output;
    add_device_operation_instances(instances,
                                   std::tuple<Candidate<2, float, 64, 64, 32, 1>,
                                              Candidate<2, float, 64, 64, 32, 4>,
                                              Candidate<2, float, 128, 128, 32, 1>,
                                              Candidate<2, float, 128, 128, 32, 4>,
                                              Candidate<2, float, 128, 128, 64, 1>,
                                              Candidate<2, float, 128, 128, 64, 4>>{});
}
#endif

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
