// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "device_gemm_xdl_universal_streamk_bf16_bf16_bf16_mk_nk_mn.hpp"
#include "ck/host_utility/device_prop.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_xdl_universal_streamk_bf16_bf16_bf16_mk_nk_mn_comp_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemm_Streamk_V2<Row,
                                                      Col,
                                                      Row,
                                                      BF16,
                                                      BF16,
                                                      BF16,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_xdl_universal_streamk_bf16_bf16_bf16_mk_nk_mn_comp_instances<GemmKPadding>{});

    if(ck::get_device_name() != "gfx950")
    {
        add_device_operation_instances(
            instances,
            device_gemm_xdl_universal_streamk_bf16_bf16_bf16_mk_nk_mn_comp_instances_part2<
                GemmKPadding>{});
    }
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
