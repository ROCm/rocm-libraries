// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn.hpp"
#include "ck/host_utility/device_prop.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_mem_instances<Intrawave, GemmKPadding>{});

    if(ck::get_device_name() != "gfx950")
    {
        add_device_operation_instances(
            instances,
            device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_mem_instances_part2<Intrawave,
                                                                                  GemmKPadding>{});
    }
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
