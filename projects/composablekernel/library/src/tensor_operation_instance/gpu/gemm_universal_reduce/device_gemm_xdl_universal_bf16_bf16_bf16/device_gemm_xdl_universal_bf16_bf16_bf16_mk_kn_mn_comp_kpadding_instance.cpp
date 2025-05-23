// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn.hpp"
#include "ck/host_utility/device_prop.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_xdl_universal_reduce_bf16_bf16_bf16_mk_kn_mn_comp_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               BF16,
                                               BF16,
                                               DsDataType,
                                               BF16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_xdl_universal_reduce_bf16_bf16_bf16_mk_kn_mn_comp_instances<GemmKPadding>{});

    if(ck::get_device_name() != "gfx950")
    {
        add_device_operation_instances(
            instances,
            device_gemm_xdl_universal_reduce_bf16_bf16_bf16_mk_kn_mn_comp_instances_part2<
                GemmKPadding>{});
    }
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
