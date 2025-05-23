// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "device_batched_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn.hpp"
#include "ck/host_utility/device_prop.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_batched_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_comp_default_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmV2MultiD<Row,
                                                          Col,
                                                          ck::Tuple<>,
                                                          Row,
                                                          BF16,
                                                          BF16,
                                                          ck::Tuple<>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_batched_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_comp_instances<GemmDefault>{});

    if(ck::get_device_name() != "gfx950")
    {
        add_device_operation_instances(
            instances,
            device_batched_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_comp_instances_part2<
                GemmDefault>{});
    }
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
