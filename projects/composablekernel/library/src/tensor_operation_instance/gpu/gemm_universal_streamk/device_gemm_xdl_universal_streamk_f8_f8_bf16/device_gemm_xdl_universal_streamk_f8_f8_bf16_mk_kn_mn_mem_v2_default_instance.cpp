// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "device_gemm_xdl_universal_streamk_f8_f8_bf16_mk_kn_mn.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_xdl_universal_streamk_f8_f8_bf16_mk_kn_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemm_Streamk_V2<Row, Row, Row, F8, F8, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_xdl_universal_streamk_f8_f8_bf16_mk_kn_mn_mem_instances<Interwave,
                                                                            GemmDefault>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
