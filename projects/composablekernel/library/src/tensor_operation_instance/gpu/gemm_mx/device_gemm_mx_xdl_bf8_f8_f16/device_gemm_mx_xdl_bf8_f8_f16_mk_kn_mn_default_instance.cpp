// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "device_gemm_mx_xdl_bf8_f8_f16_mk_kn_mn.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_mx_xdl_bf8_f8_f16_mk_kn_mn_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMX<Row,
                                             Row,
                                             Row,
                                             BF8,
                                             E8M0,
                                             F8,
                                             E8M0,
                                             F16,
                                             32,
                                             PassThrough,
                                             PassThrough,
                                             PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances, device_gemm_mx_xdl_bf8_f8_f16_mk_kn_mn_instances<Intrawave, GemmDefault>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
