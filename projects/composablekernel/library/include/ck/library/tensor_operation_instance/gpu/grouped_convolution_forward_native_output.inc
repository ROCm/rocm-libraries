// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#if defined(CK_USE_WMMA) && defined(CK_ENABLE_FP16)
void add_device_grouped_conv2d_fwd_native_output_nhwgc_gkyxc_ngkhw_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NGKHW,
                                                                half_t,
                                                                half_t,
                                                                Empty_Tuple,
                                                                half_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

#if defined(CK_USE_WMMA) && defined(CK_ENABLE_BF16)
void add_device_grouped_conv2d_fwd_native_output_nhwgc_gkyxc_ngkhw_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NGKHW,
                                                                bhalf_t,
                                                                bhalf_t,
                                                                Empty_Tuple,
                                                                bhalf_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

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
                                                                PassThrough>>>& instances);
#endif

#if defined(CK_USE_WMMA) && defined(CK_ENABLE_FP16)
void add_device_grouped_conv3d_fwd_native_output_ndhwgc_gkzyxc_ngkdhw_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NGKDHW,
                                                                half_t,
                                                                half_t,
                                                                Empty_Tuple,
                                                                half_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

#if defined(CK_USE_WMMA) && defined(CK_ENABLE_BF16)
void add_device_grouped_conv3d_fwd_native_output_ndhwgc_gkzyxc_ngkdhw_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NGKDHW,
                                                                bhalf_t,
                                                                bhalf_t,
                                                                Empty_Tuple,
                                                                bhalf_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

#if defined(CK_USE_XDL) && defined(CK_ENABLE_FP32)
void add_device_grouped_conv3d_fwd_native_output_ndhwgc_gkzyxc_ngkdhw_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NGKDHW,
                                                                float,
                                                                float,
                                                                Empty_Tuple,
                                                                float,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
