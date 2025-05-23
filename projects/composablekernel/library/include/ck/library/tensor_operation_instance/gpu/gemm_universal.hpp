// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_v2.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

#ifdef CK_USE_WMMA
#include "gemm_universal_wmma.inc"
#endif
#ifdef CK_USE_XDL
#include "gemm_universal_xdl.inc"
#endif

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmV2<ALayout,
                                               BLayout,
                                               CLayout,
                                               ADataType,
                                               BDataType,
                                               CDataType,
                                               ck::tensor_operation::element_wise::PassThrough,
                                               ck::tensor_operation::element_wise::PassThrough,
                                               ck::tensor_operation::element_wise::PassThrough>>
{
    using DeviceOp = DeviceGemmV2<ALayout,
                                  BLayout,
                                  CLayout,
                                  ADataType,
                                  BDataType,
                                  CDataType,
                                  ck::tensor_operation::element_wise::PassThrough,
                                  ck::tensor_operation::element_wise::PassThrough,
                                  ck::tensor_operation::element_wise::PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_USE_WMMA
#ifdef CK_ENABLE_FP16
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_wmma_universal_f16_f16_f16_mk_kn_mn_comp_default_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_wmma_universal_f16_f16_f16_mk_nk_mn_comp_default_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_wmma_universal_f16_f16_f16_km_kn_mn_comp_default_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_wmma_universal_f16_f16_f16_km_nk_mn_comp_default_instances(op_ptrs);
            }
        }
#endif
#ifdef CK_ENABLE_BF16
        if constexpr(is_same_v<ADataType, bhalf_t> && is_same_v<BDataType, bhalf_t> &&
                     is_same_v<CDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_wmma_universal_bf16_bf16_bf16_mk_kn_mn_comp_default_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_wmma_universal_bf16_bf16_bf16_mk_nk_mn_comp_default_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_wmma_universal_bf16_bf16_bf16_km_kn_mn_comp_default_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_wmma_universal_bf16_bf16_bf16_km_nk_mn_comp_default_instances(
                    op_ptrs);
            }
        }
#endif
#if(defined(CK_ENABLE_BF16) && defined(CK_ENABLE_FP8))
        if constexpr(is_same_v<ADataType, f8_t> && is_same_v<BDataType, f8_t> &&
                     is_same_v<CDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_wmma_universal_f8_f8_bf16_mk_kn_mn_comp_default_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_wmma_universal_f8_f8_bf16_mk_nk_mn_comp_default_instances(op_ptrs);
            }
        }
#endif
#endif // CK_USE_WMMA

#ifdef CK_USE_XDL
#ifdef CK_ENABLE_FP16
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_comp_default_instances(op_ptrs);
                add_device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_comp_kpadding_instances(op_ptrs);
                add_device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_comp_mnpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_comp_mnkpadding_instances(
                    op_ptrs);

                add_device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_mem_v1_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_mem_v1_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_mem_v1_mnkpadding_instances(
                    op_ptrs);

                add_device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_mem_v2_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_mem_v2_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_mem_v2_mnkpadding_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_universal_f16_f16_f16_mk_nk_mn_comp_default_instances(op_ptrs);
                add_device_gemm_xdl_universal_f16_f16_f16_mk_nk_mn_comp_kpadding_instances(op_ptrs);

                add_device_gemm_xdl_universal_f16_f16_f16_mk_nk_mn_mem_v1_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_f16_f16_f16_mk_nk_mn_mem_v1_kpadding_instances(
                    op_ptrs);

                add_device_gemm_xdl_universal_f16_f16_f16_mk_nk_mn_mem_v2_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_f16_f16_f16_mk_nk_mn_mem_v2_kpadding_instances(
                    op_ptrs);
            }
        }
#endif
#if(defined(CK_ENABLE_FP16) && defined(CK_ENABLE_FP8))
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, f8_t> &&
                     is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_universal_f16_f8_f16_mk_nk_mn_comp_default_instances(op_ptrs);
                add_device_gemm_xdl_universal_f16_f8_f16_mk_nk_mn_comp_kpadding_instances(op_ptrs);

                add_device_gemm_xdl_universal_f16_f8_f16_mk_nk_mn_mem_v1_default_instances(op_ptrs);
                add_device_gemm_xdl_universal_f16_f8_f16_mk_nk_mn_mem_v1_kpadding_instances(
                    op_ptrs);

                add_device_gemm_xdl_universal_f16_f8_f16_mk_nk_mn_mem_v2_default_instances(op_ptrs);
                add_device_gemm_xdl_universal_f16_f8_f16_mk_nk_mn_mem_v2_kpadding_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_universal_f16_f8_f16_mk_kn_mn_comp_default_instances(op_ptrs);
                add_device_gemm_xdl_universal_f16_f8_f16_mk_kn_mn_comp_kpadding_instances(op_ptrs);
                add_device_gemm_xdl_universal_f16_f8_f16_mk_kn_mn_comp_mnpadding_instances(op_ptrs);
                add_device_gemm_xdl_universal_f16_f8_f16_mk_kn_mn_comp_mnkpadding_instances(
                    op_ptrs);

                add_device_gemm_xdl_universal_f16_f8_f16_mk_kn_mn_mem_v1_default_instances(op_ptrs);
                add_device_gemm_xdl_universal_f16_f8_f16_mk_kn_mn_mem_v1_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_f16_f8_f16_mk_kn_mn_mem_v1_mnkpadding_instances(
                    op_ptrs);

                add_device_gemm_xdl_universal_f16_f8_f16_mk_kn_mn_mem_v2_default_instances(op_ptrs);
                add_device_gemm_xdl_universal_f16_f8_f16_mk_kn_mn_mem_v2_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_f16_f8_f16_mk_kn_mn_mem_v2_mnkpadding_instances(
                    op_ptrs);
            }
        }
        else if constexpr(is_same_v<ADataType, f8_t> && is_same_v<BDataType, half_t> &&
                          is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_universal_f8_f16_f16_mk_nk_mn_comp_default_instances(op_ptrs);
                add_device_gemm_xdl_universal_f8_f16_f16_mk_nk_mn_comp_kpadding_instances(op_ptrs);

                add_device_gemm_xdl_universal_f8_f16_f16_mk_nk_mn_mem_v1_default_instances(op_ptrs);
                add_device_gemm_xdl_universal_f8_f16_f16_mk_nk_mn_mem_v1_kpadding_instances(
                    op_ptrs);

                add_device_gemm_xdl_universal_f8_f16_f16_mk_nk_mn_mem_v2_default_instances(op_ptrs);
                add_device_gemm_xdl_universal_f8_f16_f16_mk_nk_mn_mem_v2_kpadding_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_universal_f8_f16_f16_mk_kn_mn_comp_default_instances(op_ptrs);
                add_device_gemm_xdl_universal_f8_f16_f16_mk_kn_mn_comp_kpadding_instances(op_ptrs);
                add_device_gemm_xdl_universal_f8_f16_f16_mk_kn_mn_comp_mnpadding_instances(op_ptrs);
                add_device_gemm_xdl_universal_f8_f16_f16_mk_kn_mn_comp_mnkpadding_instances(
                    op_ptrs);

                add_device_gemm_xdl_universal_f8_f16_f16_mk_kn_mn_mem_v1_default_instances(op_ptrs);
                add_device_gemm_xdl_universal_f8_f16_f16_mk_kn_mn_mem_v1_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_f8_f16_f16_mk_kn_mn_mem_v1_mnkpadding_instances(
                    op_ptrs);

                add_device_gemm_xdl_universal_f8_f16_f16_mk_kn_mn_mem_v2_default_instances(op_ptrs);
                add_device_gemm_xdl_universal_f8_f16_f16_mk_kn_mn_mem_v2_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_f8_f16_f16_mk_kn_mn_mem_v2_mnkpadding_instances(
                    op_ptrs);
            }
        }
#endif
#ifdef CK_ENABLE_BF16
        if constexpr(is_same_v<ADataType, bhalf_t> && is_same_v<BDataType, bhalf_t> &&
                     is_same_v<CDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_comp_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_comp_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_comp_mnpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_comp_mnkpadding_instances(
                    op_ptrs);

                add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_mem_v1_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_mem_v1_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_mem_v1_mnkpadding_instances(
                    op_ptrs);

                add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_mem_v2_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_mem_v2_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_mem_v2_mnkpadding_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_comp_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_comp_kpadding_instances(
                    op_ptrs);

                add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_mem_v1_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_mem_v1_kpadding_instances(
                    op_ptrs);

                add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_mem_v2_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_mem_v2_kpadding_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_comp_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_comp_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_comp_mnpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_comp_mnkpadding_instances(
                    op_ptrs);

                add_device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_mem_v1_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_mem_v1_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_mem_v1_mnkpadding_instances(
                    op_ptrs);

                add_device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_mem_v2_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_mem_v2_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_mem_v2_mnkpadding_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_comp_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_comp_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_comp_mpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_comp_mkpadding_instances(
                    op_ptrs);

                add_device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_mem_v1_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_mem_v1_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_mem_v1_mkpadding_instances(
                    op_ptrs);

                add_device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_mem_v2_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_mem_v2_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_mem_v2_mkpadding_instances(
                    op_ptrs);
            }
        }
#endif
#if(defined(CK_ENABLE_BF16) && defined(CK_ENABLE_FP8))
        if constexpr(is_same_v<ADataType, f8_t> && is_same_v<BDataType, f8_t> &&
                     is_same_v<CDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_universal_f8_f8_bf16_mk_kn_mn_comp_default_instances(op_ptrs);
                add_device_gemm_xdl_universal_f8_f8_bf16_mk_kn_mn_comp_kpadding_instances(op_ptrs);
                add_device_gemm_xdl_universal_f8_f8_bf16_mk_kn_mn_comp_nkpadding_instances(op_ptrs);

                add_device_gemm_xdl_universal_f8_f8_bf16_mk_kn_mn_mem_v1_default_instances(op_ptrs);
                add_device_gemm_xdl_universal_f8_f8_bf16_mk_kn_mn_mem_v1_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_f8_f8_bf16_mk_kn_mn_mem_v1_nkpadding_instances(
                    op_ptrs);

                add_device_gemm_xdl_universal_f8_f8_bf16_mk_kn_mn_mem_v2_default_instances(op_ptrs);
                add_device_gemm_xdl_universal_f8_f8_bf16_mk_kn_mn_mem_v2_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_f8_f8_bf16_mk_kn_mn_mem_v2_nkpadding_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_comp_default_instances(op_ptrs);
                add_device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_comp_kpadding_instances(op_ptrs);

                add_device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_v1_default_instances(op_ptrs);
                add_device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_v1_kpadding_instances(
                    op_ptrs);

                add_device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_v2_default_instances(op_ptrs);
                add_device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_v2_kpadding_instances(
                    op_ptrs);
            }
        }
#endif

#ifdef CK_ENABLE_FP16
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, pk_i4_t> &&
                     is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_universal_f16_i4_f16_mk_nk_mn_mem_v2_default_instances(op_ptrs);
            }
        }
#endif
#ifdef CK_ENABLE_BF16
        if constexpr(is_same_v<ADataType, bhalf_t> && is_same_v<BDataType, pk_i4_t> &&
                     is_same_v<CDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_universal_bf16_i4_bf16_mk_nk_mn_mem_v2_default_instances(
                    op_ptrs);
            }
        }
#endif
#endif // CK_USE_XDL

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
