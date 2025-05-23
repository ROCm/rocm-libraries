// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_ENABLE_FP16
void add_device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_comp_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_comp_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_comp_mnpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_comp_mnkpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_mem_v1_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_mem_v1_mnkpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_mem_v2_mnkpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f16_f16_mk_nk_mn_comp_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f16_f16_mk_nk_mn_comp_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f16_f16_mk_nk_mn_mem_v1_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f16_f16_mk_nk_mn_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f16_f16_mk_nk_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f16_f16_mk_nk_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);
#endif
#if(defined(CK_ENABLE_FP16) && defined(CK_ENABLE_FP8))
void add_device_gemm_xdl_universal_f16_f8_f16_mk_kn_mn_comp_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f8_f16_mk_kn_mn_comp_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f8_f16_mk_kn_mn_comp_mnpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f8_f16_mk_kn_mn_comp_mnkpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f8_f16_mk_kn_mn_mem_v1_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f8_f16_mk_kn_mn_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f8_f16_mk_kn_mn_mem_v1_mnkpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f8_f16_mk_kn_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f8_f16_mk_kn_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f8_f16_mk_kn_mn_mem_v2_mnkpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);
void add_device_gemm_xdl_universal_f16_f8_f16_mk_nk_mn_comp_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f8_f16_mk_nk_mn_comp_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f8_f16_mk_nk_mn_mem_v1_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f8_f16_mk_nk_mn_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f8_f16_mk_nk_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_i4_f16_mk_nk_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F16, I4, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_i4_bf16_mk_nk_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, BF16, I4, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f16_f8_f16_mk_nk_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f16_f16_mk_kn_mn_comp_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f16_f16_mk_kn_mn_comp_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f16_f16_mk_kn_mn_comp_mnpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f16_f16_mk_kn_mn_comp_mnkpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f16_f16_mk_kn_mn_mem_v1_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f16_f16_mk_kn_mn_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f16_f16_mk_kn_mn_mem_v1_mnkpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f16_f16_mk_kn_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f16_f16_mk_kn_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f16_f16_mk_kn_mn_mem_v2_mnkpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);
void add_device_gemm_xdl_universal_f8_f16_f16_mk_nk_mn_comp_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f16_f16_mk_nk_mn_comp_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f16_f16_mk_nk_mn_mem_v1_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f16_f16_mk_nk_mn_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f16_f16_mk_nk_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f16_f16_mk_nk_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_comp_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_comp_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_comp_mnpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_comp_mnkpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_mem_v1_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_mem_v1_mnkpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_mem_v2_mnkpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_comp_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_comp_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_mem_v1_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);
void add_device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_comp_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_comp_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_comp_mnpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_comp_mnkpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_mem_v1_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_mem_v1_mnkpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_mem_v2_mnkpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_comp_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_comp_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_comp_mpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_comp_mkpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_mem_v1_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_mem_v1_mkpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_mem_v2_mkpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);
#endif
#if(defined(CK_ENABLE_BF16) && defined(CK_ENABLE_FP8))
void add_device_gemm_xdl_universal_f8_f8_bf16_mk_kn_mn_comp_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F8, F8, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f8_bf16_mk_kn_mn_comp_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F8, F8, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f8_bf16_mk_kn_mn_comp_nkpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F8, F8, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f8_bf16_mk_kn_mn_mem_v1_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F8, F8, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f8_bf16_mk_kn_mn_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F8, F8, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f8_bf16_mk_kn_mn_mem_v1_nkpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F8, F8, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f8_bf16_mk_kn_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F8, F8, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f8_bf16_mk_kn_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F8, F8, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f8_bf16_mk_kn_mn_mem_v2_nkpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F8, F8, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_comp_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F8, F8, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_comp_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F8, F8, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_v1_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F8, F8, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F8, F8, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F8, F8, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F8, F8, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);
#endif

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
