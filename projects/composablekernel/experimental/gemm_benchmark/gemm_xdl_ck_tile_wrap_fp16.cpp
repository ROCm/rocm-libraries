// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#include "common.hpp"

#include "gemm_xdl_ck_tile_wrap.hpp"

#if 1
using ADataType       = ck::half_t;
using BDataType       = ck::half_t;
using ComputeDataType = ck::half_t;
#else
using ADataType       = ck::pk_i4_t;
using BDataType       = ck::pk_i4_t;
using ComputeDataType = ck::f8_t;
#endif
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
using CDataType        = ck::half_t;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

using AElementOp = ck::tensor_operation::element_wise::PassThrough;
using BElementOp = ck::tensor_operation::element_wise::PassThrough;
using CElementOp = ck::tensor_operation::element_wise::PassThrough;

using GemmDefault = ck_tile::sequence<false, false, false>; // M/N/K Pad

// clang-format off
using DeviceGemmV2Instance = ck::tensor_operation::device::DeviceGemm_Xdl_CkTileWrap<
    ALayout, BLayout, CLayout,
    ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,
    AElementOp, BElementOp, CElementOp,
    GemmDefault,
    128, 128, 128,                                      // M/N/K PerBlock
    16, 16, ck_tile::get_k_warp_tile<decltype(ck::GetCkTileDataType<ComputeDataType>()), 16>(),  // M/N/K PerXDL
    2, 2, 1,                                         // M/N/K Warp
    2,
    ComputeDataType,
    1, 1,
    ck_tile::GemmPipelineScheduler::Intrawave,
    ck_tile::GemmPipeline::COMPUTE_V3>;
// clang-format on

#include "run_gemm_example_v2.inc"

int main(int argc, char* argv[]) { return !run_gemm_splitk_example(argc, argv); }
