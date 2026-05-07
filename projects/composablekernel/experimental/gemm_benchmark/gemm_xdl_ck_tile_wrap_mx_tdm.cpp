// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#define CK_TILE_WARP_ENABLE_MX 1
#define CK_TILE_WRAP_ENABLE_BPRESHUFFLE 1
#include "common.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_mx_gemm.hpp"
#include "gemm_xdl_ck_tile_wrap.hpp"

using ADataType        = ck::f8_t;
using BDataType        = ck::f8_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
using CDataType        = ck::half_t;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

using AElementOp = ck::tensor_operation::element_wise::PassThrough;
using BElementOp = ck::tensor_operation::element_wise::PassThrough;
using CElementOp = ck::tensor_operation::element_wise::PassThrough;

using GemmDefault                    = ck_tile::sequence<false, false, false>; // M/N/K Pad
using XDataType                      = ck::e8m0_bexp_t;
using XPackedDataType                = ck::e8m0_bexp_t;
constexpr ck::index_t ScaleBlockSize = 32; // scaling block size
constexpr int KPack                  = 16; // Equal with KThreadChunk

// clang-format off
using DeviceOpInstance = ck::tensor_operation::device::DeviceGemm_Xdl_CkTileWrap<
    ALayout, BLayout, CLayout,
    ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,
    AElementOp, BElementOp, CElementOp,
    GemmDefault,
    128,   128,  128,  32,   32,  ck_tile::get_k_warp_tile<ck_tile::fp8_t, 32>(),  1,   4,     1, 1,
    ADataType,
    1,
    1,
    ck_tile::GemmPipelineScheduler::Intrawave,
    ck_tile::GemmPipeline::PRESHUFFLE_MX_TDM>;
// clang-format on

#include "run_mx_gemm_example_v2.inc"

int main(int argc, char* argv[]) { return !run_mx_gemm_splitk_example<true>(argc, argv); }
