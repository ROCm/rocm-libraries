// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"

using ADataType = ck::half_t;
using BDataType = ck::half_t;

using AccDataType      = float;
using CShuffleDataType = float;
using CDataType        = float;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

// clang-format off
using DeviceGemmV2Instance = 
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
        ALayout,   BLayout,  CLayout,   
        ADataType,   BDataType,  CDataType,  AccDataType,  CShuffleDataType, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        64,
        32, 32,
        16, 8, 8,
        16,   16,
        2,    2,
        S<2, 2, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        S<2, 2, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        1, 1, S<1, 16, 1, 4>, 8,
        ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1,
        ADataType, BDataType,
        false, false,
        0, false,
        true>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

#include "run_gemm_example_v2.inc"

int main(int argc, char* argv[]) { return !run_gemm_splitk_example(argc, argv); }
