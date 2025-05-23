// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_mx_common.hpp"

using ADataType = ck::bf8_t;
using BDataType = ck::bf8_t;

using XDataType = ck::e8m0_bexp_t;

using CDataType        = ck::bhalf_t;
using AccDataType      = float;
using CShuffleDataType = CDataType;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough; // elementwise transformation for A matrix
using BElementOp = PassThrough; // elementwise transformation for B matrix
using CElementOp = PassThrough; // elementwise transformation for C matrix

constexpr ck::index_t ScaleBlockSize = 32; // scaling block size
constexpr ck::index_t KPerBlock      = 128;

constexpr auto GemmSpec      = ck::tensor_operation::device::GemmSpecialization::Default;
constexpr auto BlkGemmPSched = ck::BlockGemmPipelineScheduler::Intrawave;
constexpr auto BlkGemmPVer   = ck::BlockGemmPipelineVersion::v1;

using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMX_Xdl_CShuffleV3<
    ALayout,          // ALayout
    BLayout,          // BLayout
    CLayout,          // CLayout
    ADataType,        // ADataType
    XDataType,        // AScaleDataType
    BDataType,        // BDataType
    XDataType,        // BScaleDataType
    CDataType,        // CDataType
    AccDataType,      // GemmAccDataType
    CShuffleDataType, // CShuffleDataType
    AElementOp,       // AElementwiseOperation
    BElementOp,       // BElementwiseOperation
    CElementOp,       // CElementwiseOperation
    GemmSpec,         // GemmSpec
    ScaleBlockSize,   // ScaleBlockSize: Scaling block size
    128,              // BlockSize: Thread block size
    128,              // MPerBlock
    16,               // NPerBlock
    KPerBlock,        // KPerBlock
    16,               // AK1
    16,               // BK1
    16,               // MPerXDL
    16,               // NPerXDL
    4,                // MXdlPerWave
    1,                // NXdlPerWave
    S<8, 16, 1>,      // ABlockTransferThreadClusterLengths_AK0_M_AK1
    S<1, 0, 2>,       // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,       // ABlockTransferSrcAccessOrder
    2,                // ABlockTransferSrcVectorDim
    16,               // ABlockTransferSrcScalarPerVector
    16,               // ABlockTransferDstScalarPerVector_AK1
    false,            // ABlockLdsExtraM
    S<8, 16, 1>,      // BBlockTransferThreadClusterLengths_BK0_N_BK1
    S<1, 0, 2>,       // BBlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,       // BBlockTransferSrcAccessOrder
    2,                // BBlockTransferSrcVectorDim
    16,               // BBlockTransferSrcScalarPerVector
    16,               // BBlockTransferDstScalarPerVector_BK1
    false,            // BBlockLdsExtraN
    1,                // CShuffleMXdlPerWavePerShuffle
    1,                // CShuffleNXdlPerWavePerShuffle
    S<1, 16, 1, 8>,   // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    2,                // CShuffleBlockTransferScalarPerVector_NPerBlock
    BlkGemmPSched,    // BlkGemmPipeSched
    BlkGemmPVer,      // BlkGemmPipelineVer
    ADataType,        // ComputeTypeA
    BDataType         // ComputeTypeB
    >;

int main(int argc, char* argv[])
{
    return run_mx_gemm_example<DeviceOpInstance,
                               ADataType,
                               BDataType,
                               XDataType,
                               CDataType,
                               ALayout,
                               BLayout,
                               CLayout,
                               AElementOp,
                               BElementOp,
                               CElementOp,
                               AccDataType,
                               CShuffleDataType,
                               ScaleBlockSize>(argc, argv)
               ? 0
               : -1;
}
