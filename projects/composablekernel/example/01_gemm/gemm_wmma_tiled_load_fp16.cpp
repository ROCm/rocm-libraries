// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_gfx13.hpp"

using ADataType        = ck::half_t;
using BDataType        = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = float;
using CDataType        = ck::half_t;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

// clang-format off
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmWmma_GFX13
         < ALayout,
           BLayout,
           CLayout,
           ADataType,
           BDataType,
           CDataType,
           AccDataType,
           CShuffleDataType,
           AElementOp,
           BElementOp,
           CElementOp,
           GemmDefault,
           1,      // Prefetch stage
           32, //128,    // BlockSize
           16, //64,     // MPerBlock
           16, //128,    // NPerBlock
           16, //64,     // KPerBlock
           2,            // K1
           16,           // MPerWmma
           16,           // NPerWmma
           16,           // KPerWmma
           1, //2,       // M-Repeat // M-PerWmma / M-Repeat = M-Wave
           1, //4,       // N-Repeat // N-PerWmma / N-Repeat = N-Wave
           S<8, 4, 1>, //S<32, 4, 1>,
           S<0, 1, 2>,
           S<0, 1, 2>,
           2,
           2,
           2,
           false,
           false,
           false,  // AEnableGlobalTRLoad
           true,   // AEnableGlobalTiledLoad
           ck::TensorLoadOption::DEFAULT_LOAD, // ALoadOption
           1,      // A_cluster_size
           S<8, 4, 1>, //S<32, 4, 1>,
           S<0, 1, 2>,
           S<0, 1, 2>,
           2,
           2,
           2,
           false,
           false,
           false, // BEnableGlobalTRLoad
           true,  // BEnableGlobalTiledLoad
           ck::TensorLoadOption::DEFAULT_LOAD, // BLoadOption
           1, // B_cluster_size
           1,     // C shuffle (M Repeat) Per store
           1,     // C shuffle (N Repeat) Per store
           S<1, 8, 1, 4>,//S<1, 32, 1, 4>,
           4,
           false,
           false,
           ck::LoopScheduler::Default,
           ck::PipelineVersion::v1>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

using ReferenceGemmInstanceGPU = ck::tensor_operation::device::ReferenceGemm<ALayout,
                                                                             BLayout,
                                                                             CLayout,
                                                                             ADataType,
                                                                             BDataType,
                                                                             CDataType,
                                                                             AccDataType,
                                                                             AElementOp,
                                                                             BElementOp,
                                                                             CElementOp>;

#include "run_gemm_example.inc"

int main(int argc, char* argv[]) { return !run_gemm_example(argc, argv); }
