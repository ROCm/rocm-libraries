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
           256,    // BlockSize
           32, //64,     // MPerBlock
           128,          // NPerBlock
           16, //64,     // KPerBlock
           2,            // K1
           16,           // MPerWmma
           16,           // NPerWmma
           16,           // KPerWmma
           1, //2,       // M-Repeat // M-PerWmma / M-Repeat = M-Wave
           4,            // N-Repeat // N-PerWmma / N-Repeat = N-Wave
           S<16, 4, 1>, //S<32, 4, 1>,
           S<0, 1, 2>,
           S<0, 1, 2>,
           2,
#ifdef GEMM_A_DISABLE_LDS
           2,
           2,
#else
           8,
           8,
#endif
           false,
           false,
           false,  // AEnableGlobalTRLoad
           false,  // AEnableGlobalTiledLoad
           ck::TensorLoadOption::WGP_MULTICAST_LOAD,      // ALoadOption: wgp multicast load: 2
           1,
           S<16, 4, 1>, //S<32, 4, 1>,
           S<0, 1, 2>,
           S<0, 1, 2>,
           2,
#ifdef GEMM_B_DISABLE_LDS
           2,
           2,
#else
           8,
           8,
#endif
           false,
           false,
           false, // BEnableGlobalTRLoad
           false, // BEnableGlobalTiledLoad
           ck::TensorLoadOption::DEFAULT_LOAD,     // BLoadOption
           1,
           1,     // C shuffle (M Repeat) Per store
           1,     // C shuffle (N Repeat) Per store
           S<1, 16, 1, 4>, //S<1, 32, 1, 4>,
           2,
           false,
           true, //EnableWaveGroup
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
