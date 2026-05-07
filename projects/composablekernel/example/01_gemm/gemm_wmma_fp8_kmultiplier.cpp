// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_gfx13.hpp"

#ifdef A_ENABLE_BF8
using ADataType = ck::bf8_t;
#else
using ADataType = ck::f8_t;
#endif

#ifdef B_ENABLE_BF8
using BDataType = ck::bf8_t;
#else
using BDataType = ck::f8_t;
#endif

using AccDataType      = float; // ck::half_t;
using CShuffleDataType = float;
using CDataType        = ck::f8_t;

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
           1,            // Prefetch stage
           128,          // BlockSize
           64,           // MPerBlock
           128,          // NPerBlock
#ifdef GEMM_KMULTIPLIER_8
           128,          // KPerBlock
#else
           64,           // KPerBlock
#endif
           4,            // K1
           16,           // MPerWmma
           16,           // NPerWmma
#ifdef GEMM_KMULTIPLIER_8
           128,          // KPerWmma
#else
           32,           // KPerWmma
#endif
           2,            // M-Repeat // M-PerWmma / M-Repeat = M-Wave
           4,            // N-Repeat // N-PerWmma / N-Repeat = N-Wave
           S<32, 4, 1>,  // M-K0-K1
           S<0, 1, 2>,
           S<0, 1, 2>,
           2,
           8,
           8,
           false,
           true,
           false,
           false,
           ck::TensorLoadOption::DEFAULT_LOAD,
           1,
           S<32, 4, 1>,
           S<0, 1, 2>,
           S<0, 1, 2>,
           2,
           8,
           8,
           false,
           true,
           false,
           false,
           ck::TensorLoadOption::DEFAULT_LOAD,
           1,
           1,           // C shuffle (M Repeat) Per store
           1,           // C shuffle (N Repeat) Per store
           S<1, 32, 1, 4>,
           4,
           false,
           false,
           ck::LoopScheduler::Default,
           ck::PipelineVersion::v5>;
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
