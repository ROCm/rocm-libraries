// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.
#ifdef CK_EXTENSION_MX_TYPE
#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_gfx13.hpp"

using ADataType        = ck::MxType_t<ck::MTX_FMT::MTX_FMT_FP8_E4M3>;
using BDataType        = ck::MxType_t<ck::MTX_FMT::MTX_FMT_FP8_E4M3>;
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
           1,            // Prefetch stage
           256,          // BlockSize
           64,           // MPerBlock
           128,          // NPerBlock
           256,           // KPerBlock
           1,            // K1
           16,           // MPerWmma
           16,           // NPerWmma
           64,           // KPerWmma
           2,            // M-Repeat // M-PerWmma / M-Repeat = M-Wave
           4,            // N-Repeat // N-PerWmma / N-Repeat = N-Wave
           S<32, 4, 1>,  // M-K0-K1
           S<0, 1, 2>,
           S<0, 1, 2>,
           2,
           4,
           4,
           false,
           false,
           false,
           false,
           ck::TensorLoadOption::DEFAULT_LOAD,
           1,
           S<32, 4, 1>,
           S<0, 1, 2>,
           S<0, 1, 2>,
           2,
           4,
           4,
           false,
           false,
           false,
           false,
           ck::TensorLoadOption::DEFAULT_LOAD,
           1,
           1,           // C shuffle (M Repeat) Per store
           1,           // C shuffle (N Repeat) Per store
           S<1, 32, 1, 4>,
           4,
           false,
           true,
           ck::LoopScheduler::Default,
           ck::PipelineVersion::v1>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<typename ADataType::type_t,
                                                                        typename BDataType::type_t,
                                                                        CDataType,
                                                                        AccDataType,
                                                                        AElementOp,
                                                                        BElementOp,
                                                                        CElementOp>;

#include "run_gemm_example_v3.inc"

int main(int argc, char* argv[]) { return !run_gemm_example(argc, argv); }
#endif
