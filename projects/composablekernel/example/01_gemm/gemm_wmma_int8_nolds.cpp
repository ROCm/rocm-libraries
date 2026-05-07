// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma.hpp"

using ADataType        = std::uint8_t;
using BDataType        = std::uint8_t;
using AccDataType      = std::int32_t;
using CShuffleDataType = std::int32_t;
using CDataType        = std::uint8_t;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

// clang-format off
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmWmma_CShuffle
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
           1,           // Prefetch stage
           32,          // BlockSize
           16,          // MPerBlock
           16,          // NPerBlock
           16,          // KPerBlock
           4,           // K1
           16,          // MPerWmma
           16,          // NPerWmma
           1,           // M-Repeat // M-PerWmma / M-Repeat = M-Wave
           1,           // N-Repeat // N-PerWmma / N-Repeat = N-Wave
           S<4, 8, 1>,
           S<1, 0, 2>,
           S<1, 0, 2>,
           2,
           4,
           4,
           true,
           S<4, 8, 1>,
           S<1, 0, 2>,
           S<1, 0, 2>,
           2,
           4,
           4,
           true,
           1,           // C shuffle (M Repeat) Per store
           1,           // C shuffle (N Repeat) Per store
           S<1, 8, 1, 4>,
           4>;
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
