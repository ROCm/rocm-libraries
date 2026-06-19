// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>

#include <hip/hip_runtime.h>

#include "MatmulUtils.hpp"

namespace test_mx_matmul_common
{

// The current device's gcnArchName, or an empty string if it cannot be queried.
inline std::string currentDeviceArchName()
{
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, 0) != hipSuccess)
    {
        return {};
    }
    return {props.gcnArchName};
}

// MX block-scaled GEMM (FP8/FP6/FP4) is supported only on gfx950 and gfx1250.
inline bool isMxSupportedArch(const std::string& archName)
{
    return archName.rfind("gfx950", 0) == 0 || archName.rfind("gfx125", 0) == 0;
}

/// MX-specific GEMM shapes. A is [M, K] transposed (col-major, opA=T) and B is
/// [K, N] non-transposed (row-major, opB=N); all shapes satisfy the VEC32_UE8M0
/// constraints enforced by isApplicable: m%16==0, n%16==0, K%128==0, batch==1.
inline std::vector<test_matmul_common::MatmulTestCase> getMxMatmulTestCases()
{
    const unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();
    return {
        // 2D — small square
        {{32, 128}, {128, 32}, true, false, seed},
        // 2D — rectangular, wide K
        {{64, 256}, {256, 64}, true, false, seed},
        // 2D — rectangular, wide N
        {{32, 128}, {128, 128}, true, false, seed},
        // 2D — larger
        {{128, 128}, {128, 128}, true, false, seed},
        // 3D — single (unit) batch
        {{1, 32, 128}, {1, 128, 32}, true, false, seed},
        // 4D — single (unit) batch
        {{1, 1, 32, 128}, {1, 1, 128, 32}, true, false, seed},
    };
}

} // namespace test_mx_matmul_common
