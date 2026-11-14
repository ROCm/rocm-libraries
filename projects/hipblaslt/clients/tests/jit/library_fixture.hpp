// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-code-object.hpp"
#include "hipblaslt-jit-library.hpp"
#include "test_helpers.hpp"
#include <Tensile/Contractions.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <catch2/catch_test_macros.hpp>
#include <hip/hip_runtime_api.h>

// The device target, cache key and GEMM the publish test writes and the
// loader and launch tests read.
namespace hipblaslt_jit_test
{
    inline hipblaslt_jit::Target deviceTarget(const hipDeviceProp_t& properties)
    {
        auto target = hipblaslt_jit::Target::fromTargetId(properties.gcnArchName);
        JIT_REQUIRE("Unknown target ID " + std::string(properties.gcnArchName), target.has_value());
        return *target;
    }

    // The current HIP device as a DeviceTarget, with its TensileLite hardware.
    inline hipblaslt_jit::DeviceTarget currentDeviceTarget()
    {
        hipblaslt_jit::DeviceTarget target;
        JIT_REQUIRE("Cannot select the current HIP device",
                    hipGetDevice(&target.device) == hipSuccess);
        JIT_REQUIRE("Cannot query the current HIP device",
                    hipGetDeviceProperties(&target.properties, target.device) == hipSuccess);
        static_cast<hipblaslt_jit::Target&>(target) = deviceTarget(target.properties);
        target.wavefrontSize                        = target.properties.warpSize;
        target.cuCount                              = target.properties.multiProcessorCount;
        target.hardware = TensileLite::hip::GetDevice(target.properties, target.device);
        REQUIRE((target.hardware != nullptr));
        return target;
    }

    inline hipblaslt_jit::CacheKey pregeneratedKey(const hipDeviceProp_t& properties)
    {
        hipblaslt_jit::CacheKey key;
        key.target            = deviceTarget(properties);
        key.wavefrontSize     = properties.warpSize;
        key.backendId         = "pregenerated";
        key.backendVersion    = 1;
        key.codeObjectVersion = hipblaslt_jit::jitCodeObjectVersion;
        return key;
    }

    // FP16 A, B, C and D with FP32 alpha, beta and accumulation, as the
    // committed kernels' entries expect.
    inline TensileLite::ContractionProblemGemm fp16Gemm(bool transA, size_t K)
    {
        constexpr size_t M = 256, N = 128;
        const auto       half = rocisa::DataType::Half;
        const size_t     lda  = transA ? K : M;
        auto             problem = TensileLite::ContractionProblemGemm::GEMM_Strides(
            transA, false, half, half, half, half, M, N, K, 1, lda, lda * (transA ? M : K), K,
            K * N, M, M * N, M, M * N, 0.5);
        problem.setComputeInputTypeA(half);
        problem.setComputeInputTypeB(half);
        problem.setAlphaType(rocisa::DataType::Float);
        problem.setBetaType(rocisa::DataType::Float);
        problem.setHighPrecisionAccumulate(true);
        problem.setStridedBatched(true);
        problem.setUseDeviceUserArguments(false);
        problem.setAlphaRestriction(TensileLite::toScalarValueEnum(1.25));
        problem.setBetaRestriction(TensileLite::toScalarValueEnum(0.5));
        problem.setCEqualsD(false);
        return problem;
    }
}
