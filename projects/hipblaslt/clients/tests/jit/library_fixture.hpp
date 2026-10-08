// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-code-object.hpp"
#include "hipblaslt-jit-library.hpp"
#include <Tensile/Contractions.hpp>
#include <catch2/catch_test_macros.hpp>
#include <hip/hip_runtime_api.h>

#ifndef _WIN32
extern "C" char** environ;
#endif

// The device target, cache key and GEMM the publish test writes and the
// loader and launch tests read.
namespace hipblaslt_jit_test
{
    inline hipblaslt_jit::Target deviceTarget(const hipDeviceProp_t& properties)
    {
        auto target = hipblaslt_jit::Target::fromTargetId(properties.gcnArchName);
        {
            INFO(("Unknown target ID " + std::string(properties.gcnArchName)));
            REQUIRE((target.has_value()));
        }
        target->asicRevision = properties.asicRevision;
        return *target;
    }

    inline hipblaslt_jit::CacheKey pregeneratedKey(const hipDeviceProp_t& properties)
    {
        hipblaslt_jit::CacheKey key;
        key.target            = deviceTarget(properties);
        key.wavefrontSize     = properties.warpSize;
        key.backendId         = "pregenerated";
        key.backendVersion    = 1;
        key.codeObjectVersion = hipblaslt_jit::jitCodeObjectVersion;
        key.comgr             = hipblaslt_jit::code_object::comgrIdentity();
        key.rocmPath          = hipblaslt_jit::code_object::rocmPath();
#ifndef _WIN32
        key.environment = hipblaslt_jit::compilerEnvironment(environ);
#endif
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
