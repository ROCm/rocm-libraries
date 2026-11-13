// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The checked-in rocRoller kernels (RR_GEMM_*) are for the build without
// rocRoller, which serves block-scaled GEMMs from them with rocRoller's own
// pick. The build with rocRoller compiles those problems itself, so its Tensile
// paths must select exactly what they select without these kernels: the C API
// with HIPBLASLT_USE_ROCROLLER=0, the hipblaslt_ext Gemm (both setProblem
// forms) and GroupedGemm heuristics, getAllAlgos, and an index lookup followed
// by a support check. On the build without rocRoller the same problems resolve
// to rocRoller's pick and an index lookup is supported.
//
// One problem per checked-in kernel family (A, B and D types), each with the
// kernel rocRoller picks for it on the build with rocRoller. Selection only:
// nothing here launches a kernel.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>

#include "scoped_env.hpp"

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace
{
    using test_env::ScopedEnv;

    constexpr size_t kWorkspace = 128ull << 20;
    constexpr int    kAlgoCount = 32;
    float            kAlpha     = 1.f;
    float            kBeta      = 0.f;

    struct Fixture
    {
        hipDataType a;
        hipDataType b;
        hipDataType d;
        int64_t     m;
        int64_t     n;
        int64_t     k;
        const char* kernel;
    };

    // TN, batch 1, C of the type of D. The kernel is the first candidate of
    // rocRoller's walk for the problem on the build with rocRoller.
    const std::vector<Fixture>& fixtures()
    {
        static const std::vector<Fixture> table = {
            // clang-format off
            {HIP_R_6F_E3M2, HIP_R_6F_E3M2, HIP_R_16BF, 64, 64,   128,  "RR_GEMM_TN_BF6_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_6F_E3M2, HIP_R_6F_E3M2, HIP_R_16F,  64, 64,   128,  "RR_GEMM_TN_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_6F_E3M2, HIP_R_6F_E3M2, HIP_R_32F,  64, 64,   128,  "RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_6F_E3M2, HIP_R_8F_E5M2, HIP_R_16BF, 16, 8192, 1024, "RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2"},
            {HIP_R_6F_E3M2, HIP_R_8F_E5M2, HIP_R_16F,  16, 8192, 1024, "RR_GEMM_TN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2"},
            {HIP_R_6F_E3M2, HIP_R_8F_E5M2, HIP_R_32F,  16, 8192, 1024, "RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2"},
            {HIP_R_6F_E3M2, HIP_R_4F_E2M1, HIP_R_16BF, 64, 64,   128,  "RR_GEMM_TN_BF6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_6F_E3M2, HIP_R_4F_E2M1, HIP_R_16F,  64, 64,   128,  "RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_6F_E3M2, HIP_R_4F_E2M1, HIP_R_32F,  64, 64,   128,  "RR_GEMM_TN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_6F_E3M2, HIP_R_6F_E2M3, HIP_R_16BF, 64, 64,   128,  "RR_GEMM_TN_BF6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_6F_E3M2, HIP_R_6F_E2M3, HIP_R_16F,  64, 64,   128,  "RR_GEMM_TN_BF6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_6F_E3M2, HIP_R_6F_E2M3, HIP_R_32F,  64, 64,   128,  "RR_GEMM_TN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_6F_E3M2, HIP_R_8F_E4M3, HIP_R_16BF, 16, 8192, 1024, "RR_GEMM_TN_BF6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2"},
            {HIP_R_6F_E3M2, HIP_R_8F_E4M3, HIP_R_16F,  16, 8192, 1024, "RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2"},
            {HIP_R_6F_E3M2, HIP_R_8F_E4M3, HIP_R_32F,  16, 8192, 1024, "RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2"},
            {HIP_R_8F_E5M2, HIP_R_6F_E3M2, HIP_R_16BF, 64, 64,   128,  "RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E5M2, HIP_R_6F_E3M2, HIP_R_16F,  64, 64,   128,  "RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E5M2, HIP_R_6F_E3M2, HIP_R_32F,  64, 64,   128,  "RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E5M2, HIP_R_8F_E5M2, HIP_R_16BF, 64, 64,   128,  "RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E5M2, HIP_R_8F_E5M2, HIP_R_16F,  64, 64,   128,  "RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E5M2, HIP_R_8F_E5M2, HIP_R_32F,  64, 64,   128,  "RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E5M2, HIP_R_4F_E2M1, HIP_R_16BF, 64, 64,   128,  "RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E5M2, HIP_R_4F_E2M1, HIP_R_16F,  64, 64,   128,  "RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E5M2, HIP_R_4F_E2M1, HIP_R_32F,  64, 64,   128,  "RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E5M2, HIP_R_6F_E2M3, HIP_R_16BF, 64, 64,   128,  "RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E5M2, HIP_R_6F_E2M3, HIP_R_16F,  64, 64,   128,  "RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E5M2, HIP_R_6F_E2M3, HIP_R_32F,  64, 64,   128,  "RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E5M2, HIP_R_8F_E4M3, HIP_R_16BF, 64, 64,   128,  "RR_GEMM_TN_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E5M2, HIP_R_8F_E4M3, HIP_R_16F,  64, 64,   128,  "RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E5M2, HIP_R_8F_E4M3, HIP_R_32F,  64, 64,   128,  "RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_4F_E2M1, HIP_R_6F_E3M2, HIP_R_16BF, 64, 64,   128,  "RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_4F_E2M1, HIP_R_6F_E3M2, HIP_R_16F,  64, 64,   128,  "RR_GEMM_TN_FP4_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_4F_E2M1, HIP_R_6F_E3M2, HIP_R_32F,  64, 64,   128,  "RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_4F_E2M1, HIP_R_8F_E5M2, HIP_R_16BF, 64, 64,   128,  "RR_GEMM_TN_FP4_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_4F_E2M1, HIP_R_8F_E5M2, HIP_R_16F,  64, 64,   128,  "RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_4F_E2M1, HIP_R_8F_E5M2, HIP_R_32F,  64, 64,   128,  "RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_4F_E2M1, HIP_R_4F_E2M1, HIP_R_16BF, 64, 64,   128,  "RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_4F_E2M1, HIP_R_4F_E2M1, HIP_R_16F,  64, 64,   128,  "RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_4F_E2M1, HIP_R_4F_E2M1, HIP_R_32F,  64, 64,   128,  "RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_4F_E2M1, HIP_R_6F_E2M3, HIP_R_16BF, 64, 64,   128,  "RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_4F_E2M1, HIP_R_6F_E2M3, HIP_R_16F,  64, 64,   128,  "RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_4F_E2M1, HIP_R_6F_E2M3, HIP_R_32F,  64, 64,   128,  "RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_4F_E2M1, HIP_R_8F_E4M3, HIP_R_16BF, 64, 64,   128,  "RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_4F_E2M1, HIP_R_8F_E4M3, HIP_R_16F,  64, 64,   128,  "RR_GEMM_TN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_4F_E2M1, HIP_R_8F_E4M3, HIP_R_32F,  64, 64,   128,  "RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_6F_E2M3, HIP_R_6F_E3M2, HIP_R_16BF, 64, 64,   128,  "RR_GEMM_TN_FP6_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_6F_E2M3, HIP_R_6F_E3M2, HIP_R_16F,  64, 64,   128,  "RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_6F_E2M3, HIP_R_6F_E3M2, HIP_R_32F,  64, 64,   128,  "RR_GEMM_TN_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_6F_E2M3, HIP_R_8F_E5M2, HIP_R_16BF, 16, 8192, 1024, "RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2"},
            {HIP_R_6F_E2M3, HIP_R_8F_E5M2, HIP_R_16F,  16, 8192, 1024, "RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2"},
            {HIP_R_6F_E2M3, HIP_R_8F_E5M2, HIP_R_32F,  16, 8192, 1024, "RR_GEMM_TN_FP6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2"},
            {HIP_R_6F_E2M3, HIP_R_4F_E2M1, HIP_R_16BF, 64, 64,   128,  "RR_GEMM_TN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_6F_E2M3, HIP_R_4F_E2M1, HIP_R_16F,  64, 64,   128,  "RR_GEMM_TN_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_6F_E2M3, HIP_R_4F_E2M1, HIP_R_32F,  64, 64,   128,  "RR_GEMM_TN_FP6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_6F_E2M3, HIP_R_6F_E2M3, HIP_R_16BF, 64, 64,   128,  "RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_6F_E2M3, HIP_R_6F_E2M3, HIP_R_16F,  64, 64,   128,  "RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_6F_E2M3, HIP_R_6F_E2M3, HIP_R_32F,  64, 64,   128,  "RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_6F_E2M3, HIP_R_8F_E4M3, HIP_R_16BF, 16, 8192, 1024, "RR_GEMM_TN_FP6_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2"},
            {HIP_R_6F_E2M3, HIP_R_8F_E4M3, HIP_R_16F,  16, 8192, 1024, "RR_GEMM_TN_FP6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2"},
            {HIP_R_6F_E2M3, HIP_R_8F_E4M3, HIP_R_32F,  16, 8192, 1024, "RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2"},
            {HIP_R_8F_E4M3, HIP_R_6F_E3M2, HIP_R_16BF, 64, 64,   128,  "RR_GEMM_TN_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E4M3, HIP_R_6F_E3M2, HIP_R_16F,  64, 64,   128,  "RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E4M3, HIP_R_6F_E3M2, HIP_R_32F,  64, 64,   128,  "RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E4M3, HIP_R_8F_E5M2, HIP_R_16BF, 64, 64,   128,  "RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E4M3, HIP_R_8F_E5M2, HIP_R_16F,  64, 64,   128,  "RR_GEMM_TN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E4M3, HIP_R_8F_E5M2, HIP_R_32F,  64, 64,   128,  "RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E4M3, HIP_R_4F_E2M1, HIP_R_16BF, 64, 64,   128,  "RR_GEMM_TN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E4M3, HIP_R_4F_E2M1, HIP_R_16F,  64, 64,   128,  "RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E4M3, HIP_R_4F_E2M1, HIP_R_32F,  64, 64,   128,  "RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E4M3, HIP_R_6F_E2M3, HIP_R_16BF, 64, 64,   128,  "RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E4M3, HIP_R_6F_E2M3, HIP_R_16F,  64, 64,   128,  "RR_GEMM_TN_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E4M3, HIP_R_6F_E2M3, HIP_R_32F,  64, 64,   128,  "RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E4M3, HIP_R_8F_E4M3, HIP_R_16BF, 64, 64,   128,  "RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E4M3, HIP_R_8F_E4M3, HIP_R_16F,  64, 64,   128,  "RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            {HIP_R_8F_E4M3, HIP_R_8F_E4M3, HIP_R_32F,  64, 64,   128,  "RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"},
            // clang-format on
        };
        return table;
    }

    bool isCheckedIn(const std::string& kernelName)
    {
        return kernelName.rfind("RR_GEMM_", 0) == 0;
    }

    bool archIsGfx950()
    {
        int             device = 0;
        hipDeviceProp_t prop{};
        if(hipGetDevice(&device) != hipSuccess
           || hipGetDeviceProperties(&prop, device) != hipSuccess)
            return false;
        return std::string(prop.gcnArchName).find("gfx950") != std::string::npos;
    }

    struct OwnedHandle
    {
        hipblasLtHandle_t handle = nullptr;
        ~OwnedHandle()
        {
            if(handle)
                hipblasLtDestroy(handle);
        }
    };

    // The descriptors and buffers of one fixture, with packed E8M0 scales on A
    // and B. The buffers are only ever passed to selection.
    struct Problem
    {
        const Fixture&              f;
        hipblasLtMatmulDesc_t       desc   = nullptr;
        hipblasLtMatrixLayout_t     a      = nullptr;
        hipblasLtMatrixLayout_t     b      = nullptr;
        hipblasLtMatrixLayout_t     d      = nullptr;
        hipblasLtMatmulPreference_t pref   = nullptr;
        void*                       buffer = nullptr;
        void*                       scale  = nullptr;

        explicit Problem(const Fixture& fixture)
            : f(fixture)
        {
        }

        ~Problem()
        {
            if(pref)
                hipblasLtMatmulPreferenceDestroy(pref);
            if(desc)
                hipblasLtMatmulDescDestroy(desc);
            for(auto layout : {a, b, d})
                if(layout)
                    hipblasLtMatrixLayoutDestroy(layout);
            for(void* p : {buffer, scale})
                if(p)
                    static_cast<void>(hipFree(p));
        }
    };

    void makeProblem(Problem& q)
    {
        const Fixture& f = q.f;
        ASSERT_EQ(hipblasLtMatmulDescCreate(&q.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);
        hipblasOperation_t opA = HIPBLAS_OP_T;
        hipblasOperation_t opB = HIPBLAS_OP_N;
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      q.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      q.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)),
                  HIPBLAS_STATUS_SUCCESS);

        ASSERT_EQ(hipMalloc(&q.buffer, 256), hipSuccess);
        ASSERT_EQ(hipMalloc(&q.scale, 256), hipSuccess);
        hipblasLtMatmulMatrixScale_t mode = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        for(auto [pointer, scaleMode] :
            {std::pair{HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE},
             std::pair{HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE}})
        {
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(q.desc, pointer, &q.scale, sizeof(void*)),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(q.desc, scaleMode, &mode, sizeof(mode)),
                      HIPBLAS_STATUS_SUCCESS);
        }

        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&q.a, f.a, f.k, f.m, f.k), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&q.b, f.b, f.k, f.n, f.k), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&q.d, f.d, f.m, f.n, f.m), HIPBLAS_STATUS_SUCCESS);

        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&q.pref), HIPBLAS_STATUS_SUCCESS);
        uint64_t workspace = kWorkspace;
        ASSERT_EQ(
            hipblasLtMatmulPreferenceSetAttribute(
                q.pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace)),
            HIPBLAS_STATUS_SUCCESS);
    }

    std::vector<std::string> names(hipblasLtHandle_t                                    handle,
                                   const std::vector<hipblasLtMatmulHeuristicResult_t>& results)
    {
        std::vector<std::string> out;
        for(auto result : results)
            out.push_back(hipblaslt_ext::getKernelNameFromAlgo(handle, result.algo));
        return out;
    }

    void expectNoneCheckedIn(const std::vector<std::string>& kernelNames)
    {
        for(const auto& name : kernelNames)
            EXPECT_FALSE(isCheckedIn(name)) << name;
    }

    std::vector<hipblasLtMatmulHeuristicResult_t>
        heuristic(hipblasLtHandle_t handle, Problem& q, int count)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> raw(count);
        int                                           returned = 0;
        // No algorithm is a status of its own, which is an answer here.
        static_cast<void>(hipblasLtMatmulAlgoGetHeuristic(
            handle, q.desc, q.a, q.b, q.d, q.d, q.pref, count, raw.data(), &returned));
        std::vector<hipblasLtMatmulHeuristicResult_t> out;
        for(int i = 0; i < returned; ++i)
            if(raw[i].state == HIPBLAS_STATUS_SUCCESS)
                out.push_back(raw[i]);
        return out;
    }

    // hipblaslt_ext::isSolutionSupported checks nothing for an algo whose first
    // byte is 0, so callers test only the algos it does check.
    bool cCheckable(const hipblasLtMatmulAlgo_t& algo)
    {
        return algo.data[0] != 0;
    }

    hipblasStatus_t
        cSupported(hipblasLtHandle_t handle, Problem& q, const hipblasLtMatmulAlgo_t& algo)
    {
        hipblasLtMatmulHeuristicResult_t result{};
        result.algo      = algo;
        size_t workspace = 0;
        int    count     = 0;
        return hipblaslt_ext::isSolutionSupported(
            &result, handle, q.desc, &kAlpha, q.a, q.b, &kBeta, q.d, q.d, &workspace, &count);
    }

    hipblaslt_ext::Gemm extGemm(hipblasLtHandle_t handle, const Fixture& f)
    {
        return hipblaslt_ext::Gemm(
            handle, HIPBLAS_OP_T, HIPBLAS_OP_N, f.a, f.b, f.d, f.d, HIPBLAS_COMPUTE_32F);
    }

    // hipblaslt-bench --api_method mix: the problem comes from the C descriptors.
    hipblasStatus_t setMixProblem(hipblaslt_ext::Gemm& gemm, Problem& q)
    {
        return gemm.setProblem(
            q.desc, &kAlpha, q.buffer, q.a, q.buffer, q.b, &kBeta, q.buffer, q.d, q.buffer, q.d);
    }

    // hipblaslt-bench --api_method cpp: sizes, epilogue, inputs and problem type.
    hipblasStatus_t setCppProblem(hipblaslt_ext::Gemm& gemm, Problem& q)
    {
        const Fixture&                 f = q.f;
        hipblaslt_ext::GemmProblemType type;
        type.setOpA(HIPBLAS_OP_T);
        type.setOpB(HIPBLAS_OP_N);
        type.setTypeA(f.a);
        type.setTypeB(f.b);
        type.setTypeC(f.d);
        type.setTypeD(f.d);
        type.setTypeCompute(HIPBLAS_COMPUTE_32F);
        hipblaslt_ext::GemmEpilogue epilogue;
        epilogue.setMode(HIPBLASLT_EPILOGUE_DEFAULT);
        epilogue.setScalingAType(HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
        epilogue.setScalingBType(HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
        hipblaslt_ext::GemmInputs inputs;
        inputs.setA(q.buffer);
        inputs.setB(q.buffer);
        inputs.setC(q.buffer);
        inputs.setD(q.buffer);
        inputs.setAlpha(&kAlpha);
        inputs.setBeta(&kBeta);
        inputs.setScaleA(q.scale);
        inputs.setScaleB(q.scale);
        return gemm.setProblem(f.m,
                               f.n,
                               f.k,
                               1,
                               f.k,
                               f.k,
                               f.m,
                               f.m,
                               f.k * f.m,
                               f.k * f.n,
                               f.m * f.n,
                               f.m * f.n,
                               epilogue,
                               inputs,
                               type);
    }

    std::vector<hipblasLtMatmulHeuristicResult_t> extHeuristic(hipblaslt_ext::GemmInstance& gemm,
                                                               int                          count)
    {
        hipblaslt_ext::GemmPreference pref;
        pref.setMaxWorkspaceBytes(kWorkspace);
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        static_cast<void>(gemm.algoGetHeuristic(count, pref, results));
        return results;
    }

    using ContractionLibrary
        = TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>;

    // The lazy master library the runtime loads: HIPBLASLT_TENSILE_LIBPATH,
    // then the build and install layouts next to the loaded hipBLASLt.
    std::shared_ptr<ContractionLibrary> masterLibrary(std::string& reason)
    {
        static std::string                         failure;
        static std::shared_ptr<ContractionLibrary> library = []() {
            const std::string                  file = "TensileLibrary_lazy_gfx950.dat";
            std::vector<std::filesystem::path> dirs;
            if(const char* env = std::getenv("HIPBLASLT_TENSILE_LIBPATH"))
            {
                dirs.emplace_back(env);
                dirs.emplace_back(std::filesystem::path(env) / "gfx950");
            }
#ifndef _WIN32
            Dl_info info{};
            void*   entry = dlsym(RTLD_DEFAULT, "hipblasLtCreate");
            if(entry != nullptr && dladdr(entry, &info) != 0 && info.dli_fname != nullptr)
            {
                const auto libDir = std::filesystem::weakly_canonical(info.dli_fname).parent_path();
                for(const auto& root : {libDir / "hipblaslt" / "library",
                                        libDir.parent_path() / "Tensile" / "library"})
                    dirs.push_back(root / "gfx950");
            }
#endif
            std::string tried;
            for(const auto& dir : dirs)
            {
                const auto logical = (dir / file).string();
                tried += (tried.empty() ? "" : ", ") + logical;
                if(!std::filesystem::exists(logical) && !std::filesystem::exists(logical + ".zlib"))
                    continue;
                auto master = std::dynamic_pointer_cast<ContractionLibrary>(
                    TensileLite::LoadLibraryFile<TensileLite::ContractionProblemGemm>(logical));
                if(master && master->initLibraryMapping(logical))
                    return master;
            }
            failure = "no gfx950 Tensile library found; tried " + tried;
            return std::shared_ptr<ContractionLibrary>();
        }();
        reason = failure;
        return library;
    }

    // Kernel name to one solution index, for every checked-in kernel. Only the
    // block-scaled shards are parsed, which is where these kernels live.
    const std::map<std::string, int>& checkedInIndexes(std::string& reason)
    {
        static std::string                failure;
        static std::map<std::string, int> indexes = []() {
            std::map<std::string, int> out;
            auto                       library = masterLibrary(failure);
            if(!library)
                return out;
            for(const auto& [start, prefix] : library->libraryMapping)
                if(prefix.find("_MXAE8B32_MXBE8B32_") != std::string::npos)
                    library->loadLibrary(start);
            auto record = [&](int index, const auto& solution) {
                if(solution && isCheckedIn(solution->kernelName))
                    out.emplace(solution->kernelName, index);
            };
            for(const auto& [index, solution] : library->solutions)
                record(index, solution);
            for(const auto& [prefix, cache] : library->solutionSources)
                if(cache)
                    for(int index : cache->indices())
                        record(index, cache->get(index));
            if(out.empty())
                failure = "the block-scaled shards hold no checked-in rocRoller kernel";
            return out;
        }();
        reason = failure;
        return indexes;
    }

    class RocRollerCheckedInKernels : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if(!archIsGfx950())
                GTEST_SKIP() << "the checked-in rocRoller kernels are gfx950 kernels";
        }
    };

#ifdef HIPBLASLT_USE_ROCROLLER

    TEST_F(RocRollerCheckedInKernels, smoke_CApiWithRocRollerOffNeverSelectsThem)
    {
        ScopedEnv   env("HIPBLASLT_USE_ROCROLLER", "0");
        OwnedHandle owner;
        ASSERT_EQ(hipblasLtCreate(&owner.handle), HIPBLAS_STATUS_SUCCESS);
        for(const auto& f : fixtures())
        {
            SCOPED_TRACE(f.kernel);
            Problem q(f);
            makeProblem(q);
            if(HasFatalFailure())
                return;
            for(int count : {1, kAlgoCount})
                expectNoneCheckedIn(names(owner.handle, heuristic(owner.handle, q, count)));
        }
    }

    TEST_F(RocRollerCheckedInKernels, smoke_ExtApisNeverSelectThem)
    {
        ScopedEnv   env("HIPBLASLT_USE_ROCROLLER", nullptr);
        OwnedHandle owner;
        ASSERT_EQ(hipblasLtCreate(&owner.handle), HIPBLAS_STATUS_SUCCESS);
        int groupedQueried = 0;
        for(const auto& f : fixtures())
        {
            SCOPED_TRACE(f.kernel);
            Problem q(f);
            makeProblem(q);
            if(HasFatalFailure())
                return;

            auto mix = extGemm(owner.handle, f);
            if(setMixProblem(mix, q) == HIPBLAS_STATUS_SUCCESS)
                expectNoneCheckedIn(names(owner.handle, extHeuristic(mix, kAlgoCount)));
            auto cpp = extGemm(owner.handle, f);
            if(setCppProblem(cpp, q) == HIPBLAS_STATUS_SUCCESS)
                expectNoneCheckedIn(names(owner.handle, extHeuristic(cpp, kAlgoCount)));

            hipblaslt_ext::GroupedGemm grouped(
                owner.handle, HIPBLAS_OP_T, HIPBLAS_OP_N, f.a, f.b, f.d, f.d, HIPBLAS_COMPUTE_32F);
            std::vector<hipblasLtMatmulDesc_t>   desc(2, q.desc);
            std::vector<void*>                   alpha(2, &kAlpha), beta(2, &kBeta);
            std::vector<void*>                   buffers(2, q.buffer);
            std::vector<hipblasLtMatrixLayout_t> layA(2, q.a), layB(2, q.b), layD(2, q.d);
            // Throws for a rocblaslt status with no hipBLAS equivalent; refused
            // either way.
            hipblasStatus_t status;
            try
            {
                status = grouped.setProblem(
                    desc, alpha, buffers, layA, buffers, layB, beta, buffers, layD, buffers, layD);
            }
            catch(hipblasStatus_t thrown)
            {
                status = thrown;
            }
            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                ++groupedQueried;
                expectNoneCheckedIn(names(owner.handle, extHeuristic(grouped, kAlgoCount)));
            }
        }
        RecordProperty("grouped_queried", groupedQueried);
    }

    TEST_F(RocRollerCheckedInKernels, smoke_GetAllAlgosListsNoneOfThem)
    {
        OwnedHandle owner;
        ASSERT_EQ(hipblasLtCreate(&owner.handle), HIPBLAS_STATUS_SUCCESS);
        for(const auto& f : fixtures())
        {
            SCOPED_TRACE(f.kernel);
            std::vector<hipblasLtMatmulHeuristicResult_t> all;
            static_cast<void>(hipblaslt_ext::getAllAlgos(owner.handle,
                                                         hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                                         HIPBLAS_OP_T,
                                                         HIPBLAS_OP_N,
                                                         f.a,
                                                         f.b,
                                                         f.d,
                                                         f.d,
                                                         HIPBLAS_COMPUTE_32F,
                                                         all));
            expectNoneCheckedIn(names(owner.handle, all));
        }
    }

    // To a lookup the index does not exist, as without these kernels. An algo
    // made from it by hand is refused by every support check.
    TEST_F(RocRollerCheckedInKernels, smoke_IndexLookupIsRefused)
    {
        std::string reason;
        const auto& indexes = checkedInIndexes(reason);
        ASSERT_FALSE(indexes.empty()) << reason;
        OwnedHandle owner;
        ASSERT_EQ(hipblasLtCreate(&owner.handle), HIPBLAS_STATUS_SUCCESS);
        for(const auto& f : fixtures())
        {
            SCOPED_TRACE(f.kernel);
            auto it = indexes.find(f.kernel);
            ASSERT_NE(it, indexes.end()) << "not in the loaded library";
            Problem q(f);
            makeProblem(q);
            if(HasFatalFailure())
                return;
            auto cpp = extGemm(owner.handle, f);
            ASSERT_EQ(setCppProblem(cpp, q), HIPBLAS_STATUS_SUCCESS);

            std::vector<hipblasLtMatmulHeuristicResult_t> results;
            std::vector<int>                              index{it->second};
            EXPECT_NE(hipblaslt_ext::getAlgosFromIndex(owner.handle, index, results),
                      HIPBLAS_STATUS_SUCCESS);
            EXPECT_TRUE(results.empty());

            hipblasLtMatmulAlgo_t algo;
            std::memcpy(algo.data, &it->second, sizeof(int));
            algo.max_workspace_bytes = kWorkspace;
            size_t workspace         = 0;
            EXPECT_NE(cpp.isAlgoSupported(algo, workspace), HIPBLAS_STATUS_SUCCESS);
            if(!cCheckable(algo))
                continue;
            EXPECT_NE(cSupported(owner.handle, q, algo), HIPBLAS_STATUS_SUCCESS);
            ScopedEnv env("HIPBLASLT_USE_ROCROLLER", "0");
            EXPECT_NE(cSupported(owner.handle, q, algo), HIPBLAS_STATUS_SUCCESS);
        }
    }

#else

    // rocRoller's pick on the build with rocRoller is what this build selects.
    TEST_F(RocRollerCheckedInKernels, smoke_SelectsRocRollersPick)
    {
        ScopedEnv   env("HIPBLASLT_USE_ROCROLLER", nullptr);
        OwnedHandle owner;
        ASSERT_EQ(hipblasLtCreate(&owner.handle), HIPBLAS_STATUS_SUCCESS);
        for(const auto& f : fixtures())
        {
            SCOPED_TRACE(f.kernel);
            Problem q(f);
            makeProblem(q);
            if(HasFatalFailure())
                return;
            auto results = heuristic(owner.handle, q, 1);
            ASSERT_EQ(results.size(), 1u);
            EXPECT_EQ(hipblaslt_ext::getKernelNameFromAlgo(owner.handle, results[0].algo),
                      f.kernel);
            if(cCheckable(results[0].algo))
                EXPECT_EQ(cSupported(owner.handle, q, results[0].algo), HIPBLAS_STATUS_SUCCESS);

            auto mix = extGemm(owner.handle, f);
            ASSERT_EQ(setMixProblem(mix, q), HIPBLAS_STATUS_SUCCESS);
            auto extResults = extHeuristic(mix, 1);
            ASSERT_EQ(extResults.size(), 1u);
            EXPECT_EQ(hipblaslt_ext::getKernelNameFromAlgo(owner.handle, extResults[0].algo),
                      f.kernel);
        }
    }

    TEST_F(RocRollerCheckedInKernels, smoke_IndexLookupIsSupported)
    {
        std::string reason;
        const auto& indexes = checkedInIndexes(reason);
        ASSERT_FALSE(indexes.empty()) << reason;
        OwnedHandle owner;
        ASSERT_EQ(hipblasLtCreate(&owner.handle), HIPBLAS_STATUS_SUCCESS);
        for(const auto& f : fixtures())
        {
            SCOPED_TRACE(f.kernel);
            auto it = indexes.find(f.kernel);
            ASSERT_NE(it, indexes.end()) << "not in the loaded library";
            Problem q(f);
            makeProblem(q);
            if(HasFatalFailure())
                return;
            auto cpp = extGemm(owner.handle, f);
            ASSERT_EQ(setCppProblem(cpp, q), HIPBLAS_STATUS_SUCCESS);

            std::vector<hipblasLtMatmulHeuristicResult_t> results;
            std::vector<int>                              index{it->second};
            ASSERT_EQ(hipblaslt_ext::getAlgosFromIndex(owner.handle, index, results),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(results.size(), 1u);
            EXPECT_EQ(hipblaslt_ext::getKernelNameFromAlgo(owner.handle, results[0].algo),
                      f.kernel);
            size_t workspace = 0;
            EXPECT_EQ(cpp.isAlgoSupported(results[0].algo, workspace), HIPBLAS_STATUS_SUCCESS);
            if(cCheckable(results[0].algo))
                EXPECT_EQ(cSupported(owner.handle, q, results[0].algo), HIPBLAS_STATUS_SUCCESS);
        }
    }

#endif

    // A kernel develop ships for block-scaled FP8, which the term must leave
    // alone on both builds: scale 1001 with HIPBLASLT_USE_ROCROLLER=0 is served
    // by Tensile on either.
    TEST_F(RocRollerCheckedInKernels, smoke_DevelopKernelsStaySelectable)
    {
        ScopedEnv   env("HIPBLASLT_USE_ROCROLLER", "0");
        OwnedHandle owner;
        ASSERT_EQ(hipblasLtCreate(&owner.handle), HIPBLAS_STATUS_SUCCESS);
        const Fixture fp8{HIP_R_8F_E4M3, HIP_R_8F_E4M3, HIP_R_32F, 128, 128, 256, ""};
        Problem       q(fp8);
        makeProblem(q);
        if(HasFatalFailure())
            return;
        // The scale buffers of the 32x8 pre-swizzle of this shape.
        void* scale = nullptr;
        ASSERT_EQ(hipMalloc(&scale, 1u << 20), hipSuccess);
        hipblasLtMatmulMatrixScale_t mode = HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT;
        for(auto [pointer, scaleMode] :
            {std::pair{HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE},
             std::pair{HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE}})
        {
            EXPECT_EQ(hipblasLtMatmulDescSetAttribute(q.desc, pointer, &scale, sizeof(void*)),
                      HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipblasLtMatmulDescSetAttribute(q.desc, scaleMode, &mode, sizeof(mode)),
                      HIPBLAS_STATUS_SUCCESS);
        }
        auto results = heuristic(owner.handle, q, kAlgoCount);
        EXPECT_FALSE(results.empty());
        for(auto result : results)
        {
            SCOPED_TRACE(hipblaslt_ext::getKernelNameFromAlgo(owner.handle, result.algo));
            EXPECT_FALSE(
                isCheckedIn(hipblaslt_ext::getKernelNameFromAlgo(owner.handle, result.algo)));
            if(cCheckable(result.algo))
                EXPECT_EQ(cSupported(owner.handle, q, result.algo), HIPBLAS_STATUS_SUCCESS);
        }
        static_cast<void>(hipFree(scale));
    }
}
