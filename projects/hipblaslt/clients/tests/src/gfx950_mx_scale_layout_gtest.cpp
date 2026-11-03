// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// gfx950, HIPBLASLT_USE_ROCROLLER=0. The Tensile path in tensile_host.cpp
// keeps HostPreSwizzle kernels (name token MXSFHPS) for scale 1001 and drops
// them for packed scale 3. This runs hipblasLtMatmulAlgoGetHeuristic,
// hipblaslt_ext::isSolutionSupported, and hipblasLtMatmul, so removing those
// checks in getBestSolutions, getAllSolutions, isSolutionSupported, or
// runContractionProblem fails this test.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <cstdlib>
#include <string>
#include <vector>

namespace
{
    constexpr int64_t kM = 128;
    constexpr int64_t kN = 128;
    constexpr int64_t kK = 256;

    bool archIsGfx950()
    {
        hipDeviceProp_t prop{};
        if(hipGetDeviceProperties(&prop, 0) != hipSuccess)
            return false;
        return std::string(prop.gcnArchName).find("gfx950") != std::string::npos;
    }

    struct EnvGuard
    {
        bool        had = false;
        std::string old;

        explicit EnvGuard(const char* value)
        {
            if(const char* cur = std::getenv("HIPBLASLT_USE_ROCROLLER"))
            {
                had = true;
                old = cur;
            }
            setenv("HIPBLASLT_USE_ROCROLLER", value, 1);
        }

        ~EnvGuard()
        {
            if(had)
                setenv("HIPBLASLT_USE_ROCROLLER", old.c_str(), 1);
            else
                unsetenv("HIPBLASLT_USE_ROCROLLER");
        }
    };

    struct Query
    {
        hipblasLtHandle_t           handle = nullptr;
        hipblasLtMatmulDesc_t       desc   = nullptr;
        hipblasLtMatrixLayout_t     a      = nullptr;
        hipblasLtMatrixLayout_t     b      = nullptr;
        hipblasLtMatrixLayout_t     d      = nullptr;
        hipblasLtMatmulPreference_t pref   = nullptr;
        void*                       scaleA = nullptr;
        void*                       scaleB = nullptr;

        void release()
        {
            if(pref)
                hipblasLtMatmulPreferenceDestroy(pref);
            if(desc)
                hipblasLtMatmulDescDestroy(desc);
            if(a)
                hipblasLtMatrixLayoutDestroy(a);
            if(b)
                hipblasLtMatrixLayoutDestroy(b);
            if(d)
                hipblasLtMatrixLayoutDestroy(d);
            if(scaleA)
                static_cast<void>(hipFree(scaleA));
            if(scaleB)
                static_cast<void>(hipFree(scaleB));
            pref = nullptr;
            desc = nullptr;
            a = b = d = nullptr;
            scaleA = scaleB = nullptr;
        }
    };

    // A is K x M (transA = T), B is K x N (transB = N). FP8 E4M3 in, FP32 out.
    // The scale buffers are large enough for the 32x8 pre-swizzle of this shape.
    void makeQuery(Query& q, hipblasLtMatmulMatrixScale_t mode)
    {
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
        for(auto attr : {HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE})
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(q.desc, attr, &mode, sizeof(mode)),
                      HIPBLAS_STATUS_SUCCESS);

        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&q.a, HIP_R_8F_E4M3, kK, kM, kK),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&q.b, HIP_R_8F_E4M3, kK, kN, kK),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&q.d, HIP_R_32F, kM, kN, kM), HIPBLAS_STATUS_SUCCESS);

        constexpr size_t kScaleBytes = 1u << 20;
        ASSERT_EQ(hipMalloc(&q.scaleA, kScaleBytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&q.scaleB, kScaleBytes), hipSuccess);
        ASSERT_EQ(hipMemset(q.scaleA, 0, kScaleBytes), hipSuccess);
        ASSERT_EQ(hipMemset(q.scaleB, 0, kScaleBytes), hipSuccess);
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      q.desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &q.scaleA, sizeof(void*)),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      q.desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &q.scaleB, sizeof(void*)),
                  HIPBLAS_STATUS_SUCCESS);

        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&q.pref), HIPBLAS_STATUS_SUCCESS);
        uint64_t workspace = 64ull << 20;
        ASSERT_EQ(hipblasLtMatmulPreferenceSetAttribute(q.pref,
                                                        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                        &workspace,
                                                        sizeof(workspace)),
                  HIPBLAS_STATUS_SUCCESS);
    }

    struct Heuristic
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        std::vector<std::string>                      names;
    };

    Heuristic query(Query& q)
    {
        hipblasLtMatmulHeuristicResult_t raw[32]{};
        int                              count = 0;
        EXPECT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                      q.handle, q.desc, q.a, q.b, q.d, q.d, q.pref, 32, raw, &count),
                  HIPBLAS_STATUS_SUCCESS);
        Heuristic out;
        for(int i = 0; i < count; ++i)
        {
            if(raw[i].state != HIPBLAS_STATUS_SUCCESS)
                continue;
            out.results.push_back(raw[i]);
            out.names.push_back(hipblaslt_ext::getKernelNameFromAlgo(q.handle, raw[i].algo));
        }
        return out;
    }

    // isSolutionSupported ignores an algo whose first data byte is 0.
    const hipblasLtMatmulHeuristicResult_t* preswizzledAlgo(const Heuristic& pre)
    {
        for(const auto& result : pre.results)
        {
            if(result.algo.data[0] != 0)
                return &result;
        }
        return nullptr;
    }

    hipblasStatus_t launch(Query&                                     q,
                            const hipblasLtMatmulHeuristicResult_t& result,
                            void*                                   a,
                            void*                                   b,
                            void*                                   d,
                            void*                                   workspace)
    {
        const float alpha = 1.f;
        const float beta  = 0.f;
        return hipblasLtMatmul(q.handle,
                               q.desc,
                               &alpha,
                               a,
                               q.a,
                               b,
                               q.b,
                               &beta,
                               d,
                               q.d,
                               d,
                               q.d,
                               &result.algo,
                               workspace,
                               result.workspaceSize,
                               nullptr);
    }

    TEST(Gfx950MxScaleLayout, packed_scale_does_not_return_preswizzled_kernels)
    {
        if(!archIsGfx950())
            GTEST_SKIP() << "requires gfx950";

        // 0 selects the Tensile path. The handle reads this in its constructor.
        EnvGuard rocroller("0");

        Query packed;
        Query preswizzled;
        ASSERT_EQ(hipblasLtCreate(&packed.handle), HIPBLAS_STATUS_SUCCESS);
        preswizzled.handle = packed.handle;
        makeQuery(packed, HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
        makeQuery(preswizzled, HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT);
        if(HasFatalFailure())
            return;

        const Heuristic packedH = query(packed);
        const Heuristic preH    = query(preswizzled);

        ASSERT_FALSE(preH.names.empty()) << "scale 1001 returned no kernel";
        for(const auto& name : preH.names)
            EXPECT_NE(name.find("MXSFHPS"), std::string::npos) << name;
        for(const auto& name : packedH.names)
            EXPECT_EQ(name.find("MXSFHPS"), std::string::npos) << name;

        const hipblasLtMatmulHeuristicResult_t* algo = preswizzledAlgo(preH);
        ASSERT_NE(algo, nullptr);

        const float alpha = 1.f;
        const float beta  = 0.f;
        int         supported = 0;
        size_t      workspaceBytes = 0;
        hipblasLtMatmulHeuristicResult_t forPreswizzled = *algo;
        EXPECT_EQ(hipblaslt_ext::isSolutionSupported(&forPreswizzled,
                                                     packed.handle,
                                                     preswizzled.desc,
                                                     &alpha,
                                                     preswizzled.a,
                                                     preswizzled.b,
                                                     &beta,
                                                     preswizzled.d,
                                                     preswizzled.d,
                                                     &workspaceBytes,
                                                     &supported),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(supported, 1);

        // The same algo against packed scales is the call isSolutionSupported
        // makes from the heuristic top-up. Success here means that check is gone.
        int    refused = 0;
        size_t ignored = 0;
        hipblasLtMatmulHeuristicResult_t forPacked = *algo;
        EXPECT_NE(hipblaslt_ext::isSolutionSupported(&forPacked,
                                                     packed.handle,
                                                     packed.desc,
                                                     &alpha,
                                                     packed.a,
                                                     packed.b,
                                                     &beta,
                                                     packed.d,
                                                     packed.d,
                                                     &ignored,
                                                     &refused),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(refused, 0);

        void* a = nullptr;
        void* b = nullptr;
        void* d = nullptr;
        void* workspace = nullptr;
        ASSERT_EQ(hipMalloc(&a, static_cast<size_t>(kK * kM)), hipSuccess);
        ASSERT_EQ(hipMalloc(&b, static_cast<size_t>(kK * kN)), hipSuccess);
        ASSERT_EQ(hipMalloc(&d, static_cast<size_t>(kM * kN * sizeof(float))), hipSuccess);
        ASSERT_EQ(hipMemset(a, 0, static_cast<size_t>(kK * kM)), hipSuccess);
        ASSERT_EQ(hipMemset(b, 0, static_cast<size_t>(kK * kN)), hipSuccess);
        ASSERT_EQ(hipMemset(d, 0, static_cast<size_t>(kM * kN * sizeof(float))), hipSuccess);
        if(algo->workspaceSize > 0)
            ASSERT_EQ(hipMalloc(&workspace, algo->workspaceSize), hipSuccess);

        // Launching it for scale 1001 is implemented. Launching it for packed
        // scale 3 is the runContractionProblem check: not_implemented, which
        // hipblasLtMatmul reports as an internal error.
        EXPECT_EQ(launch(preswizzled, *algo, a, b, d, workspace), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipStreamSynchronize(nullptr), hipSuccess);
        EXPECT_EQ(launch(packed, *algo, a, b, d, workspace), HIPBLAS_STATUS_INTERNAL_ERROR);

        static_cast<void>(hipFree(workspace));
        static_cast<void>(hipFree(d));
        static_cast<void>(hipFree(b));
        static_cast<void>(hipFree(a));
        preswizzled.handle = nullptr;
        preswizzled.release();
        packed.release();
        hipblasLtDestroy(packed.handle);
    }
}
