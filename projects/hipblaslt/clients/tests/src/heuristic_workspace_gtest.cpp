// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The C++ heuristic query must apply a workspace limit of 2 GiB or more as the C query does.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cstdint>
#include <tuple>
#include <vector>

namespace
{
    // FP16 NN with a long K, so that split-K and Stream-K solutions, which need a
    // workspace, rank among the first results.
    constexpr int64_t     kM         = 256;
    constexpr int64_t     kN         = 256;
    constexpr int64_t     kK         = 16384;
    constexpr hipDataType kType      = HIP_R_16F;
    constexpr int         kRequested = 32;

    // Solution index, required workspace and workspace limit of each result.
    std::vector<std::tuple<int, size_t, size_t>>
        summary(const std::vector<hipblasLtMatmulHeuristicResult_t>& results)
    {
        std::vector<std::tuple<int, size_t, size_t>> out;
        for(const auto& result : results)
            out.emplace_back(*reinterpret_cast<const int*>(result.algo.data),
                             result.workspaceSize,
                             result.algo.max_workspace_bytes);
        return out;
    }

    // The buffers are never read: only the heuristic queries run.
    class HeuristicWorkspaceLimit : public ::testing::Test
    {
    protected:
        hipblasLtHandle_t       handle  = nullptr;
        hipblasLtMatmulDesc_t   desc    = nullptr;
        hipblasLtMatrixLayout_t layoutA = nullptr;
        hipblasLtMatrixLayout_t layoutB = nullptr;
        hipblasLtMatrixLayout_t layoutD = nullptr;
        void*                   a       = nullptr;
        void*                   b       = nullptr;
        void*                   d       = nullptr;
        float                   alpha   = 1.0f;
        float                   beta    = 0.0f;

        void SetUp() override
        {
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layoutA, kType, kM, kK, kM),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layoutB, kType, kK, kN, kK),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layoutD, kType, kM, kN, kM),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipMalloc(&a, kM * kK * sizeof(uint16_t)), hipSuccess);
            ASSERT_EQ(hipMalloc(&b, kK * kN * sizeof(uint16_t)), hipSuccess);
            ASSERT_EQ(hipMalloc(&d, kM * kN * sizeof(uint16_t)), hipSuccess);
        }

        void TearDown() override
        {
            for(void* buffer : {a, b, d})
                static_cast<void>(hipFree(buffer));
            for(hipblasLtMatrixLayout_t layout : {layoutA, layoutB, layoutD})
                if(layout)
                    hipblasLtMatrixLayoutDestroy(layout);
            if(desc)
                hipblasLtMatmulDescDestroy(desc);
            if(handle)
                hipblasLtDestroy(handle);
        }

        hipblasStatus_t cHeuristic(uint64_t                                       limit,
                                   std::vector<hipblasLtMatmulHeuristicResult_t>& out)
        {
            hipblasLtMatmulPreference_t pref   = nullptr;
            hipblasStatus_t             status = hipblasLtMatmulPreferenceCreate(&pref);
            if(status != HIPBLAS_STATUS_SUCCESS)
                return status;
            status = hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit));
            out.assign(kRequested, {});
            int count = 0;
            if(status == HIPBLAS_STATUS_SUCCESS)
                status = hipblasLtMatmulAlgoGetHeuristic(handle,
                                                         desc,
                                                         layoutA,
                                                         layoutB,
                                                         layoutD,
                                                         layoutD,
                                                         pref,
                                                         kRequested,
                                                         out.data(),
                                                         &count);
            out.resize(status == HIPBLAS_STATUS_SUCCESS ? count : 0);
            hipblasLtMatmulPreferenceDestroy(pref);
            return status;
        }

        hipblasStatus_t cppHeuristic(uint64_t                                       limit,
                                     std::vector<hipblasLtMatmulHeuristicResult_t>& out)
        {
            hipblaslt_ext::Gemm gemm(handle,
                                     HIPBLAS_OP_N,
                                     HIPBLAS_OP_N,
                                     kType,
                                     kType,
                                     kType,
                                     kType,
                                     HIPBLAS_COMPUTE_32F);
            hipblaslt_ext::GemmEpilogue epilogue;
            hipblaslt_ext::GemmInputs   in;
            in.setA(a);
            in.setB(b);
            in.setC(d);
            in.setD(d);
            in.setAlpha(&alpha);
            in.setBeta(&beta);
            hipblasStatus_t status = gemm.setProblem(kM, kN, kK, 1, epilogue, in);
            if(status != HIPBLAS_STATUS_SUCCESS)
                return status;
            hipblaslt_ext::GemmPreference pref;
            pref.setMaxWorkspaceBytes(limit);
            return gemm.algoGetHeuristic(kRequested, pref, out);
        }
    };

    TEST_F(HeuristicWorkspaceLimit, smoke_CppQueryKeepsLimitsOf2GiBAndMore)
    {
        for(uint64_t limit : {2ull << 30, 3ull << 30, 4ull << 30})
        {
            SCOPED_TRACE(limit);
            std::vector<hipblasLtMatmulHeuristicResult_t> c, cpp;
            if(cHeuristic(limit, c) != HIPBLAS_STATUS_SUCCESS || c.empty())
                GTEST_SKIP() << "No solution for this problem in the loaded library";
            ASSERT_EQ(cppHeuristic(limit, cpp), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(c.front().algo.max_workspace_bytes, limit);
            EXPECT_EQ(summary(cpp), summary(c));
        }
    }
} // namespace
