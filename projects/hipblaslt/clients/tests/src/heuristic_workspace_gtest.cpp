// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The C++ heuristic query must apply a workspace limit from 2 GiB to UINT32_MAX as the C query
// does. Kernels address the workspace with 32-bit offsets, so both APIs clamp anything larger to
// UINT32_MAX. Callers pass SIZE_MAX to mean "no limit", so a larger value must not be an error.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cstdint>
#include <limits>
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
    constexpr uint64_t    kGiB       = uint64_t{1} << 30;
    constexpr uint64_t    kUint32Max = std::numeric_limits<uint32_t>::max();
    // Covers any solution the heuristic picks for this problem, whose split-K partials are
    // capped at 128 MiB.
    constexpr size_t kWorkspace = size_t{128} << 20;

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

    // The buffers hold no data: the tests check statuses and heuristic results only.
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
        void*                   ws      = nullptr;
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
            ASSERT_EQ(hipMalloc(&ws, kWorkspace), hipSuccess);
        }

        void TearDown() override
        {
            for(void* buffer : {a, b, d, ws})
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

        hipblasStatus_t setLimit(uint64_t limit)
        {
            hipblasLtMatmulPreference_t pref   = nullptr;
            hipblasStatus_t             status = hipblasLtMatmulPreferenceCreate(&pref);
            if(status != HIPBLAS_STATUS_SUCCESS)
                return status;
            status = hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit));
            hipblasLtMatmulPreferenceDestroy(pref);
            return status;
        }

        hipblasStatus_t setProblem(hipblaslt_ext::Gemm& gemm)
        {
            hipblaslt_ext::GemmEpilogue epilogue;
            hipblaslt_ext::GemmInputs   in;
            in.setA(a);
            in.setB(b);
            in.setC(d);
            in.setD(d);
            in.setAlpha(&alpha);
            in.setBeta(&beta);
            return gemm.setProblem(kM, kN, kK, 1, epilogue, in);
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
            hipblasStatus_t     status = setProblem(gemm);
            if(status != HIPBLAS_STATUS_SUCCESS)
                return status;
            hipblaslt_ext::GemmPreference pref;
            pref.setMaxWorkspaceBytes(limit);
            return gemm.algoGetHeuristic(kRequested, pref, out);
        }
    };

    TEST_F(HeuristicWorkspaceLimit, smoke_CppQueryKeepsLimitsFrom2GiBToUint32Max)
    {
        for(uint64_t limit : {2 * kGiB, 3 * kGiB, kUint32Max})
        {
            SCOPED_TRACE(limit);
            ASSERT_EQ(setLimit(limit), HIPBLAS_STATUS_SUCCESS);
            std::vector<hipblasLtMatmulHeuristicResult_t> c, cpp;
            if(cHeuristic(limit, c) != HIPBLAS_STATUS_SUCCESS || c.empty())
                GTEST_SKIP() << "No solution for this problem in the loaded library";
            ASSERT_EQ(cppHeuristic(limit, cpp), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(c.front().algo.max_workspace_bytes, limit);
            EXPECT_EQ(summary(cpp), summary(c));
        }
    }

    // The preference stores a limit above UINT32_MAX as UINT32_MAX, so no solution library is
    // needed.
    TEST_F(HeuristicWorkspaceLimit, smoke_PreferenceClampsLimitAboveUint32Max)
    {
        for(uint64_t limit : {4 * kGiB, std::numeric_limits<uint64_t>::max()})
        {
            SCOPED_TRACE(limit);
            hipblasLtMatmulPreference_t pref = nullptr;
            ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipblasLtMatmulPreferenceSetAttribute(
                          pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit)),
                      HIPBLAS_STATUS_SUCCESS);
            uint64_t stored  = 0;
            size_t   written = 0;
            EXPECT_EQ(
                hipblasLtMatmulPreferenceGetAttribute(pref,
                                                      HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &stored,
                                                      sizeof(stored),
                                                      &written),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(stored, kUint32Max);
            hipblasLtMatmulPreferenceDestroy(pref);
        }
    }

    // Every entry point that takes a workspace size or limit treats a value above UINT32_MAX
    // as UINT32_MAX.
    TEST_F(HeuristicWorkspaceLimit, smoke_ClampsWorkspaceAboveUint32Max)
    {
        // Results at the cap are the reference. Solutions that fit the real buffer are run.
        std::vector<hipblasLtMatmulHeuristicResult_t> capped, fitting;
        if(cHeuristic(kUint32Max, capped) != HIPBLAS_STATUS_SUCCESS || capped.empty()
           || cHeuristic(kWorkspace, fitting) != HIPBLAS_STATUS_SUCCESS || fitting.empty())
            GTEST_SKIP() << "No solution for this problem in the loaded library";

        for(uint64_t limit : {4 * kGiB, std::numeric_limits<uint64_t>::max()})
        {
            SCOPED_TRACE(limit);
            std::vector<hipblasLtMatmulHeuristicResult_t> c, cpp;
            ASSERT_EQ(cHeuristic(limit, c), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(summary(c), summary(capped));
            ASSERT_EQ(cppHeuristic(limit, cpp), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(summary(cpp), summary(capped));

            // The algo's own workspace limit keeps the kernel inside the real buffer.
            EXPECT_EQ(hipblasLtMatmul(handle,
                                      desc,
                                      &alpha,
                                      a,
                                      layoutA,
                                      b,
                                      layoutB,
                                      &beta,
                                      d,
                                      layoutD,
                                      d,
                                      layoutD,
                                      &fitting.front().algo,
                                      ws,
                                      limit,
                                      nullptr),
                      HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipStreamSynchronize(nullptr), hipSuccess);

            hipblaslt_ext::Gemm gemm(handle,
                                     HIPBLAS_OP_N,
                                     HIPBLAS_OP_N,
                                     kType,
                                     kType,
                                     kType,
                                     kType,
                                     HIPBLAS_COMPUTE_32F);
            ASSERT_EQ(setProblem(gemm), HIPBLAS_STATUS_SUCCESS);
            gemm.setMaxWorkspaceBytes(limit);
            hipblasLtMatmulAlgo_t algo = fitting.front().algo;
            EXPECT_EQ(gemm.initialize(algo, ws), HIPBLAS_STATUS_SUCCESS);
        }
    }
} // namespace
