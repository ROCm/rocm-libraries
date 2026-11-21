// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The C++ heuristic query must apply a workspace limit from 2 GiB to UINT32_MAX as the C query
// does. Kernels address the workspace with 32-bit offsets, so both APIs clamp anything larger to
// UINT32_MAX. Callers pass SIZE_MAX to mean "no limit", so a larger value must not be an error.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <Tensile/SolutionLibrary.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <tuple>
#include <vector>

#include "../../../library/src/amd_detail/rocblaslt/src/include/get_all_search_type_for_test.hpp"

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
                                   std::vector<hipblasLtMatmulHeuristicResult_t>& out,
                                   int requested = kRequested)
        {
            hipblasLtMatmulPreference_t pref   = nullptr;
            hipblasStatus_t             status = hipblasLtMatmulPreferenceCreate(&pref);
            if(status != HIPBLAS_STATUS_SUCCESS)
                return status;
            status = hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit));
            out.assign(requested, {});
            int count = 0;
            if(status == HIPBLAS_STATUS_SUCCESS)
                status = hipblasLtMatmulAlgoGetHeuristic(handle,
                                                         desc,
                                                         layoutA,
                                                         layoutB,
                                                         layoutD,
                                                         layoutD,
                                                         pref,
                                                         requested,
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
                                     std::vector<hipblasLtMatmulHeuristicResult_t>& out,
                                     int requested = kRequested)
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
            return gemm.algoGetHeuristic(requested, pref, out);
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

    // Ranking a single algo does not call getAllSolutions. Asking for more algos
    // than the ranking returns does, through both top-up call sites, and the
    // search type is the no-ranking one. The hook is cleared before each query.
    TEST_F(HeuristicWorkspaceLimit, smoke_TopUpPassesNoRankingSearch)
    {
        constexpr int kNoCall = -1;
        constexpr int kNoRank
            = static_cast<int>(TensileLite::SolutionLibrarySearchType::GEMM_TYPE_ONLY_NO_RANKING);
        constexpr int kMany = 100000;

        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        rocblaslt_testing_reset_last_get_all_search_type_for_test();
        if(cHeuristic(kWorkspace, results, 1) != HIPBLAS_STATUS_SUCCESS || results.empty())
            GTEST_SKIP() << "No solution for this problem in the loaded library";
        EXPECT_EQ(rocblaslt_testing_last_get_all_search_type_for_test(), kNoCall)
            << "hipblasLtMatmulAlgoGetHeuristic called getAllSolutions while the ranking "
               "already filled the request";

        rocblaslt_testing_reset_last_get_all_search_type_for_test();
        ASSERT_EQ(cHeuristic(kWorkspace, results, kMany), HIPBLAS_STATUS_SUCCESS);
        ASSERT_LT(results.size(), static_cast<size_t>(kMany));
        EXPECT_EQ(rocblaslt_testing_last_get_all_search_type_for_test(), kNoRank)
            << "topUpWithSizeIndependentSolutions did not pass a no-ranking search";

        rocblaslt_testing_reset_last_get_all_search_type_for_test();
        ASSERT_EQ(cppHeuristic(kWorkspace, results, 1), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(results.size(), 1u);
        EXPECT_EQ(rocblaslt_testing_last_get_all_search_type_for_test(), kNoCall)
            << "rocblaslt_algo_get_heuristic_cpp called getAllSolutions while the ranking "
               "already filled the request";

        rocblaslt_testing_reset_last_get_all_search_type_for_test();
        ASSERT_EQ(cppHeuristic(kWorkspace, results, kMany), HIPBLAS_STATUS_SUCCESS);
        ASSERT_LT(results.size(), static_cast<size_t>(kMany));
        EXPECT_EQ(rocblaslt_testing_last_get_all_search_type_for_test(), kNoRank)
            << "rocblaslt_algo_get_heuristic_cpp top-up did not pass a no-ranking search";
    }
} // namespace

namespace
{
    // ROCM-34680: query time is CPU work and is invisible to GPU-event timing.
    class SmallGemmHeuristic : public ::testing::Test
    {
    protected:
        static constexpr int        size           = 16;
        static constexpr uint64_t   workspaceBytes = 32 * 1024 * 1024;
        hipblasLtHandle_t           handle         = nullptr;
        hipblasLtMatmulDesc_t       desc           = nullptr;
        hipblasLtMatrixLayout_t     layout         = nullptr;
        hipblasLtMatmulPreference_t pref           = nullptr;
        void*                       a              = nullptr;
        void*                       b              = nullptr;
        void*                       c              = nullptr;
        void*                       d              = nullptr;
        void*                       workspace      = nullptr;

        void SetUp() override
        {
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layout, HIP_R_32F, size, size, size),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(
                hipblasLtMatmulPreferenceSetAttribute(pref,
                                                      HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &workspaceBytes,
                                                      sizeof(workspaceBytes)),
                HIPBLAS_STATUS_SUCCESS);
            for(void** buffer : {&a, &b, &c, &d})
                ASSERT_EQ(hipMalloc(buffer, size * size * sizeof(float)), hipSuccess);
            ASSERT_EQ(hipMalloc(&workspace, workspaceBytes), hipSuccess);
        }

        void TearDown() override
        {
            for(void* buffer : {a, b, c, d, workspace})
                if(buffer)
                    EXPECT_EQ(hipFree(buffer), hipSuccess);
            if(pref)
                EXPECT_EQ(hipblasLtMatmulPreferenceDestroy(pref), HIPBLAS_STATUS_SUCCESS);
            if(layout)
                EXPECT_EQ(hipblasLtMatrixLayoutDestroy(layout), HIPBLAS_STATUS_SUCCESS);
            if(desc)
                EXPECT_EQ(hipblasLtMatmulDescDestroy(desc), HIPBLAS_STATUS_SUCCESS);
            if(handle)
                EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
        }
    };

    TEST_F(SmallGemmHeuristic, smoke_Fp32CorrectnessAndHostQueryTime)
    {
        std::array<float, size * size>  hostA, hostB, actual;
        std::array<double, size * size> expected{};
        for(int i = 0; i < size * size; ++i)
        {
            hostA[i] = (i % 11 - 5) * 0.25f;
            hostB[i] = (i % 7 - 3) * 0.5f;
        }
        // Column-major NN product, accumulated independently in CPU double precision.
        for(int col = 0; col < size; ++col)
            for(int row = 0; row < size; ++row)
                for(int k = 0; k < size; ++k)
                    expected[row + col * size]
                        += double(hostA[row + k * size]) * double(hostB[k + col * size]);
        ASSERT_EQ(hipMemcpy(a, hostA.data(), sizeof(hostA), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(b, hostB.data(), sizeof(hostB), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemset(c, 0, sizeof(hostA)), hipSuccess);
        const float         alpha = 1.0f, beta = 0.0f;
        std::vector<double> samples;
        // Preserve the single-result query from the original 16x16x16 reproducer.
        for(int trial = 0; trial < 24; ++trial)
        {
            SCOPED_TRACE(trial);
            hipblasLtMatmulHeuristicResult_t result{};
            int                              count  = 0;
            const auto                       start  = std::chrono::steady_clock::now();
            const auto                       status = hipblasLtMatmulAlgoGetHeuristic(
                handle, desc, layout, layout, layout, layout, pref, 1, &result, &count);
            const auto end = std::chrono::steady_clock::now();
            samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(count, 1);
            ASSERT_EQ(result.state, HIPBLAS_STATUS_SUCCESS);
            ASSERT_LE(result.workspaceSize, workspaceBytes);
            // NaN poison prevents an earlier successful call from hiding a missing write.
            ASSERT_EQ(hipMemset(d, 0xff, sizeof(hostA)), hipSuccess);
            ASSERT_EQ(hipblasLtMatmul(handle,
                                      desc,
                                      &alpha,
                                      a,
                                      layout,
                                      b,
                                      layout,
                                      &beta,
                                      c,
                                      layout,
                                      d,
                                      layout,
                                      &result.algo,
                                      workspace,
                                      result.workspaceSize,
                                      nullptr),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipStreamSynchronize(nullptr), hipSuccess);
            ASSERT_EQ(hipMemcpy(actual.data(), d, sizeof(actual), hipMemcpyDeviceToHost),
                      hipSuccess);
            for(int i = 0; i < size * size; ++i)
                // These small dyadic inputs and sums are exactly representable in FP32.
                ASSERT_EQ(double(actual[i]), expected[i]) << "element " << i;
        }
        std::ostringstream raw;
        raw << std::setprecision(17);
        for(size_t i = 0; i < samples.size(); ++i)
            raw << (i ? "," : "") << samples[i];
        RecordProperty("heuristic_host_us", raw.str());
        RecordProperty("first_query_host_us", std::to_string(samples.front()));
        // First query + three warmups precede the twenty steady-state samples.
        std::vector<double> warm(samples.begin() + 4, samples.end());
        std::sort(warm.begin(), warm.end());
        RecordProperty("warm_median_host_us", std::to_string((warm[9] + warm[10]) / 2));
    }
} // namespace
