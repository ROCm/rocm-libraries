// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// hipblasLtMatmul with a null algo selects the algorithm itself. It must find one wherever
// hipblasLtMatmulAlgoGetHeuristic does for the same problem and workspace, and when no algorithm
// supports the problem it must say so with HIPBLAS_STATUS_NOT_SUPPORTED, which callers can act
// on, rather than HIPBLAS_STATUS_INTERNAL_ERROR.

#include "scoped_env.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

#include "../../../library/src/amd_detail/rocblaslt/src/include/solution_lookup_miss_for_test.hpp"

namespace
{
    // Restores normal lookups when a test ends, including on a failed assertion.
    class LookupMiss
    {
    public:
        explicit LookupMiss(rocblaslt_testing_solution_lookup_miss miss)
        {
            rocblaslt_testing_set_solution_lookup_miss_for_test(miss);
        }
        ~LookupMiss()
        {
            rocblaslt_testing_set_solution_lookup_miss_for_test(
                rocblaslt_testing_solution_lookup_miss_none);
        }
        LookupMiss(const LookupMiss&)            = delete;
        LookupMiss& operator=(const LookupMiss&) = delete;
    };

    struct Types
    {
        const char* name;
        hipDataType ab, cd;
    };

    // A column-major D = op(A) * op(B) problem with no workspace. The buffers are sized for the
    // largest problem the tests use and hold zeros: the tests check statuses only.
    class NullAlgoMatmul : public ::testing::Test
    {
    public:
        static constexpr size_t kBufferBytes = size_t(1) << 20;

        hipblasLtHandle_t           handle = nullptr;
        hipblasLtMatmulPreference_t pref   = nullptr;
        void*                       a      = nullptr;
        void*                       b      = nullptr;
        void*                       d      = nullptr;

        void SetUp() override
        {
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
            for(void** buffer : {&a, &b, &d})
            {
                ASSERT_EQ(hipMalloc(buffer, kBufferBytes), hipSuccess);
                ASSERT_EQ(hipMemset(*buffer, 0, kBufferBytes), hipSuccess);
            }
        }

        void TearDown() override
        {
            for(void* buffer : {a, b, d})
                static_cast<void>(hipFree(buffer));
            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtDestroy(handle);
        }

        struct Problem
        {
            using DescPtr   = std::unique_ptr<hipblasLtMatmulDescOpaque_t,
                                            decltype(&hipblasLtMatmulDescDestroy)>;
            using LayoutPtr = std::unique_ptr<hipblasLtMatrixLayoutOpaque_t,
                                              decltype(&hipblasLtMatrixLayoutDestroy)>;
            DescPtr   desc{nullptr, &hipblasLtMatmulDescDestroy};
            LayoutPtr la{nullptr, &hipblasLtMatrixLayoutDestroy};
            LayoutPtr lb{nullptr, &hipblasLtMatrixLayoutDestroy};
            LayoutPtr ld{nullptr, &hipblasLtMatrixLayoutDestroy};
        };

        static Problem makeProblem(const Types&       types,
                                   hipblasOperation_t opA,
                                   hipblasOperation_t opB,
                                   int64_t            m,
                                   int64_t            n,
                                   int64_t            k)
        {
            Problem                 p;
            hipblasLtMatmulDesc_t   desc = nullptr;
            hipblasLtMatrixLayout_t la = nullptr, lb = nullptr, ld = nullptr;
            EXPECT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                      HIPBLAS_STATUS_SUCCESS);
            p.desc.reset(desc);
            EXPECT_EQ(
                hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(
                hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)),
                HIPBLAS_STATUS_SUCCESS);
            const int64_t aRows = opA == HIPBLAS_OP_N ? m : k, aCols = opA == HIPBLAS_OP_N ? k : m;
            const int64_t bRows = opB == HIPBLAS_OP_N ? k : n, bCols = opB == HIPBLAS_OP_N ? n : k;
            EXPECT_EQ(hipblasLtMatrixLayoutCreate(&la, types.ab, aRows, aCols, aRows),
                      HIPBLAS_STATUS_SUCCESS);
            p.la.reset(la);
            EXPECT_EQ(hipblasLtMatrixLayoutCreate(&lb, types.ab, bRows, bCols, bRows),
                      HIPBLAS_STATUS_SUCCESS);
            p.lb.reset(lb);
            EXPECT_EQ(hipblasLtMatrixLayoutCreate(&ld, types.cd, m, n, m), HIPBLAS_STATUS_SUCCESS);
            p.ld.reset(ld);
            return p;
        }

        int heuristicCount(const Problem& p)
        {
            hipblasLtMatmulHeuristicResult_t result{};
            int                              returned = -1;
            EXPECT_EQ(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                      p.desc.get(),
                                                      p.la.get(),
                                                      p.lb.get(),
                                                      p.ld.get(),
                                                      p.ld.get(),
                                                      pref,
                                                      1,
                                                      &result,
                                                      &returned),
                      HIPBLAS_STATUS_SUCCESS);
            return returned;
        }

        hipblasStatus_t nullAlgoMatmul(const Problem& p)
        {
            const float           alpha = 1.0f, beta = 0.0f;
            const hipblasStatus_t status = hipblasLtMatmul(handle,
                                                           p.desc.get(),
                                                           &alpha,
                                                           a,
                                                           p.la.get(),
                                                           b,
                                                           p.lb.get(),
                                                           &beta,
                                                           d,
                                                           p.ld.get(),
                                                           d,
                                                           p.ld.get(),
                                                           nullptr,
                                                           nullptr,
                                                           0,
                                                           nullptr);
            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
            return status;
        }
    };

    constexpr Types kFp32{"f32", HIP_R_32F, HIP_R_32F};

    std::string describe(const Types&       types,
                         hipblasOperation_t opA,
                         hipblasOperation_t opB,
                         int64_t            m,
                         int64_t            n,
                         int64_t            k)
    {
        return std::string(types.name) + " op " + (opA == HIPBLAS_OP_N ? "N" : "T") + "/"
               + (opB == HIPBLAS_OP_N ? "N" : "T") + " " + std::to_string(m) + "x"
               + std::to_string(n) + "x" + std::to_string(k);
    }

    TEST_F(NullAlgoMatmul, smoke_NoAlgorithmIsNotSupported)
    {
        const Problem p = makeProblem(kFp32, HIPBLAS_OP_N, HIPBLAS_OP_N, 64, 64, 64);
        if(heuristicCount(p) == 0)
            GTEST_SKIP() << "the loaded library has no FP32 algorithm for the control problem";
        ASSERT_EQ(nullAlgoMatmul(p), HIPBLAS_STATUS_SUCCESS) << "control, with normal lookups";

        const LookupMiss miss(rocblaslt_testing_solution_lookup_miss_all);
        EXPECT_EQ(heuristicCount(p), 0);
        EXPECT_EQ(nullAlgoMatmul(p), HIPBLAS_STATUS_NOT_SUPPORTED);
    }

    TEST_F(NullAlgoMatmul, smoke_NoAlgorithmIsLogged)
    {
        // The logger reads its environment once per process, so the call runs in a newly
        // started copy of this binary and the test matches the copy's stderr.
        GTEST_FLAG_SET(death_test_style, "threadsafe");
        const test_env::ScopedEnv level("HIPBLASLT_LOG_LEVEL", "1");
        const test_env::ScopedEnv mask("HIPBLASLT_LOG_MASK", nullptr);
        const test_env::ScopedEnv file("HIPBLASLT_LOG_FILE", nullptr);
        EXPECT_EXIT(
            {
                const Problem    p = makeProblem(kFp32, HIPBLAS_OP_N, HIPBLAS_OP_N, 64, 64, 64);
                const LookupMiss miss(rocblaslt_testing_solution_lookup_miss_all);
                std::exit(nullAlgoMatmul(p) == HIPBLAS_STATUS_NOT_SUPPORTED ? 0 : 1);
            },
            ::testing::ExitedWithCode(0),
            "no algorithm supports this problem: arch=.* m=64 n=64 k=64 .*type_a=f32_r");
    }

    // Over these types, ops and sizes, a null algo runs wherever the heuristic finds an algorithm
    // and reports HIPBLAS_STATUS_NOT_SUPPORTED where it finds none. Returns how many problems the
    // heuristic found an algorithm for.
    int expectNullAlgoAgreesWithHeuristic(NullAlgoMatmul& t)
    {
        const Types   types[]    = {kFp32,
                                    {"f16", HIP_R_16F, HIP_R_16F},
                                    {"f16->f32", HIP_R_16F, HIP_R_32F},
                                    {"bf16", HIP_R_16BF, HIP_R_16BF},
                                    {"bf16->f32", HIP_R_16BF, HIP_R_32F}};
        const int64_t sizes[][3] = {{2, 4, 3}, {37, 19, 11}, {100, 60, 70}};
        int           found      = 0;
        for(const Types& types_ : types)
            for(auto opA : {HIPBLAS_OP_N, HIPBLAS_OP_T})
                for(auto opB : {HIPBLAS_OP_N, HIPBLAS_OP_T})
                    for(const auto& s : sizes)
                    {
                        const auto p = t.makeProblem(types_, opA, opB, s[0], s[1], s[2]);
                        const std::string which = describe(types_, opA, opB, s[0], s[1], s[2]);
                        if(t.heuristicCount(p) > 0)
                        {
                            found++;
                            EXPECT_EQ(t.nullAlgoMatmul(p), HIPBLAS_STATUS_SUCCESS) << which;
                        }
                        else
                            EXPECT_EQ(t.nullAlgoMatmul(p), HIPBLAS_STATUS_NOT_SUPPORTED) << which;
                    }
        return found;
    }

    // As for a problem no ranked library row covers, such as one outside an exact-size table:
    // the heuristic tops up with size-independent algorithms, and a null algo must too.
    TEST_F(NullAlgoMatmul, smoke_RankedMissUsesSizeIndependentAlgorithms)
    {
        const LookupMiss miss(rocblaslt_testing_solution_lookup_miss_ranked);
        if(expectNullAlgoAgreesWithHeuristic(*this) == 0)
            GTEST_SKIP() << "the loaded library has no size-independent algorithm for these problems";
    }

    // With the libraries as shipped, at sizes outside the tuned tables.
    TEST_F(NullAlgoMatmul, smoke_RunsWhereverHeuristicFindsAnAlgorithm)
    {
        if(expectNullAlgoAgreesWithHeuristic(*this) == 0)
            GTEST_SKIP() << "the loaded library has no algorithm for any of these problems";
    }
} // namespace
