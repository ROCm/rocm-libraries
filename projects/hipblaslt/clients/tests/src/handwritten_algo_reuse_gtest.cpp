// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// An algo for a hand-written kernel is refused, through the C and the ext API, when
// the problem it is launched for fails the kernel's problem predicates or is a
// pointer-array batch, which hand-written kernels cannot address. The gfx950 skinny TN
// kernel for n == 1 asserts n == 1; its algo is reused for n == 2 and for a
// pointer-array batch of the same problem.
//
// Grouped GEMM refuses such an algo on the host-argument path only: with user args,
// every launch reads its sizes from device memory. Hand-written grouped kernels ship
// for gfx942 only (FP16 and FP8 NN, Custom_*_GG_*_UserArgs).

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace
{
    constexpr int64_t m = 4096;
    constexpr int64_t k = 4096;

    class HandwrittenAlgoReuse_smoke : public ::testing::Test
    {
    protected:
        hipblasLtHandle_t handle = nullptr;
        void*             a      = nullptr;
        void*             b      = nullptr;
        void*             d      = nullptr;
        void*             arrayA = nullptr;
        void*             arrayB = nullptr;
        void*             arrayD = nullptr;
        const float       alpha  = 1.0f;
        const float       beta   = 0.0f;
        // Room for the n == 2 problem.
        const size_t dBytes = m * 2 * sizeof(uint16_t);

        void SetUp() override
        {
            int             count = 0;
            hipDeviceProp_t props{};
            if(hipGetDeviceCount(&count) != hipSuccess || count == 0
               || hipGetDeviceProperties(&props, 0) != hipSuccess)
                GTEST_SKIP() << "No GPU available";
            if(std::string(props.gcnArchName).rfind("gfx950", 0) != 0)
                GTEST_SKIP() << "The skinny TN kernels are registered for gfx950 only";
            ASSERT_EQ(hipMalloc(&a, m * k * sizeof(uint16_t)), hipSuccess);
            ASSERT_EQ(hipMalloc(&b, k * 2 * sizeof(uint16_t)), hipSuccess);
            ASSERT_EQ(hipMalloc(&d, dBytes), hipSuccess);
            ASSERT_EQ(hipMemset(a, 0, m * k * sizeof(uint16_t)), hipSuccess);
            ASSERT_EQ(hipMemset(b, 0, k * 2 * sizeof(uint16_t)), hipSuccess);
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
        }

        void TearDown() override
        {
            if(handle)
                EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
            for(void* p : {a, b, d, arrayA, arrayB, arrayD})
                if(p)
                    EXPECT_EQ(hipFree(p), hipSuccess);
        }

        // Fills D with a pattern no GEMM of zero inputs writes, so a launch shows.
        void poisonD()
        {
            ASSERT_EQ(hipMemset(d, 0x7f, dBytes), hipSuccess);
        }

        bool dUntouched()
        {
            for(uint8_t byte : download(d, dBytes))
                if(byte != 0x7f)
                    return false;
            return true;
        }

        std::vector<uint8_t> download(const void* p, size_t bytes)
        {
            std::vector<uint8_t> host(bytes);
            EXPECT_EQ(hipMemcpy(host.data(), p, bytes, hipMemcpyDeviceToHost), hipSuccess);
            return host;
        }

        // A one-batch pointer array, as large as the tensor it points to so a kernel
        // that read it as that tensor would stay inside it.
        void makePointerArray(void*& array, void* tensor, size_t bytes)
        {
            ASSERT_EQ(hipMalloc(&array, bytes), hipSuccess);
            ASSERT_EQ(hipMemset(array, 0x7f, bytes), hipSuccess);
            ASSERT_EQ(hipMemcpy(array, &tensor, sizeof(tensor), hipMemcpyHostToDevice), hipSuccess);
        }

        void makePointerArrays()
        {
            makePointerArray(arrayA, a, m * k * sizeof(uint16_t));
            makePointerArray(arrayB, b, k * 2 * sizeof(uint16_t));
            makePointerArray(arrayD, d, dBytes);
        }

        struct CProblem
        {
            hipblasLtMatmulDesc_t   desc = nullptr;
            hipblasLtMatrixLayout_t la = nullptr, lb = nullptr, ld = nullptr;

            explicit CProblem(int64_t n)
            {
                const hipblasOperation_t opA = HIPBLAS_OP_T;
                const hipblasOperation_t opB = HIPBLAS_OP_N;
                EXPECT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                          HIPBLAS_STATUS_SUCCESS);
                EXPECT_EQ(
                    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)),
                    HIPBLAS_STATUS_SUCCESS);
                EXPECT_EQ(
                    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)),
                    HIPBLAS_STATUS_SUCCESS);
                EXPECT_EQ(hipblasLtMatrixLayoutCreate(&la, HIP_R_16BF, k, m, k), HIPBLAS_STATUS_SUCCESS);
                EXPECT_EQ(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16BF, k, n, k), HIPBLAS_STATUS_SUCCESS);
                EXPECT_EQ(hipblasLtMatrixLayoutCreate(&ld, HIP_R_16BF, m, n, m), HIPBLAS_STATUS_SUCCESS);
            }

            void setPointerArray()
            {
                const hipblasLtBatchMode_t mode = HIPBLASLT_BATCH_MODE_POINTER_ARRAY;
                for(auto* layout : {la, lb, ld})
                    EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(
                                  layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE, &mode, sizeof(mode)),
                              HIPBLAS_STATUS_SUCCESS);
            }

            ~CProblem()
            {
                for(auto* layout : {la, lb, ld})
                    hipblasLtMatrixLayoutDestroy(layout);
                hipblasLtMatmulDescDestroy(desc);
            }
        };

        hipblasStatus_t matmul(const CProblem& p, const hipblasLtMatmulAlgo_t& algo)
        {
            return matmul(p, algo, a, b, d);
        }

        hipblasStatus_t matmul(const CProblem&              p,
                               const hipblasLtMatmulAlgo_t& algo,
                               const void*                  A,
                               const void*                  B,
                               void*                        D)
        {
            return hipblasLtMatmul(
                handle, p.desc, &alpha, A, p.la, B, p.lb, &beta, D, p.ld, D, p.ld, &algo, nullptr, 0, nullptr);
        }

        void setExtProblem(hipblaslt_ext::Gemm& gemm, int64_t n)
        {
            hipblaslt_ext::GemmProblemType pt(HIPBLAS_OP_T,
                                              HIPBLAS_OP_N,
                                              HIP_R_16BF,
                                              HIP_R_16BF,
                                              HIP_R_16BF,
                                              HIP_R_16BF,
                                              HIPBLAS_COMPUTE_32F);
            hipblaslt_ext::GemmEpilogue epilogue;
            hipblaslt_ext::GemmInputs   inputs;
            inputs.setA(a);
            inputs.setB(b);
            inputs.setC(d);
            inputs.setD(d);
            inputs.setAlpha(&alpha);
            inputs.setBeta(&beta);
            ASSERT_EQ(gemm.setProblem(m, n, k, 1, k, k, m, m, k * m, k * n, m * n, m * n, epilogue, inputs, pt),
                      HIPBLAS_STATUS_SUCCESS);
        }

        hipblasLtMatmulAlgo_t skinnyAlgo()
        {
            CProblem                         p(1);
            hipblasLtMatmulPreference_t      pref = nullptr;
            hipblasLtMatmulHeuristicResult_t result{};
            int                              count = 0;
            EXPECT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                          handle, p.desc, p.la, p.lb, p.ld, p.ld, pref, 1, &result, &count),
                      HIPBLAS_STATUS_SUCCESS);
            hipblasLtMatmulPreferenceDestroy(pref);
            EXPECT_EQ(count, 1);
            EXPECT_EQ(hipblaslt_ext::getSolutionNameFromAlgo(handle, result.algo)
                          .rfind("wvSplitK_vllm_bf16_tn_n1_", 0),
                      0u)
                << "the n == 1 pick is not the hand-written skinny kernel";
            return result.algo;
        }
    };

    TEST_F(HandwrittenAlgoReuse_smoke, CApiRefusesAlgoForAnotherProblem)
    {
        const hipblasLtMatmulAlgo_t algo = skinnyAlgo();
        if(HasFailure())
            return;

        EXPECT_EQ(matmul(CProblem(1), algo), HIPBLAS_STATUS_SUCCESS);

        poisonD();
        EXPECT_EQ(matmul(CProblem(2), algo), HIPBLAS_STATUS_INVALID_VALUE);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_TRUE(dUntouched()) << "the refused algo launched";
    }

    TEST_F(HandwrittenAlgoReuse_smoke, ExtApiRefusesAlgoForAnotherProblem)
    {
        const hipblasLtMatmulAlgo_t algo = skinnyAlgo();
        if(HasFailure())
            return;

        hipblaslt_ext::Gemm gemm(
            handle, HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F);
        setExtProblem(gemm, 1);
        EXPECT_EQ(gemm.initialize(algo, nullptr), HIPBLAS_STATUS_SUCCESS);

        setExtProblem(gemm, 2);
        EXPECT_EQ(gemm.initialize(algo, nullptr), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(HandwrittenAlgoReuse_smoke, CApiRefusesAlgoForPointerArrayBatch)
    {
        const hipblasLtMatmulAlgo_t algo = skinnyAlgo();
        if(HasFailure())
            return;
        makePointerArrays();
        if(HasFailure())
            return;

        CProblem p(1);
        p.setPointerArray();
        poisonD();
        const std::vector<uint8_t> arrayDBefore = download(arrayD, dBytes);
        EXPECT_EQ(matmul(p, algo, arrayA, arrayB, arrayD), HIPBLAS_STATUS_INVALID_VALUE);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_TRUE(dUntouched()) << "the refused algo launched";
        EXPECT_EQ(download(arrayD, dBytes), arrayDBefore) << "the refused algo launched";
    }

    TEST_F(HandwrittenAlgoReuse_smoke, ExtApiRefusesAlgoForPointerArrayBatch)
    {
        const hipblasLtMatmulAlgo_t algo = skinnyAlgo();
        if(HasFailure())
            return;
        makePointerArrays();
        if(HasFailure())
            return;

        CProblem p(1);
        p.setPointerArray();
        hipblaslt_ext::Gemm gemm(
            handle, p.desc, &alpha, arrayA, p.la, arrayB, p.lb, &beta, arrayD, p.ld, arrayD, p.ld);
        EXPECT_EQ(gemm.initialize(algo, nullptr), HIPBLAS_STATUS_INVALID_VALUE);
    }

    // Two FP16 NN groups. The second group's M is either kSize or kOddM, which the
    // chosen hand-written kernel's M-alignment predicate refuses.
    class HandwrittenGroupedAlgoReuse_smoke : public ::testing::Test
    {
    protected:
        static constexpr int64_t kSize      = 256;
        static constexpr int64_t kOddM      = 130;
        static constexpr size_t  kGroups    = 2;
        static constexpr size_t  kWorkspace = size_t(32) << 20;
        static constexpr size_t  kBytes     = kSize * kSize * sizeof(uint16_t);

        hipblasLtHandle_t  handle    = nullptr;
        void*              a         = nullptr;
        void*              b         = nullptr;
        void*              workspace = nullptr;
        std::vector<void*> d         = std::vector<void*>(kGroups, nullptr);
        const float        alpha     = 1.0f;
        const float        beta      = 0.0f;

        void SetUp() override
        {
            int             count = 0;
            hipDeviceProp_t props{};
            if(hipGetDeviceCount(&count) != hipSuccess || count == 0
               || hipGetDeviceProperties(&props, 0) != hipSuccess)
                GTEST_SKIP() << "No GPU available";
            if(std::string(props.gcnArchName).rfind("gfx942", 0) != 0)
                GTEST_SKIP() << "Hand-written grouped kernels ship for gfx942 only";
            ASSERT_EQ(hipMalloc(&a, kBytes), hipSuccess);
            ASSERT_EQ(hipMalloc(&b, kBytes), hipSuccess);
            ASSERT_EQ(hipMemset(a, 0, kBytes), hipSuccess);
            ASSERT_EQ(hipMemset(b, 0, kBytes), hipSuccess);
            for(void*& p : d)
                ASSERT_EQ(hipMalloc(&p, kBytes), hipSuccess);
            ASSERT_EQ(hipMalloc(&workspace, kWorkspace), hipSuccess);
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
            // Hand-written kernels are not uniform-summation-order safe, and with the
            // mode off the support check evaluates the problem predicates the launch
            // re-checks.
            ASSERT_EQ(hipblasLtSetUniformSummationOrder(handle, 0), HIPBLAS_STATUS_SUCCESS);
        }

        void TearDown() override
        {
            if(handle)
                EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
            for(void* p : {a, b, workspace, d[0], d[1]})
                if(p)
                    EXPECT_EQ(hipFree(p), hipSuccess);
        }

        hipblaslt_ext::GroupedGemm makeGroupedGemm()
        {
            return hipblaslt_ext::GroupedGemm(
                handle, HIPBLAS_OP_N, HIPBLAS_OP_N, HIP_R_16F, HIP_R_16F, HIP_R_16F, HIP_R_16F, HIPBLAS_COMPUTE_32F);
        }

        void setProblem(hipblaslt_ext::GroupedGemm& grouped, int64_t secondM)
        {
            std::vector<int64_t> m{kSize, secondM}, n(kGroups, kSize), k(kGroups, kSize),
                batch(kGroups, 1);
            std::vector<hipblaslt_ext::GemmEpilogue> epilogues(kGroups);
            std::vector<hipblaslt_ext::GemmInputs>   inputs(kGroups);
            for(size_t i = 0; i < kGroups; i++)
            {
                inputs[i].setA(a);
                inputs[i].setB(b);
                inputs[i].setC(d[i]);
                inputs[i].setD(d[i]);
                inputs[i].setAlpha(&alpha);
                inputs[i].setBeta(&beta);
            }
            ASSERT_EQ(grouped.setProblem(m, n, k, batch, epilogues, inputs), HIPBLAS_STATUS_SUCCESS);
            grouped.setMaxWorkspaceBytes(kWorkspace);
        }

        static bool isHandwritten(const std::string& name)
        {
            return name.rfind("Custom_", 0) == 0;
        }

        // A hand-written grouped algo the support check accepts for the aligned
        // problem and refuses for the odd one. Probed on objects of its own, so the
        // tests' objects never ran a support check before initialize.
        bool handwrittenAlgo(hipblasLtMatmulAlgo_t& algo)
        {
            std::vector<hipblasLtMatmulHeuristicResult_t> results;
            EXPECT_EQ(hipblaslt_ext::getAllAlgos(handle,
                                                 hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM,
                                                 HIPBLAS_OP_N,
                                                 HIPBLAS_OP_N,
                                                 HIP_R_16F,
                                                 HIP_R_16F,
                                                 HIP_R_16F,
                                                 HIP_R_16F,
                                                 HIPBLAS_COMPUTE_32F,
                                                 results),
                      HIPBLAS_STATUS_SUCCESS);
            auto aligned = makeGroupedGemm();
            auto odd     = makeGroupedGemm();
            setProblem(aligned, kSize);
            setProblem(odd, kOddM);
            for(auto& result : results)
            {
                if(!isHandwritten(hipblaslt_ext::getKernelNameFromAlgo(handle, result.algo))
                   && !isHandwritten(hipblaslt_ext::getSolutionNameFromAlgo(handle, result.algo)))
                    continue;
                size_t required = 0;
                if(aligned.isAlgoSupported(result.algo, required) != HIPBLAS_STATUS_SUCCESS
                   || required > kWorkspace)
                    continue;
                if(odd.isAlgoSupported(result.algo, required) == HIPBLAS_STATUS_SUCCESS)
                    continue;
                algo = result.algo;
                return true;
            }
            return false;
        }

        void poisonD()
        {
            for(void* p : d)
                ASSERT_EQ(hipMemset(p, 0x7f, kBytes), hipSuccess);
        }

        bool dUntouched()
        {
            std::vector<uint8_t> host(kBytes);
            for(void* p : d)
            {
                EXPECT_EQ(hipMemcpy(host.data(), p, kBytes, hipMemcpyDeviceToHost), hipSuccess);
                for(uint8_t byte : host)
                    if(byte != 0x7f)
                        return false;
            }
            return true;
        }
    };

    TEST_F(HandwrittenGroupedAlgoReuse_smoke, HostArgsInitializeRefusesAlgoForAnotherProblem)
    {
        hipblasLtMatmulAlgo_t algo{};
        ASSERT_TRUE(handwrittenAlgo(algo))
            << "no hand-written grouped algo that refuses M = " << kOddM;

        auto grouped = makeGroupedGemm();
        setProblem(grouped, kSize);
        EXPECT_EQ(grouped.initialize(algo, workspace, false), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(grouped.run(nullptr), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        auto refused = makeGroupedGemm();
        setProblem(refused, kOddM);
        poisonD();
        EXPECT_EQ(refused.initialize(algo, workspace, false), HIPBLAS_STATUS_INVALID_VALUE);
        // A run after the refusal has no kernels to launch.
        static_cast<void>(refused.run(nullptr));
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_TRUE(dUntouched()) << "the refused algo launched";
    }

    TEST_F(HandwrittenGroupedAlgoReuse_smoke, UserArgsInitializeLeavesSizesToTheCaller)
    {
        hipblasLtMatmulAlgo_t algo{};
        ASSERT_TRUE(handwrittenAlgo(algo))
            << "no hand-written grouped algo that refuses M = " << kOddM;

        auto grouped = makeGroupedGemm();
        setProblem(grouped, kSize);
        ASSERT_EQ(grouped.initialize(algo, workspace, true), HIPBLAS_STATUS_SUCCESS);
        std::vector<hipblaslt_ext::UserArguments> userArgs(kGroups);
        ASSERT_EQ(grouped.getDefaultValueForDeviceUserArguments(userArgs.data()),
                  HIPBLAS_STATUS_SUCCESS);
        void* deviceUserArgs = nullptr;
        const size_t userArgsBytes = kGroups * sizeof(hipblaslt_ext::UserArguments);
        ASSERT_EQ(hipMalloc(&deviceUserArgs, userArgsBytes), hipSuccess);
        auto freeBuffer = [](void* p) { static_cast<void>(hipFree(p)); };
        const std::unique_ptr<void, decltype(freeBuffer)> owner(deviceUserArgs, freeBuffer);
        ASSERT_EQ(hipMemcpy(deviceUserArgs, userArgs.data(), userArgsBytes, hipMemcpyHostToDevice),
                  hipSuccess);
        EXPECT_EQ(grouped.run(deviceUserArgs, nullptr), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        // The host sizes may be placeholders for the device ones, so they are not
        // checked. Not run: these sizes are outside the kernel's predicates.
        auto placeholders = makeGroupedGemm();
        setProblem(placeholders, kOddM);
        EXPECT_EQ(placeholders.initialize(algo, workspace, true), HIPBLAS_STATUS_SUCCESS);
    }
}
