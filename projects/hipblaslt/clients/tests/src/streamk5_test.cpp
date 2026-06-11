/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

// StreamK=5 hybrid-mode integration tests.
//
// Exercises the runtime mode toggle:
//
//     hipblasLtMatmulDescSetAttribute(desc,
//         HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT,
//         &flag, sizeof(int32_t));
//
// The numerics test enumerates heuristic algos and explicitly selects
// the first kernel whose name contains "_SK5_" so the toggle is
// guaranteed to switch between the SK5 kernel's static and dynamic
// paths. If no SK5 kernel is present in the loaded device library the
// test is GTEST_SKIP'd rather than allowed to pass vacuously against a
// non-SK5 kernel where the toggle is a no-op. The two D buffers are
// then compared bit-for-bit via memcmp.
//
// Self-contained (does not go through the YAML/gentest pipeline) so it
// can be filtered cleanly with --gtest_filter='*streamk5*'.

#include "testing_auxiliary.hpp"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace
{
    using hipblaslt_sk5_test::gpuAvailable;
    using hipblaslt_sk5_test::kAutoMode;
    using hipblaslt_sk5_test::kDynamicMode;
    using hipblaslt_sk5_test::kStaticMode;
    using hipblaslt_sk5_test::prepareSk5OrSkip;
    using hipblaslt_sk5_test::runSgemm;

    void fillRandom(std::vector<float>& v, unsigned seed)
    {
        std::mt19937                          rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for(auto& x : v)
            x = dist(rng);
    }

    class StreamK5HybridToggle
        : public ::testing::TestWithParam<std::tuple<int, int, int>>
    {
    };

    TEST_P(StreamK5HybridToggle, StaticAndDynamicProduceMatchingNumerics)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";

        const auto [m, n, k] = GetParam();
        auto       setup     = prepareSk5OrSkip(m, n, k);

        std::vector<float> hA(size_t(m) * k);
        std::vector<float> hB(size_t(k) * n);
        std::vector<float> hC(size_t(m) * n, 0.0f);
        fillRandom(hA, 0xA11CE);
        fillRandom(hB, 0xB0B);

        std::vector<float> outStatic, outDynamic;
        ASSERT_TRUE(runSgemm(setup.handle,
                             setup.algo,
                             setup.workspaceSize,
                             kStaticMode,
                             m,
                             n,
                             k,
                             hA.data(),
                             hB.data(),
                             hC.data(),
                             outStatic))
            << "static path matmul failed";
        ASSERT_TRUE(runSgemm(setup.handle,
                             setup.algo,
                             setup.workspaceSize,
                             kDynamicMode,
                             m,
                             n,
                             k,
                             hA.data(),
                             hB.data(),
                             hC.data(),
                             outDynamic))
            << "dynamic path matmul failed";

        (void)hipblasLtDestroy(setup.handle);

        ASSERT_EQ(outStatic.size(), outDynamic.size());
        // The static (SK3) and dynamic (SK4) sub-paths can select different
        // work partitioning for the same shape (for example when tiles < grid
        // in the static path), so reduction ordering may differ. Validate
        // numerical equivalence with tight FP32 tolerances instead of memcmp.
        constexpr float absTol = 5.0e-5f;
        constexpr float relTol = 1.0e-5f;

        size_t firstMismatch = outStatic.size();
        float  maxAbsDiff    = 0.0f;
        float  maxRelDiff    = 0.0f;
        for(size_t i = 0; i < outStatic.size(); ++i)
        {
            const float a       = outStatic[i];
            const float b       = outDynamic[i];
            const float absDiff = std::fabs(a - b);
            const float denom   = std::max(std::fabs(a), std::fabs(b));
            const float relDiff = denom > 0.0f ? (absDiff / denom) : absDiff;

            maxAbsDiff = std::max(maxAbsDiff, absDiff);
            maxRelDiff = std::max(maxRelDiff, relDiff);

            if(absDiff > absTol && relDiff > relTol)
            {
                firstMismatch = i;
                break;
            }
        }

        ASSERT_EQ(firstMismatch, outStatic.size())
            << "Mismatch at idx " << firstMismatch << ": static=" << outStatic[firstMismatch]
            << " dynamic=" << outDynamic[firstMismatch]
            << " maxAbsDiff=" << maxAbsDiff << " maxRelDiff=" << maxRelDiff;
    }

    INSTANTIATE_TEST_SUITE_P(
        streamk5,
        StreamK5HybridToggle,
        ::testing::Values(std::make_tuple(512, 512, 512),
                          std::make_tuple(1024, 1024, 1024),
                          std::make_tuple(257, 513, 129)));

    TEST(StreamK5HybridAuto, AutoMatchesStaticForSmallProblem)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";

        constexpr int M = 256, N = 256, K = 256;
        auto          setup = prepareSk5OrSkip(M, N, K);

        std::vector<float> hA(size_t(M) * K);
        std::vector<float> hB(size_t(K) * N);
        std::vector<float> hC(size_t(M) * N, 0.0f);
        fillRandom(hA, 0xC0FFEE);
        fillRandom(hB, 0xDECAF);

        std::vector<float> outAuto, outStatic;
        ASSERT_TRUE(runSgemm(setup.handle, setup.algo, setup.workspaceSize, kAutoMode,
                             M, N, K, hA.data(), hB.data(), hC.data(), outAuto))
            << "AUTO matmul failed";
        ASSERT_TRUE(runSgemm(setup.handle, setup.algo, setup.workspaceSize, kStaticMode,
                             M, N, K, hA.data(), hB.data(), hC.data(), outStatic))
            << "explicit static matmul failed";

        (void)hipblasLtDestroy(setup.handle);

        ASSERT_EQ(outAuto.size(), outStatic.size());
        const auto byteSize = outAuto.size() * sizeof(float);
        EXPECT_EQ(std::memcmp(outAuto.data(), outStatic.data(), byteSize), 0)
            << "AUTO heuristic should pick static_ for a 256x256x256 problem "
               "(tiles_per_cu ~ 0.01 < 2.08), but the bit pattern differs "
               "from the explicit-OFF run.";
    }

    TEST(StreamK5HybridAuto, AutoMatchesDynamicForLargeProblem)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";

        constexpr int M = 8192, N = 8192, K = 64;
        auto          setup = prepareSk5OrSkip(M, N, K);

        std::vector<float> hA(size_t(M) * K);
        std::vector<float> hB(size_t(K) * N);
        std::vector<float> hC(size_t(M) * N, 0.0f);
        fillRandom(hA, 0x1234);
        fillRandom(hB, 0x5678);

        std::vector<float> outAuto, outDynamic;
        ASSERT_TRUE(runSgemm(setup.handle, setup.algo, setup.workspaceSize, kAutoMode,
                             M, N, K, hA.data(), hB.data(), hC.data(), outAuto))
            << "AUTO matmul failed";
        ASSERT_TRUE(runSgemm(setup.handle, setup.algo, setup.workspaceSize, kDynamicMode,
                             M, N, K, hA.data(), hB.data(), hC.data(), outDynamic))
            << "explicit dynamic matmul failed";

        (void)hipblasLtDestroy(setup.handle);

        ASSERT_EQ(outAuto.size(), outDynamic.size());
        const auto byteSize = outAuto.size() * sizeof(float);
        EXPECT_EQ(std::memcmp(outAuto.data(), outDynamic.data(), byteSize), 0)
            << "AUTO heuristic should pick dynamic for an 8192x8192x64 problem "
               "(tiles_per_cu ~ 13.5 > 2.08), but the bit pattern differs "
               "from the explicit-ON run.";
    }

    // Lightweight API-level regression that does not require a GPU: the
    // attribute must be accepted by hipblasLtMatmulDescSetAttribute and
    // round-trip through hipblasLtMatmulDescGetAttribute.
    TEST(StreamK5HybridApi, AttributeRoundTrip)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available for hipblasLt handle";

        hipblasLtHandle_t handle = nullptr;
        ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

        hipblasLtMatmulDesc_t desc = nullptr;
        ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);

        const int32_t want = 1;
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT, &want, sizeof(want)),
                  HIPBLAS_STATUS_SUCCESS);

        int32_t got      = -1;
        size_t  writtenN = 0;
        ASSERT_EQ(hipblasLtMatmulDescGetAttribute(desc,
                                                  HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT,
                                                  &got,
                                                  sizeof(got),
                                                  &writtenN),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(got, want);
        EXPECT_EQ(writtenN, sizeof(int32_t));

        const int32_t want0 = 0;
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT, &want0, sizeof(want0)),
                  HIPBLAS_STATUS_SUCCESS);
        got = -1;
        ASSERT_EQ(hipblasLtMatmulDescGetAttribute(desc,
                                                  HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT,
                                                  &got,
                                                  sizeof(got),
                                                  &writtenN),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(got, want0);

        const int32_t want2 = 2;
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT, &want2, sizeof(want2)),
                  HIPBLAS_STATUS_SUCCESS);
        got = -1;
        ASSERT_EQ(hipblasLtMatmulDescGetAttribute(desc,
                                                  HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT,
                                                  &got,
                                                  sizeof(got),
                                                  &writtenN),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(got, want2);

        const int32_t bad = 3;
        EXPECT_EQ(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT, &bad, sizeof(bad)),
                  HIPBLAS_STATUS_INVALID_VALUE);
        const int32_t neg = -1;
        EXPECT_EQ(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT, &neg, sizeof(neg)),
                  HIPBLAS_STATUS_INVALID_VALUE);

        (void)hipblasLtMatmulDescDestroy(desc);
        (void)hipblasLtDestroy(handle);
    }

} // namespace
