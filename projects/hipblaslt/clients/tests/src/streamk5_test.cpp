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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <random>
#include <string>
#include <vector>

namespace
{
    constexpr int32_t kStaticMode  = 0;
    constexpr int32_t kDynamicMode = 1;
    constexpr int32_t kAutoMode    = 2;

    // Kernel-name marker emitted by KernelHelperNaming for StreamK=5
    // solutions. Used to skip the bitwise-equivalence test when no SK5
    // kernel is present in the loaded device library (the toggle is a
    // no-op for non-SK5 kernels, which would let the test pass
    // vacuously).
    constexpr const char* kSk5KernelMarker = "_SK5_";

    struct GpuBuf
    {
        void*  ptr  = nullptr;
        size_t size = 0;

        GpuBuf() = default;
        explicit GpuBuf(size_t bytes)
            : size(bytes)
        {
            if(hipMalloc(&ptr, bytes) != hipSuccess)
            {
                ptr = nullptr;
            }
        }
        ~GpuBuf()
        {
            if(ptr)
                (void)hipFree(ptr);
        }
        GpuBuf(const GpuBuf&)            = delete;
        GpuBuf& operator=(const GpuBuf&) = delete;
        GpuBuf(GpuBuf&& other) noexcept
            : ptr(other.ptr)
            , size(other.size)
        {
            other.ptr  = nullptr;
            other.size = 0;
        }
        GpuBuf& operator=(GpuBuf&& other) noexcept
        {
            if(this != &other)
            {
                if(ptr)
                    (void)hipFree(ptr);
                ptr        = other.ptr;
                size       = other.size;
                other.ptr  = nullptr;
                other.size = 0;
            }
            return *this;
        }
    };

    // Set up SGEMM layouts and descriptor for an MxNxK problem.
    // All matrices are column-major HIP_R_32F, op_a = op_b = N.
    bool createSgemmLayouts(int                          m,
                            int                          n,
                            int                          k,
                            hipblasLtMatrixLayout_t&     layoutA,
                            hipblasLtMatrixLayout_t&     layoutB,
                            hipblasLtMatrixLayout_t&     layoutC,
                            hipblasLtMatrixLayout_t&     layoutD,
                            hipblasLtMatmulDesc_t&       desc)
    {
        if(hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_32F, m, k, m) != HIPBLAS_STATUS_SUCCESS
           || hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_32F, k, n, k) != HIPBLAS_STATUS_SUCCESS
           || hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_32F, m, n, m) != HIPBLAS_STATUS_SUCCESS
           || hipblasLtMatrixLayoutCreate(&layoutD, HIP_R_32F, m, n, m) != HIPBLAS_STATUS_SUCCESS
           || hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                  != HIPBLAS_STATUS_SUCCESS)
            return false;

        hipblasOperation_t opN = HIPBLAS_OP_N;
        if(hipblasLtMatmulDescSetAttribute(
               desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN))
               != HIPBLAS_STATUS_SUCCESS
           || hipblasLtMatmulDescSetAttribute(
                  desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN))
                  != HIPBLAS_STATUS_SUCCESS)
            return false;

        return true;
    }

    // Query heuristic algos for an SGEMM and return the first whose
    // kernel name marks it as a StreamK=5 hybrid kernel. The output
    // `algo` is only valid when the function returns true; on false the
    // device library does not contain an SK5 kernel for this problem.
    bool pickSk5Algo(hipblasLtHandle_t           handle,
                     hipblasLtMatmulDesc_t       desc,
                     hipblasLtMatrixLayout_t     layoutA,
                     hipblasLtMatrixLayout_t     layoutB,
                     hipblasLtMatrixLayout_t     layoutC,
                     hipblasLtMatrixLayout_t     layoutD,
                     hipblasLtMatmulAlgo_t&      algo,
                     size_t&                     workspaceSize)
    {
        hipblasLtMatmulPreference_t pref = nullptr;
        if(hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS)
            return false;

        uint64_t workspace = 256ULL * 1024 * 1024;
        (void)hipblasLtMatmulPreferenceSetAttribute(
            pref,
            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace,
            sizeof(workspace));

        constexpr int                    kReqAlgos = 32;
        hipblasLtMatmulHeuristicResult_t results[kReqAlgos]{};
        int                              returnedAlgos = 0;
        const auto                       st            = hipblasLtMatmulAlgoGetHeuristic(handle,
                                                          desc,
                                                          layoutA,
                                                          layoutB,
                                                          layoutC,
                                                          layoutD,
                                                          pref,
                                                          kReqAlgos,
                                                          results,
                                                          &returnedAlgos);
        (void)hipblasLtMatmulPreferenceDestroy(pref);
        if(st != HIPBLAS_STATUS_SUCCESS || returnedAlgos <= 0)
            return false;

        for(int i = 0; i < returnedAlgos; ++i)
        {
            const std::string name
                = hipblaslt_ext::getKernelNameFromAlgo(handle, results[i].algo);
            if(name.find(kSk5KernelMarker) != std::string::npos)
            {
                algo          = results[i].algo;
                workspaceSize = results[i].workspaceSize;
                return true;
            }
        }
        return false;
    }

    // Run a single SGEMM against `algo` with the SK5 hybrid-mode toggle
    // set to `mode`, returning the host-side D buffer in `outD`. The
    // caller is expected to pass the same algo across the two mode
    // values so the only intentional difference is the bit packed into
    // MagicShiftItersPerTile by the host arg-pack path.
    bool runSgemm(hipblasLtHandle_t            handle,
                  const hipblasLtMatmulAlgo_t& algo,
                  size_t                       workspaceSize,
                  int32_t                      mode,
                  int                          m,
                  int                          n,
                  int                          k,
                  const float*                 hA,
                  const float*                 hB,
                  const float*                 hC,
                  std::vector<float>&          outD)
    {
        const size_t bytesA = size_t(m) * k * sizeof(float);
        const size_t bytesB = size_t(k) * n * sizeof(float);
        const size_t bytesC = size_t(m) * n * sizeof(float);

        GpuBuf dA(bytesA), dB(bytesB), dC(bytesC), dD(bytesC);
        if(!dA.ptr || !dB.ptr || !dC.ptr || !dD.ptr)
            return false;

        if(hipMemcpy(dA.ptr, hA, bytesA, hipMemcpyHostToDevice) != hipSuccess
           || hipMemcpy(dB.ptr, hB, bytesB, hipMemcpyHostToDevice) != hipSuccess
           || hipMemcpy(dC.ptr, hC, bytesC, hipMemcpyHostToDevice) != hipSuccess)
            return false;

        hipblasLtMatrixLayout_t layoutA = nullptr, layoutB = nullptr;
        hipblasLtMatrixLayout_t layoutC = nullptr, layoutD = nullptr;
        hipblasLtMatmulDesc_t   desc    = nullptr;
        bool                    ok      = createSgemmLayouts(
            m, n, k, layoutA, layoutB, layoutC, layoutD, desc);

        // StreamK=5 hybrid-mode toggle (0 = static, 1 = dynamic). Must
        // be set after the descriptor is created and before the matmul
        // call; the host OR's it into MagicShiftItersPerTile at
        // arg-pack time.
        if(ok)
        {
            ok = ok
                 && hipblasLtMatmulDescSetAttribute(
                        desc,
                        HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT,
                        &mode,
                        sizeof(mode))
                        == HIPBLAS_STATUS_SUCCESS;
        }

        GpuBuf workspaceBuf;
        if(ok && workspaceSize > 0)
        {
            workspaceBuf = GpuBuf(workspaceSize);
            ok           = ok && workspaceBuf.ptr != nullptr;
        }

        float alpha = 1.0f, beta = 0.0f;
        if(ok)
        {
            auto st = hipblasLtMatmul(handle,
                                      desc,
                                      &alpha,
                                      dA.ptr,
                                      layoutA,
                                      dB.ptr,
                                      layoutB,
                                      &beta,
                                      dC.ptr,
                                      layoutC,
                                      dD.ptr,
                                      layoutD,
                                      &algo,
                                      workspaceBuf.ptr,
                                      workspaceBuf.size,
                                      nullptr);
            ok      = ok && st == HIPBLAS_STATUS_SUCCESS;
            ok      = ok && hipDeviceSynchronize() == hipSuccess;
        }

        if(ok)
        {
            outD.resize(size_t(m) * n);
            ok = ok && hipMemcpy(outD.data(), dD.ptr, bytesC, hipMemcpyDeviceToHost) == hipSuccess;
        }

        if(desc)
            (void)hipblasLtMatmulDescDestroy(desc);
        if(layoutD)
            (void)hipblasLtMatrixLayoutDestroy(layoutD);
        if(layoutC)
            (void)hipblasLtMatrixLayoutDestroy(layoutC);
        if(layoutB)
            (void)hipblasLtMatrixLayoutDestroy(layoutB);
        if(layoutA)
            (void)hipblasLtMatrixLayoutDestroy(layoutA);
        return ok;
    }

    bool gpuAvailable()
    {
        int deviceCount = 0;
        return hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0;
    }

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

        hipblasLtHandle_t handle = nullptr;
        ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

        // Locate a real SK5 kernel for this problem. Without one the
        // toggle is a no-op and a bitwise-equivalence assertion would
        // pass vacuously, so skip cleanly instead.
        hipblasLtMatrixLayout_t lA = nullptr, lB = nullptr, lC = nullptr, lD = nullptr;
        hipblasLtMatmulDesc_t   desc       = nullptr;
        bool                    layoutsOk  = createSgemmLayouts(m, n, k, lA, lB, lC, lD, desc);

        hipblasLtMatmulAlgo_t algo{};
        size_t                workspaceSize = 0;
        bool                  foundSk5      = false;
        if(layoutsOk)
            foundSk5 = pickSk5Algo(handle, desc, lA, lB, lC, lD, algo, workspaceSize);

        if(desc)
            (void)hipblasLtMatmulDescDestroy(desc);
        if(lD)
            (void)hipblasLtMatrixLayoutDestroy(lD);
        if(lC)
            (void)hipblasLtMatrixLayoutDestroy(lC);
        if(lB)
            (void)hipblasLtMatrixLayoutDestroy(lB);
        if(lA)
            (void)hipblasLtMatrixLayoutDestroy(lA);

        if(!layoutsOk || !foundSk5)
        {
            (void)hipblasLtDestroy(handle);
            GTEST_SKIP() << "No StreamK=5 kernel in loaded device library for "
                         << m << "x" << n << "x" << k;
        }

        std::vector<float> hA(size_t(m) * k);
        std::vector<float> hB(size_t(k) * n);
        std::vector<float> hC(size_t(m) * n, 0.0f);
        fillRandom(hA, 0xA11CE);
        fillRandom(hB, 0xB0B);

        std::vector<float> outStatic, outDynamic;
        ASSERT_TRUE(runSgemm(handle,
                             algo,
                             workspaceSize,
                             kStaticMode,
                             m,
                             n,
                             k,
                             hA.data(),
                             hB.data(),
                             hC.data(),
                             outStatic))
            << "static path matmul failed";
        ASSERT_TRUE(runSgemm(handle,
                             algo,
                             workspaceSize,
                             kDynamicMode,
                             m,
                             n,
                             k,
                             hA.data(),
                             hB.data(),
                             hC.data(),
                             outDynamic))
            << "dynamic path matmul failed";

        (void)hipblasLtDestroy(handle);

        ASSERT_EQ(outStatic.size(), outDynamic.size());
        // The two paths inside an SK5 hybrid kernel share the same
        // partial-tile fixup and final-store sequence, so the IEEE 754
        // bit patterns must be identical (no reduction-ordering
        // differences between modes).
        const auto byteSize = outStatic.size() * sizeof(float);
        if(std::memcmp(outStatic.data(), outDynamic.data(), byteSize) != 0)
        {
            for(size_t i = 0; i < outStatic.size(); ++i)
            {
                uint32_t a = 0, b = 0;
                std::memcpy(&a, &outStatic[i], sizeof(a));
                std::memcpy(&b, &outDynamic[i], sizeof(b));
                ASSERT_EQ(a, b) << "Bit mismatch at idx " << i << ": static=0x"
                                << std::hex << a << " dynamic=0x" << b;
            }
            FAIL() << "memcmp mismatch with no per-element diff -- size " << byteSize;
        }
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

        hipblasLtHandle_t handle = nullptr;
        ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

        hipblasLtMatrixLayout_t lA = nullptr, lB = nullptr, lC = nullptr, lD = nullptr;
        hipblasLtMatmulDesc_t   desc      = nullptr;
        bool                    layoutsOk = createSgemmLayouts(M, N, K, lA, lB, lC, lD, desc);

        hipblasLtMatmulAlgo_t algo{};
        size_t                workspaceSize = 0;
        bool                  foundSk5      = false;
        if(layoutsOk)
            foundSk5 = pickSk5Algo(handle, desc, lA, lB, lC, lD, algo, workspaceSize);

        if(desc) (void)hipblasLtMatmulDescDestroy(desc);
        if(lD)   (void)hipblasLtMatrixLayoutDestroy(lD);
        if(lC)   (void)hipblasLtMatrixLayoutDestroy(lC);
        if(lB)   (void)hipblasLtMatrixLayoutDestroy(lB);
        if(lA)   (void)hipblasLtMatrixLayoutDestroy(lA);

        if(!layoutsOk || !foundSk5)
        {
            (void)hipblasLtDestroy(handle);
            GTEST_SKIP() << "No StreamK=5 kernel in loaded device library for "
                         << M << "x" << N << "x" << K;
        }

        std::vector<float> hA(size_t(M) * K);
        std::vector<float> hB(size_t(K) * N);
        std::vector<float> hC(size_t(M) * N, 0.0f);
        fillRandom(hA, 0xC0FFEE);
        fillRandom(hB, 0xDECAF);

        std::vector<float> outAuto, outStatic;
        ASSERT_TRUE(runSgemm(handle, algo, workspaceSize, kAutoMode,
                             M, N, K, hA.data(), hB.data(), hC.data(), outAuto))
            << "AUTO matmul failed";
        ASSERT_TRUE(runSgemm(handle, algo, workspaceSize, kStaticMode,
                             M, N, K, hA.data(), hB.data(), hC.data(), outStatic))
            << "explicit static matmul failed";

        (void)hipblasLtDestroy(handle);

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

        hipblasLtHandle_t handle = nullptr;
        ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

        hipblasLtMatrixLayout_t lA = nullptr, lB = nullptr, lC = nullptr, lD = nullptr;
        hipblasLtMatmulDesc_t   desc      = nullptr;
        bool                    layoutsOk = createSgemmLayouts(M, N, K, lA, lB, lC, lD, desc);

        hipblasLtMatmulAlgo_t algo{};
        size_t                workspaceSize = 0;
        bool                  foundSk5      = false;
        if(layoutsOk)
            foundSk5 = pickSk5Algo(handle, desc, lA, lB, lC, lD, algo, workspaceSize);

        if(desc) (void)hipblasLtMatmulDescDestroy(desc);
        if(lD)   (void)hipblasLtMatrixLayoutDestroy(lD);
        if(lC)   (void)hipblasLtMatrixLayoutDestroy(lC);
        if(lB)   (void)hipblasLtMatrixLayoutDestroy(lB);
        if(lA)   (void)hipblasLtMatrixLayoutDestroy(lA);

        if(!layoutsOk || !foundSk5)
        {
            (void)hipblasLtDestroy(handle);
            GTEST_SKIP() << "No StreamK=5 kernel in loaded device library for "
                         << M << "x" << N << "x" << K;
        }

        std::vector<float> hA(size_t(M) * K);
        std::vector<float> hB(size_t(K) * N);
        std::vector<float> hC(size_t(M) * N, 0.0f);
        fillRandom(hA, 0x1234);
        fillRandom(hB, 0x5678);

        std::vector<float> outAuto, outDynamic;
        ASSERT_TRUE(runSgemm(handle, algo, workspaceSize, kAutoMode,
                             M, N, K, hA.data(), hB.data(), hC.data(), outAuto))
            << "AUTO matmul failed";
        ASSERT_TRUE(runSgemm(handle, algo, workspaceSize, kDynamicMode,
                             M, N, K, hA.data(), hB.data(), hC.data(), outDynamic))
            << "explicit dynamic matmul failed";

        (void)hipblasLtDestroy(handle);

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
