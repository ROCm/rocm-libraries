/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

// StreamK Synchronizer self-clean regression tests.
//
// Runs the SK5 hybrid GEMM in both its dynamic (SK4) and static (SK3)
// sub-paths with HIPBLASLT_CHECK_STREAMK_SYNC=1 and asserts the shared
// Synchronizer work-queue buffer is left clean after each launch. Catches
// StreamK kernels that don't reset their work-queue state on exit, which
// would otherwise corrupt the next launch reusing the same buffer.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

// See streamk5_test.cpp for why hipblaslt_debug_reload is needed: it updates
// the Debug singleton inside libhipblaslt.so after an in-process setenv().
extern "C" void hipblaslt_debug_reload();
extern "C" bool hipblaslt_debug_streamk_sync_was_dirty(hipblasLtHandle_t handle);

#ifdef WIN32
int setenv(const char* name, const char* value, int overwrite)
{
    return _putenv_s(name, value);
}

int unsetenv(const char* name)
{
    return _putenv_s(name, "");
}
#endif

namespace
{
    inline bool gpuAvailable()
    {
        int deviceCount = 0;
        return hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0;
    }

    // Problem shape confirmed (via the field reproducer this test is based
    // on) to select an SK5 kernel whose dynamic sub-path leaves the
    // Synchronizer dirty pre-fix: bf16 NN, M128 x N73390 x K160.
    constexpr int64_t kM       = 128;
    constexpr int64_t kN       = 73390;
    constexpr int64_t kK       = 160;
    constexpr int64_t kWsBytes = 256LL * 1024 * 1024;

    static const char kSK5Marker[] = "TensileLite::DEBUG: SK5 hybrid mode";

    struct SyncCheckResult
    {
        bool ranSK5 = false; // whether an SK5 kernel was actually dispatched
        bool dirty  = false; // whether the Synchronizer was left dirty
    };

    // Runs one bf16 NN GEMM (kM x kN x kK) with the given tile-scheduling
    // mode and TENSILE_STREAMK5_FORCE_MODE value under
    // HIPBLASLT_CHECK_STREAMK_SYNC=1. Calls ADD_FAILURE() on any API error.
    // HIPBLASLT_CHECK_STREAMK_SYNC is read once at handle creation, so it
    // must be set before hipblasLtCreate.
    SyncCheckResult runGemmAndCheckSync(int32_t schedMode, int forceMode)
    {
        std::ostringstream    cap;
        std::streambuf* const savedCerr = std::cerr.rdbuf(cap.rdbuf());

        const char*       priorDb  = std::getenv("TENSILE_DB");
        const std::string savedDb  = priorDb ? priorDb : std::string();
        const bool        hadDb    = (priorDb != nullptr);
        const char*       priorSk5 = std::getenv("TENSILE_STREAMK5_FORCE_MODE");
        const std::string savedSk5 = priorSk5 ? priorSk5 : std::string();
        const bool        hadSk5   = (priorSk5 != nullptr);
        setenv("TENSILE_DB", "0x100000", /*overwrite=*/1);
        setenv("TENSILE_STREAMK5_FORCE_MODE", std::to_string(forceMode).c_str(), 1);
        setenv("HIPBLASLT_CHECK_STREAMK_SYNC", "1", /*overwrite=*/1);
        hipblaslt_debug_reload();

        auto cleanup = [&]() {
            std::cerr.rdbuf(savedCerr);
            if(hadDb)
                setenv("TENSILE_DB", savedDb.c_str(), 1);
            else
                unsetenv("TENSILE_DB");
            if(hadSk5)
                setenv("TENSILE_STREAMK5_FORCE_MODE", savedSk5.c_str(), 1);
            else
                unsetenv("TENSILE_STREAMK5_FORCE_MODE");
            unsetenv("HIPBLASLT_CHECK_STREAMK_SYNC");
            hipblaslt_debug_reload();
        };

        SyncCheckResult res;

        hipblasLtHandle_t handle = nullptr;
        if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
        {
            cleanup();
            ADD_FAILURE() << "hipblasLtCreate failed";
            return res;
        }

        hipStream_t stream = nullptr;
        hipStreamCreate(&stream);

        const size_t elemBytes = sizeof(uint16_t);
        void *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_ws = nullptr;
        if(hipMalloc(&d_a, static_cast<size_t>(kM * kK) * elemBytes) != hipSuccess
           || hipMalloc(&d_b, static_cast<size_t>(kK * kN) * elemBytes) != hipSuccess
           || hipMalloc(&d_c, static_cast<size_t>(kM * kN) * elemBytes) != hipSuccess
           || hipMalloc(&d_ws, static_cast<size_t>(kWsBytes)) != hipSuccess)
        {
            hipFree(d_a);
            hipFree(d_b);
            hipFree(d_c);
            hipFree(d_ws);
            hipStreamDestroy(stream);
            hipblasLtDestroy(handle);
            cleanup();
            ADD_FAILURE() << "hipMalloc failed (insufficient device memory?)";
            return res;
        }

        hipblasLtMatrixLayout_t matA = nullptr, matB = nullptr, matC = nullptr;
        hipblasLtMatrixLayoutCreate(&matA, HIP_R_16BF, kM, kK, kM);
        hipblasLtMatrixLayoutCreate(&matB, HIP_R_16BF, kK, kN, kK);
        hipblasLtMatrixLayoutCreate(&matC, HIP_R_16BF, kM, kN, kM);

        hipblasLtMatmulDesc_t desc = nullptr;
        hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
        hipblasLtMatmulDescSetAttribute(
            desc, HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT, &schedMode, sizeof(schedMode));

        hipblasLtMatmulPreference_t pref    = nullptr;
        uint64_t                    wsBytes = static_cast<uint64_t>(kWsBytes);
        hipblasLtMatmulPreferenceCreate(&pref);
        hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBytes, sizeof(wsBytes));

        hipblasLtMatmulHeuristicResult_t result{};
        int                              returnedCount = 0;
        hipblasLtMatmulAlgoGetHeuristic(
            handle, desc, matA, matB, matC, matC, pref, 1, &result, &returnedCount);

        if(returnedCount > 0)
        {
            float alpha = 1.f, beta = 0.f;
            hipblasLtMatmul(handle,
                            desc,
                            &alpha,
                            d_a,
                            matA,
                            d_b,
                            matB,
                            &beta,
                            d_c,
                            matC,
                            d_c,
                            matC,
                            &result.algo,
                            d_ws,
                            result.workspaceSize,
                            stream);
            hipStreamSynchronize(stream);
            res.ranSK5 = (cap.str().find(kSK5Marker) != std::string::npos);
        }
        res.dirty = hipblaslt_debug_streamk_sync_was_dirty(handle);

        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatmulDescDestroy(desc);
        hipblasLtMatrixLayoutDestroy(matC);
        hipblasLtMatrixLayoutDestroy(matB);
        hipblasLtMatrixLayoutDestroy(matA);
        hipFree(d_ws);
        hipFree(d_c);
        hipFree(d_b);
        hipFree(d_a);
        hipStreamDestroy(stream);
        hipblasLtDestroy(handle);

        cleanup();
        return res;
    }

    // Mode ON: SK4 dynamic per-XCD work-queue sub-path -- the sub-path that
    // was found leaving the Synchronizer dirty.
    TEST(StreamKSynchronizer, DynamicModeLeavesBufferClean)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";

        const SyncCheckResult res
            = runGemmAndCheckSync(HIPBLASLT_STREAMK_TILE_SCHEDULING_ON, /*forceMode=*/1);
        if(!res.ranSK5)
            GTEST_SKIP() << "No SK5 kernel selected for this problem/device";

        EXPECT_FALSE(res.dirty) << "StreamK Synchronizer left dirty after a dynamic launch";
    }

    // Mode OFF: SK3 static sub-path.
    TEST(StreamKSynchronizer, StaticModeLeavesBufferClean)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";

        const SyncCheckResult res
            = runGemmAndCheckSync(HIPBLASLT_STREAMK_TILE_SCHEDULING_OFF, /*forceMode=*/0);
        if(!res.ranSK5)
            GTEST_SKIP() << "No SK5 kernel selected for this problem/device";

        EXPECT_FALSE(res.dirty) << "StreamK Synchronizer left dirty after a static launch";
    }

} // namespace
