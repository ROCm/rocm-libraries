/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

// StreamK=5 API-level tests.
//
// Tests that exercise the HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT
// attribute at the API layer:
//   - AttributeRoundTrip: set/get sanity checks (no GPU kernel required).
//   - StreamK5HybridDebug/*: end-to-end fp16 GEMM tests that enable the
//     TENSILE_DB=0x100000 diagnostic bit and capture std::cerr to verify the
//     SK5 hybrid mode selection log line for each scheduling mode.
//     Each debug test calls GTEST_SKIP() when no SK5 debug line is emitted,
//     because SK5 kernel availability depends on the loaded device library.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

// Calls Debug::Instance().reloadDebugBitsForTest() inside libhipblaslt.so,
// updating the singleton that actually gates the SK5 debug messages.  Using
// this instead of calling the method directly avoids the duplicate-singleton
// problem that arises because tensilelite-host is statically linked into
// both libhipblaslt.so and the test binary.
extern "C" void hipblaslt_debug_reload();

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

    // -----------------------------------------------------------------------
    // Attribute round-trip
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // SK5 hybrid mode debug-line helpers
    // -----------------------------------------------------------------------

    // Problem size confirmed to select an SK5 kernel on gfx950 with fp16 NN.
    // 4096³ is the bench-verified size; the debug line fires on that target.
    constexpr int64_t kM       = 4096;
    constexpr int64_t kN       = 4096;
    constexpr int64_t kK       = 4096;
    constexpr int64_t kWsBytes = 256LL * 1024 * 1024;

    // Parameters controlling a single SK5 debug-capture run.
    struct Sk5Params
    {
        int32_t schedMode     = HIPBLASLT_STREAMK_TILE_SCHEDULING_OFF;
        int32_t smCountTarget = 0;
        bool    forceEnvSet   = false;
        int     forceEnvVal   = 0;
    };

    // Runs a single fp16 NN GEMM (kM×kN×kK) with the given SK5 settings and
    // returns the text emitted to std::cerr during the heuristic + matmul calls.
    //
    // Sets TENSILE_DB=0x100000 for the duration so the SK5 debug line fires.
    // Optionally sets TENSILE_STREAMK5_FORCE_MODE. Both are restored on return.
    // Calls ADD_FAILURE() and returns an empty string on any API error.
    std::string runGemmCaptureStderr(const Sk5Params& p)
    {
        // Redirect std::cerr for the entire function scope.
        std::ostringstream    cap;
        std::streambuf* const savedCerr = std::cerr.rdbuf(cap.rdbuf());

        // Snapshot prior env values so we can restore them (not just unsetenv).
        const char*       priorDb  = std::getenv("TENSILE_DB");
        const std::string savedDb  = priorDb ? priorDb : std::string();
        const bool        hadDb    = (priorDb != nullptr);
        const char*       priorSk5 = std::getenv("TENSILE_STREAMK5_FORCE_MODE");
        const std::string savedSk5 = priorSk5 ? priorSk5 : std::string();
        const bool        hadSk5   = (priorSk5 != nullptr);

        // Environment setup — must precede any TensileLite call.
        setenv("TENSILE_DB", "0x100000", /*overwrite=*/1);
        if(p.forceEnvSet)
            setenv("TENSILE_STREAMK5_FORCE_MODE",
                   std::to_string(p.forceEnvVal).c_str(), 1);
        // Update the Debug singleton inside libhipblaslt.so.
        hipblaslt_debug_reload();

        // Called on every exit path to restore env and cerr to prior state.
        auto cleanup = [&]() {
            if(hadDb)
                setenv("TENSILE_DB", savedDb.c_str(), 1);
            else
                unsetenv("TENSILE_DB");
            if(p.forceEnvSet)
            {
                if(hadSk5)
                    setenv("TENSILE_STREAMK5_FORCE_MODE", savedSk5.c_str(), 1);
                else
                    unsetenv("TENSILE_STREAMK5_FORCE_MODE");
            }
            hipblaslt_debug_reload();
            std::cerr.rdbuf(savedCerr);
        };

        hipblasLtHandle_t handle = nullptr;
        if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
        {
            cleanup();
            ADD_FAILURE() << "hipblasLtCreate failed";
            return {};
        }

        hipStream_t stream = nullptr;
        hipStreamCreate(&stream);

        // fp16 device buffers; GEMM output correctness is not verified.
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
            return {};
        }

        // Column-major NN layouts.
        hipblasLtMatrixLayout_t matA = nullptr, matB = nullptr, matC = nullptr;
        hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, kM, kK, kM);
        hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, kK, kN, kK);
        hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, kM, kN, kM);

        // Matmul descriptor with the requested tile scheduling mode.
        hipblasLtMatmulDesc_t desc = nullptr;
        hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
        hipblasLtMatmulDescSetAttribute(desc,
                                        HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT,
                                        &p.schedMode,
                                        sizeof(p.schedMode));
        if(p.smCountTarget != 0)
            hipblasLtMatmulDescSetAttribute(desc,
                                            HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
                                            &p.smCountTarget,
                                            sizeof(p.smCountTarget));

        // Preference with a generous workspace budget.
        hipblasLtMatmulPreference_t pref    = nullptr;
        uint64_t                    wsBytes = static_cast<uint64_t>(kWsBytes);
        hipblasLtMatmulPreferenceCreate(&pref);
        hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBytes, sizeof(wsBytes));

        // Heuristic selection — the debug line may fire here for some dispatch paths.
        hipblasLtMatmulHeuristicResult_t result{};
        int                              returnedCount = 0;
        hipblasLtMatmulAlgoGetHeuristic(
            handle, desc, matA, matB, matC, matC, pref, 1, &result, &returnedCount);

        // hipblasLtMatmul is the primary site where streamK5EffectiveDynamic fires.
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
        }

        // Teardown — order mirrors creation.
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
        return cap.str();
    }

    // Sentinel present in every SK5 hybrid mode debug line.
    static const char kSK5Marker[] = "TensileLite::DEBUG: SK5 hybrid mode";

    // -----------------------------------------------------------------------
    // StreamK5HybridDebug test cases
    // -----------------------------------------------------------------------

    // Mode OFF: static sub-path, reason=api-off-static.
    TEST(StreamK5HybridDebug, OffModeSelectsStaticTile)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";

        Sk5Params p;
        p.schedMode       = HIPBLASLT_STREAMK_TILE_SCHEDULING_OFF;
        const std::string out = runGemmCaptureStderr(p);

        if(out.find(kSK5Marker) == std::string::npos)
            GTEST_SKIP() << "No SK5 kernel selected for this problem/device; "
                            "SK5 debug line not emitted";

        EXPECT_NE(out.find("reason=api-off-static"), std::string::npos)
            << "Full captured stderr:\n" << out;
        EXPECT_NE(out.find("effective=static(SK3"), std::string::npos)
            << "Full captured stderr:\n" << out;
    }

    // Mode ON: dynamic work-queue sub-path, reason=api-on.
    TEST(StreamK5HybridDebug, OnModeSelectsDynamicWorkQueue)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";

        Sk5Params p;
        p.schedMode       = HIPBLASLT_STREAMK_TILE_SCHEDULING_ON;
        const std::string out = runGemmCaptureStderr(p);

        if(out.find(kSK5Marker) == std::string::npos)
            GTEST_SKIP() << "No SK5 kernel selected for this problem/device; "
                            "SK5 debug line not emitted";

        EXPECT_NE(out.find("reason=api-on"), std::string::npos)
            << "Full captured stderr:\n" << out;
        EXPECT_NE(out.find("effective=dynamic(SK4"), std::string::npos)
            << "Full captured stderr:\n" << out;
    }

    // Mode AUTO: heuristic decides, reason=api-auto-heuristic.
    TEST(StreamK5HybridDebug, AutoModeRunsHeuristic)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";

        Sk5Params p;
        p.schedMode       = HIPBLASLT_STREAMK_TILE_SCHEDULING_AUTO;
        const std::string out = runGemmCaptureStderr(p);

        if(out.find(kSK5Marker) == std::string::npos)
            GTEST_SKIP() << "No SK5 kernel selected for this problem/device; "
                            "SK5 debug line not emitted";

        EXPECT_NE(out.find("reason=api-auto-heuristic"), std::string::npos)
            << "Full captured stderr:\n" << out;
        // AUTO may resolve to either sub-path depending on tile/CU geometry.
        const bool hasStatic  = out.find("effective=static")  != std::string::npos;
        const bool hasDynamic = out.find("effective=dynamic") != std::string::npos;
        EXPECT_TRUE(hasStatic || hasDynamic)
            << "Expected effective=static(...) or effective=dynamic(...) in:\n" << out;
    }

    // Mode OFF with smCountTarget=128: heuristic still runs, reason=api-off-heuristic.
    TEST(StreamK5HybridDebug, OffWithSmCountTargetRunsHeuristic)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";

        Sk5Params p;
        p.schedMode     = HIPBLASLT_STREAMK_TILE_SCHEDULING_OFF;
        p.smCountTarget = 128;
        const std::string out = runGemmCaptureStderr(p);

        if(out.find(kSK5Marker) == std::string::npos)
            GTEST_SKIP() << "No SK5 kernel selected for this problem/device; "
                            "SK5 debug line not emitted";

        EXPECT_NE(out.find("reason=api-off-heuristic"), std::string::npos)
            << "Full captured stderr:\n" << out;
        EXPECT_NE(out.find("smCountTarget=128"), std::string::npos)
            << "Full captured stderr:\n" << out;
    }

    // TENSILE_STREAMK5_FORCE_MODE=1 overrides API OFF: reason=force-env, effective=dynamic.
    TEST(StreamK5HybridDebug, ForceEnvOverridesOffMode)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";

        Sk5Params p;
        p.schedMode   = HIPBLASLT_STREAMK_TILE_SCHEDULING_OFF;
        p.forceEnvSet = true;
        p.forceEnvVal = 1;
        const std::string out = runGemmCaptureStderr(p);

        if(out.find(kSK5Marker) == std::string::npos)
            GTEST_SKIP() << "No SK5 kernel selected for this problem/device; "
                            "SK5 debug line not emitted";

        EXPECT_NE(out.find("reason=force-env"), std::string::npos)
            << "Full captured stderr:\n" << out;
        EXPECT_NE(out.find("effective=dynamic(SK4"), std::string::npos)
            << "Full captured stderr:\n" << out;
    }

} // namespace
