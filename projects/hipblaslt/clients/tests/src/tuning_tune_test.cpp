// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tune mode tests.
//
// Everything here drives the public API with HIPBLASLT_TUNING_MODE and
// HIPBLASLT_TUNING_CACHE_PATH, and asserts on the library's own counters and on
// the rows it writes rather than on log text. Replay of existing files, in
// cache mode and through HIPBLASLT_TUNING_OVERRIDE_FILE, is tested in
// tuning_cache_test.cpp, and the file format and store without a device in
// tuning_store_test.cpp.
//
// The mode switch is latched on first use and the loaded-path set lives for the
// process, so each test starts by calling hipblaslt_tuning_reset_for_test().
// Without it, only the first mode set in the binary would ever take effect.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <streambuf>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

extern "C" void   hipblaslt_tuning_reset_for_test();
extern "C" int    hipblaslt_tuning_last_launch_for_test();
extern "C" void   hipblaslt_tuning_inject_failure_for_test(int stage);
extern "C" size_t hipblaslt_tuning_stats_for_test(uint64_t* values, size_t count);
extern "C" size_t hipblaslt_tuning_tensor_span_for_test(size_t elementSize,
                                                        size_t rows,
                                                        size_t cols,
                                                        size_t colStride,
                                                        size_t batchCount,
                                                        size_t batchStride,
                                                        int*   expanded);
extern "C" size_t hipblaslt_tuning_scratch_plan_for_test(void*       handle,
                                                         void*       matmulDesc,
                                                         void*       matA,
                                                         void*       matB,
                                                         void*       matC,
                                                         void*       matD,
                                                         const void* A,
                                                         const void* B,
                                                         const void* C,
                                                         void*       D,
                                                         float       beta,
                                                         size_t      workspaceBytes,
                                                         size_t      rotatingBytes,
                                                         size_t      capBytes,
                                                         int32_t     coldIters,
                                                         int32_t     hotIters,
                                                         uint64_t*   values,
                                                         size_t      count);
extern "C" size_t hipblaslt_tuning_candidates_for_test(void*  handle,
                                                       void*  matmulDesc,
                                                       void*  matA,
                                                       void*  matB,
                                                       void*  matC,
                                                       void*  matD,
                                                       size_t workspaceBytes,
                                                       int*   indexes,
                                                       size_t count);

#ifdef WIN32
static int setenv(const char* name, const char* value, int overwrite)
{
    return _putenv_s(name, value);
}
static int unsetenv(const char* name)
{
    return _putenv_s(name, "");
}
#endif

namespace
{
    bool gpuAvailable()
    {
        int deviceCount = 0;
        return hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0;
    }

    /**
     * The library's tallies, in the order hipblaslt_tuning_stats_for_test
     * documents. The counters count lookups; attempts counts searches that
     * reached tuning-start; shapes, matched, fellback and tunedShapes count
     * distinct shapes, the way the summary line does.
     */
    struct Counters
    {
        uint64_t loaded = 0, hits = 0, misses = 0, invalidated = 0;
        uint64_t shapes = 0, matched = 0, fellback = 0;
        uint64_t tuned = 0, skipped = 0, attempts = 0, tunedShapes = 0;
    };

    Counters counters()
    {
        uint64_t     v[11] = {};
        const size_t known = hipblaslt_tuning_stats_for_test(v, 11);
        EXPECT_GE(known, 11u) << "the library reports fewer tallies than this reads";

        Counters c;
        c.loaded      = v[0];
        c.hits        = v[1];
        c.misses      = v[2];
        c.invalidated = v[3];
        c.shapes      = v[4];
        c.matched     = v[5];
        c.fellback    = v[6];
        c.tuned       = v[7];
        c.skipped     = v[8];
        c.attempts    = v[9];
        c.tunedShapes = v[10];
        return c;
    }

    std::string tempCachePath(const char* stem)
    {
        std::ostringstream oss;
#ifdef WIN32
        const auto pid = _getpid();
#else
        const auto pid = getpid();
#endif
        oss << ::testing::TempDir() << "hipblaslt_" << stem << "_" << static_cast<long long>(pid)
            << ".tuning";
        return oss.str();
    }

    void enterMode(const char* mode, const std::string& path)
    {
        if(mode)
            setenv("HIPBLASLT_TUNING_MODE", mode, 1);
        else
            unsetenv("HIPBLASLT_TUNING_MODE");

        if(path.empty())
            unsetenv("HIPBLASLT_TUNING_CACHE_PATH");
        else
            setenv("HIPBLASLT_TUNING_CACHE_PATH", path.c_str(), 1);

        // These tests check the cache mechanism, not tuning quality, so keep the
        // search small. The shipping defaults measure every kernel over 2000
        // launches apiece, which costs minutes across the suite for no extra
        // coverage here.
        setenv("HIPBLASLT_TUNING_ALL_KERNELS", "0", 1);
        setenv("HIPBLASLT_TUNING_MAX_CANDIDATES", "16", 1);
        setenv("HIPBLASLT_TUNING_COLD_ITERS", "2", 1);
        setenv("HIPBLASLT_TUNING_HOT_ITERS", "5", 1);
        setenv("HIPBLASLT_TUNING_BUDGET_MS_PER_SHAPE", "0", 1);
        setenv("HIPBLASLT_TUNING_SCRATCH_MAX_BYTES", "1073741824", 1);

        // Left at the shipping default so the flush path is exercised, but the
        // per-device calibration burst is the dominant cost at these tiny
        // iteration counts, so it is measured once and reused across tests.
        setenv("HIPBLASLT_TUNING_FLUSH_ICACHE", "1", 1);

        // Large enough to rotate over several blocks rather than collapsing to
        // one, so the seeding and per-block pointer arithmetic are exercised;
        // the launch counts above bound the rotation at five blocks. Still far
        // below the 512 MiB default.
        setenv("HIPBLASLT_TUNING_ROTATING_MB", "160", 1);

        hipblaslt_tuning_reset_for_test();
    }

    /** fp16 1.0 and the exact fp16 encoding of a small whole number. */
    uint16_t halfBits(float value)
    {
        const _Float16 h = static_cast<_Float16>(value);
        uint16_t       bits;
        std::memcpy(&bits, &h, sizeof(bits));
        return bits;
    }

    /**
     * One fp16 GEMM through the public C API, heuristic then matmul.
     *
     * verifyProduct fills A and B with ones so the result is a real product
     * rather than zeros, and checks it. Without it these tests pass even if the
     * tuned winner is never launched, because zero inputs make every possible
     * answer identical.
     */
    // selectedIndex reports the solution the heuristic handed back, which in
    // cache mode is the entry replay chose. Counters say a key matched; only the
    // index says which of its rows won.
    bool runGemm(int64_t m,
                 int64_t n,
                 int64_t k,
                 float   betaValue     = 0.0f,
                 bool    inPlace       = true,
                 bool    verifyFirst   = false,
                 bool    verifyProduct = false,
                 int*    selectedIndex = nullptr)
    {
        hipblasLtHandle_t handle = nullptr;
        if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
            return false;

        void*        dA      = nullptr;
        void*        dB      = nullptr;
        void*        dC      = nullptr;
        void*        dD      = nullptr;
        void*        dWs     = nullptr;
        const size_t wsBytes = 32 * 1024 * 1024;

        bool ok = hipMalloc(&dA, m * k * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dB, k * n * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dC, m * n * sizeof(uint16_t)) == hipSuccess
                  && (inPlace || hipMalloc(&dD, m * n * sizeof(uint16_t)) == hipSuccess)
                  && hipMalloc(&dWs, wsBytes) == hipSuccess;

        if(ok)
        {
            ok = hipMemset(dA, 0, m * k * sizeof(uint16_t)) == hipSuccess
                 && hipMemset(dB, 0, k * n * sizeof(uint16_t)) == hipSuccess
                 && hipMemset(dC, 0, m * n * sizeof(uint16_t)) == hipSuccess
                 && (inPlace || hipMemset(dD, 0, m * n * sizeof(uint16_t)) == hipSuccess);
            if(ok && verifyFirst)
            {
                // IEEE fp16 1.0. A and B are zero, so beta=2 must turn this
                // element into fp16 2.0 (0x4000) in the caller's final launch.
                const uint16_t one = 0x3c00;
                ok = hipMemcpy(dC, &one, sizeof(one), hipMemcpyHostToDevice) == hipSuccess;
            }
            if(ok && verifyProduct)
            {
                const std::vector<uint16_t> onesA(static_cast<size_t>(m * k), halfBits(1.0f));
                const std::vector<uint16_t> onesB(static_cast<size_t>(k * n), halfBits(1.0f));
                ok = hipMemcpy(
                         dA, onesA.data(), onesA.size() * sizeof(uint16_t), hipMemcpyHostToDevice)
                         == hipSuccess
                     && hipMemcpy(dB,
                                  onesB.data(),
                                  onesB.size() * sizeof(uint16_t),
                                  hipMemcpyHostToDevice)
                            == hipSuccess;
            }
        }

        hipblasLtMatrixLayout_t     layoutA = nullptr, layoutB = nullptr, layoutC = nullptr;
        hipblasLtMatmulDesc_t       desc = nullptr;
        hipblasLtMatmulPreference_t pref = nullptr;

        if(ok)
        {
            ok = hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_16F, m, k, m) == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_16F, k, n, k)
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_16F, m, n, m)
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS;
        }

        if(ok)
        {
            const uint64_t maxWs = wsBytes;
            ok                   = hipblasLtMatmulPreferenceSetAttribute(
                     pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs))
                 == HIPBLAS_STATUS_SUCCESS;
        }

        if(ok)
        {
            hipblasLtMatmulHeuristicResult_t heuristic[1];
            int                              returned = 0;
            ok                                        = hipblasLtMatmulAlgoGetHeuristic(handle,
                                                 desc,
                                                 layoutA,
                                                 layoutB,
                                                 layoutC,
                                                 layoutC,
                                                 pref,
                                                 1,
                                                 heuristic,
                                                 &returned)
                     == HIPBLAS_STATUS_SUCCESS
                 && returned > 0;

            if(ok && selectedIndex)
                *selectedIndex = hipblaslt_ext::getIndexFromAlgo(heuristic[0].algo);

            if(ok)
            {
                const float alpha = 1.0f;
                ok                = hipblasLtMatmul(handle,
                                     desc,
                                     &alpha,
                                     dA,
                                     layoutA,
                                     dB,
                                     layoutB,
                                     &betaValue,
                                     dC,
                                     layoutC,
                                     inPlace ? dC : dD,
                                     layoutC,
                                     &heuristic[0].algo,
                                     dWs,
                                     wsBytes,
                                     nullptr)
                     == HIPBLAS_STATUS_SUCCESS;
                ok = ok && hipDeviceSynchronize() == hipSuccess;
                if(ok && verifyFirst)
                {
                    uint16_t first = 0;
                    ok = hipMemcpy(&first, inPlace ? dC : dD, sizeof(first), hipMemcpyDeviceToHost)
                             == hipSuccess
                         && first == 0x4000;
                }
                if(ok && verifyProduct)
                {
                    // Every element is a sum of k products of one, accumulated
                    // in fp32, so the fp16 result is exactly k for the small k
                    // used here.
                    std::vector<uint16_t> out(static_cast<size_t>(m * n), 0);
                    ok = hipMemcpy(out.data(),
                                   inPlace ? dC : dD,
                                   out.size() * sizeof(uint16_t),
                                   hipMemcpyDeviceToHost)
                         == hipSuccess;

                    const uint16_t expected = halfBits(static_cast<float>(k));
                    for(size_t i = 0; ok && i < out.size(); i++)
                        ok = out[i] == expected;
                }
            }
        }

        if(pref)
            hipblasLtMatmulPreferenceDestroy(pref);
        if(desc)
            hipblasLtMatmulDescDestroy(desc);
        if(layoutC)
            hipblasLtMatrixLayoutDestroy(layoutC);
        if(layoutB)
            hipblasLtMatrixLayoutDestroy(layoutB);
        if(layoutA)
            hipblasLtMatrixLayoutDestroy(layoutA);
        static_cast<void>(hipFree(dWs));
        static_cast<void>(hipFree(dD));
        static_cast<void>(hipFree(dC));
        static_cast<void>(hipFree(dB));
        static_cast<void>(hipFree(dA));
        hipblasLtDestroy(handle);

        return ok;
    }

    /** Where runGemmWith takes the algo it hands hipblasLtMatmul from. */
    enum class AlgoFrom
    {
        Heuristic, // the top heuristic result, queried just before the call
        Index, // a given solution index, the way a caller reuses one algo
        Null, // no algo, which leaves the choice to the library
    };

    /**
     * One fp16 GEMM with separate C and D, reporting the solution index the
     * library actually launched.
     *
     * The heuristic's answer and the counters only say which kernel the cache
     * offered. What reaches dispatch is what an explicit algo or a null one has
     * to be judged on.
     *
     * verifyProduct fills A and B with ones and checks every element of D is k,
     * which is what proves the launched kernel ran on the caller's buffers.
     * streamKMode, when not negative, is set as the descriptor's stream-K tile
     * scheduling mode.
     */
    bool runGemmWith(int64_t            m,
                     int64_t            n,
                     int64_t            k,
                     AlgoFrom           from,
                     int                index,
                     int*               launched,
                     bool               verifyProduct = false,
                     hipblasOperation_t opA           = HIPBLAS_OP_N,
                     int32_t            streamKMode   = -1)
    {
        *launched = -1;

        hipblasLtHandle_t handle = nullptr;
        if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
            return false;

        void*        dA      = nullptr;
        void*        dB      = nullptr;
        void*        dC      = nullptr;
        void*        dD      = nullptr;
        void*        dWs     = nullptr;
        const size_t wsBytes = 32 * 1024 * 1024;

        bool ok = hipMalloc(&dA, m * k * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dB, k * n * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dC, m * n * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dD, m * n * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dWs, wsBytes) == hipSuccess;

        if(ok)
        {
            const uint16_t              fill = verifyProduct ? halfBits(1.0f) : 0;
            const std::vector<uint16_t> a(static_cast<size_t>(m * k), fill);
            const std::vector<uint16_t> b(static_cast<size_t>(k * n), fill);
            ok = hipMemcpy(dA, a.data(), a.size() * sizeof(uint16_t), hipMemcpyHostToDevice)
                     == hipSuccess
                 && hipMemcpy(dB, b.data(), b.size() * sizeof(uint16_t), hipMemcpyHostToDevice)
                        == hipSuccess
                 && hipMemset(dC, 0, m * n * sizeof(uint16_t)) == hipSuccess
                 && hipMemset(dD, 0, m * n * sizeof(uint16_t)) == hipSuccess;
        }

        hipblasLtMatrixLayout_t     layoutA = nullptr, layoutB = nullptr, layoutC = nullptr;
        hipblasLtMatmulDesc_t       desc = nullptr;
        hipblasLtMatmulPreference_t pref = nullptr;

        if(ok)
        {
            const uint64_t maxWs  = wsBytes;
            const bool     plainA = opA == HIPBLAS_OP_N;
            ok                    = hipblasLtMatrixLayoutCreate(
                     &layoutA, HIP_R_16F, plainA ? m : k, plainA ? k : m, plainA ? m : k)
                     == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_16F, k, n, k)
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_16F, m, n, m)
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulDescSetAttribute(
                        desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA))
                        == HIPBLAS_STATUS_SUCCESS
                 && (streamKMode < 0
                     || hipblasLtMatmulDescSetAttribute(
                            desc,
                            HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT,
                            &streamKMode,
                            sizeof(streamKMode))
                            == HIPBLAS_STATUS_SUCCESS)
                 && hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulPreferenceSetAttribute(
                        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs))
                        == HIPBLAS_STATUS_SUCCESS;
        }

        const float                  alpha   = 1.0f;
        const float                  beta    = 0.0f;
        hipblasLtMatmulAlgo_t        algo    = {};
        const hipblasLtMatmulAlgo_t* algoPtr = nullptr;

        if(ok && from == AlgoFrom::Heuristic)
        {
            hipblasLtMatmulHeuristicResult_t heuristic[1];
            int                              returned = 0;
            ok                                        = hipblasLtMatmulAlgoGetHeuristic(handle,
                                                 desc,
                                                 layoutA,
                                                 layoutB,
                                                 layoutC,
                                                 layoutC,
                                                 pref,
                                                 1,
                                                 heuristic,
                                                 &returned)
                     == HIPBLAS_STATUS_SUCCESS
                 && returned > 0;
            algo    = heuristic[0].algo;
            algoPtr = &algo;
        }
        else if(ok && from == AlgoFrom::Index)
        {
            std::vector<int>                              indexes{index};
            std::vector<hipblasLtMatmulHeuristicResult_t> results;
            size_t                                        required = 0;
            ok = hipblaslt_ext::getAlgosFromIndex(handle, indexes, results)
                     == HIPBLAS_STATUS_SUCCESS
                 && !results.empty()
                 && hipblaslt_ext::matmulIsAlgoSupported(handle,
                                                         desc,
                                                         &alpha,
                                                         layoutA,
                                                         layoutB,
                                                         &beta,
                                                         layoutC,
                                                         layoutC,
                                                         results[0].algo,
                                                         required)
                        == HIPBLAS_STATUS_SUCCESS
                 && required <= wsBytes;
            if(ok)
            {
                algo    = results[0].algo;
                algoPtr = &algo;
            }
        }

        if(ok)
        {
            ok = hipblasLtMatmul(handle,
                                 desc,
                                 &alpha,
                                 dA,
                                 layoutA,
                                 dB,
                                 layoutB,
                                 &beta,
                                 dC,
                                 layoutC,
                                 dD,
                                 layoutC,
                                 algoPtr,
                                 dWs,
                                 wsBytes,
                                 nullptr)
                     == HIPBLAS_STATUS_SUCCESS
                 && hipDeviceSynchronize() == hipSuccess;
            *launched = hipblaslt_tuning_last_launch_for_test();
        }

        if(ok && verifyProduct)
        {
            std::vector<uint16_t> out(static_cast<size_t>(m * n), 0);
            ok = hipMemcpy(out.data(), dD, out.size() * sizeof(uint16_t), hipMemcpyDeviceToHost)
                 == hipSuccess;

            const uint16_t expected = halfBits(static_cast<float>(k));
            for(size_t i = 0; ok && i < out.size(); i++)
                ok = out[i] == expected;
        }

        if(pref)
            hipblasLtMatmulPreferenceDestroy(pref);
        if(desc)
            hipblasLtMatmulDescDestroy(desc);
        if(layoutC)
            hipblasLtMatrixLayoutDestroy(layoutC);
        if(layoutB)
            hipblasLtMatrixLayoutDestroy(layoutB);
        if(layoutA)
            hipblasLtMatrixLayoutDestroy(layoutA);
        static_cast<void>(hipFree(dWs));
        static_cast<void>(hipFree(dD));
        static_cast<void>(hipFree(dC));
        static_cast<void>(hipFree(dB));
        static_cast<void>(hipFree(dA));
        hipblasLtDestroy(handle);

        return ok;
    }

    /** What runGradientGemm found. */
    enum class GradientResult
    {
        Correct, // D is the gradient of the caller's E, which came back unchanged
        Unsupported, // the heuristic offers no algorithm for the problem
        Wrong, // a call failed, D is wrong, or E changed
    };

    /**
     * One fp16 DGELU matmul with separate C and D and no algo, with A and B all
     * ones and E +8 on even rows and -8 on odd ones.
     *
     * The GELU derivative is 1 at +8 and 0 at -8 to well within fp16, in the
     * tanh form and the erf form alike, so D must be exactly k on even rows and
     * zero on odd ones. A product scaled by a zeroed E would be k / 2.
     */
    GradientResult runGradientGemm(int64_t m, int64_t n, int64_t k)
    {
        hipblasLtHandle_t handle = nullptr;
        if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
            return GradientResult::Wrong;

        const size_t          rows     = static_cast<size_t>(m);
        const size_t          elements = static_cast<size_t>(m * n);
        std::vector<uint16_t> e(elements);
        for(size_t i = 0; i < elements; i++)
            e[i] = halfBits((i % rows) % 2 == 0 ? 8.0f : -8.0f);

        void*        dA      = nullptr;
        void*        dB      = nullptr;
        void*        dC      = nullptr;
        void*        dD      = nullptr;
        void*        dE      = nullptr;
        void*        dWs     = nullptr;
        const size_t wsBytes = 32 * 1024 * 1024;

        bool ok = hipMalloc(&dA, m * k * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dB, k * n * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dC, elements * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dD, elements * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dE, elements * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dWs, wsBytes) == hipSuccess;

        if(ok)
        {
            const std::vector<uint16_t> onesA(static_cast<size_t>(m * k), halfBits(1.0f));
            const std::vector<uint16_t> onesB(static_cast<size_t>(k * n), halfBits(1.0f));
            ok = hipMemcpy(dA, onesA.data(), onesA.size() * sizeof(uint16_t), hipMemcpyHostToDevice)
                     == hipSuccess
                 && hipMemcpy(
                        dB, onesB.data(), onesB.size() * sizeof(uint16_t), hipMemcpyHostToDevice)
                        == hipSuccess
                 && hipMemset(dC, 0, elements * sizeof(uint16_t)) == hipSuccess
                 && hipMemset(dD, 0, elements * sizeof(uint16_t)) == hipSuccess
                 && hipMemcpy(dE, e.data(), elements * sizeof(uint16_t), hipMemcpyHostToDevice)
                        == hipSuccess;
        }

        hipblasLtMatrixLayout_t     layoutA = nullptr, layoutB = nullptr, layoutC = nullptr;
        hipblasLtMatmulDesc_t       desc     = nullptr;
        hipblasLtMatmulPreference_t pref     = nullptr;
        const uint32_t              epilogue = HIPBLASLT_EPILOGUE_DGELU;
        const int64_t               ldE      = m;
        const uint64_t              maxWs    = wsBytes;

        if(ok)
        {
            ok = hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_16F, m, k, m) == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_16F, k, n, k)
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_16F, m, n, m)
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulDescSetAttribute(
                        desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue))
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulDescSetAttribute(
                        desc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &dE, sizeof(dE))
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulDescSetAttribute(
                        desc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ldE, sizeof(ldE))
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulPreferenceSetAttribute(
                        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs))
                        == HIPBLAS_STATUS_SUCCESS;
        }

        GradientResult result = GradientResult::Wrong;
        if(ok)
        {
            // Only asks whether anything can run the problem. The matmul passes
            // no algo, so the call that tunes launches the winner.
            hipblasLtMatmulHeuristicResult_t heuristic[1];
            int                              returned = 0;
            if(hipblasLtMatmulAlgoGetHeuristic(
                   handle, desc, layoutA, layoutB, layoutC, layoutC, pref, 1, heuristic, &returned)
                   != HIPBLAS_STATUS_SUCCESS
               || returned == 0)
                result = GradientResult::Unsupported;
            else
            {
                const float alpha = 1.0f;
                const float beta  = 0.0f;
                ok                = hipblasLtMatmul(handle,
                                     desc,
                                     &alpha,
                                     dA,
                                     layoutA,
                                     dB,
                                     layoutB,
                                     &beta,
                                     dC,
                                     layoutC,
                                     dD,
                                     layoutC,
                                     nullptr,
                                     dWs,
                                     wsBytes,
                                     nullptr)
                         == HIPBLAS_STATUS_SUCCESS
                     && hipDeviceSynchronize() == hipSuccess;

                std::vector<uint16_t> d(elements), eAfter(elements);
                ok = ok
                     && hipMemcpy(d.data(), dD, elements * sizeof(uint16_t), hipMemcpyDeviceToHost)
                            == hipSuccess
                     && hipMemcpy(
                            eAfter.data(), dE, elements * sizeof(uint16_t), hipMemcpyDeviceToHost)
                            == hipSuccess;

                // The odd rows may come out as either signed zero.
                const uint16_t product = halfBits(static_cast<float>(k));
                for(size_t i = 0; ok && i < elements; i++)
                    ok = (i % rows) % 2 == 0 ? d[i] == product : (d[i] & 0x7fff) == 0;
                ok     = ok && eAfter == e;
                result = ok ? GradientResult::Correct : GradientResult::Wrong;
            }
        }

        if(pref)
            hipblasLtMatmulPreferenceDestroy(pref);
        if(desc)
            hipblasLtMatmulDescDestroy(desc);
        for(auto layout : {layoutA, layoutB, layoutC})
            if(layout)
                hipblasLtMatrixLayoutDestroy(layout);
        for(void* buffer : {dWs, dE, dD, dC, dB, dA})
            static_cast<void>(hipFree(buffer));
        hipblasLtDestroy(handle);

        return result;
    }

    /**
     * Distinct (solution index, kernel name) pairs the heuristic offers for this
     * shape, newest API surface only.
     *
     * A hand-built cache row has to name a kernel that really can run the
     * problem, or replay rejects it on identity or support and the row order
     * under test never gets to matter. Taking the pairs from the heuristic is
     * what guarantees that.
     */
    std::vector<std::pair<int, std::string>>
        candidateIdentities(int64_t m, int64_t n, int64_t k, int want)
    {
        std::vector<std::pair<int, std::string>> found;

        hipblasLtHandle_t handle = nullptr;
        if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
            return found;

        hipblasLtMatrixLayout_t     layoutA = nullptr, layoutB = nullptr, layoutC = nullptr;
        hipblasLtMatmulDesc_t       desc = nullptr;
        hipblasLtMatmulPreference_t pref = nullptr;

        bool ok
            = hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_16F, m, k, m) == HIPBLAS_STATUS_SUCCESS
              && hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_16F, k, n, k) == HIPBLAS_STATUS_SUCCESS
              && hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_16F, m, n, m) == HIPBLAS_STATUS_SUCCESS
              && hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                     == HIPBLAS_STATUS_SUCCESS
              && hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS;

        if(ok)
        {
            const uint64_t maxWs = 32 * 1024 * 1024;
            ok                   = hipblasLtMatmulPreferenceSetAttribute(
                     pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs))
                 == HIPBLAS_STATUS_SUCCESS;
        }

        if(ok)
        {
            std::vector<hipblasLtMatmulHeuristicResult_t> heuristic(want);
            int                                           returned = 0;
            if(hipblasLtMatmulAlgoGetHeuristic(handle,
                                               desc,
                                               layoutA,
                                               layoutB,
                                               layoutC,
                                               layoutC,
                                               pref,
                                               want,
                                               heuristic.data(),
                                               &returned)
               == HIPBLAS_STATUS_SUCCESS)
            {
                for(int i = 0; i < returned; i++)
                {
                    const int  index = hipblaslt_ext::getIndexFromAlgo(heuristic[i].algo);
                    const auto name
                        = hipblaslt_ext::getKernelNameFromAlgo(handle, heuristic[i].algo);
                    if(name.empty())
                        continue;

                    bool seen = false;
                    for(const auto& p : found)
                        seen = seen || p.first == index;
                    if(!seen)
                        found.emplace_back(index, name);
                }
            }
        }

        if(pref)
            hipblasLtMatmulPreferenceDestroy(pref);
        if(desc)
            hipblasLtMatmulDescDestroy(desc);
        if(layoutC)
            hipblasLtMatrixLayoutDestroy(layoutC);
        if(layoutB)
            hipblasLtMatrixLayoutDestroy(layoutB);
        if(layoutA)
            hipblasLtMatrixLayoutDestroy(layoutA);
        hipblasLtDestroy(handle);

        return found;
    }

    /**
     * One FP8 GEMM, or false if this device has no FP8 solution for it.
     *
     * Separate from runGemm because FP8 needs one-byte inputs, an fp16 output
     * and a transposed A, and because it exists only to prove that the datatype
     * a row records is the datatype it parses back as.
     */
    bool runFp8Gemm(int64_t m, int64_t n, int64_t k, bool* unsupported)
    {
        *unsupported = false;

        hipblasLtHandle_t handle = nullptr;
        if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
            return false;

        void*        dA      = nullptr;
        void*        dB      = nullptr;
        void*        dD      = nullptr;
        void*        dWs     = nullptr;
        const size_t wsBytes = 32 * 1024 * 1024;

        bool ok = hipMalloc(&dA, m * k) == hipSuccess && hipMalloc(&dB, k * n) == hipSuccess
                  && hipMalloc(&dD, m * n * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dWs, wsBytes) == hipSuccess;
        if(ok)
            ok = hipMemset(dA, 0, m * k) == hipSuccess && hipMemset(dB, 0, k * n) == hipSuccess
                 && hipMemset(dD, 0, m * n * sizeof(uint16_t)) == hipSuccess;

        hipblasLtMatrixLayout_t     layoutA = nullptr, layoutB = nullptr, layoutD = nullptr;
        hipblasLtMatmulDesc_t       desc = nullptr;
        hipblasLtMatmulPreference_t pref = nullptr;

        if(ok)
        {
            // transA = T, so A is stored k by m.
            const hipblasOperation_t opT = HIPBLAS_OP_T;
            const hipblasOperation_t opN = HIPBLAS_OP_N;
            ok = hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_8F_E4M3_FNUZ, k, m, k)
                     == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_8F_E4M3_FNUZ, k, n, k)
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatrixLayoutCreate(&layoutD, HIP_R_16F, m, n, m)
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulDescSetAttribute(
                        desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT))
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulDescSetAttribute(
                        desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN))
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS;
        }

        if(ok)
        {
            const uint64_t maxWs = wsBytes;
            ok                   = hipblasLtMatmulPreferenceSetAttribute(
                     pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs))
                 == HIPBLAS_STATUS_SUCCESS;
        }

        if(ok)
        {
            hipblasLtMatmulHeuristicResult_t heuristic[1];
            int                              returned = 0;
            const bool                       gotAlgo  = hipblasLtMatmulAlgoGetHeuristic(handle,
                                                                 desc,
                                                                 layoutA,
                                                                 layoutB,
                                                                 layoutD,
                                                                 layoutD,
                                                                 pref,
                                                                 1,
                                                                 heuristic,
                                                                 &returned)
                                     == HIPBLAS_STATUS_SUCCESS
                                 && returned > 0;

            if(!gotAlgo)
            {
                *unsupported = true;
            }
            else
            {
                const float alpha = 1.0f;
                const float beta  = 0.0f;
                ok                = hipblasLtMatmul(handle,
                                     desc,
                                     &alpha,
                                     dA,
                                     layoutA,
                                     dB,
                                     layoutB,
                                     &beta,
                                     dD,
                                     layoutD,
                                     dD,
                                     layoutD,
                                     &heuristic[0].algo,
                                     dWs,
                                     wsBytes,
                                     nullptr)
                     == HIPBLAS_STATUS_SUCCESS;
                ok = ok && hipDeviceSynchronize() == hipSuccess;
            }
        }

        if(pref)
            hipblasLtMatmulPreferenceDestroy(pref);
        if(desc)
            hipblasLtMatmulDescDestroy(desc);
        if(layoutD)
            hipblasLtMatrixLayoutDestroy(layoutD);
        if(layoutB)
            hipblasLtMatrixLayoutDestroy(layoutB);
        if(layoutA)
            hipblasLtMatrixLayoutDestroy(layoutA);
        static_cast<void>(hipFree(dWs));
        static_cast<void>(hipFree(dD));
        static_cast<void>(hipFree(dB));
        static_cast<void>(hipFree(dA));
        hipblasLtDestroy(handle);

        return ok;
    }

    // The fp16 problem runGemm and runGemmWith launch, described with their
    // workspace limit.
    struct Fp16Problem
    {
        hipblasLtHandle_t           handle  = nullptr;
        hipblasLtMatrixLayout_t     layoutA = nullptr;
        hipblasLtMatrixLayout_t     layoutB = nullptr;
        hipblasLtMatrixLayout_t     layoutC = nullptr;
        hipblasLtMatmulDesc_t       desc    = nullptr;
        hipblasLtMatmulPreference_t pref    = nullptr;
        uint64_t                    maxWs   = 32 * 1024 * 1024;
        bool                        ok      = false;

        Fp16Problem(int64_t m, int64_t n, int64_t k)
        {
            ok = hipblasLtCreate(&handle) == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_16F, m, k, m)
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_16F, k, n, k)
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_16F, m, n, m)
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulPreferenceSetAttribute(
                        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs))
                        == HIPBLAS_STATUS_SUCCESS;
        }

        ~Fp16Problem()
        {
            if(pref)
                hipblasLtMatmulPreferenceDestroy(pref);
            if(desc)
                hipblasLtMatmulDescDestroy(desc);
            for(auto layout : {layoutA, layoutB, layoutC})
                if(layout)
                    hipblasLtMatrixLayoutDestroy(layout);
            if(handle)
                hipblasLtDestroy(handle);
        }

        Fp16Problem(const Fp16Problem&)            = delete;
        Fp16Problem& operator=(const Fp16Problem&) = delete;

        // The solution indexes the heuristic offers, up to `requested`, in its order.
        std::vector<int> offered(int requested) const
        {
            std::vector<hipblasLtMatmulHeuristicResult_t> heuristic(requested);
            int                                           returned = 0;
            std::vector<int>                              indexes;
            if(ok
               && hipblasLtMatmulAlgoGetHeuristic(handle,
                                                  desc,
                                                  layoutA,
                                                  layoutB,
                                                  layoutC,
                                                  layoutC,
                                                  pref,
                                                  requested,
                                                  heuristic.data(),
                                                  &returned)
                      == HIPBLAS_STATUS_SUCCESS)
                for(int i = 0; i < returned; i++)
                    indexes.push_back(hipblaslt_ext::getIndexFromAlgo(heuristic[i].algo));
            return indexes;
        }

        // The solution indexes tune mode would measure under the current
        // settings, in order and before the baseline is moved to the front.
        std::vector<int> tuningCandidates() const
        {
            if(!ok)
                return {};
            std::vector<int> indexes(hipblaslt_tuning_candidates_for_test(
                handle, desc, layoutA, layoutB, layoutC, layoutC, maxWs, nullptr, 0));
            hipblaslt_tuning_candidates_for_test(handle,
                                                 desc,
                                                 layoutA,
                                                 layoutB,
                                                 layoutC,
                                                 layoutC,
                                                 maxWs,
                                                 indexes.data(),
                                                 indexes.size());
            return indexes;
        }
    };

    std::vector<std::string> readLines(const std::string& path)
    {
        std::vector<std::string> lines;
        std::ifstream            in(path);
        std::string              line;
        while(std::getline(in, line))
            lines.push_back(line);
        return lines;
    }

    std::vector<std::string> splitCells(const std::string& s)
    {
        std::vector<std::string> out;
        std::stringstream        ss(s);
        std::string              cell;
        while(std::getline(ss, cell, ','))
        {
            const auto b = cell.find_first_not_of(" \t");
            const auto e = cell.find_last_not_of(" \t");
            out.push_back(b == std::string::npos ? "" : cell.substr(b, e - b + 1));
        }
        return out;
    }

    /** Overwrite one named column in every value row. */
    ::testing::AssertionResult
        rewriteColumn(const std::string& path, const std::string& column, const std::string& value)
    {
        auto lines = readLines(path);
        if(lines.empty())
            return ::testing::AssertionFailure() << "no tuning file to rewrite at " << path;

        auto split = splitCells;

        bool changed = false;
        for(size_t i = 0; i + 1 < lines.size(); i++)
        {
            if(lines[i].find("transA") == std::string::npos)
                continue;

            const auto names  = split(lines[i]);
            auto       values = split(lines[i + 1]);
            for(size_t c = 0; c < names.size() && c < values.size(); c++)
            {
                if(names[c] == column)
                {
                    values[c] = value;
                    changed   = true;
                }
            }

            std::ostringstream rebuilt;
            for(size_t c = 0; c < values.size(); c++)
                rebuilt << (c ? "," : "") << values[c];
            lines[i + 1] = rebuilt.str();
            i++;
        }

        if(!changed)
            return ::testing::AssertionFailure() << "no " << column << " cell in " << path;

        std::ofstream out(path, std::ios::trunc);
        for(const auto& l : lines)
            out << l << "\n";
        out.close();

        if(!out)
            return ::testing::AssertionFailure() << "could not write tuning file " << path;
        return ::testing::AssertionSuccess();
    }

    bool fileHasColumn(const std::string& path, const std::string& column)
    {
        for(const auto& line : readLines(path))
            if(line.find("transA") != std::string::npos && line.find(column) != std::string::npos)
                return true;
        return false;
    }

    /**
     * Collects what the library writes to std::cerr for the life of the object.
     *
     * Enough for the no-log-level cases, which are the ones worth asserting on:
     * with logging off the lifecycle router writes to the std::cerr object
     * itself, so swapping its buffer catches it. It cannot see anything sent to
     * a log file or written straight to file descriptor 2, and it is deliberately
     * scoped tightly around one call because unrelated warnings share the stream.
     */
    class CerrCapture
    {
    public:
        CerrCapture()
            : m_saved(std::cerr.rdbuf(m_buffer.rdbuf()))
        {
        }

        ~CerrCapture()
        {
            std::cerr.rdbuf(m_saved);
        }

        CerrCapture(const CerrCapture&)            = delete;
        CerrCapture& operator=(const CerrCapture&) = delete;

        std::string str() const
        {
            return m_buffer.str();
        }

    private:
        std::ostringstream m_buffer;
        std::streambuf*    m_saved;
    };

    size_t countOccurrences(const std::string& haystack, const std::string& needle)
    {
        size_t n = 0;
        for(size_t at = haystack.find(needle); at != std::string::npos;
            at        = haystack.find(needle, at + needle.size()))
            n++;
        return n;
    }

    /**
     * Whether tuning output is predictable enough to assert on.
     *
     * The logger latches its environment at first use and cannot be reset, so a
     * run that already has a level or a log file set sends the lifecycle lines
     * somewhere this fixture cannot see.
     */
    bool loggingEnvIsClean()
    {
        return getenv("HIPBLASLT_LOG_LEVEL") == nullptr && getenv("HIPBLASLT_LOG_MASK") == nullptr
               && getenv("HIPBLASLT_LOG_FILE") == nullptr;
    }

    size_t valueRowCount(const std::string& path)
    {
        size_t     n     = 0;
        const auto lines = readLines(path);
        for(size_t i = 0; i + 1 < lines.size(); i++)
        {
            if(lines[i].find("transA") != std::string::npos)
            {
                n++;
                i++;
            }
        }
        return n;
    }

    /**
     * Whether the C++ extension heuristic, asked with this stream-K tile mode,
     * is served by the cache: a hit is counted only when an entry matched the
     * key that path looks up.
     */
    // benchStyle sets the problem up the way hipblaslt-bench's C++ API run does:
    // explicit strides and a GemmProblemType, and the epilogue's A and B
    // scaling types set to scalar although there is no scale.
    bool extensionHeuristicHits(
        int64_t m, int64_t n, int64_t k, int32_t streamKMode, bool benchStyle = false)
    {
        hipblasLtHandle_t handle = nullptr;
        if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
            return false;

        void* dA = nullptr;
        void* dB = nullptr;
        void* dD = nullptr;
        bool  ok = hipMalloc(&dA, m * k * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dB, k * n * sizeof(uint16_t)) == hipSuccess
                  && hipMalloc(&dD, m * n * sizeof(uint16_t)) == hipSuccess;

        const uint64_t before = counters().hits;
        if(ok)
        {
            const float alpha = 1.0f;
            const float beta  = 0.0f;

            hipblaslt_ext::GemmPreference pref;
            pref.setMaxWorkspaceBytes(32 * 1024 * 1024);
            if(streamKMode >= 0)
                pref.setStreamKTileSchedulingMode(
                    static_cast<hipblasLtStreamKTileSchedulingMode_t>(streamKMode));

            hipblaslt_ext::Gemm gemm(handle,
                                     HIPBLAS_OP_N,
                                     HIPBLAS_OP_N,
                                     HIP_R_16F,
                                     HIP_R_16F,
                                     HIP_R_16F,
                                     HIP_R_16F,
                                     HIPBLAS_COMPUTE_32F);

            hipblaslt_ext::GemmEpilogue epilogue;
            hipblaslt_ext::GemmInputs   inputs;
            inputs.setA(dA);
            inputs.setB(dB);
            inputs.setC(dD);
            inputs.setD(dD);
            inputs.setAlpha(&alpha);
            inputs.setBeta(&beta);
            if(benchStyle)
            {
                epilogue.setScalingAType(HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F);
                epilogue.setScalingBType(HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F);

                hipblaslt_ext::GemmProblemType problemType;
                problemType.setOpA(HIPBLAS_OP_N);
                problemType.setOpB(HIPBLAS_OP_N);
                problemType.setTypeA(HIP_R_16F);
                problemType.setTypeB(HIP_R_16F);
                problemType.setTypeC(HIP_R_16F);
                problemType.setTypeD(HIP_R_16F);
                problemType.setTypeCompute(HIPBLAS_COMPUTE_32F);
                gemm.setProblem(m,
                                n,
                                k,
                                1,
                                m,
                                k,
                                m,
                                m,
                                m * k,
                                k * n,
                                m * n,
                                m * n,
                                epilogue,
                                inputs,
                                problemType);
            }
            else
            {
                gemm.setProblem(m, n, k, 1, epilogue, inputs);
            }

            std::vector<hipblasLtMatmulHeuristicResult_t> results;
            ok = gemm.algoGetHeuristic(1, pref, results) == HIPBLAS_STATUS_SUCCESS
                 && !results.empty();
        }

        static_cast<void>(hipFree(dD));
        static_cast<void>(hipFree(dB));
        static_cast<void>(hipFree(dA));
        hipblasLtDestroy(handle);

        return ok && counters().hits > before;
    }

    /** One named column's value from each value row, in file order. */
    std::vector<std::string> columnValues(const std::string& path, const std::string& column)
    {
        std::vector<std::string> out;
        const auto               lines = readLines(path);
        for(size_t i = 0; i + 1 < lines.size(); i++)
        {
            if(lines[i].find("transA") == std::string::npos)
                continue;

            const auto names  = splitCells(lines[i]);
            const auto values = splitCells(lines[i + 1]);
            for(size_t c = 0; c < names.size() && c < values.size(); c++)
                if(names[c] == column)
                    out.push_back(values[c]);
            i++;
        }
        return out;
    }

    class TuningTune_pre_checkin : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            static const char* const names[] = {"HIPBLASLT_TUNING_MODE",
                                                "HIPBLASLT_TUNING_CACHE_PATH",
                                                "HIPBLASLT_TUNING_OVERRIDE_FILE",
                                                "HIPBLASLT_TUNING_ALL_KERNELS",
                                                "HIPBLASLT_TUNING_MAX_CANDIDATES",
                                                "HIPBLASLT_TUNING_COLD_ITERS",
                                                "HIPBLASLT_TUNING_HOT_ITERS",
                                                "HIPBLASLT_TUNING_ROTATING_MB",
                                                "HIPBLASLT_TUNING_FLUSH_ICACHE",
                                                "HIPBLASLT_TUNING_BUDGET_MS_PER_SHAPE",
                                                "HIPBLASLT_TUNING_SCRATCH_MAX_BYTES",
                                                // Set by one logging case, and
                                                // left set would make every
                                                // later case skip itself.
                                                "HIPBLASLT_LOG_FILE"};
            for(const char* name : names)
            {
                const char* value = getenv(name);
                m_savedEnv.emplace_back(name,
                                        value ? std::optional<std::string>(value) : std::nullopt);
            }

            if(!gpuAvailable())
                GTEST_SKIP() << "No GPU available";

            // A user's legacy override must not leak into a cache-mode test.
            unsetenv("HIPBLASLT_TUNING_OVERRIDE_FILE");
            m_path = tempCachePath(::testing::UnitTest::GetInstance()->current_test_info()->name());
            std::remove(m_path.c_str());
        }

        void TearDown() override
        {
            std::remove(m_path.c_str());
            for(const auto& [name, value] : m_savedEnv)
            {
                if(value)
                    setenv(name.c_str(), value->c_str(), 1);
                else
                    unsetenv(name.c_str());
            }
            hipblaslt_tuning_reset_for_test();
        }

        /**
         * enterMode with the search cut down to almost nothing.
         *
         * The logging cases care about which lines appear, not about tuning
         * quality, and the shared enterMode settings pay for rotation seeding and
         * an icache calibration burst that none of these assertions look at.
         * Keeping them separate also guarantees each search finishes well inside
         * the ten-second progress interval, so the heartbeat cannot appear and
         * make the negative assertions flaky.
         */
        void enterModeForLogging(const char* mode, const std::string& path)
        {
            enterMode(mode, path);
            setenv("HIPBLASLT_TUNING_ALL_KERNELS", "0", 1);
            setenv("HIPBLASLT_TUNING_MAX_CANDIDATES", "2", 1);
            setenv("HIPBLASLT_TUNING_COLD_ITERS", "0", 1);
            setenv("HIPBLASLT_TUNING_HOT_ITERS", "1", 1);
            setenv("HIPBLASLT_TUNING_FLUSH_ICACHE", "0", 1);
            setenv("HIPBLASLT_TUNING_ROTATING_MB", "0", 1);
            hipblaslt_tuning_reset_for_test();
        }

        std::string                                                     m_path;
        std::vector<std::pair<std::string, std::optional<std::string>>> m_savedEnv;
    };

    // Nothing is written and nothing is consulted when tuning is not asked for.
    TEST_F(TuningTune_pre_checkin, OffModeWritesNothing)
    {
        enterMode("off", m_path);
        ASSERT_TRUE(runGemm(256, 256, 256));

        std::ifstream probe(m_path);
        EXPECT_FALSE(probe.good()) << "off mode created " << m_path;

        const auto c = counters();
        EXPECT_EQ(c.tuned, 0u);
        EXPECT_EQ(c.hits, 0u);
    }

    // A tuned row must carry everything a later lookup rebuilds the key from,
    // plus the identity fields validation depends on.
    TEST_F(TuningTune_pre_checkin, TuneWritesRowWithIdentityAndKeyFields)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        // Asserted rather than skipped. Selection takes the fastest candidate
        // outright and records even when that is the default pick, so on a
        // working GPU tune mode always writes a row for this shape; no row means
        // the benchmarker or the writer is broken, which is exactly what these
        // tests exist to catch. Skipping instead let every one of them go green
        // on a build where tuning silently did nothing.
        ASSERT_TRUE(std::ifstream(m_path).good()) << "tune mode recorded nothing";

        EXPECT_TRUE(fileHasColumn(m_path, "kernel_name"));
        EXPECT_TRUE(fileHasColumn(m_path, "schema_version"));
        EXPECT_TRUE(fileHasColumn(m_path, "compute_input_type_a"));
        EXPECT_TRUE(fileHasColumn(m_path, "gcnArchName"));
        EXPECT_TRUE(fileHasColumn(m_path, "lde"));
        EXPECT_TRUE(fileHasColumn(m_path, "stride_e"));
        EXPECT_TRUE(fileHasColumn(m_path, "uniform_summation_order"));

        // Nothing reads these back, so writing them would commit the format to
        // data with no consumer.
        EXPECT_FALSE(fileHasColumn(m_path, "baseline_index"));
        EXPECT_FALSE(fileHasColumn(m_path, "baseline_us"));
    }

    // Exercise the shipping-default exhaustive enumeration independently from
    // the fast ranked-prefix settings used by the rest of this suite.
    TEST_F(TuningTune_pre_checkin, ExhaustiveEnumerationWritesEntry)
    {
        enterMode("tune", m_path);
        setenv("HIPBLASLT_TUNING_ALL_KERNELS", "1", 1);
        setenv("HIPBLASLT_TUNING_COLD_ITERS", "0", 1);
        setenv("HIPBLASLT_TUNING_HOT_ITERS", "1", 1);
        setenv("HIPBLASLT_TUNING_ROTATING_MB", "0", 1);
        hipblaslt_tuning_reset_for_test();

        ASSERT_TRUE(runGemm(64, 64, 64));
        ASSERT_GT(valueRowCount(m_path), 0u) << "exhaustive tuning recorded nothing";
    }

    // A ranked search measures the list hipblasLtMatmulAlgoGetHeuristic returns,
    // in its order, including what it adds from the size-independent libraries
    // when the ranked ones return fewer than asked for, so tune mode and
    // hipblaslt-bench search the same candidates.
    TEST_F(TuningTune_pre_checkin, RankedCandidatesAreTheHeuristicsList)
    {
        const Fp16Problem problem(1024, 512, 1024);
        ASSERT_TRUE(problem.ok);
        enterMode("tune", m_path);
        for(int cap : {1, 16, 256})
        {
            SCOPED_TRACE(cap);
            setenv("HIPBLASLT_TUNING_MAX_CANDIDATES", std::to_string(cap).c_str(), 1);
            hipblaslt_tuning_reset_for_test();

            const auto offered = problem.offered(cap);
            ASSERT_FALSE(offered.empty());
            EXPECT_EQ(problem.tuningCandidates(), offered);
        }
    }

    // The winner has to actually run on the caller's buffers and be right.
    //
    // Separate C and D, and nonzero inputs, because the rest of the suite tunes
    // in-place with zeroed operands: there every candidate and every bug
    // produces the same all-zero output, so a tuning path that reported success
    // without ever launching the winner would still pass.
    TEST_F(TuningTune_pre_checkin, TunedLaunchProducesCorrectResult)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(128, 128, 128, 0.0f, false, false, true));
        ASSERT_GT(valueRowCount(m_path), 0u) << "tune mode recorded nothing";

        enterMode("cache", m_path);
        ASSERT_TRUE(runGemm(128, 128, 128, 0.0f, false, false, true));
        EXPECT_GE(counters().hits, 1u) << "tuned entry did not replay";
    }

    // A gradient epilogue reads E, which tune mode copies into its scratch for
    // the candidates. The call that tunes must still compute the gradient of
    // the caller's E, and leave that E as it was.
    TEST_F(TuningTune_pre_checkin, TunedGradientEpilogueReadsTheCallersAux)
    {
        enterMode("tune", m_path);
        const auto result = runGradientGemm(128, 128, 128);
        if(result == GradientResult::Unsupported)
            GTEST_SKIP() << "the heuristic offers no DGELU algorithm for this problem";
        EXPECT_TRUE(result == GradientResult::Correct)
            << "the tuned gradient is wrong, or tuning changed the caller's E";
        EXPECT_EQ(counters().tuned, 1u) << "the gradient call was not tuned";
    }

    // A recorded datatype must parse back as the datatype it was written from.
    //
    // The FNUZ and OCP FP8 types share one bench spelling, "f8_r", which the
    // parser resolves to the OCP type. gfx942 FP8 is FNUZ, so every FP8 row it
    // wrote described a different type than the one measured: the row never
    // matched its own key, the shape re-tuned on every process start, and the
    // cache file grew a duplicate each time.
    TEST_F(TuningTune_pre_checkin, Fp8FnuzEntryReplays)
    {
        bool unsupported = false;

        enterMode("tune", m_path);
        ASSERT_TRUE(runFp8Gemm(512, 512, 512, &unsupported));
        if(unsupported)
            GTEST_SKIP() << "no FP8 FNUZ solution on this device";
        ASSERT_GT(valueRowCount(m_path), 0u) << "tune mode recorded nothing";

        enterMode("cache", m_path);
        ASSERT_TRUE(runFp8Gemm(512, 512, 512, &unsupported));
        EXPECT_GE(counters().hits, 1u) << "FP8 entry did not replay";

        // The shape is already tuned, so a second tune run must not append.
        const size_t rows = valueRowCount(m_path);
        enterMode("tune", m_path);
        ASSERT_TRUE(runFp8Gemm(512, 512, 512, &unsupported));
        EXPECT_EQ(valueRowCount(m_path), rows) << "FP8 shape re-tuned and appended a duplicate";
    }

    // In-place nonzero-beta timing repeatedly feeds D back as C. Until the
    // timing loop can reset each reused block outside its event span, tuning
    // must decline while still running the caller's real GEMM correctly.
    TEST_F(TuningTune_pre_checkin, InPlaceNonzeroBetaRunsWithoutTuning)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(384, 256, 128, 2.0f, true, true));
        EXPECT_EQ(valueRowCount(m_path), 0u);
        EXPECT_GE(counters().skipped, 1u);
    }

    // The round trip the whole feature exists for.
    TEST_F(TuningTune_pre_checkin, TunedEntryReplaysInCacheMode)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        ASSERT_GT(valueRowCount(m_path), 0u) << "tune mode recorded nothing";

        enterMode("cache", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        const auto c = counters();
        EXPECT_GE(c.hits, 1u) << "tuned entry did not replay";
        EXPECT_EQ(c.invalidated, 0u);
    }

    // An entry whose recorded kernel is not what the index resolves to must be
    // refused, and the shape must still run.
    TEST_F(TuningTune_pre_checkin, TamperedKernelNameIsRejected)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        ASSERT_GT(valueRowCount(m_path), 0u) << "tune mode recorded nothing";
        ASSERT_TRUE(rewriteColumn(m_path, "kernel_name", "NotARealKernelName"));

        enterMode("cache", m_path);
        EXPECT_TRUE(runGemm(1024, 512, 1024)) << "rejecting an entry must not fail the call";

        const auto c = counters();
        EXPECT_GE(c.invalidated, 1u);
        EXPECT_EQ(c.hits, 0u);
    }

    // Re-tuning must be possible once every entry for a shape has gone stale,
    // and the replacement must survive a reload of the file.
    TEST_F(TuningTune_pre_checkin, InvalidatedEntryIsRetunedAndReplays)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        const size_t before = valueRowCount(m_path);
        ASSERT_GT(before, 0u) << "tune mode recorded nothing";
        ASSERT_TRUE(rewriteColumn(m_path, "kernel_name", "NotARealKernelName"));

        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        EXPECT_GT(valueRowCount(m_path), before) << "stale entry was never replaced";

        // Fresh load, as a later process would see it.
        enterMode("cache", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        const auto c = counters();
        EXPECT_GE(c.hits, 1u) << "replacement entry did not replay after reload";
    }

    // A stale row must not suppress the fresh row that supersedes it when the two
    // share a solution index.
    //
    // InvalidatedEntryIsRetunedAndReplays exercises the same append-and-reload
    // path, but only catches this when re-tuning happens to land on a different
    // index, which on a noisy shape it usually does. Here the two rows are the
    // same recorded row with only the name changed, so the collision is
    // guaranteed and the load order is the one the file format produces:
    // superseded row first, replacement after.
    TEST_F(TuningTune_pre_checkin, StaleRowDoesNotSuppressItsReplacement)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        auto lines = readLines(m_path);
        ASSERT_GE(lines.size(), 2u) << "tune mode recorded nothing";

        // Prepend a copy carrying a name no kernel can have. Same index, same
        // problem key, so the only thing separating the two rows is the name.
        ASSERT_TRUE(rewriteColumn(m_path, "kernel_name", "NotARealKernelName"));
        auto stale = readLines(m_path);

        std::ofstream out(m_path, std::ios::trunc);
        for(const auto& l : stale)
            out << l << "\n";
        for(const auto& l : lines)
            out << l << "\n";
        out.close();
        ASSERT_FALSE(out.fail()) << "could not write tuning file " << m_path;

        enterMode("cache", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        const auto c = counters();
        EXPECT_GE(c.hits, 1u) << "stale row hid the valid row that followed it";
    }

    // Sequential tuning on two devices must allocate and reuse scratch in each
    // device's address space rather than handing device 1 the pointer allocated
    // while device 0 was current.
    TEST_F(TuningTune_pre_checkin, ScratchIsDeviceLocal)
    {
        int deviceCount = 0;
        ASSERT_EQ(hipGetDeviceCount(&deviceCount), hipSuccess);
        if(deviceCount < 2)
            GTEST_SKIP() << "A second GPU is required";

        int originalDevice = 0;
        ASSERT_EQ(hipGetDevice(&originalDevice), hipSuccess);
        struct RestoreDevice
        {
            int device;
            ~RestoreDevice()
            {
                static_cast<void>(hipSetDevice(device));
            }
        } restore{originalDevice};

        enterMode("tune", m_path);
        ASSERT_EQ(hipSetDevice(0), hipSuccess);
        ASSERT_TRUE(runGemm(320, 256, 128));
        ASSERT_EQ(hipSetDevice(1), hipSuccess);
        ASSERT_TRUE(runGemm(384, 256, 128));

        EXPECT_EQ(valueRowCount(m_path), 2u);
    }

    // Turning tuning on has to explain the pause it causes, without the user
    // also having to turn on library logging.
    TEST_F(TuningTune_pre_checkin, TuneLifecycleIsVisibleWithoutLogLevel)
    {
        if(!loggingEnvIsClean())
            GTEST_SKIP() << "external logging environment redirects tuning output";

        enterModeForLogging("tune", m_path);

        std::string out;
        {
            CerrCapture capture;
            ASSERT_TRUE(runGemm(256, 256, 256));
            out = capture.str();
        }

        EXPECT_NE(out.find("tuning-cache: mode=tune"), std::string::npos) << out;
        EXPECT_NE(out.find("tuning-start"), std::string::npos) << out;
        EXPECT_NE(out.find("tuning-done"), std::string::npos) << out;

        // Everything below belongs to the info bit and must stay off by default.
        EXPECT_EQ(out.find("progress"), std::string::npos) << out;
        EXPECT_EQ(out.find("setup candidates="), std::string::npos) << out;
        EXPECT_EQ(out.find("cache-hit"), std::string::npos) << out;
        EXPECT_EQ(out.find("cache-miss"), std::string::npos) << out;
    }

    // A matmul-only caller never makes the heuristic query that reports a mode
    // set without a path, so its own call has to explain why nothing is tuned.
    TEST_F(TuningTune_pre_checkin, NullAlgoWithoutPathReportsWhyNothingIsTuned)
    {
        if(!loggingEnvIsClean())
            GTEST_SKIP() << "external logging environment redirects tuning output";

        enterModeForLogging("tune", "");

        std::string out;
        {
            CerrCapture capture;
            int         launched = -1;
            ASSERT_TRUE(runGemmWith(256, 256, 256, AlgoFrom::Null, -1, &launched));
            out = capture.str();
        }

        EXPECT_NE(out.find("load=no-path"), std::string::npos) << out;
        EXPECT_EQ(out.find("tuning-start"), std::string::npos) << out;
        EXPECT_EQ(counters().attempts, 0u);
    }

    // A skip records no winner, so the shape stays uncached and the whole attempt
    // runs again on the next matmul. Unbounded, that is a line per call.
    TEST_F(TuningTune_pre_checkin, RepeatedSkipReportsItsReasonOnce)
    {
        if(!loggingEnvIsClean())
            GTEST_SKIP() << "external logging environment redirects tuning output";

        enterModeForLogging("tune", m_path);

        std::string out;
        {
            CerrCapture capture;
            for(int i = 0; i < 3; i++)
                ASSERT_TRUE(runGemm(384, 256, 128, 2.0f, true, true));
            out = capture.str();
        }

        ASSERT_GE(counters().skipped, 3u) << "the shape was not retried, so nothing was bounded";
        EXPECT_EQ(countOccurrences(out, "tuning-skipped"), 1u) << out;

        // This decline is made before the search is announced, which is what
        // lets it collapse. A skip that had announced one would owe an ending.
        EXPECT_EQ(out.find("tuning-start"), std::string::npos) << out;
    }

    // Bounding a skip must not spend the shape's one terminal line.
    //
    // beta and C/D aliasing are deliberately outside the cache key, so the
    // declined in-place call below and the ordinary call after it are one key
    // with two different outcomes. The second one blocks for the whole search,
    // which is exactly the pause these notices exist to explain.
    TEST_F(TuningTune_pre_checkin, SkipDoesNotSilenceALaterTuneOfTheSameShape)
    {
        if(!loggingEnvIsClean())
            GTEST_SKIP() << "external logging environment redirects tuning output";

        enterModeForLogging("tune", m_path);

        std::string out;
        {
            CerrCapture capture;
            ASSERT_TRUE(runGemm(384, 256, 128, 2.0f, true, true));
            ASSERT_TRUE(runGemm(384, 256, 128));
            out = capture.str();
        }

        ASSERT_GE(counters().skipped, 1u) << "the in-place call was not declined";
        EXPECT_NE(out.find("tuning-skipped"), std::string::npos) << out;
        EXPECT_NE(out.find("tuning-done"), std::string::npos) << out;
    }

    // A winner that could not be written was still tuned: it is in the in-memory
    // cache and serves every later call in this process. Reporting tuned=0 next
    // to a "tuning-done ... persisted=no" line contradicts it.
    TEST_F(TuningTune_pre_checkin, TuneThatCannotPersistIsStillCountedAsTuned)
    {
        // No such directory, so the load finds nothing and the append cannot
        // create the file. Tuning itself is unaffected.
        const std::string unwritable = "/nonexistent-hipblaslt-review-dir/cache.tuning";

        enterModeForLogging("tune", unwritable);
        ASSERT_TRUE(runGemm(256, 256, 256));

        const auto tally = counters();

        EXPECT_EQ(tally.tunedShapes, 1u)
            << "a successful tune vanished from the summary because the "
               "winner could not be written";

        // The persist-only counter is what stayed at zero, and that is the
        // distinction the outcome line reports.
        EXPECT_EQ(counters().tuned, 0u) << "the winner was somehow written to " << unwritable;
    }

    // A tuned entry closes the gate, or every tune-mode process would pay the
    // whole search again on a warm cache.
    TEST_F(TuningTune_pre_checkin, TunedEntryIsNotRetuned)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        const size_t before = valueRowCount(m_path);
        ASSERT_GT(before, 0u) << "tune mode recorded nothing";

        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        EXPECT_EQ(valueRowCount(m_path), before) << "a tuned entry was re-tuned";
    }

    // A search the budget cuts short records nothing and is attempted once per
    // process. The retry would run under the same ceiling and stop in the same
    // place, so leaving it unlatched spends the whole ceiling on every call for
    // the life of the process rather than once.
    TEST_F(TuningTune_pre_checkin, BudgetExhaustedShapeIsNotRetried)
    {
        enterMode("tune", m_path);
        setenv("HIPBLASLT_TUNING_BUDGET_MS_PER_SHAPE", "1", 1);
        hipblaslt_tuning_reset_for_test();
        // Read as spent at once, however quickly this machine runs the search.
        hipblaslt_tuning_inject_failure_for_test(5);

        for(int i = 0; i < 3; i++)
            ASSERT_TRUE(runGemm(1024, 512, 1024));

        const auto c = counters();
        EXPECT_EQ(c.skipped, 1u)
            << "the shape re-tuned after its budget ran out, so every call pays the ceiling";
        EXPECT_EQ(c.tuned, 0u);
        EXPECT_EQ(valueRowCount(m_path), 0u) << "a search the budget stopped recorded a row";
    }

    // Stopped after one measured candidate, a search has a best candidate, but
    // only of an arbitrary prefix, so it still records nothing.
    TEST_F(TuningTune_pre_checkin, SearchTheBudgetStopsAfterMeasuringRecordsNothing)
    {
        // The stop is asked before each candidate, so a search of one candidate
        // finishes before it is asked again.
        if(Fp16Problem(1024, 512, 1024).offered(2).size() < 2)
            GTEST_SKIP() << "the heuristic offers one algorithm for this problem";
        enterMode("tune", m_path);
        hipblaslt_tuning_inject_failure_for_test(4);

        int launched = -1;
        for(int i = 0; i < 2; i++)
            ASSERT_TRUE(runGemmWith(1024, 512, 1024, AlgoFrom::Null, -1, &launched, true))
                << "a stopped search failed the matmul";

        const auto c = counters();
        EXPECT_EQ(c.attempts, 1u) << "the stopped search was repeated";
        EXPECT_EQ(c.skipped, 1u);
        EXPECT_EQ(c.tuned, 0u);
        EXPECT_EQ(valueRowCount(m_path), 0u);
    }

    // One stale row is met three times in a tune-mode call: the heuristic
    // lookup, the execution probe, and the recheck under the tuning lock.
    TEST_F(TuningTune_pre_checkin, StaleEntryIsCountedAsOneInvalidation)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        ASSERT_EQ(valueRowCount(m_path), 1u) << "expected exactly one tuned row to spoil";
        ASSERT_TRUE(rewriteColumn(m_path, "kernel_name", "NotARealKernelName"));

        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        EXPECT_EQ(counters().invalidated, 1u) << "one rejected entry was counted more than once";
    }

    // Query once and reuse the algo, the pattern the API documents. Tuning on the
    // first call records a winner for later lookups, but must not change what
    // that algo launches, on the tuning call or any after it.
    TEST_F(TuningTune_pre_checkin, ReusedAlgoKeepsLaunchingItsOwnKernel)
    {
        const auto identities = candidateIdentities(1024, 512, 1024, 1);
        ASSERT_FALSE(identities.empty());
        const int given = identities[0].first;

        enterMode("tune", m_path);
        for(int call = 0; call < 2; call++)
        {
            int launched = -1;
            ASSERT_TRUE(runGemmWith(1024, 512, 1024, AlgoFrom::Index, given, &launched, true));
            EXPECT_EQ(launched, given) << "call " << call << " did not launch the caller's algo";
        }

        EXPECT_EQ(valueRowCount(m_path), 1u) << "the first call did not tune";
        EXPECT_EQ(counters().hits, 0u);
    }

    // With no algo, the call that tunes is the one that gets to use the winner,
    // on the caller's buffers.
    TEST_F(TuningTune_pre_checkin, NullAlgoLaunchesTheWinnerOnTheTuningCall)
    {
        enterMode("tune", m_path);
        int launched = -1;
        ASSERT_TRUE(runGemmWith(128, 128, 128, AlgoFrom::Null, -1, &launched, true));

        const auto winners = columnValues(m_path, "solution_index");
        ASSERT_EQ(winners.size(), 1u) << "tune mode recorded nothing";
        EXPECT_EQ(launched, std::stoi(winners[0]));
    }

    // A search that announced itself and then failed is not started again in
    // this process. At the default log level a repeat would be a silent stall,
    // since a shape's start and its failure are each reported once.
    TEST_F(TuningTune_pre_checkin, FailedSearchIsNotRepeated)
    {
        // Setup, enumeration, and an exception after one candidate was measured.
        for(int stage = 1; stage <= 3; stage++)
        {
            SCOPED_TRACE("failure stage " + std::to_string(stage));
            std::remove(m_path.c_str());
            enterMode("tune", m_path);
            hipblaslt_tuning_inject_failure_for_test(stage);

            for(int call = 0; call < 3; call++)
                ASSERT_TRUE(runGemm(256, 256, 256)) << "a failed search failed the matmul";

            EXPECT_EQ(counters().attempts, 1u);
            EXPECT_EQ(valueRowCount(m_path), 0u);
        }
    }

    // A call whose search throws still launched default selection, so it is a
    // miss and its shape one that fell back, like any other call the cache
    // could not serve.
    TEST_F(TuningTune_pre_checkin, CallWhoseSearchThrowsIsStillCounted)
    {
        enterMode("tune", m_path);
        hipblaslt_tuning_inject_failure_for_test(3);

        int launched = -1;
        ASSERT_TRUE(runGemmWith(256, 256, 256, AlgoFrom::Null, -1, &launched))
            << "a failed search failed the matmul";
        EXPECT_EQ(counters().attempts, 1u);

        const auto c = counters();
        EXPECT_EQ(c.hits, 0u);
        EXPECT_EQ(c.misses, 1u);
        EXPECT_EQ(c.shapes, 1u);
        EXPECT_EQ(c.fellback, 1u);
    }

    // Two threads that meet two untuned shapes at the same time both get them
    // tuned. Tune mode was asked for, so what it records must not depend on
    // which thread reached the tuning lock first.
    TEST_F(TuningTune_pre_checkin, ConcurrentShapesAreAllTuned)
    {
        enterMode("tune", m_path);

        std::atomic<int> arrived{0};
        bool             ok[2] = {false, false};
        auto             run   = [&](int slot, int64_t m) {
            arrived++;
            while(arrived.load() < 2)
                std::this_thread::yield();
            ok[slot] = runGemm(m, 256, 128);
        };

        std::thread first(run, 0, 320);
        std::thread second(run, 1, 384);
        first.join();
        second.join();

        ASSERT_TRUE(ok[0] && ok[1]);
        EXPECT_EQ(valueRowCount(m_path), 2u)
            << "a shape went untuned because another thread held the tuning lock";
    }

    // Two threads meeting one untuned shape at once, with no algo: one tunes,
    // the other waits for it, and both launch the winner. The thread that waited
    // is served by the cache, so it counts as a hit rather than a second miss.
    TEST_F(TuningTune_pre_checkin, ThreadThatWaitedForTheSearchLaunchesTheWinner)
    {
        enterMode("tune", m_path);

        std::atomic<int> arrived{0};
        bool             ok[2]       = {false, false};
        int              launched[2] = {-1, -1};
        auto             run         = [&](int slot) {
            arrived++;
            while(arrived.load() < 2)
                std::this_thread::yield();
            ok[slot] = runGemmWith(1024, 512, 1024, AlgoFrom::Null, -1, &launched[slot]);
        };

        std::thread first(run, 0);
        std::thread second(run, 1);
        first.join();
        second.join();

        ASSERT_TRUE(ok[0] && ok[1]);
        const auto winners = columnValues(m_path, "solution_index");
        ASSERT_EQ(winners.size(), 1u) << "the shape was not tuned exactly once";
        EXPECT_EQ(launched[0], std::stoi(winners[0]));
        EXPECT_EQ(launched[1], std::stoi(winners[0]));
        EXPECT_EQ(counters().attempts, 1u);

        const auto c = counters();
        EXPECT_EQ(c.hits, 1u);
        EXPECT_EQ(c.misses, 1u);
    }

    // A conjugate transpose is keyed apart from a plain one. Both are "not N",
    // which is all the historical key records, so without the distinction a row
    // tuned for one would serve the other.
    TEST_F(TuningTune_pre_checkin, ConjugateTransposeRowDoesNotServeATranspose)
    {
        int launched = -1;

        enterMode("tune", m_path);
        ASSERT_TRUE(
            runGemmWith(1024, 512, 1024, AlgoFrom::Heuristic, -1, &launched, false, HIPBLAS_OP_T));
        ASSERT_EQ(valueRowCount(m_path), 1u) << "tune mode recorded nothing";
        ASSERT_EQ(columnValues(m_path, "transA").at(0), "T");

        enterMode("cache", m_path);
        ASSERT_TRUE(
            runGemmWith(1024, 512, 1024, AlgoFrom::Heuristic, -1, &launched, false, HIPBLAS_OP_T));
        ASSERT_GE(counters().hits, 1u) << "the transposed row did not serve its own problem";

        ASSERT_TRUE(rewriteColumn(m_path, "transA", "C"));
        enterMode("cache", m_path);
        ASSERT_TRUE(
            runGemmWith(1024, 512, 1024, AlgoFrom::Heuristic, -1, &launched, false, HIPBLAS_OP_T));

        const auto c = counters();
        EXPECT_EQ(c.loaded, 1u) << "a conjugate-transpose row was rejected rather than keyed";
        EXPECT_EQ(c.hits, 0u) << "a conjugate-transpose row served a plain transpose";
    }

    // The C++ extension builds its key when the gemm is created, and only later
    // does the preference set the stream-K tile mode, which is part of the key.
    // A row tuned with that mode through the C API must serve the extension
    // asking with the same mode, and must not serve it asking with another.
    TEST_F(TuningTune_pre_checkin, ExtensionLookupKeysOnThePreferenceStreamKMode)
    {
        const int32_t autoMode = HIPBLASLT_STREAMK_TILE_SCHEDULING_AUTO;
        int           launched = -1;

        enterMode("tune", m_path);
        ASSERT_TRUE(runGemmWith(
            1024, 512, 1024, AlgoFrom::Heuristic, -1, &launched, false, HIPBLAS_OP_N, autoMode));
        ASSERT_EQ(valueRowCount(m_path), 1u) << "tune mode recorded nothing";
        ASSERT_EQ(columnValues(m_path, "streamk_tile_scheduling").at(0), std::to_string(autoMode));

        enterMode("cache", m_path);
        EXPECT_TRUE(extensionHeuristicHits(1024, 512, 1024, autoMode))
            << "the extension looked up the mode its gemm was created with, not the one its "
               "preference set";
        EXPECT_FALSE(
            extensionHeuristicHits(1024, 512, 1024, HIPBLASLT_STREAMK_TILE_SCHEDULING_OFF));
    }

    // A scalar scaling type with no scale scales nothing, the same as the unset
    // type the C API leaves there, so a row tuned through the C API must serve
    // an extension caller that sets it. hipblaslt-bench's C++ API run is one,
    // and this is how it finds what its tuning pass recorded.
    TEST_F(TuningTune_pre_checkin, ExtensionScalarScalingWithoutScalesFindsTheCApiRow)
    {
        int launched = -1;

        enterMode("tune", m_path);
        ASSERT_TRUE(runGemmWith(1024, 512, 1024, AlgoFrom::Heuristic, -1, &launched));
        ASSERT_EQ(valueRowCount(m_path), 1u) << "tune mode recorded nothing";

        enterMode("cache", m_path);
        EXPECT_TRUE(extensionHeuristicHits(1024, 512, 1024, -1, true))
            << "the scaling type was keyed although nothing is scaled";
    }

    // The scratch plan decides what memory a candidate can touch. An output
    // span that is too small lets a kernel write past its block, and an input
    // span that is only an upper bound must never become the length of a copy
    // from the caller's buffer. These cases check the plan's numbers directly,
    // which a measurement cannot show.

    struct Span
    {
        size_t bytes    = 0;
        bool   expanded = false;
    };

    Span tensorSpan(size_t elementSize,
                    size_t rows,
                    size_t cols,
                    size_t colStride,
                    size_t batchCount  = 1,
                    size_t batchStride = 0)
    {
        int          expanded = -1;
        const size_t bytes    = hipblaslt_tuning_tensor_span_for_test(
            elementSize, rows, cols, colStride, batchCount, batchStride, &expanded);
        EXPECT_NE(expanded, -1) << "the span did not report whether it rounded a stride up";
        return {bytes, expanded == 1};
    }

    TEST(TuningScratchSpan_pre_checkin, SpanFollowsTheStridesGiven)
    {
        // The last column ends rows past the start of the last column, and the
        // last batch likewise after the start of the last batch.
        EXPECT_EQ(tensorSpan(2, 8, 4, 8).bytes, 64u);
        EXPECT_EQ(tensorSpan(2, 8, 4, 10).bytes, (10u * 3 + 8) * 2);
        EXPECT_EQ(tensorSpan(2, 8, 4, 8, 3, 40).bytes, (40u * 2 + 32) * 2);
        EXPECT_FALSE(tensorSpan(2, 8, 4, 10, 3, 40).expanded);
    }

    // A stride too small to keep columns or batches apart is rounded up to
    // dense and flagged: as a span to write into that is safe, but the caller's
    // buffer is smaller than it.
    TEST(TuningScratchSpan_pre_checkin, OverlappingStridesAreRoundedUpAndFlagged)
    {
        const auto narrowColumns = tensorSpan(2, 8, 4, 5);
        EXPECT_EQ(narrowColumns.bytes, 64u);
        EXPECT_TRUE(narrowColumns.expanded);

        const auto broadcast = tensorSpan(2, 8, 4, 8, 3, 0);
        EXPECT_EQ(broadcast.bytes, 3u * 32 * 2);
        EXPECT_TRUE(broadcast.expanded);

        const auto overlapping = tensorSpan(2, 8, 4, 8, 3, 16);
        EXPECT_EQ(overlapping.bytes, 3u * 32 * 2);
        EXPECT_TRUE(overlapping.expanded);

        EXPECT_FALSE(tensorSpan(2, 8, 4, 8, 1, 0).expanded)
            << "a batch stride was rounded up for a single batch, which never uses it";
    }

    // A span too large for size_t is 0, never a wrapped value small enough to
    // pass for a real one.
    TEST(TuningScratchSpan_pre_checkin, SpanThatOverflowsIsZero)
    {
        const size_t most = std::numeric_limits<size_t>::max();
        EXPECT_EQ(tensorSpan(1, 2, 3, most / 2).bytes, 0u) << "columns";
        EXPECT_EQ(tensorSpan(1, 4, 1, 4, 3, most / 2).bytes, 0u) << "batches";
        EXPECT_EQ(tensorSpan(8, most / 4, 1, most / 4).bytes, 0u) << "bytes";
        EXPECT_EQ(tensorSpan(2, 0, 4, 8).bytes, 0u) << "an empty tensor";
    }

    /** A matmul for the scratch plan, fp16 throughout, as a caller describes it. */
    struct PlanCase
    {
        int64_t            m = 64, n = 32, k = 128;
        hipblasOperation_t opA = HIPBLAS_OP_N, opB = HIPBLAS_OP_N;

        // Leading dimensions, where 0 is dense, and batch strides, where -1 is
        // dense.
        int64_t lda = 0, ldb = 0, ldc = 0, ldd = 0;
        int32_t batch   = 1;
        int64_t strideA = -1, strideB = -1, strideC = -1, strideD = -1;

        bool                inPlace    = false;
        float               beta       = 0.0f;
        hipblasLtEpilogue_t epilogue   = HIPBLASLT_EPILOGUE_DEFAULT;
        int32_t             biasStride = 0;
        int64_t             auxStride  = -1;
        int64_t             auxLD      = 0;

        // A scaled by an M vector and B by an N vector, and alpha a vector.
        bool vectorScales = false;
        bool alphaVector  = false;

        size_t  workspaceBytes = 0;
        size_t  rotatingBytes  = 0;
        size_t  cap            = std::numeric_limits<size_t>::max();
        int32_t coldIters      = 1000;
        int32_t hotIters       = 1000;
    };

    struct Plan
    {
        bool     usable = false;
        uint64_t blocks = 0, total = 0;
        uint64_t bytesA = 0, bytesB = 0, bytesC = 0, bytesD = 0, bytesInPlaceC = 0, bytesE = 0,
                 bytesBias = 0, bytesBiasIn = 0, copyBiasIn = 0, bytesScaleA = 0, bytesScaleB = 0,
                 bytesScaleAlphaVec = 0;
    };

    bool planFor(const PlanCase& c, Plan* plan)
    {
        hipblasLtHandle_t handle = nullptr;
        if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
            return false;

        hipblasLtMatrixLayout_t layouts[4] = {};
        hipblasLtMatmulDesc_t   desc       = nullptr;

        auto layout = [&](hipblasLtMatrixLayout_t* out,
                          int64_t                  rows,
                          int64_t                  cols,
                          int64_t                  leading,
                          int64_t                  stride) {
            const int64_t ld          = leading ? leading : rows;
            const int64_t batchStride = stride >= 0 ? stride : ld * cols;
            return hipblasLtMatrixLayoutCreate(out, HIP_R_16F, rows, cols, ld)
                       == HIPBLAS_STATUS_SUCCESS
                   && hipblasLtMatrixLayoutSetAttribute(
                          *out, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &c.batch, sizeof(c.batch))
                          == HIPBLAS_STATUS_SUCCESS
                   && hipblasLtMatrixLayoutSetAttribute(
                          *out,
                          HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                          &batchStride,
                          sizeof(batchStride))
                          == HIPBLAS_STATUS_SUCCESS;
        };

        // Only compared, never dereferenced, so any distinct addresses do.
        static char    buffers[8];
        const void*    A      = &buffers[0];
        const void*    B      = &buffers[1];
        const void*    C      = &buffers[2];
        void*          D      = c.inPlace ? &buffers[2] : &buffers[3];
        void*          bias   = &buffers[4];
        void*          aux    = &buffers[5];
        void*          scaleA = &buffers[6];
        void*          scaleB = &buffers[7];
        const bool     nA   = c.opA == HIPBLAS_OP_N;
        const bool     nB   = c.opB == HIPBLAS_OP_N;
        const int32_t  opA = c.opA, opB = c.opB;
        const uint32_t epilogue = c.epilogue;
        const int32_t  biasType = HIP_R_32F;

        bool ok = layout(&layouts[0], nA ? c.m : c.k, nA ? c.k : c.m, c.lda, c.strideA)
                  && layout(&layouts[1], nB ? c.k : c.n, nB ? c.n : c.k, c.ldb, c.strideB)
                  && layout(&layouts[2], c.m, c.n, c.ldc, c.strideC)
                  && layout(&layouts[3], c.m, c.n, c.ldd, c.strideD)
                  && hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                         == HIPBLAS_STATUS_SUCCESS
                  && hipblasLtMatmulDescSetAttribute(
                         desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA))
                         == HIPBLAS_STATUS_SUCCESS
                  && hipblasLtMatmulDescSetAttribute(
                         desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB))
                         == HIPBLAS_STATUS_SUCCESS
                  && hipblasLtMatmulDescSetAttribute(
                         desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue))
                         == HIPBLAS_STATUS_SUCCESS;
        if(ok && c.epilogue != HIPBLASLT_EPILOGUE_DEFAULT)
            ok = hipblasLtMatmulDescSetAttribute(
                     desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias))
                     == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulDescSetAttribute(
                        desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &biasType, sizeof(biasType))
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulDescSetAttribute(desc,
                                                    HIPBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE,
                                                    &c.biasStride,
                                                    sizeof(c.biasStride))
                        == HIPBLAS_STATUS_SUCCESS;

        if(ok
           && (c.epilogue == HIPBLASLT_EPILOGUE_DGELU || c.epilogue == HIPBLASLT_EPILOGUE_GELU_AUX))
        {
            const int64_t ld     = c.auxLD ? c.auxLD : c.m;
            const int64_t stride = c.auxStride >= 0 ? c.auxStride : ld * c.n;
            ok                   = hipblasLtMatmulDescSetAttribute(
                     desc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &aux, sizeof(aux))
                     == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulDescSetAttribute(
                        desc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ld, sizeof(ld))
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulDescSetAttribute(desc,
                                                    HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE,
                                                    &stride,
                                                    sizeof(stride))
                        == HIPBLAS_STATUS_SUCCESS;
        }

        if(ok && c.vectorScales)
        {
            const int32_t vector = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
            ok                   = hipblasLtMatmulDescSetAttribute(
                     desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scaleA, sizeof(scaleA))
                     == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulDescSetAttribute(
                        desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &vector, sizeof(vector))
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulDescSetAttribute(
                        desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scaleB, sizeof(scaleB))
                        == HIPBLAS_STATUS_SUCCESS
                 && hipblasLtMatmulDescSetAttribute(
                        desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &vector, sizeof(vector))
                        == HIPBLAS_STATUS_SUCCESS;
        }

        if(ok && c.alphaVector)
        {
            const int32_t mode = HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
            ok                 = hipblasLtMatmulDescSetAttribute(
                     desc, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &mode, sizeof(mode))
                 == HIPBLAS_STATUS_SUCCESS;
        }

        uint64_t v[15] = {};
        ok             = ok
             && hipblaslt_tuning_scratch_plan_for_test(handle,
                                                       desc,
                                                       layouts[0],
                                                       layouts[1],
                                                       layouts[2],
                                                       layouts[3],
                                                       A,
                                                       B,
                                                       C,
                                                       D,
                                                       c.beta,
                                                       c.workspaceBytes,
                                                       c.rotatingBytes,
                                                       c.cap,
                                                       c.coldIters,
                                                       c.hotIters,
                                                       v,
                                                       15)
                    >= 15;
        if(ok)
            *plan = {v[0] != 0,
                     v[1],
                     v[2],
                     v[3],
                     v[4],
                     v[5],
                     v[6],
                     v[7],
                     v[8],
                     v[9],
                     v[10],
                     v[11],
                     v[12],
                     v[13],
                     v[14]};

        if(desc)
            hipblasLtMatmulDescDestroy(desc);
        for(auto l : layouts)
            if(l)
                hipblasLtMatrixLayoutDestroy(l);
        hipblasLtDestroy(handle);
        return ok;
    }

    class TuningScratchPlan_pre_checkin : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // The plan needs no GPU work, but a handle needs a device.
            if(!gpuAvailable())
                GTEST_SKIP() << "No GPU available";
        }
    };

    // Each tensor is sized from its own leading dimension, including C and D,
    // which need not share one.
    TEST_F(TuningScratchPlan_pre_checkin, EachSpanFollowsItsOwnLeadingDimension)
    {
        PlanCase padded;
        padded.lda = 80;
        padded.ldb = 136;
        padded.ldc = 72;
        padded.ldd = 96;

        Plan plan;
        ASSERT_TRUE(planFor(padded, &plan));
        ASSERT_TRUE(plan.usable);
        EXPECT_EQ(plan.bytesA, (80u * 127 + 64) * 2);
        EXPECT_EQ(plan.bytesB, (136u * 31 + 128) * 2);
        EXPECT_EQ(plan.bytesC, (72u * 31 + 64) * 2);
        EXPECT_EQ(plan.bytesD, (96u * 31 + 64) * 2);
    }

    // In place, C is read through D's block, so that block covers whichever
    // of the two reaches further.
    TEST_F(TuningScratchPlan_pre_checkin, InPlaceBlockCoversTheWiderOfCAndD)
    {
        PlanCase inPlace;
        inPlace.inPlace = true;
        inPlace.ldc     = 96;

        Plan plan;
        ASSERT_TRUE(planFor(inPlace, &plan));
        ASSERT_TRUE(plan.usable);
        EXPECT_EQ(plan.bytesInPlaceC, (96u * 31 + 64) * 2);
        EXPECT_EQ(plan.bytesD, plan.bytesInPlaceC);
        EXPECT_EQ(plan.bytesC, 0u) << "C has a block of its own as well as D's";
    }

    // A broadcast input's span is only an upper bound on the caller's buffer,
    // so it is never copied for rotation: the search measures with one block
    // rather than reading past the end of a one-batch allocation.
    TEST_F(TuningScratchPlan_pre_checkin, BroadcastInputIsNotRotated)
    {
        PlanCase batched;
        batched.batch         = 2;
        batched.rotatingBytes = size_t(64) << 20;

        Plan dense;
        ASSERT_TRUE(planFor(batched, &dense));
        ASSERT_TRUE(dense.usable);
        ASSERT_GT(dense.blocks, 1u) << "a dense batched problem does not rotate either";

        for(const char* input : {"A", "C"})
        {
            SCOPED_TRACE(input);
            PlanCase broadcast                                        = batched;
            (input[0] == 'A' ? broadcast.strideA : broadcast.strideC) = 0;

            Plan plan;
            ASSERT_TRUE(planFor(broadcast, &plan));
            ASSERT_TRUE(plan.usable);
            EXPECT_EQ(plan.blocks, 1u);
        }
    }

    TEST_F(TuningScratchPlan_pre_checkin, GradientAuxiliaryOverlapDoesNotCopyAnExpandedSpan)
    {
        PlanCase c;
        c.batch     = 2;
        c.auxLD     = c.m + 8;
        c.auxStride = c.m * c.n;
        c.epilogue  = HIPBLASLT_EPILOGUE_DGELU;
        Plan plan;
        ASSERT_TRUE(planFor(c, &plan));
        EXPECT_FALSE(plan.usable);
        c.auxStride = c.auxLD * c.n;
        ASSERT_TRUE(planFor(c, &plan));
        EXPECT_TRUE(plan.usable);
        c.auxStride = c.m * c.n;
        c.epilogue  = HIPBLASLT_EPILOGUE_GELU_AUX;
        ASSERT_TRUE(planFor(c, &plan));
        EXPECT_TRUE(plan.usable) << "Writable forward auxiliary scratch can be expanded";
    }

    // A gradient epilogue writes the bias, as long as the longer side of D,
    // once per batch at the bias batch stride.
    TEST_F(TuningScratchPlan_pre_checkin, GradientBiasCoversEveryBatchAtItsStride)
    {
        PlanCase grad;
        grad.epilogue   = HIPBLASLT_EPILOGUE_BGRADB;
        grad.batch      = 2;
        grad.biasStride = 80;

        Plan plan;
        ASSERT_TRUE(planFor(grad, &plan));
        ASSERT_TRUE(plan.usable);
        EXPECT_EQ(plan.bytesBias, (80u + 64) * sizeof(float));
    }

    // A cap below the full rotation trims blocks rather than declining. A cap
    // below a single block is left for the scratch allocation to refuse.
    TEST_F(TuningScratchPlan_pre_checkin, CapTrimsTheRotation)
    {
        PlanCase c;
        c.rotatingBytes = size_t(64) << 20;

        Plan full;
        ASSERT_TRUE(planFor(c, &full));
        ASSERT_TRUE(full.usable);
        ASSERT_GE(full.blocks, 4u);

        // hipblaslt-bench's footprint for this dense problem: A, B and D, as a
        // zero beta leaves C unread.
        const uint64_t footprint = full.bytesA + full.bytesB + full.bytesD;

        PlanCase threeBlocks      = c;
        threeBlocks.rotatingBytes = 3 * footprint;
        Plan three;
        ASSERT_TRUE(planFor(threeBlocks, &three));
        ASSERT_EQ(three.blocks, 3u);

        PlanCase capped = c;
        capped.cap      = three.total;
        Plan plan;
        ASSERT_TRUE(planFor(capped, &plan));
        EXPECT_TRUE(plan.usable);
        EXPECT_EQ(plan.blocks, 3u);
        EXPECT_LE(plan.total, capped.cap);

        capped.cap = three.total - 1;
        ASSERT_TRUE(planFor(capped, &plan));
        EXPECT_EQ(plan.blocks, 2u);
        EXPECT_LE(plan.total, capped.cap);

        capped.cap = 1;
        ASSERT_TRUE(planFor(capped, &plan));
        EXPECT_TRUE(plan.usable);
        EXPECT_EQ(plan.blocks, 1u);
        EXPECT_GT(plan.total, capped.cap);
    }

    // With C and D apart, fp16 1024^3 occupies 6 MiB for A, B and D, and
    // 8 MiB once beta reads C. These are actual tensor bytes; the benchmark's
    // C estimate uses sizeof(hipDataType) instead of the element size.
    TEST_F(TuningScratchPlan_pre_checkin, BlockCountUsesActualTensorBytes)
    {
        PlanCase c;
        c.m = c.n = c.k = 1024;
        c.rotatingBytes = size_t(512) << 20;

        Plan plan;
        ASSERT_TRUE(planFor(c, &plan));
        ASSERT_TRUE(plan.usable);
        EXPECT_EQ(plan.blocks, 86u) << "512 MiB over 6 MiB, rounded up";

        c.beta = 1.0f;
        ASSERT_TRUE(planFor(c, &plan));
        ASSERT_TRUE(plan.usable);
        EXPECT_EQ(plan.blocks, 64u) << "512 MiB over 8 MiB";
    }

    // A bias the epilogue reads, vector scales and the alpha vector rotate with
    // the tensors and count toward the block count, as hipblaslt-bench's do.
    // Only the caller's extent of the bias is copied into its block.
    TEST_F(TuningScratchPlan_pre_checkin, ReadVectorsRotateWithTheTensors)
    {
        PlanCase c;
        c.epilogue     = HIPBLASLT_EPILOGUE_BIAS;
        c.vectorScales = true;
        c.alphaVector  = true;

        Plan bare;
        ASSERT_TRUE(planFor(c, &bare));
        ASSERT_TRUE(bare.usable);
        EXPECT_EQ(bare.bytesBias, 0u) << "a read bias took a written one's block";
        EXPECT_EQ(bare.bytesBiasIn, std::max(c.m, c.n) * sizeof(float));
        EXPECT_EQ(bare.copyBiasIn, c.m * sizeof(float)) << "a bias added along D is M long";
        EXPECT_EQ(bare.bytesScaleA, c.m * sizeof(float));
        EXPECT_EQ(bare.bytesScaleB, c.n * sizeof(float));
        EXPECT_EQ(bare.bytesScaleAlphaVec, c.m * sizeof(float));

        // A, B and D, then the bias, the two scales and the alpha vector.
        const uint64_t footprint
            = bare.bytesA + bare.bytesB + bare.bytesD + (3 * c.m + c.n) * sizeof(float);
        c.rotatingBytes = 20 * footprint;
        Plan plan;
        ASSERT_TRUE(planFor(c, &plan));
        ASSERT_TRUE(plan.usable);
        EXPECT_EQ(plan.blocks, 20u);

        // Once per batch at its stride, or a single vector when that is zero.
        c.batch = 2;
        ASSERT_TRUE(planFor(c, &plan));
        ASSERT_TRUE(plan.usable);
        EXPECT_EQ(plan.copyBiasIn, c.m * sizeof(float));
        c.biasStride = 80;
        ASSERT_TRUE(planFor(c, &plan));
        ASSERT_TRUE(plan.usable);
        EXPECT_EQ(plan.copyBiasIn, (80 + c.m) * sizeof(float));
    }

    // Every block shares one workspace, so however large the caller's limit,
    // it takes no rotating budget from the tensors.
    TEST_F(TuningScratchPlan_pre_checkin, WorkspaceIsSharedByEveryBlock)
    {
        PlanCase c;
        c.rotatingBytes = size_t(1) << 20;

        Plan bare;
        ASSERT_TRUE(planFor(c, &bare));
        ASSERT_TRUE(bare.usable);
        ASSERT_GT(bare.blocks, 1u);

        PlanCase withWorkspace       = c;
        withWorkspace.workspaceBytes = size_t(128) << 20;
        Plan plan;
        ASSERT_TRUE(planFor(withWorkspace, &plan));
        ASSERT_TRUE(plan.usable);
        EXPECT_EQ(plan.blocks, bare.blocks);
        EXPECT_EQ(plan.total, bare.total + withWorkspace.workspaceBytes);
    }

    // However small the problem, there are never more rotation blocks than the
    // larger of a candidate's cold and timed launch counts, as in hipblaslt-bench:
    // each block costs a solve per candidate, and blocks past the last launch
    // would go unused.
    TEST_F(TuningScratchPlan_pre_checkin, RotationIsBoundedByTheLaunchCount)
    {
        PlanCase tiny;
        tiny.m = tiny.n = tiny.k = 16;
        tiny.rotatingBytes       = size_t(1) << 30;
        tiny.coldIters           = 200;
        tiny.hotIters            = 300;

        Plan plan;
        ASSERT_TRUE(planFor(tiny, &plan));
        ASSERT_TRUE(plan.usable);
        EXPECT_EQ(plan.blocks, 300u);

        tiny.coldIters = 400;
        ASSERT_TRUE(planFor(tiny, &plan));
        EXPECT_EQ(plan.blocks, 400u);
    }

    // A layout whose span does not fit in size_t is declined rather than
    // sized from a wrapped value. hipBLASLt refuses sizes, leading dimensions
    // and batch strides of 2^32 elements or more, so both cases stay below it.
    TEST_F(TuningScratchPlan_pre_checkin, ShapeWhoseSpanOverflowsIsDeclined)
    {
        const int64_t largest = (int64_t(1) << 32) - 1;

        PlanCase large;
        large.batch   = 4;
        large.strideD = largest;
        Plan plan;
        ASSERT_TRUE(planFor(large, &plan));
        ASSERT_TRUE(plan.usable) << "a large but representable stride is declined too";

        PlanCase wide;
        wide.n = largest;
        ASSERT_TRUE(planFor(wide, &plan));
        ASSERT_TRUE(plan.usable) << "the widest D hipBLASLt accepts is declined too";

        // With its leading dimension at the limit too, D's span is about 2^64
        // halves. Its single batch never uses the stride.
        PlanCase overflowing = wide;
        overflowing.ldd      = largest;
        overflowing.strideD  = 0;
        ASSERT_TRUE(planFor(overflowing, &plan));
        EXPECT_FALSE(plan.usable);
        EXPECT_EQ(plan.bytesD, 0u);
    }
} // namespace
