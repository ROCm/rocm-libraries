/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

// User-driven tuning cache tests.
//
// Everything here drives the public API with HIPBLASLT_TUNING_MODE and
// HIPBLASLT_TUNING_CACHE_PATH, and asserts on the library's own hit / miss /
// invalidated / tuned counters rather than on log text.
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

extern "C" void hipblaslt_tuning_reset_for_test();
extern "C" void hipblaslt_tuning_counters_for_test(uint64_t* loaded,
                                                   uint64_t* hits,
                                                   uint64_t* misses,
                                                   uint64_t* invalidated,
                                                   uint64_t* tuned,
                                                   uint64_t* skipped);
extern "C" void hipblaslt_tuning_lookup_tally_for_test(uint64_t* shapes,
                                                       uint64_t* matched,
                                                       uint64_t* fellback,
                                                       uint64_t* tuned);
extern "C" int      hipblaslt_tuning_last_launch_for_test();
extern "C" uint64_t hipblaslt_tuning_attempts_for_test();
extern "C" void     hipblaslt_tuning_inject_failure_for_test(int stage);
extern "C" void     hipblaslt_tuning_reset_as_privileged_for_test();

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

    struct Counters
    {
        uint64_t loaded = 0, hits = 0, misses = 0, invalidated = 0, tuned = 0, skipped = 0;
    };

    Counters counters()
    {
        Counters c;
        hipblaslt_tuning_counters_for_test(
            &c.loaded, &c.hits, &c.misses, &c.invalidated, &c.tuned, &c.skipped);
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
        oss << "hipblaslt_" << stem << "_" << static_cast<long long>(pid) << ".tuning";
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
        // one. runGemm asks for a 32 MiB workspace, and the workspace is part of
        // the per-block footprint, so a budget below that silently disables
        // rotation and leaves the seeding and per-block pointer arithmetic
        // untested. Still far below the 512 MiB default.
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
            const uint64_t maxWs = wsBytes;
            const bool     plainA = opA == HIPBLAS_OP_N;
            ok = hipblasLtMatrixLayoutCreate(
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
                     || hipblasLtMatmulDescSetAttribute(desc,
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
            ok = hipblasLtMatmulAlgoGetHeuristic(
                     handle, desc, layoutA, layoutB, layoutC, layoutC, pref, 1, heuristic, &returned)
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
            ok = hipblaslt_ext::getAlgosFromIndex(handle, indexes, results) == HIPBLAS_STATUS_SUCCESS
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
    bool rewriteColumn(const std::string& path, const std::string& column, const std::string& value)
    {
        auto lines = readLines(path);
        if(lines.empty())
            return false;

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
            return false;

        std::ofstream out(path, std::ios::trunc);
        for(const auto& l : lines)
            out << l << "\n";
        return true;
    }

    /**
     * Delete one named column, header cell and value cell, from every row.
     *
     * Produces what a file written by a build that predates the column looks
     * like, which is the shape of every cache a user upgrades with.
     */
    bool dropColumn(const std::string& path, const std::string& column)
    {
        auto lines = readLines(path);
        if(lines.empty())
            return false;

        auto split = [](const std::string& s) {
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
        };

        auto join = [](const std::vector<std::string>& cells) {
            std::ostringstream out;
            for(size_t c = 0; c < cells.size(); c++)
                out << (c ? "," : "") << cells[c];
            return out.str();
        };

        bool changed = false;
        for(size_t i = 0; i + 1 < lines.size(); i++)
        {
            if(lines[i].find("transA") == std::string::npos)
                continue;

            auto names  = split(lines[i]);
            auto values = split(lines[i + 1]);
            for(size_t c = 0; c < names.size(); c++)
            {
                if(names[c] != column)
                    continue;
                names.erase(names.begin() + c);
                if(c < values.size())
                    values.erase(values.begin() + c);
                changed = true;
                break;
            }

            lines[i]     = "    " + join(names);
            lines[i + 1] = join(values);
            i++;
        }

        if(!changed)
            return false;

        std::ofstream out(path, std::ios::trunc);
        for(const auto& l : lines)
            out << l << "\n";
        return true;
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
     * Rebuild the file as two value rows cloned from its first, each with the
     * given column overrides applied.
     *
     * Cloning rather than composing a row from scratch keeps every key column
     * byte-identical to what the writer emits, so both rows land under one key.
     * That is the shape an append-only cache takes once a later run finishes a
     * search the budget had truncated.
     */
    bool writeTwoRowsFromFirst(const std::string&                        path,
                               const std::map<std::string, std::string>& firstOverrides,
                               const std::map<std::string, std::string>& secondOverrides)
    {
        const auto lines  = readLines(path);
        size_t     header = lines.size();
        for(size_t i = 0; i + 1 < lines.size(); i++)
        {
            if(lines[i].find("transA") != std::string::npos)
            {
                header = i;
                break;
            }
        }
        if(header == lines.size())
            return false;

        const auto names = splitCells(lines[header]);

        auto apply = [&](const std::map<std::string, std::string>& overrides) {
            auto values = splitCells(lines[header + 1]);
            for(size_t c = 0; c < names.size() && c < values.size(); c++)
            {
                const auto it = overrides.find(names[c]);
                if(it != overrides.end())
                    values[c] = it->second;
            }

            std::ostringstream row;
            for(size_t c = 0; c < values.size(); c++)
                row << (c ? "," : "") << values[c];
            return row.str();
        };

        const auto first  = apply(firstOverrides);
        const auto second = apply(secondOverrides);

        std::ofstream out(path, std::ios::trunc);
        if(!out)
            return false;

        // Whatever the writer put above the first header, including the version
        // line the parser reads before any row.
        for(size_t i = 0; i < header; i++)
            out << lines[i] << "\n";

        out << lines[header] << "\n" << first << "\n";
        out << lines[header] << "\n" << second << "\n";
        return out.good();
    }

    /**
     * Rewrite the file as its first row reduced to the historical column set:
     * the ten problem columns plus solution_index, with no schema_version and
     * no name of either kind. That is what hipblaslt-bench has always written.
     */
    bool reduceToLegacyRow(const std::string& path)
    {
        static const char* const kLegacy[] = {"transA",
                                              "transB",
                                              "batch_count",
                                              "m",
                                              "n",
                                              "k",
                                              "a_type",
                                              "b_type",
                                              "c_type",
                                              "compute_type",
                                              "solution_index"};

        const auto lines = readLines(path);
        for(size_t i = 0; i + 1 < lines.size(); i++)
        {
            if(lines[i].find("transA") == std::string::npos)
                continue;

            const auto names = splitCells(lines[i]);
            const auto vals  = splitCells(lines[i + 1]);

            std::ostringstream header, values;
            bool               first = true;
            for(const char* want : kLegacy)
            {
                for(size_t c = 0; c < names.size() && c < vals.size(); c++)
                {
                    if(names[c] != want)
                        continue;
                    header << (first ? "" : ",") << names[c];
                    values << (first ? "" : ",") << vals[c];
                    first = false;
                }
            }

            std::ofstream out(path, std::ios::trunc);
            out << lines[0] << "\n    " << header.str() << "\n" << values.str() << "\n";
            return out.good();
        }
        return false;
    }

    /**
     * Cut every value row short just before the named column, leaving its header
     * whole: what a process that died mid-append leaves behind.
     */
    bool truncateValueRowsBefore(const std::string& path, const std::string& column)
    {
        auto lines   = readLines(path);
        bool changed = false;
        for(size_t i = 0; i + 1 < lines.size(); i++)
        {
            if(lines[i].find("transA") == std::string::npos)
                continue;

            const auto names  = splitCells(lines[i]);
            const auto values = splitCells(lines[i + 1]);
            for(size_t c = 0; c < names.size() && c <= values.size(); c++)
            {
                if(names[c] != column)
                    continue;

                std::ostringstream kept;
                for(size_t v = 0; v < c; v++)
                    kept << (v ? "," : "") << values[v];
                lines[i + 1] = kept.str();
                changed      = true;
                break;
            }
            i++;
        }

        if(!changed)
            return false;

        std::ofstream out(path, std::ios::trunc);
        for(const auto& l : lines)
            out << l << "\n";
        return out.good();
    }

    /**
     * Whether the C++ extension heuristic, asked with this stream-K tile mode,
     * is served by the cache: a hit is counted only when an entry matched the
     * key that path looks up.
     */
    bool extensionHeuristicHits(int64_t m, int64_t n, int64_t k, int32_t streamKMode)
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
            gemm.setProblem(m, n, k, 1, epilogue, inputs);

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

    class TuningCache : public ::testing::Test
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

        std::string m_path;
        std::vector<std::pair<std::string, std::optional<std::string>>> m_savedEnv;
    };

    // Nothing is written and nothing is consulted when tuning is not asked for.
    TEST_F(TuningCache, OffModeWritesNothing)
    {
        enterMode("off", m_path);
        ASSERT_TRUE(runGemm(256, 256, 256));

        std::ifstream probe(m_path);
        EXPECT_FALSE(probe.good()) << "off mode created " << m_path;

        const auto c = counters();
        EXPECT_EQ(c.tuned, 0u);
        EXPECT_EQ(c.hits, 0u);
    }

    // cache/tune do nothing without somewhere to keep results, and must not
    // fall back to the legacy override variable.
    TEST_F(TuningCache, CacheModeWithoutPathIsInert)
    {
        enterMode("cache", "");
        EXPECT_TRUE(runGemm(256, 256, 256));

        const auto c = counters();
        EXPECT_EQ(c.hits, 0u);
        EXPECT_EQ(c.tuned, 0u);
    }

    // A tuned row must carry everything a later lookup rebuilds the key from,
    // plus the identity fields validation depends on.
    TEST_F(TuningCache, TuneWritesRowWithIdentityAndKeyFields)
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

        // What the search covered, so a later run can tell whether it would
        // search more than this one did.
        EXPECT_TRUE(fileHasColumn(m_path, "search_all_kernels"));
        EXPECT_TRUE(fileHasColumn(m_path, "search_max_candidates"));
        EXPECT_TRUE(fileHasColumn(m_path, "search_workspace"));
        EXPECT_TRUE(fileHasColumn(m_path, "hot_iters"));

        // Nothing reads these back, so writing them would commit the format to
        // data with no consumer.
        EXPECT_FALSE(fileHasColumn(m_path, "baseline_index"));
        EXPECT_FALSE(fileHasColumn(m_path, "baseline_us"));
    }

    // Exercise the shipping-default exhaustive enumeration independently from
    // the fast ranked-prefix settings used by the rest of this suite.
    TEST_F(TuningCache, ExhaustiveEnumerationWritesEntry)
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

    // The winner has to actually run on the caller's buffers and be right.
    //
    // Separate C and D, and nonzero inputs, because the rest of the suite tunes
    // in-place with zeroed operands: there every candidate and every bug
    // produces the same all-zero output, so a tuning path that reported success
    // without ever launching the winner would still pass.
    TEST_F(TuningCache, TunedLaunchProducesCorrectResult)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(128, 128, 128, 0.0f, false, false, true));
        ASSERT_GT(valueRowCount(m_path), 0u) << "tune mode recorded nothing";

        enterMode("cache", m_path);
        ASSERT_TRUE(runGemm(128, 128, 128, 0.0f, false, false, true));
        EXPECT_GE(counters().hits, 1u) << "tuned entry did not replay";
    }

    // A recorded datatype must parse back as the datatype it was written from.
    //
    // The FNUZ and OCP FP8 types share one bench spelling, "f8_r", which the
    // parser resolves to the OCP type. gfx942 FP8 is FNUZ, so every FP8 row it
    // wrote described a different type than the one measured: the row never
    // matched its own key, the shape re-tuned on every process start, and the
    // cache file grew a duplicate each time.
    TEST_F(TuningCache, Fp8FnuzEntryReplays)
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
    TEST_F(TuningCache, InPlaceNonzeroBetaRunsWithoutTuning)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(384, 256, 128, 2.0f, true, true));
        EXPECT_EQ(valueRowCount(m_path), 0u);
        EXPECT_GE(counters().skipped, 1u);
    }

    // The round trip the whole feature exists for.
    TEST_F(TuningCache, TunedEntryReplaysInCacheMode)
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
    TEST_F(TuningCache, TamperedKernelNameIsRejected)
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
    TEST_F(TuningCache, InvalidatedEntryIsRetunedAndReplays)
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
    TEST_F(TuningCache, StaleRowDoesNotSuppressItsReplacement)
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

        enterMode("cache", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        const auto c = counters();
        EXPECT_GE(c.hits, 1u) << "stale row hid the valid row that followed it";
    }

    // A row written by a newer hipBLASLt must be ignored, not read as if it were
    // this build's format.
    //
    // A future schema keys on fields this parser cannot see, so interpreting one
    // as the current version matches on the subset it happens to recognise and
    // applies a kernel chosen for a problem that differs in the rest. The
    // version was previously accepted as current whenever it was at least the
    // current number, so this row replayed.
    TEST_F(TuningCache, UnknownSchemaVersionIsRejected)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        ASSERT_GT(valueRowCount(m_path), 0u) << "tune mode recorded nothing";

        ASSERT_TRUE(rewriteColumn(m_path, "schema_version", "99"));

        enterMode("cache", m_path);
        EXPECT_TRUE(runGemm(1024, 512, 1024)) << "ignoring a row must not fail the call";

        const auto c = counters();
        EXPECT_EQ(c.hits, 0u) << "a row from an unknown schema was replayed";
    }

    // Some strings are recognized as hipDataType values but are not legal GEMM
    // tensor types. They must be rejected before the assert-based Tensile
    // converter is called.
    TEST_F(TuningCache, UnsupportedCurrentDatatypeIsRejected)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        ASSERT_TRUE(rewriteColumn(m_path, "a_type", "e8_r"));

        enterMode("cache", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        EXPECT_EQ(counters().hits, 0u);
    }

    // A current-schema row missing a value for a key field must be dropped.
    //
    // stride_a is blanked because this shape's real batch stride is 0, so the
    // empty cell parses to a value that matches and the row replays. That is the
    // hazard: an empty cell is indistinguishable from a real zero, and a row
    // truncated in one field is not trustworthy in the others either.
    TEST_F(TuningCache, CurrentSchemaRowWithBlankKeyFieldIsRejected)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        ASSERT_GT(valueRowCount(m_path), 0u) << "tune mode recorded nothing";

        ASSERT_TRUE(rewriteColumn(m_path, "stride_a", ""));

        enterMode("cache", m_path);
        EXPECT_TRUE(runGemm(1024, 512, 1024)) << "ignoring a row must not fail the call";

        const auto c = counters();
        EXPECT_EQ(c.hits, 0u) << "a row with no value for a key field was replayed";
    }

    TEST_F(TuningCache, CurrentSchemaNumericPrefixIsRejected)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        ASSERT_TRUE(rewriteColumn(m_path, "m", "1024junk"));

        enterMode("cache", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        EXPECT_EQ(counters().hits, 0u);
    }

    // Recover after a process dies between writing a header and its value row.
    // The orphan must not consume the next complete record's header.
    TEST_F(TuningCache, TornHeaderDoesNotHideFollowingEntry)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        const auto complete = readLines(m_path);
        ASSERT_GE(complete.size(), 3u);

        {
            std::ofstream out(m_path, std::ios::trunc);
            out << complete[0] << "\n"
                << complete[1] << "\n"
                << complete[1] << "\n"
                << complete[2] << "\n";
        }

        enterMode("cache", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        EXPECT_GE(counters().hits, 1u);
    }

    // Sequential tuning on two devices must allocate and reuse scratch in each
    // device's address space rather than handing device 1 the pointer allocated
    // while device 0 was current.
    TEST_F(TuningCache, ScratchIsDeviceLocal)
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

    // Files written before the widened key must keep matching on the fields the
    // old format recorded.
    TEST_F(TuningCache, LegacyRowStillMatches)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        ASSERT_TRUE(reduceToLegacyRow(m_path)) << "tune mode recorded nothing";

        enterMode("cache", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        const auto c = counters();
        EXPECT_GE(c.loaded, 1u) << "legacy row was not loaded";
        EXPECT_GE(c.hits, 1u) << "legacy row did not match";
    }

    // Turning tuning on has to explain the pause it causes, without the user
    // also having to turn on library logging.
    TEST_F(TuningCache, TuneLifecycleIsVisibleWithoutLogLevel)
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

    // A legacy override user never opted into any of this and must not start
    // seeing new output on stderr because the feature exists.
    TEST_F(TuningCache, OffModeWithLegacyOverrideEmitsNoLifecycle)
    {
        if(!loggingEnvIsClean())
            GTEST_SKIP() << "external logging environment redirects tuning output";

        enterModeForLogging("tune", m_path);
        ASSERT_TRUE(runGemm(256, 256, 256));
        ASSERT_GT(valueRowCount(m_path), 0u) << "tune mode recorded nothing";

        enterMode("off", "");
        setenv("HIPBLASLT_TUNING_OVERRIDE_FILE", m_path.c_str(), 1);
        hipblaslt_tuning_reset_for_test();

        std::string out;
        {
            CerrCapture capture;
            ASSERT_TRUE(runGemm(256, 256, 256));
            out = capture.str();
        }

        // Asserted before the silence check, because silence is also what a run
        // that never consulted the override file looks like. The tune phase
        // above wrote this exact shape, so off mode must replay it from the
        // legacy variable to get here.
        ASSERT_GE(counters().hits, 1u) << "the legacy override file was never consulted";

        EXPECT_EQ(out.find("tuning-cache:"), std::string::npos) << out;
    }

    // Replay is the steady state, so it has to be silent per call. Otherwise the
    // feature costs a line of stderr for every matmul an application runs.
    TEST_F(TuningCache, RepeatedCacheHitsAddNoOutput)
    {
        if(!loggingEnvIsClean())
            GTEST_SKIP() << "external logging environment redirects tuning output";

        enterModeForLogging("tune", m_path);
        ASSERT_TRUE(runGemm(256, 256, 256));
        ASSERT_GT(valueRowCount(m_path), 0u) << "tune mode recorded nothing";

        enterModeForLogging("cache", m_path);

        // First call carries the once-per-process startup line.
        {
            CerrCapture capture;
            ASSERT_TRUE(runGemm(256, 256, 256));
        }

        std::string later;
        {
            CerrCapture capture;
            ASSERT_TRUE(runGemm(256, 256, 256));
            ASSERT_TRUE(runGemm(256, 256, 256));
            later = capture.str();
        }

        EXPECT_EQ(later, "") << later;
    }

    // A skip records no winner, so the shape stays uncached and the whole attempt
    // runs again on the next matmul. Unbounded, that is a line per call.
    TEST_F(TuningCache, RepeatedSkipReportsItsReasonOnce)
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
    TEST_F(TuningCache, SkipDoesNotSilenceALaterTuneOfTheSameShape)
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

    // A cache that served nothing must not read like a process that asked
    // nothing. Without the shape recorded here the summary says shapes=0, which
    // is the same thing it says when tuning was never reached at all.
    TEST_F(TuningCache, ShapesThatMissAnEmptyCacheAreStillCounted)
    {
        enterModeForLogging("cache", m_path);
        ASSERT_FALSE(std::ifstream(m_path).good()) << "the cache file was supposed to be absent";

        ASSERT_TRUE(runGemm(256, 256, 256));

        uint64_t shapes = 0, matched = 0, fellback = 0, tuned = 0;
        hipblaslt_tuning_lookup_tally_for_test(&shapes, &matched, &fellback, &tuned);

        EXPECT_EQ(shapes, 1u);
        EXPECT_EQ(matched, 0u);
        EXPECT_EQ(fellback, 1u);
        EXPECT_EQ(tuned, 0u);
    }

    // The summary is read against loaded=N, so it counts problems rather than
    // lookups. A replay loop over one shape is one shape.
    TEST_F(TuningCache, LookupTallyCountsShapesNotCalls)
    {
        enterModeForLogging("tune", m_path);
        ASSERT_TRUE(runGemm(256, 256, 256));
        ASSERT_GT(valueRowCount(m_path), 0u) << "tune mode recorded nothing";

        enterModeForLogging("cache", m_path);
        for(int i = 0; i < 4; i++)
            ASSERT_TRUE(runGemm(256, 256, 256));

        uint64_t shapes = 0, matched = 0, fellback = 0, tuned = 0;
        hipblaslt_tuning_lookup_tally_for_test(&shapes, &matched, &fellback, &tuned);

        EXPECT_EQ(shapes, 1u);
        EXPECT_EQ(matched, 1u);
        EXPECT_EQ(fellback, 0u);
    }

    // A winner that could not be written was still tuned: it is in the in-memory
    // cache and serves every later call in this process. Reporting tuned=0 next
    // to a "tuning-done ... persisted=no" line contradicts it.
    TEST_F(TuningCache, TuneThatCannotPersistIsStillCountedAsTuned)
    {
        // No such directory, so the load finds nothing and the append cannot
        // create the file. Tuning itself is unaffected.
        const std::string unwritable = "/nonexistent-hipblaslt-review-dir/cache.tuning";

        enterModeForLogging("tune", unwritable);
        ASSERT_TRUE(runGemm(256, 256, 256));

        uint64_t shapes = 0, matched = 0, fellback = 0, tuned = 0;
        hipblaslt_tuning_lookup_tally_for_test(&shapes, &matched, &fellback, &tuned);

        EXPECT_EQ(tuned, 1u) << "a successful tune vanished from the summary because the "
                                "winner could not be written";

        // The persist-only counter is what stayed at zero, and that is the
        // distinction the outcome line reports.
        EXPECT_EQ(counters().tuned, 0u) << "the winner was somehow written to " << unwritable;
    }

    // An entry the budget cut short is usable but not final: it is replayed like
    // any other, and a run that can finish the search replaces it rather than
    // living with the best of an arbitrary prefix.
    TEST_F(TuningCache, PartialEntryReplaysAndIsRetuned)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        ASSERT_TRUE(fileHasColumn(m_path, "complete")) << "completeness was not recorded";

        // Demote the row to what a search stopped by a one-second ceiling would
        // have written. Doing it this way rather than by inducing a real
        // truncation keeps the test off the clock: a budget that stops the
        // search partway through is a race.
        ASSERT_TRUE(rewriteColumn(m_path, "complete", "0"));
        ASSERT_TRUE(rewriteColumn(m_path, "budget_ms", "1000"));
        const size_t before = valueRowCount(m_path);

        enterMode("cache", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        EXPECT_GE(counters().hits, 1u) << "a partial entry was not replayed";

        // The fixture runs with no ceiling, which beats the recorded one.
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        EXPECT_GT(valueRowCount(m_path), before) << "a partial entry was never re-tuned";
    }

    // Re-tuning is worth a stall only when this run can get further than the one
    // that gave up. Under the same ceiling it would measure the same prefix,
    // stop in the same place, and append an identical row, so a shape that keeps
    // truncating must not cost the ceiling on every process that opens the file.
    TEST_F(TuningCache, PartialEntryIsNotRetunedUnderTheSameCeiling)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        ASSERT_TRUE(rewriteColumn(m_path, "complete", "0"));
        ASSERT_TRUE(rewriteColumn(m_path, "budget_ms", "1000"));

        const size_t before = valueRowCount(m_path);
        ASSERT_GT(before, 0u) << "tune mode recorded nothing";

        enterMode("tune", m_path);
        setenv("HIPBLASLT_TUNING_BUDGET_MS_PER_SHAPE", "1000", 1);
        hipblaslt_tuning_reset_for_test();
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        EXPECT_EQ(valueRowCount(m_path), before)
            << "a partial entry was re-tuned under the ceiling that already stopped it";
    }

    // The file is append-only, so a shape tuned partially and later properly
    // ends up holding both rows. The superseded partial row must not keep the
    // shape looking untuned, or the finished search is repeated on every process
    // start for the life of the file.
    TEST_F(TuningCache, PartialRowBesideCompleteRowDoesNotRetune)
    {
        // Two searches over different candidate pools, so the second is likely
        // to land on a different kernel and leave two distinct entries under one
        // key. If they do coincide the later row simply refreshes the earlier
        // one, which this must also survive.
        enterMode("tune", m_path);
        setenv("HIPBLASLT_TUNING_MAX_CANDIDATES", "2", 1);
        hipblaslt_tuning_reset_for_test();
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        ASSERT_TRUE(rewriteColumn(m_path, "complete", "0"));
        ASSERT_TRUE(rewriteColumn(m_path, "budget_ms", "1000"));

        enterMode("tune", m_path);
        setenv("HIPBLASLT_TUNING_MAX_CANDIDATES", "16", 1);
        hipblaslt_tuning_reset_for_test();
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        const size_t before = valueRowCount(m_path);
        ASSERT_GT(before, 1u) << "the finishing run did not append beside the partial row";

        enterMode("tune", m_path);
        setenv("HIPBLASLT_TUNING_MAX_CANDIDATES", "16", 1);
        hipblaslt_tuning_reset_for_test();
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        EXPECT_EQ(valueRowCount(m_path), before)
            << "a superseded partial row kept the shape looking untuned";
    }

    // The other half of that pair, and the one the row count cannot see. Not
    // re-tuning is only correct if the completed row is also the row that runs.
    // The file is append-only, so the finishing run leaves its winner behind the
    // partial it supersedes, and equal keys come back from the multimap in
    // insertion order: replay took the partial while needsRetune saw the
    // complete row and declined to tune again, so the finished winner was
    // written to the file and then never used.
    TEST_F(TuningCache, CompleteRowWinsOverTheOlderPartialRow)
    {
        // Two solutions the heuristic itself offers, so both rows below are
        // genuinely valid and replay has to choose between them on order alone.
        // Two real tuning runs would be more lifelike but not deterministic:
        // a wider candidate pool often reaches the same winner, and then there
        // is nothing to observe.
        const auto identities = candidateIdentities(1024, 512, 1024, 8);
        if(identities.size() < 2)
            GTEST_SKIP() << "this device offers one solution for the shape";

        const auto& partial  = identities[0];
        const auto& finished = identities[1];

        // A real row first, so every key column is exactly what the writer emits
        // and both clones match the lookup key.
        enterMode("tune", m_path);
        hipblaslt_tuning_reset_for_test();
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        ASSERT_EQ(valueRowCount(m_path), 1u) << "tune mode did not record exactly one row";

        ASSERT_TRUE(writeTwoRowsFromFirst(m_path,
                                          {{"solution_index", std::to_string(partial.first)},
                                           {"kernel_name", partial.second},
                                           {"complete", "0"},
                                           {"budget_ms", "1000"}},
                                          {{"solution_index", std::to_string(finished.first)},
                                           {"kernel_name", finished.second},
                                           {"complete", "1"},
                                           {"budget_ms", "0"}}));

        // Reset so the rows come back from the file the way a later process sees
        // them. Within one process the tuner replaces its own entry, which is
        // what kept this ordering hidden.
        enterMode("cache", m_path);
        hipblaslt_tuning_reset_for_test();

        int replayed = -1;
        ASSERT_TRUE(runGemm(1024, 512, 1024, 0.0f, true, false, false, &replayed));
        EXPECT_EQ(replayed, finished.first) << "replay used the superseded partial row "
                                            << partial.first << " instead of the completed search";
    }

    // The other half of that gate. A finished entry must close it, or every
    // tune-mode process would pay the whole search again on a warm cache.
    TEST_F(TuningCache, CompleteEntryIsNotRetuned)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        const size_t before = valueRowCount(m_path);
        ASSERT_GT(before, 0u) << "tune mode recorded nothing";

        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        EXPECT_EQ(valueRowCount(m_path), before) << "a complete entry was re-tuned";
    }

    // A row in the format hipblaslt-bench has always written carries no
    // completeness and no search, and must read as final. Reading it as partial
    // would re-tune every shape of an existing tuning file on the first tune run,
    // which is the stall this whole feature is trying to avoid.
    TEST_F(TuningCache, LegacyRowIsNotRetuned)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        ASSERT_TRUE(reduceToLegacyRow(m_path)) << "tune mode recorded nothing";
        ASSERT_EQ(valueRowCount(m_path), 1u);

        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        EXPECT_EQ(valueRowCount(m_path), 1u) << "a legacy row was re-tuned as if it were partial";
    }

    // A current-schema row that lost its completion column is not trusted:
    // whether tune mode revisits a row depends on it, and a missing value can
    // no longer be told apart from a finished search.
    TEST_F(TuningCache, CurrentSchemaRowWithoutItsCompletionColumnIsRejected)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        ASSERT_TRUE(dropColumn(m_path, "complete"));

        enterMode("cache", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        const auto c = counters();
        EXPECT_EQ(c.loaded, 0u) << "a row with no completion column was loaded";
        EXPECT_EQ(c.hits, 0u);
    }

    // A process that dies partway through an append leaves a value row shorter
    // than its header. Cut before the completion columns, or between them, the
    // prefix that survives is well formed and would otherwise read as a finished
    // search.
    TEST_F(TuningCache, TruncatedCurrentSchemaRowIsRejected)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        const auto written = readLines(m_path);
        ASSERT_EQ(valueRowCount(m_path), 1u) << "tune mode did not record exactly one row";

        for(const char* cutBefore : {"complete", "budget_ms"})
        {
            SCOPED_TRACE(std::string("value row cut before ") + cutBefore);
            {
                std::ofstream out(m_path, std::ios::trunc);
                for(const auto& l : written)
                    out << l << "\n";
            }
            ASSERT_TRUE(truncateValueRowsBefore(m_path, cutBefore));

            enterMode("cache", m_path);
            ASSERT_TRUE(runGemm(1024, 512, 1024));

            const auto c = counters();
            EXPECT_EQ(c.loaded, 0u) << "a truncated row was loaded";
            EXPECT_EQ(c.hits, 0u);
        }
    }

    // A search the budget cuts short is attempted once per process. The retry
    // would run under the same ceiling and stop in the same place, so leaving it
    // unlatched spends the whole ceiling on every call for the life of the
    // process rather than once.
    TEST_F(TuningCache, BudgetExhaustedShapeIsNotRetried)
    {
        enterMode("tune", m_path);

        // A ceiling this small stops the search either before the first
        // candidate is measured or just after, and which one is a matter of how
        // long setup happened to take. Both are correct, and they end
        // differently: nothing measured is a skip that records no row, whereas
        // one candidate measured is a partial winner that records exactly one.
        // What must hold either way is that it happened once.
        setenv("HIPBLASLT_TUNING_BUDGET_MS_PER_SHAPE", "1", 1);
        hipblaslt_tuning_reset_for_test();

        for(int i = 0; i < 3; i++)
            ASSERT_TRUE(runGemm(1024, 512, 1024));

        const auto c = counters();
        EXPECT_EQ(c.skipped + c.tuned, 1u)
            << "the shape re-tuned after its budget ran out, so every call pays the ceiling";
        EXPECT_LE(valueRowCount(m_path), 1u) << "a truncated search appended more than one row";
    }

    // One stale row is met three times in a tune-mode call: the heuristic
    // lookup, the execution probe, and the recheck under the tuning lock.
    TEST_F(TuningCache, StaleEntryIsCountedAsOneInvalidation)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        ASSERT_EQ(valueRowCount(m_path), 1u) << "expected exactly one tuned row to spoil";
        ASSERT_TRUE(rewriteColumn(m_path, "kernel_name", "NotARealKernelName"));

        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));

        EXPECT_EQ(counters().invalidated, 1u) << "one rejected entry was counted more than once";
    }

    // The router must never open the log file itself: doing so would create and
    // truncate it for a process that asked for tuning but not for logging.
    TEST_F(TuningCache, LogFileWithoutLevelIsNotCreated)
    {
        if(!loggingEnvIsClean())
            GTEST_SKIP() << "external logging environment redirects tuning output";

        const std::string logPath = tempCachePath("LogFileProbe") + ".log";
        std::remove(logPath.c_str());
        setenv("HIPBLASLT_LOG_FILE", logPath.c_str(), 1);

        enterModeForLogging("tune", m_path);

        bool ran = false;
        {
            CerrCapture capture;
            ran = runGemm(256, 256, 256);
        }

        unsetenv("HIPBLASLT_LOG_FILE");
        ASSERT_TRUE(ran);

        const bool created = std::ifstream(logPath).good();
        std::remove(logPath.c_str());

        EXPECT_FALSE(created) << "the lifecycle router created " << logPath
                              << " even though no log level was set";
    }

    // A caller that passes an algo gets that algo, whatever the cache holds. The
    // cache chooses only when asked to, through the heuristic query or by a
    // matmul given no algo.
    TEST_F(TuningCache, ExplicitAlgoLaunchesAsGivenInCacheMode)
    {
        const auto identities = candidateIdentities(1024, 512, 1024, 8);
        if(identities.size() < 2)
            GTEST_SKIP() << "this device offers one solution for the shape";

        const auto& given  = identities[0];
        const auto& cached = identities[1];

        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        ASSERT_EQ(valueRowCount(m_path), 1u) << "tune mode did not record exactly one row";
        ASSERT_TRUE(rewriteColumn(m_path, "solution_index", std::to_string(cached.first)));
        ASSERT_TRUE(rewriteColumn(m_path, "kernel_name", cached.second));

        enterMode("cache", m_path);
        int launched = -1;
        ASSERT_TRUE(runGemmWith(1024, 512, 1024, AlgoFrom::Index, given.first, &launched));
        EXPECT_EQ(launched, given.first)
            << "the cache replaced the caller's algo with " << cached.first;
        EXPECT_EQ(counters().hits, 0u) << "a hit was counted for a launch the cache did not choose";
    }

    // A matmul given no algo leaves the choice to the library, so a usable cache
    // entry is what it launches.
    TEST_F(TuningCache, NullAlgoLaunchesTheCachedKernel)
    {
        const auto identities = candidateIdentities(1024, 512, 1024, 8);
        if(identities.size() < 2)
            GTEST_SKIP() << "this device offers one solution for the shape";

        const auto& cached = identities[1];

        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        ASSERT_EQ(valueRowCount(m_path), 1u) << "tune mode did not record exactly one row";
        ASSERT_TRUE(rewriteColumn(m_path, "solution_index", std::to_string(cached.first)));
        ASSERT_TRUE(rewriteColumn(m_path, "kernel_name", cached.second));

        enterMode("cache", m_path);
        int launched = -1;
        ASSERT_TRUE(runGemmWith(1024, 512, 1024, AlgoFrom::Null, -1, &launched, true));
        EXPECT_EQ(launched, cached.first) << "a matmul with no algo ignored the cached entry";
        EXPECT_EQ(counters().hits, 1u);
    }

    // With no usable entry a null algo still runs, on default selection's pick,
    // and the lookup is counted as the fallback it is.
    TEST_F(TuningCache, NullAlgoWithoutAnEntryLaunchesDefaultSelection)
    {
        const auto identities = candidateIdentities(1024, 512, 1024, 1);
        ASSERT_FALSE(identities.empty());

        enterMode("cache", m_path);
        int launched = -1;
        ASSERT_TRUE(runGemmWith(1024, 512, 1024, AlgoFrom::Null, -1, &launched, true));
        EXPECT_EQ(launched, identities[0].first);

        const auto c = counters();
        EXPECT_EQ(c.hits, 0u);
        EXPECT_GE(c.misses, 1u);
    }

    // Query once and reuse the algo, the pattern the API documents. Tuning on the
    // first call records a winner for later lookups, but must not change what
    // that algo launches, on the tuning call or any after it.
    TEST_F(TuningCache, ReusedAlgoKeepsLaunchingItsOwnKernel)
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
    TEST_F(TuningCache, NullAlgoLaunchesTheWinnerOnTheTuningCall)
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
    TEST_F(TuningCache, FailedSearchIsNotRepeated)
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

            EXPECT_EQ(hipblaslt_tuning_attempts_for_test(), 1u);
            EXPECT_EQ(valueRowCount(m_path), 0u);
        }
    }

    // Two threads that meet two untuned shapes at the same time both get them
    // tuned. Tune mode was asked for, so what it records must not depend on
    // which thread reached the tuning lock first.
    TEST_F(TuningCache, ConcurrentShapesAreAllTuned)
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

    // Among rows that are all partial, the newest runs, whatever ceilings they
    // were written under. In the normal workflow a later partial row exists only
    // because a later run had a more generous ceiling, so the newest is also the
    // most thorough; the older row's larger ceiling here pins the rule itself.
    TEST_F(TuningCache, NewestPartialRowWinsAmongPartialRows)
    {
        const auto identities = candidateIdentities(1024, 512, 1024, 8);
        if(identities.size() < 2)
            GTEST_SKIP() << "this device offers one solution for the shape";

        const auto& older = identities[0];
        const auto& newer = identities[1];

        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        ASSERT_EQ(valueRowCount(m_path), 1u) << "tune mode did not record exactly one row";

        ASSERT_TRUE(writeTwoRowsFromFirst(m_path,
                                          {{"solution_index", std::to_string(older.first)},
                                           {"kernel_name", older.second},
                                           {"complete", "0"},
                                           {"budget_ms", "60000"}},
                                          {{"solution_index", std::to_string(newer.first)},
                                           {"kernel_name", newer.second},
                                           {"complete", "0"},
                                           {"budget_ms", "1000"}}));

        enterMode("cache", m_path);
        int replayed = -1;
        ASSERT_TRUE(runGemm(1024, 512, 1024, 0.0f, true, false, false, &replayed));
        EXPECT_EQ(replayed, newer.first) << "replay used the older partial row " << older.first;
    }

    // A finished search of a ranked prefix says nothing about the kernels past
    // it, so a run that searches more re-tunes the shape. A run that searches
    // less has nothing to add, and must not.
    TEST_F(TuningCache, NarrowerFinishedSearchIsWidenedButNotRepeated)
    {
        enterMode("tune", m_path);
        setenv("HIPBLASLT_TUNING_MAX_CANDIDATES", "2", 1);
        hipblaslt_tuning_reset_for_test();
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        ASSERT_EQ(valueRowCount(m_path), 1u) << "tune mode did not record exactly one row";

        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        EXPECT_EQ(valueRowCount(m_path), 2u)
            << "a finished two-candidate search stopped a sixteen-candidate run from tuning";

        enterMode("tune", m_path);
        setenv("HIPBLASLT_TUNING_MAX_CANDIDATES", "2", 1);
        hipblaslt_tuning_reset_for_test();
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        EXPECT_EQ(valueRowCount(m_path), 2u)
            << "a narrower run re-tuned a shape that a wider search had finished";
    }

    // Each of these is part of the key, so a row that differs from the problem
    // in one of them is loaded but serves nothing.
    TEST_F(TuningCache, RowDifferingInAKeyColumnDoesNotMatch)
    {
        enterMode("tune", m_path);
        ASSERT_TRUE(runGemm(1024, 512, 1024));
        const auto written = readLines(m_path);
        ASSERT_EQ(valueRowCount(m_path), 1u) << "tune mode did not record exactly one row";

        const std::vector<std::pair<std::string, std::string>> changes
            = {{"lde", "12345"}, {"stride_e", "12345"}, {"uniform_summation_order", "1"}};
        for(const auto& [column, value] : changes)
        {
            SCOPED_TRACE(column + "=" + value);
            {
                std::ofstream out(m_path, std::ios::trunc);
                for(const auto& l : written)
                    out << l << "\n";
            }
            ASSERT_TRUE(rewriteColumn(m_path, column, value));

            enterMode("cache", m_path);
            ASSERT_TRUE(runGemm(1024, 512, 1024));

            const auto c = counters();
            EXPECT_EQ(c.loaded, 1u) << "the row was rejected rather than keyed on the column";
            EXPECT_EQ(c.hits, 0u);
        }
    }

    // A conjugate transpose is keyed apart from a plain one. Both are "not N",
    // which is all the historical key records, so without the distinction a row
    // tuned for one would serve the other.
    TEST_F(TuningCache, ConjugateTransposeRowDoesNotServeATranspose)
    {
        int launched = -1;

        enterMode("tune", m_path);
        ASSERT_TRUE(runGemmWith(
            1024, 512, 1024, AlgoFrom::Heuristic, -1, &launched, false, HIPBLAS_OP_T));
        ASSERT_EQ(valueRowCount(m_path), 1u) << "tune mode recorded nothing";
        ASSERT_EQ(columnValues(m_path, "transA").at(0), "T");

        enterMode("cache", m_path);
        ASSERT_TRUE(runGemmWith(
            1024, 512, 1024, AlgoFrom::Heuristic, -1, &launched, false, HIPBLAS_OP_T));
        ASSERT_GE(counters().hits, 1u) << "the transposed row did not serve its own problem";

        ASSERT_TRUE(rewriteColumn(m_path, "transA", "C"));
        enterMode("cache", m_path);
        ASSERT_TRUE(runGemmWith(
            1024, 512, 1024, AlgoFrom::Heuristic, -1, &launched, false, HIPBLAS_OP_T));

        const auto c = counters();
        EXPECT_EQ(c.loaded, 1u) << "a conjugate-transpose row was rejected rather than keyed";
        EXPECT_EQ(c.hits, 0u) << "a conjugate-transpose row served a plain transpose";
    }

    // The C++ extension builds its key when the gemm is created, and only later
    // does the preference set the stream-K tile mode, which is part of the key.
    // A row tuned with that mode through the C API must serve the extension
    // asking with the same mode, and must not serve it asking with another.
    TEST_F(TuningCache, ExtensionLookupKeysOnThePreferenceStreamKMode)
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
        EXPECT_FALSE(extensionHeuristicHits(1024, 512, 1024, HIPBLASLT_STREAMK_TILE_SCHEDULING_OFF));
    }

    // A process in a secure execution context (set-user-ID and the like) must
    // not let an inherited environment turn tuning on: that would choose a file
    // for it to write and minutes of GPU work for it to spend.
    TEST_F(TuningCache, PrivilegedProcessIgnoresTheTuningEnvironment)
    {
        enterMode("tune", m_path);
        hipblaslt_tuning_reset_as_privileged_for_test();

        ASSERT_TRUE(runGemm(256, 256, 256));

        EXPECT_FALSE(std::ifstream(m_path).good()) << "a privileged process wrote " << m_path;
        EXPECT_EQ(hipblaslt_tuning_attempts_for_test(), 0u);

        const auto c = counters();
        EXPECT_EQ(c.hits + c.misses, 0u) << "a privileged process consulted the cache";
    }
} // namespace
