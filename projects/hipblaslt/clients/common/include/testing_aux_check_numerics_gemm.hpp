/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

// Gtest counterpart of sample 28_hipblaslt_gemm_check_numerics: drives N
// back-to-back fp32 matmuls per scenario through the public hipBLASLt API and
// asserts hipblasLtCheckNumericsDrain() reports the expected call_id under
// each HIPBLASLT_CHECK_NUMERICS_SCAN_* knob (default sampling, SCAN_EVERY,
// SCAN_FROM/SCAN_UNTIL window). Public-API only -- works in every build
// flavor regardless of CODE_COVERAGE.

#include "hipblaslt_test.hpp"

#include <hipblaslt/hipblaslt.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

// MSVC CRT lacks POSIX setenv/unsetenv. Mirror the existing _putenv_s shim
// pattern from shared/origami/tests/include/common.hpp:37 and
// projects/hipblaslt/clients/common/src/utility.cpp:299 so existing
// setenv/unsetenv call sites below stay portable. _putenv_s with an empty
// value removes the variable on Windows, matching POSIX unsetenv semantics
// (return-value shape differs -- POSIX is 0/-1, _putenv_s is 0/errno_t --
// but no caller here inspects the return). The !defined() guards keep us
// safe if a future Windows toolchain ever exposes the POSIX names. This
// header must not be included from non-test TUs.
#ifdef _WIN32
#if !defined(setenv)
inline int setenv(const char* name, const char* value, int /*overwrite*/)
{
    return _putenv_s(name, value);
}
#endif
#if !defined(unsetenv)
inline int unsetenv(const char* name)
{
    return _putenv_s(name, "");
}
#endif
#endif

inline void testing_aux_check_numerics_gemm(const Arguments& arg)
{
    (void)arg;

    // setenv/unsetenv are not thread-safe; serialize the body so concurrent
    // RUN_TEST_ON_THREADS_STREAMS threads can't interleave env mutations with
    // hipblasLtCreate() reads of those vars.
    static std::mutex           env_mutex;
    std::lock_guard<std::mutex> env_lock(env_mutex);

    auto reset_env = []() {
        unsetenv("HIPBLASLT_CHECK_NUMERICS");
        unsetenv("HIPBLASLT_CHECK_NUMERICS_SCAN_EVERY");
        unsetenv("HIPBLASLT_CHECK_NUMERICS_SCAN_FROM");
        unsetenv("HIPBLASLT_CHECK_NUMERICS_SCAN_UNTIL");
        unsetenv("HIPBLASLT_CHECK_NUMERICS_STOP_ON_FIRST");
    };

    // Sentinel for "no NaN injection" -- the loop is 1-indexed so call_ids
    // 1..total are valid injection points; 0 means clean run. A literal
    // constant + the assert below guard against a future refactor that
    // 0-indexes the loop and silently turns the clean-run scenario into
    // "inject at first call".
    constexpr int kNoInjection = 0;

    // Drives the scenario into out_reported. Lambda returns void so the
    // gtest ASSERT_EQ inside EXPECT_HIPBLAS_STATUS / CHECK_HIP_ERROR can
    // fail-fast cleanly (those macros use `return;` on failure, which
    // requires void context). All allocations are hoisted outside the
    // per-call loop -- only the H2D copy of A and the C zero-init vary
    // per call, so we malloc/create once and reuse, drastically cutting
    // the test's wallclock cost in CI.
    auto run_scenario = [&](int total, int inject_at, uint32_t* out_reported) {
        ASSERT_GE(inject_at, kNoInjection);
        ASSERT_LE(inject_at, total);
        ASSERT_NE(out_reported, nullptr);
        *out_reported = ~uint32_t(0);

        constexpr int M = 8, N = 8, K = 8;

        hipblasLtHandle_t handle = nullptr;
        EXPECT_HIPBLAS_STATUS(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
        hipStream_t stream = nullptr;
        CHECK_HIP_ERROR(hipStreamCreate(&stream));

        std::vector<float> A_clean(M * K, 1.0f);
        std::vector<float> B_h(K * N, 1.0f);
        std::vector<float> A_dirty = A_clean;
        A_dirty[0]                 = std::numeric_limits<float>::quiet_NaN();

        float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr;
        CHECK_HIP_ERROR(hipMalloc(&dA, sizeof(float) * M * K));
        CHECK_HIP_ERROR(hipMalloc(&dB, sizeof(float) * K * N));
        CHECK_HIP_ERROR(hipMalloc(&dC, sizeof(float) * M * N));
        CHECK_HIP_ERROR(hipMalloc(&dD, sizeof(float) * M * N));
        CHECK_HIP_ERROR(hipMemcpy(dB, B_h.data(), sizeof(float) * K * N, hipMemcpyHostToDevice));

        hipblasLtMatrixLayout_t lA = nullptr, lB = nullptr, lC = nullptr, lD = nullptr;
        EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutCreate(&lA, HIP_R_32F, M, K, M),
                              HIPBLAS_STATUS_SUCCESS);
        EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutCreate(&lB, HIP_R_32F, K, N, K),
                              HIPBLAS_STATUS_SUCCESS);
        EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutCreate(&lC, HIP_R_32F, M, N, M),
                              HIPBLAS_STATUS_SUCCESS);
        EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutCreate(&lD, HIP_R_32F, M, N, M),
                              HIPBLAS_STATUS_SUCCESS);

        hipblasLtMatmulDesc_t md = nullptr;
        EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescCreate(&md, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                              HIPBLAS_STATUS_SUCCESS);
        int32_t op = HIPBLAS_OP_N;
        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op)),
            HIPBLAS_STATUS_SUCCESS);
        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op)),
            HIPBLAS_STATUS_SUCCESS);

        hipblasLtMatmulPreference_t pref = nullptr;
        EXPECT_HIPBLAS_STATUS(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
        // 8x8x8 fp32 GEMM heuristic returns algos requiring well under 1 MiB;
        // 1 MiB is plenty and keeps per-scenario hipMalloc cheap.
        uint64_t ws_size = 1ull * 1024ull * 1024ull;
        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatmulPreferenceSetAttribute(pref,
                                                  HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &ws_size,
                                                  sizeof(ws_size)),
            HIPBLAS_STATUS_SUCCESS);

        hipblasLtMatmulHeuristicResult_t heur[1]{};
        int                              got = 0;
        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatmulAlgoGetHeuristic(handle, md, lA, lB, lC, lD, pref, 1, heur, &got),
            HIPBLAS_STATUS_SUCCESS);
        ASSERT_GT(got, 0);

        void* dW = nullptr;
        if(heur[0].workspaceSize > 0)
            CHECK_HIP_ERROR(hipMalloc(&dW, heur[0].workspaceSize));

        // Per-call: copy A (clean or dirty), zero C, run matmul, sync.
        // 1-indexed deliberately -- see kNoInjection comment above.
        for(int i = 1; i <= total; ++i)
        {
            const bool   dirty = (i == inject_at) && (inject_at != kNoInjection);
            const float* hA    = dirty ? A_dirty.data() : A_clean.data();
            CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(float) * M * K, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemset(dC, 0, sizeof(float) * M * N));

            float alpha = 1.0f, beta = 0.0f;
            EXPECT_HIPBLAS_STATUS(hipblasLtMatmul(handle,
                                                  md,
                                                  &alpha,
                                                  dA,
                                                  lA,
                                                  dB,
                                                  lB,
                                                  &beta,
                                                  dC,
                                                  lC,
                                                  dD,
                                                  lD,
                                                  &heur[0].algo,
                                                  dW,
                                                  heur[0].workspaceSize,
                                                  stream),
                                  HIPBLAS_STATUS_SUCCESS);
            CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        }

        EXPECT_HIPBLAS_STATUS(hipblasLtCheckNumericsDrain(handle, out_reported),
                              HIPBLAS_STATUS_SUCCESS);

        if(dW)
            CHECK_HIP_ERROR(hipFree(dW));
        CHECK_HIP_ERROR(hipFree(dD));
        CHECK_HIP_ERROR(hipFree(dC));
        CHECK_HIP_ERROR(hipFree(dB));
        CHECK_HIP_ERROR(hipFree(dA));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(md));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(lD));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(lC));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(lB));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(lA));
        CHECK_HIP_ERROR(hipStreamDestroy(stream));
        EXPECT_HIPBLAS_STATUS(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
    };

    uint32_t reported = 0;

    // Default sampling (every call), inject NaN at #5, expect 5.
    reset_env();
    setenv("HIPBLASLT_CHECK_NUMERICS", "1", 1);
    run_scenario(/*total=*/8, /*inject_at=*/5, &reported);
    EXPECT_EQ(reported, 5u);

    // First-call boundary (inject at #1) -- catches off-by-one in the call
    // counter where 0-indexed internal state would mis-report 0 or 2.
    reset_env();
    setenv("HIPBLASLT_CHECK_NUMERICS", "1", 1);
    run_scenario(/*total=*/4, /*inject_at=*/1, &reported);
    EXPECT_EQ(reported, 1u);

    // SCAN_EVERY=5 samples calls 5,10,...; with that modulus, call #7 is
    // NOT in the sampled set, so injecting NaN at #7 reaches D undetected
    // and the next sampled call (#10) writes a clean output that
    // overwrites it -- documented "miss" case in the sample. Catches a
    // regression that flips the modulus check (e.g. `% N == 1`).
    reset_env();
    setenv("HIPBLASLT_CHECK_NUMERICS", "1", 1);
    setenv("HIPBLASLT_CHECK_NUMERICS_SCAN_EVERY", "5", 1);
    run_scenario(/*total=*/12, /*inject_at=*/7, &reported);
    EXPECT_EQ(reported, 0u);

    // SCAN_EVERY=5, inject at #10 (sampled) -- expect 10.
    reset_env();
    setenv("HIPBLASLT_CHECK_NUMERICS", "1", 1);
    setenv("HIPBLASLT_CHECK_NUMERICS_SCAN_EVERY", "5", 1);
    run_scenario(/*total=*/12, /*inject_at=*/10, &reported);
    EXPECT_EQ(reported, 10u);

    // SCAN_FROM=10, SCAN_UNTIL=15: lower-bound inclusivity (#10) -- catches
    // a `<` vs `<=` regression on the FROM bound.
    reset_env();
    setenv("HIPBLASLT_CHECK_NUMERICS", "2", 1);
    setenv("HIPBLASLT_CHECK_NUMERICS_SCAN_FROM", "10", 1);
    setenv("HIPBLASLT_CHECK_NUMERICS_SCAN_UNTIL", "15", 1);
    run_scenario(/*total=*/15, /*inject_at=*/10, &reported);
    EXPECT_EQ(reported, 10u);

    // Same window: interior (#12), expect 12.
    reset_env();
    setenv("HIPBLASLT_CHECK_NUMERICS", "2", 1);
    setenv("HIPBLASLT_CHECK_NUMERICS_SCAN_FROM", "10", 1);
    setenv("HIPBLASLT_CHECK_NUMERICS_SCAN_UNTIL", "15", 1);
    run_scenario(/*total=*/15, /*inject_at=*/12, &reported);
    EXPECT_EQ(reported, 12u);

    // Same window: upper-bound inclusivity (#15) -- catches a `<` vs `<=`
    // regression on the UNTIL bound.
    reset_env();
    setenv("HIPBLASLT_CHECK_NUMERICS", "2", 1);
    setenv("HIPBLASLT_CHECK_NUMERICS_SCAN_FROM", "10", 1);
    setenv("HIPBLASLT_CHECK_NUMERICS_SCAN_UNTIL", "15", 1);
    run_scenario(/*total=*/15, /*inject_at=*/15, &reported);
    EXPECT_EQ(reported, 15u);

    // Same window: outside (#2) -- expect 0.
    reset_env();
    setenv("HIPBLASLT_CHECK_NUMERICS", "2", 1);
    setenv("HIPBLASLT_CHECK_NUMERICS_SCAN_FROM", "10", 1);
    setenv("HIPBLASLT_CHECK_NUMERICS_SCAN_UNTIL", "15", 1);
    run_scenario(/*total=*/15, /*inject_at=*/2, &reported);
    EXPECT_EQ(reported, 0u);

    // Clean run (no injection) -- catches a regression where the scanner
    // spuriously flags clean GEMMs.
    reset_env();
    setenv("HIPBLASLT_CHECK_NUMERICS", "1", 1);
    run_scenario(/*total=*/6, /*inject_at=*/kNoInjection, &reported);
    EXPECT_EQ(reported, 0u);

    // RAII redirect of std::cerr -> ostringstream for the lifetime of one
    // run_scenario call, so we can assert on the CHECK_NUMERICS log output.
    // The CHECK_NUMERICS log helper writes to get_logger_os() and falls
    // through to &std::cerr when the singleton's log_os is null (the
    // default when HIPBLASLT_LOG_LEVEL is unset, the typical CI config).
    // If HIPBLASLT_LOG_FILE is set in the environment, the helper bypasses
    // std::cerr entirely and these log assertions will fail loudly --
    // documented and acceptable; reproducible CI runs should not set it.
    struct CerrCapture
    {
        std::ostringstream oss;
        std::streambuf*    prev;
        CerrCapture()
            : prev(std::cerr.rdbuf(oss.rdbuf()))
        {
        }
        ~CerrCapture()
        {
            std::cerr.rdbuf(prev);
        }
        std::string str() const
        {
            return oss.str();
        }
    };

    // Counts non-overlapping occurrences of `needle` in `hay`.
    auto count_substr = [](const std::string& hay, const std::string& needle) {
        size_t n = 0, pos = 0;
        while((pos = hay.find(needle, pos)) != std::string::npos)
        {
            ++n;
            pos += needle.size();
        }
        return n;
    };

    // STOP_ON_FIRST=1: NaN at #3 trips the host-peek auto-drain on a
    // later call; the device flag is sticky so the on-demand drain still
    // reports 3. The per-call hipStreamSynchronize in run_scenario is what
    // makes #3's atomicCAS host-visible to the peek.
    //
    // Behavioral coverage of the new auto-drain log path:
    //   1. Exactly one "auto-drain on host peek" log line (CAS-elect dedup
    //      in scan_D's host-peek block).
    //   2. That line must reference call #3 (the injected first-NaN id).
    //   3. The on-demand drain that follows must NOT re-emit a duplicate
    //      log line for the same NaN (drain helper's CAS-elect on
    //      short_circuit suppresses the second log).
    {
        reset_env();
        setenv("HIPBLASLT_CHECK_NUMERICS", "1", 1);
        setenv("HIPBLASLT_CHECK_NUMERICS_STOP_ON_FIRST", "1", 1);

        std::string captured_log;
        {
            CerrCapture cap;
            run_scenario(/*total=*/8, /*inject_at=*/3, &reported);
            captured_log = cap.str();
        }

        EXPECT_EQ(reported, 3u);
        EXPECT_EQ(count_substr(captured_log, "auto-drain on host peek"), 1u)
            << "expected exactly one auto-drain log line; captured:\n"
            << captured_log;
        EXPECT_NE(captured_log.find("call #3"), std::string::npos)
            << "auto-drain log should reference the first-NaN call id (#3); captured:\n"
            << captured_log;
        // Both on-demand drain (hipblasLtCheckNumericsDrain) and handle
        // teardown drain (~handle dtor) call the drain helper with
        // short_circuit_out non-null; the CAS-elect must suppress both
        // duplicate log emissions for the same first-NaN event.
        EXPECT_EQ(count_substr(captured_log, "on-demand drain"), 0u)
            << "drain helper must suppress the duplicate on-demand log; captured:\n"
            << captured_log;
        // Match the drain helper's NaN-event line specifically -- the bare
        // substring "handle teardown" also appears in the STOP_ON_FIRST tail
        // report ("handle teardown: matmul calls (X..Y] were intentionally
        // skipped...") emitted from a separate code path in handle.cpp, which
        // is not what the dedup is meant to suppress.
        EXPECT_EQ(count_substr(captured_log, "handle teardown: first NaN"), 0u)
            << "drain helper must suppress the duplicate teardown log; captured:\n"
            << captured_log;
    }

    // STOP_ON_FIRST=1 + SCAN_EVERY=2: covers the under-tested interaction
    // where the host-peek path runs under sampling. Calls 2,4,6,... are
    // sampled; injecting NaN at #4 (sampled) trips the auto-drain on a
    // later call. The bisect-hint branch in the log helper fires when
    // every>1 -- this case ensures the auto-drain path correctly threads
    // scan_every into the log line and the sticky drain still reports 4.
    {
        reset_env();
        setenv("HIPBLASLT_CHECK_NUMERICS", "1", 1);
        setenv("HIPBLASLT_CHECK_NUMERICS_STOP_ON_FIRST", "1", 1);
        setenv("HIPBLASLT_CHECK_NUMERICS_SCAN_EVERY", "2", 1);

        std::string captured_log;
        {
            CerrCapture cap;
            run_scenario(/*total=*/10, /*inject_at=*/4, &reported);
            captured_log = cap.str();
        }

        EXPECT_EQ(reported, 4u);
        EXPECT_EQ(count_substr(captured_log, "auto-drain on host peek"), 1u)
            << "expected exactly one auto-drain log line under SCAN_EVERY=2; captured:\n"
            << captured_log;
        EXPECT_NE(captured_log.find("call #4"), std::string::npos)
            << "auto-drain log should reference call #4; captured:\n"
            << captured_log;
        EXPECT_NE(captured_log.find("scan_every=2"), std::string::npos)
            << "auto-drain log should thread the effective scan_every; captured:\n"
            << captured_log;
        EXPECT_EQ(count_substr(captured_log, "on-demand drain"), 0u)
            << "drain helper must suppress the duplicate on-demand log under "
               "sampling; captured:\n"
            << captured_log;
        // See note on the matching assertion above: tighten to the drain
        // helper's NaN-event prefix so the always-fires STOP_ON_FIRST tail
        // report doesn't false-trigger this check.
        EXPECT_EQ(count_substr(captured_log, "handle teardown: first NaN"), 0u)
            << "drain helper must suppress the duplicate teardown log under "
               "sampling; captured:\n"
            << captured_log;
    }

    reset_env();
}
