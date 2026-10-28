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

#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
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

    // STOP_ON_FIRST=1: NaN at #3 is claimed by atomicCAS on the device flag,
    // so the on-demand drain reports 3 on every GPU. The CAS winner also
    // plain-stores that id into a host mirror. When the mirror is visible by
    // the next matmul (the per-call hipStreamSynchronize in run_scenario),
    // scan_D emits exactly one "auto-drain on host peek" line and the drain
    // helper's CAS-elect suppresses the duplicate on-demand and teardown
    // lines. A link that cannot complete PCIe AtomicOps
    // (hipDeviceAttributeHostNativeAtomicSupported == 0, typical of a
    // PCH-attached GPU) may not observe that store; the on-demand drain then
    // emits the single NaN report instead. Either way the call id is reported
    // once (ROCM-30523).
    int host_native_atomics = 0;
    ASSERT_EQ(hipDeviceGetAttribute(&host_native_atomics,
                                    hipDeviceAttributeHostNativeAtomicSupported,
                                    /*device=*/0),
              hipSuccess);
    auto expect_one_nan_report = [&](const std::string& captured_log,
                                     const char*        call_token,
                                     const char*        extra_token) {
        const auto peek_n  = count_substr(captured_log, "auto-drain on host peek");
        const auto drain_n = count_substr(captured_log, "on-demand drain");
        if(host_native_atomics)
        {
            EXPECT_EQ(peek_n, 1u)
                << "host native atomics are supported; expected the host-peek "
                   "auto-drain line; captured:\n"
                << captured_log;
            EXPECT_EQ(drain_n, 0u)
                << "drain helper must suppress the duplicate on-demand log; captured:\n"
                << captured_log;
        }
        else
        {
            EXPECT_EQ(peek_n + drain_n, 1u)
                << "expected exactly one NaN report (host peek or on-demand drain); captured:\n"
                << captured_log;
        }
        EXPECT_NE(captured_log.find(call_token), std::string::npos)
            << "NaN report should reference " << call_token << "; captured:\n"
            << captured_log;
        if(extra_token != nullptr)
        {
            EXPECT_NE(captured_log.find(extra_token), std::string::npos)
                << "NaN report should contain \"" << extra_token << "\"; captured:\n"
                << captured_log;
        }
        // Match the drain helper's NaN-event line specifically -- the bare
        // substring "handle teardown" also appears in the STOP_ON_FIRST tail
        // report ("handle teardown: matmul calls (X..Y] were intentionally
        // skipped...") emitted from a separate code path in handle.cpp, which
        // is not what the dedup is meant to suppress.
        EXPECT_EQ(count_substr(captured_log, "handle teardown: first NaN"), 0u)
            << "drain helper must suppress the duplicate teardown log; captured:\n"
            << captured_log;
    };
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
        expect_one_nan_report(captured_log, "call #3", nullptr);
    }

    // STOP_ON_FIRST=1 + SCAN_EVERY=2: calls 2,4,6,... are sampled; injecting
    // NaN at #4 (sampled) is the one the drain reports. The bisect-hint
    // branch in the log helper fires when every>1, so the single NaN report
    // (host peek or on-demand drain; see above) must thread scan_every.
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
        expect_one_nan_report(captured_log, "call #4", "scan_every=2");
    }

    // Pointer-array batched matmul (HIPBLASLT_BATCH_MODE_POINTER_ARRAY): D is a
    // device array holding one matrix pointer per batch, so the scanner has to
    // read each pointer instead of striding from D. Reading it as a strided
    // buffer runs past the pointer array and faults the GPU. The NaN goes in
    // the last batch so a scanner that only looks at batch 0 also fails.
    // 128x128x128 fp32 matches matmul_pointer_array_distinct_c_d_layout_smoke,
    // which has pointer-array solutions on every architecture that runs it.
    constexpr int kNoNanBatch = -1;
    auto run_pointer_array_scenario = [&](int       nan_batch,
                                          int64_t   d_offset,
                                          bool      prefix_nan,
                                          bool      use_cpp,
                                          uint32_t* out_reported) {
        ASSERT_NE(out_reported, nullptr);
        *out_reported = ~uint32_t(0);

        constexpr int     M = 128, N = 128, K = 128;
        constexpr int32_t kBatch = 3;
        ASSERT_LT(nan_batch, kBatch);
        ASSERT_GE(d_offset, 0);

        hipblasLtHandle_t handle = nullptr;
        EXPECT_HIPBLAS_STATUS(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
        hipStream_t stream = nullptr;
        CHECK_HIP_ERROR(hipStreamCreate(&stream));

        std::vector<float> A_clean(M * K, 1.0f);
        std::vector<float> A_dirty = A_clean;
        A_dirty[M - 1]             = std::numeric_limits<float>::quiet_NaN();
        std::vector<float> B_h(K * N, 1.0f);

        std::vector<float> D_initial(d_offset + M * N, 0.0f);
        if(prefix_nan)
            std::fill(D_initial.begin(),
                      D_initial.begin() + d_offset,
                      std::numeric_limits<float>::quiet_NaN());

        std::vector<float*> hA(kBatch, nullptr), hB(kBatch, nullptr), hC(kBatch, nullptr),
            hD(kBatch, nullptr);
        for(int32_t b = 0; b < kBatch; ++b)
        {
            CHECK_HIP_ERROR(hipMalloc(&hA[b], sizeof(float) * M * K));
            CHECK_HIP_ERROR(hipMalloc(&hB[b], sizeof(float) * K * N));
            CHECK_HIP_ERROR(hipMalloc(&hC[b], sizeof(float) * M * N));
            CHECK_HIP_ERROR(hipMalloc(&hD[b], sizeof(float) * (d_offset + M * N)));
            CHECK_HIP_ERROR(hipMemcpy(
                hD[b], D_initial.data(), sizeof(float) * D_initial.size(), hipMemcpyHostToDevice));
            const float* srcA = (b == nan_batch) ? A_dirty.data() : A_clean.data();
            CHECK_HIP_ERROR(
                hipMemcpy(hA[b], srcA, sizeof(float) * M * K, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(
                hipMemcpy(hB[b], B_h.data(), sizeof(float) * K * N, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemset(hC[b], 0, sizeof(float) * M * N));
        }

        float **dA = nullptr, **dB = nullptr, **dC = nullptr, **dD = nullptr;
        CHECK_HIP_ERROR(hipMalloc(&dA, sizeof(float*) * kBatch));
        CHECK_HIP_ERROR(hipMalloc(&dB, sizeof(float*) * kBatch));
        CHECK_HIP_ERROR(hipMalloc(&dC, sizeof(float*) * kBatch));
        CHECK_HIP_ERROR(hipMalloc(&dD, sizeof(float*) * kBatch));
        CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(float*) * kBatch, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(float*) * kBatch, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dC, hC.data(), sizeof(float*) * kBatch, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dD, hD.data(), sizeof(float*) * kBatch, hipMemcpyHostToDevice));

        hipblasLtMatrixLayout_t lA = nullptr, lB = nullptr, lC = nullptr, lD = nullptr;
        EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutCreate(&lA, HIP_R_32F, M, K, M),
                              HIPBLAS_STATUS_SUCCESS);
        EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutCreate(&lB, HIP_R_32F, K, N, K),
                              HIPBLAS_STATUS_SUCCESS);
        EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutCreate(&lC, HIP_R_32F, M, N, M),
                              HIPBLAS_STATUS_SUCCESS);
        EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutCreate(&lD, HIP_R_32F, M, N, M),
                              HIPBLAS_STATUS_SUCCESS);
        const int32_t batch_count = kBatch;
        const int32_t batch_mode  = HIPBLASLT_BATCH_MODE_POINTER_ARRAY;
        for(hipblasLtMatrixLayout_t l : {lA, lB, lC, lD})
        {
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(
                    l, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(
                    l, HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batch_mode, sizeof(batch_mode)),
                HIPBLAS_STATUS_SUCCESS);
        }

        EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                                  lD, HIPBLASLT_MATRIX_LAYOUT_OFFSET, &d_offset, sizeof(d_offset)),
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
        uint64_t ws_size = 32ull * 1024ull * 1024ull;
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

        float alpha = 1.0f, beta = 0.0f;
        if(use_cpp)
        {
            hipblaslt_ext::Gemm gemm(handle, md, &alpha, dA, lA, dB, lB, &beta, dC, lC, dD, lD);
            gemm.setMaxWorkspaceBytes(heur[0].workspaceSize);
            EXPECT_HIPBLAS_STATUS(gemm.initialize(heur[0].algo, dW), HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(gemm.run(stream), HIPBLAS_STATUS_SUCCESS);
        }
        else
        {
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
        }
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));

        std::vector<float> D_actual(M * N);
        CHECK_HIP_ERROR(hipMemcpy(D_actual.data(),
                                  hD[kBatch - 1] + d_offset,
                                  sizeof(float) * M * N,
                                  hipMemcpyDeviceToHost));
        ASSERT_EQ(D_actual[0], 128.0f);
        ASSERT_EQ(std::isnan(D_actual.back()), nan_batch == kBatch - 1);

        EXPECT_HIPBLAS_STATUS(hipblasLtCheckNumericsDrain(handle, out_reported),
                              HIPBLAS_STATUS_SUCCESS);

        if(dW)
            CHECK_HIP_ERROR(hipFree(dW));
        for(float** p : {dA, dB, dC, dD})
            CHECK_HIP_ERROR(hipFree(p));
        for(int32_t b = 0; b < kBatch; ++b)
        {
            CHECK_HIP_ERROR(hipFree(hA[b]));
            CHECK_HIP_ERROR(hipFree(hB[b]));
            CHECK_HIP_ERROR(hipFree(hC[b]));
            CHECK_HIP_ERROR(hipFree(hD[b]));
        }
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(md));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(lD));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(lC));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(lB));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(lA));
        CHECK_HIP_ERROR(hipStreamDestroy(stream));
        EXPECT_HIPBLAS_STATUS(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
    };

    reset_env();
    setenv("HIPBLASLT_CHECK_NUMERICS", "1", 1);
    run_pointer_array_scenario(/*nan_batch=*/2,
                               /*d_offset=*/0,
                               /*prefix_nan=*/false,
                               /*use_cpp=*/false,
                               &reported);
    EXPECT_EQ(reported, 1u) << "NaN in the last pointer-array batch must be reported";

    reset_env();
    setenv("HIPBLASLT_CHECK_NUMERICS", "1", 1);
    run_pointer_array_scenario(kNoNanBatch,
                               /*d_offset=*/0,
                               /*prefix_nan=*/false,
                               /*use_cpp=*/false,
                               &reported);
    EXPECT_EQ(reported, 0u) << "clean pointer-array matmul must not be flagged";

    // HIPBLASLT_MATRIX_LAYOUT_OFFSET places each output after a full matrix of
    // unused data. Check both a NaN at the output's end and NaNs confined to
    // that unused prefix, through the C API and the extension API.
    constexpr int64_t kDOffset = 128 * 128;
    for(bool use_cpp : {false, true})
    {
        SCOPED_TRACE(use_cpp ? "extension API" : "C API");
        reset_env();
        setenv("HIPBLASLT_CHECK_NUMERICS", "1", 1);
        run_pointer_array_scenario(
            /*nan_batch=*/2, kDOffset, /*prefix_nan=*/false, use_cpp, &reported);
        EXPECT_EQ(reported, 1u) << "NaN in offset pointer-array output must be reported";

        reset_env();
        setenv("HIPBLASLT_CHECK_NUMERICS", "1", 1);
        run_pointer_array_scenario(
            kNoNanBatch, kDOffset, /*prefix_nan=*/true, use_cpp, &reported);
        EXPECT_EQ(reported, 0u) << "NaNs before offset pointer-array output must be ignored";
    }

    reset_env();
}
