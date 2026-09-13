/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

// Targeted tests for ROCBLAS_LAYER=0x10 (rocblas_layer_mode_log_kernel_select).
// Self-contained: drives the C API directly with hipMalloc/hipMemcpy, sets env vars
// per-test, and parses the trace file to verify schema-level invariants. The selected
// kernel name depends on GPU arch and backend build, so we never golden-file the line.

#include "client_utility.hpp"
#include "device_vector.hpp"
#include "host_vector.hpp"
#include "rocblas_test.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <sstream>
#include <string>
#include <vector>

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error no filesystem found
#endif

#ifdef WIN32
#define setenv(A, B, C) _putenv_s(A, B)
#define unsetenv(A) _putenv_s(A, "")
#endif

namespace
{
    // Reads the entire trace file produced by a test into a single string for substring matching.
    std::string read_trace(const std::string& path)
    {
        std::ifstream     ifs(path);
        std::stringstream ss;
        ss << ifs.rdbuf();
        return ss.str();
    }

    // Counts occurrences of `needle` in `haystack`. Plain substring scan, no regex.
    size_t count_occurrences(const std::string& haystack, const std::string& needle)
    {
        if(needle.empty())
            return 0;
        size_t n = 0;
        for(size_t pos = 0; (pos = haystack.find(needle, pos)) != std::string::npos;
            pos += needle.size())
            ++n;
        return n;
    }

    // Generates a unique, temp-dir-rooted trace-file path for this test invocation
    // using rocblas_tempname().
    std::string make_trace_path(const char* test_name)
    {
        static const std::string tmp_dir = rocblas_tempname();
        const fs::path           fspath
            = tmp_dir + std::string("kernel_select_") + test_name + std::string(".log");
        const std::string p = fspath.generic_string();
        std::remove(p.c_str());
        return p;
    }

    // RAII: snapshot the env vars we mutate, restore on scope exit so test ordering doesn't matter.
    struct env_guard
    {
        env_guard()
            : m_layer(getenv("ROCBLAS_LAYER") ? getenv("ROCBLAS_LAYER") : "")
            , m_path(getenv("ROCBLAS_LOG_TRACE_PATH") ? getenv("ROCBLAS_LOG_TRACE_PATH") : "")
            , m_uselt(getenv("ROCBLAS_USE_HIPBLASLT") ? getenv("ROCBLAS_USE_HIPBLASLT") : "")
            , m_layer_set(getenv("ROCBLAS_LAYER") != nullptr)
            , m_path_set(getenv("ROCBLAS_LOG_TRACE_PATH") != nullptr)
            , m_uselt_set(getenv("ROCBLAS_USE_HIPBLASLT") != nullptr)
        {
        }
        ~env_guard()
        {
            auto restore = [](const char* name, const std::string& val, bool was_set) {
                if(was_set)
                    setenv(name, val.c_str(), 1);
                else
                    unsetenv(name);
            };
            restore("ROCBLAS_LAYER", m_layer, m_layer_set);
            restore("ROCBLAS_LOG_TRACE_PATH", m_path, m_path_set);
            restore("ROCBLAS_USE_HIPBLASLT", m_uselt, m_uselt_set);
        }

    private:
        std::string m_layer, m_path, m_uselt;
        bool        m_layer_set, m_path_set, m_uselt_set;
    };

} // namespace

// (1) trsm decomposition produces many sub-problem lines, all attributed to rocblas_strsm.
// Covers: the rocblas_internal_api_scope at trsm's _impl, multi-line decomposition output,
// and demonstrates the primary use-case the feature was designed for.
TEST(KernelSelect, TrsmDecompositionParentApi)
{
    env_guard         guard;
    const std::string trace_path = make_trace_path("trsm");
    constexpr int     m          = 512;
    constexpr int     n          = 512;
    constexpr float   alpha      = 1.0f;

    setenv("ROCBLAS_LAYER", "0x10", 1);
    setenv("ROCBLAS_LOG_TRACE_PATH", trace_path.c_str(), 1);

    {
        rocblas_handle handle;
        ASSERT_EQ(rocblas_create_handle(&handle), rocblas_status_success);

        host_vector<float> hA(size_t(m) * m);
        host_vector<float> hB(size_t(m) * n);
        for(int col = 0; col < m; ++col)
            for(int row = 0; row < m; ++row)
                hA[size_t(col) * m + row] = row == col ? float(m) : (row < col ? 0.5f : 0.0f);
        for(size_t i = 0; i < size_t(m) * n; ++i)
            hB[i] = 1.0f;

        device_vector<float> dA(size_t(m) * m);
        device_vector<float> dB(size_t(m) * n);
        ASSERT_EQ(dA.transfer_from(hA), hipSuccess);
        ASSERT_EQ(dB.transfer_from(hB), hipSuccess);

        ASSERT_EQ(rocblas_strsm(handle,
                                rocblas_side_left,
                                rocblas_fill_upper,
                                rocblas_operation_transpose,
                                rocblas_diagonal_non_unit,
                                m,
                                n,
                                &alpha,
                                dA,
                                m,
                                dB,
                                m),
                  rocblas_status_success);

        ASSERT_EQ(rocblas_destroy_handle(handle), rocblas_status_success);
    }

    const std::string trace = read_trace(trace_path);
    // trsm at this size decomposes into >>1 sub-problems; require multiple lines all carrying
    // the trsm parent. (Exact count depends on internal block size, so we only lower-bound.)
    const size_t lines = count_occurrences(trace, "parent_api=rocblas_strsm");
    EXPECT_GT(lines, size_t(1)) << "expected multiple sub-problems attributed to rocblas_strsm, "
                                << "got " << lines << " in " << trace_path;
    // Non-batched sub-problems (batch_count == 1) replay as gemm_ex.
    EXPECT_NE(trace.find("-f gemm_ex"), std::string::npos);
    EXPECT_NE(trace.find("# source="), std::string::npos);

    std::remove(trace_path.c_str());
}

// (2) Forcing ROCBLAS_USE_HIPBLASLT=0 should route every GEMM through Tensile.
// Covers: source=tensile branch, capture of solution->name(), and the Tensile kernel-name
// format (Cijk_* is Tensile's canonical naming convention).
TEST(KernelSelect, TensileBackendForced)
{
#ifndef BUILD_WITH_TENSILE
    GTEST_SKIP() << "Tensile not built in; cannot test tensile backend path";
#else
    env_guard         guard;
    const std::string trace_path = make_trace_path("tensile");
    constexpr int     m          = 256;
    constexpr int     n          = 256;
    constexpr int     k          = 256;
    constexpr float   alpha      = 1.0f;
    constexpr float   beta       = 0.0f;

    setenv("ROCBLAS_LAYER", "0x10", 1);
    setenv("ROCBLAS_LOG_TRACE_PATH", trace_path.c_str(), 1);
    setenv("ROCBLAS_USE_HIPBLASLT", "0", 1);

    {
        rocblas_handle handle;
        ASSERT_EQ(rocblas_create_handle(&handle), rocblas_status_success);

        device_vector<float> dA(size_t(m) * k);
        device_vector<float> dB(size_t(k) * n);
        device_vector<float> dC(size_t(m) * n);

        ASSERT_EQ(rocblas_sgemm(handle,
                                rocblas_operation_none,
                                rocblas_operation_none,
                                m,
                                n,
                                k,
                                &alpha,
                                dA,
                                m,
                                dB,
                                k,
                                &beta,
                                dC,
                                m),
                  rocblas_status_success);

        ASSERT_EQ(rocblas_destroy_handle(handle), rocblas_status_success);
    }

    const std::string trace = read_trace(trace_path);
    EXPECT_NE(trace.find("source=tensile"), std::string::npos)
        << "with ROCBLAS_USE_HIPBLASLT=0 expected source=tensile, got:\n"
        << trace;
    // Tensile solution names start with "Cijk_" by convention; if absent the capture is broken.
    EXPECT_NE(trace.find("kernel=Cijk_"), std::string::npos)
        << "expected Tensile-style kernel name, got:\n"
        << trace;
    EXPECT_EQ(trace.find("source=hipblaslt"), std::string::npos)
        << "ROCBLAS_USE_HIPBLASLT=0 should suppress hipblaslt source, got:\n"
        << trace;

    std::remove(trace_path.c_str());
#endif
}

// (3) Every emitted line must include the fallback_from= field, regardless of backend.
// Covers: a weak schema check the existing logging test omits — guards against accidental
// removal of the field or a code path that forgets to format it.
TEST(KernelSelect, FallbackFromFieldAlwaysPresent)
{
    env_guard         guard;
    const std::string trace_path = make_trace_path("fallback_field");
    constexpr int     m          = 256;
    constexpr int     n          = 256;
    constexpr int     k          = 256;
    constexpr float   alpha      = 1.0f;
    constexpr float   beta       = 0.0f;

    setenv("ROCBLAS_LAYER", "0x10", 1);
    setenv("ROCBLAS_LOG_TRACE_PATH", trace_path.c_str(), 1);

    {
        rocblas_handle handle;
        ASSERT_EQ(rocblas_create_handle(&handle), rocblas_status_success);

        device_vector<float> dA(size_t(m) * k);
        device_vector<float> dB(size_t(k) * n);
        device_vector<float> dC(size_t(m) * n);

        ASSERT_EQ(rocblas_sgemm(handle,
                                rocblas_operation_none,
                                rocblas_operation_none,
                                m,
                                n,
                                k,
                                &alpha,
                                dA,
                                m,
                                dB,
                                k,
                                &beta,
                                dC,
                                m),
                  rocblas_status_success);

        ASSERT_EQ(rocblas_destroy_handle(handle), rocblas_status_success);
    }

    const std::string trace = read_trace(trace_path);
    ASSERT_NE(trace.find("# source="), std::string::npos)
        << "no kernel-select line emitted, trace was:\n"
        << trace;
    // Every kernel-select line should have the field — counting source= vs fallback_from= gives
    // a uniform per-line check without per-line parsing.
    const size_t source_lines   = count_occurrences(trace, "# source=");
    const size_t fallback_lines = count_occurrences(trace, "fallback_from=");
    EXPECT_EQ(source_lines, fallback_lines)
        << "every kernel-select line must carry fallback_from=, got " << source_lines
        << " source= but " << fallback_lines << " fallback_from=";

    std::remove(trace_path.c_str());
}

// (4) Bits 0x8 and 0x10 are independent and can be combined; the same trace stream carries both.
// Covers: env-parse OR of multiple bits, and the back-compat invariant that 0x8 still works.
TEST(KernelSelect, CombinedWithInternalLayer)
{
    env_guard         guard;
    const std::string trace_path = make_trace_path("combined");
    constexpr int     m          = 256;
    constexpr int     n          = 256;
    constexpr int     k          = 256;
    constexpr float   alpha      = 1.0f;
    constexpr float   beta       = 0.0f;

    setenv("ROCBLAS_LAYER", "0x18", 1); // 0x8 internal + 0x10 kernel-select
    setenv("ROCBLAS_LOG_TRACE_PATH", trace_path.c_str(), 1);

    {
        rocblas_handle handle;
        ASSERT_EQ(rocblas_create_handle(&handle), rocblas_status_success);

        device_vector<float> dA(size_t(m) * k);
        device_vector<float> dB(size_t(k) * n);
        device_vector<float> dC(size_t(m) * n);

        ASSERT_EQ(rocblas_sgemm(handle,
                                rocblas_operation_none,
                                rocblas_operation_none,
                                m,
                                n,
                                k,
                                &alpha,
                                dA,
                                m,
                                dB,
                                k,
                                &beta,
                                dC,
                                m),
                  rocblas_status_success);

        ASSERT_EQ(rocblas_destroy_handle(handle), rocblas_status_success);
    }

    const std::string trace = read_trace(trace_path);
    // 0x8 emits one of these backend tags per GEMM.
    const bool has_0x8 = trace.find("rocblas_gemm_hipblaslt_backend") != std::string::npos
                         || trace.find("rocblas_gemm_tensile_backend") != std::string::npos
                         || trace.find("rocblas_gemm_source_backend") != std::string::npos;
    EXPECT_TRUE(has_0x8) << "0x8 internal backend line missing from combined trace:\n" << trace;
    // 0x10 emits the bench-style line.
    EXPECT_NE(trace.find("# source="), std::string::npos)
        << "0x10 kernel-select line missing from combined trace:\n"
        << trace;

    std::remove(trace_path.c_str());
}
