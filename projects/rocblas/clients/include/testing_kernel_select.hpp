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

#pragma once

// Tests for ROCBLAS_LAYER=0x10 (rocblas_layer_mode_log_kernel_select). The selected kernel
// depends on the GPU and the backends built in, so the lines are checked for schema and for the
// fields each path controls, never against a golden file.

#include "testing_common.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#ifdef WIN32
#include <stdlib.h>
#define setenv(A, B, C) _putenv_s(A, B)
#define unsetenv(A) _putenv_s(A, "")
#endif

namespace kernel_select
{
    inline bool contains(const std::string& s, const char* needle)
    {
        return s.find(needle) != std::string::npos;
    }

    inline bool ends_with(const std::string& s, const std::string& suffix)
    {
        return s.size() >= suffix.size()
               && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    // Sets an environment variable for the lifetime of the object, then restores it.
    class scoped_env
    {
        std::string m_name;
        std::string m_old;
        bool        m_was_set;

    public:
        scoped_env(const char* name, const std::string& value)
            : m_name(name)
            , m_was_set(getenv(name) != nullptr)
        {
            if(m_was_set)
                m_old = getenv(name);
            setenv(name, value.c_str(), 1);
        }
        ~scoped_env()
        {
            if(m_was_set)
                setenv(m_name.c_str(), m_old.c_str(), 1);
            else
                unsetenv(m_name.c_str());
        }
        scoped_env(const scoped_env&) = delete;
        scoped_env& operator=(const scoped_env&) = delete;
    };

    struct trace
    {
        std::string              text;
        std::vector<std::string> lines; // the kernel-select lines only
    };

    // Runs fn(handle) on a fresh handle with ROCBLAS_LAYER=layer and returns the trace.
    template <typename F>
    trace capture(const Arguments& arg, const char* layer, F&& fn)
    {
        static const std::string base  = rocblas_tempname();
        static int               count = 0;
        const std::string        path = base + "_kernel_select_" + std::to_string(count++) + ".log";
        trace                    result;
        {
            scoped_env layer_env("ROCBLAS_LAYER", layer);
            scoped_env path_env("ROCBLAS_LOG_TRACE_PATH", path);

            rocblas_local_handle handle{arg};
            fn(handle);
        } // the handle destructor flushes the trace stream

        std::ifstream ifs(path);
        for(std::string line; std::getline(ifs, line);)
        {
            result.text += line + "\n";
            if(contains(line, "# source="))
                result.lines.push_back(line);
        }
        ifs.close();
        std::remove(path.c_str());
        return result;
    }

    // Checks the fields every kernel-select line must carry.
    inline void expect_schema(const trace& t)
    {
        for(const auto& line : t.lines)
        {
            const std::string bench = line.substr(0, line.find('#'));
            EXPECT_EQ(line.rfind("./rocblas-bench", 0), size_t(0)) << line;
            EXPECT_TRUE(contains(bench, " -f gemm")) << line;
            EXPECT_TRUE(contains(line, " fallback_from=")) << line;
            EXPECT_TRUE(contains(line, " parent_api=")) << line;
            // Only gemm_strided_batched_ex takes strides, as in rocblas-bench.
            EXPECT_EQ(contains(bench, "--stride_a"), contains(bench, "-f gemm_strided_batched_ex "))
                << line;
        }
    }

    // Fills A, B and C of an m x n x k non-transposed GEMM with initialized data.
    template <typename T>
    struct gemm_data
    {
        host_matrix<T>   hA, hB, hC;
        device_matrix<T> dA, dB, dC;

        gemm_data(const Arguments& arg, int64_t m, int64_t n, int64_t k)
            : hA(m, k, m)
            , hB(k, n, k)
            , hC(m, n, m)
            , dA(m, k, m)
            , dB(k, n, k)
            , dC(m, n, m)
        {
            rocblas_init_matrix(
                hA, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix, true);
            rocblas_init_matrix(
                hB, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix);
            rocblas_init_matrix(
                hC, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix);
            EXPECT_EQ(dA.transfer_from(hA), hipSuccess);
            EXPECT_EQ(dB.transfer_from(hB), hipSuccess);
            EXPECT_EQ(dC.transfer_from(hC), hipSuccess);
        }
    };

    constexpr int64_t c_dim   = 64;
    constexpr float   c_alpha = 1.0f;
    constexpr float   c_beta  = 0.0f;

    // Runs rocblas_gemm_ex on a c_dim x n x c_dim float problem with C == D.
    inline rocblas_status run_sgemm_ex(rocblas_handle    handle,
                                       gemm_data<float>& d,
                                       int64_t           n,
                                       rocblas_gemm_algo algo,
                                       int32_t           solution_index,
                                       uint32_t          flags)
    {
        return rocblas_gemm_ex(handle,
                               rocblas_operation_none,
                               rocblas_operation_none,
                               c_dim,
                               n,
                               c_dim,
                               &c_alpha,
                               d.dA,
                               rocblas_datatype_f32_r,
                               c_dim,
                               d.dB,
                               rocblas_datatype_f32_r,
                               c_dim,
                               &c_beta,
                               d.dC,
                               rocblas_datatype_f32_r,
                               c_dim,
                               d.dC,
                               rocblas_datatype_f32_r,
                               c_dim,
                               rocblas_datatype_f32_r,
                               algo,
                               solution_index,
                               flags);
    }
} // namespace kernel_select

// trsm decomposes into several GEMM sub-problems, all attributed to rocblas_strsm.
inline void testing_kernel_select_trsm(const Arguments& arg)
{
    using namespace kernel_select;
    constexpr int64_t m = 512;
    constexpr int64_t n = 512;

    HOST_MEMCHECK(host_matrix<float>, hA, (m, m, m));
    HOST_MEMCHECK(host_matrix<float>, hB, (m, n, m));
    DEVICE_MEMCHECK(device_matrix<float>, dA, (m, m, m));
    DEVICE_MEMCHECK(device_matrix<float>, dB, (m, n, m));

    // Diagonally dominant upper-triangular A keeps the solve well conditioned.
    float* A = hA[0];
    for(int64_t col = 0; col < m; ++col)
        for(int64_t row = 0; row < m; ++row)
            A[col * m + row] = row == col ? float(m) : (row < col ? 0.5f : 0.0f);
    rocblas_init_matrix(hB, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix, true);
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));

    const trace t = capture(arg, "0x10", [&](rocblas_handle handle) {
        CHECK_ROCBLAS_ERROR(rocblas_strsm(handle,
                                          rocblas_side_left,
                                          rocblas_fill_upper,
                                          rocblas_operation_transpose,
                                          rocblas_diagonal_non_unit,
                                          m,
                                          n,
                                          &c_alpha,
                                          dA,
                                          m,
                                          dB,
                                          m));
    });

    expect_schema(t);
    EXPECT_GT(t.lines.size(), size_t(1)) << t.text;
    for(const auto& line : t.lines)
        EXPECT_TRUE(ends_with(line, "parent_api=rocblas_strsm")) << line;
}

// gemm, gemm_batched and gemm_strided_batched replay as gemm_ex, gemm_batched_ex and
// gemm_strided_batched_ex respectively.
inline void testing_kernel_select_layout(const Arguments& arg)
{
    using namespace kernel_select;
    constexpr int64_t d           = c_dim;
    constexpr int64_t batch_count = 3;
    constexpr int64_t stride      = d * d;

    {
        gemm_data<float> g(arg, d, d, d);
        const trace      t = capture(arg, "0x10", [&](rocblas_handle handle) {
            CHECK_ROCBLAS_ERROR(rocblas_sgemm(handle,
                                              rocblas_operation_none,
                                              rocblas_operation_none,
                                              d,
                                              d,
                                              d,
                                              &c_alpha,
                                              g.dA,
                                              d,
                                              g.dB,
                                              d,
                                              &c_beta,
                                              g.dC,
                                              d));
        });
        expect_schema(t);
        ASSERT_EQ(t.lines.size(), size_t(1)) << t.text;
        EXPECT_TRUE(contains(t.lines[0], "-f gemm_ex ")) << t.lines[0];
        EXPECT_FALSE(contains(t.lines[0], "--batch_count")) << t.lines[0];
        EXPECT_TRUE(ends_with(t.lines[0], "parent_api=rocblas_sgemm")) << t.lines[0];
    }

    {
        HOST_MEMCHECK(host_batch_matrix<float>, hA, (d, d, d, batch_count));
        HOST_MEMCHECK(host_batch_matrix<float>, hB, (d, d, d, batch_count));
        HOST_MEMCHECK(host_batch_matrix<float>, hC, (d, d, d, batch_count));
        DEVICE_MEMCHECK(device_batch_matrix<float>, dA, (d, d, d, batch_count));
        DEVICE_MEMCHECK(device_batch_matrix<float>, dB, (d, d, d, batch_count));
        DEVICE_MEMCHECK(device_batch_matrix<float>, dC, (d, d, d, batch_count));
        rocblas_init_matrix(
            hA, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix, true);
        rocblas_init_matrix(hB, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix);
        rocblas_init_matrix(hC, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix);
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
        CHECK_HIP_ERROR(dC.transfer_from(hC));

        const trace t = capture(arg, "0x10", [&](rocblas_handle handle) {
            CHECK_ROCBLAS_ERROR(rocblas_sgemm_batched(handle,
                                                      rocblas_operation_none,
                                                      rocblas_operation_none,
                                                      d,
                                                      d,
                                                      d,
                                                      &c_alpha,
                                                      dA.ptr_on_device(),
                                                      d,
                                                      dB.ptr_on_device(),
                                                      d,
                                                      &c_beta,
                                                      dC.ptr_on_device(),
                                                      d,
                                                      batch_count));
        });
        expect_schema(t);
        ASSERT_EQ(t.lines.size(), size_t(1)) << t.text;
        EXPECT_TRUE(contains(t.lines[0], "-f gemm_batched_ex ")) << t.lines[0];
        EXPECT_TRUE(contains(t.lines[0], "--batch_count 3 ")) << t.lines[0];
    }

    {
        HOST_MEMCHECK(host_strided_batch_matrix<float>, hA, (d, d, d, stride, batch_count));
        HOST_MEMCHECK(host_strided_batch_matrix<float>, hB, (d, d, d, stride, batch_count));
        HOST_MEMCHECK(host_strided_batch_matrix<float>, hC, (d, d, d, stride, batch_count));
        DEVICE_MEMCHECK(device_strided_batch_matrix<float>, dA, (d, d, d, stride, batch_count));
        DEVICE_MEMCHECK(device_strided_batch_matrix<float>, dB, (d, d, d, stride, batch_count));
        DEVICE_MEMCHECK(device_strided_batch_matrix<float>, dC, (d, d, d, stride, batch_count));
        rocblas_init_matrix(
            hA, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix, true);
        rocblas_init_matrix(hB, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix);
        rocblas_init_matrix(hC, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix);
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
        CHECK_HIP_ERROR(dC.transfer_from(hC));

        const trace t = capture(arg, "0x10", [&](rocblas_handle handle) {
            CHECK_ROCBLAS_ERROR(rocblas_sgemm_strided_batched(handle,
                                                              rocblas_operation_none,
                                                              rocblas_operation_none,
                                                              d,
                                                              d,
                                                              d,
                                                              &c_alpha,
                                                              dA,
                                                              d,
                                                              stride,
                                                              dB,
                                                              d,
                                                              stride,
                                                              &c_beta,
                                                              dC,
                                                              d,
                                                              stride,
                                                              batch_count));
        });
        expect_schema(t);
        ASSERT_EQ(t.lines.size(), size_t(1)) << t.text;
        EXPECT_TRUE(contains(t.lines[0], "-f gemm_strided_batched_ex ")) << t.lines[0];
        EXPECT_TRUE(contains(t.lines[0], "--stride_a 4096 ")) << t.lines[0];
        EXPECT_TRUE(contains(t.lines[0], "--batch_count 3 ")) << t.lines[0];
    }
}

// Bits 0x8 and 0x10 are independent and share the trace stream.
inline void testing_kernel_select_combined(const Arguments& arg)
{
    using namespace kernel_select;
    gemm_data<float> g(arg, c_dim, c_dim, c_dim);

    const trace t = capture(arg, "0x18", [&](rocblas_handle handle) {
        CHECK_ROCBLAS_ERROR(
            run_sgemm_ex(handle, g, c_dim, rocblas_gemm_algo_standard, 0, rocblas_gemm_flags_none));
    });

    expect_schema(t);
    EXPECT_EQ(t.lines.size(), size_t(1)) << t.text;
    EXPECT_TRUE(contains(t.text, "rocblas_gemm_hipblaslt_backend")
                || contains(t.text, "rocblas_gemm_tensile_backend")
                || contains(t.text, "rocblas_gemm_source_backend"))
        << t.text;
}

// An explicit source solution index runs the rocBLAS source kernels without trying a backend.
inline void testing_kernel_select_source(const Arguments& arg)
{
    using namespace kernel_select;
    constexpr int32_t c_source_solution = -2;
    gemm_data<float>  g(arg, c_dim, c_dim, c_dim);

    const trace t = capture(arg, "0x10", [&](rocblas_handle handle) {
        CHECK_ROCBLAS_ERROR(run_sgemm_ex(handle,
                                         g,
                                         c_dim,
                                         rocblas_gemm_algo_solution_index,
                                         c_source_solution,
                                         rocblas_gemm_flags_none));
    });

    expect_schema(t);
    ASSERT_EQ(t.lines.size(), size_t(1)) << t.text;
    const std::string& line = t.lines[0];
    EXPECT_TRUE(contains(line, "-f gemm_ex ")) << line;
    EXPECT_TRUE(contains(line, "--solution_index -2 ")) << line;
    EXPECT_TRUE(contains(
        line, "# source=source kernel=rocblas_gemm_source_solution_64 fallback_from=none "))
        << line;
    EXPECT_TRUE(ends_with(line, "parent_api=rocblas_gemm_ex")) << line;
}

// An explicit GEMV solution index runs an n == 1 GEMM through the GEMV kernels.
inline void testing_kernel_select_gemv(const Arguments& arg)
{
#if !defined(BUILD_WITH_TENSILE) && !defined(BUILD_WITH_HIPBLASLT)
    GTEST_SKIP() << "gemm_ex routes to GEMV only when a GEMM backend is built in";
#else
    using namespace kernel_select;
    constexpr int32_t c_gemv_solution = -9;
    gemm_data<float>  g(arg, c_dim, 1, c_dim);

    const trace t = capture(arg, "0x10", [&](rocblas_handle handle) {
        CHECK_ROCBLAS_ERROR(run_sgemm_ex(handle,
                                         g,
                                         1,
                                         rocblas_gemm_algo_solution_index,
                                         c_gemv_solution,
                                         rocblas_gemm_flags_none));
    });

    expect_schema(t);
    ASSERT_EQ(t.lines.size(), size_t(1)) << t.text;
    EXPECT_TRUE(contains(t.lines[0], " -n 1 ")) << t.lines[0];
    EXPECT_TRUE(contains(t.lines[0],
                         "# source=gemv kernel=rocblas_internal_gemv_launcher fallback_from=none "))
        << t.lines[0];
#endif
}

// With hipBLASLt off, Tensile runs the GEMM and its kernel name is reported.
inline void testing_kernel_select_tensile(const Arguments& arg)
{
#ifndef BUILD_WITH_TENSILE
    GTEST_SKIP() << "Tensile is not built in";
#else
    using namespace kernel_select;
    gemm_data<float> g(arg, c_dim, c_dim, c_dim);

    const trace t = capture(arg, "0x10", [&](rocblas_handle handle) {
        CHECK_ROCBLAS_ERROR(
            run_sgemm_ex(handle, g, c_dim, rocblas_gemm_algo_standard, 0, rocblas_gemm_flags_none));
    });

    expect_schema(t);
    ASSERT_EQ(t.lines.size(), size_t(1)) << t.text;
    EXPECT_TRUE(contains(t.lines[0], "# source=tensile kernel=Cijk_")) << t.lines[0];
    EXPECT_TRUE(contains(t.lines[0], " fallback_from=none ")) << t.lines[0];
#endif
}

// check_solution_index validates a solution without launching it, so nothing is logged.
inline void testing_kernel_select_check_only(const Arguments& arg)
{
#ifndef BUILD_WITH_TENSILE
    GTEST_SKIP() << "Tensile is not built in";
#else
    using namespace kernel_select;
    gemm_data<float> g(arg, c_dim, c_dim, c_dim);

    const trace t = capture(arg, "0x10", [&](rocblas_handle handle) {
        CHECK_ROCBLAS_ERROR(run_sgemm_ex(handle,
                                         g,
                                         c_dim,
                                         rocblas_gemm_algo_standard,
                                         0,
                                         rocblas_gemm_flags_check_solution_index));
    });

    EXPECT_TRUE(t.lines.empty()) << t.text;
#endif
}
