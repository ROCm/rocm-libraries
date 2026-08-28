/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

// JIRA: AIROCBLAS-1372
//
// Tests for chunked workspace allocation in batched syrk/herk.
//
// The fix changes peak workspace from batch_count * tri(n) * sizeof(T) to
// min(batch_count, 65535) * tri(n) * sizeof(T).  The host-side launcher
// processes at most c_YZ_grid_launch_limit (65535) batches per iteration and
// reuses the same workspace buffer, so only one chunk's worth of triangle
// slots must be live at once.
//
// Scope: int32 strided-batched and batched (pointer-array) syrk/herk.
// Tests exercise all architectures: on gfx90a/gfx942 where the gemm-only
// workspace path is active, workspace-specific assertions (chunked formula,
// canary guard) are checked; on other architectures, the API calls are still
// exercised for correctness with no workspace.
//
// This file uses raw TEST() macros instead of the RocBLAS_Test / YAML-driven
// pattern because it tests workspace infrastructure plumbing rather than
// general BLAS correctness, and does not need parameterized matrix dimensions.

#include "client_utility.hpp"
#include "device_batch_vector.hpp"
#include "device_vector.hpp"
#include "rocblas.hpp"
#include "rocblas_test.hpp"
#include <algorithm>
#include <numeric>
#include <vector>

namespace
{
    // tri(n) = n*(n-1)/2 -- off-diagonal element count per batch in workspace.
    static size_t tri_n(rocblas_int n)
    {
        return (size_t(n) * size_t(n - 1)) / 2;
    }

    // Expected workspace bytes under the chunked scheme.
    // chunk = min(batch_count, c_YZ_grid_launch_limit).
    // Mirrors the production formula in rocblas_syrk_herk.hpp.
    template <typename T>
    static size_t chunked_workspace_bytes(rocblas_int n, rocblas_int batch_count)
    {
        constexpr rocblas_int limit = (1 << 16) - 1; // 65535 -- must match c_YZ_grid_launch_limit
        size_t                chunk = size_t(std::min(batch_count, limit));
        return tri_n(n) * sizeof(T) * chunk;
    }

    // -----------------------------------------------------------------------
    // Templated workspace-size query helpers.
    //
    // NOTE: The complex-T branch uses real-of-T for alpha/beta, matching the
    // herk API signature.  This is correct for herk but would be wrong for
    // csyrk / zsyrk (which take complex alpha/beta).  Do NOT reuse these
    // helpers for complex-syrk queries without adding a separate code path.
    //
    // The two overloads cover strided-batched and batched API variants.
    // -----------------------------------------------------------------------

    // Strided-batched workspace query.
    template <typename T, typename ApiFunc>
    static bool query_strided_workspace(rocblas_handle handle,
                                        ApiFunc        api,
                                        rocblas_int    n,
                                        rocblas_int    k,
                                        rocblas_int    lda,
                                        rocblas_int    ldc,
                                        rocblas_int    batch_count,
                                        size_t*        bytes)
    {
        const rocblas_fill      uplo    = rocblas_fill_lower;
        const rocblas_operation transA  = rocblas_operation_none;
        const rocblas_stride    strideA = rocblas_stride(lda) * k;
        const rocblas_stride    strideC = rocblas_stride(ldc) * n;

        rocblas_status st = rocblas_start_device_memory_size_query(handle);
        EXPECT_EQ(st, rocblas_status_success);
        if(st != rocblas_status_success)
            return false;

        // herk takes real alpha/beta; syrk takes T-typed alpha/beta.
        if constexpr(rocblas_is_complex<T>)
        {
            using U       = decltype(std::real(T{}));
            const U alpha = U(1);
            const U beta  = U(0);
            st            = api(handle,
                     uplo,
                     transA,
                     n,
                     k,
                     &alpha,
                     nullptr,
                     lda,
                     strideA,
                     &beta,
                     nullptr,
                     ldc,
                     strideC,
                     batch_count);
        }
        else
        {
            const T alpha = T(1);
            const T beta  = T(0);
            st            = api(handle,
                     uplo,
                     transA,
                     n,
                     k,
                     &alpha,
                     nullptr,
                     lda,
                     strideA,
                     &beta,
                     nullptr,
                     ldc,
                     strideC,
                     batch_count);
        }

        EXPECT_TRUE(st == rocblas_status_size_increased || st == rocblas_status_size_unchanged)
            << "size query returned " << rocblas_status_to_string(st);
        if(st != rocblas_status_size_increased && st != rocblas_status_size_unchanged)
            return false;

        st = rocblas_stop_device_memory_size_query(handle, bytes);
        EXPECT_EQ(st, rocblas_status_success);
        return st == rocblas_status_success;
    }

    // Batched (pointer-array) workspace query.
    template <typename T, typename ApiFunc>
    static bool query_batched_workspace(rocblas_handle handle,
                                        ApiFunc        api,
                                        rocblas_int    n,
                                        rocblas_int    k,
                                        rocblas_int    lda,
                                        rocblas_int    ldc,
                                        rocblas_int    batch_count,
                                        size_t*        bytes)
    {
        const rocblas_fill      uplo   = rocblas_fill_lower;
        const rocblas_operation transA = rocblas_operation_none;

        rocblas_status st = rocblas_start_device_memory_size_query(handle);
        EXPECT_EQ(st, rocblas_status_success);
        if(st != rocblas_status_success)
            return false;

        if constexpr(rocblas_is_complex<T>)
        {
            using U       = decltype(std::real(T{}));
            const U alpha = U(1);
            const U beta  = U(0);
            st            = api(
                handle, uplo, transA, n, k, &alpha, nullptr, lda, &beta, nullptr, ldc, batch_count);
        }
        else
        {
            const T alpha = T(1);
            const T beta  = T(0);
            st            = api(
                handle, uplo, transA, n, k, &alpha, nullptr, lda, &beta, nullptr, ldc, batch_count);
        }

        EXPECT_TRUE(st == rocblas_status_size_increased || st == rocblas_status_size_unchanged)
            << "size query returned " << rocblas_status_to_string(st);
        if(st != rocblas_status_size_increased && st != rocblas_status_size_unchanged)
            return false;

        st = rocblas_stop_device_memory_size_query(handle, bytes);
        EXPECT_EQ(st, rocblas_status_success);
        return st == rocblas_status_success;
    }

    // -----------------------------------------------------------------------
    // Workspace size query tests
    //
    // When the gemm-only workspace path is active:
    //   batch_count > 65535: workspace equals the chunked bound and is
    //     strictly less than the old un-capped formula.
    //   batch_count <= 65535: workspace equals batch_count * tri(n) * sizeof(T).
    //
    // When the workspace path is NOT active (non-target arch or thresholds
    // not met), the query returns 0 and the test verifies that fact without
    // skipping.
    // -----------------------------------------------------------------------

    // When the workspace path is active, check that the reported value
    // matches the chunked formula.  When batch_count > limit, also verify
    // capping.  When the path is NOT active, the query returned 0 — no
    // assertion needed.
    template <typename T>
    static void ws_check_capped(size_t reported, rocblas_int n, rocblas_int bc)
    {
        if(reported > 0)
        {
            EXPECT_EQ(reported, chunked_workspace_bytes<T>(n, bc));
            EXPECT_LT(reported, tri_n(n) * sizeof(T) * size_t(bc)) << "workspace not capped";
        }
    }

    template <typename T>
    static void ws_check_uncapped(size_t reported, rocblas_int n, rocblas_int bc)
    {
        if(reported > 0)
        {
            EXPECT_EQ(reported, tri_n(n) * sizeof(T) * size_t(bc));
        }
    }

    // ssyrk strided-batched
    TEST(syrk_herk_chunked_workspace_size, ssyrk_strided_131070)
    {
        rocblas_local_handle handle;
        const rocblas_int    n = 32, k = 500, bc = 131070;
        size_t               reported = 0;
        ASSERT_TRUE(query_strided_workspace<float>(
            handle, rocblas_ssyrk_strided_batched, n, k, n, n, bc, &reported));
        ws_check_capped<float>(reported, n, bc);
    }

    TEST(syrk_herk_chunked_workspace_size, ssyrk_strided_65536)
    {
        rocblas_local_handle handle;
        const rocblas_int    n = 32, k = 500, bc = 65536;
        size_t               reported = 0;
        ASSERT_TRUE(query_strided_workspace<float>(
            handle, rocblas_ssyrk_strided_batched, n, k, n, n, bc, &reported));
        ws_check_capped<float>(reported, n, bc);
    }

    TEST(syrk_herk_chunked_workspace_size, ssyrk_strided_65535)
    {
        rocblas_local_handle handle;
        const rocblas_int    n = 32, k = 500, bc = 65535;
        size_t               reported = 0;
        ASSERT_TRUE(query_strided_workspace<float>(
            handle, rocblas_ssyrk_strided_batched, n, k, n, n, bc, &reported));
        ws_check_uncapped<float>(reported, n, bc);
    }

    TEST(syrk_herk_chunked_workspace_size, ssyrk_strided_100)
    {
        rocblas_local_handle handle;
        const rocblas_int    n = 32, k = 500, bc = 100;
        size_t               reported = 0;
        ASSERT_TRUE(query_strided_workspace<float>(
            handle, rocblas_ssyrk_strided_batched, n, k, n, n, bc, &reported));
        ws_check_uncapped<float>(reported, n, bc);
    }

    // ssyrk batched (pointer-array)
    TEST(syrk_herk_chunked_workspace_size, ssyrk_batched_131070)
    {
        rocblas_local_handle handle;
        const rocblas_int    n = 32, k = 500, bc = 131070;
        size_t               reported = 0;
        ASSERT_TRUE(query_batched_workspace<float>(
            handle, rocblas_ssyrk_batched, n, k, n, n, bc, &reported));
        ws_check_capped<float>(reported, n, bc);
    }

    TEST(syrk_herk_chunked_workspace_size, ssyrk_batched_65535)
    {
        rocblas_local_handle handle;
        const rocblas_int    n = 32, k = 500, bc = 65535;
        size_t               reported = 0;
        ASSERT_TRUE(query_batched_workspace<float>(
            handle, rocblas_ssyrk_batched, n, k, n, n, bc, &reported));
        ws_check_uncapped<float>(reported, n, bc);
    }

    // dsyrk strided-batched (double precision)
    TEST(syrk_herk_chunked_workspace_size, dsyrk_strided_131070)
    {
        rocblas_local_handle handle;
        const rocblas_int    n = 32, k = 500, bc = 131070;
        size_t               reported = 0;
        ASSERT_TRUE(query_strided_workspace<double>(
            handle, rocblas_dsyrk_strided_batched, n, k, n, n, bc, &reported));
        ws_check_capped<double>(reported, n, bc);
    }

    // cherk strided-batched
    TEST(syrk_herk_chunked_workspace_size, cherk_strided_131070)
    {
        rocblas_local_handle handle;
        const rocblas_int    n = 32, k = 500, bc = 131070;
        size_t               reported = 0;
        ASSERT_TRUE(query_strided_workspace<rocblas_float_complex>(
            handle, rocblas_cherk_strided_batched, n, k, n, n, bc, &reported));
        ws_check_capped<rocblas_float_complex>(reported, n, bc);
    }

    TEST(syrk_herk_chunked_workspace_size, cherk_strided_65536)
    {
        rocblas_local_handle handle;
        const rocblas_int    n = 32, k = 500, bc = 65536;
        size_t               reported = 0;
        ASSERT_TRUE(query_strided_workspace<rocblas_float_complex>(
            handle, rocblas_cherk_strided_batched, n, k, n, n, bc, &reported));
        ws_check_capped<rocblas_float_complex>(reported, n, bc);
    }

    TEST(syrk_herk_chunked_workspace_size, cherk_strided_65535)
    {
        rocblas_local_handle handle;
        const rocblas_int    n = 32, k = 500, bc = 65535;
        size_t               reported = 0;
        ASSERT_TRUE(query_strided_workspace<rocblas_float_complex>(
            handle, rocblas_cherk_strided_batched, n, k, n, n, bc, &reported));
        ws_check_uncapped<rocblas_float_complex>(reported, n, bc);
    }

    TEST(syrk_herk_chunked_workspace_size, cherk_strided_100)
    {
        rocblas_local_handle handle;
        const rocblas_int    n = 32, k = 500, bc = 100;
        size_t               reported = 0;
        ASSERT_TRUE(query_strided_workspace<rocblas_float_complex>(
            handle, rocblas_cherk_strided_batched, n, k, n, n, bc, &reported));
        ws_check_uncapped<rocblas_float_complex>(reported, n, bc);
    }

    // cherk batched (pointer-array)
    TEST(syrk_herk_chunked_workspace_size, cherk_batched_131070)
    {
        rocblas_local_handle handle;
        const rocblas_int    n = 32, k = 500, bc = 131070;
        size_t               reported = 0;
        ASSERT_TRUE(query_batched_workspace<rocblas_float_complex>(
            handle, rocblas_cherk_batched, n, k, n, n, bc, &reported));
        ws_check_capped<rocblas_float_complex>(reported, n, bc);
    }

    TEST(syrk_herk_chunked_workspace_size, cherk_batched_65535)
    {
        rocblas_local_handle handle;
        const rocblas_int    n = 32, k = 500, bc = 65535;
        size_t               reported = 0;
        ASSERT_TRUE(query_batched_workspace<rocblas_float_complex>(
            handle, rocblas_cherk_batched, n, k, n, n, bc, &reported));
        ws_check_uncapped<rocblas_float_complex>(reported, n, bc);
    }

    // -----------------------------------------------------------------------
    // Canary / correctness tests
    //
    // Allocate an exact-sized workspace matching the new chunked formula and
    // append a contiguous guard region filled with a sentinel byte.  If the
    // kernel writes past the workspace boundary the guard bytes change and the
    // test fails.
    // -----------------------------------------------------------------------

    constexpr unsigned char c_canary_byte  = 0xA5;
    constexpr size_t        c_canary_bytes = 8192;

    static bool fill_canary(void* base, size_t workspace_bytes)
    {
        hipError_t err
            = hipMemset(static_cast<char*>(base) + workspace_bytes, c_canary_byte, c_canary_bytes);
        EXPECT_EQ(err, hipSuccess) << "hipMemset guard: " << hipGetErrorString(err);
        if(err != hipSuccess)
            return false;
        err = hipStreamSynchronize(nullptr);
        EXPECT_EQ(err, hipSuccess) << "sync guard fill: " << hipGetErrorString(err);
        return err == hipSuccess;
    }

    static void expect_canary_clean(rocblas_handle handle, void* base, size_t workspace_bytes)
    {
        hipStream_t stream = nullptr;
        ASSERT_EQ(rocblas_get_stream(handle, &stream), rocblas_status_success);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        std::vector<unsigned char> host(c_canary_bytes);
        ASSERT_EQ(hipMemcpy(host.data(),
                            static_cast<char*>(base) + workspace_bytes,
                            c_canary_bytes,
                            hipMemcpyDeviceToHost),
                  hipSuccess);

        size_t differing = 0;
        size_t first     = c_canary_bytes;
        for(size_t i = 0; i < c_canary_bytes; ++i)
            if(host[i] != c_canary_byte)
            {
                if(first == c_canary_bytes)
                    first = i;
                ++differing;
            }
        EXPECT_EQ(differing, size_t(0))
            << differing << " guard byte(s) differ; first at +" << first << " byte(s) past the "
            << workspace_bytes << "-byte workspace";
    }

    // Strided-batched canary helper, templated over T.
    // On workspace-path architectures: allocates exact-sized workspace with
    // canary guard, runs the API, verifies the guard is clean.
    // On other architectures: exercises the API call with a small batch to
    // avoid unnecessary large allocations.
    template <typename T, typename ApiFunc>
    static void run_strided_canary(ApiFunc      api,
                                   rocblas_int  n,
                                   rocblas_int  k,
                                   rocblas_int  batch_count,
                                   rocblas_fill uplo = rocblas_fill_lower)
    {
        rocblas_local_handle handle;

        size_t queried = 0;
        ASSERT_TRUE(query_strided_workspace<T>(handle, api, n, k, n, n, batch_count, &queried));

        const bool ws_active = queried > 0;

        // On non-workspace archs, use a small batch to avoid large allocations;
        // multi-chunk iteration is only meaningful when the workspace path is active.
        const rocblas_int run_bc = ws_active ? batch_count : std::min(batch_count, rocblas_int(2));

        void* d_ws = nullptr;
        if(ws_active)
        {
            EXPECT_EQ(queried, chunked_workspace_bytes<T>(n, batch_count))
                << "queried workspace diverges from test formula";
            ASSERT_EQ((hipMalloc)(&d_ws, queried + c_canary_bytes), hipSuccess);
            ASSERT_TRUE(fill_canary(d_ws, queried));
            ASSERT_EQ(rocblas_set_workspace(handle, d_ws, queried), rocblas_status_success);
        }

        // A aliased across all batches via stride_A=0 (read-only, safe to share).
        const size_t a_elems = size_t(n) * size_t(k);
        const size_t c_elems = size_t(n) * size_t(n) * size_t(run_bc);

        device_vector<T> dA(a_elems);
        device_vector<T> dC(c_elems);
        ASSERT_EQ(dA.memcheck(), hipSuccess);
        ASSERT_EQ(dC.memcheck(), hipSuccess);
        ASSERT_EQ(hipMemset((T*)dA, 0, a_elems * sizeof(T)), hipSuccess);
        ASSERT_EQ(hipMemset((T*)dC, 0, c_elems * sizeof(T)), hipSuccess);

        // herk takes real alpha/beta; syrk takes T-typed alpha/beta.
        if constexpr(rocblas_is_complex<T>)
        {
            using U       = decltype(std::real(T{}));
            const U alpha = U(1);
            const U beta  = U(1);
            ASSERT_EQ(api(handle,
                          uplo,
                          rocblas_operation_none,
                          n,
                          k,
                          &alpha,
                          (T*)dA,
                          n,
                          0,
                          &beta,
                          (T*)dC,
                          n,
                          rocblas_stride(n) * n,
                          run_bc),
                      rocblas_status_success);
        }
        else
        {
            const T alpha = T(1);
            const T beta  = T(1);
            ASSERT_EQ(api(handle,
                          uplo,
                          rocblas_operation_none,
                          n,
                          k,
                          &alpha,
                          (T*)dA,
                          n,
                          0,
                          &beta,
                          (T*)dC,
                          n,
                          rocblas_stride(n) * n,
                          run_bc),
                      rocblas_status_success);
        }

        if(ws_active)
        {
            expect_canary_clean(handle, d_ws, queried);
            ASSERT_EQ(rocblas_set_workspace(handle, nullptr, 0), rocblas_status_success);
            ASSERT_EQ((hipFree)(d_ws), hipSuccess);
        }
    }

    // Batched (pointer-array) canary helper, templated over T.
    // Uses a single contiguous hipMemset (device_batch_vector allocates one
    // contiguous block internally).
    template <typename T, typename ApiFunc>
    static void
        run_batched_canary(ApiFunc api, rocblas_int n, rocblas_int k, rocblas_int batch_count)
    {
        rocblas_local_handle handle;

        size_t queried = 0;
        ASSERT_TRUE(query_batched_workspace<T>(handle, api, n, k, n, n, batch_count, &queried));

        const bool ws_active = queried > 0;

        const rocblas_int run_bc = ws_active ? batch_count : std::min(batch_count, rocblas_int(2));

        void* d_ws = nullptr;
        if(ws_active)
        {
            EXPECT_EQ(queried, chunked_workspace_bytes<T>(n, batch_count))
                << "queried workspace diverges from test formula";
            ASSERT_EQ((hipMalloc)(&d_ws, queried + c_canary_bytes), hipSuccess);
            ASSERT_TRUE(fill_canary(d_ws, queried));
            ASSERT_EQ(rocblas_set_workspace(handle, d_ws, queried), rocblas_status_success);
        }

        const size_t a_elems = size_t(n) * size_t(k);
        const size_t c_elems = size_t(n) * size_t(n);

        device_batch_vector<T> dA(a_elems, 1, run_bc);
        device_batch_vector<T> dC(c_elems, 1, run_bc);
        ASSERT_EQ(dA.memcheck(), hipSuccess);
        ASSERT_EQ(dC.memcheck(), hipSuccess);
        ASSERT_EQ(hipMemset(dA[0], 0, a_elems * sizeof(T) * run_bc), hipSuccess);
        ASSERT_EQ(hipMemset(dC[0], 0, c_elems * sizeof(T) * run_bc), hipSuccess);

        if constexpr(rocblas_is_complex<T>)
        {
            using U       = decltype(std::real(T{}));
            const U alpha = U(1);
            const U beta  = U(1);
            ASSERT_EQ(api(handle,
                          rocblas_fill_lower,
                          rocblas_operation_none,
                          n,
                          k,
                          &alpha,
                          dA.ptr_on_device(),
                          n,
                          &beta,
                          dC.ptr_on_device(),
                          n,
                          run_bc),
                      rocblas_status_success);
        }
        else
        {
            const T alpha = T(1);
            const T beta  = T(1);
            ASSERT_EQ(api(handle,
                          rocblas_fill_lower,
                          rocblas_operation_none,
                          n,
                          k,
                          &alpha,
                          dA.ptr_on_device(),
                          n,
                          &beta,
                          dC.ptr_on_device(),
                          n,
                          run_bc),
                      rocblas_status_success);
        }

        if(ws_active)
        {
            expect_canary_clean(handle, d_ws, queried);
            ASSERT_EQ(rocblas_set_workspace(handle, nullptr, 0), rocblas_status_success);
            ASSERT_EQ((hipFree)(d_ws), hipSuccess);
        }
    }

    // -----------------------------------------------------------------------
    // Canary tests: strided-batched
    //
    // batch_count = 131070: two saturated grid-z passes.
    // batch_count = 65539:  boundary case (one element into the second pass).
    // batch_count = 1:      degenerate single-batch (regression guard).
    // -----------------------------------------------------------------------

    TEST(syrk_herk_chunked_workspace_correctness, ssyrk_strided_batched_131070)
    {
        run_strided_canary<float>(rocblas_ssyrk_strided_batched, 2, 500, 131070);
    }

    TEST(syrk_herk_chunked_workspace_correctness, ssyrk_strided_batched_65539)
    {
        run_strided_canary<float>(rocblas_ssyrk_strided_batched, 2, 500, 65539);
    }

    TEST(syrk_herk_chunked_workspace_correctness, ssyrk_strided_batched_1)
    {
        run_strided_canary<float>(rocblas_ssyrk_strided_batched, 2, 500, 1);
    }

    // fill_upper: exercises the is_upper=true template branch with distinct
    // triangle index arithmetic.
    TEST(syrk_herk_chunked_workspace_correctness, ssyrk_strided_batched_upper_131070)
    {
        run_strided_canary<float>(
            rocblas_ssyrk_strided_batched, 2, 500, 131070, rocblas_fill_upper);
    }

    TEST(syrk_herk_chunked_workspace_correctness, cherk_strided_batched_131070)
    {
        run_strided_canary<rocblas_float_complex>(rocblas_cherk_strided_batched, 2, 500, 131070);
    }

    TEST(syrk_herk_chunked_workspace_correctness, cherk_strided_batched_65539)
    {
        run_strided_canary<rocblas_float_complex>(rocblas_cherk_strided_batched, 2, 500, 65539);
    }

    TEST(syrk_herk_chunked_workspace_correctness, cherk_strided_batched_upper_131070)
    {
        run_strided_canary<rocblas_float_complex>(
            rocblas_cherk_strided_batched, 2, 500, 131070, rocblas_fill_upper);
    }

    // -----------------------------------------------------------------------
    // Canary tests: batched (pointer-array)
    // -----------------------------------------------------------------------

    TEST(syrk_herk_chunked_workspace_correctness, ssyrk_batched_131070)
    {
        run_batched_canary<float>(rocblas_ssyrk_batched, 2, 500, 131070);
    }

    TEST(syrk_herk_chunked_workspace_correctness, ssyrk_batched_65539)
    {
        run_batched_canary<float>(rocblas_ssyrk_batched, 2, 500, 65539);
    }

    TEST(syrk_herk_chunked_workspace_correctness, cherk_batched_131070)
    {
        run_batched_canary<rocblas_float_complex>(rocblas_cherk_batched, 2, 500, 131070);
    }

    TEST(syrk_herk_chunked_workspace_correctness, cherk_batched_65539)
    {
        run_batched_canary<rocblas_float_complex>(rocblas_cherk_batched, 2, 500, 65539);
    }

    // -----------------------------------------------------------------------
    // Numerical correctness test
    //
    // Verifies that syrk produces correct results by computing
    // C = alpha * A * A^T + beta * C with known data and comparing to a
    // host-side reference.  On workspace-path architectures this exercises
    // the chunked save-GEMM-restore cycle; on other architectures it
    // exercises the standard syrk kernel.  Expected values are the same
    // regardless of which internal path is taken.
    //
    // batch_count is set above 65535 to exercise the multi-chunk path
    // when active.  Batches 0 and (batch_count-1) are spot-checked; the
    // middle batches share the same aliased A (stride_A=0) and
    // zero-initialised C.
    // -----------------------------------------------------------------------

    TEST(syrk_herk_chunked_workspace_correctness, ssyrk_strided_numerical)
    {
        rocblas_local_handle handle;

        const rocblas_int n = 2, k = 500;
        // Just above the grid limit to force two chunks on workspace-path archs.
        const rocblas_int batch_count = 65536;

        // A is a 2x500 matrix.  Fill with a simple pattern: A[i,j] = 1.0f
        // so that A * A^T = k * ones(n,n) = [[500,500],[500,500]].
        const size_t a_elems = size_t(n) * size_t(k);
        const size_t c_elems = size_t(n) * size_t(n);

        // Host-side reference: C_ref = alpha * A * A^T + beta * C_init
        // With alpha=1, beta=1, A=all-ones, C_init=identity:
        //   A*A^T = [[k, k],[k, k]]
        //   C_ref (lower fill, including diagonal) = [[k+1, k],[k, k+1]]
        // The workspace path operates as: (1) save upper triangle of C to W_C,
        // (2) GEMM overwrites all of C, (3) restore saved upper triangle from
        // W_C back to C.  So the lower triangle gets the GEMM result and the
        // upper triangle is restored to its C_init values.
        const float alpha = 1.0f;
        const float beta  = 1.0f;

        std::vector<float> h_A(a_elems, 1.0f);
        // C_init: identity (diagonal = 1, off-diagonal = 0) for each batch.
        std::vector<float> h_C(c_elems * batch_count, 0.0f);
        for(rocblas_int b = 0; b < batch_count; ++b)
        {
            for(rocblas_int i = 0; i < n; ++i)
                h_C[b * c_elems + i + i * n] = 1.0f;
        }

        device_vector<float> dA(a_elems);
        device_vector<float> dC(c_elems * batch_count);
        ASSERT_EQ(dA.memcheck(), hipSuccess);
        ASSERT_EQ(dC.memcheck(), hipSuccess);
        ASSERT_EQ(hipMemcpy((float*)dA, h_A.data(), a_elems * sizeof(float), hipMemcpyHostToDevice),
                  hipSuccess);
        ASSERT_EQ(hipMemcpy((float*)dC,
                            h_C.data(),
                            c_elems * batch_count * sizeof(float),
                            hipMemcpyHostToDevice),
                  hipSuccess);

        ASSERT_EQ(rocblas_ssyrk_strided_batched(handle,
                                                rocblas_fill_lower,
                                                rocblas_operation_none,
                                                n,
                                                k,
                                                &alpha,
                                                (float*)dA,
                                                n,
                                                0, // stride_A=0: alias A
                                                &beta,
                                                (float*)dC,
                                                n,
                                                rocblas_stride(n) * n,
                                                batch_count),
                  rocblas_status_success);

        // Copy result back.
        std::vector<float> h_result(c_elems * batch_count);
        ASSERT_EQ(hipMemcpy(h_result.data(),
                            (float*)dC,
                            c_elems * batch_count * sizeof(float),
                            hipMemcpyDeviceToHost),
                  hipSuccess);

        // Expected values for lower fill (column-major 2x2):
        //   C[0,0] = k + 1 = 501 (diagonal)
        //   C[1,0] = k     = 500 (lower off-diagonal)
        //   C[0,1] = 0  -- upper triangle is not written by syrk (standard path),
        //                   or saved/restored around GEMM (workspace path).
        //   C[1,1] = k + 1 = 501 (diagonal)

        auto check_batch = [&](rocblas_int b) {
            const float* C_b = h_result.data() + b * c_elems;
            // Column-major: element (row, col) = C_b[row + col*n]
            EXPECT_FLOAT_EQ(C_b[0 + 0 * n], float(k + 1)) << "batch " << b << " C[0,0]";
            EXPECT_FLOAT_EQ(C_b[1 + 0 * n], float(k)) << "batch " << b << " C[1,0]";
            EXPECT_FLOAT_EQ(C_b[0 + 1 * n], 0.0f)
                << "batch " << b << " C[0,1] (upper triangle, should be restored)";
            EXPECT_FLOAT_EQ(C_b[1 + 1 * n], float(k + 1)) << "batch " << b << " C[1,1]";
        };

        // Spot-check batches: first (chunk 0), last (chunk 1), and a middle one.
        check_batch(0);
        check_batch(batch_count / 2);
        check_batch(batch_count - 1);
    }

    // Degenerate dimension: n=1 means tri(n)=0, no workspace used.
    // Verify no crash and correct scalar result.
    TEST(syrk_herk_chunked_workspace_correctness, ssyrk_strided_n1)
    {
        rocblas_local_handle handle;

        const rocblas_int n = 1, k = 500;
        const rocblas_int batch_count = 65536;

        // For n=1, workspace is 0 (tri(1)=0).  Syrk should still produce
        // correct scalar results: C = alpha*A*A^T + beta*C.
        // A is 1x500 = all-ones, so A*A^T = [k] = [500].
        // C_init = [1], alpha=1, beta=1 => C = 500 + 1 = 501.

        const float alpha = 1.0f;
        const float beta  = 1.0f;

        std::vector<float> h_A(size_t(k), 1.0f);
        std::vector<float> h_C(size_t(batch_count), 1.0f);

        device_vector<float> dA(size_t(k));
        device_vector<float> dC(size_t(batch_count));
        ASSERT_EQ(dA.memcheck(), hipSuccess);
        ASSERT_EQ(dC.memcheck(), hipSuccess);
        ASSERT_EQ(hipMemcpy((float*)dA, h_A.data(), k * sizeof(float), hipMemcpyHostToDevice),
                  hipSuccess);
        ASSERT_EQ(
            hipMemcpy((float*)dC, h_C.data(), batch_count * sizeof(float), hipMemcpyHostToDevice),
            hipSuccess);

        ASSERT_EQ(rocblas_ssyrk_strided_batched(handle,
                                                rocblas_fill_lower,
                                                rocblas_operation_none,
                                                n,
                                                k,
                                                &alpha,
                                                (float*)dA,
                                                n,
                                                0, // stride_A=0
                                                &beta,
                                                (float*)dC,
                                                n,
                                                1, // stride_C=1 (single scalar)
                                                batch_count),
                  rocblas_status_success);

        std::vector<float> h_result(size_t(batch_count));
        ASSERT_EQ(
            hipMemcpy(
                h_result.data(), (float*)dC, batch_count * sizeof(float), hipMemcpyDeviceToHost),
            hipSuccess);

        // Check first batch, last batch (second chunk), and a middle one.
        EXPECT_FLOAT_EQ(h_result[0], float(k + 1));
        EXPECT_FLOAT_EQ(h_result[batch_count / 2], float(k + 1));
        EXPECT_FLOAT_EQ(h_result[batch_count - 1], float(k + 1));
    }

} // namespace
