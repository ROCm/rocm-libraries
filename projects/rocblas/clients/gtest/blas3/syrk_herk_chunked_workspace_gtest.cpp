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
// reuses the same workspace slice, so only one chunk's worth of triangle slots
// must be live at once.
//
// Scope: int32 batched and strided-batched syrk/herk on gfx90a (910) and
// gfx942 when rocblas_use_only_gemm selects the workspace path (k >= 500,
// n below the per-arch ceiling).

#include "client_utility.hpp"
#include "device_vector.hpp"
#include "rocblas.hpp"
#include "rocblas_test.hpp"
#include <algorithm>
#include <cstring>
#include <vector>

namespace
{
    // tri(n) = n*(n-1)/2 -- off-diagonal element count per batch in workspace.
    static inline size_t tri_n(rocblas_int n)
    {
        return (size_t(n) * size_t(n - 1)) / 2;
    }

    // Expected workspace bytes under the chunked scheme.
    // chunk = min(batch_count, c_YZ_grid_launch_limit) = min(batch_count, 65535).
    template <typename T>
    static size_t chunked_workspace_bytes(rocblas_int n, rocblas_int batch_count)
    {
        constexpr rocblas_int limit = (1 << 16) - 1; // 65535
        rocblas_int           chunk = std::min(batch_count, limit);
        return tri_n(n) * sizeof(T) * size_t(chunk);
    }

    // True when the device is gfx90a (arch 910) or gfx942.
    // These are the only architectures where rocblas_use_only_gemm selects
    // the workspace path for syrk/herk.
    static bool is_gemm_only_arch(rocblas_handle handle)
    {
        int arch = rocblas_handle(handle)->getArch();
        return arch == 910 || arch == 942;
    }

    // Issue a workspace size query for ssyrk_strided_batched and return the
    // reported bytes.  Returns false and leaves *bytes unset on failure.
    static bool query_ssyrk_workspace(rocblas_handle handle,
                                      rocblas_int    n,
                                      rocblas_int    k,
                                      rocblas_int    lda,
                                      rocblas_int    ldc,
                                      rocblas_int    batch_count,
                                      size_t*        bytes)
    {
        const rocblas_fill      uplo    = rocblas_fill_lower;
        const rocblas_operation transA  = rocblas_operation_none;
        const float             alpha   = 1.0f;
        const float             beta    = 0.0f;
        const rocblas_stride    strideA = rocblas_stride(lda) * k;
        const rocblas_stride    strideC = rocblas_stride(ldc) * n;

        rocblas_status st = rocblas_start_device_memory_size_query(handle);
        EXPECT_EQ(st, rocblas_status_success);
        if(st != rocblas_status_success)
            return false;

        st = rocblas_ssyrk_strided_batched(handle,
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
        EXPECT_TRUE(st == rocblas_status_size_increased || st == rocblas_status_size_unchanged)
            << "size query returned " << rocblas_status_to_string(st);
        if(st != rocblas_status_size_increased && st != rocblas_status_size_unchanged)
            return false;

        st = rocblas_stop_device_memory_size_query(handle, bytes);
        EXPECT_EQ(st, rocblas_status_success);
        return st == rocblas_status_success;
    }

    // Issue a workspace size query for cherk_strided_batched and return the
    // reported bytes.  Returns false and leaves *bytes unset on failure.
    static bool query_cherk_workspace(rocblas_handle handle,
                                      rocblas_int    n,
                                      rocblas_int    k,
                                      rocblas_int    lda,
                                      rocblas_int    ldc,
                                      rocblas_int    batch_count,
                                      size_t*        bytes)
    {
        const rocblas_fill      uplo    = rocblas_fill_lower;
        const rocblas_operation transA  = rocblas_operation_none;
        const float             alpha   = 1.0f;
        const float             beta    = 0.0f;
        const rocblas_stride    strideA = rocblas_stride(lda) * k;
        const rocblas_stride    strideC = rocblas_stride(ldc) * n;

        rocblas_status st = rocblas_start_device_memory_size_query(handle);
        EXPECT_EQ(st, rocblas_status_success);
        if(st != rocblas_status_success)
            return false;

        st = rocblas_cherk_strided_batched(handle,
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
        EXPECT_TRUE(st == rocblas_status_size_increased || st == rocblas_status_size_unchanged)
            << "size query returned " << rocblas_status_to_string(st);
        if(st != rocblas_status_size_increased && st != rocblas_status_size_unchanged)
            return false;

        st = rocblas_stop_device_memory_size_query(handle, bytes);
        EXPECT_EQ(st, rocblas_status_success);
        return st == rocblas_status_success;
    }

    // -----------------------------------------------------------------------
    // Workspace size query tests (syrk_herk_chunked_workspace_size suite)
    //
    // Each test verifies that the workspace reported by the size-query API
    // satisfies the chunked bound: it must be at least
    // min(batch_count, 65535) * tri(n) * sizeof(T) and, when
    // batch_count > 65535, strictly less than batch_count * tri(n) * sizeof(T)
    // (i.e. the old un-capped formula is no longer used).
    // -----------------------------------------------------------------------

    // batch_count = 131070 (two full passes of 65535): workspace must be capped.
    TEST(syrk_herk_chunked_workspace_size, ssyrk_131070)
    {
        rocblas_local_handle handle;
        if(!is_gemm_only_arch(handle))
            GTEST_SKIP() << "gemm-only path not active on this arch";

        const rocblas_int n = 32, k = 500, batch_count = 131070;
        size_t            reported = 0;
        ASSERT_TRUE(query_ssyrk_workspace(handle, n, k, n, n, batch_count, &reported));

        // Must allocate at least chunked_workspace_bytes, but not the full
        // batch_count * tri(n) * sizeof(float) that the old code would use.
        EXPECT_GE(reported, chunked_workspace_bytes<float>(n, batch_count));
        EXPECT_LT(reported, tri_n(n) * sizeof(float) * size_t(batch_count))
            << "workspace not capped: old batch_count-proportional formula still in use";
    }

    // batch_count = 65536 (one element over the grid limit): first value where
    // the chunk cap reduces workspace.
    TEST(syrk_herk_chunked_workspace_size, ssyrk_65536)
    {
        rocblas_local_handle handle;
        if(!is_gemm_only_arch(handle))
            GTEST_SKIP() << "gemm-only path not active on this arch";

        const rocblas_int n = 32, k = 500, batch_count = 65536;
        size_t            reported = 0;
        ASSERT_TRUE(query_ssyrk_workspace(handle, n, k, n, n, batch_count, &reported));

        EXPECT_GE(reported, chunked_workspace_bytes<float>(n, batch_count));
        EXPECT_LT(reported, tri_n(n) * sizeof(float) * size_t(batch_count))
            << "workspace not capped at batch_count=65536";
    }

    // batch_count = 65535 (exactly at the grid limit): no capping needed.
    TEST(syrk_herk_chunked_workspace_size, ssyrk_65535)
    {
        rocblas_local_handle handle;
        if(!is_gemm_only_arch(handle))
            GTEST_SKIP() << "gemm-only path not active on this arch";

        const rocblas_int n = 32, k = 500, batch_count = 65535;
        size_t            reported = 0;
        ASSERT_TRUE(query_ssyrk_workspace(handle, n, k, n, n, batch_count, &reported));

        EXPECT_GE(reported, chunked_workspace_bytes<float>(n, batch_count));
    }

    // batch_count = 100 (well below the limit): behaviour unchanged.
    TEST(syrk_herk_chunked_workspace_size, ssyrk_100)
    {
        rocblas_local_handle handle;
        if(!is_gemm_only_arch(handle))
            GTEST_SKIP() << "gemm-only path not active on this arch";

        const rocblas_int n = 32, k = 500, batch_count = 100;
        size_t            reported = 0;
        ASSERT_TRUE(query_ssyrk_workspace(handle, n, k, n, n, batch_count, &reported));

        EXPECT_GE(reported, chunked_workspace_bytes<float>(n, batch_count));
    }

    TEST(syrk_herk_chunked_workspace_size, cherk_131070)
    {
        rocblas_local_handle handle;
        if(!is_gemm_only_arch(handle))
            GTEST_SKIP() << "gemm-only path not active on this arch";

        const rocblas_int n = 32, k = 500, batch_count = 131070;
        size_t            reported = 0;
        ASSERT_TRUE(query_cherk_workspace(handle, n, k, n, n, batch_count, &reported));

        EXPECT_GE(reported, chunked_workspace_bytes<rocblas_float_complex>(n, batch_count));
        EXPECT_LT(reported,
                  tri_n(n) * sizeof(rocblas_float_complex) * size_t(batch_count))
            << "herk workspace not capped";
    }

    TEST(syrk_herk_chunked_workspace_size, cherk_65536)
    {
        rocblas_local_handle handle;
        if(!is_gemm_only_arch(handle))
            GTEST_SKIP() << "gemm-only path not active on this arch";

        const rocblas_int n = 32, k = 500, batch_count = 65536;
        size_t            reported = 0;
        ASSERT_TRUE(query_cherk_workspace(handle, n, k, n, n, batch_count, &reported));

        EXPECT_GE(reported, chunked_workspace_bytes<rocblas_float_complex>(n, batch_count));
        EXPECT_LT(reported,
                  tri_n(n) * sizeof(rocblas_float_complex) * size_t(batch_count))
            << "herk workspace not capped at batch_count=65536";
    }

    TEST(syrk_herk_chunked_workspace_size, cherk_65535)
    {
        rocblas_local_handle handle;
        if(!is_gemm_only_arch(handle))
            GTEST_SKIP() << "gemm-only path not active on this arch";

        const rocblas_int n = 32, k = 500, batch_count = 65535;
        size_t            reported = 0;
        ASSERT_TRUE(query_cherk_workspace(handle, n, k, n, n, batch_count, &reported));

        EXPECT_GE(reported, chunked_workspace_bytes<rocblas_float_complex>(n, batch_count));
    }

    TEST(syrk_herk_chunked_workspace_size, cherk_100)
    {
        rocblas_local_handle handle;
        if(!is_gemm_only_arch(handle))
            GTEST_SKIP() << "gemm-only path not active on this arch";

        const rocblas_int n = 32, k = 500, batch_count = 100;
        size_t            reported = 0;
        ASSERT_TRUE(query_cherk_workspace(handle, n, k, n, n, batch_count, &reported));

        EXPECT_GE(reported, chunked_workspace_bytes<rocblas_float_complex>(n, batch_count));
    }

    // -----------------------------------------------------------------------
    // Canary / correctness tests (syrk_herk_chunked_workspace_correctness)
    //
    // Allocate an exact-sized workspace matching the new chunked formula and
    // append a contiguous guard region filled with a sentinel byte.  If the
    // kernel writes past the workspace boundary the guard bytes change and the
    // test fails.
    //
    // batch_count = 131070: two saturated grid-z passes; the strongest
    // reproducer for the old over-allocation bug.
    // batch_count = 65539: boundary case (one element into the second pass).
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

    static void expect_canary_clean(rocblas_handle handle,
                                    void*          base,
                                    size_t         workspace_bytes)
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
            << differing << " guard byte(s) differ; first at +" << first
            << " byte(s) past the " << workspace_bytes << "-byte workspace";
    }

    // Runs ssyrk_strided_batched with exact chunked workspace + guard.
    static void run_ssyrk_canary(rocblas_int n, rocblas_int k, rocblas_int batch_count)
    {
        rocblas_local_handle handle;
        if(!is_gemm_only_arch(handle))
        {
            GTEST_SKIP() << "gemm-only path not active on this arch";
            return;
        }

        const size_t ws_bytes = chunked_workspace_bytes<float>(n, batch_count);
        // Workspace size query must return at least ws_bytes; skip if the path
        // is not live (returned 0 means rocblas_use_only_gemm returned false).
        size_t queried = 0;
        ASSERT_TRUE(query_ssyrk_workspace(handle, n, k, n, n, batch_count, &queried));
        if(queried == 0)
        {
            GTEST_SKIP() << "size query returned 0; gemm-only path not selected";
            return;
        }

        // Allocate workspace + guard.
        void* d_ws = nullptr;
        ASSERT_EQ((hipMalloc)(&d_ws, ws_bytes + c_canary_bytes), hipSuccess);
        ASSERT_TRUE(fill_canary(d_ws, ws_bytes));

        ASSERT_EQ(rocblas_set_workspace(handle, d_ws, ws_bytes), rocblas_status_success);

        // Strided A aliased across all batches (stride_A = 0).
        const size_t a_elems = size_t(n) * size_t(k); // single A matrix
        const size_t c_elems = size_t(n) * size_t(n) * size_t(batch_count);

        device_vector<float> dA(a_elems);
        device_vector<float> dC(c_elems);
        ASSERT_EQ(dA.memcheck(), hipSuccess);
        ASSERT_EQ(dC.memcheck(), hipSuccess);
        ASSERT_EQ(hipMemset((float*)dA, 0, a_elems * sizeof(float)), hipSuccess);
        ASSERT_EQ(hipMemset((float*)dC, 0, c_elems * sizeof(float)), hipSuccess);

        const float alpha = 1.0f;
        const float beta  = 1.0f;

        // stride_A = 0 aliases every batch to the same A (read-only access).
        ASSERT_EQ(rocblas_ssyrk_strided_batched(handle,
                                                 rocblas_fill_lower,
                                                 rocblas_operation_none,
                                                 n,
                                                 k,
                                                 &alpha,
                                                 (float*)dA,
                                                 n,
                                                 0,
                                                 &beta,
                                                 (float*)dC,
                                                 n,
                                                 rocblas_stride(n) * n,
                                                 batch_count),
                  rocblas_status_success);

        expect_canary_clean(handle, d_ws, ws_bytes);

        ASSERT_EQ(rocblas_set_workspace(handle, nullptr, 0), rocblas_status_success);
        ASSERT_EQ((hipFree)(d_ws), hipSuccess);
    }

    // Runs cherk_strided_batched with exact chunked workspace + guard.
    static void run_cherk_canary(rocblas_int n, rocblas_int k, rocblas_int batch_count)
    {
        rocblas_local_handle handle;
        if(!is_gemm_only_arch(handle))
        {
            GTEST_SKIP() << "gemm-only path not active on this arch";
            return;
        }

        const size_t ws_bytes = chunked_workspace_bytes<rocblas_float_complex>(n, batch_count);
        size_t       queried  = 0;
        ASSERT_TRUE(query_cherk_workspace(handle, n, k, n, n, batch_count, &queried));
        if(queried == 0)
        {
            GTEST_SKIP() << "size query returned 0; gemm-only path not selected";
            return;
        }

        void* d_ws = nullptr;
        ASSERT_EQ((hipMalloc)(&d_ws, ws_bytes + c_canary_bytes), hipSuccess);
        ASSERT_TRUE(fill_canary(d_ws, ws_bytes));

        ASSERT_EQ(rocblas_set_workspace(handle, d_ws, ws_bytes), rocblas_status_success);

        const size_t a_elems = size_t(n) * size_t(k);
        const size_t c_elems = size_t(n) * size_t(n) * size_t(batch_count);

        device_vector<rocblas_float_complex> dA(a_elems);
        device_vector<rocblas_float_complex> dC(c_elems);
        ASSERT_EQ(dA.memcheck(), hipSuccess);
        ASSERT_EQ(dC.memcheck(), hipSuccess);
        ASSERT_EQ(hipMemset((rocblas_float_complex*)dA,
                            0,
                            a_elems * sizeof(rocblas_float_complex)),
                  hipSuccess);
        ASSERT_EQ(hipMemset((rocblas_float_complex*)dC,
                            0,
                            c_elems * sizeof(rocblas_float_complex)),
                  hipSuccess);

        const float alpha = 1.0f;
        const float beta  = 1.0f;

        ASSERT_EQ(rocblas_cherk_strided_batched(handle,
                                                 rocblas_fill_lower,
                                                 rocblas_operation_none,
                                                 n,
                                                 k,
                                                 &alpha,
                                                 (rocblas_float_complex*)dA,
                                                 n,
                                                 0,
                                                 &beta,
                                                 (rocblas_float_complex*)dC,
                                                 n,
                                                 rocblas_stride(n) * n,
                                                 batch_count),
                  rocblas_status_success);

        expect_canary_clean(handle, d_ws, ws_bytes);

        ASSERT_EQ(rocblas_set_workspace(handle, nullptr, 0), rocblas_status_success);
        ASSERT_EQ((hipFree)(d_ws), hipSuccess);
    }

    // Two full grid-z passes (131070 = 2 * 65535).
    // If the chunked workspace is too small the guard bytes will be overwritten.
    TEST(syrk_herk_chunked_workspace_correctness, ssyrk_strided_batched_131070)
    {
        run_ssyrk_canary(2, 500, 131070);
    }

    TEST(syrk_herk_chunked_workspace_correctness, cherk_strided_batched_131070)
    {
        run_cherk_canary(2, 500, 131070);
    }

    // One element into the second pass (65539 = 65535 + 4).
    TEST(syrk_herk_chunked_workspace_correctness, ssyrk_strided_batched_65539)
    {
        run_ssyrk_canary(2, 500, 65539);
    }

    TEST(syrk_herk_chunked_workspace_correctness, cherk_strided_batched_65539)
    {
        run_cherk_canary(2, 500, 65539);
    }

} // namespace
