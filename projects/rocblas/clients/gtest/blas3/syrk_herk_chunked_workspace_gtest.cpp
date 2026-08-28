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
// Scope: int32 strided-batched and batched (pointer-array) syrk/herk on
// gfx90a (910) and gfx942 when rocblas_use_only_gemm selects the workspace
// path (k >= 500, n below the per-arch ceiling).

#include "client_utility.hpp"
#include "device_batch_vector.hpp"
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
        size_t                chunk = size_t(std::min(batch_count, limit));
        return tri_n(n) * sizeof(T) * chunk;
    }

    // True when the device is gfx90a (arch 910) or gfx942.
    // These are the only architectures where rocblas_use_only_gemm selects
    // the workspace path for syrk/herk.
    static bool is_gemm_only_arch(rocblas_handle handle)
    {
        int arch = rocblas_handle(handle)->getArch();
        return arch == 910 || arch == 942;
    }

    // Skip helper that emits a visible ADD_FAILURE note on unsupported hardware
    // when running in a context where it matters (non-CI), and always sets
    // skip so the test is not counted as a failure in normal usage.
    // The RecordProperty call ensures a skipped arch is logged in CI XML.
    #define SKIP_IF_NOT_GEMM_ONLY_ARCH(handle)                                       \
        do                                                                            \
        {                                                                             \
            if(!is_gemm_only_arch(handle))                                           \
            {                                                                        \
                ::testing::Test::RecordProperty(                                     \
                    "SkipReason", "gemm-only workspace path not active on this arch"); \
                GTEST_SKIP() << "gemm-only workspace path not active on this arch "  \
                             << "(requires gfx90a/gfx942 with k>=500)";             \
            }                                                                        \
        } while(0)

    // Issue a workspace size query for ssyrk_strided_batched and return the
    // reported bytes.  Returns false and leaves *bytes unset on failure.
    static bool query_ssyrk_strided_workspace(rocblas_handle handle,
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

    // Issue a workspace size query for ssyrk_batched (pointer-array) and return
    // the reported bytes.  Returns false and leaves *bytes unset on failure.
    static bool query_ssyrk_batched_workspace(rocblas_handle handle,
                                              rocblas_int    n,
                                              rocblas_int    k,
                                              rocblas_int    lda,
                                              rocblas_int    ldc,
                                              rocblas_int    batch_count,
                                              size_t*        bytes)
    {
        const rocblas_fill      uplo   = rocblas_fill_lower;
        const rocblas_operation transA = rocblas_operation_none;
        const float             alpha  = 1.0f;
        const float             beta   = 0.0f;

        rocblas_status st = rocblas_start_device_memory_size_query(handle);
        EXPECT_EQ(st, rocblas_status_success);
        if(st != rocblas_status_success)
            return false;

        st = rocblas_ssyrk_batched(handle,
                                   uplo,
                                   transA,
                                   n,
                                   k,
                                   &alpha,
                                   nullptr,
                                   lda,
                                   &beta,
                                   nullptr,
                                   ldc,
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
    static bool query_cherk_strided_workspace(rocblas_handle handle,
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

    // Issue a workspace size query for cherk_batched (pointer-array) and return
    // the reported bytes.  Returns false and leaves *bytes unset on failure.
    static bool query_cherk_batched_workspace(rocblas_handle handle,
                                              rocblas_int    n,
                                              rocblas_int    k,
                                              rocblas_int    lda,
                                              rocblas_int    ldc,
                                              rocblas_int    batch_count,
                                              size_t*        bytes)
    {
        const rocblas_fill      uplo   = rocblas_fill_lower;
        const rocblas_operation transA = rocblas_operation_none;
        const float             alpha  = 1.0f;
        const float             beta   = 0.0f;

        rocblas_status st = rocblas_start_device_memory_size_query(handle);
        EXPECT_EQ(st, rocblas_status_success);
        if(st != rocblas_status_success)
            return false;

        st = rocblas_cherk_batched(handle,
                                   uplo,
                                   transA,
                                   n,
                                   k,
                                   &alpha,
                                   nullptr,
                                   lda,
                                   &beta,
                                   nullptr,
                                   ldc,
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
    // For batch_count > 65535: workspace must equal the chunked bound (65535
    // slots) and be strictly less than the old un-capped formula.
    // For batch_count <= 65535: workspace must equal batch_count * tri(n) *
    // sizeof(T) (no change in behaviour, just a regression guard).
    // -----------------------------------------------------------------------

    // batch_count = 131070 (two full passes of 65535): workspace must be capped.
    TEST(syrk_herk_chunked_workspace_size, ssyrk_strided_131070)
    {
        rocblas_local_handle handle;
        SKIP_IF_NOT_GEMM_ONLY_ARCH(handle);

        const rocblas_int n = 32, k = 500, batch_count = 131070;
        size_t            reported = 0;
        ASSERT_TRUE(query_ssyrk_strided_workspace(handle, n, k, n, n, batch_count, &reported));

        const size_t chunked = chunked_workspace_bytes<float>(n, batch_count);
        EXPECT_EQ(reported, chunked)
            << "expected exactly " << chunked << " bytes (chunked formula)";
        EXPECT_LT(reported, tri_n(n) * sizeof(float) * size_t(batch_count))
            << "workspace not capped: old batch_count-proportional formula still in use";
    }

    // batch_count = 65536 (one element over the grid limit).
    TEST(syrk_herk_chunked_workspace_size, ssyrk_strided_65536)
    {
        rocblas_local_handle handle;
        SKIP_IF_NOT_GEMM_ONLY_ARCH(handle);

        const rocblas_int n = 32, k = 500, batch_count = 65536;
        size_t            reported = 0;
        ASSERT_TRUE(query_ssyrk_strided_workspace(handle, n, k, n, n, batch_count, &reported));

        const size_t chunked = chunked_workspace_bytes<float>(n, batch_count);
        EXPECT_EQ(reported, chunked)
            << "expected exactly " << chunked << " bytes (chunked formula)";
        EXPECT_LT(reported, tri_n(n) * sizeof(float) * size_t(batch_count))
            << "workspace not capped at batch_count=65536";
    }

    // batch_count = 65535 (exactly at the grid limit): chunk == batch_count,
    // so reported must equal batch_count * tri(n) * sizeof(T).
    TEST(syrk_herk_chunked_workspace_size, ssyrk_strided_65535)
    {
        rocblas_local_handle handle;
        SKIP_IF_NOT_GEMM_ONLY_ARCH(handle);

        const rocblas_int n = 32, k = 500, batch_count = 65535;
        size_t            reported = 0;
        ASSERT_TRUE(query_ssyrk_strided_workspace(handle, n, k, n, n, batch_count, &reported));

        const size_t expected = tri_n(n) * sizeof(float) * size_t(batch_count);
        EXPECT_EQ(reported, expected)
            << "at batch_count=65535 workspace should equal full batch_count * tri(n) * sizeof(T)";
    }

    // batch_count = 100 (well below the limit): behaviour unchanged from pre-fix.
    TEST(syrk_herk_chunked_workspace_size, ssyrk_strided_100)
    {
        rocblas_local_handle handle;
        SKIP_IF_NOT_GEMM_ONLY_ARCH(handle);

        const rocblas_int n = 32, k = 500, batch_count = 100;
        size_t            reported = 0;
        ASSERT_TRUE(query_ssyrk_strided_workspace(handle, n, k, n, n, batch_count, &reported));

        const size_t expected = tri_n(n) * sizeof(float) * size_t(batch_count);
        EXPECT_EQ(reported, expected);
    }

    // BATCHED (pointer-array) variants: the workspace formula is shared between
    // strided-batched and batched, so the same caps apply.
    TEST(syrk_herk_chunked_workspace_size, ssyrk_batched_131070)
    {
        rocblas_local_handle handle;
        SKIP_IF_NOT_GEMM_ONLY_ARCH(handle);

        const rocblas_int n = 32, k = 500, batch_count = 131070;
        size_t            reported = 0;
        ASSERT_TRUE(query_ssyrk_batched_workspace(handle, n, k, n, n, batch_count, &reported));

        const size_t chunked = chunked_workspace_bytes<float>(n, batch_count);
        EXPECT_EQ(reported, chunked);
        EXPECT_LT(reported, tri_n(n) * sizeof(float) * size_t(batch_count))
            << "batched workspace not capped";
    }

    TEST(syrk_herk_chunked_workspace_size, ssyrk_batched_65535)
    {
        rocblas_local_handle handle;
        SKIP_IF_NOT_GEMM_ONLY_ARCH(handle);

        const rocblas_int n = 32, k = 500, batch_count = 65535;
        size_t            reported = 0;
        ASSERT_TRUE(query_ssyrk_batched_workspace(handle, n, k, n, n, batch_count, &reported));

        const size_t expected = tri_n(n) * sizeof(float) * size_t(batch_count);
        EXPECT_EQ(reported, expected);
    }

    TEST(syrk_herk_chunked_workspace_size, cherk_strided_131070)
    {
        rocblas_local_handle handle;
        SKIP_IF_NOT_GEMM_ONLY_ARCH(handle);

        const rocblas_int n = 32, k = 500, batch_count = 131070;
        size_t            reported = 0;
        ASSERT_TRUE(query_cherk_strided_workspace(handle, n, k, n, n, batch_count, &reported));

        const size_t chunked = chunked_workspace_bytes<rocblas_float_complex>(n, batch_count);
        EXPECT_EQ(reported, chunked);
        EXPECT_LT(reported, tri_n(n) * sizeof(rocblas_float_complex) * size_t(batch_count))
            << "herk workspace not capped";
    }

    TEST(syrk_herk_chunked_workspace_size, cherk_strided_65536)
    {
        rocblas_local_handle handle;
        SKIP_IF_NOT_GEMM_ONLY_ARCH(handle);

        const rocblas_int n = 32, k = 500, batch_count = 65536;
        size_t            reported = 0;
        ASSERT_TRUE(query_cherk_strided_workspace(handle, n, k, n, n, batch_count, &reported));

        const size_t chunked = chunked_workspace_bytes<rocblas_float_complex>(n, batch_count);
        EXPECT_EQ(reported, chunked);
        EXPECT_LT(reported, tri_n(n) * sizeof(rocblas_float_complex) * size_t(batch_count))
            << "herk workspace not capped at batch_count=65536";
    }

    TEST(syrk_herk_chunked_workspace_size, cherk_strided_65535)
    {
        rocblas_local_handle handle;
        SKIP_IF_NOT_GEMM_ONLY_ARCH(handle);

        const rocblas_int n = 32, k = 500, batch_count = 65535;
        size_t            reported = 0;
        ASSERT_TRUE(query_cherk_strided_workspace(handle, n, k, n, n, batch_count, &reported));

        const size_t expected = tri_n(n) * sizeof(rocblas_float_complex) * size_t(batch_count);
        EXPECT_EQ(reported, expected);
    }

    TEST(syrk_herk_chunked_workspace_size, cherk_strided_100)
    {
        rocblas_local_handle handle;
        SKIP_IF_NOT_GEMM_ONLY_ARCH(handle);

        const rocblas_int n = 32, k = 500, batch_count = 100;
        size_t            reported = 0;
        ASSERT_TRUE(query_cherk_strided_workspace(handle, n, k, n, n, batch_count, &reported));

        const size_t expected = tri_n(n) * sizeof(rocblas_float_complex) * size_t(batch_count);
        EXPECT_EQ(reported, expected);
    }

    TEST(syrk_herk_chunked_workspace_size, cherk_batched_131070)
    {
        rocblas_local_handle handle;
        SKIP_IF_NOT_GEMM_ONLY_ARCH(handle);

        const rocblas_int n = 32, k = 500, batch_count = 131070;
        size_t            reported = 0;
        ASSERT_TRUE(query_cherk_batched_workspace(handle, n, k, n, n, batch_count, &reported));

        const size_t chunked = chunked_workspace_bytes<rocblas_float_complex>(n, batch_count);
        EXPECT_EQ(reported, chunked);
        EXPECT_LT(reported, tri_n(n) * sizeof(rocblas_float_complex) * size_t(batch_count))
            << "batched herk workspace not capped";
    }

    TEST(syrk_herk_chunked_workspace_size, cherk_batched_65535)
    {
        rocblas_local_handle handle;
        SKIP_IF_NOT_GEMM_ONLY_ARCH(handle);

        const rocblas_int n = 32, k = 500, batch_count = 65535;
        size_t            reported = 0;
        ASSERT_TRUE(query_cherk_batched_workspace(handle, n, k, n, n, batch_count, &reported));

        const size_t expected = tri_n(n) * sizeof(rocblas_float_complex) * size_t(batch_count);
        EXPECT_EQ(reported, expected);
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
    static void run_ssyrk_strided_canary(rocblas_int n, rocblas_int k, rocblas_int batch_count)
    {
        rocblas_local_handle handle;
        SKIP_IF_NOT_GEMM_ONLY_ARCH(handle);

        const size_t ws_bytes = chunked_workspace_bytes<float>(n, batch_count);
        // Skip if gemm-only path was not activated (workspace query returned 0).
        size_t queried = 0;
        ASSERT_TRUE(query_ssyrk_strided_workspace(handle, n, k, n, n, batch_count, &queried));
        if(queried == 0)
        {
            ::testing::Test::RecordProperty("SkipReason", "size query returned 0");
            GTEST_SKIP() << "size query returned 0; gemm-only path not selected "
                         << "(check k=" << k << " n=" << n << " thresholds)";
        }

        // Allocate workspace + guard region.
        void* d_ws = nullptr;
        ASSERT_EQ((hipMalloc)(&d_ws, ws_bytes + c_canary_bytes), hipSuccess);
        ASSERT_TRUE(fill_canary(d_ws, ws_bytes));

        ASSERT_EQ(rocblas_set_workspace(handle, d_ws, ws_bytes), rocblas_status_success);

        // A aliased across all batches via stride_A=0 (read-only, safe to share).
        const size_t a_elems = size_t(n) * size_t(k);
        const size_t c_elems = size_t(n) * size_t(n) * size_t(batch_count);

        device_vector<float> dA(a_elems);
        device_vector<float> dC(c_elems);
        ASSERT_EQ(dA.memcheck(), hipSuccess);
        ASSERT_EQ(dC.memcheck(), hipSuccess);
        ASSERT_EQ(hipMemset((float*)dA, 0, a_elems * sizeof(float)), hipSuccess);
        ASSERT_EQ(hipMemset((float*)dC, 0, c_elems * sizeof(float)), hipSuccess);

        const float alpha = 1.0f;
        const float beta  = 1.0f;

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

    // Runs ssyrk_batched (pointer-array) with exact chunked workspace + guard.
    static void run_ssyrk_batched_canary(rocblas_int n, rocblas_int k, rocblas_int batch_count)
    {
        rocblas_local_handle handle;
        SKIP_IF_NOT_GEMM_ONLY_ARCH(handle);

        const size_t ws_bytes = chunked_workspace_bytes<float>(n, batch_count);
        size_t       queried  = 0;
        ASSERT_TRUE(query_ssyrk_batched_workspace(handle, n, k, n, n, batch_count, &queried));
        if(queried == 0)
        {
            ::testing::Test::RecordProperty("SkipReason", "size query returned 0");
            GTEST_SKIP() << "size query returned 0; gemm-only path not selected";
        }

        void* d_ws = nullptr;
        ASSERT_EQ((hipMalloc)(&d_ws, ws_bytes + c_canary_bytes), hipSuccess);
        ASSERT_TRUE(fill_canary(d_ws, ws_bytes));

        ASSERT_EQ(rocblas_set_workspace(handle, d_ws, ws_bytes), rocblas_status_success);

        const size_t a_elems = size_t(n) * size_t(k);
        const size_t c_elems = size_t(n) * size_t(n);

        // device_batch_vector allocates batch_count independent device arrays.
        device_batch_vector<float> dA(a_elems, 1, batch_count);
        device_batch_vector<float> dC(c_elems, 1, batch_count);
        ASSERT_EQ(dA.memcheck(), hipSuccess);
        ASSERT_EQ(dC.memcheck(), hipSuccess);
        // Zero-initialise all batches.
        for(rocblas_int b = 0; b < batch_count; ++b)
        {
            ASSERT_EQ(hipMemset(dA[b], 0, a_elems * sizeof(float)), hipSuccess);
            ASSERT_EQ(hipMemset(dC[b], 0, c_elems * sizeof(float)), hipSuccess);
        }

        const float alpha = 1.0f;
        const float beta  = 1.0f;

        ASSERT_EQ(rocblas_ssyrk_batched(handle,
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
                                         batch_count),
                  rocblas_status_success);

        expect_canary_clean(handle, d_ws, ws_bytes);

        ASSERT_EQ(rocblas_set_workspace(handle, nullptr, 0), rocblas_status_success);
        ASSERT_EQ((hipFree)(d_ws), hipSuccess);
    }

    // Runs cherk_strided_batched with exact chunked workspace + guard.
    static void run_cherk_strided_canary(rocblas_int n, rocblas_int k, rocblas_int batch_count)
    {
        rocblas_local_handle handle;
        SKIP_IF_NOT_GEMM_ONLY_ARCH(handle);

        const size_t ws_bytes = chunked_workspace_bytes<rocblas_float_complex>(n, batch_count);
        size_t       queried  = 0;
        ASSERT_TRUE(query_cherk_strided_workspace(handle, n, k, n, n, batch_count, &queried));
        if(queried == 0)
        {
            ::testing::Test::RecordProperty("SkipReason", "size query returned 0");
            GTEST_SKIP() << "size query returned 0; gemm-only path not selected";
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

    // Runs cherk_batched (pointer-array) with exact chunked workspace + guard.
    static void run_cherk_batched_canary(rocblas_int n, rocblas_int k, rocblas_int batch_count)
    {
        rocblas_local_handle handle;
        SKIP_IF_NOT_GEMM_ONLY_ARCH(handle);

        const size_t ws_bytes = chunked_workspace_bytes<rocblas_float_complex>(n, batch_count);
        size_t       queried  = 0;
        ASSERT_TRUE(query_cherk_batched_workspace(handle, n, k, n, n, batch_count, &queried));
        if(queried == 0)
        {
            ::testing::Test::RecordProperty("SkipReason", "size query returned 0");
            GTEST_SKIP() << "size query returned 0; gemm-only path not selected";
        }

        void* d_ws = nullptr;
        ASSERT_EQ((hipMalloc)(&d_ws, ws_bytes + c_canary_bytes), hipSuccess);
        ASSERT_TRUE(fill_canary(d_ws, ws_bytes));

        ASSERT_EQ(rocblas_set_workspace(handle, d_ws, ws_bytes), rocblas_status_success);

        const size_t a_elems = size_t(n) * size_t(k);
        const size_t c_elems = size_t(n) * size_t(n);

        device_batch_vector<rocblas_float_complex> dA(a_elems, 1, batch_count);
        device_batch_vector<rocblas_float_complex> dC(c_elems, 1, batch_count);
        ASSERT_EQ(dA.memcheck(), hipSuccess);
        ASSERT_EQ(dC.memcheck(), hipSuccess);
        for(rocblas_int b = 0; b < batch_count; ++b)
        {
            ASSERT_EQ(hipMemset(dA[b], 0, a_elems * sizeof(rocblas_float_complex)), hipSuccess);
            ASSERT_EQ(hipMemset(dC[b], 0, c_elems * sizeof(rocblas_float_complex)), hipSuccess);
        }

        const float alpha = 1.0f;
        const float beta  = 1.0f;

        ASSERT_EQ(rocblas_cherk_batched(handle,
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
                                         batch_count),
                  rocblas_status_success);

        expect_canary_clean(handle, d_ws, ws_bytes);

        ASSERT_EQ(rocblas_set_workspace(handle, nullptr, 0), rocblas_status_success);
        ASSERT_EQ((hipFree)(d_ws), hipSuccess);
    }

    // -----------------------------------------------------------------------
    // Correctness / canary tests
    //
    // Two batch_count values exercised:
    //   131070 = 2 * 65535 : exactly two saturated grid-z passes.
    //   65539  = 65535 + 4 : one element into the second pass (boundary).
    // -----------------------------------------------------------------------

    // --- ssyrk strided-batched ---
    TEST(syrk_herk_chunked_workspace_correctness, ssyrk_strided_batched_131070)
    {
        run_ssyrk_strided_canary(2, 500, 131070);
    }

    TEST(syrk_herk_chunked_workspace_correctness, ssyrk_strided_batched_65539)
    {
        run_ssyrk_strided_canary(2, 500, 65539);
    }

    // --- ssyrk batched (pointer-array) ---
    TEST(syrk_herk_chunked_workspace_correctness, ssyrk_batched_131070)
    {
        run_ssyrk_batched_canary(2, 500, 131070);
    }

    TEST(syrk_herk_chunked_workspace_correctness, ssyrk_batched_65539)
    {
        run_ssyrk_batched_canary(2, 500, 65539);
    }

    // --- cherk strided-batched ---
    TEST(syrk_herk_chunked_workspace_correctness, cherk_strided_batched_131070)
    {
        run_cherk_strided_canary(2, 500, 131070);
    }

    TEST(syrk_herk_chunked_workspace_correctness, cherk_strided_batched_65539)
    {
        run_cherk_strided_canary(2, 500, 65539);
    }

    // --- cherk batched (pointer-array) ---
    TEST(syrk_herk_chunked_workspace_correctness, cherk_batched_131070)
    {
        run_cherk_batched_canary(2, 500, 131070);
    }

    TEST(syrk_herk_chunked_workspace_correctness, cherk_batched_65539)
    {
        run_cherk_batched_canary(2, 500, 65539);
    }

} // namespace
