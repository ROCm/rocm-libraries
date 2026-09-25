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
// Scope: int32 strided-batched and batched (pointer-array) syrk/herk, over
// every type the two operations instantiate -- s/d/c/z syrk and c/z herk.
// The workspace formula scales with sizeof(T) and the chunk loop is shared by
// all six, so each is dispatched through the same templated battery.
//
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
#include "host_batch_vector.hpp"
#include "rocblas.hpp"
#include "rocblas_test.hpp"
#include <algorithm>
#include <type_traits>
#include <vector>

namespace
{
    // -----------------------------------------------------------------------
    // Type dispatch
    //
    // syrk takes alpha/beta of type T for every T, including complex.  herk
    // takes real alpha/beta even though C is complex.  Selecting on the
    // operation rather than on rocblas_is_complex<T> is what lets csyrk/zsyrk
    // share this battery with cherk/zherk.
    // -----------------------------------------------------------------------

    enum class op_kind
    {
        syrk,
        herk
    };

    template <typename T>
    struct real_of
    {
        using type = T;
    };
    template <>
    struct real_of<rocblas_float_complex>
    {
        using type = float;
    };
    template <>
    struct real_of<rocblas_double_complex>
    {
        using type = double;
    };

    template <typename T>
    using real_t = typename real_of<T>::type;

    // Scalar type of alpha/beta for operation K on element type T.
    template <typename T, op_kind K>
    using scalar_t = std::conditional_t<K == op_kind::herk, real_t<T>, T>;

    template <typename T>
    static T make_val(double re, double im)
    {
        if constexpr(rocblas_is_complex<T>)
            return T(real_t<T>(re), real_t<T>(im));
        else
            return T(re);
    }

    template <typename T>
    static real_t<T> re_of(const T& v)
    {
        if constexpr(rocblas_is_complex<T>)
            return std::real(v);
        else
            return v;
    }

    template <typename T>
    static real_t<T> im_of(const T& v)
    {
        if constexpr(rocblas_is_complex<T>)
            return std::imag(v);
        else
            return real_t<T>(0);
    }

    // -----------------------------------------------------------------------
    // Workspace size model
    // -----------------------------------------------------------------------

    // tri(n) = n*(n-1)/2 -- off-diagonal element count per batch in workspace.
    static size_t tri_n(rocblas_int n)
    {
        return (size_t(n) * size_t(n - 1)) / 2;
    }

    // Batches per chunk. Must match c_i64_grid_YZ_chunk, which is library-internal
    // and so cannot be referenced from a client test. The value is the 16-bit grid
    // ceiling rounded down to a multiple of 16, which is also the stride
    // rocblas_internal_gemm_64 uses for its own batch loop.
    constexpr rocblas_int limit = ((1 << 16) - 1) & ~0xf; // 65520

    // Expected workspace bytes under the chunked scheme.
    // Mirrors the production formula in rocblas_syrk_herk.hpp.
    template <typename T>
    static size_t chunked_workspace_bytes(rocblas_int n, rocblas_int batch_count)
    {
        return tri_n(n) * sizeof(T) * size_t(std::min(batch_count, limit));
    }

    // A size query does not report the raw request: handle.hpp rounds it up to
    // a 64-byte chunk. Compare against the rounded value, or every case whose
    // raw size is not already a multiple of 64 fails.
    static size_t rounded(size_t bytes)
    {
        constexpr size_t chunk = 64;
        return ((bytes + chunk - 1) / chunk) * chunk;
    }

    // -----------------------------------------------------------------------
    // Workspace size query helpers, one per API shape.
    // Passing null A/C is legal during a size query: no memory is touched.
    // -----------------------------------------------------------------------

    template <typename T, op_kind K, typename ApiFunc>
    static bool query_strided_workspace(rocblas_handle handle,
                                        ApiFunc        api,
                                        rocblas_int    n,
                                        rocblas_int    k,
                                        rocblas_int    lda,
                                        rocblas_int    ldc,
                                        rocblas_int    batch_count,
                                        size_t*        bytes)
    {
        using S = scalar_t<T, K>;

        rocblas_status st = rocblas_start_device_memory_size_query(handle);
        EXPECT_EQ(st, rocblas_status_success);
        if(st != rocblas_status_success)
            return false;

        const S alpha = S(1);
        const S beta  = S(0);

        st = api(handle,
                 rocblas_fill_lower,
                 rocblas_operation_none,
                 n,
                 k,
                 &alpha,
                 nullptr,
                 lda,
                 rocblas_stride(lda) * k,
                 &beta,
                 nullptr,
                 ldc,
                 rocblas_stride(ldc) * n,
                 batch_count);

        EXPECT_TRUE(st == rocblas_status_size_increased || st == rocblas_status_size_unchanged)
            << "size query returned " << rocblas_status_to_string(st);
        if(st != rocblas_status_size_increased && st != rocblas_status_size_unchanged)
            return false;

        st = rocblas_stop_device_memory_size_query(handle, bytes);
        EXPECT_EQ(st, rocblas_status_success);
        return st == rocblas_status_success;
    }

    template <typename T, op_kind K, typename ApiFunc>
    static bool query_batched_workspace(rocblas_handle handle,
                                        ApiFunc        api,
                                        rocblas_int    n,
                                        rocblas_int    k,
                                        rocblas_int    lda,
                                        rocblas_int    ldc,
                                        rocblas_int    batch_count,
                                        size_t*        bytes)
    {
        using S = scalar_t<T, K>;

        rocblas_status st = rocblas_start_device_memory_size_query(handle);
        EXPECT_EQ(st, rocblas_status_success);
        if(st != rocblas_status_success)
            return false;

        const S alpha = S(1);
        const S beta  = S(0);

        st = api(handle,
                 rocblas_fill_lower,
                 rocblas_operation_none,
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
    // Canary guard
    //
    // Allocate an exact-sized workspace matching the chunked formula and append
    // a contiguous guard region filled with a sentinel byte.  If the kernel
    // writes past the workspace boundary the guard bytes change.
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

    // Owns a device array of batch_count pointers all aliasing one buffer.
    // A is read-only, so sharing it keeps the A allocation independent of
    // batch_count; a distinct A per batch would cost gigabytes at the
    // >65535 batch counts these tests need for 16-byte types.
    template <typename T>
    class aliased_ptr_array
    {
    public:
        aliased_ptr_array(T* base, rocblas_int batch_count)
        {
            std::vector<T*> host(size_t(batch_count), base);
            if((hipMalloc)(&m_device, sizeof(T*) * size_t(batch_count)) != hipSuccess)
            {
                m_device = nullptr;
                return;
            }
            if(hipMemcpy(
                   m_device, host.data(), sizeof(T*) * size_t(batch_count), hipMemcpyHostToDevice)
               != hipSuccess)
            {
                (void)(hipFree)(m_device);
                m_device = nullptr;
            }
        }

        ~aliased_ptr_array()
        {
            if(m_device)
                (void)(hipFree)(m_device);
        }

        aliased_ptr_array(const aliased_ptr_array&) = delete;
        aliased_ptr_array& operator=(const aliased_ptr_array&) = delete;

        bool valid() const
        {
            return m_device != nullptr;
        }
        T* const* ptr_on_device() const
        {
            return m_device;
        }

    private:
        T** m_device = nullptr;
    };

    // -----------------------------------------------------------------------
    // Shared test geometry.
    //
    // n is small so that C (n*n*batch_count) stays affordable at the >65535
    // batch counts; k is large enough to keep the gemm-only workspace path
    // selected on the architectures that have it.
    // -----------------------------------------------------------------------

    constexpr rocblas_int c_n = 2;
    constexpr rocblas_int c_k = 500;

    // -----------------------------------------------------------------------
    // Size-query battery: no device allocation, so it runs on every arch.
    // -----------------------------------------------------------------------

    template <typename T, op_kind K, typename StridedFn, typename BatchedFn>
    static void run_size_queries(StridedFn strided_api, BatchedFn batched_api)
    {
        rocblas_local_handle handle;
        const rocblas_int    n = 32, k = c_k;

        // Above the chunk limit the reported size must equal the chunked bound
        // and be strictly below the old per-batch formula.
        for(rocblas_int bc : {limit + 1, 65536, 131070})
        {
            size_t reported = 0;
            ASSERT_TRUE(
                (query_strided_workspace<T, K>(handle, strided_api, n, k, n, n, bc, &reported)))
                << "strided query failed at batch_count=" << bc;
            if(reported)
            {
                EXPECT_EQ(reported, rounded(chunked_workspace_bytes<T>(n, bc)))
                    << "batch_count=" << bc;
                EXPECT_LT(reported, tri_n(n) * sizeof(T) * size_t(bc))
                    << "workspace not capped at batch_count=" << bc;
            }

            reported = 0;
            ASSERT_TRUE(
                (query_batched_workspace<T, K>(handle, batched_api, n, k, n, n, bc, &reported)))
                << "batched query failed at batch_count=" << bc;
            if(reported)
                EXPECT_EQ(reported, rounded(chunked_workspace_bytes<T>(n, bc)))
                    << "batch_count=" << bc;
        }

        // At or below the limit the chunked formula degenerates to the
        // original per-batch size, so the fix must not change these.
        for(rocblas_int bc : {1, 100, limit})
        {
            size_t reported = 0;
            ASSERT_TRUE(
                (query_strided_workspace<T, K>(handle, strided_api, n, k, n, n, bc, &reported)))
                << "strided query failed at batch_count=" << bc;
            if(reported)
                EXPECT_EQ(reported, rounded(tri_n(n) * sizeof(T) * size_t(bc)))
                    << "batch_count=" << bc;
        }
    }

    // -----------------------------------------------------------------------
    // Canary battery: exact-sized workspace plus guard, over both API shapes
    // and both fill modes.
    // -----------------------------------------------------------------------

    template <typename T, op_kind K, typename ApiFunc>
    static void run_strided_canary(ApiFunc api, rocblas_int batch_count, rocblas_fill uplo)
    {
        using S = scalar_t<T, K>;

        rocblas_local_handle handle;
        const rocblas_int    n = c_n, k = c_k;

        size_t queried = 0;
        ASSERT_TRUE(
            (query_strided_workspace<T, K>(handle, api, n, k, n, n, batch_count, &queried)));

        const bool ws_active = queried > 0;

        // Without the workspace path, multi-chunk iteration is meaningless;
        // run a token batch rather than allocating for nothing.
        const rocblas_int run_bc = ws_active ? batch_count : std::min(batch_count, rocblas_int(2));

        void* d_ws = nullptr;
        if(ws_active)
        {
            EXPECT_EQ(queried, rounded(chunked_workspace_bytes<T>(n, batch_count)))
                << "queried workspace diverges from test formula";
            ASSERT_EQ((hipMalloc)(&d_ws, queried + c_canary_bytes), hipSuccess);
            ASSERT_TRUE(fill_canary(d_ws, queried));
            ASSERT_EQ(rocblas_set_workspace(handle, d_ws, queried), rocblas_status_success);
        }

        // A aliased across all batches via stride_A=0 (read-only, safe to share).
        device_vector<T> dA(size_t(n) * size_t(k));
        device_vector<T> dC(size_t(n) * size_t(n) * size_t(run_bc));
        ASSERT_EQ(dA.memcheck(), hipSuccess);
        ASSERT_EQ(dC.memcheck(), hipSuccess);
        ASSERT_EQ(hipMemset((T*)dA, 0, size_t(n) * size_t(k) * sizeof(T)), hipSuccess);
        ASSERT_EQ(hipMemset((T*)dC, 0, size_t(n) * size_t(n) * size_t(run_bc) * sizeof(T)),
                  hipSuccess);

        const S alpha = S(1);
        const S beta  = S(1);

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

        if(ws_active)
        {
            expect_canary_clean(handle, d_ws, queried);
            ASSERT_EQ(rocblas_set_workspace(handle, nullptr, 0), rocblas_status_success);
            ASSERT_EQ((hipFree)(d_ws), hipSuccess);
        }
    }

    template <typename T, op_kind K, typename ApiFunc>
    static void run_batched_canary(ApiFunc api, rocblas_int batch_count, rocblas_fill uplo)
    {
        using S = scalar_t<T, K>;

        rocblas_local_handle handle;
        const rocblas_int    n = c_n, k = c_k;

        size_t queried = 0;
        ASSERT_TRUE(
            (query_batched_workspace<T, K>(handle, api, n, k, n, n, batch_count, &queried)));

        const bool        ws_active = queried > 0;
        const rocblas_int run_bc = ws_active ? batch_count : std::min(batch_count, rocblas_int(2));

        void* d_ws = nullptr;
        if(ws_active)
        {
            EXPECT_EQ(queried, rounded(chunked_workspace_bytes<T>(n, batch_count)))
                << "queried workspace diverges from test formula";
            ASSERT_EQ((hipMalloc)(&d_ws, queried + c_canary_bytes), hipSuccess);
            ASSERT_TRUE(fill_canary(d_ws, queried));
            ASSERT_EQ(rocblas_set_workspace(handle, d_ws, queried), rocblas_status_success);
        }

        device_vector<T> dA(size_t(n) * size_t(k));
        ASSERT_EQ(dA.memcheck(), hipSuccess);
        ASSERT_EQ(hipMemset((T*)dA, 0, size_t(n) * size_t(k) * sizeof(T)), hipSuccess);

        aliased_ptr_array<T> dA_ptrs((T*)dA, run_bc);
        ASSERT_TRUE(dA_ptrs.valid()) << "failed to allocate A pointer array";

        device_batch_vector<T> dC(size_t(n) * size_t(n), 1, run_bc);
        ASSERT_EQ(dC.memcheck(), hipSuccess);
        ASSERT_EQ(hipMemset(dC[0], 0, size_t(n) * size_t(n) * size_t(run_bc) * sizeof(T)),
                  hipSuccess);

        const S alpha = S(1);
        const S beta  = S(1);

        ASSERT_EQ(api(handle,
                      uplo,
                      rocblas_operation_none,
                      n,
                      k,
                      &alpha,
                      dA_ptrs.ptr_on_device(),
                      n,
                      &beta,
                      dC.ptr_on_device(),
                      n,
                      run_bc),
                  rocblas_status_success);

        if(ws_active)
        {
            expect_canary_clean(handle, d_ws, queried);
            ASSERT_EQ(rocblas_set_workspace(handle, nullptr, 0), rocblas_status_success);
            ASSERT_EQ((hipFree)(d_ws), hipSuccess);
        }
    }

    template <typename T, op_kind K, typename StridedFn, typename BatchedFn>
    static void run_canaries(StridedFn strided_api, BatchedFn batched_api)
    {
        // 131070 = two full chunks, 65539 = one full chunk plus a short one,
        // 1 = the degenerate single-batch case.
        for(rocblas_int bc : {131070, 65539, 1})
        {
            run_strided_canary<T, K>(strided_api, bc, rocblas_fill_lower);
            run_batched_canary<T, K>(batched_api, bc, rocblas_fill_lower);
        }
        // Upper fill saves the opposite triangle.
        run_strided_canary<T, K>(strided_api, 131070, rocblas_fill_upper);
        run_batched_canary<T, K>(batched_api, 131070, rocblas_fill_upper);
    }

    // -----------------------------------------------------------------------
    // Numerical battery
    //
    // A is all ones, so alpha*A*A^T (or A*A^H) is the constant k in every
    // element.  Each batch starts with a distinct value in the upper triangle,
    // which a lower-fill syrk/herk must leave untouched: on the workspace path
    // it is saved to W_C before the GEMM overwrites all of C and restored
    // afterwards.  Zero-filled data would hide a chunk loop that failed to
    // advance the C pointer, because every batch would look alike; distinct
    // per-batch values make a wrong batch index observable.
    //
    // For herk the imaginary part of the diagonal is defined to be zero, so
    // the diagonal seeds are real; the upper triangle carries a non-zero
    // imaginary part for the complex types.
    // -----------------------------------------------------------------------

    template <typename T>
    static T expected_upper(rocblas_int b)
    {
        // Exactly representable in float and distinct across a chunk boundary.
        const double v = double(1 + (b % 4096));
        return make_val<T>(v, rocblas_is_complex<T> ? -v : 0.0);
    }

    template <typename T>
    static void expect_val_eq(const T& got, const T& want, const char* what, rocblas_int b)
    {
        EXPECT_EQ(re_of(got), re_of(want)) << what << " real part, batch " << b;
        EXPECT_EQ(im_of(got), im_of(want)) << what << " imaginary part, batch " << b;
    }

    // Fills one column-major n x n C block: real diagonal, zero lower
    // off-diagonal (the GEMM overwrites it), distinct upper off-diagonal.
    template <typename T>
    static void seed_c_block(T* C, rocblas_int n, rocblas_int b)
    {
        C[0 + 0 * n] = make_val<T>(1.0, 0.0);
        C[1 + 1 * n] = make_val<T>(1.0, 0.0);
        C[1 + 0 * n] = make_val<T>(0.0, 0.0);
        C[0 + 1 * n] = expected_upper<T>(b);
    }

    template <typename T>
    static void check_c_block(const T* C, rocblas_int n, rocblas_int k, rocblas_int b)
    {
        expect_val_eq(C[0 + 0 * n], make_val<T>(double(k) + 1.0, 0.0), "C[0,0]", b);
        expect_val_eq(C[1 + 0 * n], make_val<T>(double(k), 0.0), "C[1,0]", b);
        expect_val_eq(C[1 + 1 * n], make_val<T>(double(k) + 1.0, 0.0), "C[1,1]", b);
        expect_val_eq(C[0 + 1 * n], expected_upper<T>(b), "C[0,1] (upper triangle preserved)", b);
    }

    template <typename T, op_kind K, typename ApiFunc>
    static void run_strided_numerical(ApiFunc api, rocblas_int batch_count)
    {
        using S = scalar_t<T, K>;

        rocblas_local_handle handle;
        const rocblas_int    n = c_n, k = 64;

        const size_t a_elems = size_t(n) * size_t(k);
        const size_t c_elems = size_t(n) * size_t(n);

        std::vector<T> h_A(a_elems, make_val<T>(1.0, 0.0));
        std::vector<T> h_C(c_elems * size_t(batch_count));
        for(rocblas_int b = 0; b < batch_count; ++b)
            seed_c_block(h_C.data() + size_t(b) * c_elems, n, b);

        device_vector<T> dA(a_elems);
        device_vector<T> dC(c_elems * size_t(batch_count));
        ASSERT_EQ(dA.memcheck(), hipSuccess);
        ASSERT_EQ(dC.memcheck(), hipSuccess);
        ASSERT_EQ(hipMemcpy((T*)dA, h_A.data(), a_elems * sizeof(T), hipMemcpyHostToDevice),
                  hipSuccess);
        ASSERT_EQ(hipMemcpy((T*)dC,
                            h_C.data(),
                            c_elems * size_t(batch_count) * sizeof(T),
                            hipMemcpyHostToDevice),
                  hipSuccess);

        const S alpha = S(1);
        const S beta  = S(1);

        ASSERT_EQ(api(handle,
                      rocblas_fill_lower,
                      rocblas_operation_none,
                      n,
                      k,
                      &alpha,
                      (T*)dA,
                      n,
                      0, // stride_A = 0: alias A across batches
                      &beta,
                      (T*)dC,
                      n,
                      rocblas_stride(n) * n,
                      batch_count),
                  rocblas_status_success);

        std::vector<T> h_result(c_elems * size_t(batch_count));
        ASSERT_EQ(hipMemcpy(h_result.data(),
                            (T*)dC,
                            c_elems * size_t(batch_count) * sizeof(T),
                            hipMemcpyDeviceToHost),
                  hipSuccess);

        // Last batch of chunk 0, first and last of chunk 1.
        for(rocblas_int b : {0, limit - 1, limit, batch_count - 1})
            check_c_block(h_result.data() + size_t(b) * c_elems, n, k, b);
    }

    template <typename T, op_kind K, typename ApiFunc>
    static void run_batched_numerical(ApiFunc api, rocblas_int batch_count)
    {
        using S = scalar_t<T, K>;

        rocblas_local_handle handle;
        const rocblas_int    n = c_n, k = 64;

        const size_t a_elems = size_t(n) * size_t(k);
        const size_t c_elems = size_t(n) * size_t(n);

        std::vector<T> h_A(a_elems, make_val<T>(1.0, 0.0));

        host_batch_vector<T> h_C(c_elems, 1, batch_count);
        ASSERT_EQ(h_C.memcheck(), hipSuccess);
        for(rocblas_int b = 0; b < batch_count; ++b)
            seed_c_block(h_C[b], n, b);

        device_vector<T> dA(a_elems);
        ASSERT_EQ(dA.memcheck(), hipSuccess);
        ASSERT_EQ(hipMemcpy((T*)dA, h_A.data(), a_elems * sizeof(T), hipMemcpyHostToDevice),
                  hipSuccess);

        aliased_ptr_array<T> dA_ptrs((T*)dA, batch_count);
        ASSERT_TRUE(dA_ptrs.valid()) << "failed to allocate A pointer array";

        device_batch_vector<T> dC(c_elems, 1, batch_count);
        ASSERT_EQ(dC.memcheck(), hipSuccess);
        ASSERT_EQ(dC.transfer_from(h_C), hipSuccess);

        const S alpha = S(1);
        const S beta  = S(1);

        ASSERT_EQ(api(handle,
                      rocblas_fill_lower,
                      rocblas_operation_none,
                      n,
                      k,
                      &alpha,
                      dA_ptrs.ptr_on_device(),
                      n,
                      &beta,
                      dC.ptr_on_device(),
                      n,
                      batch_count),
                  rocblas_status_success);

        host_batch_vector<T> h_result(c_elems, 1, batch_count);
        ASSERT_EQ(h_result.memcheck(), hipSuccess);
        ASSERT_EQ(h_result.transfer_from(dC), hipSuccess);

        for(rocblas_int b : {0, limit - 1, limit, batch_count - 1})
            check_c_block(h_result[b], n, k, b);
    }

    template <typename T, op_kind K, typename StridedFn, typename BatchedFn>
    static void run_numerical(StridedFn strided_api, BatchedFn batched_api)
    {
        // Just above the chunk limit so the workspace path runs two chunks.
        constexpr rocblas_int bc = 65536;
        run_strided_numerical<T, K>(strided_api, bc);
        run_batched_numerical<T, K>(batched_api, bc);
    }

    // n=1 makes tri(n) zero, so no workspace is needed at any batch count.
    // Verify the chunk loop still terminates and the scalar result is right.
    template <typename T, op_kind K, typename ApiFunc>
    static void run_degenerate_n1(ApiFunc api)
    {
        using S = scalar_t<T, K>;

        rocblas_local_handle handle;
        const rocblas_int    n = 1, k = 64;
        const rocblas_int    batch_count = 65536;

        std::vector<T> h_A(size_t(k), make_val<T>(1.0, 0.0));
        std::vector<T> h_C(size_t(batch_count), make_val<T>(1.0, 0.0));

        // Braces, not parens: device_vector<T> dA(size_t(k)) declares a function.
        device_vector<T> dA{size_t(k)};
        device_vector<T> dC{size_t(batch_count)};
        ASSERT_EQ(dA.memcheck(), hipSuccess);
        ASSERT_EQ(dC.memcheck(), hipSuccess);
        ASSERT_EQ(hipMemcpy((T*)dA, h_A.data(), size_t(k) * sizeof(T), hipMemcpyHostToDevice),
                  hipSuccess);
        ASSERT_EQ(
            hipMemcpy((T*)dC, h_C.data(), size_t(batch_count) * sizeof(T), hipMemcpyHostToDevice),
            hipSuccess);

        const S alpha = S(1);
        const S beta  = S(1);

        ASSERT_EQ(api(handle,
                      rocblas_fill_lower,
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
                      1,
                      batch_count),
                  rocblas_status_success);

        std::vector<T> h_result(size_t(batch_count), T{});
        ASSERT_EQ(
            hipMemcpy(
                h_result.data(), (T*)dC, size_t(batch_count) * sizeof(T), hipMemcpyDeviceToHost),
            hipSuccess);

        const T want = make_val<T>(double(k) + 1.0, 0.0);
        for(rocblas_int b : {0, limit - 1, limit, batch_count - 1})
            expect_val_eq(h_result[b], want, "C[0,0]", b);
    }

    // -----------------------------------------------------------------------
    // Instantiate the battery for every type/operation syrk and herk provide.
    // -----------------------------------------------------------------------

#define SYRK_HERK_CHUNKED_TESTS(prefix, T, KIND, STRIDED_FN, BATCHED_FN)   \
    TEST(syrk_herk_chunked_workspace_size_pre_checkin, prefix)             \
    {                                                                      \
        run_size_queries<T, KIND>(STRIDED_FN, BATCHED_FN);                 \
    }                                                                      \
    TEST(syrk_herk_chunked_workspace_canary_pre_checkin, prefix)           \
    {                                                                      \
        run_canaries<T, KIND>(STRIDED_FN, BATCHED_FN);                     \
    }                                                                      \
    TEST(syrk_herk_chunked_workspace_correctness_pre_checkin, prefix)      \
    {                                                                      \
        run_numerical<T, KIND>(STRIDED_FN, BATCHED_FN);                    \
    }                                                                      \
    TEST(syrk_herk_chunked_workspace_correctness_pre_checkin, prefix##_n1) \
    {                                                                      \
        run_degenerate_n1<T, KIND>(STRIDED_FN);                            \
    }

    SYRK_HERK_CHUNKED_TESTS(
        ssyrk, float, op_kind::syrk, rocblas_ssyrk_strided_batched, rocblas_ssyrk_batched)
    SYRK_HERK_CHUNKED_TESTS(
        dsyrk, double, op_kind::syrk, rocblas_dsyrk_strided_batched, rocblas_dsyrk_batched)
    SYRK_HERK_CHUNKED_TESTS(csyrk,
                            rocblas_float_complex,
                            op_kind::syrk,
                            rocblas_csyrk_strided_batched,
                            rocblas_csyrk_batched)
    SYRK_HERK_CHUNKED_TESTS(zsyrk,
                            rocblas_double_complex,
                            op_kind::syrk,
                            rocblas_zsyrk_strided_batched,
                            rocblas_zsyrk_batched)
    SYRK_HERK_CHUNKED_TESTS(cherk,
                            rocblas_float_complex,
                            op_kind::herk,
                            rocblas_cherk_strided_batched,
                            rocblas_cherk_batched)
    SYRK_HERK_CHUNKED_TESTS(zherk,
                            rocblas_double_complex,
                            op_kind::herk,
                            rocblas_zherk_strided_batched,
                            rocblas_zherk_batched)

#undef SYRK_HERK_CHUNKED_TESTS

} // namespace
