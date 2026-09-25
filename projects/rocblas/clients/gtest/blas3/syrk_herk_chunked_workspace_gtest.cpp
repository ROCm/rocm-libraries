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
// The fix caps peak workspace at one chunk's worth of triangle slots rather
// than batch_count * tri(n) * sizeof(T).  The chunk is whatever fits a byte
// budget, so a problem already under the budget runs in a single pass, and the
// launcher reuses the one buffer for every chunk otherwise.
//
// Splitting a problem into several chunks requires its unchunked workspace to
// exceed the budget, and C is about twice that, so the cheapest shape that
// splits costs a few GB.  The pre_checkin batteries therefore all run in a
// single chunk; the nightly battery at the end pays that cost and is the only
// place the host chunk loop iterates.
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
#include <limits>
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

    // real_t<T> comes from the library via client_utility.hpp.

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

    // Mirrors of two library-internal constants, which a client test cannot
    // reference. c_gemm_stride is the 16-bit grid ceiling rounded down to a
    // multiple of 16, the stride rocblas_internal_gemm_64 uses for its own batch
    // loop; c_budget is the workspace byte cap.
    constexpr rocblas_int c_gemm_stride = ((1 << 16) - 1) & ~0xf; // 65520

    // Mirror of c_syrk_herk_workspace_max_bytes.
    //
    // NOTE ON COVERAGE: at this budget every shape these tests can afford to
    // allocate fits in a single chunk, because C is about twice the unchunked
    // workspace and so forcing a split needs several GB of it. The host chunk
    // loop therefore runs exactly one iteration here, and the multi-chunk
    // indexing -- batch_off, the A and C pointer advances, the local-to-absolute
    // batch mapping -- is verified only through the size-query battery's
    // arithmetic, not through the kernels. The kernel's grid-stride sweep does
    // run, since chunk_size exceeds the grid ceiling.
    constexpr size_t c_budget = size_t(1024) * 1024 * 1024;

    // Port of rocblas_syrk_herk_chunk_size. Kept deliberately literal so a change
    // to the production rule shows up here as a test failure rather than silently
    // agreeing.
    template <typename T>
    static rocblas_int model_chunk(rocblas_int n, rocblas_int batch_count)
    {
        const size_t per_batch = tri_n(n) * sizeof(T);
        if(!per_batch)
            return batch_count;

        size_t chunk = c_budget / per_batch;
        if(chunk < 1)
            chunk = 1;
        if(chunk > size_t(batch_count))
            chunk = size_t(batch_count);
        if(chunk < size_t(batch_count) && chunk > size_t(c_gemm_stride))
            chunk = (chunk / size_t(c_gemm_stride)) * size_t(c_gemm_stride);

        return rocblas_int(chunk);
    }

    // Expected workspace bytes: one chunk's worth of triangle slots.
    template <typename T>
    static size_t chunked_workspace_bytes(rocblas_int n, rocblas_int batch_count)
    {
        return tri_n(n) * sizeof(T) * size_t(model_chunk<T>(n, batch_count));
    }

    // Batches worth spot-checking: the ends, plus the batches either side of the
    // first chunk boundary when the shape actually produces one. Checking a fixed
    // index would silently stop straddling the boundary whenever the chunk rule
    // changes.
    template <typename T>
    static std::vector<rocblas_int> chunk_probe_batches(rocblas_int n, rocblas_int batch_count)
    {
        std::vector<rocblas_int> probes{0, batch_count - 1};

        const rocblas_int chunk = model_chunk<T>(n, batch_count);
        if(chunk < batch_count)
        {
            probes.push_back(chunk - 1); // last of chunk 0
            probes.push_back(chunk); // first of chunk 1
        }
        return probes;
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
            : aliased_ptr_array(std::vector<T*>(size_t(batch_count), base))
        {
        }

        // Caller-supplied mapping, so a batch can be pointed at one of several
        // buffers rather than all sharing one.
        explicit aliased_ptr_array(const std::vector<T*>& host)
        {
            const size_t bytes = sizeof(T*) * host.size();
            if((hipMalloc)(&m_device, bytes) != hipSuccess)
            {
                m_device = nullptr;
                return;
            }
            if(hipMemcpy(m_device, host.data(), bytes, hipMemcpyHostToDevice) != hipSuccess)
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
        const rocblas_int    k = c_k;

        // A size query allocates nothing, so this sweeps shapes far larger than
        // the correctness tests can afford. n is varied as well as batch_count
        // because the cap is a byte budget: large n reaches it at a small batch
        // count, and small n may never reach it at all.
        struct
        {
            rocblas_int n, bc;
        } cases[] = {
            {32, 1},
            {32, 100},
            {32, c_gemm_stride},
            {32, c_gemm_stride + 1},
            {32, 65536},
            {32, 131070},
            {2, 131070}, // smallest triangle: the budget never binds
            {256, 100000}, // budget binds: tens of chunks
            {1024, 50000}, // budget binds hard: hundreds of chunks
        };

        for(const auto& c : cases)
        {
            size_t reported = 0;
            ASSERT_TRUE((query_strided_workspace<T, K>(
                handle, strided_api, c.n, k, c.n, c.n, c.bc, &reported)))
                << "strided query failed at n=" << c.n << " batch_count=" << c.bc;
            if(reported)
            {
                EXPECT_EQ(reported, rounded(chunked_workspace_bytes<T>(c.n, c.bc)))
                    << "n=" << c.n << " batch_count=" << c.bc;
                // The budget is only honoured while a single batch fits in
                // it. Past that the chunk clamps to one batch and the
                // allocation is that batch, budget or not.  That branch
                // cannot be reached at the shipped budget, since
                // rocblas_use_only_gemm caps n below 4000 and so caps a
                // triangle at about 122MB; it is asserted anyway so the
                // rule stays right if the budget is ever lowered.
                const size_t per_batch = tri_n(c.n) * sizeof(T);
                if(per_batch <= c_budget)
                    EXPECT_LE(reported, rounded(c_budget))
                        << "workspace exceeds the byte budget at n=" << c.n
                        << " batch_count=" << c.bc;
                else
                    EXPECT_EQ(reported, rounded(per_batch))
                        << "a single batch exceeds the budget, so the chunk must be "
                           "exactly one batch, at n="
                        << c.n << " batch_count=" << c.bc;
                EXPECT_LE(reported, rounded(tri_n(c.n) * sizeof(T) * size_t(c.bc)))
                    << "chunking must never ask for more than the unchunked size";
            }

            reported = 0;
            ASSERT_TRUE((query_batched_workspace<T, K>(
                handle, batched_api, c.n, k, c.n, c.n, c.bc, &reported)))
                << "batched query failed at n=" << c.n << " batch_count=" << c.bc;
            if(reported)
                EXPECT_EQ(reported, rounded(chunked_workspace_bytes<T>(c.n, c.bc)))
                    << "n=" << c.n << " batch_count=" << c.bc;
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
        // Several chunks, a single chunk, and the degenerate single-batch case.
        for(rocblas_int bc : {600000, 65539, 1})
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
    // Every A element is s+si (s for real types), so alpha*A*A^T and
    // alpha*A*A^H are constant across the matrix but differ from each other:
    // 2s^2*k imaginary for syrk against 2s^2*k real for herk.  An all-real A
    // would make the two indistinguishable.
    //
    // Each batch starts with a distinct value in the upper triangle,
    // which a lower-fill syrk/herk must leave untouched: on the workspace path
    // it is saved to W_C before the GEMM overwrites all of C and restored
    // afterwards.  Zero-filled data would hide a chunk loop that failed to
    // advance the C pointer, because every batch would look alike; distinct
    // per-batch values make a wrong batch index observable.
    //
    // The complex diagonal is seeded with a non-zero imaginary part, which herk
    // must discard (its output diagonal is real by definition) and syrk must
    // keep.  The strictly-upper seed also carries one, so preserving it is a
    // check on the full value rather than just its real part.
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

    // Every A element. Complex A carries a non-zero imaginary part so that
    // A*A^T and A*A^H differ: an all-real A makes syrk and herk produce the
    // same numbers, and the expectations below would hold even if the library
    // confused the two.
    template <typename T>
    static T a_value(int scale)
    {
        return make_val<T>(double(scale), rocblas_is_complex<T> ? double(scale) : 0.0);
    }

    // One element of the GEMM contribution, summed over k terms of a_value.
    //   real:    1 * 1        =  1     -> k
    //   syrk:    (s+si)^2     =  2s^2 i  (purely imaginary)
    //   herk:    (s+si)(s-si) =  2s^2    (purely real)
    template <typename T, op_kind K>
    static T gemm_term(rocblas_int k, int scale)
    {
        const double s2 = 2.0 * double(scale) * double(scale) * double(k);

        if constexpr(!rocblas_is_complex<T>)
            return make_val<T>(double(scale) * double(scale) * double(k), 0.0);
        else if constexpr(K == op_kind::herk)
            return make_val<T>(s2, 0.0);
        else
            return make_val<T>(0.0, s2);
    }

    // Diagonal seed. The imaginary part is non-zero for complex types so that
    // herk's rule that the diagonal is real on output is actually exercised;
    // syrk has no such rule and must leave it alone.
    template <typename T>
    static T diag_seed()
    {
        return make_val<T>(1.0, rocblas_is_complex<T> ? 7.0 : 0.0);
    }

    // Fills one column-major n x n C block: seeded diagonal, zero lower
    // off-diagonal (the GEMM overwrites it), distinct upper off-diagonal.
    template <typename T>
    static void seed_c_block(T* C, rocblas_int n, rocblas_int b)
    {
        C[0 + 0 * n] = diag_seed<T>();
        C[1 + 1 * n] = diag_seed<T>();
        C[1 + 0 * n] = make_val<T>(0.0, 0.0);
        C[0 + 1 * n] = expected_upper<T>(b);
    }

    // alpha = beta = 1, so the diagonal is the GEMM term plus its seed, with
    // herk forcing the imaginary part to zero and syrk leaving it alone.
    template <typename T, op_kind K>
    static T expected_diag(rocblas_int k, int scale)
    {
        const T g = gemm_term<T, K>(k, scale);
        return make_val<T>(double(re_of(g)) + double(re_of(diag_seed<T>())),
                           K == op_kind::herk ? 0.0
                                              : double(im_of(g)) + double(im_of(diag_seed<T>())));
    }

    // scale selects which A buffer this batch used; see run_batched_numerical.
    template <typename T, op_kind K>
    static void check_c_block(const T* C, rocblas_int n, rocblas_int k, rocblas_int b, int scale)
    {
        const T g         = gemm_term<T, K>(k, scale);
        const T want_diag = expected_diag<T, K>(k, scale);

        expect_val_eq(C[0 + 0 * n], want_diag, "C[0,0] diagonal", b);
        expect_val_eq(C[1 + 1 * n], want_diag, "C[1,1] diagonal", b);
        expect_val_eq(C[1 + 0 * n], g, "C[1,0] lower (GEMM result)", b);
        expect_val_eq(C[0 + 1 * n], expected_upper<T>(b), "C[0,1] (upper triangle preserved)", b);
    }

    template <typename T, op_kind K, typename ApiFunc>
    static void run_strided_numerical(ApiFunc api, rocblas_int batch_count)
    {
        using S = scalar_t<T, K>;

        rocblas_local_handle handle;
        // k must stay at or above syrk_k_lower_threshold, or rocblas_use_only_gemm
        // is false and the call never reaches the chunked workspace path at all.
        const rocblas_int n = c_n, k = c_k;

        const size_t a_elems = size_t(n) * size_t(k);
        const size_t c_elems = size_t(n) * size_t(n);

        std::vector<T> h_A(a_elems, a_value<T>(1));
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

        // stride_A = 0 aliases A, so every batch used the same buffer: scale 1.
        for(rocblas_int b : chunk_probe_batches<T>(n, batch_count))
            check_c_block<T, K>(h_result.data() + size_t(b) * c_elems, n, k, b, 1);
    }

    template <typename T, op_kind K, typename ApiFunc>
    static void run_batched_numerical(ApiFunc api, rocblas_int batch_count)
    {
        using S = scalar_t<T, K>;

        rocblas_local_handle handle;
        // k must stay at or above syrk_k_lower_threshold, or rocblas_use_only_gemm
        // is false and the call never reaches the chunked workspace path at all.
        const rocblas_int n = c_n, k = c_k;

        const size_t a_elems = size_t(n) * size_t(k);
        const size_t c_elems = size_t(n) * size_t(n);

        // One A buffer shared by every batch. This shape runs in a single chunk,
        // so there is no A-side advance to observe; the multi-chunk battery
        // covers that with two buffers.
        std::vector<T> h_A(a_elems, a_value<T>(1));

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

        for(rocblas_int b : chunk_probe_batches<T>(n, batch_count))
            check_c_block<T, K>(h_result[b], n, k, b, 1);
    }

    template <typename T, op_kind K, typename StridedFn, typename BatchedFn>
    static void run_numerical(StridedFn strided_api, BatchedFn batched_api)
    {
        // 65536 fits one chunk at the test budget, so it covers the single-pass
        // case where the arguments are identical to the unchunked code, while
        // still being large enough that the kernel sweeps. 600000 splits into
        // several chunks for every element type, so the host loop, a non-zero
        // batch_offset and the sweep are all live at once.
        for(rocblas_int bc : {65536, 600000})
        {
            run_strided_numerical<T, K>(strided_api, bc);
            run_batched_numerical<T, K>(batched_api, bc);
        }
    }

    // n=1 makes tri(n) zero, so no workspace is needed at any batch count.
    // Verify the chunk loop still terminates and the scalar result is right.
    template <typename T, op_kind K, typename ApiFunc>
    static void run_degenerate_n1(ApiFunc api)
    {
        using S = scalar_t<T, K>;

        rocblas_local_handle handle;
        // k must stay at or above syrk_k_lower_threshold, or rocblas_use_only_gemm
        // is false and the call never reaches the chunked workspace path at all.
        const rocblas_int n = 1, k = c_k;
        const rocblas_int batch_count = 65536;

        std::vector<T> h_A(size_t(k), a_value<T>(1));
        // Distinct per batch, so a wrong batch index is visible here too. n=1
        // means the whole matrix is the diagonal, and herk discards the
        // imaginary part of that, so the variation has to be in the real part.
        auto n1_seed = [](rocblas_int b) { return make_val<T>(double(1 + (b % 4096)), 0.0); };

        std::vector<T> h_C(size_t(batch_count), make_val<T>(0.0, 0.0));
        for(rocblas_int b = 0; b < batch_count; ++b)
            h_C[size_t(b)] = n1_seed(b);

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

        // alpha = beta = 1, so each batch is its own seed plus the GEMM term.
        const T g = gemm_term<T, K>(k, 1);
        for(rocblas_int b : chunk_probe_batches<T>(n, batch_count))
        {
            const T want = make_val<T>(double(re_of(g)) + double(re_of(n1_seed(b))),
                                       K == op_kind::herk ? 0.0 : double(im_of(g)));
            expect_val_eq(h_result[b], want, "C[0,0]", b);
        }
    }

    // -----------------------------------------------------------------------
    // Multi-chunk battery (nightly)
    //
    // The batteries above all run in a single chunk: a split needs the
    // unchunked workspace to exceed the budget, and C is about twice that, so
    // the smallest shape that splits costs a few GB.  Without this the host
    // chunk loop is never entered more than once and batch_off, the A and C
    // pointer advances and the local-to-absolute batch mapping are covered only
    // by the size-query arithmetic -- which is exactly the code the original
    // defect was in.
    //
    // Memory is kept to C plus one workspace by building a single n x n block on
    // the host and patching only the one element that has to differ per batch.
    // -----------------------------------------------------------------------

    // n is under every gemm-only threshold (the lowest is 1600) so the path is
    // selected, while being large enough that the triangle per batch is big and
    // the batch count needed to overflow the budget stays small.
    constexpr rocblas_int c_multichunk_n = 512;

    // Enough batches for a second chunk with a substantial number of live slots,
    // rather than the degenerate one-batch tail.
    template <typename T>
    static rocblas_int multichunk_batch_count()
    {
        const rocblas_int chunk
            = model_chunk<T>(c_multichunk_n, std::numeric_limits<rocblas_int>::max());
        return chunk + chunk / 2;
    }

    // Seeds one n x n block: real diagonal, zero strictly-lower (the GEMM
    // overwrites it), constant strictly-upper.  The per-batch distinct value is
    // patched into (0,1) afterwards so the bulk of the block can be shared.
    template <typename T>
    static void fill_template_block(std::vector<T>& block, rocblas_int n)
    {
        block.assign(size_t(n) * size_t(n), make_val<T>(0.0, 0.0));
        for(rocblas_int col = 0; col < n; ++col)
            for(rocblas_int row = 0; row < n; ++row)
            {
                T& e = block[size_t(row) + size_t(col) * size_t(n)];
                if(row == col)
                    e = diag_seed<T>();
                else if(row < col)
                    e = expected_upper<T>(0); // patched per batch at (0,1)
            }
    }

    template <typename T, op_kind K, typename ApiFunc>
    static void run_multichunk_strided(ApiFunc api)
    {
        using S = scalar_t<T, K>;

        rocblas_local_handle handle;
        const rocblas_int    n = c_multichunk_n, k = c_k;
        const rocblas_int    batch_count = multichunk_batch_count<T>();
        const rocblas_int    chunk       = model_chunk<T>(n, batch_count);

        ASSERT_LT(chunk, batch_count) << "shape does not split; the battery would test nothing";

        const size_t a_elems = size_t(n) * size_t(k);
        const size_t c_elems = size_t(n) * size_t(n);

        std::vector<T> h_A(a_elems, a_value<T>(1));
        std::vector<T> block;
        fill_template_block(block, n);

        device_vector<T> dA(a_elems);
        device_vector<T> dC(c_elems * size_t(batch_count));
        ASSERT_EQ(dA.memcheck(), hipSuccess);
        ASSERT_EQ(dC.memcheck(), hipSuccess);
        ASSERT_EQ(hipMemcpy((T*)dA, h_A.data(), a_elems * sizeof(T), hipMemcpyHostToDevice),
                  hipSuccess);

        for(rocblas_int b = 0; b < batch_count; ++b)
        {
            ASSERT_EQ(hipMemcpy((T*)dC + size_t(b) * c_elems,
                                block.data(),
                                c_elems * sizeof(T),
                                hipMemcpyHostToDevice),
                      hipSuccess);

            const T upper = expected_upper<T>(b);
            ASSERT_EQ(hipMemcpy((T*)dC + size_t(b) * c_elems + size_t(n), // (0,1)
                                &upper,
                                sizeof(T),
                                hipMemcpyHostToDevice),
                      hipSuccess);
        }

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

        // Only the corner of each probed batch is read back, so the check costs
        // four elements per probe rather than a full mirror of C.
        for(rocblas_int b : chunk_probe_batches<T>(n, batch_count))
        {
            T corner[4];
            for(int i = 0; i < 4; ++i)
            {
                const size_t off = size_t(b) * c_elems + size_t(i % 2) + size_t(i / 2) * size_t(n);
                ASSERT_EQ(hipMemcpy(&corner[i], (T*)dC + off, sizeof(T), hipMemcpyDeviceToHost),
                          hipSuccess);
            }

            const T g = gemm_term<T, K>(k, 1);
            expect_val_eq(corner[0], expected_diag<T, K>(k, 1), "C[0,0] diagonal", b);
            expect_val_eq(corner[1], g, "C[1,0] lower (GEMM result)", b);
            expect_val_eq(corner[3], expected_diag<T, K>(k, 1), "C[1,1] diagonal", b);
            expect_val_eq(corner[2], expected_upper<T>(b), "C[0,1] (upper preserved)", b);
        }
    }

    // Batched form of the multi-chunk battery.  This is the only place the
    // A-side chunk advance is observable: A is read-only, so a test that points
    // every batch at one buffer cannot tell whether the launcher advanced the
    // pointer array. Two buffers keyed on chunk index can -- dropping the
    // advance feeds chunk 0's A to every chunk, which the chunk-1 probe catches.
    // Keying on batch parity would not work, because the chunk sizes here are
    // even and a batch one chunk away shares its parity.
    template <typename T, op_kind K, typename ApiFunc>
    static void run_multichunk_batched(ApiFunc api)
    {
        using S = scalar_t<T, K>;

        rocblas_local_handle handle;
        const rocblas_int    n = c_multichunk_n, k = c_k;
        const rocblas_int    batch_count = multichunk_batch_count<T>();
        const rocblas_int    chunk       = model_chunk<T>(n, batch_count);

        ASSERT_LT(chunk, batch_count) << "shape does not split; the battery would test nothing";

        const size_t a_elems = size_t(n) * size_t(k);
        const size_t c_elems = size_t(n) * size_t(n);

        auto scale_of = [chunk](rocblas_int b) { return ((b / chunk) % 2) ? 2 : 1; };

        std::vector<T> h_A(a_elems * 2);
        std::fill(h_A.begin(), h_A.begin() + a_elems, a_value<T>(1));
        std::fill(h_A.begin() + a_elems, h_A.end(), a_value<T>(2));

        std::vector<T> block;
        fill_template_block(block, n);

        device_vector<T> dA(a_elems * 2);
        device_vector<T> dC(c_elems * size_t(batch_count));
        ASSERT_EQ(dA.memcheck(), hipSuccess);
        ASSERT_EQ(dC.memcheck(), hipSuccess);
        ASSERT_EQ(hipMemcpy((T*)dA, h_A.data(), a_elems * 2 * sizeof(T), hipMemcpyHostToDevice),
                  hipSuccess);

        std::vector<T*> a_ptrs(size_t(batch_count), nullptr);
        std::vector<T*> c_ptrs(size_t(batch_count), nullptr);
        for(rocblas_int b = 0; b < batch_count; ++b)
        {
            a_ptrs[size_t(b)] = (T*)dA + (scale_of(b) == 2 ? a_elems : 0);
            c_ptrs[size_t(b)] = (T*)dC + size_t(b) * c_elems;

            ASSERT_EQ(
                hipMemcpy(
                    c_ptrs[size_t(b)], block.data(), c_elems * sizeof(T), hipMemcpyHostToDevice),
                hipSuccess);

            const T upper = expected_upper<T>(b);
            ASSERT_EQ(hipMemcpy(c_ptrs[size_t(b)] + size_t(n), // (0,1)
                                &upper,
                                sizeof(T),
                                hipMemcpyHostToDevice),
                      hipSuccess);
        }

        aliased_ptr_array<T> dA_ptrs(a_ptrs);
        aliased_ptr_array<T> dC_ptrs(c_ptrs);
        ASSERT_TRUE(dA_ptrs.valid()) << "failed to allocate A pointer array";
        ASSERT_TRUE(dC_ptrs.valid()) << "failed to allocate C pointer array";

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
                      dC_ptrs.ptr_on_device(),
                      n,
                      batch_count),
                  rocblas_status_success);

        for(rocblas_int b : chunk_probe_batches<T>(n, batch_count))
        {
            T corner[4];
            for(int i = 0; i < 4; ++i)
            {
                const size_t off = size_t(b) * c_elems + size_t(i % 2) + size_t(i / 2) * size_t(n);
                ASSERT_EQ(hipMemcpy(&corner[i], (T*)dC + off, sizeof(T), hipMemcpyDeviceToHost),
                          hipSuccess);
            }

            const int scale = scale_of(b);
            const T   g     = gemm_term<T, K>(k, scale);
            expect_val_eq(corner[0], expected_diag<T, K>(k, scale), "C[0,0] diagonal", b);
            expect_val_eq(corner[1], g, "C[1,0] lower (GEMM result)", b);
            expect_val_eq(corner[3], expected_diag<T, K>(k, scale), "C[1,1] diagonal", b);
            expect_val_eq(corner[2], expected_upper<T>(b), "C[0,1] (upper preserved)", b);
        }
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

    // Nightly only: these allocate a few GB, which is the price of a genuine
    // chunk split at the shipped budget. float syrk and double-complex herk are
    // the extremes of element width and cover both alpha/beta conventions; the
    // chunk indexing under test does not vary with type.
    TEST(syrk_herk_chunked_workspace_multichunk_nightly, ssyrk_strided)
    {
        run_multichunk_strided<float, op_kind::syrk>(rocblas_ssyrk_strided_batched);
    }

    TEST(syrk_herk_chunked_workspace_multichunk_nightly, zherk_strided)
    {
        run_multichunk_strided<rocblas_double_complex, op_kind::herk>(
            rocblas_zherk_strided_batched);
    }

    TEST(syrk_herk_chunked_workspace_multichunk_nightly, ssyrk_batched)
    {
        run_multichunk_batched<float, op_kind::syrk>(rocblas_ssyrk_batched);
    }

    TEST(syrk_herk_chunked_workspace_multichunk_nightly, zherk_batched)
    {
        run_multichunk_batched<rocblas_double_complex, op_kind::herk>(rocblas_zherk_batched);
    }

#undef SYRK_HERK_CHUNKED_TESTS

} // namespace
