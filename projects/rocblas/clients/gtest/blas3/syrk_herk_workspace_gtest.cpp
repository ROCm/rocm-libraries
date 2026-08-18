/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "client_utility.hpp"
#include "device_vector.hpp"
#include "rocblas.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace
{
    // Regression for rocblas_copy_triangular_syrk_herk_kernel batch-sweep pointer
    // mutation. Values alone cannot detect it: save/restore use the same map.
    // Supply an exact-sized workspace with a contiguous canary behind it.
    //
    // batch_count notes (pre-fix compounded W_C offset bug):
    //   65536 — intentionally omitted: second grid-z pass has blockIdx.z=0 only,
    //           so the extra offset is zero and the canary cannot fail.
    //   65539 — small-magnitude detector (second pass z=2,3 write past the end).
    //   131070 — two saturated grid-z passes; strongest reproducer for n=2.
    //
    // ILP64 (_64) batched syrk/herk above c_i64_grid_YZ_chunk (65520 production)
    // is chunked into rocblas_internal_syr2k_her2k_template and never reaches
    // this kernel; no _64 canary is required for this defect class.
    constexpr unsigned char c_guard_byte = 0xA5;
    // tri(n)=1 at n=2; pre-fix worst two-pass error is 65534 slots (~262 KiB for
    // float). 8192 bytes catches the leading guard trip for this fixed shape.
    constexpr size_t c_guard_bytes = 8192;

    bool fill_guard(void* workspace, size_t workspace_bytes)
    {
        hipError_t err = hipMemset(static_cast<char*>(workspace) + workspace_bytes,
                                   c_guard_byte,
                                   c_guard_bytes);
        EXPECT_EQ(err, hipSuccess) << "could not fill the guard: " << hipGetErrorString(err);
        if(err != hipSuccess)
            return false;
        err = hipStreamSynchronize(nullptr);
        EXPECT_EQ(err, hipSuccess) << "could not sync the guard fill: " << hipGetErrorString(err);
        return err == hipSuccess;
    }

    void expect_guard_clean(rocblas_handle handle, void* workspace, size_t workspace_bytes)
    {
        hipStream_t stream = nullptr;
        ASSERT_EQ(rocblas_get_stream(handle, &stream), rocblas_status_success);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        std::vector<unsigned char> host(c_guard_bytes);
        ASSERT_EQ(hipMemcpy(host.data(),
                            static_cast<char*>(workspace) + workspace_bytes,
                            c_guard_bytes,
                            hipMemcpyDeviceToHost),
                  hipSuccess);

        size_t differing = 0;
        size_t first     = c_guard_bytes;
        for(size_t i = 0; i < c_guard_bytes; ++i)
        {
            if(host[i] != c_guard_byte)
            {
                if(first == c_guard_bytes)
                    first = i;
                ++differing;
            }
        }
        EXPECT_EQ(differing, size_t(0))
            << differing << " guard byte(s) differ, the first " << first
            << " byte(s) past the end of the " << workspace_bytes
            << " byte user allocated workspace";
    }

    // Owns an exact-sized workspace plus trailing guard; clears handle binding on teardown.
    class canary_workspace
    {
    public:
        canary_workspace(rocblas_handle handle, size_t workspace_bytes)
            : m_handle(handle)
            , m_workspace_bytes(workspace_bytes)
            , m_storage(workspace_bytes + c_guard_bytes)
        {
            if(m_storage.memcheck() != hipSuccess)
                return;
            if(!fill_guard(m_storage, workspace_bytes))
                return;
            if(rocblas_set_workspace(m_handle, m_storage, workspace_bytes) != rocblas_status_success)
                return;
            m_valid = true;
        }

        ~canary_workspace()
        {
            if(m_valid)
                EXPECT_EQ(rocblas_set_workspace(m_handle, nullptr, 0), rocblas_status_success);
        }

        canary_workspace(const canary_workspace&)            = delete;
        canary_workspace& operator=(const canary_workspace&) = delete;

        bool   valid() const { return m_valid; }
        void*  ptr() const { return m_storage; }
        size_t bytes() const { return m_workspace_bytes; }

    private:
        rocblas_handle               m_handle;
        size_t                       m_workspace_bytes = 0;
        device_vector<unsigned char> m_storage;
        bool                         m_valid           = false;
    };

    template <typename T>
    bool query_workspace(rocblas_handle    handle,
                         rocblas_operation transA,
                         rocblas_fill      uplo,
                         rocblas_int       n,
                         rocblas_int       k,
                         rocblas_int       lda,
                         rocblas_int       ldc,
                         rocblas_int       batch_count,
                         bool              strided,
                         bool              herk,
                         size_t*           bytes)
    {
        *bytes = 0;
        auto st = rocblas_start_device_memory_size_query(handle);
        EXPECT_EQ(st, rocblas_status_success);
        if(st != rocblas_status_success)
            return false;

        const T         alpha_t(1);
        const T         beta_t(1);
        const float     alpha_r = 1;
        const float     beta_r  = 1;
        const T* const* null_A  = nullptr;
        T* const*       null_C  = nullptr;
        const T*        null_As = nullptr;
        T*              null_Cs = nullptr;

        if(herk)
        {
            if(strided)
            {
                st = rocblas_herk_strided_batched<T>(handle,
                                                     uplo,
                                                     transA,
                                                     n,
                                                     k,
                                                     &alpha_r,
                                                     null_As,
                                                     lda,
                                                     0,
                                                     &beta_r,
                                                     null_Cs,
                                                     ldc,
                                                     0,
                                                     batch_count);
            }
            else
            {
                st = rocblas_herk_batched<T>(handle,
                                             uplo,
                                             transA,
                                             n,
                                             k,
                                             &alpha_r,
                                             null_A,
                                             lda,
                                             &beta_r,
                                             null_C,
                                             ldc,
                                             batch_count);
            }
        }
        else
        {
            if(strided)
            {
                st = rocblas_syrk_strided_batched<T>(handle,
                                                     uplo,
                                                     transA,
                                                     n,
                                                     k,
                                                     &alpha_t,
                                                     null_As,
                                                     lda,
                                                     0,
                                                     &beta_t,
                                                     null_Cs,
                                                     ldc,
                                                     0,
                                                     batch_count);
            }
            else
            {
                st = rocblas_syrk_batched<T>(handle,
                                             uplo,
                                             transA,
                                             n,
                                             k,
                                             &alpha_t,
                                             null_A,
                                             lda,
                                             &beta_t,
                                             null_C,
                                             ldc,
                                             batch_count);
            }
        }

        EXPECT_TRUE(st == rocblas_status_size_increased || st == rocblas_status_size_unchanged)
            << "size query call returned " << rocblas_status_to_string(st);
        if(!(st == rocblas_status_size_increased || st == rocblas_status_size_unchanged))
            return false;

        st = rocblas_stop_device_memory_size_query(handle, bytes);
        EXPECT_EQ(st, rocblas_status_success);
        return st == rocblas_status_success;
    }

    template <typename T>
    void run_workspace_canary(const Arguments& arg, bool strided, bool herk)
    {
        const rocblas_int  n           = arg.N;
        const rocblas_int  k           = arg.K;
        const rocblas_int  lda         = arg.lda;
        const rocblas_int  ldc         = arg.ldc;
        const rocblas_int  batch_count = arg.batch_count;
        const rocblas_fill uplo        = char2rocblas_fill(arg.uplo);

        // Default-constructed handle: do not go through user_allocated_workspace.
        rocblas_local_handle handle;

        const rocblas_operation transA
            = herk ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose;

        size_t workspace_bytes = 0;
        ASSERT_TRUE(query_workspace<T>(handle,
                                       transA,
                                       uplo,
                                       n,
                                       k,
                                       lda,
                                       ldc,
                                       batch_count,
                                       strided,
                                       herk,
                                       &workspace_bytes));
        // Path must be live: a zero query means rocblas_use_only_gemm did not select
        // the workspace path (yaml gpu_arch also restricts these cases to gfx90a/gfx942).
        if(workspace_bytes == 0)
            GTEST_SKIP() << "size query returned 0; gemm-only workspace path was not selected";

        canary_workspace workspace(handle, workspace_bytes);
        ASSERT_TRUE(workspace.valid()) << "could not provision the canary workspace";

        // Alias A across batches (read-only). Keep C distinct so writers do not race.
        // For transA = T/C, A is k x n with lda >= k. Storage lda * n.
        const size_t a_storage = size_t(lda) * size_t(n);
        const size_t c_elems   = size_t(ldc) * size_t(n) * size_t(batch_count);

        device_vector<T> dA(a_storage);
        device_vector<T> dC(c_elems);
        ASSERT_EQ(dA.memcheck(), hipSuccess);
        ASSERT_EQ(dC.memcheck(), hipSuccess);
        ASSERT_EQ(hipMemset((T*)dA, 0, a_storage * sizeof(T)), hipSuccess);
        ASSERT_EQ(hipMemset((T*)dC, 0, c_elems * sizeof(T)), hipSuccess);

        const T     alpha_t(1);
        const T     beta_t(1);
        const float alpha_r = 1;
        const float beta_r  = 1;

        if(strided)
        {
            // stride_a = 0 aliases every batch to the same A.
            if(herk)
            {
                ASSERT_EQ(rocblas_herk_strided_batched<T>(handle,
                                                          uplo,
                                                          transA,
                                                          n,
                                                          k,
                                                          &alpha_r,
                                                          (const T*)dA,
                                                          lda,
                                                          0,
                                                          &beta_r,
                                                          (T*)dC,
                                                          ldc,
                                                          rocblas_stride(ldc) * n,
                                                          batch_count),
                          rocblas_status_success);
            }
            else
            {
                ASSERT_EQ(rocblas_syrk_strided_batched<T>(handle,
                                                          uplo,
                                                          transA,
                                                          n,
                                                          k,
                                                          &alpha_t,
                                                          (const T*)dA,
                                                          lda,
                                                          0,
                                                          &beta_t,
                                                          (T*)dC,
                                                          ldc,
                                                          rocblas_stride(ldc) * n,
                                                          batch_count),
                          rocblas_status_success);
            }
        }
        else
        {
            std::vector<const T*> hA(batch_count, (const T*)dA);
            std::vector<T*>       hC(batch_count);
            for(rocblas_int b = 0; b < batch_count; ++b)
                hC[b] = (T*)dC + size_t(b) * size_t(ldc) * size_t(n);

            device_vector<const T*> dA_array(batch_count);
            device_vector<T*>       dC_array(batch_count);
            ASSERT_EQ(dA_array.memcheck(), hipSuccess);
            ASSERT_EQ(dC_array.memcheck(), hipSuccess);
            ASSERT_EQ(hipMemcpy(dA_array,
                                hA.data(),
                                sizeof(const T*) * size_t(batch_count),
                                hipMemcpyHostToDevice),
                      hipSuccess);
            ASSERT_EQ(hipMemcpy(dC_array,
                                hC.data(),
                                sizeof(T*) * size_t(batch_count),
                                hipMemcpyHostToDevice),
                      hipSuccess);

            if(herk)
            {
                ASSERT_EQ(rocblas_herk_batched<T>(handle,
                                                  uplo,
                                                  transA,
                                                  n,
                                                  k,
                                                  &alpha_r,
                                                  dA_array,
                                                  lda,
                                                  &beta_r,
                                                  dC_array,
                                                  ldc,
                                                  batch_count),
                          rocblas_status_success);
            }
            else
            {
                ASSERT_EQ(rocblas_syrk_batched<T>(handle,
                                                  uplo,
                                                  transA,
                                                  n,
                                                  k,
                                                  &alpha_t,
                                                  dA_array,
                                                  lda,
                                                  &beta_t,
                                                  dC_array,
                                                  ldc,
                                                  batch_count),
                          rocblas_status_success);
            }
        }

        expect_guard_clean(handle, workspace.ptr(), workspace.bytes());
    }

    bool is_workspace_canary_function(const char* fn)
    {
        return !strcmp(fn, "syrk_batched_workspace_canary")
               || !strcmp(fn, "syrk_strided_batched_workspace_canary")
               || !strcmp(fn, "herk_batched_workspace_canary")
               || !strcmp(fn, "herk_strided_batched_workspace_canary")
               || !strcmp(fn, "dsyrk_batched_workspace_canary")
               || !strcmp(fn, "dsyrk_strided_batched_workspace_canary")
               || !strcmp(fn, "zherk_batched_workspace_canary")
               || !strcmp(fn, "zherk_strided_batched_workspace_canary");
    }

    void testing_syrk_herk_workspace(const Arguments& arg)
    {
        const char* fn = arg.function;
        if(!strcmp(fn, "syrk_batched_workspace_canary"))
            run_workspace_canary<float>(arg, false, false);
        else if(!strcmp(fn, "syrk_strided_batched_workspace_canary"))
            run_workspace_canary<float>(arg, true, false);
        else if(!strcmp(fn, "herk_batched_workspace_canary"))
            run_workspace_canary<rocblas_float_complex>(arg, false, true);
        else if(!strcmp(fn, "herk_strided_batched_workspace_canary"))
            run_workspace_canary<rocblas_float_complex>(arg, true, true);
        else if(!strcmp(fn, "dsyrk_batched_workspace_canary"))
            run_workspace_canary<double>(arg, false, false);
        else if(!strcmp(fn, "dsyrk_strided_batched_workspace_canary"))
            run_workspace_canary<double>(arg, true, false);
        else if(!strcmp(fn, "zherk_batched_workspace_canary"))
            run_workspace_canary<rocblas_double_complex>(arg, false, true);
        else if(!strcmp(fn, "zherk_strided_batched_workspace_canary"))
            run_workspace_canary<rocblas_double_complex>(arg, true, true);
        else
            FAIL() << "unexpected function " << fn;
    }

    template <typename...>
    struct testing_syrk_herk_workspace_fun : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            testing_syrk_herk_workspace(arg);
        }
    };

    struct syrk_herk_workspace_gtest
        : RocBLAS_Test<syrk_herk_workspace_gtest, testing_syrk_herk_workspace_fun>
    {
        static bool type_filter(const Arguments&)
        {
            return true;
        }

        static bool function_filter(const Arguments& arg)
        {
            return is_workspace_canary_function(arg.function);
        }

        static std::string name_suffix(const Arguments& arg)
        {
            std::string name = RocBLAS_TestName<syrk_herk_workspace_gtest>(arg.name);
            name += arg.function;
            return name;
        }
    };

    TEST_P(syrk_herk_workspace_gtest, pre_checkin)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            testing_syrk_herk_workspace_fun<>{}(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(syrk_herk_workspace_gtest)
} // namespace
