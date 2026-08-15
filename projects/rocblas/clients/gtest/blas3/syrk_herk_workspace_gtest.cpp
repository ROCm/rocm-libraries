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
#include "rocblas_test.hpp"

#include "../../library/src/include/utility.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace
{
    // Regression for rocblas_copy_triangular_syrk_herk_kernel batch-sweep pointer
    // mutation. Values alone cannot detect it: save/restore use the same map.
    // Supply an exact-sized workspace with a contiguous canary behind it.
    constexpr rocblas_int   c_n           = 2;
    constexpr rocblas_int   c_k           = 500; // syrk_k_lower_threshold
    constexpr rocblas_int   c_batch_count = 131070; // two saturated grid-z passes
    constexpr rocblas_int   c_lda         = 500;
    constexpr rocblas_int   c_ldc         = 2;
    constexpr unsigned char c_guard_byte  = 0xA5;
    constexpr size_t        c_guard_bytes = 8192;

    bool arch_has_gemm_only_path()
    {
        const std::string arch = rocblas_internal_get_arch_name();
        return arch == "gfx90a" || arch == "gfx942";
    }

    void fill_guard(void* workspace, size_t workspace_bytes)
    {
        ASSERT_EQ(hipMemset(static_cast<char*>(workspace) + workspace_bytes,
                            c_guard_byte,
                            c_guard_bytes),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(nullptr), hipSuccess);
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

    template <typename T>
    bool query_workspace(rocblas_handle    handle,
                         rocblas_operation transA,
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
                                                     rocblas_fill_upper,
                                                     transA,
                                                     c_n,
                                                     c_k,
                                                     &alpha_r,
                                                     null_As,
                                                     c_lda,
                                                     0,
                                                     &beta_r,
                                                     null_Cs,
                                                     c_ldc,
                                                     0,
                                                     c_batch_count);
            }
            else
            {
                st = rocblas_herk_batched<T>(handle,
                                             rocblas_fill_upper,
                                             transA,
                                             c_n,
                                             c_k,
                                             &alpha_r,
                                             null_A,
                                             c_lda,
                                             &beta_r,
                                             null_C,
                                             c_ldc,
                                             c_batch_count);
            }
        }
        else
        {
            if(strided)
            {
                st = rocblas_syrk_strided_batched<T>(handle,
                                                     rocblas_fill_upper,
                                                     transA,
                                                     c_n,
                                                     c_k,
                                                     &alpha_t,
                                                     null_As,
                                                     c_lda,
                                                     0,
                                                     &beta_t,
                                                     null_Cs,
                                                     c_ldc,
                                                     0,
                                                     c_batch_count);
            }
            else
            {
                st = rocblas_syrk_batched<T>(handle,
                                             rocblas_fill_upper,
                                             transA,
                                             c_n,
                                             c_k,
                                             &alpha_t,
                                             null_A,
                                             c_lda,
                                             &beta_t,
                                             null_C,
                                             c_ldc,
                                             c_batch_count);
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
    void run_workspace_canary(bool strided, bool herk)
    {
        if(!arch_has_gemm_only_path())
            GTEST_SKIP() << "gemm-only syrk/herk path is live only on gfx90a/gfx942";

        // Default-constructed handle: do not go through user_allocated_workspace.
        rocblas_local_handle handle;

        const rocblas_operation transA
            = herk ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose;

        size_t workspace_bytes = 0;
        ASSERT_TRUE(query_workspace<T>(handle, transA, strided, herk, &workspace_bytes));
        // Path must be live: a zero query means we never reached the workspace
        // kernel, so a clean canary would be a false pass.
        ASSERT_GT(workspace_bytes, size_t(0))
            << "size query returned 0; gemm-only workspace path was not selected";

        void* workspace = nullptr;
        ASSERT_EQ(hipMalloc(&workspace, workspace_bytes + c_guard_bytes), hipSuccess);
        fill_guard(workspace, workspace_bytes);
        ASSERT_EQ(rocblas_set_workspace(handle, workspace, workspace_bytes), rocblas_status_success);

        // Alias A across batches (read-only). Keep C distinct so writers do not race.
        // For transA = T/C, A is k x n with lda >= k. Storage lda * n.
        const size_t a_storage = size_t(c_lda) * size_t(c_n);
        const size_t c_elems   = size_t(c_ldc) * size_t(c_n) * size_t(c_batch_count);

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
                                                          rocblas_fill_upper,
                                                          transA,
                                                          c_n,
                                                          c_k,
                                                          &alpha_r,
                                                          (const T*)dA,
                                                          c_lda,
                                                          0,
                                                          &beta_r,
                                                          (T*)dC,
                                                          c_ldc,
                                                          rocblas_stride(c_ldc) * c_n,
                                                          c_batch_count),
                          rocblas_status_success);
            }
            else
            {
                ASSERT_EQ(rocblas_syrk_strided_batched<T>(handle,
                                                          rocblas_fill_upper,
                                                          transA,
                                                          c_n,
                                                          c_k,
                                                          &alpha_t,
                                                          (const T*)dA,
                                                          c_lda,
                                                          0,
                                                          &beta_t,
                                                          (T*)dC,
                                                          c_ldc,
                                                          rocblas_stride(c_ldc) * c_n,
                                                          c_batch_count),
                          rocblas_status_success);
            }
        }
        else
        {
            std::vector<const T*> hA(c_batch_count, (const T*)dA);
            std::vector<T*>       hC(c_batch_count);
            for(rocblas_int b = 0; b < c_batch_count; ++b)
                hC[b] = (T*)dC + size_t(b) * size_t(c_ldc) * size_t(c_n);

            const T** dA_array = nullptr;
            T**       dC_array = nullptr;
            ASSERT_EQ(hipMalloc(&dA_array, sizeof(const T*) * size_t(c_batch_count)), hipSuccess);
            ASSERT_EQ(hipMalloc(&dC_array, sizeof(T*) * size_t(c_batch_count)), hipSuccess);
            ASSERT_EQ(hipMemcpy(dA_array,
                                hA.data(),
                                sizeof(const T*) * size_t(c_batch_count),
                                hipMemcpyHostToDevice),
                      hipSuccess);
            ASSERT_EQ(hipMemcpy(dC_array,
                                hC.data(),
                                sizeof(T*) * size_t(c_batch_count),
                                hipMemcpyHostToDevice),
                      hipSuccess);

            if(herk)
            {
                ASSERT_EQ(rocblas_herk_batched<T>(handle,
                                                  rocblas_fill_upper,
                                                  transA,
                                                  c_n,
                                                  c_k,
                                                  &alpha_r,
                                                  dA_array,
                                                  c_lda,
                                                  &beta_r,
                                                  dC_array,
                                                  c_ldc,
                                                  c_batch_count),
                          rocblas_status_success);
            }
            else
            {
                ASSERT_EQ(rocblas_syrk_batched<T>(handle,
                                                  rocblas_fill_upper,
                                                  transA,
                                                  c_n,
                                                  c_k,
                                                  &alpha_t,
                                                  dA_array,
                                                  c_lda,
                                                  &beta_t,
                                                  dC_array,
                                                  c_ldc,
                                                  c_batch_count),
                          rocblas_status_success);
            }
            EXPECT_EQ(hipFree(dA_array), hipSuccess);
            EXPECT_EQ(hipFree(dC_array), hipSuccess);
        }

        expect_guard_clean(handle, workspace, workspace_bytes);

        // Drop the workspace before freeing so the handle does not retain it.
        EXPECT_EQ(rocblas_set_workspace(handle, nullptr, 0), rocblas_status_success);
        EXPECT_EQ(hipFree(workspace), hipSuccess);
    }

    void testing_syrk_herk_workspace(const Arguments& arg)
    {
        const char* fn = arg.function;
        if(!strcmp(fn, "syrk_batched_workspace_canary"))
            run_workspace_canary<float>(false, false);
        else if(!strcmp(fn, "syrk_strided_batched_workspace_canary"))
            run_workspace_canary<float>(true, false);
        else if(!strcmp(fn, "herk_batched_workspace_canary"))
            run_workspace_canary<rocblas_float_complex>(false, true);
        else if(!strcmp(fn, "herk_strided_batched_workspace_canary"))
            run_workspace_canary<rocblas_float_complex>(true, true);
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
            return !strcmp(arg.function, "syrk_batched_workspace_canary")
                   || !strcmp(arg.function, "syrk_strided_batched_workspace_canary")
                   || !strcmp(arg.function, "herk_batched_workspace_canary")
                   || !strcmp(arg.function, "herk_strided_batched_workspace_canary");
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
