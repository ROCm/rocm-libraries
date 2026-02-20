/* ************************************************************************
 * Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "testing.hpp"

template <typename T>
void testing_copy_info_bad_arg(const Arguments& arg)
{
    // Test invalid parameters for rocsparse_copy_mat_info
    rocsparse_mat_info src;
    rocsparse_mat_info dest;

    rocsparse_local_handle    handle;
    rocsparse_local_mat_descr descr;

    // Create source and destination info structures
    CHECK_ROCSPARSE_ERROR(rocsparse_create_mat_info(&dest));
    CHECK_ROCSPARSE_ERROR(rocsparse_create_mat_info(&src));

    // Pass valid created source or destination along with nullptr
    EXPECT_ROCSPARSE_STATUS(rocsparse_copy_mat_info(nullptr, src),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_copy_mat_info(dest, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_copy_mat_info(nullptr, nullptr),
                            rocsparse_status_invalid_pointer);

    rocsparse_matrix_factory<T> matrix_factory(arg);

    rocsparse_int M1 = arg.M;
    rocsparse_int N1 = arg.N;
    rocsparse_int M2 = 2 * arg.M;
    rocsparse_int N2 = 2 * arg.N;

    host_csr_matrix<T> hA1;
    matrix_factory.init_csr(hA1, M1, N1);
    device_csr_matrix<T> dA1(hA1);

    // Fill a valid source info structure
    CHECK_ROCSPARSE_ERROR(rocsparse_csrmv_analysis<T>(
        handle, arg.transA, dA1.m, dA1.n, dA1.nnz, descr, dA1.val, dA1.ptr, dA1.ind, src));

    host_csr_matrix<T> hA2;
    matrix_factory.init_csr(hA2, M2, N2);
    device_csr_matrix<T> dA2(hA2);

    // Fill another valid destination info structure with different sparsity pattern
    CHECK_ROCSPARSE_ERROR(rocsparse_csrmv_analysis<T>(
        handle, arg.transA, dA2.m, dA2.n, dA2.nnz, descr, dA2.val, dA2.ptr, dA2.ind, dest));

    // Try and copy src info structure to dest info structure
    EXPECT_ROCSPARSE_STATUS(rocsparse_copy_mat_info(dest, src), rocsparse_status_invalid_pointer);

    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_mat_info(dest));
    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_mat_info(src));

    // Test invalid parameters for rocsparse_copy_hyb_mat
    rocsparse_hyb_mat hyb_dest;
    rocsparse_hyb_mat hyb_src;

    // Create source and destination info structures
    CHECK_ROCSPARSE_ERROR(rocsparse_create_hyb_mat(&hyb_dest));
    CHECK_ROCSPARSE_ERROR(rocsparse_create_hyb_mat(&hyb_src));

    // Pass valid created source or destination along with nullptr
    EXPECT_ROCSPARSE_STATUS(rocsparse_copy_hyb_mat(nullptr, hyb_src),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_copy_hyb_mat(hyb_dest, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_copy_hyb_mat(nullptr, nullptr),
                            rocsparse_status_invalid_pointer);

    // Fill a valid source info structure
    rocsparse_int user_ell_width1 = dA1.nnz / dA1.m;
    CHECK_ROCSPARSE_ERROR(rocsparse_csr2hyb<T>(handle,
                                               dA1.m,
                                               dA1.n,
                                               descr,
                                               dA1.val,
                                               dA1.ptr,
                                               dA1.ind,
                                               hyb_src,
                                               user_ell_width1,
                                               rocsparse_hyb_partition_auto));

    // Fill another valid destination info structure with different sparsity pattern
    rocsparse_int user_ell_width2 = dA2.nnz / dA2.m;
    CHECK_ROCSPARSE_ERROR(rocsparse_csr2hyb<T>(handle,
                                               dA2.m,
                                               dA2.n,
                                               descr,
                                               dA2.val,
                                               dA2.ptr,
                                               dA2.ind,
                                               hyb_dest,
                                               user_ell_width2,
                                               rocsparse_hyb_partition_auto));

    // Try and copy src info structure to dest info structure
    EXPECT_ROCSPARSE_STATUS(rocsparse_copy_hyb_mat(hyb_dest, hyb_src),
                            rocsparse_status_invalid_pointer);

    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_hyb_mat(hyb_dest));
    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_hyb_mat(hyb_src));
}

template <typename T>
void testing_copy_info(const Arguments& arg)
{
    rocsparse_int M = arg.M;
    rocsparse_int N = arg.N;

    rocsparse_mat_info src;
    rocsparse_mat_info dest;

    rocsparse_local_handle    handle;
    rocsparse_local_mat_descr descr;

    // Create source and destination info structures
    CHECK_ROCSPARSE_ERROR(rocsparse_create_mat_info(&dest));
    CHECK_ROCSPARSE_ERROR(rocsparse_create_mat_info(&src));

    rocsparse_matrix_factory<T> matrix_factory(arg);

    host_csr_matrix<T> hA;
    matrix_factory.init_csr(hA, M, N);
    device_csr_matrix<T> dA(hA);

    // Fill a valid source info structure
    CHECK_ROCSPARSE_ERROR(rocsparse_csrmv_analysis<T>(
        handle, arg.transA, dA.m, dA.n, dA.nnz, descr, dA.val, dA.ptr, dA.ind, src));

    // Try and copy src info structure to dest info structure
    for(int i = 0; i < 4; i++)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_copy_mat_info(dest, src));
    }

    // Obtain required buffer size
    size_t temp1 = 0;
    size_t temp2 = 0;
    size_t temp3 = 0;
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_buffer_size<T>(
        handle, arg.transA, dA.m, dA.nnz, descr, dA.val, dA.ptr, dA.ind, src, &temp1));
    CHECK_ROCSPARSE_ERROR(rocsparse_csric0_buffer_size<T>(
        handle, dA.m, dA.nnz, descr, dA.val, dA.ptr, dA.ind, src, &temp2));
    CHECK_ROCSPARSE_ERROR(rocsparse_csrilu0_buffer_size<T>(
        handle, dA.m, dA.nnz, descr, dA.val, dA.ptr, dA.ind, src, &temp3));

    size_t buffer_size = 0;
    buffer_size        = std::max(buffer_size, temp1);
    buffer_size        = std::max(buffer_size, temp2);
    buffer_size        = std::max(buffer_size, temp3);

    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_analysis<T>(handle,
                                                      arg.transA,
                                                      dA.m,
                                                      dA.nnz,
                                                      descr,
                                                      dA.val,
                                                      dA.ptr,
                                                      dA.ind,
                                                      src,
                                                      rocsparse_analysis_policy_force,
                                                      rocsparse_solve_policy_auto,
                                                      dbuffer));

    // Try and copy src info structure to dest info structure
    for(int i = 0; i < 4; i++)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_copy_mat_info(dest, src));
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_csric0_analysis<T>(handle,
                                                       dA.m,
                                                       dA.nnz,
                                                       descr,
                                                       dA.val,
                                                       dA.ptr,
                                                       dA.ind,
                                                       src,
                                                       rocsparse_analysis_policy_force,
                                                       rocsparse_solve_policy_auto,
                                                       dbuffer));

    // Try and copy src info structure to dest info structure
    for(int i = 0; i < 4; i++)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_copy_mat_info(dest, src));
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_csric0_analysis<T>(handle,
                                                       dA.m,
                                                       dA.nnz,
                                                       descr,
                                                       dA.val,
                                                       dA.ptr,
                                                       dA.ind,
                                                       src,
                                                       rocsparse_analysis_policy_force,
                                                       rocsparse_solve_policy_auto,
                                                       dbuffer));

    // Try and copy src info structure to dest info structure
    for(int i = 0; i < 4; i++)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_copy_mat_info(dest, src));
    }

    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));

    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_mat_info(dest));
    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_mat_info(src));

    rocsparse_hyb_mat hyb_dest;
    rocsparse_hyb_mat hyb_src;

    // Create source and destination info structures
    CHECK_ROCSPARSE_ERROR(rocsparse_create_hyb_mat(&hyb_dest));
    CHECK_ROCSPARSE_ERROR(rocsparse_create_hyb_mat(&hyb_src));

    rocsparse_int user_ell_width = dA.nnz / dA.m;
    CHECK_ROCSPARSE_ERROR(rocsparse_csr2hyb<T>(handle,
                                               dA.m,
                                               dA.n,
                                               descr,
                                               dA.val,
                                               dA.ptr,
                                               dA.ind,
                                               hyb_src,
                                               user_ell_width,
                                               rocsparse_hyb_partition_auto));

    // Try and copy src info structure to dest info structure
    for(int i = 0; i < 4; i++)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_copy_hyb_mat(hyb_dest, hyb_src));
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_hyb_mat(hyb_dest));
    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_hyb_mat(hyb_src));
}

#define INSTANTIATE(TYPE)                                                \
    template void testing_copy_info_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_copy_info<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_copy_info_extra(const Arguments& arg) {}
