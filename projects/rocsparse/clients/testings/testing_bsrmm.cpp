/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse.hpp"
#include "testing.hpp"
#include "utility.hpp"

#include "rocsparse_enum.hpp"

template <typename T>
void testing_bsrmm_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_descr;

    T h_alpha = static_cast<T>(1);
    T h_beta  = static_cast<T>(1);

    // local declaration
    rocsparse_handle     handle      = local_handle;
    rocsparse_direction  dir         = rocsparse_direction_row;
    rocsparse_operation  trans_A     = rocsparse_operation_none;
    rocsparse_operation  trans_B     = rocsparse_operation_none;
    rocsparse_int        mb          = safe_size;
    rocsparse_int        n           = safe_size;
    rocsparse_int        kb          = safe_size;
    rocsparse_int        nnzb        = safe_size;
    const T*             alpha       = &h_alpha;
    rocsparse_mat_descr  descr       = local_descr;
    const T*             bsr_val     = (const T*)0x4;
    const rocsparse_int* bsr_row_ptr = (const rocsparse_int*)0x4;
    const rocsparse_int* bsr_col_ind = (const rocsparse_int*)0x4;
    rocsparse_int        block_dim   = safe_size;
    const T*             dense_B     = (const T*)0x4;
    rocsparse_int        ldb         = safe_size;
    const T*             beta        = &h_beta;
    T*                   dense_C     = (T*)0x4;
    rocsparse_int        ldc         = safe_size;

#define PARAMS                                                                          \
    handle, dir, trans_A, trans_B, mb, n, kb, nnzb, alpha, descr, bsr_val, bsr_row_ptr, \
        bsr_col_ind, block_dim, dense_B, ldb, beta, dense_C, ldc

    //
    // Auto testing.
    //
    bad_arg_analysis(rocsparse_bsrmm<T>, PARAMS);

    //
    // LOOP OVER MATRIX TYPES DIFFERENT FROM TYPE_GENERAL.
    //
    for(auto val : rocsparse_matrix_type_t::values)
    {
        if(val != rocsparse_matrix_type_general)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, val));
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_not_implemented);
        }
    }
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_requires_sorted_storage);
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_sorted));

    //
    // NOT IMPLEMENTED
    //
    {
        auto tmp = trans_A;
        trans_A  = rocsparse_operation_transpose;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_not_implemented);
        trans_A = rocsparse_operation_conjugate_transpose;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_not_implemented);
        trans_A = tmp;
    }

    {
        auto tmp = trans_B;
        trans_B  = rocsparse_operation_conjugate_transpose;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_not_implemented);
        trans_B = tmp;
    }

    //
    // INVALID SIZE
    //
    {
        auto tmp  = block_dim;
        block_dim = 0;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_invalid_size);
        block_dim = tmp;
    }

    //
    // Testing wrong leading dimensions.
    //
    {
        //
        // op(A) = A, op(B) = B
        //
        mb      = 3;
        n       = 14;
        kb      = 32;
        trans_A = rocsparse_operation_none;
        trans_B = rocsparse_operation_none;

        //  ldb < k
        ldb = kb * block_dim - 1;
        ldc = mb * block_dim;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < m
        ldb = kb * block_dim;
        ldc = mb * block_dim - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_invalid_size);
    }

    {
        //
        // op(A) = A, op(B) = B^T
        //
        mb      = 3;
        n       = 14;
        kb      = 32;
        trans_A = rocsparse_operation_none;
        trans_B = rocsparse_operation_transpose;

        //  ldb < n
        ldb = n - 1;
        ldc = mb * block_dim;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < m
        ldb = n;
        ldc = mb * block_dim - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_invalid_size);

#ifdef ROCSPARSE_BSRMM_CONJUGATE_TRANSPOSE_B_SUPPORTED
        trans_B = rocsparse_operation_conjugate_transpose;
        //  ldb < n
        ldb = n - 1;
        ldc = mb * block_dim;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < m
        ldb = n;
        ldc = mb * block_dim - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_invalid_size);
#endif
    }

#ifdef ROCSPARSE_BSRMM_TRANSPOSE_A_SUPPORTED
    {
        //
        // op(A) = A^T, op(B) = B
        //
        mb      = 3;
        n       = 14;
        kb      = 32;
        trans_A = rocsparse_operation_transpose;
        trans_B = rocsparse_operation_none;

        //  ldb < m
        ldb = mb * block_dim - 1;
        ldc = kb * block_dim;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < k
        ldb = mb * block_dim;
        ldc = kb * block_dim - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_invalid_size);

#ifdef ROCSPARSE_BSRMM_CONJUGATE_TRANSPOSE_A_SUPPORTED
        trans_A = rocsparse_operation_conjugate_transpose;
        //  ldb < m
        ldb = mb * block_dim - 1;
        ldc = kb * block_dim;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < k
        ldb = mb * block_dim;
        ldc = kb * block_dim - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_invalid_size);
#endif
    }

    {
        //
        // op(A) = A^T, op(B) = B^T
        //
        mb      = 3;
        n       = 14;
        kb      = 32;
        trans_A = rocsparse_operation_transpose;
        trans_B = rocsparse_operation_transpose;

        //  ldb < n
        ldb = n - 1;
        ldc = kb * block_dim;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < k
        ldb = n;
        ldc = kb * block_dim - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_invalid_size);

#ifdef ROCSPARSE_BSRMM_CONJUGATE_TRANSPOSE_A_SUPPORTED
        trans_A = rocsparse_operation_conjugate_transpose;
        //  ldb < n
        ldb = n - 1;
        ldc = kb * block_dim;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < k
        ldb = n;
        ldc = kb * block_dim - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_invalid_size);
#endif

#ifdef ROCSPARSE_BSRMM_CONJUGATE_TRANSPOSE_B_SUPPORTED
        trans_A = rocsparse_operation_transpose;
        trans_B = rocsparse_operation_conjugate_transpose;

        //  ldb < n
        ldb = n - 1;
        ldc = kb * block_dim;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < k
        ldb = n;
        ldc = kb * block_dim - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_invalid_size);

        trans_A = rocsparse_operation_conjugate_transpose;
        //  ldb < n
        ldb = n - 1;
        ldc = kb * block_dim;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < k
        ldb = n;
        ldc = kb * block_dim - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(PARAMS), rocsparse_status_invalid_size);
#endif
    }
#endif

#undef PARAMS

    // Additional tests for invalid zero matrices
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmm<T>(handle,
                                               dir,
                                               trans_A,
                                               trans_B,
                                               mb,
                                               n,
                                               kb,
                                               nnzb,
                                               alpha,
                                               descr,
                                               nullptr,
                                               bsr_row_ptr,
                                               nullptr,
                                               block_dim,
                                               dense_B,
                                               ldb,
                                               beta,
                                               dense_C,
                                               ldc),
                            rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_bsrmm(const Arguments& arg)
{
    rocsparse_int        M               = arg.M;
    rocsparse_int        N               = arg.N;
    rocsparse_int        K               = arg.K;
    rocsparse_int        block_dim       = arg.block_dim;
    rocsparse_operation  transA          = arg.transA;
    rocsparse_operation  transB          = arg.transB;
    rocsparse_direction  direction       = arg.direction;
    rocsparse_index_base base            = arg.baseA;
    rocsparse_int        ld_multiplier_B = arg.ld_multiplier_B;
    rocsparse_int        ld_multiplier_C = arg.ld_multiplier_C;

    rocsparse_int Mb = (M + block_dim - 1) / block_dim;
    rocsparse_int Kb = (K + block_dim - 1) / block_dim;

    host_scalar<T> h_alpha(arg.get_alpha<T>());
    host_scalar<T> h_beta(arg.get_beta<T>());

    device_scalar<T> d_alpha(h_alpha);
    device_scalar<T> d_beta(h_beta);

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // Allocate host memory for output BSR matrix
    rocsparse_matrix_factory<T> matrix_factory(arg);

    host_gebsr_matrix<T>   hA;
    device_gebsr_matrix<T> dA;

    matrix_factory.init_bsr(hA, dA, Mb, Kb, base);

    M = Mb * dA.row_block_dim;
    K = Kb * dA.col_block_dim;

    // Allocate B matrix
    host_dense_matrix<T> hB_temp((transB == rocsparse_operation_none)
                                     ? int64_t(ld_multiplier_B) * K
                                     : int64_t(ld_multiplier_B) * N,
                                 (transB == rocsparse_operation_none) ? N : K);
    device_dense_matrix<T> dB_temp((transB == rocsparse_operation_none)
                                       ? int64_t(ld_multiplier_B) * K
                                       : int64_t(ld_multiplier_B) * N,
                                   (transB == rocsparse_operation_none) ? N : K);
    rocsparse_matrix_utils::init(hB_temp);
    dB_temp.transfer_from(hB_temp);

    // Layout of B matrix
    host_dense_matrix_view<T> hB((transB == rocsparse_operation_none) ? K : N,
                                 (transB == rocsparse_operation_none) ? N : K,
                                 hB_temp.data(),
                                 (transB == rocsparse_operation_none)
                                     ? int64_t(ld_multiplier_B) * K
                                     : int64_t(ld_multiplier_B) * N);

    device_dense_matrix_view<T> dB((transB == rocsparse_operation_none) ? K : N,
                                   (transB == rocsparse_operation_none) ? N : K,
                                   dB_temp.data(),
                                   (transB == rocsparse_operation_none)
                                       ? int64_t(ld_multiplier_B) * K
                                       : int64_t(ld_multiplier_B) * N);

    // Allocate C matrix
    host_dense_matrix<T>   hC_temp(int64_t(ld_multiplier_C) * M, N);
    device_dense_matrix<T> dC_temp(int64_t(ld_multiplier_C) * M, N);
    rocsparse_matrix_utils::init(hC_temp);
    dC_temp.transfer_from(hC_temp);

    // Layout of C matrix
    host_dense_matrix_view<T>   hC(M, N, hC_temp.data(), int64_t(ld_multiplier_C) * M);
    device_dense_matrix_view<T> dC(M, N, dC_temp.data(), int64_t(ld_multiplier_C) * M);

#define PARAMS(alpha_, dA_, dB_, beta_, dC_)                                                 \
    handle, direction, transA, transB, Mb, N, Kb, dA_.nnzb, alpha_, descr, dA_.val, dA_.ptr, \
        dA_.ind, dA_.row_block_dim, dB_, dB_.ld, beta_, dC_, dC_.ld

    if(arg.unit_check)
    {
        //
        // Pointer mode host
        //
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_bsrmm<T>(PARAMS(h_alpha, dA, dB, h_beta, dC)));

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("C pointer mode host", dC);
        }
        //
        // Compute on host.
        //
        {
            host_dense_matrix<T> hC_copy(hC);
            host_bsrmm<T, rocsparse_int, rocsparse_int, T, T, T>(handle,
                                                                 direction,
                                                                 transA,
                                                                 transB,
                                                                 Mb,
                                                                 N,
                                                                 Kb,
                                                                 hA.nnzb,
                                                                 *h_alpha,
                                                                 hA.val,
                                                                 hA.ptr,
                                                                 hA.ind,
                                                                 hA.row_block_dim,
                                                                 hB,
                                                                 hB.ld,
                                                                 rocsparse_order_column,
                                                                 *h_beta,
                                                                 hC,
                                                                 hC.ld,
                                                                 rocsparse_order_column,
                                                                 base);
            hC.near_check(dC);
            dC = hC_copy;
        }

        //
        // Pointer mode device
        //
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_bsrmm<T>(PARAMS(d_alpha, dA, dB, d_beta, dC)));
        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("C pointer mode device", dC);
        }

        hC.near_check(dC);
    }

    if(arg.timing)
    {

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        const double gpu_time_used = rocsparse_clients::run_benchmark(
            arg, rocsparse_bsrmm<T>, PARAMS(h_alpha, dA, dB, h_beta, dC));

        double gflop_count
            = bsrmm_gflop_count(N, dA.nnzb, block_dim, dC.m * dC.n, *h_beta != static_cast<T>(0));
        double gbyte_count = bsrmm_gbyte_count<T>(
            Mb, dA.nnzb, block_dim, dB.m * dB.n, dC.m * dC.n, *h_beta != static_cast<T>(0));

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::K,
                            K,
                            display_key_t::dir,
                            direction,
                            display_key_t::trans_A,
                            transA,
                            display_key_t::trans_B,
                            transB,
                            display_key_t::nnzb,
                            dA.nnzb,
                            display_key_t::bdim,
                            block_dim,
                            display_key_t::nnz_B,
                            dB.m * dB.n,
                            display_key_t::nnz_C,
                            dC.m * dC.n,
                            display_key_t::alpha,
                            *h_alpha,
                            display_key_t::beta,
                            *h_beta,
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                            \
    template void testing_bsrmm_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_bsrmm<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_bsrmm_extra(const Arguments& arg) {}
