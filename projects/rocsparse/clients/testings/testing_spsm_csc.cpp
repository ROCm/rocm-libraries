/* ************************************************************************
* Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

template <typename I, typename J, typename T>
void testing_spsm_csc_bad_arg(const Arguments& arg)
{
    J        m           = 100;
    J        n           = 100;
    J        k           = 16;
    I        nnz         = 100;
    const T  local_alpha = 0.6;
    const T* alpha       = &local_alpha;

    rocsparse_operation  trans_A = rocsparse_operation_none;
    rocsparse_operation  trans_B = rocsparse_operation_none;
    rocsparse_index_base base    = rocsparse_index_base_zero;
    rocsparse_spsm_alg   alg     = rocsparse_spsm_alg_default;

    // Index and data type
    rocsparse_indextype itype        = get_indextype<I>();
    rocsparse_indextype jtype        = get_indextype<J>();
    rocsparse_datatype  compute_type = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // SpSM structures — CSC format
    static const bool     csc_format = true;
    rocsparse_local_spmat local_A(m,
                                  n,
                                  nnz,
                                  (void*)0x4,
                                  (void*)0x4,
                                  (void*)0x4,
                                  itype,
                                  jtype,
                                  base,
                                  compute_type,
                                  csc_format);
    rocsparse_local_dnmat local_B(m, k, m, (void*)0x4, compute_type, rocsparse_order_column);
    rocsparse_local_dnmat local_C(m, k, m, (void*)0x4, compute_type, rocsparse_order_column);

    rocsparse_handle      handle = local_handle;
    rocsparse_spmat_descr matA   = local_A;
    rocsparse_dnmat_descr matB   = local_B;
    rocsparse_dnmat_descr matC   = local_C;

    size_t  local_buffer_size;
    size_t* buffer_size = &local_buffer_size;
    void*   temp_buffer = (void*)0x4;

#define PARAMS_BUFFER_SIZE                                                                    \
    handle, trans_A, trans_B, alpha, matA, matB, matC, compute_type, alg, stage, buffer_size, \
        temp_buffer

#define PARAMS_ANALYSIS                                                                       \
    handle, trans_A, trans_B, alpha, matA, matB, matC, compute_type, alg, stage, buffer_size, \
        temp_buffer

#define PARAMS_SOLVE                                                                          \
    handle, trans_A, trans_B, alpha, matA, matB, matC, compute_type, alg, stage, buffer_size, \
        temp_buffer

    {
        static constexpr int       nargs_to_exclude                  = 1;
        const int                  args_to_exclude[nargs_to_exclude] = {11};
        const rocsparse_spsm_stage stage = rocsparse_spsm_stage_buffer_size;
        select_bad_arg_analysis(
            rocsparse_spsm, nargs_to_exclude, args_to_exclude, PARAMS_BUFFER_SIZE);
    }

    {
        static constexpr int nargs_to_exclude                  = 2;
        const int            args_to_exclude[nargs_to_exclude] = {10, 11};

        const rocsparse_spsm_stage stage = rocsparse_spsm_stage_preprocess;
        select_bad_arg_analysis(rocsparse_spsm, nargs_to_exclude, args_to_exclude, PARAMS_ANALYSIS);
    }

    {
        static constexpr int nargs_to_exclude                  = 2;
        const int            args_to_exclude[nargs_to_exclude] = {10, 11};

        const rocsparse_spsm_stage stage = rocsparse_spsm_stage_compute;
        select_bad_arg_analysis(rocsparse_spsm, nargs_to_exclude, args_to_exclude, PARAMS_SOLVE);
    }

#undef PARAMS_BUFFER_SIZE
#undef PARAMS_ANALYSIS
#undef PARAMS_SOLVE
}

template <typename I, typename J, typename T>
void testing_spsm_csc(const Arguments& arg)
{
    J                    M               = arg.M;
    J                    N               = arg.N;
    J                    K               = arg.K;
    rocsparse_operation  trans_A         = arg.transA;
    rocsparse_operation  trans_B         = arg.transB;
    rocsparse_index_base base            = arg.baseA;
    rocsparse_spsm_alg   alg             = arg.spsm_alg;
    rocsparse_diag_type  diag            = arg.diag;
    rocsparse_fill_mode  uplo            = arg.uplo;
    rocsparse_order      order_B         = arg.orderB;
    rocsparse_order      order_C         = arg.orderC;
    rocsparse_int        ld_multiplier_B = arg.ld_multiplier_B;
    rocsparse_int        ld_multiplier_C = arg.ld_multiplier_C;

    // In the generic routines, C is always non-transposed (op(A) * C = op(B))
    rocsparse_operation trans_C = rocsparse_operation_none;

    T halpha = arg.get_alpha<T>();

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_indextype jtype = get_indextype<J>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    rocsparse_matrix_factory<T, I, J> matrix_factory(arg);

    // Allocate host memory for CSC matrix
    host_vector<I> hcsc_col_ptr;
    host_vector<J> hcsc_row_ind;
    host_vector<T> hcsc_val;

    // Sample matrix in CSC format
    I nnz_A;
    matrix_factory.init_csc(hcsc_col_ptr, hcsc_row_ind, hcsc_val, M, N, nnz_A, base);

    // Non-squared matrices are not supported
    if(M != N)
    {
        return;
    }

    //
    // Scale values to improve conditioning.
    //
    {
        const size_t       size = hcsc_val.size();
        floating_data_t<T> mx   = floating_data_t<T>(0);
        for(size_t i = 0; i < size; ++i)
        {
            mx = std::max(mx, std::abs(hcsc_val[i]));
        }
        mx = (mx != floating_data_t<T>(0)) ? floating_data_t<T>(1.0) / mx : floating_data_t<T>(1.0);
        for(size_t i = 0; i < size; ++i)
        {
            hcsc_val[i] *= mx;
        }
    }

    J B_m = (trans_B == rocsparse_operation_none) ? M : K;
    J B_n = (trans_B == rocsparse_operation_none) ? K : M;

    J C_m = M;
    J C_n = K;

    int64_t ldb = (order_B == rocsparse_order_column)
                      ? ((trans_B == rocsparse_operation_none) ? (int64_t(ld_multiplier_B) * M)
                                                               : (int64_t(ld_multiplier_B) * K))
                      : ((trans_B == rocsparse_operation_none) ? (int64_t(ld_multiplier_B) * K)
                                                               : (int64_t(ld_multiplier_B) * M));

    int64_t ldc = (order_C == rocsparse_order_column) ? (int64_t(ld_multiplier_C) * M)
                                                      : (int64_t(ld_multiplier_C) * K);

    ldb = std::max(int64_t(1), ldb);
    ldc = std::max(int64_t(1), ldc);

    int64_t nrowB = (order_B == rocsparse_order_column) ? ldb : B_m;
    int64_t ncolB = (order_B == rocsparse_order_column) ? B_n : ldb;
    int64_t nrowC = (order_C == rocsparse_order_column) ? ldc : C_m;
    int64_t ncolC = (order_C == rocsparse_order_column) ? C_n : ldc;

    // Allocate host memory for dense matrices
    host_dense_matrix<T> htemp(B_m, B_n);
    host_dense_matrix<T> hB(nrowB, ncolB);
    host_dense_matrix<T> hC_1(nrowC, ncolC);
    host_dense_matrix<T> hC_2(nrowC, ncolC);
    host_dense_matrix<T> hC_gold(nrowC, ncolC);

    rocsparse_matrix_utils::init(htemp);

    // Copy B to C
    if(order_B == rocsparse_order_column)
    {
        for(J j = 0; j < B_n; j++)
        {
            for(J i = 0; i < B_m; i++)
            {
                hB[i + ldb * j] = htemp[i + B_m * j];
            }
        }

        if(trans_B == rocsparse_operation_none)
        {
            if(order_C == rocsparse_order_column)
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        hC_1[i + ldc * j] = hB[i + ldb * j];
                    }
                }
            }
            else
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        hC_1[i * ldc + j] = hB[i + ldb * j];
                    }
                }
            }
        }
        else
        {
            if(order_C == rocsparse_order_column)
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        hC_1[i * ldc + j] = hB[i + ldb * j];
                    }
                }
            }
            else
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        hC_1[i + ldc * j] = hB[i + ldb * j];
                    }
                }
            }
        }
    }
    else
    {
        for(J i = 0; i < B_m; i++)
        {
            for(J j = 0; j < B_n; j++)
            {
                hB[ldb * i + j] = htemp[B_n * i + j];
            }
        }

        if(trans_B == rocsparse_operation_none)
        {
            if(order_C == rocsparse_order_column)
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        hC_1[i + ldc * j] = hB[ldb * i + j];
                    }
                }
            }
            else
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        hC_1[i * ldc + j] = hB[ldb * i + j];
                    }
                }
            }
        }
        else
        {
            if(order_C == rocsparse_order_column)
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        hC_1[i * ldc + j] = hB[ldb * i + j];
                    }
                }
            }
            else
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        hC_1[i + ldc * j] = hB[ldb * i + j];
                    }
                }
            }
        }
    }

    hC_2    = hC_1;
    hC_gold = hC_1;

    if(trans_B == rocsparse_operation_conjugate_transpose)
    {
        if(order_C == rocsparse_order_column)
        {
            for(J j = 0; j < C_n; j++)
            {
                for(J i = 0; i < C_m; i++)
                {
                    hC_gold[i + ldc * j] = rocsparse_conj<T>(hC_gold[i + ldc * j]);
                }
            }
        }
        else
        {
            for(J i = 0; i < C_m; i++)
            {
                for(J j = 0; j < C_n; j++)
                {
                    hC_gold[ldc * i + j] = rocsparse_conj<T>(hC_gold[ldc * i + j]);
                }
            }
        }
    }

    // Allocate device memory
    // CSC: col_ptr has length N+1, row_ind has length nnz
    device_vector<I>       dcsc_col_ptr(N + 1);
    device_vector<J>       dcsc_row_ind(nnz_A);
    device_vector<T>       dcsc_val(nnz_A);
    device_dense_matrix<T> dB(nrowB, ncolB);
    device_dense_matrix<T> dC_1(nrowC, ncolC);
    device_dense_matrix<T> dC_2(nrowC, ncolC);
    device_vector<T>       dalpha(1);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcsc_col_ptr, hcsc_col_ptr.data(), sizeof(I) * (N + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsc_row_ind, hcsc_row_ind.data(), sizeof(J) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsc_val, hcsc_val.data(), sizeof(T) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * nrowB * ncolB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_1, hC_1, sizeof(T) * nrowC * ncolC, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_2, hC_2, sizeof(T) * nrowC * ncolC, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dalpha, &halpha, sizeof(T), hipMemcpyHostToDevice));

    // Create CSC descriptor
    static const bool     csc_format = true;
    rocsparse_local_spmat A(
        M, N, nnz_A, dcsc_col_ptr, dcsc_row_ind, dcsc_val, itype, jtype, base, ttype, csc_format);
    rocsparse_local_dnmat B(B_m, B_n, ldb, dB, ttype, order_B);
    rocsparse_local_dnmat C1(C_m, C_n, ldc, dC_1, ttype, order_C);
    rocsparse_local_dnmat C2(C_m, C_n, ldc, dC_2, ttype, order_C);

    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &uplo, sizeof(uplo)));

    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_diag_type, &diag, sizeof(diag)));

    // Query SpSM buffer
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_spsm(handle,
                                         trans_A,
                                         trans_B,
                                         &halpha,
                                         A,
                                         B,
                                         C1,
                                         ttype,
                                         alg,
                                         rocsparse_spsm_stage_buffer_size,
                                         &buffer_size,
                                         nullptr));

    // Allocate buffer
    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    // Perform analysis on host
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_spsm(handle,
                                         trans_A,
                                         trans_B,
                                         &halpha,
                                         A,
                                         B,
                                         C1,
                                         ttype,
                                         alg,
                                         rocsparse_spsm_stage_preprocess,
                                         &buffer_size,
                                         dbuffer));

    // Perform analysis on device
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
    CHECK_ROCSPARSE_ERROR(rocsparse_spsm(handle,
                                         trans_A,
                                         trans_B,
                                         dalpha,
                                         A,
                                         B,
                                         C2,
                                         ttype,
                                         alg,
                                         rocsparse_spsm_stage_preprocess,
                                         &buffer_size,
                                         dbuffer));

    //
    // The buffer must be non persistent, let's put garbage in it.
    //
    CHECK_HIP_ERROR(hipMemset(dbuffer, 255 - 1, buffer_size));

    if(arg.unit_check)
    {
        // Solve on host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_spsm(handle,
                                                      trans_A,
                                                      trans_B,
                                                      &halpha,
                                                      A,
                                                      B,
                                                      C1,
                                                      ttype,
                                                      alg,
                                                      rocsparse_spsm_stage_compute,
                                                      &buffer_size,
                                                      dbuffer));

        // Solve on device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_spsm(handle,
                                                      trans_A,
                                                      trans_B,
                                                      dalpha,
                                                      A,
                                                      B,
                                                      C2,
                                                      ttype,
                                                      alg,
                                                      rocsparse_spsm_stage_compute,
                                                      &buffer_size,
                                                      dbuffer));

        CHECK_HIP_ERROR(hipDeviceSynchronize());

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("dC_1", dC_1);
        }
        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("dC_2", dC_2);
        }

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hC_1, dC_1, sizeof(T) * nrowC * ncolC, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hC_2, dC_2, sizeof(T) * nrowC * ncolC, hipMemcpyDeviceToHost));

        // CPU gold check: host_cscsm wraps host_csrsm, handling the CSC->CSR
        // mapping (transpose operation on A, flipped fill mode and, for the
        // conjugate transpose, the value conjugation) internally. The operation
        // on the right-hand side (trans_C) is passed through unchanged.
        J analysis_pivot = -1;
        J solve_pivot    = -1;
        host_cscsm<I, J, T>(M,
                            K,
                            nnz_A,
                            trans_A,
                            trans_C,
                            halpha,
                            hcsc_col_ptr.data(),
                            hcsc_row_ind.data(),
                            hcsc_val.data(),
                            hC_gold,
                            ldc,
                            order_C,
                            diag,
                            uplo,
                            base,
                            &analysis_pivot,
                            &solve_pivot);

        if(analysis_pivot == -1 && solve_pivot == -1)
        {
            hC_gold.near_check(hC_1);
            hC_gold.near_check(hC_2);
        }
    }

    if(arg.timing)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_spsm,
                                                                      handle,
                                                                      trans_A,
                                                                      trans_B,
                                                                      &halpha,
                                                                      A,
                                                                      B,
                                                                      C1,
                                                                      ttype,
                                                                      alg,
                                                                      rocsparse_spsm_stage_compute,
                                                                      &buffer_size,
                                                                      dbuffer);

        double gflop_count = spsv_gflop_count(M, nnz_A, diag) * K;
        double gpu_gflops  = get_gpu_gflops(gpu_time_used, gflop_count);

        double gbyte_count = csrsv_gbyte_count<T>(M, nnz_A) * K;
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::nnz_A,
                            nnz_A,
                            display_key_t::nrhs,
                            K,
                            display_key_t::alpha,
                            halpha,
                            display_key_t::algorithm,
                            rocsparse_spsmalg2string(alg),
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                               \
    template void testing_spsm_csc_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_spsm_csc<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);

void testing_spsm_csc_extra(const Arguments& arg) {}
