/* ************************************************************************
* Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
void testing_spsm_csr_bad_arg(const Arguments& arg)
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

    // SpSM structures
    rocsparse_local_spmat local_A(
        m, n, nnz, (void*)0x4, (void*)0x4, (void*)0x4, itype, jtype, base, compute_type);
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
void testing_spsm_csr(const Arguments& arg)
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

    // Allocate host memory for matrix
    host_vector<I> hcsr_row_ptr;
    host_vector<J> hcsr_col_ind;
    host_vector<T> hcsr_val;

    // Sample matrix
    I nnz_A;
    matrix_factory.init_csr(hcsr_row_ptr, hcsr_col_ind, hcsr_val, M, N, nnz_A, base);

    //
    // Scale values.
    //
    {
        const size_t       size = hcsr_val.size();
        floating_data_t<T> mx   = floating_data_t<T>(0);
        for(size_t i = 0; i < size; ++i)
        {
            mx = std::max(mx, std::abs(hcsr_val[i]));
        }
        mx = floating_data_t<T>(1.0) / mx;
        for(size_t i = 0; i < size; ++i)
        {
            hcsr_val[i] *= mx;
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

    // Non-squared matrices are not supported
    if(M != N)
    {
        return;
    }

    // Allocate host memory for vectors
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
    device_vector<I>       dcsr_row_ptr(M + 1);
    device_vector<J>       dcsr_col_ind(nnz_A);
    device_vector<T>       dcsr_val(nnz_A);
    device_dense_matrix<T> dB(nrowB, ncolB);
    device_dense_matrix<T> dC_1(nrowC, ncolC);
    device_dense_matrix<T> dC_2(nrowC, ncolC);
    device_vector<T>       dalpha(1);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(I) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(J) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(T) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * nrowB * ncolB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_1, hC_1, sizeof(T) * nrowC * ncolC, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_2, hC_2, sizeof(T) * nrowC * ncolC, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dalpha, &halpha, sizeof(T), hipMemcpyHostToDevice));

    // Create descriptors
    rocsparse_local_spmat A(
        M, N, nnz_A, dcsr_row_ptr, dcsr_col_ind, dcsr_val, itype, jtype, base, ttype);

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
                                         nullptr,
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
                                         nullptr,
                                         dbuffer));

    //
    // The buffer must be be non persistent, let's put garbage in it.
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

        // CPU csrsm
        J analysis_pivot = -1;
        J solve_pivot    = -1;
        host_csrsm<I, J, T>(M,
                            K,
                            nnz_A,
                            trans_A,
                            trans_C,
                            halpha,
                            hcsr_row_ptr,
                            hcsr_col_ind,
                            hcsr_val,
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
    template void testing_spsm_csr_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_spsm_csr<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

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

static void testing_spsm_csr_extra0(const Arguments& arg)
{
    static const bool         verbose = false;
    const rocsparse_operation trans_A = rocsparse_operation_none;
    rocsparse_operation       trans_B = rocsparse_operation_transpose;

    //     1 0 0
    // A = 0 2 0
    //     0 0 3
    const rocsparse_int    m   = 3;
    const rocsparse_int    n   = 2;
    const rocsparse_int    nnz = 3;
    host_csr_matrix<float> hA(m, m, nnz, rocsparse_index_base_zero);
    hA.ptr[0] = 0;
    hA.ptr[1] = 1;
    hA.ptr[2] = 2;
    hA.ptr[3] = 3;
    hA.ind[0] = 0;
    hA.ind[1] = 1;
    hA.ind[2] = 2;
    hA.val[0] = 1;
    hA.val[1] = 2;
    hA.val[2] = 4;

    device_csr_matrix<float> dA(hA);
    host_dense_matrix<float> hB(m, n);
    host_dense_matrix<float> hB_T(n, m);
    host_dense_matrix<float> hC(m, n);

    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            hC[hC.ld * j + i] = 777;
        }
    }

    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            hB[hB.ld * j + i] = static_cast<float>(j + 1);
        }
    }

    //     1 2
    // B = 1 2
    //     1 2
    if(verbose)
    {
        std::cout << "hB" << std::endl;
        for(size_t i = 0; i < m; ++i)
        {
            for(size_t j = 0; j < n; ++j)
                fprintf(stdout, " %8.5e", hB[j * m + i]);
            fprintf(stdout, "\n");
        }
    }
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            hB_T[n * j + i] = static_cast<float>(i + 1);
        }
    }

    if(verbose)
    {
        std::cout << "hB_T" << std::endl;
        for(size_t i = 0; i < n; ++i)
        {
            for(size_t j = 0; j < m; ++j)
                fprintf(stdout, " %8.5e", hB_T[j * n + i]);
            fprintf(stdout, "\n");
        }
    }
    // Scalar alpha
    float alpha = 1.0f;
    // Offload data to device
    device_dense_matrix<float> dB(hB), dB_T(hB_T), dC(hC);
    rocsparse_handle           handle;
    rocsparse_local_spmat      matA(dA);
    rocsparse_local_dnmat      matB(dB);
    rocsparse_local_dnmat      matB_T(dB_T);
    rocsparse_local_dnmat      matC(dC);
    CHECK_ROCSPARSE_ERROR(rocsparse_create_handle(&handle));
    rocsparse_datatype compute_type = rocsparse_datatype_f32_r;
    {
        // Call spsv to get buffer size
        size_t buffer_size;
        trans_B = rocsparse_operation_transpose;
        CHECK_ROCSPARSE_ERROR(rocsparse_spsm(handle,
                                             trans_A,
                                             trans_B,
                                             &alpha,
                                             matA,
                                             matB_T,
                                             matC,
                                             compute_type,
                                             rocsparse_spsm_alg_default,
                                             rocsparse_spsm_stage_buffer_size,
                                             &buffer_size,
                                             nullptr));
        void* temp_buffer;
        CHECK_HIP_ERROR(rocsparse_hipMalloc((void**)&temp_buffer, buffer_size));
        // Call spsv to perform analysis
        CHECK_ROCSPARSE_ERROR(rocsparse_spsm(handle,
                                             trans_A,
                                             trans_B,
                                             &alpha,
                                             matA,
                                             matB_T,
                                             matC,
                                             compute_type,
                                             rocsparse_spsm_alg_default,
                                             rocsparse_spsm_stage_preprocess,
                                             &buffer_size,
                                             temp_buffer));
        // Call spsv to perform computation
        CHECK_ROCSPARSE_ERROR(rocsparse_spsm(handle,
                                             trans_A,
                                             trans_B,
                                             &alpha,
                                             matA,
                                             matB_T,
                                             matC,
                                             compute_type,
                                             rocsparse_spsm_alg_default,
                                             rocsparse_spsm_stage_compute,
                                             &buffer_size,
                                             temp_buffer));
        CHECK_HIP_ERROR(rocsparse_hipFree(temp_buffer));
        // Copy result back to host
        hC.transfer_from(dC);

        if(verbose)
        {

            std::cout << "hC with B transpose" << std::endl;
            for(size_t i = 0; i < m; ++i)
            {
                for(size_t j = 0; j < n; ++j)
                {
                    fprintf(stdout, " %8.5e", hC[j * m + i]);
                }
                fprintf(stdout, "\n");
            }
        }
    }

    host_dense_matrix<float> hC2(hC);
    {
        // Call spsv to get buffer size
        size_t buffer_size;
        trans_B = rocsparse_operation_none;
        CHECK_ROCSPARSE_ERROR(rocsparse_spsm(handle,
                                             trans_A,
                                             trans_B,
                                             &alpha,
                                             matA,
                                             matB,
                                             matC,
                                             compute_type,
                                             rocsparse_spsm_alg_default,
                                             rocsparse_spsm_stage_buffer_size,
                                             &buffer_size,
                                             nullptr));
        void* temp_buffer;
        CHECK_HIP_ERROR(rocsparse_hipMalloc((void**)&temp_buffer, buffer_size));
        // Call spsv to perform analysis
        CHECK_ROCSPARSE_ERROR(rocsparse_spsm(handle,
                                             trans_A,
                                             trans_B,
                                             &alpha,
                                             matA,
                                             matB,
                                             matC,
                                             compute_type,
                                             rocsparse_spsm_alg_default,
                                             rocsparse_spsm_stage_preprocess,
                                             &buffer_size,
                                             temp_buffer));
        // Call spsv to perform computation
        CHECK_ROCSPARSE_ERROR(rocsparse_spsm(handle,
                                             trans_A,
                                             trans_B,
                                             &alpha,
                                             matA,
                                             matB,
                                             matC,
                                             compute_type,
                                             rocsparse_spsm_alg_default,
                                             rocsparse_spsm_stage_compute,
                                             &buffer_size,
                                             temp_buffer));
        CHECK_HIP_ERROR(rocsparse_hipFree(temp_buffer));
        // Copy result back to host
        hC.transfer_from(dC);
    }
    if(verbose)
    {

        std::cout << "hC with B none" << std::endl;
        for(size_t i = 0; i < m; ++i)
        {
            for(size_t j = 0; j < n; ++j)
            {
                fprintf(stdout, " %8.5e", hC[j * m + i]);
            }
            fprintf(stdout, "\n");
        }
    }
    hC.unit_check(hC2);
    // Clear rocSPARSE
    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_handle(handle));
}

static void
    spsm_csr_B_conjugate(const Arguments& arg, rocsparse_order order_B, rocsparse_order order_C)
{
    // This test verifies that the conjugate transpose is working correctly
    static const bool    verbose = false;
    rocsparse_operation  trans_A = rocsparse_operation_none;
    rocsparse_operation  trans_B = rocsparse_operation_conjugate_transpose;
    rocsparse_index_base base    = rocsparse_index_base_zero;

    rocsparse_spsm_alg  alg  = rocsparse_spsm_alg_default;
    rocsparse_fill_mode uplo = rocsparse_fill_mode_lower;
    rocsparse_diag_type diag = rocsparse_diag_type_non_unit;

    rocsparse_int m   = 3;
    rocsparse_int k   = 2;
    rocsparse_int ldb = (order_B == rocsparse_order_column) ? k : m;
    rocsparse_int ldc = (order_C == rocsparse_order_column) ? m : k;

    rocsparse_indextype itype        = get_indextype<rocsparse_int>();
    rocsparse_indextype jtype        = get_indextype<rocsparse_int>();
    rocsparse_datatype  compute_type = get_datatype<rocsparse_float_complex>();

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    rocsparse_float_complex halpha = rocsparse_float_complex(1, 0);

    //     1 0 0
    // A = 3 2 0
    //     4 2 -2
    rocsparse_int                        nnz_A        = 6;
    host_vector<rocsparse_int>           hcsr_row_ptr = {0, 1, 3, 6};
    host_vector<rocsparse_int>           hcsr_col_ind = {0, 0, 1, 0, 1, 2};
    host_vector<rocsparse_float_complex> hcsr_val     = {1, 3, 2, 4, 2, -2};

    // B = (1, 1)  (2, -1)  (0,  1)
    //     (1, 0)  (-1, 1)  (3, -2)
    const host_vector<rocsparse_float_complex> hB_row_order = {rocsparse_float_complex(1, 1),
                                                               rocsparse_float_complex(2, -1),
                                                               rocsparse_float_complex(0, 1),
                                                               rocsparse_float_complex(1, 0),
                                                               rocsparse_float_complex(-1, 1),
                                                               rocsparse_float_complex(3, -2)};
    const host_vector<rocsparse_float_complex> hB_col_order = {rocsparse_float_complex(1, 1),
                                                               rocsparse_float_complex(1, 0),
                                                               rocsparse_float_complex(2, -1),
                                                               rocsparse_float_complex(-1, 1),
                                                               rocsparse_float_complex(0, 1),
                                                               rocsparse_float_complex(3, -2)};

    // C = (1, 1)  (2, 2)
    //     (3, 3)  (4, 4)
    //     (5, 5)  (6, 6)
    const host_vector<rocsparse_float_complex> hC_row_order = {rocsparse_float_complex(1, 1),
                                                               rocsparse_float_complex(2, 2),
                                                               rocsparse_float_complex(3, 3),
                                                               rocsparse_float_complex(4, 4),
                                                               rocsparse_float_complex(5, 5),
                                                               rocsparse_float_complex(6, 6)};
    const host_vector<rocsparse_float_complex> hC_col_order = {rocsparse_float_complex(1, 1),
                                                               rocsparse_float_complex(3, 3),
                                                               rocsparse_float_complex(5, 5),
                                                               rocsparse_float_complex(2, 2),
                                                               rocsparse_float_complex(4, 4),
                                                               rocsparse_float_complex(6, 6)};

    host_vector<rocsparse_float_complex> hB(k * m);
    host_vector<rocsparse_float_complex> hC(m * k);

    for(rocsparse_int i = 0; i < k * m; i++)
    {
        hB[i] = (order_B == rocsparse_order_column) ? hB_col_order[i] : hB_row_order[i];
    }

    for(rocsparse_int i = 0; i < m * k; i++)
    {
        hC[i] = (order_C == rocsparse_order_column) ? hC_col_order[i] : hC_row_order[i];
    }

    if(verbose)
    {
        std::cout << "A" << std::endl;
        for(rocsparse_int i = 0; i < m; i++)
        {
            rocsparse_int                        start = hcsr_row_ptr[i] - base;
            rocsparse_int                        end   = hcsr_row_ptr[i + 1] - base;
            std::vector<rocsparse_float_complex> temp(m, 0);
            for(rocsparse_int j = start; j < end; j++)
            {
                temp[hcsr_col_ind[j] - base] = hcsr_val[j];
            }

            for(rocsparse_int j = 0; j < m; j++)
            {
                std::cout << temp[j] << " ";
            }
            std::cout << "" << std::endl;
        }
        std::cout << "" << std::endl;

        std::cout << "B" << std::endl;
        if(order_B == rocsparse_order_column)
        {
            for(rocsparse_int i = 0; i < m; i++)
            {
                for(rocsparse_int j = 0; j < k; j++)
                {
                    std::cout << hB[i + ldb * j] << " ";
                }
                std::cout << "" << std::endl;
            }
        }
        else
        {
            for(rocsparse_int i = 0; i < k; i++)
            {
                for(rocsparse_int j = 0; j < m; j++)
                {
                    std::cout << hB[ldb * i + j] << " ";
                }
                std::cout << "" << std::endl;
            }
        }
        std::cout << "" << std::endl;

        std::cout << "C" << std::endl;
        if(order_C == rocsparse_order_column)
        {
            for(rocsparse_int i = 0; i < m; i++)
            {
                for(rocsparse_int j = 0; j < k; j++)
                {
                    std::cout << hC[i + ldc * j] << " ";
                }
                std::cout << "" << std::endl;
            }
        }
        else
        {
            for(rocsparse_int i = 0; i < m; i++)
            {
                for(rocsparse_int j = 0; j < k; j++)
                {
                    std::cout << hC[ldc * i + j] << " ";
                }
                std::cout << "" << std::endl;
            }
        }
        std::cout << "" << std::endl;
    }

    device_vector<rocsparse_int>           dcsr_row_ptr(m + 1);
    device_vector<rocsparse_int>           dcsr_col_ind(nnz_A);
    device_vector<rocsparse_float_complex> dcsr_val(nnz_A);
    device_vector<rocsparse_float_complex> dB(k * m);
    device_vector<rocsparse_float_complex> dC(m * k);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_col_ind, hcsr_col_ind.data(), sizeof(rocsparse_int) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_val, hcsr_val.data(), sizeof(rocsparse_float_complex) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dB, hB, sizeof(rocsparse_float_complex) * k * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dC, hC, sizeof(rocsparse_float_complex) * m * k, hipMemcpyHostToDevice));

    // Create descriptors
    rocsparse_local_spmat A(
        m, m, nnz_A, dcsr_row_ptr, dcsr_col_ind, dcsr_val, itype, jtype, base, compute_type);

    rocsparse_local_dnmat B(k, m, ldb, dB, compute_type, order_B);
    rocsparse_local_dnmat C(m, k, ldc, dC, compute_type, order_C);

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
                                         C,
                                         compute_type,
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
                                         C,
                                         compute_type,
                                         alg,
                                         rocsparse_spsm_stage_preprocess,
                                         nullptr,
                                         dbuffer));

    CHECK_ROCSPARSE_ERROR(testing::rocsparse_spsm(handle,
                                                  trans_A,
                                                  trans_B,
                                                  &halpha,
                                                  A,
                                                  B,
                                                  C,
                                                  compute_type,
                                                  alg,
                                                  rocsparse_spsm_stage_compute,
                                                  &buffer_size,
                                                  dbuffer));

    CHECK_HIP_ERROR(hipDeviceSynchronize());

    // Copy output to host
    CHECK_HIP_ERROR(
        hipMemcpy(hC, dC, sizeof(rocsparse_float_complex) * m * k, hipMemcpyDeviceToHost));

    if(verbose)
    {
        std::cout << "C" << std::endl;
        if(order_C == rocsparse_order_column)
        {
            for(rocsparse_int i = 0; i < m; i++)
            {
                for(rocsparse_int j = 0; j < k; j++)
                {
                    std::cout << hC[i + ldc * j] << " ";
                }
                std::cout << "" << std::endl;
            }
        }
        else
        {
            for(rocsparse_int i = 0; i < m; i++)
            {
                for(rocsparse_int j = 0; j < k; j++)
                {
                    std::cout << hC[ldc * i + j] << " ";
                }
                std::cout << "" << std::endl;
            }
        }
        std::cout << "" << std::endl;
    }

    // Manually computed correct solution
    // C = (1, -1)    (1, 0)
    //     (-0.5, 2)  (-2, -0.5)
    //     (1.5, 0.5) (-1.5, -1.5)
    const host_vector<rocsparse_float_complex> hC_solution_row_order
        = {rocsparse_float_complex(1, -1),
           rocsparse_float_complex(1, 0),
           rocsparse_float_complex(-0.5, 2),
           rocsparse_float_complex(-2, -0.5),
           rocsparse_float_complex(1.5, 0.5),
           rocsparse_float_complex(-1.5, -1.5)};
    const host_vector<rocsparse_float_complex> hC_solution_col_order
        = {rocsparse_float_complex(1, -1),
           rocsparse_float_complex(-0.5, 2),
           rocsparse_float_complex(1.5, 0.5),
           rocsparse_float_complex(1, 0),
           rocsparse_float_complex(-2, -0.5),
           rocsparse_float_complex(-1.5, -1.5)};

    host_dense_matrix<rocsparse_float_complex> hC_gpu(m, k);
    for(rocsparse_int i = 0; i < m * k; i++)
    {
        hC_gpu[i] = hC[i];
    }
    host_dense_matrix<rocsparse_float_complex> hC_cpu(m, k);

    for(rocsparse_int i = 0; i < m * k; i++)
    {
        hC_cpu[i] = (order_C == rocsparse_order_column) ? hC_solution_col_order[i]
                                                        : hC_solution_row_order[i];
    }

    hC_cpu.unit_check(hC_gpu);

    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

static void testing_spsm_csr_extra1(const Arguments& arg)
{
    spsm_csr_B_conjugate(arg, rocsparse_order_row, rocsparse_order_row);
    spsm_csr_B_conjugate(arg, rocsparse_order_row, rocsparse_order_column);
    spsm_csr_B_conjugate(arg, rocsparse_order_column, rocsparse_order_row);
    spsm_csr_B_conjugate(arg, rocsparse_order_column, rocsparse_order_column);
}

void testing_spsm_csr_extra(const Arguments& arg)
{
    testing_spsm_csr_extra0(arg);
    testing_spsm_csr_extra1(arg);
}
