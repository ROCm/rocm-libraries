/* ************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

template <typename I, typename T>
void testing_sptrsm_coo_bad_arg(const Arguments& arg)
{
    //
    // Bad args of sptrsm is already tested in testing_sptrsm_csr_bad_arg
    //
}

template <typename I, typename T>
void testing_sptrsm_coo(const Arguments& arg)
{
    I M = arg.M;
    I N = arg.N;
    if(M != N)
    {
        return;
    }
    const I                    K               = arg.K;
    const rocsparse_operation  trans_A         = arg.transA;
    const rocsparse_operation  trans_B         = arg.transB;
    const rocsparse_index_base base            = arg.baseA;
    const rocsparse_spsm_alg   alg             = arg.spsm_alg;
    const rocsparse_diag_type  diag            = arg.diag;
    const rocsparse_fill_mode  uplo            = arg.uplo;
    const rocsparse_order      order_B         = arg.orderB;
    const rocsparse_order      order_C         = arg.orderC;
    const rocsparse_int        ld_multiplier_B = arg.ld_multiplier_B;
    const rocsparse_int        ld_multiplier_C = arg.ld_multiplier_C;

    // In the generic routines, C is always non-transposed (op(A) * C = op(B))
    const rocsparse_operation trans_C = rocsparse_operation_none;

    //    const host_scalar<T> halpha(arg.get_alpha<T>());
    const host_scalar<T> halpha(1);

    // Index and data type
    const rocsparse_datatype ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle            handle(arg);
    rocsparse_matrix_factory<T, I, I> matrix_factory(arg);

    // Sample matrix
    // Allocate host memory for matrix
    host_coo_matrix<T, I> hA;

    // Sample matrix
    matrix_factory.init_coo(hA, M, N, base);

    // Non-squared matrices are not supported
    if(M != N)
    {
        return;
    }

    const I nnz_A = hA.nnz;
    //
    // Scale values.
    //
    {
        const size_t       size = nnz_A;
        floating_data_t<T> mx   = floating_data_t<T>(0);
        for(size_t i = 0; i < size; ++i)
        {
            mx = std::max(mx, std::abs(hA.val[i]));
        }
        mx = (mx != floating_data_t<T>(0)) ? floating_data_t<T>(1.0) / mx : floating_data_t<T>(1.0);
        for(size_t i = 0; i < size; ++i)
        {
            hA.val[i] *= mx;
        }
    }

    const I B_m = (trans_B == rocsparse_operation_none) ? M : K;
    const I B_n = (trans_B == rocsparse_operation_none) ? K : M;
    const I C_m = M;
    const I C_n = K;

    const int64_t ldb = std::max(
        int64_t(1),
        ((order_B == rocsparse_order_column)
             ? ((trans_B == rocsparse_operation_none) ? (int64_t(ld_multiplier_B) * M)
                                                      : (int64_t(ld_multiplier_B) * K))
             : ((trans_B == rocsparse_operation_none) ? (int64_t(ld_multiplier_B) * K)
                                                      : (int64_t(ld_multiplier_B) * M))));

    const int64_t ldc
        = std::max(int64_t(1),
                   ((order_C == rocsparse_order_column) ? (int64_t(ld_multiplier_C) * M)
                                                        : (int64_t(ld_multiplier_C) * K)));

    const int64_t nrowB = (order_B == rocsparse_order_column) ? ldb : B_m;
    const int64_t ncolB = (order_B == rocsparse_order_column) ? B_n : ldb;
    const int64_t nrowC = (order_C == rocsparse_order_column) ? ldc : C_m;
    const int64_t ncolC = (order_C == rocsparse_order_column) ? C_n : ldc;

    // Allocate host memory for vectors
    host_dense_matrix<T> hB(nrowB, ncolB);

    {
        host_dense_matrix<T> htemp(B_m, B_n);
        rocsparse_matrix_utils::init(htemp);

        if(order_B == rocsparse_order_column)
        {
            for(I j = 0; j < B_n; j++)
            {
                for(I i = 0; i < B_m; i++)
                {
                    hB[i + ldb * j] = htemp[i + B_m * j];
                }
            }
        }
        else
        {
            for(I i = 0; i < B_m; i++)
            {
                for(I j = 0; j < B_n; j++)
                {
                    hB[ldb * i + j] = htemp[B_n * i + j];
                }
            }
        }
    }

    host_dense_matrix<T> hC(nrowC, ncolC);
    memset(hC, 0, sizeof(T) * nrowC * ncolC);

    // Copy B to C
    if(order_B == rocsparse_order_column)
    {
        if(trans_B == rocsparse_operation_none)
        {
            if(order_C == rocsparse_order_column)
            {
                for(I j = 0; j < B_n; j++)
                {
                    for(I i = 0; i < B_m; i++)
                    {
                        hC[i + ldc * j] = hB[i + ldb * j];
                    }
                }
            }
            else
            {
                for(I j = 0; j < B_n; j++)
                {
                    for(I i = 0; i < B_m; i++)
                    {
                        hC[i * ldc + j] = hB[i + ldb * j];
                    }
                }
            }
        }
        else
        {
            if(order_C == rocsparse_order_column)
            {
                for(I j = 0; j < B_n; j++)
                {
                    for(I i = 0; i < B_m; i++)
                    {
                        hC[i * ldc + j] = hB[i + ldb * j];
                    }
                }
            }
            else
            {
                for(I j = 0; j < B_n; j++)
                {
                    for(I i = 0; i < B_m; i++)
                    {
                        hC[i + ldc * j] = hB[i + ldb * j];
                    }
                }
            }
        }
    }
    else
    {
        if(trans_B == rocsparse_operation_none)
        {
            if(order_C == rocsparse_order_column)
            {
                for(I j = 0; j < B_n; j++)
                {
                    for(I i = 0; i < B_m; i++)
                    {
                        hC[i + ldc * j] = hB[ldb * i + j];
                    }
                }
            }
            else
            {
                for(I j = 0; j < B_n; j++)
                {
                    for(I i = 0; i < B_m; i++)
                    {
                        hC[i * ldc + j] = hB[ldb * i + j];
                    }
                }
            }
        }
        else
        {
            if(order_C == rocsparse_order_column)
            {
                for(I j = 0; j < B_n; j++)
                {
                    for(I i = 0; i < B_m; i++)
                    {
                        hC[i * ldc + j] = hB[ldb * i + j];
                    }
                }
            }
            else
            {
                for(I j = 0; j < B_n; j++)
                {
                    for(I i = 0; i < B_m; i++)
                    {
                        hC[i + ldc * j] = hB[ldb * i + j];
                    }
                }
            }
        }
    }

    if(trans_B == rocsparse_operation_conjugate_transpose)
    {
        if(order_C == rocsparse_order_column)
        {
            for(I j = 0; j < C_n; j++)
            {
                for(I i = 0; i < C_m; i++)
                {
                    hC[i + ldc * j] = rocsparse_conj<T>(hC[i + ldc * j]);
                }
            }
        }
        else
        {
            for(I i = 0; i < C_m; i++)
            {
                for(I j = 0; j < C_n; j++)
                {
                    hC[ldc * i + j] = rocsparse_conj<T>(hC[ldc * i + j]);
                }
            }
        }
    }

    // Allocate device memory
    device_coo_matrix<T, I> dA(hA);
    device_dense_matrix<T>  dB(hB);
    device_dense_matrix<T>  dC(nrowC, ncolC);
    CHECK_HIP_ERROR(hipMemset(dC, 0, sizeof(T) * nrowC * ncolC));

    device_scalar<T> dalpha(halpha);

    // Create descriptors
    rocsparse_local_spmat A(dA);
    rocsparse_local_dnmat B(B_m, B_n, ldb, dB, ttype, order_B);
    rocsparse_local_dnmat C(C_m, C_n, ldc, dC, ttype, order_C);
    rocsparse_error*      p_error = nullptr;

    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &uplo, sizeof(uplo)));

    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_diag_type, &diag, sizeof(diag)));

    rocsparse_sptrsm_descr sptrsm_descr;
    CHECK_ROCSPARSE_ERROR(rocsparse_create_sptrsm_descr(&sptrsm_descr));

    //
    // Set algorithm.
    //
    {
        const rocsparse_sptrsm_alg alg = rocsparse_sptrsm_alg_default;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(
            handle, sptrsm_descr, rocsparse_sptrsm_input_alg, &alg, sizeof(alg), p_error));
    }

    //
    // Set operation A
    //
    {
        const rocsparse_operation operation_A = trans_A;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(handle,
                                                         sptrsm_descr,
                                                         rocsparse_sptrsm_input_operation_A,
                                                         &operation_A,
                                                         sizeof(operation_A),
                                                         p_error));
    }

    //
    // Set operation B
    //
    {
        const rocsparse_operation operation_X = trans_B;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(handle,
                                                         sptrsm_descr,
                                                         rocsparse_sptrsm_input_operation_X,
                                                         &operation_X,
                                                         sizeof(operation_X),
                                                         p_error));
    }

    //
    // Set scalar datatype.
    //
    {
        const rocsparse_datatype scalar_datatype = ttype;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(handle,
                                                         sptrsm_descr,
                                                         rocsparse_sptrsm_input_scalar_datatype,
                                                         &scalar_datatype,
                                                         sizeof(scalar_datatype),
                                                         p_error));
    }
    {
        const rocsparse_datatype compute_datatype = ttype;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(handle,
                                                         sptrsm_descr,
                                                         rocsparse_sptrsm_input_compute_datatype,
                                                         &compute_datatype,
                                                         sizeof(compute_datatype),
                                                         p_error));
    }

    {
        const rocsparse_analysis_policy apol = arg.apol;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(handle,
                                                         sptrsm_descr,
                                                         rocsparse_sptrsm_input_analysis_policy,
                                                         &apol,
                                                         sizeof(apol),
                                                         p_error));
    }

    {
        size_t buffer_size_in_bytes;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_buffer_size(handle,
                                                           sptrsm_descr,
                                                           A,
                                                           B,
                                                           C,
                                                           rocsparse_sptrsm_stage_analysis,
                                                           &buffer_size_in_bytes,
                                                           p_error));

        void* buffer;
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size_in_bytes));

        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm(handle,
                                               sptrsm_descr,
                                               A,
                                               B,
                                               C,
                                               rocsparse_sptrsm_stage_analysis,
                                               buffer_size_in_bytes,
                                               buffer,
                                               p_error));
        CHECK_HIP_ERROR(rocsparse_hipFree(buffer));
    }

    if(arg.unit_check)
    {
        // CPU
        I analysis_pivot = -1;
        I solve_pivot    = -1;

        host_coosm<I, T>(M,
                         K,
                         nnz_A,
                         trans_A,
                         trans_C,
                         halpha[0],
                         hA.row_ind,
                         hA.col_ind,
                         hA.val,
                         hC,
                         ldc,
                         order_C,
                         diag,
                         uplo,
                         base,
                         &analysis_pivot,
                         &solve_pivot);

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(handle,
                                                         sptrsm_descr,
                                                         rocsparse_sptrsm_input_scalar_alpha,
                                                         halpha,
                                                         sizeof(const T*),
                                                         p_error));

        size_t buffer_size_in_bytes;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_buffer_size(handle,
                                                           sptrsm_descr,
                                                           A,
                                                           B,
                                                           C,
                                                           rocsparse_sptrsm_stage_compute,
                                                           &buffer_size_in_bytes,
                                                           p_error));

        void* buffer;
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size_in_bytes));

        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm(handle,
                                               sptrsm_descr,
                                               A,
                                               B,
                                               C,
                                               rocsparse_sptrsm_stage_compute,
                                               buffer_size_in_bytes,
                                               buffer,
                                               p_error));
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("dC_host", dC);
        }

        if(analysis_pivot == -1 && solve_pivot == -1)
        {
            hC.near_check(dC);
        }

        //
        // Reset.
        //
        CHECK_HIP_ERROR(hipMemset(dC, 0, sizeof(T) * nrowC * ncolC));

        // Solve on device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(handle,
                                                         sptrsm_descr,
                                                         rocsparse_sptrsm_input_scalar_alpha,
                                                         dalpha,
                                                         sizeof(const T*),
                                                         p_error));

        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm(handle,
                                               sptrsm_descr,
                                               A,
                                               B,
                                               C,
                                               rocsparse_sptrsm_stage_compute,
                                               buffer_size_in_bytes,
                                               buffer,
                                               p_error));
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("dC_device", dC);
        }

        if(analysis_pivot == -1 && solve_pivot == -1)
        {
            hC.near_check(dC);
        }
        CHECK_HIP_ERROR(rocsparse_hipFree(buffer));
    }

    if(arg.timing)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(handle,
                                                         sptrsm_descr,
                                                         rocsparse_sptrsm_input_scalar_alpha,
                                                         halpha,
                                                         sizeof(const T*),
                                                         p_error));

        size_t buffer_size_in_bytes;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_buffer_size(handle,
                                                           sptrsm_descr,
                                                           A,
                                                           B,
                                                           C,
                                                           rocsparse_sptrsm_stage_compute,
                                                           &buffer_size_in_bytes,
                                                           p_error));

        void* buffer;
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size_in_bytes));

        const double gpu_time_used
            = rocsparse_clients::run_benchmark(arg,
                                               rocsparse_sptrsm,
                                               handle,
                                               sptrsm_descr,
                                               A,
                                               B,
                                               C,
                                               rocsparse_sptrsm_stage_compute,
                                               buffer_size_in_bytes,
                                               buffer,
                                               p_error);

        CHECK_HIP_ERROR(rocsparse_hipFree(buffer));

        const double gflop_count = spsv_gflop_count(M, nnz_A, diag) * K;
        const double gpu_gflops  = get_gpu_gflops(gpu_time_used, gflop_count);

        const double gbyte_count = csrsv_gbyte_count<T>(M, nnz_A) * K;
        const double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

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
    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_sptrsm_descr(sptrsm_descr));
}

#define INSTANTIATE(ITYPE, TTYPE)                                                 \
    template void testing_sptrsm_coo_bad_arg<ITYPE, TTYPE>(const Arguments& arg); \
    template void testing_sptrsm_coo<ITYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
void testing_sptrsm_coo_extra(const Arguments& arg) {}
