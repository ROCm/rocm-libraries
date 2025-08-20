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

#include <tuple>

template <typename I, typename J, typename A, typename B, typename C, typename T>
void testing_spmm_bsr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle     handle      = local_handle;
    rocsparse_direction  dir         = rocsparse_direction_row;
    J                    mb          = safe_size;
    J                    n           = safe_size;
    J                    kb          = safe_size;
    I                    nnzb        = safe_size;
    J                    block_dim   = safe_size;
    const T*             alpha       = (const T*)0x4;
    const T*             beta        = (const T*)0x4;
    void*                bsr_val     = (void*)0x4;
    void*                bsr_row_ptr = (void*)0x4;
    void*                bsr_col_ind = (void*)0x4;
    void*                dB          = (void*)0x4;
    void*                dC          = (void*)0x4;
    rocsparse_operation  trans_A     = rocsparse_operation_none;
    rocsparse_operation  trans_B     = rocsparse_operation_none;
    rocsparse_index_base base        = rocsparse_index_base_zero;
    rocsparse_order      order_B     = rocsparse_order_column;
    rocsparse_order      order_C     = rocsparse_order_column;
    rocsparse_spmm_alg   alg         = rocsparse_spmm_alg_default;
    rocsparse_spmm_stage stage       = rocsparse_spmm_stage_compute;

    rocsparse_indextype itype        = get_indextype<I>();
    rocsparse_indextype jtype        = get_indextype<J>();
    rocsparse_datatype  atype        = get_datatype<A>();
    rocsparse_datatype  btype        = get_datatype<B>();
    rocsparse_datatype  ctype        = get_datatype<C>();
    rocsparse_datatype  compute_type = get_datatype<T>();

    J m = mb * block_dim;
    J k = kb * block_dim;

    // SpMM structures
    rocsparse_local_spmat local_mat_A(mb,
                                      kb,
                                      nnzb,
                                      dir,
                                      block_dim,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      bsr_val,
                                      itype,
                                      jtype,
                                      base,
                                      atype,
                                      rocsparse_format_bsr);
    rocsparse_local_dnmat local_mat_B(k, n, k, dB, btype, order_B);
    rocsparse_local_dnmat local_mat_C(m, n, m, dC, ctype, order_C);

    rocsparse_spmat_descr mat_A = local_mat_A;
    rocsparse_dnmat_descr mat_B = local_mat_B;
    rocsparse_dnmat_descr mat_C = local_mat_C;

    int       nargs_to_exclude   = 2;
    const int args_to_exclude[2] = {11, 12};

#define PARAMS                                                                            \
    handle, trans_A, trans_B, alpha, mat_A, mat_B, beta, mat_C, compute_type, alg, stage, \
        buffer_size, temp_buffer
    {
        size_t* buffer_size = (size_t*)0x4;
        void*   temp_buffer = (void*)0x4;
        select_bad_arg_analysis(rocsparse_spmm, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    {
        size_t* buffer_size = (size_t*)0x4;
        void*   temp_buffer = nullptr;
        select_bad_arg_analysis(rocsparse_spmm, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    {
        size_t* buffer_size = nullptr;
        void*   temp_buffer = (void*)0x4;
        select_bad_arg_analysis(rocsparse_spmm, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    {
        size_t* buffer_size = nullptr;
        void*   temp_buffer = nullptr;
        select_bad_arg_analysis(rocsparse_spmm, nargs_to_exclude, args_to_exclude, PARAMS);
    }
#undef PARAMS
}

template <typename I, typename J, typename A, typename B, typename C, typename T>
void testing_spmm_bsr(const Arguments& arg)
{
    J                    M               = arg.M;
    J                    N               = arg.N;
    J                    K               = arg.K;
    J                    block_dim       = arg.block_dim;
    rocsparse_operation  trans_A         = arg.transA;
    rocsparse_operation  trans_B         = arg.transB;
    rocsparse_direction  dir             = arg.direction;
    rocsparse_index_base base            = arg.baseA;
    rocsparse_spmm_alg   alg             = arg.spmm_alg;
    rocsparse_order      order_B         = arg.orderB;
    rocsparse_order      order_C         = arg.orderC;
    int64_t              ld_multiplier_B = arg.ld_multiplier_B;
    int64_t              ld_multiplier_C = arg.ld_multiplier_C;

    J Mb = (M + block_dim - 1) / block_dim;
    J Kb = (K + block_dim - 1) / block_dim;

    T halpha = arg.get_alpha<T>();
    T hbeta  = arg.get_beta<T>();

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_indextype jtype = get_indextype<J>();
    rocsparse_datatype  atype = get_datatype<A>();
    rocsparse_datatype  btype = get_datatype<B>();
    rocsparse_datatype  ctype = get_datatype<C>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Allocate host memory for matrix
    host_vector<I> hbsr_row_ptr;
    host_vector<J> hbsr_col_ind;
    host_vector<A> hbsr_val;

    // Allocate host memory for matrix
    rocsparse_matrix_factory<A, I, J> matrix_factory(arg);

    I nnzb_A;
    matrix_factory.init_bsr(hbsr_row_ptr,
                            hbsr_col_ind,
                            hbsr_val,
                            dir,
                            (trans_A == rocsparse_operation_none) ? Mb : Kb,
                            (trans_A == rocsparse_operation_none) ? Kb : Mb,
                            nnzb_A,
                            block_dim,
                            base);

    M = Mb * block_dim;
    K = Kb * block_dim;

    // Some matrix properties
    J A_mb = (trans_A == rocsparse_operation_none) ? Mb : Kb;
    J A_nb = (trans_A == rocsparse_operation_none) ? Kb : Mb;
    J B_m  = (trans_B == rocsparse_operation_none) ? K : N;
    J B_n  = (trans_B == rocsparse_operation_none) ? N : K;
    J C_m  = M;
    J C_n  = N;

    int64_t ldb = (order_B == rocsparse_order_column)
                      ? ((trans_B == rocsparse_operation_none) ? (ld_multiplier_B * K)
                                                               : (ld_multiplier_B * N))
                      : ((trans_B == rocsparse_operation_none) ? (ld_multiplier_B * N)
                                                               : (ld_multiplier_B * K));
    int64_t ldc
        = (order_C == rocsparse_order_column) ? (ld_multiplier_C * M) : (ld_multiplier_C * N);

    ldb = std::max(int64_t(1), ldb);
    ldc = std::max(int64_t(1), ldc);

    int64_t nrowB = (order_B == rocsparse_order_column) ? ldb : B_m;
    int64_t ncolB = (order_B == rocsparse_order_column) ? B_n : ldb;
    int64_t nrowC = (order_C == rocsparse_order_column) ? ldc : C_m;
    int64_t ncolC = (order_C == rocsparse_order_column) ? C_n : ldc;

    int64_t nnz_A = int64_t(nnzb_A) * block_dim * block_dim;
    int64_t nnz_B = nrowB * ncolB;
    int64_t nnz_C = nrowC * ncolC;

    // Allocate host memory for vectors
    host_vector<B> hB(nnz_B);
    host_vector<C> hC_1(nnz_C);
    host_vector<C> hC_2(nnz_C);
    host_vector<C> hC_gold(nnz_C);

    // Initialize data on CPU
    rocsparse_init<B>(hB, nnz_B, 1, 1, arg.convert_to_int);
    rocsparse_init<C>(hC_1, nnz_C, 1, 1, arg.convert_to_int);

    hC_2    = hC_1;
    hC_gold = hC_1;

    // Allocate device memory
    device_vector<I> dbsr_row_ptr(A_mb + 1);
    device_vector<J> dbsr_col_ind(nnzb_A);
    device_vector<A> dbsr_val(nnz_A);
    device_vector<B> dB(nnz_B);
    device_vector<C> dC_1(nnz_C);
    device_vector<C> dC_2(nnz_C);
    device_vector<T> dalpha(1);
    device_vector<T> dbeta(1);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dbsr_row_ptr, hbsr_row_ptr.data(), sizeof(I) * (A_mb + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dbsr_col_ind, hbsr_col_ind.data(), sizeof(J) * nnzb_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dbsr_val, hbsr_val.data(), sizeof(A) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(B) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_1, hC_1, sizeof(C) * nnz_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_2, hC_2, sizeof(C) * nnz_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dalpha, &halpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dbeta, &hbeta, sizeof(T), hipMemcpyHostToDevice));

    // Create descriptors
    rocsparse_local_spmat mat_A(A_mb,
                                A_nb,
                                nnzb_A,
                                dir,
                                block_dim,
                                dbsr_row_ptr,
                                dbsr_col_ind,
                                dbsr_val,
                                itype,
                                jtype,
                                base,
                                atype,
                                rocsparse_format_bsr);

    rocsparse_local_dnmat mat_B(B_m, B_n, ldb, dB, btype, order_B);
    rocsparse_local_dnmat mat_C1(C_m, C_n, ldc, dC_1, ctype, order_C);
    rocsparse_local_dnmat mat_C2(C_m, C_n, ldc, dC_2, ctype, order_C);

    // Query SpMM buffer
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_spmm(handle,
                                         trans_A,
                                         trans_B,
                                         &halpha,
                                         mat_A,
                                         mat_B,
                                         &hbeta,
                                         mat_C1,
                                         ttype,
                                         alg,
                                         rocsparse_spmm_stage_buffer_size,
                                         &buffer_size,
                                         nullptr));

    // Allocate buffer
    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    CHECK_ROCSPARSE_ERROR(rocsparse_spmm(handle,
                                         trans_A,
                                         trans_B,
                                         &halpha,
                                         mat_A,
                                         mat_B,
                                         &hbeta,
                                         mat_C1,
                                         ttype,
                                         alg,
                                         rocsparse_spmm_stage_preprocess,
                                         &buffer_size,
                                         dbuffer));

    if(arg.unit_check)
    {
        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_spmm(handle,
                                                      trans_A,
                                                      trans_B,
                                                      &halpha,
                                                      mat_A,
                                                      mat_B,
                                                      &hbeta,
                                                      mat_C1,
                                                      ttype,
                                                      alg,
                                                      rocsparse_spmm_stage_compute,
                                                      &buffer_size,
                                                      dbuffer));
        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("dC_1", dC_1);
        }

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_spmm(handle,
                                                      trans_A,
                                                      trans_B,
                                                      dalpha,
                                                      mat_A,
                                                      mat_B,
                                                      dbeta,
                                                      mat_C2,
                                                      ttype,
                                                      alg,
                                                      rocsparse_spmm_stage_compute,
                                                      &buffer_size,
                                                      dbuffer));

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("dC_2", dC_2);
        }

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hC_1, dC_1, sizeof(C) * nnz_C, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hC_2, dC_2, sizeof(C) * nnz_C, hipMemcpyDeviceToHost));

        // CPU bsrmm
        host_bsrmm<T, I, J, A, B, C>(handle,
                                     dir,
                                     trans_A,
                                     trans_B,
                                     A_mb,
                                     N,
                                     A_nb,
                                     nnzb_A,
                                     halpha,
                                     hbsr_val,
                                     hbsr_row_ptr,
                                     hbsr_col_ind,
                                     block_dim,
                                     hB,
                                     ldb,
                                     order_B,
                                     hbeta,
                                     hC_gold,
                                     ldc,
                                     order_C,
                                     base);

        hC_gold.near_check(hC_1, get_near_check_tol<C>(arg));
        hC_gold.near_check(hC_2, get_near_check_tol<C>(arg));
    }

    if(arg.timing)
    {

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_spmm,
                                                                      handle,
                                                                      trans_A,
                                                                      trans_B,
                                                                      &halpha,
                                                                      mat_A,
                                                                      mat_B,
                                                                      &hbeta,
                                                                      mat_C1,
                                                                      ttype,
                                                                      alg,
                                                                      rocsparse_spmm_stage_compute,
                                                                      &buffer_size,
                                                                      dbuffer);

        double gflop_count = spmm_gflop_count(
            N, nnzb_A * block_dim * block_dim, (I)C_m * (I)C_n, hbeta != static_cast<T>(0));
        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);

        double gbyte_count = bsrmm_gbyte_count<T>(
            A_mb, nnzb_A, block_dim, (I)B_m * (I)B_n, (I)C_m * (I)C_n, hbeta != static_cast<T>(0));
        double gpu_gbyte = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::K,
                            K,
                            display_key_t::dir,
                            dir,
                            display_key_t::trans_A,
                            trans_A,
                            display_key_t::trans_B,
                            trans_B,
                            display_key_t::nnzb,
                            nnzb_A,
                            display_key_t::bdim,
                            block_dim,
                            display_key_t::nnz_B,
                            nnz_B,
                            display_key_t::nnz_C,
                            nnz_C,
                            display_key_t::alpha,
                            halpha,
                            display_key_t::beta,
                            hbeta,
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                              \
    template void testing_spmm_bsr_bad_arg<ITYPE, JTYPE, TTYPE, TTYPE, TTYPE, TTYPE>( \
        const Arguments& arg);                                                        \
    template void testing_spmm_bsr<ITYPE, JTYPE, TTYPE, TTYPE, TTYPE, TTYPE>(const Arguments& arg)
#define INSTANTIATE_MIXED(ITYPE, JTYPE, ATYPE, XTYPE, YTYPE, TTYPE)                   \
    template void testing_spmm_bsr_bad_arg<ITYPE, JTYPE, ATYPE, XTYPE, YTYPE, TTYPE>( \
        const Arguments& arg);                                                        \
    template void testing_spmm_bsr<ITYPE, JTYPE, ATYPE, XTYPE, YTYPE, TTYPE>(const Arguments& arg)

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

INSTANTIATE_MIXED(int32_t, int32_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE_MIXED(int64_t, int32_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE_MIXED(int64_t, int64_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE_MIXED(int32_t, int32_t, int8_t, int8_t, float, float);
INSTANTIATE_MIXED(int64_t, int32_t, int8_t, int8_t, float, float);
INSTANTIATE_MIXED(int64_t, int64_t, int8_t, int8_t, float, float);
INSTANTIATE_MIXED(int32_t, int32_t, _Float16, _Float16, float, float);
INSTANTIATE_MIXED(int64_t, int32_t, _Float16, _Float16, float, float);
INSTANTIATE_MIXED(int64_t, int64_t, _Float16, _Float16, float, float);
INSTANTIATE_MIXED(int32_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);
INSTANTIATE_MIXED(int64_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);
INSTANTIATE_MIXED(int64_t, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);

void testing_spmm_bsr_extra(const Arguments& arg) {}
