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

template <typename I, typename J, typename A, typename B, typename C, typename T>
void testing_sddmm_batched_csc_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle     handle      = local_handle;
    J                    m           = safe_size;
    J                    n           = safe_size;
    J                    k           = safe_size;
    I                    nnz         = safe_size;
    void*                csc_val     = (void*)0x4;
    void*                csc_col_ptr = (void*)0x4;
    void*                csc_row_ind = (void*)0x4;
    void*                dense_A     = (void*)0x4;
    void*                dense_B     = (void*)0x4;
    size_t*              buffer_size = (size_t*)0x4;
    void*                temp_buffer = (void*)0x4;
    rocsparse_operation  trans_A     = rocsparse_operation_none;
    rocsparse_operation  trans_B     = rocsparse_operation_none;
    rocsparse_index_base base        = rocsparse_index_base_zero;
    rocsparse_order      order_A     = rocsparse_order_column;
    rocsparse_order      order_B     = rocsparse_order_column;
    rocsparse_sddmm_alg  alg         = rocsparse_sddmm_alg_default;

    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_indextype jtype = get_indextype<J>();
    rocsparse_datatype  atype = get_datatype<A>();
    rocsparse_datatype  btype = get_datatype<B>();
    rocsparse_datatype  ctype = get_datatype<C>();
    rocsparse_datatype  ttype = get_datatype<T>();

    T alpha = static_cast<T>(1.0);
    T beta  = static_cast<T>(0.0);

    // SDDMM structures: A and B are dense, C is sparse (CSC).
    static const bool     use_csc_format = true;
    rocsparse_local_dnmat local_mat_A(m, k, m, dense_A, atype, order_A);
    rocsparse_local_dnmat local_mat_B(k, n, k, dense_B, btype, order_B);
    rocsparse_local_spmat local_mat_C(
        m, n, nnz, csc_col_ptr, csc_row_ind, csc_val, itype, jtype, base, ctype, use_csc_format);

    rocsparse_dnmat_descr mat_A = local_mat_A;
    rocsparse_dnmat_descr mat_B = local_mat_B;
    rocsparse_spmat_descr mat_C = local_mat_C;

#define PARAMS_BUFFER_SIZE \
    handle, trans_A, trans_B, &alpha, mat_A, mat_B, &beta, mat_C, ttype, alg, buffer_size

#define PARAMS handle, trans_A, trans_B, &alpha, mat_A, mat_B, &beta, mat_C, ttype, alg, temp_buffer

    rocsparse_int batch_count_A;
    rocsparse_int batch_count_B;
    rocsparse_int batch_count_C;
    int64_t       batch_stride_A;
    int64_t       batch_stride_B;
    int64_t       offsets_batch_stride_C;
    int64_t       rows_values_batch_stride_C;

    // Mismatching batch counts between A and C.
    batch_count_A              = 10;
    batch_count_B              = 5;
    batch_count_C              = 5;
    batch_stride_A             = m * k;
    batch_stride_B             = k * n;
    offsets_batch_stride_C     = 0;
    rows_values_batch_stride_C = nnz;

    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_A, batch_count_A, batch_stride_A),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_B, batch_count_B, batch_stride_B),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_csc_set_strided_batch(
            mat_C, batch_count_C, offsets_batch_stride_C, rows_values_batch_stride_C),
        rocsparse_status_success);

    EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm_buffer_size(PARAMS_BUFFER_SIZE),
                            rocsparse_status_invalid_value);
    EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm_preprocess(PARAMS), rocsparse_status_invalid_value);
    EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm(PARAMS), rocsparse_status_invalid_value);

    // Mismatching batch counts between B and C.
    batch_count_A = 5;
    batch_count_B = 10;
    batch_count_C = 5;

    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_A, batch_count_A, batch_stride_A),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_B, batch_count_B, batch_stride_B),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_csc_set_strided_batch(
            mat_C, batch_count_C, offsets_batch_stride_C, rows_values_batch_stride_C),
        rocsparse_status_success);

    EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm_buffer_size(PARAMS_BUFFER_SIZE),
                            rocsparse_status_invalid_value);
    EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm_preprocess(PARAMS), rocsparse_status_invalid_value);
    EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm(PARAMS), rocsparse_status_invalid_value);

    // Batched computation with rocsparse_sddmm_alg_dense is not yet supported.
    batch_count_A = 5;
    batch_count_B = 5;
    batch_count_C = 5;
    alg           = rocsparse_sddmm_alg_dense;

    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_A, batch_count_A, batch_stride_A),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_B, batch_count_B, batch_stride_B),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_csc_set_strided_batch(
            mat_C, batch_count_C, offsets_batch_stride_C, rows_values_batch_stride_C),
        rocsparse_status_success);

    EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm_buffer_size(PARAMS_BUFFER_SIZE),
                            rocsparse_status_not_implemented);
    EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm_preprocess(PARAMS), rocsparse_status_not_implemented);
    EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm(PARAMS), rocsparse_status_not_implemented);

    alg = rocsparse_sddmm_alg_default;

#undef PARAMS
#undef PARAMS_BUFFER_SIZE
}

template <typename I, typename J, typename A, typename B, typename C, typename T>
void testing_sddmm_batched_csc(const Arguments& arg)
{
    J                    M       = arg.M;
    J                    N       = arg.N;
    J                    K       = arg.K;
    rocsparse_operation  trans_A = arg.transA;
    rocsparse_operation  trans_B = arg.transB;
    rocsparse_index_base base    = arg.baseA;
    rocsparse_sddmm_alg  alg     = arg.sddmm_alg;
    rocsparse_order      order_A = arg.order;
    rocsparse_order      order_B = arg.orderB;

    J batch_count_A = arg.batch_count_A;
    J batch_count_B = arg.batch_count_B;
    J batch_count_C = arg.batch_count_C;

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

    bool Ci_A_B_Ci   = (batch_count_A == 1 && batch_count_B == 1);
    bool Ci_A_Bi_Ci  = (batch_count_A == 1 && batch_count_B == batch_count_C);
    bool Ci_Ai_B_Ci  = (batch_count_B == 1 && batch_count_A == batch_count_C);
    bool Ci_Ai_Bi_Ci = (batch_count_A == batch_count_C && batch_count_A == batch_count_B);

    if(!Ci_A_B_Ci && !Ci_A_Bi_Ci && !Ci_Ai_B_Ci && !Ci_Ai_Bi_Ci)
    {
        return;
    }

    const J batch_count = batch_count_C;

    // Allocate host memory and generate the sparsity pattern for the output
    // sparse matrix C (shared across batches). Values will be set/computed per
    // batch below.
    rocsparse_matrix_factory<C, I, J> matrix_factory(arg);

    host_vector<I> hcsc_col_ptr_temp;
    host_vector<J> hcsc_row_ind_temp;
    host_vector<C> hcsc_val_temp;

    I nnz_C;
    matrix_factory.init_csc(hcsc_col_ptr_temp, hcsc_row_ind_temp, hcsc_val_temp, M, N, nnz_C, base);

    // Some matrix properties
    J A_m = (trans_A == rocsparse_operation_none) ? M : K;
    J A_n = (trans_A == rocsparse_operation_none) ? K : M;
    J B_m = (trans_B == rocsparse_operation_none) ? K : N;
    J B_n = (trans_B == rocsparse_operation_none) ? N : K;

    int64_t lda = (order_A == rocsparse_order_column) ? A_m : A_n;
    int64_t ldb = (order_B == rocsparse_order_column) ? B_m : B_n;

    int64_t tiny_size       = 100;
    int64_t nnz_A_per_batch = static_cast<int64_t>(A_m) * A_n + tiny_size;
    int64_t nnz_B_per_batch = static_cast<int64_t>(B_m) * B_n + tiny_size;
    int64_t nnz_C_per_batch = nnz_C + tiny_size;

    int64_t batch_stride_A             = (batch_count_A > 1) ? nnz_A_per_batch : 0;
    int64_t batch_stride_B             = (batch_count_B > 1) ? nnz_B_per_batch : 0;
    int64_t offsets_batch_stride_C     = (batch_count_C > 1) ? N + 1 : 0;
    int64_t rows_values_batch_stride_C = (batch_count_C > 1) ? nnz_C_per_batch : 0;

    // Allocate/initialize dense A and B matrices (per batch unique).
    host_vector<A> hA(batch_count_A * nnz_A_per_batch);
    host_vector<B> hB(batch_count_B * nnz_B_per_batch);
    rocsparse_init_1d_array<A>(hA,
                               batch_count_A * nnz_A_per_batch,
                               arg.convert_to_int,
                               arg.rand_gen_min,
                               arg.rand_gen_max);
    rocsparse_init_1d_array<B>(hB,
                               batch_count_B * nnz_B_per_batch,
                               arg.convert_to_int,
                               arg.rand_gen_min,
                               arg.rand_gen_max);

    host_vector<I> hcsc_col_ptr(batch_count_C * (N + 1));
    for(J i = 0; i < batch_count_C; ++i)
    {
        for(size_t j = 0; j < static_cast<size_t>(N + 1); ++j)
        {
            hcsc_col_ptr[(N + 1) * i + j] = hcsc_col_ptr_temp[j];
        }
    }

    host_vector<J> hcsc_row_ind(batch_count_C * nnz_C_per_batch);
    host_vector<C> hcsc_val_1(batch_count_C * nnz_C_per_batch);
    host_vector<C> hcsc_val_2(batch_count_C * nnz_C_per_batch);
    host_vector<C> hcsc_val_gold(batch_count_C * nnz_C_per_batch);

    for(J i = 0; i < batch_count_C; ++i)
    {
        for(I j = 0; j < nnz_C; ++j)
        {
            hcsc_row_ind[nnz_C_per_batch * i + j] = hcsc_row_ind_temp[j];
        }
    }

    rocsparse_init_1d_array<C>(hcsc_val_1,
                               batch_count_C * nnz_C_per_batch,
                               arg.convert_to_int,
                               arg.rand_gen_min,
                               arg.rand_gen_max);
    hcsc_val_2    = hcsc_val_1;
    hcsc_val_gold = hcsc_val_1;

    // Allocate device memory
    device_vector<I> dcsc_col_ptr(hcsc_col_ptr);
    device_vector<J> dcsc_row_ind(hcsc_row_ind);
    device_vector<C> dcsc_val_1(hcsc_val_1);
    device_vector<C> dcsc_val_2(hcsc_val_2);
    device_vector<A> dA(hA);
    device_vector<B> dB(hB);
    device_vector<T> dalpha(1);
    device_vector<T> dbeta(1);

    CHECK_HIP_ERROR(hipMemcpy(dalpha, &halpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dbeta, &hbeta, sizeof(T), hipMemcpyHostToDevice));

    // Create descriptors
    static const bool     use_csc_format = true;
    rocsparse_local_dnmat mat_A(A_m, A_n, std::max(int64_t(1), lda), dA, atype, order_A);
    rocsparse_local_dnmat mat_B(B_m, B_n, std::max(int64_t(1), ldb), dB, btype, order_B);

    rocsparse_local_spmat mat_C1(M,
                                 N,
                                 nnz_C,
                                 dcsc_col_ptr,
                                 dcsc_row_ind,
                                 dcsc_val_1,
                                 itype,
                                 jtype,
                                 base,
                                 ctype,
                                 use_csc_format);
    rocsparse_local_spmat mat_C2(M,
                                 N,
                                 nnz_C,
                                 dcsc_col_ptr,
                                 dcsc_row_ind,
                                 dcsc_val_2,
                                 itype,
                                 jtype,
                                 base,
                                 ctype,
                                 use_csc_format);

    CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_strided_batch(mat_A, batch_count_A, batch_stride_A));
    CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_strided_batch(mat_B, batch_count_B, batch_stride_B));
    CHECK_ROCSPARSE_ERROR(rocsparse_csc_set_strided_batch(
        mat_C1, batch_count_C, offsets_batch_stride_C, rows_values_batch_stride_C));
    CHECK_ROCSPARSE_ERROR(rocsparse_csc_set_strided_batch(
        mat_C2, batch_count_C, offsets_batch_stride_C, rows_values_batch_stride_C));

#define PARAMS(alpha_, A_, B_, beta_, C_)                                                      \
    handle, trans_A, trans_B, alpha_, (const rocsparse_dnmat_descr&)A_,                        \
        (const rocsparse_dnmat_descr&)B_, beta_, (const rocsparse_spmat_descr&)C_, ttype, alg, \
        dbuffer
#define PARAMS_BUFFER_SIZE(alpha_, A_, B_, beta_, C_)                                          \
    handle, trans_A, trans_B, alpha_, (const rocsparse_dnmat_descr&)A_,                        \
        (const rocsparse_dnmat_descr&)B_, beta_, (const rocsparse_spmat_descr&)C_, ttype, alg, \
        &buffer_size

    // Query SDDMM buffer
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(
        rocsparse_sddmm_buffer_size(PARAMS_BUFFER_SIZE(&halpha, mat_A, mat_B, &hbeta, mat_C1)));

    void* dbuffer = nullptr;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, std::max(buffer_size, sizeof(I))));

    CHECK_ROCSPARSE_ERROR(
        rocsparse_sddmm_preprocess(PARAMS(&halpha, mat_A, mat_B, &hbeta, mat_C1)));

    if(arg.unit_check)
    {
        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(
            testing::rocsparse_sddmm(PARAMS(&halpha, mat_A, mat_B, &hbeta, mat_C1)));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(
            testing::rocsparse_sddmm(PARAMS(dalpha, mat_A, mat_B, dbeta, mat_C2)));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hcsc_val_1,
                                  dcsc_val_1,
                                  sizeof(C) * batch_count_C * nnz_C_per_batch,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsc_val_2,
                                  dcsc_val_2,
                                  sizeof(C) * batch_count_C * nnz_C_per_batch,
                                  hipMemcpyDeviceToHost));

        // CPU reference: run cscddmm per batch.
        for(J i = 0; i < batch_count; ++i)
        {
            rocsparse_host<T, I, J, A, B, C>::cscddmm(
                trans_A,
                trans_B,
                order_A,
                order_B,
                M,
                N,
                K,
                nnz_C,
                &halpha,
                hA.data() + i * batch_stride_A,
                lda,
                hB.data() + i * batch_stride_B,
                ldb,
                &hbeta,
                hcsc_col_ptr.data() + i * offsets_batch_stride_C,
                hcsc_row_ind.data() + i * rows_values_batch_stride_C,
                hcsc_val_gold.data() + i * rows_values_batch_stride_C,
                base);
        }

        hcsc_val_gold.near_check(hcsc_val_1);
        hcsc_val_gold.near_check(hcsc_val_2);
    }

    if(arg.timing)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        const double gpu_time_used = rocsparse_clients::run_benchmark(
            arg, rocsparse_sddmm, PARAMS(&halpha, mat_A, mat_B, &hbeta, mat_C1));

        double gflop_count = batch_count
                             * rocsparse_gflop_count<rocsparse_format_csc>::sddmm(
                                 M, N, nnz_C, K, hbeta != static_cast<T>(0));
        double gbyte_count = batch_count
                             * rocsparse_gbyte_count<rocsparse_format_csc>::template sddmm<T>(
                                 M, N, nnz_C, K, hbeta != static_cast<T>(0));

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::trans_A,
                            rocsparse_operation2string(trans_A),
                            display_key_t::trans_B,
                            rocsparse_operation2string(trans_B),
                            display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::K,
                            K,
                            display_key_t::nnz,
                            nnz_C,
                            display_key_t::batch_count_A,
                            batch_count_A,
                            display_key_t::batch_count_B,
                            batch_count_B,
                            display_key_t::batch_count_C,
                            batch_count_C,
                            display_key_t::alpha,
                            halpha,
                            display_key_t::beta,
                            hbeta,
                            display_key_t::algorithm,
                            rocsparse_sddmmalg2string(alg),
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));

#undef PARAMS
#undef PARAMS_BUFFER_SIZE
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                       \
    template void testing_sddmm_batched_csc_bad_arg<ITYPE, JTYPE, TTYPE, TTYPE, TTYPE, TTYPE>( \
        const Arguments& arg);                                                                 \
    template void testing_sddmm_batched_csc<ITYPE, JTYPE, TTYPE, TTYPE, TTYPE, TTYPE>(         \
        const Arguments& arg)

#define INSTANTIATE_MIXED(ITYPE, JTYPE, ATYPE, BTYPE, CTYPE, TTYPE)                            \
    template void testing_sddmm_batched_csc_bad_arg<ITYPE, JTYPE, ATYPE, BTYPE, CTYPE, TTYPE>( \
        const Arguments& arg);                                                                 \
    template void testing_sddmm_batched_csc<ITYPE, JTYPE, ATYPE, BTYPE, CTYPE, TTYPE>(         \
        const Arguments& arg)

INSTANTIATE(int32_t, int32_t, _Float16);
INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);

INSTANTIATE(int64_t, int32_t, _Float16);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);

INSTANTIATE(int64_t, int64_t, _Float16);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);

INSTANTIATE_MIXED(int32_t, int32_t, _Float16, _Float16, float, float);
INSTANTIATE_MIXED(int64_t, int32_t, _Float16, _Float16, float, float);
INSTANTIATE_MIXED(int64_t, int64_t, _Float16, _Float16, float, float);
INSTANTIATE_MIXED(int32_t, int32_t, _Float16, _Float16, _Float16, float);
INSTANTIATE_MIXED(int64_t, int32_t, _Float16, _Float16, _Float16, float);
INSTANTIATE_MIXED(int64_t, int64_t, _Float16, _Float16, _Float16, float);

INSTANTIATE_MIXED(int32_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);
INSTANTIATE_MIXED(int64_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);
INSTANTIATE_MIXED(int64_t, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);
INSTANTIATE_MIXED(
    int32_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE_MIXED(
    int64_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE_MIXED(
    int64_t, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16, float);

void testing_sddmm_batched_csc_extra(const Arguments& arg) {}
