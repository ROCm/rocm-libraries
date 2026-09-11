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

template <typename I, typename A, typename B, typename C, typename T>
void testing_sddmm_batched_coo_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle     handle      = local_handle;
    I                    m           = safe_size;
    I                    n           = safe_size;
    I                    k           = safe_size;
    I                    nnz         = safe_size;
    void*                coo_val     = (void*)0x4;
    void*                coo_row_ind = (void*)0x4;
    void*                coo_col_ind = (void*)0x4;
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
    rocsparse_datatype  atype = get_datatype<A>();
    rocsparse_datatype  btype = get_datatype<B>();
    rocsparse_datatype  ctype = get_datatype<C>();
    rocsparse_datatype  ttype = get_datatype<T>();

    T alpha = static_cast<T>(1.0);
    T beta  = static_cast<T>(0.0);

    // SDDMM structures: A and B are dense, C is sparse (COO).
    rocsparse_local_dnmat local_mat_A(m, k, m, dense_A, atype, order_A);
    rocsparse_local_dnmat local_mat_B(k, n, k, dense_B, btype, order_B);
    rocsparse_local_spmat local_mat_C(
        m, n, nnz, coo_row_ind, coo_col_ind, coo_val, itype, base, ctype);

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
    int64_t       batch_stride_C;

    // C_i = (A * B_i) o C_i
    batch_count_A  = 1;
    batch_count_B  = 10;
    batch_count_C  = 5;
    batch_stride_A = 0;
    batch_stride_B = k * n;
    batch_stride_C = nnz;
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_A, batch_count_A, batch_stride_A),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_B, batch_count_B, batch_stride_B),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo_set_strided_batch(mat_C, batch_count_C, batch_stride_C),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm_buffer_size(PARAMS_BUFFER_SIZE),
                            rocsparse_status_invalid_value);
    EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm_preprocess(PARAMS), rocsparse_status_invalid_value);
    EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm(PARAMS), rocsparse_status_invalid_value);

    // C_i = (A_i * B) o C_i
    batch_count_A  = 10;
    batch_count_B  = 1;
    batch_count_C  = 5;
    batch_stride_A = m * k;
    batch_stride_B = 0;
    batch_stride_C = nnz;
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_A, batch_count_A, batch_stride_A),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_B, batch_count_B, batch_stride_B),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo_set_strided_batch(mat_C, batch_count_C, batch_stride_C),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm_buffer_size(PARAMS_BUFFER_SIZE),
                            rocsparse_status_invalid_value);
    EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm_preprocess(PARAMS), rocsparse_status_invalid_value);
    EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm(PARAMS), rocsparse_status_invalid_value);

    // C_i = (A_i * B_i) o C_i
    batch_count_A  = 10;
    batch_count_B  = 10;
    batch_count_C  = 5;
    batch_stride_A = m * k;
    batch_stride_B = k * n;
    batch_stride_C = nnz;
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_A, batch_count_A, batch_stride_A),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_B, batch_count_B, batch_stride_B),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo_set_strided_batch(mat_C, batch_count_C, batch_stride_C),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm_buffer_size(PARAMS_BUFFER_SIZE),
                            rocsparse_status_invalid_value);
    EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm_preprocess(PARAMS), rocsparse_status_invalid_value);
    EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm(PARAMS), rocsparse_status_invalid_value);

    // Batched computation with rocsparse_sddmm_alg_dense is not yet supported.
    batch_count_A  = 5;
    batch_count_B  = 5;
    batch_count_C  = 5;
    batch_stride_A = m * k;
    batch_stride_B = k * n;
    batch_stride_C = nnz;
    alg            = rocsparse_sddmm_alg_dense;

    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_A, batch_count_A, batch_stride_A),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_B, batch_count_B, batch_stride_B),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo_set_strided_batch(mat_C, batch_count_C, batch_stride_C),
                            rocsparse_status_success);

    EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm_buffer_size(PARAMS_BUFFER_SIZE),
                            rocsparse_status_not_implemented);
    EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm_preprocess(PARAMS), rocsparse_status_not_implemented);
    EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm(PARAMS), rocsparse_status_not_implemented);

    alg = rocsparse_sddmm_alg_default;

#undef PARAMS
#undef PARAMS_BUFFER_SIZE
}

template <typename I, typename A, typename B, typename C, typename T>
void testing_sddmm_batched_coo(const Arguments& arg)
{
    I                    M       = arg.M;
    I                    N       = arg.N;
    I                    K       = arg.K;
    rocsparse_operation  trans_A = arg.transA;
    rocsparse_operation  trans_B = arg.transB;
    rocsparse_index_base base    = arg.baseA;
    rocsparse_sddmm_alg  alg     = arg.sddmm_alg;
    rocsparse_order      order_A = arg.order;
    rocsparse_order      order_B = arg.orderB;

    I batch_count_A = arg.batch_count_A;
    I batch_count_B = arg.batch_count_B;
    I batch_count_C = arg.batch_count_C;

    T halpha = arg.get_alpha<T>();
    T hbeta  = arg.get_beta<T>();

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
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

    const I batch_count = batch_count_C;

    rocsparse_matrix_factory<C, I, I> matrix_factory(arg);

    host_vector<I> hcoo_row_ind_temp;
    host_vector<I> hcoo_col_ind_temp;
    host_vector<C> hcoo_val_temp;

    int64_t nnz_C = 0;
    matrix_factory.init_coo(hcoo_row_ind_temp, hcoo_col_ind_temp, hcoo_val_temp, M, N, nnz_C, base);

    // Some matrix properties
    I A_m = (trans_A == rocsparse_operation_none) ? M : K;
    I A_n = (trans_A == rocsparse_operation_none) ? K : M;
    I B_m = (trans_B == rocsparse_operation_none) ? K : N;
    I B_n = (trans_B == rocsparse_operation_none) ? N : K;

    int64_t lda = (order_A == rocsparse_order_column) ? A_m : A_n;
    int64_t ldb = (order_B == rocsparse_order_column) ? B_m : B_n;

    int64_t tiny_size       = 100;
    int64_t nnz_A_per_batch = static_cast<int64_t>(A_m) * A_n + tiny_size;
    int64_t nnz_B_per_batch = static_cast<int64_t>(B_m) * B_n + tiny_size;
    int64_t nnz_C_per_batch = nnz_C + tiny_size;

    int64_t batch_stride_A = (batch_count_A > 1) ? nnz_A_per_batch : 0;
    int64_t batch_stride_B = (batch_count_B > 1) ? nnz_B_per_batch : 0;
    int64_t batch_stride_C = (batch_count_C > 1) ? nnz_C_per_batch : 0;

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

    host_vector<I> hcoo_row_ind(batch_count_C * nnz_C_per_batch);
    host_vector<I> hcoo_col_ind(batch_count_C * nnz_C_per_batch);
    host_vector<C> hcoo_val(batch_count_C * nnz_C_per_batch);
    host_vector<C> hcoo_val_gold(batch_count_C * nnz_C_per_batch);

    for(I i = 0; i < batch_count_C; ++i)
    {
        for(I j = 0; j < nnz_C; ++j)
        {
            hcoo_row_ind[nnz_C_per_batch * i + j] = hcoo_row_ind_temp[j];
            hcoo_col_ind[nnz_C_per_batch * i + j] = hcoo_col_ind_temp[j];
        }
    }

    rocsparse_init_1d_array<C>(hcoo_val,
                               batch_count_C * nnz_C_per_batch,
                               arg.convert_to_int,
                               arg.rand_gen_min,
                               arg.rand_gen_max);
    hcoo_val_gold = hcoo_val;

    // Allocate device memory
    device_vector<I> dcoo_row_ind(hcoo_row_ind);
    device_vector<I> dcoo_col_ind(hcoo_col_ind);
    device_vector<C> dcoo_val(hcoo_val);
    device_vector<A> dA(hA);
    device_vector<B> dB(hB);
    device_vector<T> dalpha(1);
    device_vector<T> dbeta(1);

    CHECK_HIP_ERROR(hipMemcpy(dalpha, &halpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dbeta, &hbeta, sizeof(T), hipMemcpyHostToDevice));

    // Create descriptors
    rocsparse_local_dnmat mat_A(A_m, A_n, std::max(int64_t(1), lda), dA, atype, order_A);
    rocsparse_local_dnmat mat_B(B_m, B_n, std::max(int64_t(1), ldb), dB, btype, order_B);

    rocsparse_local_spmat mat_C(
        M, N, nnz_C, dcoo_row_ind, dcoo_col_ind, dcoo_val, itype, base, ctype);

    CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_strided_batch(mat_A, batch_count_A, batch_stride_A));
    CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_strided_batch(mat_B, batch_count_B, batch_stride_B));
    CHECK_ROCSPARSE_ERROR(rocsparse_coo_set_strided_batch(mat_C, batch_count_C, batch_stride_C));

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
        rocsparse_sddmm_buffer_size(PARAMS_BUFFER_SIZE(&halpha, mat_A, mat_B, &hbeta, mat_C)));

    void* dbuffer = nullptr;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, std::max(buffer_size, sizeof(I))));

    CHECK_ROCSPARSE_ERROR(rocsparse_sddmm_preprocess(PARAMS(&halpha, mat_A, mat_B, &hbeta, mat_C)));

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(
            testing::rocsparse_sddmm(PARAMS(&halpha, mat_A, mat_B, &hbeta, mat_C)));

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("P pointer mode host", dcoo_val);
        }

        CHECK_HIP_ERROR(hipMemcpy(hcoo_val,
                                  dcoo_val,
                                  sizeof(C) * batch_count_C * nnz_C_per_batch,
                                  hipMemcpyDeviceToHost));

        // Restore the initial sparse output values on the device for the next
        // run. hcoo_val_gold still carries the initial values at this point.
        CHECK_HIP_ERROR(hipMemcpy(dcoo_val,
                                  hcoo_val_gold,
                                  sizeof(C) * batch_count_C * nnz_C_per_batch,
                                  hipMemcpyHostToDevice));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_sddmm(PARAMS(dalpha, mat_A, mat_B, dbeta, mat_C)));

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("P pointer mode device", dcoo_val);
        }

        // CPU reference: run cooddmm per batch. hcoo_val_gold currently holds
        // the initial COO values and is overwritten in place to become the
        // gold result.
        for(I i = 0; i < batch_count; ++i)
        {
            rocsparse_host<T, I, I, A, B, C>::cooddmm(trans_A,
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
                                                      hcoo_row_ind.data() + i * batch_stride_C,
                                                      hcoo_col_ind.data() + i * batch_stride_C,
                                                      hcoo_val_gold.data() + i * batch_stride_C,
                                                      base);
        }

        // Check the host-pointer-mode result (still in hcoo_val from above).
        hcoo_val_gold.near_check(hcoo_val);

        // Pull the device-pointer-mode result back, overwriting hcoo_val, and
        // check it.
        CHECK_HIP_ERROR(hipMemcpy(hcoo_val,
                                  dcoo_val,
                                  sizeof(C) * batch_count_C * nnz_C_per_batch,
                                  hipMemcpyDeviceToHost));
        hcoo_val_gold.near_check(hcoo_val);
    }

    if(arg.timing)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        const double gpu_time_used = rocsparse_clients::run_benchmark(
            arg, rocsparse_sddmm, PARAMS(&halpha, mat_A, mat_B, &hbeta, mat_C));

        double gflop_count = batch_count
                             * rocsparse_gflop_count<rocsparse_format_coo>::sddmm(
                                 M, N, nnz_C, K, hbeta != static_cast<T>(0));
        double gbyte_count = batch_count
                             * rocsparse_gbyte_count<rocsparse_format_coo>::template sddmm<T>(
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

#define INSTANTIATE(ITYPE, TTYPE)                                                       \
    template void testing_sddmm_batched_coo_bad_arg<ITYPE, TTYPE, TTYPE, TTYPE, TTYPE>( \
        const Arguments& arg);                                                          \
    template void testing_sddmm_batched_coo<ITYPE, TTYPE, TTYPE, TTYPE, TTYPE>(const Arguments& arg)

#define INSTANTIATE_MIXED(ITYPE, ATYPE, BTYPE, CTYPE, TTYPE)                            \
    template void testing_sddmm_batched_coo_bad_arg<ITYPE, ATYPE, BTYPE, CTYPE, TTYPE>( \
        const Arguments& arg);                                                          \
    template void testing_sddmm_batched_coo<ITYPE, ATYPE, BTYPE, CTYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, _Float16);
INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);

INSTANTIATE(int64_t, _Float16);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);

INSTANTIATE_MIXED(int32_t, _Float16, _Float16, float, float);
INSTANTIATE_MIXED(int64_t, _Float16, _Float16, float, float);
INSTANTIATE_MIXED(int32_t, _Float16, _Float16, _Float16, float);
INSTANTIATE_MIXED(int64_t, _Float16, _Float16, _Float16, float);

INSTANTIATE_MIXED(int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);
INSTANTIATE_MIXED(int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);
INSTANTIATE_MIXED(int32_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE_MIXED(int64_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16, float);

void testing_sddmm_batched_coo_extra(const Arguments& arg) {}
