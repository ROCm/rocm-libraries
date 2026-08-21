/* ************************************************************************
* Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#include <vector>

template <typename I, typename J, typename A, typename B, typename C, typename T>
void testing_spmm_csr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle     handle      = local_handle;
    J                    m           = safe_size;
    J                    n           = safe_size;
    J                    k           = safe_size;
    I                    nnz         = safe_size;
    const T*             alpha       = (const T*)0x4;
    const T*             beta        = (const T*)0x4;
    void*                csr_val     = (void*)0x4;
    void*                csr_row_ptr = (void*)0x4;
    void*                csr_col_ind = (void*)0x4;
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

    // SpMM structures
    rocsparse_local_spmat local_mat_A(
        m, k, nnz, csr_row_ptr, csr_col_ind, csr_val, itype, jtype, base, atype);
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
void testing_spmm_csr(const Arguments& arg)
{
    J                    M               = arg.M;
    J                    N               = arg.N;
    J                    K               = arg.K;
    rocsparse_operation  trans_A         = arg.transA;
    rocsparse_operation  trans_B         = arg.transB;
    rocsparse_index_base base            = arg.baseA;
    rocsparse_spmm_alg   alg             = arg.spmm_alg;
    rocsparse_order      order_B         = arg.orderB;
    rocsparse_order      order_C         = arg.orderC;
    rocsparse_int        ld_multiplier_B = arg.ld_multiplier_B;
    rocsparse_int        ld_multiplier_C = arg.ld_multiplier_C;

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
    host_vector<I> hcsr_row_ptr;
    host_vector<J> hcsr_col_ind;
    host_vector<A> hcsr_val;

    // Allocate host memory for matrix
    rocsparse_matrix_factory<A, I, J> matrix_factory(arg);

    I nnz_A;
    matrix_factory.init_csr(hcsr_row_ptr,
                            hcsr_col_ind,
                            hcsr_val,
                            (trans_A == rocsparse_operation_none) ? M : K,
                            (trans_A == rocsparse_operation_none) ? K : M,
                            nnz_A,
                            base);

    // Redefine values
    rocsparse_init_1d_array<A>(
        hcsr_val, nnz_A, arg.convert_to_int, arg.rand_gen_min, arg.rand_gen_max);

    // For low-precision types (f16/bf16), set values to 1.0f for numerical stability
    if constexpr(is_low_precision_v<A>)
    {
        set_array_to_ones(hcsr_val.data(), nnz_A);
    }

    // Some matrix properties
    J A_m = (trans_A == rocsparse_operation_none) ? M : K;
    J A_n = (trans_A == rocsparse_operation_none) ? K : M;
    J B_m = (trans_B == rocsparse_operation_none) ? K : N;
    J B_n = (trans_B == rocsparse_operation_none) ? N : K;
    J C_m = M;
    J C_n = N;

    int64_t ldb = (order_B == rocsparse_order_column)
                      ? ((trans_B == rocsparse_operation_none) ? (int64_t(ld_multiplier_B) * K)
                                                               : (int64_t(ld_multiplier_B) * N))
                      : ((trans_B == rocsparse_operation_none) ? (int64_t(ld_multiplier_B) * N)
                                                               : (int64_t(ld_multiplier_B) * K));
    int64_t ldc = (order_C == rocsparse_order_column) ? (int64_t(ld_multiplier_C) * M)
                                                      : (int64_t(ld_multiplier_C) * N);

    int64_t nrowB = (order_B == rocsparse_order_column) ? ldb : B_m;
    int64_t ncolB = (order_B == rocsparse_order_column) ? B_n : ldb;
    int64_t nrowC = (order_C == rocsparse_order_column) ? ldc : C_m;
    int64_t ncolC = (order_C == rocsparse_order_column) ? C_n : ldc;

    int64_t nnz_B = nrowB * ncolB;
    int64_t nnz_C = nrowC * ncolC;

    // Allocate host memory for vectors
    host_vector<B> hB(nnz_B);
    host_vector<C> hC_1(nnz_C);
    host_vector<C> hC_2(nnz_C);
    host_vector<C> hC_gold(nnz_C);

    // Initialize data on CPU
    rocsparse_init_1d_array<B>(hB, nnz_B, arg.convert_to_int, arg.rand_gen_min, arg.rand_gen_max);
    rocsparse_init_1d_array<C>(hC_1, nnz_C, arg.convert_to_int, arg.rand_gen_min, arg.rand_gen_max);

    // For low-precision types (f16/bf16), set values to 1.0f for numerical stability
    if constexpr(is_low_precision_v<B>)
    {
        set_array_to_ones(hB.data(), nnz_B);
    }
    if constexpr(is_low_precision_v<C>)
    {
        set_array_to_ones(hC_1.data(), nnz_C);
    }

    hC_2    = hC_1;
    hC_gold = hC_1;

    // Allocate device memory
    device_vector<I> dcsr_row_ptr(A_m + 1);
    device_vector<J> dcsr_col_ind(nnz_A);
    device_vector<A> dcsr_val(nnz_A);
    device_vector<B> dB(nnz_B);
    device_vector<C> dC_1(nnz_C);
    device_vector<C> dC_2(nnz_C);
    device_vector<T> dalpha(1);
    device_vector<T> dbeta(1);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(I) * (A_m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(J) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(A) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(B) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_1, hC_1, sizeof(C) * nnz_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_2, hC_2, sizeof(C) * nnz_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dalpha, &halpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dbeta, &hbeta, sizeof(T), hipMemcpyHostToDevice));

    // Create descriptors
    rocsparse_local_spmat mat_A(
        A_m, A_n, nnz_A, dcsr_row_ptr, dcsr_col_ind, dcsr_val, itype, jtype, base, atype);

    ldb = std::max(int64_t(1), ldb);
    ldc = std::max(int64_t(1), ldc);

    rocsparse_local_dnmat mat_B(B_m, B_n, ldb, dB, btype, order_B);
    rocsparse_local_dnmat mat_C1(C_m, C_n, ldc, dC_1, ctype, order_C);
    rocsparse_local_dnmat mat_C2(C_m, C_n, ldc, dC_2, ctype, order_C);

    // Query SpMM buffer. Route through the graph-capture wrapper so the
    // graph_test cases capture the buffer_size stage as well. The default
    // algorithm selection runs a device reduction (kernel + D2H copy + stream
    // sync) only at buffer_size; that sequence is illegal under HIP stream
    // capture, so this exercises the capture guard (the call must still return
    // success and fall back to the capture-safe row-split default). For the
    // non-graph cases the wrapper is a plain passthrough.
    size_t buffer_size;
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
        // SpMM

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

        // CPU csrmm
        host_csrmm<T, I, J, A, B, C>(A_m,
                                     N,
                                     A_n,
                                     trans_A,
                                     trans_B,
                                     halpha,
                                     hcsr_row_ptr,
                                     hcsr_col_ind,
                                     hcsr_val,
                                     hB,
                                     ldb,
                                     order_B,
                                     hbeta,
                                     hC_gold,
                                     ldc,
                                     order_C,
                                     base,
                                     false);

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

        double gflop_count
            = spmm_gflop_count(N, nnz_A, (I)C_m * (I)C_n, hbeta != static_cast<T>(0));
        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);

        double gbyte_count = csrmm_gbyte_count<A, B, C>(
            A_m, nnz_A, (I)B_m * (I)B_n, (I)C_m * (I)C_n, hbeta != static_cast<T>(0));
        double gpu_gbyte = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::K,
                            K,
                            display_key_t::nnz_A,
                            nnz_A,
                            display_key_t::alpha,
                            halpha,
                            display_key_t::beta,
                            hbeta,
                            display_key_t::algorithm,
                            rocsparse_spmmalg2string(alg),
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
    template void testing_spmm_csr_bad_arg<ITYPE, JTYPE, TTYPE, TTYPE, TTYPE, TTYPE>( \
        const Arguments& arg);                                                        \
    template void testing_spmm_csr<ITYPE, JTYPE, TTYPE, TTYPE, TTYPE, TTYPE>(const Arguments& arg)
#define INSTANTIATE_MIXED(ITYPE, JTYPE, ATYPE, XTYPE, YTYPE, TTYPE)                   \
    template void testing_spmm_csr_bad_arg<ITYPE, JTYPE, ATYPE, XTYPE, YTYPE, TTYPE>( \
        const Arguments& arg);                                                        \
    template void testing_spmm_csr<ITYPE, JTYPE, ATYPE, XTYPE, YTYPE, TTYPE>(const Arguments& arg)

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
INSTANTIATE_MIXED(int32_t, int32_t, _Float16, _Float16, _Float16, float);
INSTANTIATE_MIXED(int64_t, int32_t, _Float16, _Float16, _Float16, float);
INSTANTIATE_MIXED(int64_t, int64_t, _Float16, _Float16, _Float16, float);
INSTANTIATE_MIXED(
    int32_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE_MIXED(
    int64_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE_MIXED(
    int64_t, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16, float);

void testing_spmm_csr_extra(const Arguments& arg)
{
    // Validate the CSR SpMM default-algorithm auto-selection end to end.
    //
    // The kernel the default resolves to is deliberately not observable through
    // the public API: buffer_size is sized conservatively for the largest
    // auto-selectable kernel, and every kernel computes the same product. So this
    // test drives the full buffer_size -> preprocess -> compute pipeline - which
    // includes the structural line-nnz profile reduction and the selection it
    // feeds - and checks that the default reproduces the analytic product in both
    // asymptotic skew regimes, and that it agrees with the explicit row-split and
    // non-zero-split kernels.
    //
    // The two synthetic matrices below sit at the asymptotic ends of the
    // dimensionless gate (longest-row share -> 1 vs -> 0), so the exercised
    // decision is independent of the device compute-unit count, and therefore of
    // the architecture.

    rocsparse_local_handle local_handle;
    rocsparse_handle       handle = local_handle;

    const rocsparse_datatype   ttype = get_datatype<double>();
    const rocsparse_indextype  itype = get_indextype<int32_t>();
    const rocsparse_indextype  jtype = get_indextype<int32_t>();
    const rocsparse_index_base base  = rocsparse_index_base_zero;
    const rocsparse_order      order = rocsparse_order_column;

    const double  alpha = 1.0;
    const double  beta  = 0.0;
    const int32_t n     = 2; // dense columns of B and C

    // Run the full three-stage CSR SpMM for one structure and algorithm and copy
    // the dense result C back to the host. Returns through an out-parameter
    // because the CHECK_*_ERROR macros expand to GoogleTest ASSERT_*, which may
    // only be used in a void-returning function. With every matrix and B entry
    // equal to one (alpha = 1, beta = 0), C[i, j] is exactly the number of
    // non-zeros in row i, an integer represented exactly in double.
    auto run_spmm = [&](const host_vector<int32_t>& hrow_ptr,
                        const host_vector<int32_t>& hcol_ind,
                        int32_t                     k,
                        rocsparse_spmm_alg          alg,
                        host_vector<double>&        hC) {
        const int32_t m   = static_cast<int32_t>(hrow_ptr.size()) - 1;
        const int64_t nnz = static_cast<int64_t>(hcol_ind.size());

        const host_vector<double> hval(nnz, 1.0);
        const host_vector<double> hB(int64_t(k) * n, 1.0);

        device_vector<int32_t> drow_ptr(m + 1);
        device_vector<int32_t> dcol_ind(nnz);
        device_vector<double>  dval(nnz);
        device_vector<double>  dB(int64_t(k) * n);
        device_vector<double>  dC(int64_t(m) * n);

        CHECK_HIP_ERROR(
            hipMemcpy(drow_ptr, hrow_ptr.data(), sizeof(int32_t) * (m + 1), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(dcol_ind, hcol_ind.data(), sizeof(int32_t) * nnz, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dval, hval.data(), sizeof(double) * nnz, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(dB, hB.data(), sizeof(double) * int64_t(k) * n, hipMemcpyHostToDevice));

        rocsparse_local_spmat mat_A(m, k, nnz, drow_ptr, dcol_ind, dval, itype, jtype, base, ttype);
        rocsparse_local_dnmat mat_B(k, n, k, dB, ttype, order);
        rocsparse_local_dnmat mat_C(m, n, m, dC, ttype, order);

        size_t buffer_size = 0;
        CHECK_ROCSPARSE_ERROR(rocsparse_spmm(handle,
                                             rocsparse_operation_none,
                                             rocsparse_operation_none,
                                             &alpha,
                                             mat_A,
                                             mat_B,
                                             &beta,
                                             mat_C,
                                             ttype,
                                             alg,
                                             rocsparse_spmm_stage_buffer_size,
                                             &buffer_size,
                                             nullptr));

        void* dbuffer = nullptr;
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size > 0 ? buffer_size : 1));

        CHECK_ROCSPARSE_ERROR(rocsparse_spmm(handle,
                                             rocsparse_operation_none,
                                             rocsparse_operation_none,
                                             &alpha,
                                             mat_A,
                                             mat_B,
                                             &beta,
                                             mat_C,
                                             ttype,
                                             alg,
                                             rocsparse_spmm_stage_preprocess,
                                             &buffer_size,
                                             dbuffer));

        CHECK_ROCSPARSE_ERROR(rocsparse_spmm(handle,
                                             rocsparse_operation_none,
                                             rocsparse_operation_none,
                                             &alpha,
                                             mat_A,
                                             mat_B,
                                             &beta,
                                             mat_C,
                                             ttype,
                                             alg,
                                             rocsparse_spmm_stage_compute,
                                             &buffer_size,
                                             dbuffer));

        hC.resize(int64_t(m) * n);
        CHECK_HIP_ERROR(
            hipMemcpy(hC.data(), dC, sizeof(double) * int64_t(m) * n, hipMemcpyDeviceToHost));

        CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
    };

    // For one structure, check that the default reproduces the analytic product
    // and matches both explicit load-balanced kernels.
    auto check_structure =
        [&](const host_vector<int32_t>& hrow_ptr, const host_vector<int32_t>& hcol_ind, int32_t k) {
            const int32_t m = static_cast<int32_t>(hrow_ptr.size()) - 1;

            // Analytic reference C[i, j] = non-zeros in row i (column-major, ld = m).
            host_vector<double> hC_ref(int64_t(m) * n);
            for(int32_t j = 0; j < n; ++j)
            {
                for(int32_t i = 0; i < m; ++i)
                {
                    hC_ref[int64_t(j) * m + i] = static_cast<double>(hrow_ptr[i + 1] - hrow_ptr[i]);
                }
            }

            host_vector<double> hC_default, hC_row, hC_nnz;
            run_spmm(hrow_ptr, hcol_ind, k, rocsparse_spmm_alg_default, hC_default);
            run_spmm(hrow_ptr, hcol_ind, k, rocsparse_spmm_alg_csr_row_split, hC_row);
            run_spmm(hrow_ptr, hcol_ind, k, rocsparse_spmm_alg_csr_nnz_split, hC_nnz);

            unit_check_general<double>(m, n, hC_ref.data(), m, hC_default.data(), m);
            unit_check_general<double>(m, n, hC_ref.data(), m, hC_row.data(), m);
            unit_check_general<double>(m, n, hC_ref.data(), m, hC_nnz.data(), m);
        };

    // Extremely row-skewed matrix: one "hub" row holds almost all non-zeros, the
    // rest a single entry. Longest-row share -> 1, so the gate is free to upgrade
    // to the non-zero-split kernel; the default must still be correct.
    {
        const int32_t hub_nnz = 65536;
        const int32_t m       = 8;
        const int32_t k       = hub_nnz;

        host_vector<int32_t> hrow_ptr(m + 1, 0);
        host_vector<int32_t> hcol_ind;
        hcol_ind.reserve(hub_nnz + (m - 1));

        for(int32_t c = 0; c < hub_nnz; ++c)
        {
            hcol_ind.push_back(c);
        }
        hrow_ptr[1] = hub_nnz;

        for(int32_t r = 1; r < m; ++r)
        {
            hcol_ind.push_back(0);
            hrow_ptr[r + 1] = hrow_ptr[r] + 1;
        }

        check_structure(hrow_ptr, hcol_ind, k);
    }

    // Extremely uniform matrix: every row has the same small length. Longest-row
    // share -> 0, so the gate keeps the row-split default; the default must stay
    // correct.
    {
        const int32_t row_nnz = 4;
        const int32_t m       = 100000;
        const int32_t k       = row_nnz;

        host_vector<int32_t> hrow_ptr(m + 1, 0);
        host_vector<int32_t> hcol_ind;
        hcol_ind.reserve(int64_t(m) * row_nnz);

        for(int32_t r = 0; r < m; ++r)
        {
            for(int32_t c = 0; c < row_nnz; ++c)
            {
                hcol_ind.push_back(c);
            }
            hrow_ptr[r + 1] = hrow_ptr[r] + row_nnz;
        }

        check_structure(hrow_ptr, hcol_ind, k);
    }
}
