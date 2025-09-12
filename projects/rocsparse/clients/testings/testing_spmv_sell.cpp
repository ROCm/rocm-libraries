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
#include "testing_spmv.hpp"

template <typename I, typename J, typename A, typename X, typename Y, typename T>
void testing_spmv_sell_bad_arg(const Arguments& arg)
{
    const J m                = 20;
    const J n                = 20;
    const J nnz              = 100;
    const J slice_size       = 2;
    const J sell_colval_size = 120;
    const T local_alpha      = static_cast<T>(6);
    const T local_beta       = static_cast<T>(2);

    rocsparse_local_handle local_handle;

    rocsparse_handle     handle = local_handle;
    rocsparse_operation  trans  = rocsparse_operation_none;
    const void*          alpha  = (const void*)&local_alpha;
    const void*          beta   = (const void*)&local_beta;
    rocsparse_spmv_alg   alg    = rocsparse_spmv_alg_default;
    rocsparse_spmv_stage stage  = rocsparse_spmv_stage_compute;
    rocsparse_index_base base   = rocsparse_index_base_zero;
    size_t               local_buffer_size;
    size_t*              buffer_size  = &local_buffer_size;
    void*                temp_buffer  = (void*)0x4;
    rocsparse_indextype  i_type       = get_indextype<I>();
    rocsparse_indextype  j_type       = get_indextype<J>();
    rocsparse_datatype   a_type       = get_datatype<A>();
    rocsparse_datatype   x_type       = get_datatype<X>();
    rocsparse_datatype   y_type       = get_datatype<Y>();
    rocsparse_datatype   compute_type = get_datatype<T>();

    // AUTOMATIC BAD ARGS.
    rocsparse_local_spmat local_mat(m,
                                    n,
                                    nnz,
                                    slice_size,
                                    sell_colval_size,
                                    (void*)0x4,
                                    (void*)0x4,
                                    (void*)0x4,
                                    i_type,
                                    j_type,
                                    base,
                                    a_type);
    rocsparse_local_dnvec local_x(n, (void*)0x4, x_type);
    rocsparse_local_dnvec local_y(m, (void*)0x4, y_type);

    rocsparse_spmat_descr mat = local_mat;
    rocsparse_dnvec_descr x   = local_x;
    rocsparse_dnvec_descr y   = local_y;

    // WITH 2 ARGUMENTS BEING SKIPPED DURING THE CHECK.
    static const int nex   = 2;
    static const int ex[2] = {10, 11};
    select_bad_arg_analysis(rocsparse_spmv,
                            nex,
                            ex,
                            handle,
                            trans,
                            alpha,
                            mat,
                            x,
                            beta,
                            y,
                            compute_type,
                            alg,
                            stage,
                            buffer_size,
                            temp_buffer);

    buffer_size = nullptr;
    temp_buffer = nullptr;
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmv(handle,
                                           trans,
                                           alpha,
                                           mat,
                                           x,
                                           beta,
                                           y,
                                           compute_type,
                                           alg,
                                           stage,
                                           buffer_size,
                                           temp_buffer),
                            rocsparse_status_invalid_pointer);
}

template <typename I, typename J, typename A, typename X, typename Y, typename T>
void testing_spmv_sell(const Arguments& arg)
{
    J                    M          = arg.M;
    J                    N          = arg.N;
    J                    slice_size = arg.slice_size;
    rocsparse_operation  trans      = arg.transA;
    rocsparse_index_base base       = arg.baseA;
    rocsparse_spmv_alg   alg        = arg.spmv_alg;

    host_scalar<T> h_alpha(arg.get_alpha<T>());
    host_scalar<T> h_beta(arg.get_beta<T>());

    device_scalar<T> d_alpha(h_alpha);
    device_scalar<T> d_beta(h_beta);

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_indextype jtype = get_indextype<J>();
    rocsparse_datatype  atype = get_datatype<A>();
    rocsparse_datatype  xtype = get_datatype<X>();
    rocsparse_datatype  ytype = get_datatype<Y>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    rocsparse_matrix_factory<A, I, J> matrix_factory(arg);

    // Allocate host memory for matrix
    host_sell_matrix<A, I, J> hA;

    matrix_factory.init_sell(hA, M, N, slice_size, base);

    // Allocate host memory for vectors
    host_vector<X> hx((trans == rocsparse_operation_none) ? N : M);
    host_vector<Y> hy((trans == rocsparse_operation_none) ? M : N);

    // Initialize data on CPU
    rocsparse_init<X>(hx, (trans == rocsparse_operation_none) ? N : M, 1, 1, arg.convert_to_int);
    rocsparse_init<Y>(hy, (trans == rocsparse_operation_none) ? M : N, 1, 1, arg.convert_to_int);

    host_vector<Y> hy_gold(hy);

    // Allocate device memory
    device_sell_matrix<A, I, J> dA(hA);
    device_vector<X>            dx((trans == rocsparse_operation_none) ? N : M);
    device_vector<Y>            dy((trans == rocsparse_operation_none) ? M : N);

    dx.transfer_from(hx);
    dy.transfer_from(hy);

    // Create descriptors
    rocsparse_local_spmat mat_A(dA.m,
                                dA.n,
                                dA.nnz,
                                slice_size,
                                dA.sell_colval_size,
                                dA.ptr,
                                dA.ind,
                                dA.val,
                                itype,
                                jtype,
                                base,
                                atype);
    rocsparse_local_dnvec vec_x((trans == rocsparse_operation_none) ? N : M, dx, xtype);
    rocsparse_local_dnvec vec_y((trans == rocsparse_operation_none) ? M : N, dy, ytype);

    // Call spmv to get buffer size
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_spmv(handle,
                                         trans,
                                         h_alpha,
                                         mat_A,
                                         vec_x,
                                         h_beta,
                                         vec_y,
                                         ttype,
                                         alg,
                                         rocsparse_spmv_stage_buffer_size,
                                         &buffer_size,
                                         nullptr));

    void* dbuffer;
    CHECK_HIP_ERROR(hipMalloc(&dbuffer, buffer_size));

    // Call spmv to perform analysis
    CHECK_ROCSPARSE_ERROR(rocsparse_spmv(handle,
                                         trans,
                                         h_alpha,
                                         mat_A,
                                         vec_x,
                                         h_beta,
                                         vec_y,
                                         ttype,
                                         alg,
                                         rocsparse_spmv_stage_preprocess,
                                         &buffer_size,
                                         dbuffer));

    if(arg.unit_check)
    {
        // CPU sellmv
        host_sellmv<T, I, J, A, X, Y>(trans,
                                      hA.m,
                                      hA.n,
                                      hA.nnz,
                                      slice_size,
                                      hA.sell_colval_size,
                                      *h_alpha,
                                      hA.ptr,
                                      hA.ind,
                                      hA.val,
                                      hx,
                                      *h_beta,
                                      hy_gold,
                                      hA.base);

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        // Call spmv to perform computation
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_spmv(handle,
                                                      trans,
                                                      h_alpha,
                                                      mat_A,
                                                      vec_x,
                                                      h_beta,
                                                      vec_y,
                                                      ttype,
                                                      alg,
                                                      rocsparse_spmv_stage_compute,
                                                      &buffer_size,
                                                      dbuffer));

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("dy", dy);
        }

        hy_gold.near_check(dy);
        dy.transfer_from(hy);

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_spmv(handle,
                                                      trans,
                                                      d_alpha,
                                                      mat_A,
                                                      vec_x,
                                                      d_beta,
                                                      vec_y,
                                                      ttype,
                                                      alg,
                                                      rocsparse_spmv_stage_compute,
                                                      &buffer_size,
                                                      dbuffer));

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("dy", dy);
        }

        hy_gold.near_check(dy);
    }

    if(arg.timing)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_spmv,
                                                                      handle,
                                                                      trans,
                                                                      h_alpha,
                                                                      mat_A,
                                                                      vec_x,
                                                                      h_beta,
                                                                      vec_y,
                                                                      ttype,
                                                                      alg,
                                                                      rocsparse_spmv_stage_compute,
                                                                      &buffer_size,
                                                                      dbuffer);

        double gflop_count = spmv_gflop_count(dA.m, dA.nnz, *h_beta != static_cast<T>(0));
        double gbyte_count = sellmv_gbyte_count<A, X, Y, I, J>(
            dA.m, dA.n, dA.nnz, slice_size, dA.sell_colval_size, *h_beta != static_cast<T>(0));

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::nnz,
                            dA.nnz,
                            display_key_t::alpha,
                            *h_alpha,
                            display_key_t::beta,
                            *h_beta,
                            display_key_t::algorithm,
                            rocsparse_spmvalg2string(alg),
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
    template void testing_spmv_sell_bad_arg<ITYPE, JTYPE, TTYPE, TTYPE, TTYPE, TTYPE>( \
        const Arguments& arg);                                                         \
    template void testing_spmv_sell<ITYPE, JTYPE, TTYPE, TTYPE, TTYPE, TTYPE>(const Arguments& arg)

#define INSTANTIATE_MIXED(ITYPE, JTYPE, ATYPE, XTYPE, YTYPE, TTYPE)                    \
    template void testing_spmv_sell_bad_arg<ITYPE, JTYPE, ATYPE, XTYPE, YTYPE, TTYPE>( \
        const Arguments& arg);                                                         \
    template void testing_spmv_sell<ITYPE, JTYPE, ATYPE, XTYPE, YTYPE, TTYPE>(const Arguments& arg)

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
INSTANTIATE_MIXED(int32_t, int32_t, float, double, double, double);
INSTANTIATE_MIXED(int64_t, int32_t, float, double, double, double);
INSTANTIATE_MIXED(int64_t, int64_t, float, double, double, double);

INSTANTIATE_MIXED(int32_t,
                  int32_t,
                  float,
                  rocsparse_float_complex,
                  rocsparse_float_complex,
                  rocsparse_float_complex);
INSTANTIATE_MIXED(int64_t,
                  int32_t,
                  float,
                  rocsparse_float_complex,
                  rocsparse_float_complex,
                  rocsparse_float_complex);
INSTANTIATE_MIXED(int64_t,
                  int64_t,
                  float,
                  rocsparse_float_complex,
                  rocsparse_float_complex,
                  rocsparse_float_complex);

INSTANTIATE_MIXED(int32_t,
                  int32_t,
                  double,
                  rocsparse_double_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);
INSTANTIATE_MIXED(int64_t,
                  int32_t,
                  double,
                  rocsparse_double_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);
INSTANTIATE_MIXED(int64_t,
                  int64_t,
                  double,
                  rocsparse_double_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);

INSTANTIATE_MIXED(int32_t,
                  int32_t,
                  rocsparse_float_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);
INSTANTIATE_MIXED(int64_t,
                  int32_t,
                  rocsparse_float_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);
INSTANTIATE_MIXED(int64_t,
                  int64_t,
                  rocsparse_float_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);

void testing_spmv_sell_extra(const Arguments& arg) {}
