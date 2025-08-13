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

template <typename I, typename A, typename B, typename C, typename T>
void testing_spmm_bell_bad_arg(const Arguments& arg)
{
    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle     handle       = local_handle;
    rocsparse_operation  trans_A      = rocsparse_operation_none;
    rocsparse_operation  trans_B      = rocsparse_operation_none;
    void*                alpha        = (void*)0x4;
    void*                beta         = (void*)0x4;
    rocsparse_datatype   compute_type = rocsparse_datatype_f32_r;
    rocsparse_spmm_alg   alg          = rocsparse_spmm_alg_bell;
    rocsparse_spmm_stage stage        = rocsparse_spmm_stage_compute;
    size_t*              buffer_size  = (size_t*)0x4;
    void*                buffer       = (void*)0x4;

    rocsparse_local_spmat local_mat_A(4,
                                      4,
                                      rocsparse_direction_row,
                                      2,
                                      2,
                                      (void*)0x4,
                                      (void*)0x4,
                                      rocsparse_indextype_i32,
                                      rocsparse_index_base_zero,
                                      compute_type);
    rocsparse_local_dnmat local_mat_B(4, 4, 4, (void*)0x4, compute_type, rocsparse_order_column);
    rocsparse_local_dnmat local_mat_C(4, 4, 4, (void*)0x4, compute_type, rocsparse_order_column);

    rocsparse_spmat_descr mat_A = local_mat_A;
    rocsparse_dnmat_descr mat_B = local_mat_B;
    rocsparse_dnmat_descr mat_C = local_mat_C;

#define PARAMS                                                                            \
    handle, trans_A, trans_B, alpha, mat_A, mat_B, beta, mat_C, compute_type, alg, stage, \
        buffer_size, buffer

    static const int nargs_to_exclude                  = 2;
    static const int args_to_exclude[nargs_to_exclude] = {11, 12};

    select_bad_arg_analysis(rocsparse_spmm, nargs_to_exclude, args_to_exclude, PARAMS);

#undef PARAMS
}

template <typename I, typename A, typename B, typename C, typename T>
void testing_spmm_bell(const Arguments& arg)
{
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_datatype  atype = get_datatype<A>();
    rocsparse_datatype  btype = get_datatype<B>();
    rocsparse_datatype  ctype = get_datatype<C>();
    rocsparse_datatype  ttype = get_datatype<T>();

    I M         = arg.M;
    I N         = arg.N;
    I K         = arg.K;
    I block_dim = arg.block_dim;

    rocsparse_operation  trans_A   = arg.transA;
    rocsparse_operation  trans_B   = arg.transB;
    rocsparse_direction  direction = arg.direction;
    rocsparse_index_base base      = arg.baseA;
    rocsparse_spmm_alg   alg       = arg.spmm_alg;
    rocsparse_order      order_B   = arg.orderB;
    rocsparse_order      order_C   = arg.orderC;

    I Mb = -1, Nb = -1, Kb = -1;

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

    rocsparse_matrix_factory<A, I, I> matrix_factory(arg);

    host_ell_matrix<A, I> hA;
    I                     hA_m = (trans_A == rocsparse_operation_none) ? M : K;
    I                     hA_n = (trans_A == rocsparse_operation_none) ? K : M;
    matrix_factory.init_ell(hA, hA_m, hA_n, base);

    Mb = hA_m;
    Nb = N;
    Kb = hA_n;

    M = Mb * block_dim;
    N = Nb * block_dim;
    K = Kb * block_dim;

    host_dense_matrix<B> hB((trans_B == rocsparse_operation_none) ? K : N,
                            (trans_B == rocsparse_operation_none) ? N : K);
    rocsparse_matrix_utils::init_exact(hB);
    device_dense_matrix<B> dB(hB);

    //
    // C
    //
    host_dense_matrix<C> hC(M, N);
    rocsparse_matrix_utils::init_exact(hC);
    device_dense_matrix<C> dC(hC);

    device_ell_matrix<A, I> dA(hA);
    host_dense_matrix<A>    hA_val(1, dA.width * Mb * block_dim * block_dim);
    rocsparse_matrix_utils::init_exact(hA_val);
    device_dense_matrix<A> dA_val(hA_val);
    rocsparse_local_spmat  mat_A(M,
                                K,
                                direction,
                                block_dim,
                                dA.width * block_dim,
                                (I*)dA.ind,
                                (A*)dA_val,
                                itype,
                                base,
                                atype);

    rocsparse_local_dnmat mat_B(
        dB.m,
        dB.n,
        std::max(rocsparse_int(1), (order_B == rocsparse_order_column) ? dB.m : dB.n),
        dB,
        btype,
        order_B);
    rocsparse_local_dnmat mat_C(
        dC.m,
        dC.n,
        std::max(rocsparse_int(1), (order_C == rocsparse_order_column) ? dC.m : dC.n),
        dC,
        ctype,
        order_C);

    // Query SpMM buffer
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_spmm(handle,
                                         trans_A,
                                         trans_B,
                                         h_alpha,
                                         mat_A,
                                         mat_B,
                                         h_beta,
                                         mat_C,
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
                                         h_alpha,
                                         mat_A,
                                         mat_B,
                                         h_beta,
                                         mat_C,
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
                                                      h_alpha,
                                                      mat_A,
                                                      mat_B,
                                                      h_beta,
                                                      mat_C,
                                                      ttype,
                                                      alg,
                                                      rocsparse_spmm_stage_compute,
                                                      &buffer_size,
                                                      dbuffer));

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("C pointer mode host", dC);
        }

        {
            host_dense_matrix<C> hC_copy(hC);

            //
            // Host calculation.
            //
            // Convert the Blocked ELL matrix to coo matrix...
            //
            I nnzb  = 0;
            I bound = hA.m * hA.width;
            for(size_t i = 0; i < bound; ++i)
            {
                if(hA.ind[i] - hA.base >= 0)
                {
                    ++nnzb;
                }
            }

            I* coo_row = new I[size_t(nnzb) * block_dim * block_dim];
            I* coo_col = new I[size_t(nnzb) * block_dim * block_dim];
            A* coo_val = new A[size_t(nnzb) * block_dim * block_dim];

            size_t at = 0;
            for(I ib = 0; ib < Mb; ++ib)
            {
                for(I l = 0; l < hA.width; ++l)
                {
                    const size_t idx = Mb * l + ib;

                    const I jb = hA.ind[idx] - hA.base;
                    if(jb >= 0)
                    {
                        if(direction == rocsparse_direction_column)
                        {
                            for(I lcol = 0; lcol < block_dim; ++lcol)
                            {
                                for(I lrow = 0; lrow < block_dim; ++lrow)
                                {
                                    coo_row[at] = ib * block_dim + lrow + hA.base;
                                    coo_col[at] = jb * block_dim + lcol + hA.base;
                                    coo_val[at] = hA_val[block_dim * block_dim * idx
                                                         + lcol * block_dim + lrow];
                                    ++at;
                                }
                            }
                        }
                        else
                        {
                            for(I lcol = 0; lcol < block_dim; ++lcol)
                            {
                                for(I lrow = 0; lrow < block_dim; ++lrow)
                                {
                                    coo_row[at] = ib * block_dim + lrow + hA.base;
                                    coo_col[at] = jb * block_dim + lcol + hA.base;
                                    coo_val[at] = hA_val[block_dim * block_dim * idx
                                                         + lrow * block_dim + lcol];
                                    ++at;
                                }
                            }
                        }
                    }
                }
            }

            host_coomm<T, I, A, B, C>(M,
                                      N,
                                      K,
                                      nnzb * block_dim * block_dim,
                                      trans_A,
                                      trans_B,
                                      *h_alpha,
                                      coo_row,
                                      coo_col,
                                      coo_val,
                                      hB,
                                      (order_B == rocsparse_order_column) ? hB.m : hB.n,
                                      order_B,
                                      *h_beta,
                                      hC,
                                      (order_C == rocsparse_order_column) ? hC.m : hC.n,
                                      order_C,
                                      base);

            delete[] coo_val;
            delete[] coo_col;
            delete[] coo_row;
            if(trans_A == rocsparse_operation_none)
            {
                hC.near_check(dC);
            }
            dC = hC_copy;
        }

        // Pointer mode device
        {

            CHECK_ROCSPARSE_ERROR(
                rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
            CHECK_ROCSPARSE_ERROR(testing::rocsparse_spmm(handle,
                                                          trans_A,
                                                          trans_B,
                                                          d_alpha,
                                                          mat_A,
                                                          mat_B,
                                                          d_beta,
                                                          mat_C,
                                                          ttype,
                                                          alg,
                                                          rocsparse_spmm_stage_compute,
                                                          &buffer_size,
                                                          dbuffer));
            if(ROCSPARSE_REPRODUCIBILITY)
            {
                rocsparse_reproducibility::save("C pointer mode device", dC);
            }
        }
    }

    if(arg.timing)
    {

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_spmm,
                                                                      handle,
                                                                      trans_A,
                                                                      trans_B,
                                                                      h_alpha,
                                                                      mat_A,
                                                                      mat_B,
                                                                      h_beta,
                                                                      mat_C,
                                                                      ttype,
                                                                      alg,
                                                                      rocsparse_spmm_stage_compute,
                                                                      &buffer_size,
                                                                      dbuffer);

        double gflop_count = spmm_gflop_count(
            N, dA.nnz, (int64_t)dC.m * (int64_t)dC.n, *h_beta != static_cast<T>(0));
        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);

        double gbyte_count = bellmm_gbyte_count<A, B, C, I>(Mb,
                                                            dA.width,
                                                            block_dim,
                                                            (int64_t)dB.m * (int64_t)dB.n,
                                                            (int64_t)dC.m * (int64_t)dC.n,
                                                            *h_beta != static_cast<T>(0));
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::K,
                            K,
                            display_key_t::nnz,
                            dA.nnz,
                            display_key_t::alpha,
                            *h_alpha,
                            display_key_t::beta,
                            *h_beta,
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

#define INSTANTIATE(ITYPE, TTYPE)                                               \
    template void testing_spmm_bell_bad_arg<ITYPE, TTYPE, TTYPE, TTYPE, TTYPE>( \
        const Arguments& arg);                                                  \
    template void testing_spmm_bell<ITYPE, TTYPE, TTYPE, TTYPE, TTYPE>(const Arguments& arg)
#define INSTANTIATE_MIXED(ITYPE, ATYPE, XTYPE, YTYPE, TTYPE)                    \
    template void testing_spmm_bell_bad_arg<ITYPE, ATYPE, XTYPE, YTYPE, TTYPE>( \
        const Arguments& arg);                                                  \
    template void testing_spmm_bell<ITYPE, ATYPE, XTYPE, YTYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);

INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);

INSTANTIATE_MIXED(int32_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE_MIXED(int64_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE_MIXED(int32_t, int8_t, int8_t, float, float);
INSTANTIATE_MIXED(int64_t, int8_t, int8_t, float, float);
INSTANTIATE_MIXED(int32_t, _Float16, _Float16, float, float);
INSTANTIATE_MIXED(int64_t, _Float16, _Float16, float, float);
INSTANTIATE_MIXED(int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);
INSTANTIATE_MIXED(int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);
void testing_spmm_bell_extra(const Arguments& arg) {}
