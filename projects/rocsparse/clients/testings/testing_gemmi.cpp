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

#include "testing.hpp"

template <typename T>
void testing_gemmi_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create rocsparse mat descriptor
    rocsparse_local_mat_descr local_descr;

    T h_alpha = static_cast<T>(1);
    T h_beta  = static_cast<T>(1);

    rocsparse_handle          handle      = local_handle;
    rocsparse_operation       trans_A     = rocsparse_operation_none;
    rocsparse_operation       trans_B     = rocsparse_operation_transpose;
    rocsparse_int             m           = safe_size;
    rocsparse_int             n           = safe_size;
    rocsparse_int             k           = safe_size;
    rocsparse_int             nnz         = safe_size;
    const T*                  alpha       = &h_alpha;
    const T*                  A           = (const T*)0x4;
    rocsparse_int             lda         = safe_size;
    const rocsparse_mat_descr descr       = local_descr;
    const T*                  csr_val     = (const T*)0x4;
    const rocsparse_int*      csr_row_ptr = (const rocsparse_int*)0x4;
    const rocsparse_int*      csr_col_ind = (const rocsparse_int*)0x4;
    const T*                  beta        = &h_beta;
    T*                        C           = (T*)0x4;
    rocsparse_int             ldc         = safe_size;

#define PARAMS                                                                          \
    handle, trans_A, trans_B, m, n, k, nnz, alpha, A, lda, descr, csr_val, csr_row_ptr, \
        csr_col_ind, beta, C, ldc

    bad_arg_analysis(rocsparse_gemmi<T>, PARAMS);

    {
        auto tmp = trans_A;
        trans_A  = rocsparse_operation_transpose;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gemmi<T>(PARAMS), rocsparse_status_not_implemented);
        trans_A = tmp;
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, rocsparse_matrix_type_symmetric));
    EXPECT_ROCSPARSE_STATUS(rocsparse_gemmi<T>(PARAMS), rocsparse_status_not_implemented);
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general));
#undef PARAMS
}

template <typename T>
void testing_gemmi(const Arguments& arg)
{
    rocsparse_int               M       = arg.M;
    rocsparse_int               N       = arg.N;
    rocsparse_int               K       = arg.K;
    rocsparse_operation         transA  = arg.transA;
    rocsparse_operation         transB  = arg.transB;
    rocsparse_index_base        base    = arg.baseA;
    rocsparse_storage_mode      storage = arg.storage;
    rocsparse_matrix_factory<T> matrix_factory(arg);

    host_scalar<T> h_alpha(arg.get_alpha<T>());
    host_scalar<T> h_beta(arg.get_beta<T>());

    device_scalar<T> d_alpha(h_alpha);
    device_scalar<T> d_beta(h_beta);

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Create rocsparse mat descriptor
    rocsparse_local_mat_descr descr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // Set matrix storage mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, storage));

    host_csr_matrix<T> hB;
    matrix_factory.init_csr(hB,
                            (transB == rocsparse_operation_none) ? K : N,
                            (transB == rocsparse_operation_none) ? N : K);

    host_dense_matrix<T> hA(M, K), hC(M, N);
    rocsparse_matrix_utils::init(hA);
    rocsparse_matrix_utils::init(hC);

    device_csr_matrix<T>   dB(hB);
    device_dense_matrix<T> dA(hA), dC(hC);

#define PARAMS(_ta, _tb, _a, _da, _db, _b, _dc)                                                 \
    handle, _ta, _tb, _dc.m, _dc.n, (_ta == rocsparse_operation_none) ? _da.n : _da.m, _db.nnz, \
        _a, _da, _da.ld, descr, _db.val, _db.ptr, _db.ind, _b, _dc, _dc.ld

    if(arg.unit_check)
    {
        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(
            testing::rocsparse_gemmi<T>(PARAMS(transA, transB, h_alpha, dA, dB, h_beta, dC)));

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("C pointer mode host", dC);
        }

        host_dense_matrix<T> hC_copy(hC);
        host_gemmi<T>(M,
                      N,
                      transA,
                      transB,
                      *h_alpha,
                      hA,
                      hA.ld,
                      hB.ptr,
                      hB.ind,
                      hB.val,
                      *h_beta,
                      hC,
                      hC.ld,
                      base);

        hC.unit_check(dC);
        dC = hC_copy;

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(
            testing::rocsparse_gemmi<T>(PARAMS(transA, transB, d_alpha, dA, dB, d_beta, dC)));
        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("C pointer mode device", dC);
        }

        hC.unit_check(dC);
    }

    const rocsparse_int nnz_A = hA.m * hA.n;
    const rocsparse_int nnz_B = hB.nnz;
    const rocsparse_int nnz_C = hC.m * hC.n;

    if(arg.timing)
    {

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        const double gpu_time_used = rocsparse_clients::run_benchmark(
            arg, rocsparse_gemmi<T>, PARAMS(transA, transB, h_alpha, dA, dB, h_beta, dC));

        double gpu_gflops = get_gpu_gflops(gpu_time_used,
                                           csrmm_gflop_count<rocsparse_int, rocsparse_int>,
                                           M,
                                           hB.nnz,
                                           nnz_C,
                                           *h_beta != static_cast<T>(0));
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used,
                                         csrmm_gbyte_count<T, T, T, rocsparse_int, rocsparse_int>,
                                         hB.m,
                                         hB.nnz,
                                         nnz_A,
                                         nnz_C,
                                         *h_beta != static_cast<T>(0));

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::K,
                            K,
                            display_key_t::trans_A,
                            rocsparse_operation2string(transA),
                            display_key_t::trans_B,
                            rocsparse_operation2string(transB),
                            display_key_t::nnz_A,
                            nnz_A,
                            display_key_t::nnz_B,
                            nnz_B,
                            display_key_t::nnz_C,
                            nnz_C,
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
#undef PARAMS
}

#define INSTANTIATE(TYPE)                                            \
    template void testing_gemmi_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_gemmi<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_gemmi_extra(const Arguments& arg) {}
