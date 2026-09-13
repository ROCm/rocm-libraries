/* ************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

template <typename I, typename A, typename X, typename Y, typename T>
void testing_spmv_ell_bad_arg(const Arguments& arg)
{
    testing_spmv_dispatch<rocsparse_format_ell, I, I, A, X, Y, T>::testing_spmv_bad_arg(arg);
}

template <typename I, typename A, typename X, typename Y, typename T>
void testing_spmv_ell(const Arguments& arg)
{
    testing_spmv_dispatch<rocsparse_format_ell, I, I, A, X, Y, T>::testing_spmv(arg);
}

#define INSTANTIATE(ITYPE, TTYPE)                                              \
    template void testing_spmv_ell_bad_arg<ITYPE, TTYPE, TTYPE, TTYPE, TTYPE>( \
        const Arguments& arg);                                                 \
    template void testing_spmv_ell<ITYPE, TTYPE, TTYPE, TTYPE, TTYPE>(const Arguments& arg)

#define INSTANTIATE_MIXED(ITYPE, ATYPE, XTYPE, YTYPE, TTYPE)                   \
    template void testing_spmv_ell_bad_arg<ITYPE, ATYPE, XTYPE, YTYPE, TTYPE>( \
        const Arguments& arg);                                                 \
    template void testing_spmv_ell<ITYPE, ATYPE, XTYPE, YTYPE, TTYPE>(const Arguments& arg)

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
INSTANTIATE_MIXED(int32_t, _Float16, _Float16, _Float16, float);
INSTANTIATE_MIXED(int64_t, _Float16, _Float16, _Float16, float);
INSTANTIATE_MIXED(int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);
INSTANTIATE_MIXED(int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float, float);
INSTANTIATE_MIXED(int32_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE_MIXED(int64_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16, float);

INSTANTIATE_MIXED(
    int32_t, float, rocsparse_float_complex, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE_MIXED(
    int64_t, float, rocsparse_float_complex, rocsparse_float_complex, rocsparse_float_complex);

INSTANTIATE_MIXED(int32_t, float, double, double, double);
INSTANTIATE_MIXED(int64_t, float, double, double, double);

INSTANTIATE_MIXED(
    int32_t, double, rocsparse_double_complex, rocsparse_double_complex, rocsparse_double_complex);
INSTANTIATE_MIXED(
    int64_t, double, rocsparse_double_complex, rocsparse_double_complex, rocsparse_double_complex);

INSTANTIATE_MIXED(int32_t,
                  rocsparse_float_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);
INSTANTIATE_MIXED(int64_t,
                  rocsparse_float_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);

void testing_spmv_ell_extra(const Arguments& arg)
{
    // Regression test for AISPARSE-658.
    //
    // The ELL SpMV kernels computed the row index as
    //     const I ai = BLOCKSIZE * hipBlockIdx_x + hipThreadIdx_x;
    // where BLOCKSIZE is a uint32_t, so the product was evaluated in 32-bit
    // unsigned arithmetic and wrapped at 2^32 even when the index type I was
    // 64-bit. For an ELL matrix declared with more than 2^32 rows, the rows
    // past the wrap point were mapped back onto low row ids, so their y entries
    // were never computed. The fix casts BLOCKSIZE to I before the multiply.
    //
    // This drives the 64-bit-index ELL path of rocsparse_spmv with m just past
    // the 2^32 boundary (one non-zero per row, as ELL pads the rest) and checks
    // that a row beyond that boundary is actually evaluated. Everything is
    // initialized on the device and a single row is probed to stay within the
    // (large) device allocation.
    using I = int64_t;
    using T = float;

    static constexpr int64_t two_pow_32 = static_cast<int64_t>(1) << 32;

    // m just beyond 2^32 so the top rows sit in blocks whose
    // (blockIdx * BLOCKSIZE) product overflows 32-bit arithmetic.
    const I m         = two_pow_32 + 512;
    const I n         = 2;
    const I ell_width = 1;
    const I ell_nnz   = m * ell_width; // one stored entry per row

    const rocsparse_index_base base  = rocsparse_index_base_zero;
    const rocsparse_datatype   ttype = get_datatype<T>();
    const rocsparse_indextype  itype = get_indextype<I>();

    rocsparse_local_handle handle(arg);

    device_vector<I> dell_col_ind(ell_nnz);
    device_vector<T> dell_val(ell_nnz);
    device_vector<T> dx(n);
    device_vector<T> dy(m);

    // Every row holds a single entry in column 0 with value 0. With the ELL
    // column-major layout (ELL_IND) and ell_width == 1 the storage index equals
    // the row id, and with x[0] == 0 these entries add nothing to y regardless
    // of the row id.
    CHECK_HIP_ERROR(hipMemset(dell_col_ind, 0, sizeof(I) * ell_nnz));
    CHECK_HIP_ERROR(hipMemset(dell_val, 0, sizeof(T) * ell_nnz));
    CHECK_HIP_ERROR(hipMemset(dy, 0, sizeof(T) * m));

    // x = [0, 1] so only column 1 contributes.
    const T hx[2] = {static_cast<T>(0), static_cast<T>(1)};
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * n, hipMemcpyHostToDevice));

    // The probe row lives past the 2^32 boundary; its single entry targets
    // column 1 with value 1, so y[probe_row] must become 1.
    const I probe_row = two_pow_32 + 5;
    const I probe_col = 1;
    const T probe_val = static_cast<T>(1);
    CHECK_HIP_ERROR(hipMemcpy(
        static_cast<I*>(dell_col_ind) + probe_row, &probe_col, sizeof(I), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        static_cast<T*>(dell_val) + probe_row, &probe_val, sizeof(T), hipMemcpyHostToDevice));

    rocsparse_spmat_descr mat;
    CHECK_ROCSPARSE_ERROR(rocsparse_create_ell_descr(&mat,
                                                     m,
                                                     n,
                                                     static_cast<I*>(dell_col_ind),
                                                     static_cast<T*>(dell_val),
                                                     ell_width,
                                                     itype,
                                                     base,
                                                     ttype));

    rocsparse_local_dnvec x(n, dx, ttype);
    rocsparse_local_dnvec y(m, dy, ttype);

    const rocsparse_operation trans = rocsparse_operation_none;
    const rocsparse_spmv_alg  alg   = rocsparse_spmv_alg_default;

    // beta == 0 clears y; alpha scales the accumulated products.
    const T halpha = static_cast<T>(1);
    const T hbeta  = static_cast<T>(0);

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    void*  dbuffer     = nullptr;
    size_t buffer_size = 0;
    CHECK_ROCSPARSE_ERROR(rocsparse_spmv(handle,
                                         trans,
                                         &halpha,
                                         mat,
                                         x,
                                         &hbeta,
                                         y,
                                         ttype,
                                         alg,
                                         rocsparse_spmv_stage_buffer_size,
                                         &buffer_size,
                                         dbuffer));
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    CHECK_ROCSPARSE_ERROR(rocsparse_spmv(handle,
                                         trans,
                                         &halpha,
                                         mat,
                                         x,
                                         &hbeta,
                                         y,
                                         ttype,
                                         alg,
                                         rocsparse_spmv_stage_preprocess,
                                         &buffer_size,
                                         dbuffer));

    CHECK_ROCSPARSE_ERROR(testing::rocsparse_spmv(handle,
                                                  trans,
                                                  &halpha,
                                                  mat,
                                                  x,
                                                  &hbeta,
                                                  y,
                                                  ttype,
                                                  alg,
                                                  rocsparse_spmv_stage_compute,
                                                  &buffer_size,
                                                  dbuffer));

    // y[probe_row] = alpha * probe_val * x[1] = 1. Before the fix the wrapped
    // row id leaves this row unprocessed, so it stays 0.
    T y_out = static_cast<T>(0);
    CHECK_HIP_ERROR(
        hipMemcpy(&y_out, static_cast<T*>(dy) + probe_row, sizeof(T), hipMemcpyDeviceToHost));

    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_spmat_descr(mat));

    unit_check_scalar<T>(static_cast<T>(1), y_out);
}
