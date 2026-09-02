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
#include "testing_spmv.hpp"

template <typename I, typename A, typename X, typename Y, typename T>
void testing_spmv_coo_aos_bad_arg(const Arguments& arg)
{
    testing_spmv_dispatch<rocsparse_format_coo_aos, I, I, A, X, Y, T>::testing_spmv_bad_arg(arg);
}

template <typename I, typename A, typename X, typename Y, typename T>
void testing_spmv_coo_aos(const Arguments& arg)
{
    testing_spmv_dispatch<rocsparse_format_coo_aos, I, I, A, X, Y, T>::testing_spmv(arg);
}

#define INSTANTIATE(ITYPE, TTYPE)                                                  \
    template void testing_spmv_coo_aos_bad_arg<ITYPE, TTYPE, TTYPE, TTYPE, TTYPE>( \
        const Arguments& arg);                                                     \
    template void testing_spmv_coo_aos<ITYPE, TTYPE, TTYPE, TTYPE, TTYPE>(const Arguments& arg)

#define INSTANTIATE_MIXED(ITYPE, ATYPE, XTYPE, YTYPE, TTYPE)                       \
    template void testing_spmv_coo_aos_bad_arg<ITYPE, ATYPE, XTYPE, YTYPE, TTYPE>( \
        const Arguments& arg);                                                     \
    template void testing_spmv_coo_aos<ITYPE, ATYPE, XTYPE, YTYPE, TTYPE>(const Arguments& arg)

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

void testing_spmv_coo_aos_extra(const Arguments& arg)
{
    // Regression test for AISPARSE-660 (COO AoS variant).
    //
    // The COO AoS atomic SpMV kernel computed the global element id as a
    // product of hipBlockIdx_x and a uint32_t block dimension, so the id
    // wrapped at 2^32 regardless of how wide the index type I was. For a COO
    // AoS matrix declared with a 64-bit index type and nnz beyond 2^32,
    // non-zeros past the wrap point were never accumulated into y. The fix
    // casts the block index to int64_t before the multiply.
    //
    // This drives the 64-bit-index atomic path of rocsparse_spmv (COO AoS) with
    // nnz just past the 2^32 boundary and checks that a non-zero beyond that
    // boundary is actually accumulated into y. Everything is initialized on the
    // device and a single element is probed to stay within the (large) device
    // allocation.
    using I = int64_t;
    using T = float;

    static constexpr int64_t two_pow_32 = static_cast<int64_t>(1) << 32;

    // nnz just beyond 2^32 so at least one block has a block index whose
    // (blockIdx * BLOCKSIZE) product overflows 32-bit arithmetic.
    const I nnz = two_pow_32 + 512;
    const I m   = 2;
    const I n   = 2;

    const rocsparse_index_base base  = rocsparse_index_base_zero;
    const rocsparse_datatype   ttype = get_datatype<T>();
    const rocsparse_indextype  itype = get_indextype<I>();

    rocsparse_local_handle handle(arg);

    // AoS stores the <row, column> pair of each non-zero contiguously, so the
    // index array holds 2 * nnz entries.
    device_vector<I> dcoo_ind(2 * nnz);
    device_vector<T> dval(nnz);
    device_vector<T> dx(n);
    device_vector<T> dy(m);

    // Filler non-zeros all sit at (row 0, col 0) with value 0, so they add
    // nothing to y regardless of ordering under the atomic accumulation.
    CHECK_HIP_ERROR(hipMemset(dcoo_ind, 0, sizeof(I) * 2 * nnz));
    CHECK_HIP_ERROR(hipMemset(dval, 0, sizeof(T) * nnz));
    CHECK_HIP_ERROR(hipMemset(dy, 0, sizeof(T) * m));

    // x = [0, 1] so only column 1 contributes.
    const T hx[2] = {static_cast<T>(0), static_cast<T>(1)};
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * n, hipMemcpyHostToDevice));

    // The probe lives past the 2^32 boundary. It is the only non-zero that
    // targets row 1 / column 1, so its contribution to y[1] is isolated from
    // the filler accumulations to y[0]. Its <row, column> pair occupies
    // coo_ind[2 * probe_idx] and coo_ind[2 * probe_idx + 1].
    const I probe_idx     = two_pow_32 + 5;
    const I probe_pair[2] = {static_cast<I>(1), static_cast<I>(1)};
    const T probe_val     = static_cast<T>(1);
    CHECK_HIP_ERROR(hipMemcpy(static_cast<I*>(dcoo_ind) + 2 * probe_idx,
                              probe_pair,
                              sizeof(I) * 2,
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(static_cast<T*>(dval) + probe_idx, &probe_val, sizeof(T), hipMemcpyHostToDevice));

    rocsparse_spmat_descr mat;
    CHECK_ROCSPARSE_ERROR(rocsparse_create_coo_aos_descr(
        &mat, m, n, nnz, dcoo_ind, dval, itype, base, ttype));
    rocsparse_local_dnvec x(n, dx, ttype);
    rocsparse_local_dnvec y(m, dy, ttype);

    const rocsparse_operation trans = rocsparse_operation_none;
    const rocsparse_spmv_alg  alg   = rocsparse_spmv_alg_coo_atomic;

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

    // y[1] = alpha * probe_val * x[1] = 1. Before the fix the wrapped element
    // id leaves the probe non-zero unprocessed, so y[1] stays 0.
    T y_out = static_cast<T>(0);
    CHECK_HIP_ERROR(hipMemcpy(&y_out, static_cast<T*>(dy) + 1, sizeof(T), hipMemcpyDeviceToHost));

    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_spmat_descr(mat));

    unit_check_scalar<T>(static_cast<T>(1), y_out);
}
