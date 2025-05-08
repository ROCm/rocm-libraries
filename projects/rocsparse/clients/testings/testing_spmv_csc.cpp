/* ************************************************************************
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
void testing_spmv_csc_bad_arg(const Arguments& arg)
{
    testing_spmv_dispatch<rocsparse_format_csc, I, J, A, X, Y, T>::testing_spmv_bad_arg(arg);
}

template <typename I, typename J, typename A, typename X, typename Y, typename T>
void testing_spmv_csc(const Arguments& arg)
{
    testing_spmv_dispatch<rocsparse_format_csc, I, J, A, X, Y, T>::testing_spmv(arg);
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                              \
    template void testing_spmv_csc_bad_arg<ITYPE, JTYPE, TTYPE, TTYPE, TTYPE, TTYPE>( \
        const Arguments& arg);                                                        \
    template void testing_spmv_csc<ITYPE, JTYPE, TTYPE, TTYPE, TTYPE, TTYPE>(const Arguments& arg)

#define INSTANTIATE_MIXED(ITYPE, JTYPE, ATYPE, XTYPE, YTYPE, TTYPE)                   \
    template void testing_spmv_csc_bad_arg<ITYPE, JTYPE, ATYPE, XTYPE, YTYPE, TTYPE>( \
        const Arguments& arg);                                                        \
    template void testing_spmv_csc<ITYPE, JTYPE, ATYPE, XTYPE, YTYPE, TTYPE>(const Arguments& arg)

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

INSTANTIATE_MIXED(int32_t, int32_t, float, double, double, double);
INSTANTIATE_MIXED(int64_t, int32_t, float, double, double, double);
INSTANTIATE_MIXED(int64_t, int64_t, float, double, double, double);

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

void testing_spmv_csc_extra(const Arguments& arg) {}
