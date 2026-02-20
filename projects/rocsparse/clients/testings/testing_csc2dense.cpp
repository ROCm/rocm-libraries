/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_csx2dense.hpp"

template <typename T>
void testing_csc2dense_bad_arg(const Arguments& arg)
{
    testing_csx2dense_bad_arg<rocsparse_direction_column, T>(arg);
}
template <typename T>
void testing_csc2dense(const Arguments& arg)
{
    testing_csx2dense<rocsparse_direction_column, T>(
        arg,
        (testing::rocsparse_csc2dense<T,
                                      rocsparse_int&,
                                      rocsparse_int&,
                                      rocsparse_mat_descr&,
                                      const T*,
                                      rocsparse_int*,
                                      rocsparse_int*,
                                      T*,
                                      rocsparse_int&>),
        rocsparse_dense2csc<T>);
}

#define INSTANTIATE(TYPE)                                                \
    template void testing_csc2dense_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csc2dense<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_csc2dense_extra(const Arguments& arg) {}
