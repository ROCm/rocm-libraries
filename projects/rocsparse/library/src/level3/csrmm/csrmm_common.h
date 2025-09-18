/*! \file */
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
#define INSTANTIATE_KERNELS(kernel_macro)                                                 \
    kernel_macro(float, int32_t, int32_t, float, float, float);                           \
    kernel_macro(float, int64_t, int32_t, float, float, float);                           \
    kernel_macro(float, int64_t, int64_t, float, float, float);                           \
    kernel_macro(double, int32_t, int32_t, double, double, double);                       \
    kernel_macro(double, int64_t, int32_t, double, double, double);                       \
    kernel_macro(double, int64_t, int64_t, double, double, double);                       \
    kernel_macro(rocsparse_float_complex,                                                 \
                 int32_t,                                                                 \
                 int32_t,                                                                 \
                 rocsparse_float_complex,                                                 \
                 rocsparse_float_complex,                                                 \
                 rocsparse_float_complex);                                                \
    kernel_macro(rocsparse_float_complex,                                                 \
                 int64_t,                                                                 \
                 int32_t,                                                                 \
                 rocsparse_float_complex,                                                 \
                 rocsparse_float_complex,                                                 \
                 rocsparse_float_complex);                                                \
    kernel_macro(rocsparse_float_complex,                                                 \
                 int64_t,                                                                 \
                 int64_t,                                                                 \
                 rocsparse_float_complex,                                                 \
                 rocsparse_float_complex,                                                 \
                 rocsparse_float_complex);                                                \
    kernel_macro(rocsparse_double_complex,                                                \
                 int32_t,                                                                 \
                 int32_t,                                                                 \
                 rocsparse_double_complex,                                                \
                 rocsparse_double_complex,                                                \
                 rocsparse_double_complex);                                               \
    kernel_macro(rocsparse_double_complex,                                                \
                 int64_t,                                                                 \
                 int32_t,                                                                 \
                 rocsparse_double_complex,                                                \
                 rocsparse_double_complex,                                                \
                 rocsparse_double_complex);                                               \
    kernel_macro(rocsparse_double_complex,                                                \
                 int64_t,                                                                 \
                 int64_t,                                                                 \
                 rocsparse_double_complex,                                                \
                 rocsparse_double_complex,                                                \
                 rocsparse_double_complex);                                               \
    kernel_macro(float, int32_t, int32_t, _Float16, _Float16, float);                     \
    kernel_macro(float, int64_t, int32_t, _Float16, _Float16, float);                     \
    kernel_macro(float, int64_t, int64_t, _Float16, _Float16, float);                     \
    kernel_macro(float, int32_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float); \
    kernel_macro(float, int64_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float); \
    kernel_macro(float, int64_t, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float); \
    kernel_macro(int32_t, int32_t, int32_t, int8_t, int8_t, int32_t);                     \
    kernel_macro(int32_t, int64_t, int32_t, int8_t, int8_t, int32_t);                     \
    kernel_macro(int32_t, int64_t, int64_t, int8_t, int8_t, int32_t);                     \
    kernel_macro(float, int32_t, int32_t, int8_t, int8_t, float);                         \
    kernel_macro(float, int64_t, int32_t, int8_t, int8_t, float);                         \
    kernel_macro(float, int64_t, int64_t, int8_t, int8_t, float);
