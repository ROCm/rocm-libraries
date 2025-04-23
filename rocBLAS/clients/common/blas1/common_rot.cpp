/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "../common_helpers.hpp"
#include "testing_rot.hpp"
#include "testing_rot_batched.hpp"
#include "testing_rot_strided_batched.hpp"
#include "testing_rotg.hpp"
#include "testing_rotg_batched.hpp"
#include "testing_rotg_strided_batched.hpp"
#include "testing_rotm.hpp"
#include "testing_rotm_batched.hpp"
#include "testing_rotm_strided_batched.hpp"
#include "testing_rotmg.hpp"
#include "testing_rotmg_batched.hpp"
#include "testing_rotmg_strided_batched.hpp"

// rot
#define INSTANTIATE_ROT(T_)            \
    INSTANTIATE_TESTS(rot, T_)         \
    INSTANTIATE_TESTS(rot_batched, T_) \
    INSTANTIATE_TESTS(rot_strided_batched, T_)

INSTANTIATE_ROT(float)
INSTANTIATE_ROT(double)
INSTANTIATE_ROT(rocblas_float_complex)
INSTANTIATE_ROT(rocblas_double_complex)

#define INSTANTIATE_ROT_HPA(T_, U_, V_)        \
    INSTANTIATE_TESTS(rot, T_, U_, V_)         \
    INSTANTIATE_TESTS(rot_batched, T_, U_, V_) \
    INSTANTIATE_TESTS(rot_strided_batched, T_, U_, V_)

INSTANTIATE_ROT_HPA(rocblas_float_complex, float, rocblas_float_complex)
INSTANTIATE_ROT_HPA(rocblas_float_complex, float, float)
INSTANTIATE_ROT_HPA(rocblas_double_complex, double, rocblas_double_complex)
INSTANTIATE_ROT_HPA(rocblas_double_complex, double, double)

// rotg
#define INSTANTIATE_ROTG(T_)            \
    INSTANTIATE_TESTS(rotg, T_)         \
    INSTANTIATE_TESTS(rotg_batched, T_) \
    INSTANTIATE_TESTS(rotg_strided_batched, T_)

INSTANTIATE_ROTG(float)
INSTANTIATE_ROTG(double)

#define INSTANTIATE_ROTG_HPA(T_, U_)        \
    INSTANTIATE_TESTS(rotg, T_, U_)         \
    INSTANTIATE_TESTS(rotg_batched, T_, U_) \
    INSTANTIATE_TESTS(rotg_strided_batched, T_, U_)

INSTANTIATE_ROTG_HPA(rocblas_float_complex, float)
INSTANTIATE_ROTG_HPA(rocblas_double_complex, double)

// rotm and rotmg
#define INSTANTIATE_ROTM_ROTMG(T_)              \
    INSTANTIATE_TESTS(rotm, T_)                 \
    INSTANTIATE_TESTS(rotm_batched, T_)         \
    INSTANTIATE_TESTS(rotm_strided_batched, T_) \
    INSTANTIATE_TESTS(rotmg, T_)                \
    INSTANTIATE_TESTS(rotmg_batched, T_)        \
    INSTANTIATE_TESTS(rotmg_strided_batched, T_)

INSTANTIATE_ROTM_ROTMG(float)
INSTANTIATE_ROTM_ROTMG(double)
