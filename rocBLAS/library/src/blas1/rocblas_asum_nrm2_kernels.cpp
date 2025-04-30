/* ************************************************************************
 * Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocblas_asum_nrm2_kernels.hpp"
#include "../blas1/rocblas_asum_nrm2.hpp"
#include "rocblas_block_sizes.h"

// clang-format off
#ifdef INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER
#error INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER IS ALREADY DEFINED
#endif

#define INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER(NB_, FETCH_, FINALIZE_, T_, U_, V_)      \
    template rocblas_status rocblas_internal_asum_nrm2_launcher<rocblas_int, NB_, FETCH_, FINALIZE_, T_, U_, V_>(rocblas_handle  handle, \
                                                          rocblas_int    n,                \
                                                          T_          x,                   \
                                                          rocblas_stride shiftx,           \
                                                          rocblas_int    incx,             \
                                                          rocblas_stride stridex,          \
                                                          rocblas_int    batch_count,      \
                                                          U_*            workspace,        \
                                                          V_*            result);

//ASUM instantiations
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER(ROCBLAS_ASUM_NB, rocblas_fetch_asum<float>, rocblas_finalize_identity, float const*, float, float)
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER(ROCBLAS_ASUM_NB, rocblas_fetch_asum<float>, rocblas_finalize_identity, float const* const*, float, float)

INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER(ROCBLAS_ASUM_NB, rocblas_fetch_asum<double>, rocblas_finalize_identity, double const*, double, double)
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER(ROCBLAS_ASUM_NB, rocblas_fetch_asum<double>, rocblas_finalize_identity, double const* const*, double, double)

INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER(ROCBLAS_ASUM_NB, rocblas_fetch_asum<float>, rocblas_finalize_identity, rocblas_float_complex const*, float, float)
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER(ROCBLAS_ASUM_NB, rocblas_fetch_asum<float>, rocblas_finalize_identity, rocblas_float_complex const* const*, float, float)

INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER(ROCBLAS_ASUM_NB, rocblas_fetch_asum<double>, rocblas_finalize_identity, rocblas_double_complex const*, double, double)
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER(ROCBLAS_ASUM_NB, rocblas_fetch_asum<double>, rocblas_finalize_identity, rocblas_double_complex const* const*, double, double)

//nrm2 and nrm2_ex instantiations
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<float>, rocblas_finalize_nrm2, float const*, float, float)
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<float>, rocblas_finalize_nrm2, float const* const*, float, float)

INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<double>, rocblas_finalize_nrm2, double const*, double, double)
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<double>, rocblas_finalize_nrm2, double const* const*, double, double)

INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<float>, rocblas_finalize_nrm2, rocblas_float_complex const*, float, float)
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<float>, rocblas_finalize_nrm2, rocblas_float_complex const* const*, float, float)

INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<double>, rocblas_finalize_nrm2, rocblas_double_complex const*, double, double)
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<double>, rocblas_finalize_nrm2, rocblas_double_complex const* const*, double, double)

INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<float>, rocblas_finalize_nrm2, _Float16 const*, float, _Float16)
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<float>, rocblas_finalize_nrm2, _Float16 const* const*, float, _Float16)

INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<float>, rocblas_finalize_nrm2, rocblas_bfloat16 const*, float, rocblas_bfloat16)
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<float>, rocblas_finalize_nrm2, rocblas_bfloat16 const* const*, float, rocblas_bfloat16)

#undef INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER

// clang-format off
