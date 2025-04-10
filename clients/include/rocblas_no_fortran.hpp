/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#ifdef CLIENTS_NO_FORTRAN

#define rocblas_axpy_ex_fortran rocblas_axpy_ex
#define rocblas_axpy_batched_ex_fortran rocblas_axpy_batched_ex
#define rocblas_axpy_strided_batched_ex_fortran rocblas_axpy_strided_batched_ex

#define rocblas_axpy_ex_64_fortran rocblas_axpy_ex_64
#define rocblas_axpy_batched_ex_64_fortran rocblas_axpy_batched_ex_64
#define rocblas_axpy_strided_batched_ex_64_fortran rocblas_axpy_strided_batched_ex_64

#define rocblas_dot_ex_fortran rocblas_dot_ex
#define rocblas_dotc_ex_fortran rocblas_dotc_ex
#define rocblas_dot_batched_ex_fortran rocblas_dot_batched_ex
#define rocblas_dotc_batched_ex_fortran rocblas_dotc_batched_ex
#define rocblas_dot_strided_batched_ex_fortran rocblas_dot_strided_batched_ex
#define rocblas_dotc_strided_batched_ex_fortran rocblas_dotc_strided_batched_ex

#define rocblas_dot_ex_64_fortran rocblas_dot_ex_64
#define rocblas_dotc_ex_64_fortran rocblas_dotc_ex_64
#define rocblas_dot_batched_ex_64_fortran rocblas_dot_batched_ex_64
#define rocblas_dotc_batched_ex_64_fortran rocblas_dotc_batched_ex_64
#define rocblas_dot_strided_batched_ex_64_fortran rocblas_dot_strided_batched_ex_64
#define rocblas_dotc_strided_batched_ex_64_fortran rocblas_dotc_strided_batched_ex_64

#define rocblas_nrm2_ex_fortran rocblas_nrm2_ex
#define rocblas_nrm2_batched_ex_fortran rocblas_nrm2_batched_ex
#define rocblas_nrm2_strided_batched_ex_fortran rocblas_nrm2_strided_batched_ex

#define rocblas_nrm2_ex_64_fortran rocblas_nrm2_ex_64
#define rocblas_nrm2_batched_ex_64_fortran rocblas_nrm2_batched_ex_64
#define rocblas_nrm2_strided_batched_ex_64_fortran rocblas_nrm2_strided_batched_ex_64

#define rocblas_rot_ex_fortran rocblas_rot_ex
#define rocblas_rot_batched_ex_fortran rocblas_rot_batched_ex
#define rocblas_rot_strided_batched_ex_fortran rocblas_rot_strided_batched_ex

#define rocblas_rot_ex_64_fortran rocblas_rot_ex_64
#define rocblas_rot_batched_ex_64_fortran rocblas_rot_batched_ex_64
#define rocblas_rot_strided_batched_ex_64_fortran rocblas_rot_strided_batched_ex_64

#define rocblas_scal_ex_fortran rocblas_scal_ex
#define rocblas_scal_batched_ex_fortran rocblas_scal_batched_ex
#define rocblas_scal_strided_batched_ex_fortran rocblas_scal_strided_batched_ex

#define rocblas_scal_ex_64_fortran rocblas_scal_ex_64
#define rocblas_scal_batched_ex_64_fortran rocblas_scal_batched_ex_64
#define rocblas_scal_strided_batched_ex_64_fortran rocblas_scal_strided_batched_ex_64

#define rocblas_gemm_ex_fortran rocblas_gemm_ex
#define rocblas_gemm_batched_ex_fortran rocblas_gemm_batched_ex
#define rocblas_gemm_strided_batched_ex_fortran rocblas_gemm_strided_batched_ex
#define rocblas_gemmt_fortran rocblas_gemmt
#define rocblas_gemmt_batched_fortran rocblas_gemmt_batched
#define rocblas_gemmt_strided_batched_fortran rocblas_gemmt_strided_batched

#define rocblas_gemm_ex_64_fortran rocblas_gemm_ex_64
#define rocblas_gemm_batched_ex_64_fortran rocblas_gemm_batched_ex_64
#define rocblas_gemm_strided_batched_ex_64_fortran rocblas_gemm_strided_batched_ex_64

#define rocblas_geam_ex_fortran rocblas_geam_ex

#define rocblas_trsm_ex_fortran rocblas_trsm_ex
#define rocblas_trsm_batched_ex_fortran rocblas_trsm_batched_ex
#define rocblas_trsm_strided_batched_ex_fortran rocblas_trsm_strided_batched_ex

#endif
