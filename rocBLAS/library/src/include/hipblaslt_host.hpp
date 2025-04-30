/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************/

/*********************************************************
 * Declaration of the rocBLAS<->hipBLASLt interface layer. *
 *********************************************************/

#pragma once

/*****************************************************************************
 * WARNING: hipBLASLt-specific data types, functions and macros should only be *
 * referenced from hipblaslt_host.cpp. This header file defines the interface  *
 * that the rest of rocBLAS uses to access hipBLASLt. If another hipBLASLt       *
 * feature needs to be accessed, the API for accessing it should be defined  *
 * in this file, without referencing any hipBLASLt-specific identifiers here.  *
 *****************************************************************************/

#include "handle.hpp"
#include "tensile_host.hpp"
#include "tuple_helper.hpp"
#include <atomic>

/*******************************************************************************
 * runContractionProblem() solves a RocblasContractionProblem                  *
 *******************************************************************************/
template <typename Ti, typename To, typename Tc>
rocblas_status runContractionProblemHipBlasLT(const RocblasContractionProblem<Ti, To, Tc>& problem,
                                              rocblas_gemm_algo algo = rocblas_gemm_algo_standard,
                                              int32_t           solution_index = 0);

template <typename Ti, typename To, typename Tc>
rocblas_status getAllSolutionsHipBlasLT(const RocblasContractionProblem<Ti, To, Tc>& prob,
                                        rocblas_tensile_get_solution_option          option,
                                        rocblas_int*                                 list_array,
                                        rocblas_int*                                 list_size);
