/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
 *
 * ************************************************************************ */

#pragma once

#include "hipsolver.h"
#include "lib_macros.hpp"
#include <cusolverDn.h>
#include <cusolverRf.h>

HIPSOLVER_BEGIN_NAMESPACE

cublasOperation_t hip2cuda_operation(hipsolverOperation_t op);

hipsolverOperation_t cuda2hip_operation(cublasOperation_t op);

cublasFillMode_t hip2cuda_fill(hipsolverFillMode_t fill);

hipsolverFillMode_t cuda2hip_fill(cublasFillMode_t fill);

cublasSideMode_t hip2cuda_side(hipsolverSideMode_t side);

hipsolverSideMode_t cuda2hip_side(cublasSideMode_t side);

cusolverEigMode_t hip2cuda_evect(hipsolverEigMode_t eig);

hipsolverEigMode_t cuda2hip_evect(cusolverEigMode_t eig);

cusolverEigType_t hip2cuda_eform(hipsolverEigType_t eig);

hipsolverEigType_t cuda2hip_eform(cusolverEigType_t eig);

cusolverEigRange_t hip2cuda_erange(hipsolverEigRange_t eig);

hipsolverEigRange_t cuda2hip_erange(cusolverEigRange_t eig);

cusolverAlgMode_t hip2cuda_algmode(hipsolverAlgMode_t mode);

hipsolverAlgMode_t cuda2hip_algmode(cusolverAlgMode_t mode);

#if(CUDART_VERSION >= 12020)
cusolverDeterministicMode_t hip2cuda_deterministic(hipsolverDeterministicMode_t mode);

hipsolverDeterministicMode_t cuda2hip_deterministic(cusolverDeterministicMode_t mode);
#endif

hipsolverStatus_t cuda2hip_status(cusolverStatus_t cuStatus);

// Dense API
cusolverDnFunction_t hip2cuda_function(hipsolverDnFunction_t func);

hipsolverDnFunction_t cuda2hip_function(cusolverDnFunction_t func);

// Refactor API
cusolverRfFactorization_t hip2cuda_factorization(hipsolverRfFactorization_t alg);

hipsolverRfFactorization_t cuda2hip_factorization(cusolverRfFactorization_t alg);

cusolverRfMatrixFormat_t hip2cuda_matrixformat(hipsolverRfMatrixFormat_t format);

hipsolverRfMatrixFormat_t cuda2hip_matrixformat(cusolverRfMatrixFormat_t format);

cusolverRfNumericBoostReport_t hip2cuda_boostrep(hipsolverRfNumericBoostReport_t nbr);

hipsolverRfNumericBoostReport_t cuda2hip_boostrep(cusolverRfNumericBoostReport_t nbr);

cusolverRfResetValuesFastMode_t hip2cuda_resetvalsfm(hipsolverRfResetValuesFastMode_t mode);

hipsolverRfResetValuesFastMode_t cuda2hip_resetvalsfm(cusolverRfResetValuesFastMode_t mode);

cusolverRfTriangularSolve_t hip2cuda_trisolve(hipsolverRfTriangularSolve_t alg);

hipsolverRfTriangularSolve_t cuda2hip_trisolve(cusolverRfTriangularSolve_t alg);

cusolverRfUnitDiagonal_t hip2cuda_unitdiag(hipsolverRfUnitDiagonal_t diag);

hipsolverRfUnitDiagonal_t cuda2hip_unitdiag(cusolverRfUnitDiagonal_t diag);

HIPSOLVER_END_NAMESPACE
