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

/*! \file
 *  \brief Methods to convert hipSOLVER enums to and from cuSOLVER enums
 */

#include "hipsolver_conversions.hpp"

HIPSOLVER_BEGIN_NAMESPACE

cublasOperation_t hip2cuda_operation(hipsolverOperation_t op)
{
    switch(op)
    {
    case HIPSOLVER_OP_N:
        return CUBLAS_OP_N;
    case HIPSOLVER_OP_T:
        return CUBLAS_OP_T;
    case HIPSOLVER_OP_C:
        return CUBLAS_OP_C;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverOperation_t cuda2hip_operation(cublasOperation_t op)
{
    switch(op)
    {
    case CUBLAS_OP_N:
        return HIPSOLVER_OP_N;
    case CUBLAS_OP_T:
        return HIPSOLVER_OP_T;
    case CUBLAS_OP_C:
        return HIPSOLVER_OP_C;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

cublasFillMode_t hip2cuda_fill(hipsolverFillMode_t fill)
{
    switch(fill)
    {
    case HIPSOLVER_FILL_MODE_UPPER:
        return CUBLAS_FILL_MODE_UPPER;
    case HIPSOLVER_FILL_MODE_LOWER:
        return CUBLAS_FILL_MODE_LOWER;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverFillMode_t cuda2hip_fill(cublasFillMode_t fill)
{
    switch(fill)
    {
    case CUBLAS_FILL_MODE_UPPER:
        return HIPSOLVER_FILL_MODE_UPPER;
    case CUBLAS_FILL_MODE_LOWER:
        return HIPSOLVER_FILL_MODE_LOWER;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

cublasSideMode_t hip2cuda_side(hipsolverSideMode_t side)
{
    switch(side)
    {
    case HIPSOLVER_SIDE_LEFT:
        return CUBLAS_SIDE_LEFT;
    case HIPSOLVER_SIDE_RIGHT:
        return CUBLAS_SIDE_RIGHT;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverSideMode_t cuda2hip_side(cublasSideMode_t side)
{
    switch(side)
    {
    case CUBLAS_SIDE_LEFT:
        return HIPSOLVER_SIDE_LEFT;
    case CUBLAS_SIDE_RIGHT:
        return HIPSOLVER_SIDE_RIGHT;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

cusolverEigMode_t hip2cuda_evect(hipsolverEigMode_t eig)
{
    switch(eig)
    {
    case HIPSOLVER_EIG_MODE_NOVECTOR:
        return CUSOLVER_EIG_MODE_NOVECTOR;
    case HIPSOLVER_EIG_MODE_VECTOR:
        return CUSOLVER_EIG_MODE_VECTOR;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverEigMode_t cuda2hip_evect(cusolverEigMode_t eig)
{
    switch(eig)
    {
    case CUSOLVER_EIG_MODE_NOVECTOR:
        return HIPSOLVER_EIG_MODE_NOVECTOR;
    case CUSOLVER_EIG_MODE_VECTOR:
        return HIPSOLVER_EIG_MODE_VECTOR;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

cusolverEigType_t hip2cuda_eform(hipsolverEigType_t eig)
{
    switch(eig)
    {
    case HIPSOLVER_EIG_TYPE_1:
        return CUSOLVER_EIG_TYPE_1;
    case HIPSOLVER_EIG_TYPE_2:
        return CUSOLVER_EIG_TYPE_2;
    case HIPSOLVER_EIG_TYPE_3:
        return CUSOLVER_EIG_TYPE_3;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverEigType_t cuda2hip_eform(cusolverEigType_t eig)
{
    switch(eig)
    {
    case CUSOLVER_EIG_TYPE_1:
        return HIPSOLVER_EIG_TYPE_1;
    case CUSOLVER_EIG_TYPE_2:
        return HIPSOLVER_EIG_TYPE_2;
    case CUSOLVER_EIG_TYPE_3:
        return HIPSOLVER_EIG_TYPE_3;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

cusolverEigRange_t hip2cuda_erange(hipsolverEigRange_t eig)
{
    switch(eig)
    {
    case HIPSOLVER_EIG_RANGE_ALL:
        return CUSOLVER_EIG_RANGE_ALL;
    case HIPSOLVER_EIG_RANGE_V:
        return CUSOLVER_EIG_RANGE_V;
    case HIPSOLVER_EIG_RANGE_I:
        return CUSOLVER_EIG_RANGE_I;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverEigRange_t cuda2hip_erange(cusolverEigRange_t eig)
{
    switch(eig)
    {
    case CUSOLVER_EIG_RANGE_ALL:
        return HIPSOLVER_EIG_RANGE_ALL;
    case CUSOLVER_EIG_RANGE_V:
        return HIPSOLVER_EIG_RANGE_V;
    case CUSOLVER_EIG_RANGE_I:
        return HIPSOLVER_EIG_RANGE_I;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

cusolverAlgMode_t hip2cuda_algmode(hipsolverAlgMode_t mode)
{
    switch(mode)
    {
    case HIPSOLVER_ALG_0:
        return CUSOLVER_ALG_0;
    case HIPSOLVER_ALG_1:
        return CUSOLVER_ALG_1;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverAlgMode_t cuda2hip_algmode(cusolverAlgMode_t mode)
{
    switch(mode)
    {
    case CUSOLVER_ALG_0:
        return HIPSOLVER_ALG_0;
    case CUSOLVER_ALG_1:
        return HIPSOLVER_ALG_1;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

#if(CUDART_VERSION >= 12020)
cusolverDeterministicMode_t hip2cuda_deterministic(hipsolverDeterministicMode_t mode)
{
    switch(mode)
    {
    case HIPSOLVER_DETERMINISTIC_RESULTS:
        return CUSOLVER_DETERMINISTIC_RESULTS;
    case HIPSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS:
        return CUSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverDeterministicMode_t cuda2hip_deterministic(cusolverDeterministicMode_t mode)
{
    switch(mode)
    {
    case CUSOLVER_DETERMINISTIC_RESULTS:
        return HIPSOLVER_DETERMINISTIC_RESULTS;
    case CUSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS:
        return HIPSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}
#endif

hipsolverStatus_t cuda2hip_status(cusolverStatus_t cuStatus)
{
    switch(cuStatus)
    {
    case CUSOLVER_STATUS_SUCCESS:
        return HIPSOLVER_STATUS_SUCCESS;
    case CUSOLVER_STATUS_NOT_INITIALIZED:
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    case CUSOLVER_STATUS_ALLOC_FAILED:
        return HIPSOLVER_STATUS_ALLOC_FAILED;
    case CUSOLVER_STATUS_INVALID_VALUE:
    case CUSOLVER_STATUS_IRS_PARAMS_INVALID:
        return HIPSOLVER_STATUS_INVALID_VALUE;
    case CUSOLVER_STATUS_MAPPING_ERROR:
        return HIPSOLVER_STATUS_MAPPING_ERROR;
    case CUSOLVER_STATUS_EXECUTION_FAILED:
        return HIPSOLVER_STATUS_EXECUTION_FAILED;
    case CUSOLVER_STATUS_INTERNAL_ERROR:
    case CUSOLVER_STATUS_IRS_INTERNAL_ERROR:
        return HIPSOLVER_STATUS_INTERNAL_ERROR;
    case CUSOLVER_STATUS_NOT_SUPPORTED:
    case CUSOLVER_STATUS_IRS_NOT_SUPPORTED:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    case CUSOLVER_STATUS_ARCH_MISMATCH:
        return HIPSOLVER_STATUS_ARCH_MISMATCH;
    case CUSOLVER_STATUS_ZERO_PIVOT:
        return HIPSOLVER_STATUS_ZERO_PIVOT;
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    default:
        return HIPSOLVER_STATUS_UNKNOWN;
    }
}

/******************** DENSE API ********************/
cusolverDnFunction_t hip2cuda_function(hipsolverDnFunction_t func)
{
    switch(func)
    {
    case HIPSOLVERDN_GETRF:
        return CUSOLVERDN_GETRF;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverDnFunction_t cuda2hip_function(cusolverDnFunction_t func)
{
    switch(func)
    {
    case CUSOLVERDN_GETRF:
        return HIPSOLVERDN_GETRF;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

/******************** REFACTOR API ********************/
cusolverRfFactorization_t hip2cuda_factorization(hipsolverRfFactorization_t alg)
{
    switch(alg)
    {
    case HIPSOLVERRF_FACTORIZATION_ALG0:
        return CUSOLVERRF_FACTORIZATION_ALG0;
    case HIPSOLVERRF_FACTORIZATION_ALG1:
        return CUSOLVERRF_FACTORIZATION_ALG1;
    case HIPSOLVERRF_FACTORIZATION_ALG2:
        return CUSOLVERRF_FACTORIZATION_ALG2;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverRfFactorization_t cuda2hip_factorization(cusolverRfFactorization_t alg)
{
    switch(alg)
    {
    case CUSOLVERRF_FACTORIZATION_ALG0:
        return HIPSOLVERRF_FACTORIZATION_ALG0;
    case CUSOLVERRF_FACTORIZATION_ALG1:
        return HIPSOLVERRF_FACTORIZATION_ALG1;
    case CUSOLVERRF_FACTORIZATION_ALG2:
        return HIPSOLVERRF_FACTORIZATION_ALG2;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

cusolverRfMatrixFormat_t hip2cuda_matrixformat(hipsolverRfMatrixFormat_t format)
{
    switch(format)
    {
    case HIPSOLVERRF_MATRIX_FORMAT_CSR:
        return CUSOLVERRF_MATRIX_FORMAT_CSR;
    case HIPSOLVERRF_MATRIX_FORMAT_CSC:
        return CUSOLVERRF_MATRIX_FORMAT_CSC;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverRfMatrixFormat_t cuda2hip_matrixformat(cusolverRfMatrixFormat_t format)
{
    switch(format)
    {
    case CUSOLVERRF_MATRIX_FORMAT_CSR:
        return HIPSOLVERRF_MATRIX_FORMAT_CSR;
    case CUSOLVERRF_MATRIX_FORMAT_CSC:
        return HIPSOLVERRF_MATRIX_FORMAT_CSC;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

cusolverRfNumericBoostReport_t hip2cuda_boostrep(hipsolverRfNumericBoostReport_t nbr)
{
    switch(nbr)
    {
    case HIPSOLVERRF_NUMERIC_BOOST_NOT_USED:
        return CUSOLVERRF_NUMERIC_BOOST_NOT_USED;
    case HIPSOLVERRF_NUMERIC_BOOST_USED:
        return CUSOLVERRF_NUMERIC_BOOST_USED;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverRfNumericBoostReport_t cuda2hip_boostrep(cusolverRfNumericBoostReport_t nbr)
{
    switch(nbr)
    {
    case CUSOLVERRF_NUMERIC_BOOST_NOT_USED:
        return HIPSOLVERRF_NUMERIC_BOOST_NOT_USED;
    case CUSOLVERRF_NUMERIC_BOOST_USED:
        return HIPSOLVERRF_NUMERIC_BOOST_USED;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

cusolverRfResetValuesFastMode_t hip2cuda_resetvalsfm(hipsolverRfResetValuesFastMode_t mode)
{
    switch(mode)
    {
    case HIPSOLVERRF_RESET_VALUES_FAST_MODE_OFF:
        return CUSOLVERRF_RESET_VALUES_FAST_MODE_OFF;
    case HIPSOLVERRF_RESET_VALUES_FAST_MODE_ON:
        return CUSOLVERRF_RESET_VALUES_FAST_MODE_ON;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverRfResetValuesFastMode_t cuda2hip_resetvalsfm(cusolverRfResetValuesFastMode_t mode)
{
    switch(mode)
    {
    case CUSOLVERRF_RESET_VALUES_FAST_MODE_OFF:
        return HIPSOLVERRF_RESET_VALUES_FAST_MODE_OFF;
    case CUSOLVERRF_RESET_VALUES_FAST_MODE_ON:
        return HIPSOLVERRF_RESET_VALUES_FAST_MODE_ON;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

cusolverRfTriangularSolve_t hip2cuda_trisolve(hipsolverRfTriangularSolve_t alg)
{
    switch(alg)
    {
    case HIPSOLVERRF_TRIANGULAR_SOLVE_ALG1:
        return CUSOLVERRF_TRIANGULAR_SOLVE_ALG1;
    case HIPSOLVERRF_TRIANGULAR_SOLVE_ALG2:
        return CUSOLVERRF_TRIANGULAR_SOLVE_ALG2;
    case HIPSOLVERRF_TRIANGULAR_SOLVE_ALG3:
        return CUSOLVERRF_TRIANGULAR_SOLVE_ALG3;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverRfTriangularSolve_t cuda2hip_trisolve(cusolverRfTriangularSolve_t alg)
{
    switch(alg)
    {
    case CUSOLVERRF_TRIANGULAR_SOLVE_ALG1:
        return HIPSOLVERRF_TRIANGULAR_SOLVE_ALG1;
    case CUSOLVERRF_TRIANGULAR_SOLVE_ALG2:
        return HIPSOLVERRF_TRIANGULAR_SOLVE_ALG2;
    case CUSOLVERRF_TRIANGULAR_SOLVE_ALG3:
        return HIPSOLVERRF_TRIANGULAR_SOLVE_ALG3;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

cusolverRfUnitDiagonal_t hip2cuda_unitdiag(hipsolverRfUnitDiagonal_t diag)
{
    switch(diag)
    {
    case HIPSOLVERRF_UNIT_DIAGONAL_STORED_L:
        return CUSOLVERRF_UNIT_DIAGONAL_STORED_L;
    case HIPSOLVERRF_UNIT_DIAGONAL_STORED_U:
        return CUSOLVERRF_UNIT_DIAGONAL_STORED_U;
    case HIPSOLVERRF_UNIT_DIAGONAL_ASSUMED_L:
        return CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L;
    case HIPSOLVERRF_UNIT_DIAGONAL_ASSUMED_U:
        return CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverRfUnitDiagonal_t cuda2hip_unitdiag(cusolverRfUnitDiagonal_t diag)
{
    switch(diag)
    {
    case CUSOLVERRF_UNIT_DIAGONAL_STORED_L:
        return HIPSOLVERRF_UNIT_DIAGONAL_STORED_L;
    case CUSOLVERRF_UNIT_DIAGONAL_STORED_U:
        return HIPSOLVERRF_UNIT_DIAGONAL_STORED_U;
    case CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L:
        return HIPSOLVERRF_UNIT_DIAGONAL_ASSUMED_L;
    case CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U:
        return HIPSOLVERRF_UNIT_DIAGONAL_ASSUMED_U;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

HIPSOLVER_END_NAMESPACE
