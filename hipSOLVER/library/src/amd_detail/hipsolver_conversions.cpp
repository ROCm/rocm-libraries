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
 *  \brief Methods to convert hipSOLVER enums to and from rocBLAS/rocSOLVER enums
 */

#include "hipsolver_conversions.hpp"

HIPSOLVER_BEGIN_NAMESPACE

rocblas_operation_ hip2rocblas_operation(hipsolverOperation_t op)
{
    switch(op)
    {
    case HIPSOLVER_OP_N:
        return rocblas_operation_none;
    case HIPSOLVER_OP_T:
        return rocblas_operation_transpose;
    case HIPSOLVER_OP_C:
        return rocblas_operation_conjugate_transpose;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverOperation_t rocblas2hip_operation(rocblas_operation_ op)
{
    switch(op)
    {
    case rocblas_operation_none:
        return HIPSOLVER_OP_N;
    case rocblas_operation_transpose:
        return HIPSOLVER_OP_T;
    case rocblas_operation_conjugate_transpose:
        return HIPSOLVER_OP_C;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

rocblas_fill_ hip2rocblas_fill(hipsolverFillMode_t fill)
{
    switch(fill)
    {
    case HIPSOLVER_FILL_MODE_UPPER:
        return rocblas_fill_upper;
    case HIPSOLVER_FILL_MODE_LOWER:
        return rocblas_fill_lower;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverFillMode_t rocblas2hip_fill(rocblas_fill_ fill)
{
    switch(fill)
    {
    case rocblas_fill_upper:
        return HIPSOLVER_FILL_MODE_UPPER;
    case rocblas_fill_lower:
        return HIPSOLVER_FILL_MODE_LOWER;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

rocblas_side_ hip2rocblas_side(hipsolverSideMode_t side)
{
    switch(side)
    {
    case HIPSOLVER_SIDE_LEFT:
        return rocblas_side_left;
    case HIPSOLVER_SIDE_RIGHT:
        return rocblas_side_right;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverSideMode_t rocblas2hip_side(rocblas_side_ side)
{
    switch(side)
    {
    case rocblas_side_left:
        return HIPSOLVER_SIDE_LEFT;
    case rocblas_side_right:
        return HIPSOLVER_SIDE_RIGHT;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

rocblas_evect_ hip2rocblas_evect(hipsolverEigMode_t eig)
{
    switch(eig)
    {
    case HIPSOLVER_EIG_MODE_NOVECTOR:
        return rocblas_evect_none;
    case HIPSOLVER_EIG_MODE_VECTOR:
        return rocblas_evect_original;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverEigMode_t rocblas2hip_evect(rocblas_evect_ eig)
{
    switch(eig)
    {
    case rocblas_evect_none:
        return HIPSOLVER_EIG_MODE_NOVECTOR;
    case rocblas_evect_original:
        return HIPSOLVER_EIG_MODE_VECTOR;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

rocblas_eform_ hip2rocblas_eform(hipsolverEigType_t eig)
{
    switch(eig)
    {
    case HIPSOLVER_EIG_TYPE_1:
        return rocblas_eform_ax;
    case HIPSOLVER_EIG_TYPE_2:
        return rocblas_eform_abx;
    case HIPSOLVER_EIG_TYPE_3:
        return rocblas_eform_bax;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverEigType_t rocblas2hip_eform(rocblas_eform_ eig)
{
    switch(eig)
    {
    case rocblas_eform_ax:
        return HIPSOLVER_EIG_TYPE_1;
    case rocblas_eform_abx:
        return HIPSOLVER_EIG_TYPE_2;
    case rocblas_eform_bax:
        return HIPSOLVER_EIG_TYPE_3;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

rocblas_erange_ hip2rocblas_erange(hipsolverEigRange_t range)
{
    switch(range)
    {
    case HIPSOLVER_EIG_RANGE_ALL:
        return rocblas_erange_all;
    case HIPSOLVER_EIG_RANGE_V:
        return rocblas_erange_value;
    case HIPSOLVER_EIG_RANGE_I:
        return rocblas_erange_index;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverEigRange_t rocblas2hip_erange(rocblas_erange_ range)
{
    switch(range)
    {
    case rocblas_erange_all:
        return HIPSOLVER_EIG_RANGE_ALL;
    case rocblas_erange_value:
        return HIPSOLVER_EIG_RANGE_V;
    case rocblas_erange_index:
        return HIPSOLVER_EIG_RANGE_I;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

rocblas_storev_ hip2rocblas_side2storev(hipsolverSideMode_t side)
{
    switch(side)
    {
    case HIPSOLVER_SIDE_LEFT:
        return rocblas_column_wise;
    case HIPSOLVER_SIDE_RIGHT:
        return rocblas_row_wise;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

rocblas_svect_ hip2rocblas_evect2svect(hipsolverEigMode_t eig, int econ)
{
    switch(eig)
    {
    case HIPSOLVER_EIG_MODE_NOVECTOR:
        return rocblas_svect_none;
    case HIPSOLVER_EIG_MODE_VECTOR:
        if(econ)
            return rocblas_svect_singular;
        else
            return rocblas_svect_all;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

rocblas_svect_ hip2rocblas_evect2overwrite(hipsolverEigMode_t eig, int econ)
{
    switch(eig)
    {
    case HIPSOLVER_EIG_MODE_NOVECTOR:
        return rocblas_svect_none;
    case HIPSOLVER_EIG_MODE_VECTOR:
        if(econ)
            return rocblas_svect_overwrite;
        else
            return rocblas_svect_all;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

rocblas_atomics_mode_ hip2rocblas_deterministic(hipsolverDeterministicMode_t mode)
{
    switch(mode)
    {
    case HIPSOLVER_DETERMINISTIC_RESULTS:
        return rocblas_atomics_not_allowed;
    case HIPSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS:
        return rocblas_atomics_allowed;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverDeterministicMode_t rocblas2hip_deterministic(rocblas_atomics_mode_ mode)
{
    switch(mode)
    {
    case rocblas_atomics_not_allowed:
        return HIPSOLVER_DETERMINISTIC_RESULTS;
    case rocblas_atomics_allowed:
        return HIPSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

rocblas_svect_ char2rocblas_svect(signed char svect)
{
    switch(svect)
    {
    case 'N':
        return rocblas_svect_none;
    case 'A':
        return rocblas_svect_all;
    case 'S':
        return rocblas_svect_singular;
    case 'O':
        return rocblas_svect_overwrite;
    default:
        throw HIPSOLVER_STATUS_INVALID_VALUE;
    }
}

hipsolverStatus_t rocblas2hip_status(rocblas_status_ error)
{
    switch(error)
    {
    case rocblas_status_size_unchanged:
    case rocblas_status_size_increased:
    case rocblas_status_success:
        return HIPSOLVER_STATUS_SUCCESS;
    case rocblas_status_invalid_handle:
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    case rocblas_status_not_implemented:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    case rocblas_status_invalid_pointer:
    case rocblas_status_invalid_size:
    case rocblas_status_invalid_value:
        return HIPSOLVER_STATUS_INVALID_VALUE;
    case rocblas_status_memory_error:
        return HIPSOLVER_STATUS_ALLOC_FAILED;
    case rocblas_status_internal_error:
        return HIPSOLVER_STATUS_INTERNAL_ERROR;
    default:
        return HIPSOLVER_STATUS_UNKNOWN;
    }
}

HIPSOLVER_END_NAMESPACE
