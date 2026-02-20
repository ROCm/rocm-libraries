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
#include "rocblas/rocblas.h"
#include "rocsolver/rocsolver.h"

HIPSOLVER_BEGIN_NAMESPACE

rocblas_operation_ hip2rocblas_operation(hipsolverOperation_t op);

hipsolverOperation_t rocblas2hip_operation(rocblas_operation_ op);

rocblas_fill_ hip2rocblas_fill(hipsolverFillMode_t fill);

hipsolverFillMode_t rocblas2hip_fill(rocblas_fill_ fill);

rocblas_side_ hip2rocblas_side(hipsolverSideMode_t side);

hipsolverSideMode_t rocblas2hip_side(rocblas_side_ side);

rocblas_evect_ hip2rocblas_evect(hipsolverEigMode_t eig);

hipsolverEigMode_t rocblas2hip_evect(rocblas_evect_ eig);

rocblas_eform_ hip2rocblas_eform(hipsolverEigType_t eig);

hipsolverEigType_t rocblas2hip_eform(rocblas_eform_ eig);

rocblas_erange_ hip2rocblas_erange(hipsolverEigRange_t range);

hipsolverEigRange_t rocblas2hip_erange(rocblas_erange_ range);

rocblas_storev_ hip2rocblas_side2storev(hipsolverSideMode_t side);

rocblas_svect_ hip2rocblas_evect2svect(hipsolverEigMode_t eig, int econ);

rocblas_svect_ hip2rocblas_evect2overwrite(hipsolverEigMode_t eig, int econ);

rocblas_atomics_mode_ hip2rocblas_deterministic(hipsolverDeterministicMode_t mode);

hipsolverDeterministicMode_t rocblas2hip_deterministic(rocblas_atomics_mode_ mode);

rocblas_svect_ char2rocblas_svect(signed char svect);

hipsolverStatus_t rocblas2hip_status(rocblas_status_ error);

HIPSOLVER_END_NAMESPACE
