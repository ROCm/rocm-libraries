/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cmath>

#include "hipsolver.h"
#include "lib_macros.hpp"
#include <hip/library_types.h>

HIPSOLVER_BEGIN_NAMESPACE

#ifdef __cplusplus
extern "C" {
#endif

void sgeev_(char*  jobvl,
            char*  jobvr,
            int*   n,
            float* A,
            int*   lda,
            float* wr,
            float* wi,
            float* vl,
            int*   ldvl,
            float* vr,
            int*   ldvr,
            float* work,
            int*   lwork,
            int*   info);
void dgeev_(char*   jobvl,
            char*   jobvr,
            int*    n,
            double* A,
            int*    lda,
            double* wr,
            double* wi,
            double* vl,
            int*    ldvl,
            double* vr,
            int*    ldvr,
            double* work,
            int*    lwork,
            int*    info);
void cgeev_(char*            jobvl,
            char*            jobvr,
            int*             n,
            hipFloatComplex* A,
            int*             lda,
            hipFloatComplex* w,
            hipFloatComplex* vl,
            int*             ldvl,
            hipFloatComplex* vr,
            int*             ldvr,
            hipFloatComplex* work,
            int*             lwork,
            float*           rwork,
            int*             info);
void zgeev_(char*             jobvl,
            char*             jobvr,
            int*              n,
            hipDoubleComplex* A,
            int*              lda,
            hipDoubleComplex* w,
            hipDoubleComplex* vl,
            int*              ldvl,
            hipDoubleComplex* vr,
            int*              ldvr,
            hipDoubleComplex* work,
            int*              lwork,
            double*           rwork,
            int*              info);

#ifdef __cplusplus
}
#endif

inline void cpu_geev(char                    jobvl,
                     char                    jobvr,
                     int                     n,
                     float*                  A,
                     int                     lda,
                     float*                  w,
                     float*                  vl,
                     int                     ldvl,
                     float*                  vr,
                     int                     ldvr,
                     float*                  work,
                     int                     lwork,
                     [[maybe_unused]] float* rwork,
                     int*                    info)
{
    // check for Infs and NaNs
    for(int j = 0; j < n; j++)
        for(int i = 0; i < n; i++)
            if(!std::isfinite(A[i + j * lda]))
                throw HIPSOLVER_STATUS_INTERNAL_ERROR;

    sgeev_(&jobvl, &jobvr, &n, A, &lda, w, w + n, vl, &ldvl, vr, &ldvr, work, &lwork, info);
}

inline void cpu_geev(char                     jobvl,
                     char                     jobvr,
                     int                      n,
                     double*                  A,
                     int                      lda,
                     double*                  w,
                     double*                  vl,
                     int                      ldvl,
                     double*                  vr,
                     int                      ldvr,
                     double*                  work,
                     int                      lwork,
                     [[maybe_unused]] double* rwork,
                     int*                     info)
{
    // check for Infs and NaNs
    for(int j = 0; j < n; j++)
        for(int i = 0; i < n; i++)
            if(!std::isfinite(A[i + j * lda]))
                throw HIPSOLVER_STATUS_INTERNAL_ERROR;

    dgeev_(&jobvl, &jobvr, &n, A, &lda, w, w + n, vl, &ldvl, vr, &ldvr, work, &lwork, info);
}

inline void cpu_geev(char             jobvl,
                     char             jobvr,
                     int              n,
                     hipFloatComplex* A,
                     int              lda,
                     hipFloatComplex* w,
                     hipFloatComplex* vl,
                     int              ldvl,
                     hipFloatComplex* vr,
                     int              ldvr,
                     hipFloatComplex* work,
                     int              lwork,
                     float*           rwork,
                     int*             info)
{
    // check for Infs and NaNs
    for(int j = 0; j < n; j++)
    {
        for(int i = 0; i < n; i++)
        {
            float re = hipCrealf(A[i + j * lda]);
            float im = hipCimagf(A[i + j * lda]);
            if(!std::isfinite(re))
                throw HIPSOLVER_STATUS_INTERNAL_ERROR;
            if(!std::isfinite(im))
                throw HIPSOLVER_STATUS_INTERNAL_ERROR;
        }
    }

    cgeev_(&jobvl, &jobvr, &n, A, &lda, w, vl, &ldvl, vr, &ldvr, work, &lwork, rwork, info);
}

inline void cpu_geev(char              jobvl,
                     char              jobvr,
                     int               n,
                     hipDoubleComplex* A,
                     int               lda,
                     hipDoubleComplex* w,
                     hipDoubleComplex* vl,
                     int               ldvl,
                     hipDoubleComplex* vr,
                     int               ldvr,
                     hipDoubleComplex* work,
                     int               lwork,
                     double*           rwork,
                     int*              info)
{
    // check for Infs and NaNs
    for(int j = 0; j < n; j++)
    {
        for(int i = 0; i < n; i++)
        {
            double re = hipCreal(A[i + j * lda]);
            double im = hipCimag(A[i + j * lda]);
            if(!std::isfinite(re))
                throw HIPSOLVER_STATUS_INTERNAL_ERROR;
            if(!std::isfinite(im))
                throw HIPSOLVER_STATUS_INTERNAL_ERROR;
        }
    }

    zgeev_(&jobvl, &jobvr, &n, A, &lda, w, vl, &ldvl, vr, &ldvr, work, &lwork, rwork, info);
}

HIPSOLVER_END_NAMESPACE
