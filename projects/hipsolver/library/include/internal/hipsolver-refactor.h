/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef HIPSOLVER_REFACTOR_H
#define HIPSOLVER_REFACTOR_H

#include "hipsolver-types.h"

typedef void* hipsolverRfHandle_t;

typedef enum
{
    HIPSOLVERRF_FACTORIZATION_ALG0 = 0,
    HIPSOLVERRF_FACTORIZATION_ALG1 = 1,
    HIPSOLVERRF_FACTORIZATION_ALG2 = 2,
} hipsolverRfFactorization_t;

typedef enum
{
    HIPSOLVERRF_MATRIX_FORMAT_CSR = 0,
    HIPSOLVERRF_MATRIX_FORMAT_CSC = 1,
} hipsolverRfMatrixFormat_t;

typedef enum
{
    HIPSOLVERRF_NUMERIC_BOOST_NOT_USED = 0,
    HIPSOLVERRF_NUMERIC_BOOST_USED     = 1,
} hipsolverRfNumericBoostReport_t;

typedef enum
{
    HIPSOLVERRF_RESET_VALUES_FAST_MODE_OFF = 0,
    HIPSOLVERRF_RESET_VALUES_FAST_MODE_ON  = 1,
} hipsolverRfResetValuesFastMode_t;

typedef enum
{
    HIPSOLVERRF_TRIANGULAR_SOLVE_ALG1 = 1,
    HIPSOLVERRF_TRIANGULAR_SOLVE_ALG2 = 2,
    HIPSOLVERRF_TRIANGULAR_SOLVE_ALG3 = 3,
} hipsolverRfTriangularSolve_t;

typedef enum
{
    HIPSOLVERRF_UNIT_DIAGONAL_STORED_L  = 0,
    HIPSOLVERRF_UNIT_DIAGONAL_STORED_U  = 1,
    HIPSOLVERRF_UNIT_DIAGONAL_ASSUMED_L = 2,
    HIPSOLVERRF_UNIT_DIAGONAL_ASSUMED_U = 3,
} hipsolverRfUnitDiagonal_t;

#ifdef __cplusplus
extern "C" {
#endif

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfCreate(hipsolverRfHandle_t* handle);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfDestroy(hipsolverRfHandle_t handle);

// non-batched routines
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfSetupDevice(int                 n,
                                                          int                 nnzA,
                                                          int*                csrRowPtrA,
                                                          int*                csrColIndA,
                                                          double*             csrValA,
                                                          int                 nnzL,
                                                          int*                csrRowPtrL,
                                                          int*                csrColIndL,
                                                          double*             csrValL,
                                                          int                 nnzU,
                                                          int*                csrRowPtrU,
                                                          int*                csrColIndU,
                                                          double*             csrValU,
                                                          int*                P,
                                                          int*                Q,
                                                          hipsolverRfHandle_t handle);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfSetupHost(int                 n,
                                                        int                 nnzA,
                                                        int*                h_csrRowPtrA,
                                                        int*                h_csrColIndA,
                                                        double*             h_csrValA,
                                                        int                 nnzL,
                                                        int*                h_csrRowPtrL,
                                                        int*                h_csrColIndL,
                                                        double*             h_csrValL,
                                                        int                 nnzU,
                                                        int*                h_csrRowPtrU,
                                                        int*                h_csrColIndU,
                                                        double*             h_csrValU,
                                                        int*                h_P,
                                                        int*                h_Q,
                                                        hipsolverRfHandle_t handle);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfAccessBundledFactorsDevice(
    hipsolverRfHandle_t handle, int* nnzM, int** Mp, int** Mi, double** Mx);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfAnalyze(hipsolverRfHandle_t handle);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfExtractBundledFactorsHost(
    hipsolverRfHandle_t handle, int* h_nnzM, int** h_Mp, int** h_Mi, double** h_Mx);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfExtractSplitFactorsHost(hipsolverRfHandle_t handle,
                                                                      int*                h_nnzL,
                                                                      int**               h_Lp,
                                                                      int**               h_Li,
                                                                      double**            h_Lx,
                                                                      int*                h_nnzU,
                                                                      int**               h_Up,
                                                                      int**               h_Ui,
                                                                      double**            h_Ux);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfGet_Algs(hipsolverRfHandle_t           handle,
                                                       hipsolverRfFactorization_t*   fact_alg,
                                                       hipsolverRfTriangularSolve_t* solve_alg);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfGetMatrixFormat(hipsolverRfHandle_t        handle,
                                                              hipsolverRfMatrixFormat_t* format,
                                                              hipsolverRfUnitDiagonal_t* diag);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfGetNumericBoostReport(
    hipsolverRfHandle_t handle, hipsolverRfNumericBoostReport_t* report);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfGetNumericProperties(hipsolverRfHandle_t handle,
                                                                   double*             zero,
                                                                   double*             boost);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfGetResetValuesFastMode(
    hipsolverRfHandle_t handle, hipsolverRfResetValuesFastMode_t* fastMode);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfRefactor(hipsolverRfHandle_t handle);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfResetValues(int                 n,
                                                          int                 nnzA,
                                                          int*                csrRowPtrA,
                                                          int*                csrColIndA,
                                                          double*             csrValA,
                                                          int*                P,
                                                          int*                Q,
                                                          hipsolverRfHandle_t handle);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfSetAlgs(hipsolverRfHandle_t          handle,
                                                      hipsolverRfFactorization_t   fact_alg,
                                                      hipsolverRfTriangularSolve_t solve_alg);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfSetMatrixFormat(hipsolverRfHandle_t       handle,
                                                              hipsolverRfMatrixFormat_t format,
                                                              hipsolverRfUnitDiagonal_t diag);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfSetNumericProperties(hipsolverRfHandle_t handle,
                                                                   double effective_zero,
                                                                   double boost_val);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfSetResetValuesFastMode(
    hipsolverRfHandle_t handle, hipsolverRfResetValuesFastMode_t fastMode);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfSolve(hipsolverRfHandle_t handle,
                                                    int*                P,
                                                    int*                Q,
                                                    int                 nrhs,
                                                    double*             Temp,
                                                    int                 ldt,
                                                    double*             XF,
                                                    int                 ldxf);

// batched routines
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfBatchSetupHost(int                 batchSize,
                                                             int                 n,
                                                             int                 nnzA,
                                                             int*                h_csrRowPtrA,
                                                             int*                h_csrColIndA,
                                                             double*             h_csrValA_array[],
                                                             int                 nnzL,
                                                             int*                h_csrRowPtrL,
                                                             int*                h_csrColIndL,
                                                             double*             h_csrValL,
                                                             int                 nnzU,
                                                             int*                h_csrRowPtrU,
                                                             int*                h_csrColIndU,
                                                             double*             h_csrValU,
                                                             int*                h_P,
                                                             int*                h_Q,
                                                             hipsolverRfHandle_t handle);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfBatchAnalyze(hipsolverRfHandle_t handle);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfBatchRefactor(hipsolverRfHandle_t handle);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfBatchResetValues(int                 batchSize,
                                                               int                 n,
                                                               int                 nnzA,
                                                               int*                csrRowPtrA,
                                                               int*                csrColIndA,
                                                               double*             csrValA_array[],
                                                               int*                P,
                                                               int*                Q,
                                                               hipsolverRfHandle_t handle);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfBatchSolve(hipsolverRfHandle_t handle,
                                                         int*                P,
                                                         int*                Q,
                                                         int                 nrhs,
                                                         double*             Temp,
                                                         int                 ldt,
                                                         double*             XF_array[],
                                                         int                 ldxf);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfBatchZeroPivot(hipsolverRfHandle_t handle,
                                                             int*                position);

#ifdef __cplusplus
}
#endif

#endif // HIPSOLVER_REFACTOR_H
