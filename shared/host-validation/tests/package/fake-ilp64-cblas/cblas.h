// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

typedef enum CBLAS_ORDER {
    CblasRowMajor = 101,
    CblasColMajor = 102
} CBLAS_ORDER;

typedef enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113
} CBLAS_TRANSPOSE;

void cblas_sgemm(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, long long,
                 long long, long long, float, const float*, long long,
                 const float*, long long, float, float*, long long);
void cblas_dgemm(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, long long,
                 long long, long long, double, const double*, long long,
                 const double*, long long, double, double*, long long);
void cblas_cgemm(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, long long,
                 long long, long long, const void*, const void*, long long,
                 const void*, long long, const void*, void*, long long);
void cblas_zgemm(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, long long,
                 long long, long long, const void*, const void*, long long,
                 const void*, long long, const void*, void*, long long);
