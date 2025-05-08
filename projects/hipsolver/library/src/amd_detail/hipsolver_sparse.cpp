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
 *  \brief Implementation of the compatibility sparse APIs that require especial
 *  calls to hipSOLVER or rocSOLVER.
 */

#include "exceptions.hpp"
#include "hipsolver.h"
#include "hipsolver_conversions.hpp"
#include "lib_macros.hpp"

#include <algorithm>
#include <climits>
#include <functional>
#include <iostream>
#include <math.h>
#include <set>
#include <vector>

#include <rocblas/internal/rocblas_device_malloc.hpp>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#include "dlopen/cholmod.hpp"
#include "dlopen/rocsparse.hpp"

extern "C" {

/******************** HANDLE ********************/
struct hipsolverSpHandle
{
    rocblas_handle   handle;
    rocsparse_handle sphandle;
    rocsolver_rfinfo rfinfo;
    cholmod_common   c_handle;

    rocblas_int h_n;
    rocblas_int d_n, d_nnzA, d_nnzT;

    rocblas_int* dPtrA;
    rocblas_int* dIndA;

    rocblas_int *dPtrT, *hPtrT;
    rocblas_int* dIndT;
    double*      dValT;

    rocblas_int* dQ;

    rocblas_int* hParent;
    rocblas_int *hWork1, *hWork2, *hWork3;

    char *d_buffer, *h_buffer;

    // Constructor
    explicit hipsolverSpHandle()
        : h_n(0)
        , d_n(0)
        , d_nnzA(0)
        , d_nnzT(0)
        , d_buffer(nullptr)
        , h_buffer(nullptr)
    {
    }

    // Allocate device memory
    hipsolverStatus_t malloc_device(int n, int nnzA, int nnzT)
    {
        if(n < 0 || nnzA < 0 || nnzT < 0)
            return HIPSOLVER_STATUS_INVALID_VALUE;

        if(this->d_n < n || this->d_nnzA < nnzA || this->d_nnzT < nnzT)
        {
            if(this->d_buffer)
            {
                if(hipFree(this->d_buffer) != hipSuccess)
                    return HIPSOLVER_STATUS_INTERNAL_ERROR;
                this->d_buffer = nullptr;
            }

            size_t size_dPtrA = sizeof(rocblas_int) * (n + 1);
            size_t size_dIndA = sizeof(rocblas_int) * nnzA;

            size_t size_dPtrT = sizeof(rocblas_int) * (n + 1);
            size_t size_dIndT = sizeof(rocblas_int) * nnzT;
            size_t size_dValT = sizeof(double) * nnzT;

            size_t size_dQ = sizeof(rocblas_int) * n;

            // 128 byte alignment
            size_dPtrA = ((size_dPtrA - 1) / 128 + 1) * 128;
            size_dIndA = ((size_dIndA - 1) / 128 + 1) * 128;
            size_dPtrT = ((size_dPtrT - 1) / 128 + 1) * 128;
            size_dIndT = ((size_dIndT - 1) / 128 + 1) * 128;
            size_dValT = ((size_dValT - 1) / 128 + 1) * 128;
            size_dQ    = ((size_dQ - 1) / 128 + 1) * 128;

            size_t size_buffer
                = size_dPtrA + size_dIndA + size_dPtrT + size_dIndT + size_dValT + size_dQ;

            if(hipMalloc(&this->d_buffer, size_buffer) != hipSuccess)
                return HIPSOLVER_STATUS_ALLOC_FAILED;

            char* temp_buf;
            this->dPtrA = (rocblas_int*)(temp_buf = this->d_buffer);
            this->dPtrT = (rocblas_int*)(temp_buf += size_dPtrA);

            this->dIndA = (rocblas_int*)(temp_buf += size_dPtrT);
            this->dIndT = (rocblas_int*)(temp_buf += size_dIndA);

            this->dQ = (rocblas_int*)(temp_buf += size_dIndT);

            this->dValT = (double*)(temp_buf += size_dQ);

            this->d_n    = n;
            this->d_nnzT = nnzT;
        }

        return HIPSOLVER_STATUS_SUCCESS;
    }

    // Allocate host memory
    hipsolverStatus_t malloc_host(int n)
    {
        if(n < 0)
            return HIPSOLVER_STATUS_INVALID_VALUE;

        if(this->h_n < n)
        {
            if(this->h_buffer)
            {
                free(this->h_buffer);
                this->h_buffer = nullptr;
            }

            size_t size_hPtrT = sizeof(rocblas_int) * (n + 1);

            size_t size_hParent = sizeof(rocblas_int) * n;
            size_t size_hWork1  = sizeof(rocblas_int) * n;
            size_t size_hWork2  = sizeof(rocblas_int) * n;
            size_t size_hWork3  = sizeof(rocblas_int) * n;

            // 128 byte alignment
            size_hPtrT   = ((size_hPtrT - 1) / 128 + 1) * 128;
            size_hParent = ((size_hParent - 1) / 128 + 1) * 128;
            size_hWork1  = ((size_hWork1 - 1) / 128 + 1) * 128;
            size_hWork2  = ((size_hWork2 - 1) / 128 + 1) * 128;
            size_hWork3  = ((size_hWork3 - 1) / 128 + 1) * 128;

            size_t size_buffer
                = size_hPtrT + size_hParent + size_hWork1 + size_hWork2 + size_hWork3;

            this->h_buffer = (char*)malloc(size_buffer);
            if(!this->h_buffer)
                return HIPSOLVER_STATUS_ALLOC_FAILED;

            char* temp_buf;
            this->hPtrT = (rocblas_int*)(temp_buf = this->h_buffer);

            this->hParent = (rocblas_int*)(temp_buf += size_hPtrT);
            this->hWork1  = (rocblas_int*)(temp_buf += size_hParent);
            this->hWork2  = (rocblas_int*)(temp_buf += size_hWork1);
            this->hWork3  = (rocblas_int*)(temp_buf += size_hWork2);

            this->h_n = n;
        }

        return HIPSOLVER_STATUS_SUCCESS;
    }

    // Free memory
    void free_all()
    {
        free(this->h_buffer);
        this->h_buffer = nullptr;

        hipFree(this->d_buffer);
        this->d_buffer = nullptr;
    }

    // Convert base one indices to base zero, and copy float values into double array
    void prep_input(rocsparse_index_base indbase,
                    int                  n,
                    int                  nnz,
                    int*                 ptr,
                    int*                 ind,
                    double*              val,
                    float*               src_val)
    {
        int count;
        if(indbase == rocsparse_index_base_one)
        {
            for(int i = 0; i <= n; i++)
                ptr[i] -= 1;
            count = std::min(nnz, ptr[n]);
            for(int i = 0; i < count; i++)
                ind[i] -= 1;
        }
        else
            count = std::min(nnz, ptr[n]);

        if(src_val)
        {
            for(int i = 0; i < count; i++)
                val[i] = (double)src_val[i];
        }
    }
    // Copy float values into double array
    void prep_input(int n, double* val, float* src_val)
    {
        if(src_val)
        {
            for(int i = 0; i < n; i++)
                val[i] = (double)src_val[i];
        }
    }

    // Convert base zero indices to base one, and copy double values into float array
    void prep_output(rocsparse_index_base indbase,
                     int                  n,
                     int                  nnz,
                     int*                 ptr,
                     int*                 ind,
                     double*              val,
                     float*               dest_val)
    {
        int count;
        if(indbase == rocsparse_index_base_one)
        {
            count = std::min(nnz, ptr[n]);
            for(int i = 0; i <= n; i++)
                ptr[i] += 1;
            for(int i = 0; i < count; i++)
                ind[i] += 1;
        }
        else
            count = std::min(nnz, ptr[n]);

        if(dest_val)
        {
            for(int i = 0; i < count; i++)
                dest_val[i] = (float)val[i];
        }
    }
    // Copy double values into float array
    void prep_output(int n, double* val, float* dest_val)
    {
        if(dest_val)
        {
            for(int i = 0; i < n; i++)
                dest_val[i] = (float)val[i];
        }
    }

    // Generates the sparsity pattern of T given the sparsity pattern of A, the elimination tree (specified by parent),
    // and the ordering (specified by new2old)
    void gen_sparsity_pattern(int               n,
                              int*              Ap,
                              int*              Ai,
                              int*              new2old,
                              int*              old2new,
                              int*              parent,
                              int*              mark,
                              int*              Tp,
                              std::vector<int>& Ti)
    {
        Tp[0] = 0;
        Ti.clear();
        for(int i = 0; i < n; i++)
        {
            old2new[new2old[i]] = i;
            mark[i]             = -1;
        }

        std::set<int> graph;
        for(int i = 0; i < n; i++)
        {
            int iold = new2old[i];
            mark[i]  = i;

            auto const kstart = Ap[iold];
            auto const kend   = Ap[iold + 1];
            for(int k = kstart; k < kend; k++)
            {

                int jold = Ai[k];
                int j    = old2new[jold];

                if(j < i)
                {
                    while((0 <= j) && (j < n) && (mark[j] != i))
                    {
                        mark[j] = i;
                        graph.insert(j);
                        j = parent[j];
                    }
                }
            }

            graph.insert(i);
            Ti.insert(Ti.end(), graph.begin(), graph.end());
            Tp[i + 1] = Ti.size();
            graph.clear();
        }
    }
};

hipsolverStatus_t hipsolverSpCreate(hipsolverSpHandle_t* handle)
try
{
#ifndef HAVE_ROCSPARSE
    if(!::hipsolver::try_load_rocsparse())
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    if(!::hipsolver::try_load_cholmod())
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
#endif // HAVE_ROCSPARSE

    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    hipsolverSpHandle* sp = new hipsolverSpHandle;
    rocblas_status     status;
    rocsparse_status   sp_status;

    if((status = rocblas_create_handle(&sp->handle)) != rocblas_status_success)
    {
        delete sp;
        return hipsolver::rocblas2hip_status(status);
    }

    if((sp_status = rocsparse_create_handle(&sp->sphandle)) != rocsparse_status_success)
    {
        rocblas_destroy_handle(sp->handle);
        delete sp;
        return HIPSOLVER_STATUS_ALLOC_FAILED;
    }

    if((status = rocsolver_create_rfinfo(&sp->rfinfo, sp->handle)) != rocblas_status_success)
    {
        rocblas_destroy_handle(sp->handle);
        rocsparse_destroy_handle(sp->sphandle);
        delete sp;
        return hipsolver::rocblas2hip_status(status);
    }

    if(cholmod_start(&sp->c_handle) != TRUE)
    {
        rocblas_destroy_handle(sp->handle);
        rocsparse_destroy_handle(sp->sphandle);
        rocsolver_destroy_rfinfo(sp->rfinfo);
        delete sp;
        return HIPSOLVER_STATUS_INTERNAL_ERROR;
    }

    *handle = sp;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverSpDestroy(hipsolverSpHandle_t handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    hipsolverSpHandle* sp = (hipsolverSpHandle*)handle;
    sp->free_all();
    rocblas_destroy_handle(sp->handle);
    rocsparse_destroy_handle(sp->sphandle);
    rocsolver_destroy_rfinfo(sp->rfinfo);
    cholmod_finish(&sp->c_handle);
    delete sp;

    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverSpSetStream(hipsolverSpHandle_t handle, hipStream_t streamId)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    hipsolverSpHandle* sp = (hipsolverSpHandle*)handle;
    return hipsolver::rocblas2hip_status(rocblas_set_stream(sp->handle, streamId));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

/******************** CSRLSVCHOL ********************/
hipsolverStatus_t hipsolverSpScsrlsvchol(hipsolverSpHandle_t       handle,
                                         int                       n,
                                         int                       nnzA,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrVal,
                                         const int*                csrRowPtr,
                                         const int*                csrColInd,
                                         const float*              b,
                                         float                     tolerance,
                                         int                       reorder,
                                         float*                    x,
                                         int*                      singularity)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(n < 0 || nnzA < 0)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtr || !csrColInd || !csrVal || !descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!b || !x || !singularity)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(reorder < 0 || reorder > 3)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    rocsparse_matrix_type mattype = rocsparse_get_mat_type((rocsparse_mat_descr)descrA);
    // rocsparse_fill_mode fillmode = rocsparse_get_mat_fill_mode((rocsparse_mat_descr)descrA);
    // rocsparse_diag_type diagtype = rocsparse_get_mat_diag_type((rocsparse_mat_descr)descrA);
    rocsparse_index_base indbase = rocsparse_get_mat_index_base((rocsparse_mat_descr)descrA);
    if(mattype != rocsparse_matrix_type_general)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    if(indbase != rocsparse_index_base_zero && indbase != rocsparse_index_base_one)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;

    hipsolverSpHandle* sp = (hipsolverSpHandle*)handle;
    *singularity          = -1;

    sp->c_handle.nmethods  = 1; // use 1 reordering method
    sp->c_handle.postorder = false; // no postordering
    sp->c_handle.final_ll  = true; // factorize as LL' not LDL'
    int ordering;
    switch(reorder)
    {
    case 1:
    case 2:
        ordering = CHOLMOD_AMD;
        break;
    case 3:
        ordering = CHOLMOD_METIS;
        break;
    default:
        ordering = CHOLMOD_NATURAL;
    }
    sp->c_handle.method[0].ordering = ordering;

    // set up A
    cholmod_sparse* c_A
        = cholmod_allocate_sparse(n, n, nnzA, true, true, 1, CHOLMOD_PATTERN, &sp->c_handle);
    CHECK_HIP_ERROR(
        hipMemcpy(c_A->p, csrRowPtr, sizeof(rocblas_int) * (n + 1), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(c_A->i, csrColInd, sizeof(rocblas_int) * nnzA, hipMemcpyDeviceToHost));
    sp->prep_input(indbase, n, nnzA, (int*)c_A->p, (int*)c_A->i, nullptr, nullptr);

    if(tolerance > 0)
        cholmod_drop(tolerance, c_A, &sp->c_handle);

    // factorize A (symbolic)
    cholmod_factor* c_L = cholmod_analyze(c_A, &sp->c_handle);

    CHECK_HIPSOLVER_ERROR(sp->malloc_host(n));
    int status = cholmod_analyze_ordering(c_A,
                                          ordering,
                                          (int*)c_L->Perm,
                                          nullptr,
                                          0,
                                          sp->hParent,
                                          sp->hWork1,
                                          nullptr,
                                          sp->hWork2,
                                          sp->hWork3,
                                          &sp->c_handle);

    std::vector<int> hIndA;
    sp->gen_sparsity_pattern(n,
                             (int*)c_A->p,
                             (int*)c_A->i,
                             (int*)c_L->Perm,
                             sp->hWork1,
                             sp->hParent,
                             sp->hWork2,
                             sp->hPtrT,
                             hIndA);

    // set up A
    int *dPtrA, *dIndA;
    if(indbase == rocsparse_index_base_zero)
    {
        // if indices are base zero, can use input arrays
        CHECK_HIPSOLVER_ERROR(sp->malloc_device(n, 0, hIndA.size()));
        dPtrA = (int*)csrRowPtr;
        dIndA = (int*)csrColInd;
    }
    else
    {
        // if indices are base one, need to use temp arrays and load base zero indices
        CHECK_HIPSOLVER_ERROR(sp->malloc_device(n, nnzA, hIndA.size()));
        dPtrA = sp->dPtrA;
        dIndA = sp->dIndA;
        CHECK_HIP_ERROR(hipMemcpy(
            (void*)sp->dPtrA, c_A->p, sizeof(rocblas_int) * (n + 1), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy((void*)sp->dIndA, c_A->i, sizeof(rocblas_int) * nnzA, hipMemcpyHostToDevice));
    }

    // set up T
    CHECK_HIP_ERROR(hipMemcpy(
        (void*)sp->dPtrT, sp->hPtrT, sizeof(rocblas_int) * (n + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        (void*)sp->dIndT, hIndA.data(), sizeof(rocblas_int) * hIndA.size(), hipMemcpyHostToDevice));

    // set up Q
    CHECK_HIP_ERROR(
        hipMemcpy((void*)sp->dQ, c_L->Perm, sizeof(rocblas_int) * n, hipMemcpyHostToDevice));

    // set up B
    CHECK_HIP_ERROR(hipMemcpy((void*)x, b, sizeof(float) * n, hipMemcpyDeviceToDevice));

    // factorize A (numeric)
    CHECK_ROCBLAS_ERROR(rocsolver_set_rfinfo_mode(sp->rfinfo, rocsolver_rfinfo_mode_cholesky));
    CHECK_ROCBLAS_ERROR(rocsolver_scsrrf_analysis(sp->handle,
                                                  n,
                                                  1,
                                                  nnzA,
                                                  dPtrA,
                                                  dIndA,
                                                  (float*)csrVal,
                                                  hIndA.size(),
                                                  sp->dPtrT,
                                                  sp->dIndT,
                                                  (float*)sp->dValT,
                                                  nullptr,
                                                  sp->dQ,
                                                  x,
                                                  n,
                                                  sp->rfinfo));
    CHECK_ROCBLAS_ERROR(rocsolver_scsrrf_refactchol(sp->handle,
                                                    n,
                                                    nnzA,
                                                    dPtrA,
                                                    dIndA,
                                                    (float*)csrVal,
                                                    hIndA.size(),
                                                    sp->dPtrT,
                                                    sp->dIndT,
                                                    (float*)sp->dValT,
                                                    sp->dQ,
                                                    sp->rfinfo));

    // solve for x
    CHECK_ROCBLAS_ERROR(rocsolver_scsrrf_solve(sp->handle,
                                               n,
                                               1,
                                               hIndA.size(),
                                               sp->dPtrT,
                                               sp->dIndT,
                                               (float*)sp->dValT,
                                               nullptr,
                                               sp->dQ,
                                               x,
                                               n,
                                               sp->rfinfo));

    // free resources
    cholmod_free_sparse(&c_A, &sp->c_handle);
    cholmod_free_factor(&c_L, &sp->c_handle);

    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverSpDcsrlsvchol(hipsolverSpHandle_t       handle,
                                         int                       n,
                                         int                       nnzA,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrVal,
                                         const int*                csrRowPtr,
                                         const int*                csrColInd,
                                         const double*             b,
                                         double                    tolerance,
                                         int                       reorder,
                                         double*                   x,
                                         int*                      singularity)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(n < 0 || nnzA < 0)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtr || !csrColInd || !csrVal || !descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!b || !x || !singularity)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(reorder < 0 || reorder > 3)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    rocsparse_matrix_type mattype = rocsparse_get_mat_type((rocsparse_mat_descr)descrA);
    // rocsparse_fill_mode fillmode = rocsparse_get_mat_fill_mode((rocsparse_mat_descr)descrA);
    // rocsparse_diag_type diagtype = rocsparse_get_mat_diag_type((rocsparse_mat_descr)descrA);
    rocsparse_index_base indbase = rocsparse_get_mat_index_base((rocsparse_mat_descr)descrA);
    if(mattype != rocsparse_matrix_type_general)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    if(indbase != rocsparse_index_base_zero && indbase != rocsparse_index_base_one)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;

    hipsolverSpHandle* sp = (hipsolverSpHandle*)handle;
    *singularity          = -1;

    sp->c_handle.nmethods  = 1; // use 1 reordering method
    sp->c_handle.postorder = false; // no postordering
    sp->c_handle.final_ll  = true; // factorize as LL' not LDL'
    int ordering;
    switch(reorder)
    {
    case 1:
    case 2:
        ordering = CHOLMOD_AMD;
        break;
    case 3:
        ordering = CHOLMOD_METIS;
        break;
    default:
        ordering = CHOLMOD_NATURAL;
    }
    sp->c_handle.method[0].ordering = ordering;

    // set up A
    cholmod_sparse* c_A
        = cholmod_allocate_sparse(n, n, nnzA, true, true, 1, CHOLMOD_PATTERN, &sp->c_handle);
    CHECK_HIP_ERROR(
        hipMemcpy(c_A->p, csrRowPtr, sizeof(rocblas_int) * (n + 1), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(c_A->i, csrColInd, sizeof(rocblas_int) * nnzA, hipMemcpyDeviceToHost));
    sp->prep_input(indbase, n, nnzA, (int*)c_A->p, (int*)c_A->i, nullptr, nullptr);

    if(tolerance > 0)
        cholmod_drop(tolerance, c_A, &sp->c_handle);

    // factorize A (symbolic)
    cholmod_factor* c_L = cholmod_analyze(c_A, &sp->c_handle);

    CHECK_HIPSOLVER_ERROR(sp->malloc_host(n));
    int status = cholmod_analyze_ordering(c_A,
                                          ordering,
                                          (int*)c_L->Perm,
                                          nullptr,
                                          0,
                                          sp->hParent,
                                          sp->hWork1,
                                          nullptr,
                                          sp->hWork2,
                                          sp->hWork3,
                                          &sp->c_handle);

    std::vector<int> hIndA;
    sp->gen_sparsity_pattern(n,
                             (int*)c_A->p,
                             (int*)c_A->i,
                             (int*)c_L->Perm,
                             sp->hWork1,
                             sp->hParent,
                             sp->hWork2,
                             sp->hPtrT,
                             hIndA);

    // set up A
    int *dPtrA, *dIndA;
    if(indbase == rocsparse_index_base_zero)
    {
        // if indices are base zero, can use input arrays
        CHECK_HIPSOLVER_ERROR(sp->malloc_device(n, 0, hIndA.size()));
        dPtrA = (int*)csrRowPtr;
        dIndA = (int*)csrColInd;
    }
    else
    {
        // if indices are base one, need to use temp arrays and load base zero indices
        CHECK_HIPSOLVER_ERROR(sp->malloc_device(n, nnzA, hIndA.size()));
        dPtrA = sp->dPtrA;
        dIndA = sp->dIndA;
        CHECK_HIP_ERROR(hipMemcpy(
            (void*)sp->dPtrA, c_A->p, sizeof(rocblas_int) * (n + 1), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy((void*)sp->dIndA, c_A->i, sizeof(rocblas_int) * nnzA, hipMemcpyHostToDevice));
    }

    // set up T
    CHECK_HIP_ERROR(hipMemcpy(
        (void*)sp->dPtrT, sp->hPtrT, sizeof(rocblas_int) * (n + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        (void*)sp->dIndT, hIndA.data(), sizeof(rocblas_int) * hIndA.size(), hipMemcpyHostToDevice));

    // set up Q
    CHECK_HIP_ERROR(
        hipMemcpy((void*)sp->dQ, c_L->Perm, sizeof(rocblas_int) * n, hipMemcpyHostToDevice));

    // set up B
    CHECK_HIP_ERROR(hipMemcpy((void*)x, b, sizeof(double) * n, hipMemcpyDeviceToDevice));

    // factorize A (numeric)
    CHECK_ROCBLAS_ERROR(rocsolver_set_rfinfo_mode(sp->rfinfo, rocsolver_rfinfo_mode_cholesky));
    CHECK_ROCBLAS_ERROR(rocsolver_dcsrrf_analysis(sp->handle,
                                                  n,
                                                  1,
                                                  nnzA,
                                                  dPtrA,
                                                  dIndA,
                                                  (double*)csrVal,
                                                  hIndA.size(),
                                                  sp->dPtrT,
                                                  sp->dIndT,
                                                  sp->dValT,
                                                  nullptr,
                                                  sp->dQ,
                                                  x,
                                                  n,
                                                  sp->rfinfo));
    CHECK_ROCBLAS_ERROR(rocsolver_dcsrrf_refactchol(sp->handle,
                                                    n,
                                                    nnzA,
                                                    dPtrA,
                                                    dIndA,
                                                    (double*)csrVal,
                                                    hIndA.size(),
                                                    sp->dPtrT,
                                                    sp->dIndT,
                                                    sp->dValT,
                                                    sp->dQ,
                                                    sp->rfinfo));

    // solve for x
    CHECK_ROCBLAS_ERROR(rocsolver_dcsrrf_solve(sp->handle,
                                               n,
                                               1,
                                               hIndA.size(),
                                               sp->dPtrT,
                                               sp->dIndT,
                                               sp->dValT,
                                               nullptr,
                                               sp->dQ,
                                               x,
                                               n,
                                               sp->rfinfo));

    // free resources
    cholmod_free_sparse(&c_A, &sp->c_handle);
    cholmod_free_factor(&c_L, &sp->c_handle);

    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

/*hipsolverStatus_t hipsolverSpCcsrlsvchol(hipsolverSpHandle_t       handle,
                                         int                       n,
                                         int                       nnzA,
                                         const hipsparseMatDescr_t descrA,
                                         const hipFloatComplex*    csrVal,
                                         const int*                csrRowPtr,
                                         const int*                csrColInd,
                                         const hipFloatComplex*    b,
                                         float                     tolerance,
                                         int                       reorder,
                                         hipFloatComplex*          x,
                                         int*                      singularity)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverSpZcsrlsvchol(hipsolverSpHandle_t       handle,
                                         int                       n,
                                         int                       nnzA,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   csrVal,
                                         const int*                csrRowPtr,
                                         const int*                csrColInd,
                                         const hipDoubleComplex*   b,
                                         double                    tolerance,
                                         int                       reorder,
                                         hipDoubleComplex*         x,
                                         int*                      singularity)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}*/

hipsolverStatus_t hipsolverSpScsrlsvcholHost(hipsolverSpHandle_t       handle,
                                             int                       n,
                                             int                       nnzA,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              csrVal,
                                             const int*                csrRowPtr,
                                             const int*                csrColInd,
                                             const float*              b,
                                             float                     tolerance,
                                             int                       reorder,
                                             float*                    x,
                                             int*                      singularity)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(n < 0 || nnzA < 0)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtr || !csrColInd || !csrVal || !descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!b || !x || !singularity)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(reorder < 0 || reorder > 3)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    rocsparse_matrix_type mattype = rocsparse_get_mat_type((rocsparse_mat_descr)descrA);
    // rocsparse_fill_mode fillmode = rocsparse_get_mat_fill_mode((rocsparse_mat_descr)descrA);
    // rocsparse_diag_type diagtype = rocsparse_get_mat_diag_type((rocsparse_mat_descr)descrA);
    rocsparse_index_base indbase = rocsparse_get_mat_index_base((rocsparse_mat_descr)descrA);
    if(mattype != rocsparse_matrix_type_general)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    if(indbase != rocsparse_index_base_zero && indbase != rocsparse_index_base_one)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;

    hipsolverSpHandle* sp = (hipsolverSpHandle*)handle;
    *singularity          = -1;

    sp->c_handle.nmethods  = 1; // use 1 reordering method
    sp->c_handle.postorder = false; // no postordering
    sp->c_handle.final_ll  = true; // factorize as LL' not LDL'
    switch(reorder)
    {
    case 1:
    case 2:
        sp->c_handle.method[0].ordering = CHOLMOD_AMD;
        break;
    case 3:
        sp->c_handle.method[0].ordering = CHOLMOD_METIS;
        break;
    default:
        sp->c_handle.method[0].ordering = CHOLMOD_NATURAL;
    }

    // set up A
    float*          sngVal = (float*)malloc(sizeof(float) * nnzA);
    cholmod_sparse* c_A
        = cholmod_allocate_sparse(n, n, nnzA, true, true, 1, CHOLMOD_REAL, &sp->c_handle);
    memcpy(c_A->p, csrRowPtr, sizeof(rocblas_int) * (n + 1));
    memcpy(c_A->i, csrColInd, sizeof(rocblas_int) * nnzA);
    sp->prep_input(indbase, n, nnzA, (int*)c_A->p, (int*)c_A->i, (double*)c_A->x, (float*)csrVal);
    free(sngVal);

    if(tolerance > 0)
        cholmod_drop(tolerance, c_A, &sp->c_handle);

    // factorize A
    cholmod_factor* c_L    = cholmod_analyze(c_A, &sp->c_handle);
    int             status = cholmod_factorize(c_A, c_L, &sp->c_handle);
    if(status != TRUE)
    {
        cholmod_free_sparse(&c_A, &sp->c_handle);
        cholmod_free_factor(&c_L, &sp->c_handle);
        return HIPSOLVER_STATUS_INTERNAL_ERROR;
    }
    if(sp->c_handle.status == CHOLMOD_NOT_POSDEF)
    {
        *singularity = c_L->minor;
        cholmod_free_sparse(&c_A, &sp->c_handle);
        cholmod_free_factor(&c_L, &sp->c_handle);
        return HIPSOLVER_STATUS_SUCCESS;
    }

    // set up B
    cholmod_dense* c_b = cholmod_allocate_dense(n, 1, n, CHOLMOD_REAL, &sp->c_handle);
    sp->prep_input(n, (double*)c_b->x, (float*)b);

    // solve for x
    cholmod_dense* c_x = cholmod_solve(CHOLMOD_A, c_L, c_b, &sp->c_handle);

    // copy back results
    sp->prep_output(n, (double*)c_x->x, (float*)x);

    // free resources
    cholmod_free_sparse(&c_A, &sp->c_handle);
    cholmod_free_factor(&c_L, &sp->c_handle);
    cholmod_free_dense(&c_b, &sp->c_handle);
    cholmod_free_dense(&c_x, &sp->c_handle);

    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverSpDcsrlsvcholHost(hipsolverSpHandle_t       handle,
                                             int                       n,
                                             int                       nnzA,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             csrVal,
                                             const int*                csrRowPtr,
                                             const int*                csrColInd,
                                             const double*             b,
                                             double                    tolerance,
                                             int                       reorder,
                                             double*                   x,
                                             int*                      singularity)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(n < 0 || nnzA < 0)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtr || !csrColInd || !csrVal || !descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!b || !x || !singularity)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(reorder < 0 || reorder > 3)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    rocsparse_matrix_type mattype = rocsparse_get_mat_type((rocsparse_mat_descr)descrA);
    // rocsparse_fill_mode fillmode = rocsparse_get_mat_fill_mode((rocsparse_mat_descr)descrA);
    // rocsparse_diag_type diagtype = rocsparse_get_mat_diag_type((rocsparse_mat_descr)descrA);
    rocsparse_index_base indbase = rocsparse_get_mat_index_base((rocsparse_mat_descr)descrA);
    if(mattype != rocsparse_matrix_type_general)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    if(indbase != rocsparse_index_base_zero && indbase != rocsparse_index_base_one)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;

    hipsolverSpHandle* sp = (hipsolverSpHandle*)handle;
    *singularity          = -1;

    sp->c_handle.nmethods  = 1; // use 1 reordering method
    sp->c_handle.postorder = false; // no postordering
    sp->c_handle.final_ll  = true; // factorize as LL' not LDL'
    switch(reorder)
    {
    case 1:
    case 2:
        sp->c_handle.method[0].ordering = CHOLMOD_AMD;
        break;
    case 3:
        sp->c_handle.method[0].ordering = CHOLMOD_METIS;
        break;
    default:
        sp->c_handle.method[0].ordering = CHOLMOD_NATURAL;
    }

    // set up A
    cholmod_sparse* c_A
        = cholmod_allocate_sparse(n, n, nnzA, true, true, 1, CHOLMOD_REAL, &sp->c_handle);
    memcpy(c_A->p, csrRowPtr, sizeof(rocblas_int) * (n + 1));
    memcpy(c_A->i, csrColInd, sizeof(rocblas_int) * nnzA);
    memcpy(c_A->x, csrVal, sizeof(double) * nnzA);
    sp->prep_input(indbase, n, nnzA, (int*)c_A->p, (int*)c_A->i, (double*)c_A->x, nullptr);

    if(tolerance > 0)
        cholmod_drop(tolerance, c_A, &sp->c_handle);

    // factorize A
    cholmod_factor* c_L    = cholmod_analyze(c_A, &sp->c_handle);
    int             status = cholmod_factorize(c_A, c_L, &sp->c_handle);
    if(status != TRUE)
    {
        cholmod_free_sparse(&c_A, &sp->c_handle);
        cholmod_free_factor(&c_L, &sp->c_handle);
        return HIPSOLVER_STATUS_INTERNAL_ERROR;
    }
    if(sp->c_handle.status == CHOLMOD_NOT_POSDEF)
    {
        *singularity = c_L->minor;
        cholmod_free_sparse(&c_A, &sp->c_handle);
        cholmod_free_factor(&c_L, &sp->c_handle);
        return HIPSOLVER_STATUS_SUCCESS;
    }

    // set up B
    cholmod_dense* c_b = cholmod_allocate_dense(n, 1, n, CHOLMOD_REAL, &sp->c_handle);
    memcpy(c_b->x, b, sizeof(double) * n);

    // solve for x
    cholmod_dense* c_x = cholmod_solve(CHOLMOD_A, c_L, c_b, &sp->c_handle);

    // copy back results
    memcpy((void*)x, c_x->x, sizeof(double) * n);

    // free resources
    cholmod_free_sparse(&c_A, &sp->c_handle);
    cholmod_free_factor(&c_L, &sp->c_handle);
    cholmod_free_dense(&c_b, &sp->c_handle);
    cholmod_free_dense(&c_x, &sp->c_handle);

    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

/*hipsolverStatus_t hipsolverSpCcsrlsvcholHost(hipsolverSpHandle_t       handle,
                                             int                       n,
                                             int                       nnzA,
                                             const hipsparseMatDescr_t descrA,
                                             const hipFloatComplex*    csrVal,
                                             const int*                csrRowPtr,
                                             const int*                csrColInd,
                                             const hipFloatComplex*    b,
                                             float                     tolerance,
                                             int                       reorder,
                                             hipFloatComplex*          x,
                                             int*                      singularity)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverSpZcsrlsvcholHost(hipsolverSpHandle_t       handle,
                                             int                       n,
                                             int                       nnzA,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   csrVal,
                                             const int*                csrRowPtr,
                                             const int*                csrColInd,
                                             const hipDoubleComplex*   b,
                                             double                    tolerance,
                                             int                       reorder,
                                             hipDoubleComplex*         x,
                                             int*                      singularity)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}*/

/******************** CSRLSVQR ********************/
hipsolverStatus_t hipsolverSpScsrlsvqr(hipsolverSpHandle_t       handle,
                                       int                       n,
                                       int                       nnz,
                                       const hipsparseMatDescr_t descrA,
                                       const float*              csrVal,
                                       const int*                csrRowPtr,
                                       const int*                csrColInd,
                                       const float*              b,
                                       double                    tolerance,
                                       int                       reorder,
                                       float*                    x,
                                       int*                      singularity)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(n < 0 || nnz < 0)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtr || !csrColInd || !csrVal || !descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!b || !x || !singularity)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    // if(reorder < 0 || reorder > 3)
    //     return HIPSOLVER_STATUS_INVALID_VALUE;

    rocsparse_matrix_type mattype = rocsparse_get_mat_type((rocsparse_mat_descr)descrA);
    rocsparse_index_base  indbase = rocsparse_get_mat_index_base((rocsparse_mat_descr)descrA);
    if(mattype != rocsparse_matrix_type_general)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    if(indbase != rocsparse_index_base_zero && indbase != rocsparse_index_base_one)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;

    hipsolverSpHandle* sp = (hipsolverSpHandle*)handle;
    *singularity          = -1;

    // set up B
    CHECK_HIP_ERROR(hipMemcpy((void*)x, b, sizeof(float) * n, hipMemcpyDeviceToDevice));

    // convert A to dense matrix
    float* denseA;
    CHECK_HIP_ERROR(hipMalloc(&denseA, sizeof(float) * n * n));
    rocsparse_scsr2dense(
        sp->sphandle, n, n, (rocsparse_mat_descr)descrA, csrVal, csrRowPtr, csrColInd, denseA, n);

    int* info;
    CHECK_HIP_ERROR(hipMalloc(&info, sizeof(int)));

    rocblas_status st
        = rocsolver_sgels(sp->handle, rocblas_operation_none, n, n, 1, denseA, n, x, n, info);

    // finalize singularity
    CHECK_HIP_ERROR(hipMemcpy((void*)singularity, info, sizeof(int), hipMemcpyDeviceToHost));
    *singularity = *singularity - 1;

    CHECK_HIP_ERROR(hipFree(denseA));
    CHECK_HIP_ERROR(hipFree(info));

    return hipsolver::rocblas2hip_status(st);
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverSpDcsrlsvqr(hipsolverSpHandle_t       handle,
                                       int                       n,
                                       int                       nnz,
                                       const hipsparseMatDescr_t descrA,
                                       const double*             csrVal,
                                       const int*                csrRowPtr,
                                       const int*                csrColInd,
                                       const double*             b,
                                       double                    tolerance,
                                       int                       reorder,
                                       double*                   x,
                                       int*                      singularity)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(n < 0 || nnz < 0)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtr || !csrColInd || !csrVal || !descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!b || !x || !singularity)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    // if(reorder < 0 || reorder > 3)
    //     return HIPSOLVER_STATUS_INVALID_VALUE;

    rocsparse_matrix_type mattype = rocsparse_get_mat_type((rocsparse_mat_descr)descrA);
    rocsparse_index_base  indbase = rocsparse_get_mat_index_base((rocsparse_mat_descr)descrA);
    if(mattype != rocsparse_matrix_type_general)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    if(indbase != rocsparse_index_base_zero && indbase != rocsparse_index_base_one)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;

    hipsolverSpHandle* sp = (hipsolverSpHandle*)handle;
    *singularity          = -1;

    // set up B
    CHECK_HIP_ERROR(hipMemcpy((void*)x, b, sizeof(double) * n, hipMemcpyDeviceToDevice));

    // convert A to dense matrix
    double* denseA;
    CHECK_HIP_ERROR(hipMalloc(&denseA, sizeof(double) * n * n));
    rocsparse_dcsr2dense(
        sp->sphandle, n, n, (rocsparse_mat_descr)descrA, csrVal, csrRowPtr, csrColInd, denseA, n);

    int* info;
    CHECK_HIP_ERROR(hipMalloc(&info, sizeof(int)));

    rocblas_status st
        = rocsolver_dgels(sp->handle, rocblas_operation_none, n, n, 1, denseA, n, x, n, info);

    // finalize singularity
    CHECK_HIP_ERROR(hipMemcpy((void*)singularity, info, sizeof(int), hipMemcpyDeviceToHost));
    *singularity = *singularity - 1;

    CHECK_HIP_ERROR(hipFree(denseA));
    CHECK_HIP_ERROR(hipFree(info));

    return hipsolver::rocblas2hip_status(st);
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

/*hipsolverStatus_t hipsolverSpCcsrlsvqr(hipsolverSpHandle_t       handle,
                                       int                       n,
                                       int                       nnz,
                                       const hipsparseMatDescr_t descrA,
                                       const hipFloatComplex*    csrVal,
                                       const int*                csrRowPtr,
                                       const int*                csrColInd,
                                       const hipFloatComplex*    b,
                                       double                    tolerance,
                                       int                       reorder,
                                       hipFloatComplex*          x,
                                       int*                      singularity)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(n < 0 || nnz < 0)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtr || !csrColInd || !csrVal || !descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!b || !x || !singularity)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(reorder < 0 || reorder > 3)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    rocsparse_matrix_type mattype = rocsparse_get_mat_type((rocsparse_mat_descr)descrA);
    rocsparse_index_base  indbase = rocsparse_get_mat_index_base((rocsparse_mat_descr)descrA);
    if(mattype != rocsparse_matrix_type_general)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    if(indbase != rocsparse_index_base_zero && indbase != rocsparse_index_base_one)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;

    hipsolverSpHandle* sp = (hipsolverSpHandle*)handle;
    *singularity          = -1;

    // set up B
    CHECK_HIP_ERROR(hipMemcpy((void*)x, b, sizeof(hipFloatComplex) * n, hipMemcpyDeviceToDevice));

    // convert A to dense matrix
    hipFloatComplex* denseA;
    CHECK_HIP_ERROR(hipMalloc(&denseA, sizeof(hipFloatComplex) * n * n));
    rocsparse_ccsr2dense(sp->sphandle,
                         n,
                         n,
                         (rocsparse_mat_descr)descrA,
                         (rocsparse_float_complex*)csrVal,
                         csrRowPtr,
                         csrColInd,
                         (rocsparse_float_complex*)denseA,
                         n);

    int* info;
    CHECK_HIP_ERROR(hipMalloc(&info, sizeof(int)));

    rocblas_status st = rocsolver_cgels(sp->handle,
                                        rocblas_operation_none,
                                        n,
                                        n,
                                        1,
                                        (rocblas_float_complex*)denseA,
                                        n,
                                        (rocblas_float_complex*)x,
                                        n,
                                        info);

    // finalize singularity
    CHECK_HIP_ERROR(hipMemcpy((void*)singularity, info, sizeof(int), hipMemcpyDeviceToHost));
    *singularity = *singularity - 1;

    CHECK_HIP_ERROR(hipFree(denseA));
    CHECK_HIP_ERROR(hipFree(info));

    return hipsolver::rocblas2hip_status(st);
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverSpZcsrlsvqr(hipsolverSpHandle_t       handle,
                                       int                       n,
                                       int                       nnz,
                                       const hipsparseMatDescr_t descrA,
                                       const hipDoubleComplex*   csrVal,
                                       const int*                csrRowPtr,
                                       const int*                csrColInd,
                                       const hipDoubleComplex*   b,
                                       double                    tolerance,
                                       int                       reorder,
                                       hipDoubleComplex*         x,
                                       int*                      singularity)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(n < 0 || nnz < 0)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtr || !csrColInd || !csrVal || !descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!b || !x || !singularity)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(reorder < 0 || reorder > 3)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    rocsparse_matrix_type mattype = rocsparse_get_mat_type((rocsparse_mat_descr)descrA);
    rocsparse_index_base  indbase = rocsparse_get_mat_index_base((rocsparse_mat_descr)descrA);
    if(mattype != rocsparse_matrix_type_general)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    if(indbase != rocsparse_index_base_zero && indbase != rocsparse_index_base_one)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;

    hipsolverSpHandle* sp = (hipsolverSpHandle*)handle;
    *singularity          = -1;

    // set up B
    CHECK_HIP_ERROR(hipMemcpy((void*)x, b, sizeof(hipDoubleComplex) * n, hipMemcpyDeviceToDevice));

    // convert A to dense matrix
    hipFloatComplex* denseA;
    CHECK_HIP_ERROR(hipMalloc(&denseA, sizeof(hipDoubleComplex) * n * n));
    rocsparse_zcsr2dense(sp->sphandle,
                         n,
                         n,
                         (rocsparse_mat_descr)descrA,
                         (rocsparse_double_complex*)csrVal,
                         csrRowPtr,
                         csrColInd,
                         (rocsparse_double_complex*)denseA,
                         n);

    int* info;
    CHECK_HIP_ERROR(hipMalloc(&info, sizeof(int)));

    rocblas_status st = rocsolver_zgels(sp->handle,
                                        rocblas_operation_none,
                                        n,
                                        n,
                                        1,
                                        (rocblas_double_complex*)denseA,
                                        n,
                                        (rocblas_double_complex*)x,
                                        n,
                                        info);

    // finalize singularity
    CHECK_HIP_ERROR(hipMemcpy((void*)singularity, info, sizeof(int), hipMemcpyDeviceToHost));
    *singularity = *singularity - 1;

    CHECK_HIP_ERROR(hipFree(denseA));
    CHECK_HIP_ERROR(hipFree(info));

    return hipsolver::rocblas2hip_status(st);
}
catch(...)
{
    return hipsolver::exception2hip_status();
}*/

} //extern C
