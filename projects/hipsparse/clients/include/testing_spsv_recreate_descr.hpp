/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#ifndef TESTING_SPSV_RECREATE_DESCR_HPP
#define TESTING_SPSV_RECREATE_DESCR_HPP

#include "hipsparse_arguments.hpp"
#include "hipsparse_local_dnvec.hpp"
#include "hipsparse_local_handle.hpp"
#include "hipsparse_local_spmat.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <memory>
#include <vector>

using namespace hipsparse_test;

template <typename I, typename J, typename T>
void testing_spsv_recreate_descr_bad_arg(const Arguments& argus)
{
}

// Solves op(A) * y = alpha * x twice for one (operation, fill mode)
// configuration: once with the same descriptors for analysis and solve, and
// once with the sparse matrix and dense vector descriptors destroyed and
// recreated between analysis and solve, solving on a different handle in
// device pointer mode. Both results must agree. make_matrix creates a new
// sparse matrix descriptor on the same device arrays on every call.
template <typename T, typename MakeMatrix>
static void call_spsv_recreate_descr(hipsparseHandle_t    handle_analysis,
                                     hipsparseHandle_t    handle_solve,
                                     MakeMatrix           make_matrix,
                                     int64_t              m,
                                     const T*             dx,
                                     const T*             d_alpha,
                                     T                    h_alpha,
                                     hipsparseOperation_t op,
                                     hipsparseFillMode_t  uplo,
                                     hipsparseDiagType_t  diag,
                                     bool                 check_result)
{
    const hipDataType        typeT = getDataType<T>();
    const hipsparseSpSVAlg_t alg   = HIPSPARSE_SPSV_ALG_DEFAULT;

    auto dy_ref_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dy_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    T*   dy_ref         = (T*)dy_ref_managed.get();
    T*   dy             = (T*)dy_managed.get();

    std::vector<T> hy_init(m);
    hipsparseInit<T>(hy_init, 1, m);
    CHECK_HIP_ERROR(hipMemcpy(dy_ref, hy_init.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy_init.data(), sizeof(T) * m, hipMemcpyHostToDevice));

    auto set_attributes = [&](hipsparseSpMatDescr_t A) {
        CHECK_HIPSPARSE_ERROR(
            hipsparseSpMatSetAttribute(A, HIPSPARSE_SPMAT_FILL_MODE, &uplo, sizeof(uplo)));
        CHECK_HIPSPARSE_ERROR(
            hipsparseSpMatSetAttribute(A, HIPSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag)));
    };

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle_analysis, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle_solve, HIPSPARSE_POINTER_MODE_DEVICE));

    // Reference: the same descriptors for analysis and solve.
    {
        auto A = make_matrix();
        set_attributes(*A);
        hipsparseLocalDnVec x(m, (void*)dx, typeT);
        hipsparseLocalDnVec y(m, dy_ref, typeT);

        hipsparseSpSVDescr_t descr;
        CHECK_HIPSPARSE_ERROR(hipsparseSpSV_createDescr(&descr));

        size_t bufferSize;
        CHECK_HIPSPARSE_ERROR(hipsparseSpSV_bufferSize(
            handle_analysis, op, &h_alpha, *A, x, y, typeT, alg, descr, &bufferSize));
        auto buffer_managed = hipsparse_unique_ptr{device_malloc(bufferSize), device_free};

        CHECK_HIPSPARSE_ERROR(hipsparseSpSV_analysis(
            handle_analysis, op, &h_alpha, *A, x, y, typeT, alg, descr, buffer_managed.get()));
        CHECK_HIPSPARSE_ERROR(
            hipsparseSpSV_solve(handle_analysis, op, &h_alpha, *A, x, y, typeT, alg, descr));
        CHECK_HIPSPARSE_ERROR(hipsparseSpSV_destroyDescr(descr));
    }

    // Recreated descriptors: only the SpSV descriptor and the external buffer
    // survive from the analysis to the solve.
    hipsparseSpSVDescr_t descr;
    CHECK_HIPSPARSE_ERROR(hipsparseSpSV_createDescr(&descr));
    hipsparse_unique_ptr buffer_managed{nullptr, device_free};
    {
        auto A = make_matrix();
        set_attributes(*A);
        hipsparseLocalDnVec x(m, (void*)dx, typeT);
        hipsparseLocalDnVec y(m, dy, typeT);

        size_t bufferSize;
        CHECK_HIPSPARSE_ERROR(hipsparseSpSV_bufferSize(
            handle_analysis, op, &h_alpha, *A, x, y, typeT, alg, descr, &bufferSize));
        buffer_managed = hipsparse_unique_ptr{device_malloc(bufferSize), device_free};

        CHECK_HIPSPARSE_ERROR(hipsparseSpSV_analysis(
            handle_analysis, op, &h_alpha, *A, x, y, typeT, alg, descr, buffer_managed.get()));
    }
    {
        auto A = make_matrix();
        set_attributes(*A);
        hipsparseLocalDnVec x(m, (void*)dx, typeT);
        hipsparseLocalDnVec y(m, dy, typeT);

#if(!defined(CUDART_VERSION))
        // The operation is fixed by the analysis.
        const hipsparseOperation_t other_op = (op == HIPSPARSE_OPERATION_NON_TRANSPOSE)
                                                  ? HIPSPARSE_OPERATION_TRANSPOSE
                                                  : HIPSPARSE_OPERATION_NON_TRANSPOSE;
        verify_hipsparse_status_invalid_value(
            hipsparseSpSV_solve(handle_solve, other_op, d_alpha, *A, x, y, typeT, alg, descr),
            "Error: operation differs from the analysis");
#endif

        CHECK_HIPSPARSE_ERROR(
            hipsparseSpSV_solve(handle_solve, op, d_alpha, *A, x, y, typeT, alg, descr));
    }
    CHECK_HIPSPARSE_ERROR(hipsparseSpSV_destroyDescr(descr));

    std::vector<T> hy_ref(m);
    std::vector<T> hy(m);
    CHECK_HIP_ERROR(hipMemcpy(hy_ref.data(), dy_ref, sizeof(T) * m, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hy.data(), dy, sizeof(T) * m, hipMemcpyDeviceToHost));

    if(check_result)
    {
        unit_check_near(1, m, 1, hy_ref.data(), hy.data());
    }
}

template <typename I, typename J, typename T>
void testing_spsv_recreate_descr(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
    J                    m        = argus.M;
    J                    n        = argus.N;
    T                    h_alpha  = argus.get_alpha<T>();
    hipsparseIndexBase_t idx_base = argus.baseA;
    std::string          filename = argus.filename;

    hipsparseIndexType_t typeI = getIndexType<I>();
    hipsparseIndexType_t typeJ = getIndexType<J>();
    hipDataType          typeT = getDataType<T>();

    hipsparseLocalHandle_t handle_analysis(argus);
    hipsparseLocalHandle_t handle_solve(argus);

    std::vector<I> hcsr_row_ptr;
    std::vector<J> hcsr_col_ind;
    std::vector<T> hcsr_val;

    srand(12345ULL);

    // SpSV requires a square sparse matrix.
    n = m;

    I nnz;
    CHECK_GENERATE_MATRIX_ERROR(
        generate_csr_matrix(filename, m, n, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base));

    // COO uses a single index type for the row and column indices.
    std::vector<I> hcoo_row_ind(nnz);
    std::vector<I> hcoo_col_ind(nnz);
    for(J i = 0; i < m; ++i)
    {
        for(I j = hcsr_row_ptr[i]; j < hcsr_row_ptr[i + 1]; ++j)
        {
            hcoo_row_ind[j - idx_base] = i + idx_base;
            hcoo_col_ind[j - idx_base] = hcsr_col_ind[j - idx_base];
        }
    }

    std::vector<T> hx(m);
    hipsparseInit<T>(hx, 1, m);

    auto dptr_managed     = hipsparse_unique_ptr{device_malloc(sizeof(I) * (m + 1)), device_free};
    auto dcol_managed     = hipsparse_unique_ptr{device_malloc(sizeof(J) * nnz), device_free};
    auto dcoo_row_managed = hipsparse_unique_ptr{device_malloc(sizeof(I) * nnz), device_free};
    auto dcoo_col_managed = hipsparse_unique_ptr{device_malloc(sizeof(I) * nnz), device_free};
    auto dval_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dx_managed       = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto d_alpha_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    I* dptr     = (I*)dptr_managed.get();
    J* dcol     = (J*)dcol_managed.get();
    I* dcoo_row = (I*)dcoo_row_managed.get();
    I* dcoo_col = (I*)dcoo_col_managed.get();
    T* dval     = (T*)dval_managed.get();
    T* dx       = (T*)dx_managed.get();
    T* d_alpha  = (T*)d_alpha_managed.get();

    CHECK_HIP_ERROR(
        hipMemcpy(dptr, hcsr_row_ptr.data(), sizeof(I) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcol, hcsr_col_ind.data(), sizeof(J) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcoo_row, hcoo_row_ind.data(), sizeof(I) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcoo_col, hcoo_col_ind.data(), sizeof(I) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    auto make_csr = [&]() {
        return std::make_unique<hipsparseLocalCsr>(
            m, n, nnz, dptr, dcol, dval, typeI, typeJ, idx_base, typeT);
    };
    auto make_coo = [&]() {
        return std::make_unique<hipsparseLocalCoo>(
            m, n, nnz, dcoo_row, dcoo_col, dval, typeI, idx_base, typeT);
    };
    // The CSR arrays of A are the CSC arrays of A^T.
    auto make_csc = [&]() {
        return std::make_unique<hipsparseLocalCsc>(
            m, n, nnz, dptr, dcol, dval, typeI, typeJ, idx_base, typeT);
    };

    const std::vector<hipsparseOperation_t> ops
        = {HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE};
    const std::vector<hipsparseFillMode_t> uplos
        = {HIPSPARSE_FILL_MODE_LOWER, HIPSPARSE_FILL_MODE_UPPER};
    const hipsparseDiagType_t diag = HIPSPARSE_DIAG_TYPE_NON_UNIT;

    for(hipsparseFillMode_t uplo : uplos)
    {
        // A structural or numerical zero pivot depends only on the diagonal, which
        // is the same for A and A^T, so one host solve decides for every format and
        // operation whether the results can be compared.
        std::vector<T> hy_host(m);
        J              struct_pivot  = -1;
        J              numeric_pivot = -1;
        host_csrsv(HIPSPARSE_OPERATION_NON_TRANSPOSE,
                   m,
                   nnz,
                   h_alpha,
                   hcsr_row_ptr.data(),
                   hcsr_col_ind.data(),
                   hcsr_val.data(),
                   hx.data(),
                   hy_host.data(),
                   diag,
                   uplo,
                   idx_base,
                   &struct_pivot,
                   &numeric_pivot);
        const bool check_result = (struct_pivot == -1 && numeric_pivot == -1);

        for(hipsparseOperation_t op : ops)
        {
            call_spsv_recreate_descr<T>(handle_analysis,
                                        handle_solve,
                                        make_csr,
                                        m,
                                        dx,
                                        d_alpha,
                                        h_alpha,
                                        op,
                                        uplo,
                                        diag,
                                        check_result);
            call_spsv_recreate_descr<T>(handle_analysis,
                                        handle_solve,
                                        make_coo,
                                        m,
                                        dx,
                                        d_alpha,
                                        h_alpha,
                                        op,
                                        uplo,
                                        diag,
                                        check_result);
            call_spsv_recreate_descr<T>(handle_analysis,
                                        handle_solve,
                                        make_csc,
                                        m,
                                        dx,
                                        d_alpha,
                                        h_alpha,
                                        op,
                                        uplo,
                                        diag,
                                        check_result);
        }
    }
#endif
}

#endif // TESTING_SPSV_RECREATE_DESCR_HPP
