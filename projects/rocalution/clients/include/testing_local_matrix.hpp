/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_LOCAL_MATRIX_HPP
#define TESTING_LOCAL_MATRIX_HPP

#include "utility.hpp"

#include <gtest/gtest.h>
#include <rocalution/rocalution.hpp>

using namespace rocalution;

template <typename T>
void testing_local_matrix_bad_args(void)
{
    int safe_size = 100;

    // Initialize rocALUTION
    set_device_rocalution(device);
    init_rocalution();

    LocalMatrix<T>       mat1;
    LocalMatrix<T>       mat2;
    LocalVector<T>       vec1;
    LocalVector<bool>    bool1;
    LocalVector<int>     int1;
    LocalVector<int64_t> int641;

    // null pointers
    int* null_int  = nullptr;
    T*   null_data = nullptr;

    // Valid pointers
    int* vint  = nullptr;
    T*   vdata = nullptr;

    allocate_host(safe_size, &vint);
    allocate_host(safe_size, &vdata);

    // Valid matrices
    LocalMatrix<T> mat3;
    mat3.AllocateCSR("valid", safe_size, safe_size, safe_size);

    // ExtractSubMatrix, ExtractSubMatrices, Extract(Inverse)Diagonal, ExtractL/U
    {
        LocalMatrix<T>*   mat_null  = nullptr;
        LocalMatrix<T>**  mat_null2 = nullptr;
        LocalMatrix<T>*** pmat      = new LocalMatrix<T>**[1];
        pmat[0]                     = new LocalMatrix<T>*[1];
        pmat[0][0]                  = new LocalMatrix<T>;
        LocalVector<T>* null_vec    = nullptr;
        ASSERT_DEATH(mat1.ExtractSubMatrix(0, 0, safe_size, safe_size, mat_null),
                     ".*Assertion.*mat != (NULL|__null)*");
        ASSERT_DEATH(mat1.ExtractSubMatrices(safe_size, safe_size, null_int, vint, pmat),
                     ".*Assertion.*row_offset != (NULL|__null)*");
        ASSERT_DEATH(mat1.ExtractSubMatrices(safe_size, safe_size, vint, null_int, pmat),
                     ".*Assertion.*col_offset != (NULL|__null)*");
        ASSERT_DEATH(mat1.ExtractSubMatrices(safe_size, safe_size, vint, vint, nullptr),
                     ".*Assertion.*mat != (NULL|__null)*");
        ASSERT_DEATH(mat1.ExtractSubMatrices(safe_size, safe_size, vint, vint, &mat_null2),
                     ".*Assertion.*mat != (NULL|__null)*");
        ASSERT_DEATH(mat1.ExtractDiagonal(null_vec), ".*Assertion.*vec_diag != (NULL|__null)*");
        ASSERT_DEATH(mat1.ExtractInverseDiagonal(null_vec),
                     ".*Assertion.*vec_inv_diag != (NULL|__null)*");
        ASSERT_DEATH(mat1.ExtractL(mat_null, true), ".*Assertion.*L != (NULL|__null)*");
        ASSERT_DEATH(mat1.ExtractU(mat_null, true), ".*Assertion.*U != (NULL|__null)*");
        delete pmat[0][0];
        delete[] pmat[0];
        delete[] pmat;
    }

    // CMK, RCMK, ConnectivityOrder, MultiColoring, MaximalIndependentSet, ZeroBlockPermutation
    {
        int               val;
        LocalVector<int>* null_vec = nullptr;
        ASSERT_DEATH(mat1.CMK(null_vec), ".*Assertion.*permutation != (NULL|__null)*");
        ASSERT_DEATH(mat1.RCMK(null_vec), ".*Assertion.*permutation != (NULL|__null)*");
        ASSERT_DEATH(mat1.ConnectivityOrder(null_vec),
                     ".*Assertion.*permutation != (NULL|__null)*");
        ASSERT_DEATH(mat1.MultiColoring(val, &vint, &int1),
                     ".*Assertion.*size_colors == (NULL|__null)*");
        ASSERT_DEATH(mat1.MultiColoring(val, &null_int, null_vec),
                     ".*Assertion.*permutation != (NULL|__null)*");
        ASSERT_DEATH(mat1.MaximalIndependentSet(val, null_vec),
                     ".*Assertion.*permutation != (NULL|__null)*");
        ASSERT_DEATH(mat1.ZeroBlockPermutation(val, null_vec),
                     ".*Assertion.*permutation != (NULL|__null)*");
    }

    // LSolve, USolve, LLSolve, LUSolve, QRSolve
    {
        LocalVector<T>* null_vec = nullptr;
        ASSERT_DEATH(mat1.LSolve(vec1, null_vec), ".*Assertion.*out != (NULL|__null)*");
        ASSERT_DEATH(mat1.USolve(vec1, null_vec), ".*Assertion.*out != (NULL|__null)*");
        ASSERT_DEATH(mat1.LLSolve(vec1, null_vec), ".*Assertion.*out != (NULL|__null)*");
        ASSERT_DEATH(mat1.LLSolve(vec1, vec1, null_vec), ".*Assertion.*out != (NULL|__null)*");
        ASSERT_DEATH(mat1.LUSolve(vec1, null_vec), ".*Assertion.*out != (NULL|__null)*");
        ASSERT_DEATH(mat1.QRSolve(vec1, null_vec), ".*Assertion.*out != (NULL|__null)*");
    }

    // ItLSolve, ItUSolve, ItLLSolve, ItLUSolve
    {
        LocalVector<T>* null_vec = nullptr;
        int             max_iter = 1;
        double          tol      = 0;
        ASSERT_DEATH(mat1.ItLSolve(max_iter, tol, true, vec1, null_vec),
                     ".*Assertion.*out != (NULL|__null)*");
        ASSERT_DEATH(mat1.ItUSolve(max_iter, tol, true, vec1, null_vec),
                     ".*Assertion.*out != (NULL|__null)*");
        ASSERT_DEATH(mat1.ItLLSolve(max_iter, tol, true, vec1, null_vec),
                     ".*Assertion.*out != (NULL|__null)*");
        ASSERT_DEATH(mat1.ItLLSolve(max_iter, tol, true, vec1, vec1, null_vec),
                     ".*Assertion.*out != (NULL|__null)*");
        ASSERT_DEATH(mat1.ItLUSolve(max_iter, tol, true, vec1, null_vec),
                     ".*Assertion.*out != (NULL|__null)*");
    }

    // ICFactorize, Householder
    {
        T               val;
        LocalVector<T>* null_vec = nullptr;
        ASSERT_DEATH(mat1.ICFactorize(null_vec), ".*Assertion.*inv_diag != (NULL|__null)*");
        ASSERT_DEATH(mat1.Householder(0, val, null_vec), ".*Assertion.*vec != (NULL|__null)*");
    }

    // CopyFrom functions
    {
        ASSERT_DEATH(mat1.UpdateValuesCSR(null_data), ".*Assertion.*val != (NULL|__null)*");
        ASSERT_DEATH(mat1.CopyFromCSR(null_int, vint, vdata),
                     ".*Assertion.*row_offsets != (NULL|__null)*");
        ASSERT_DEATH(mat3.CopyFromCSR(vint, null_int, vdata), ".*Assertion.*col != (NULL|__null)*");
        ASSERT_DEATH(mat3.CopyFromCSR(vint, vint, null_data), ".*Assertion.*val != (NULL|__null)*");
        ASSERT_DEATH(mat1.CopyToCSR(null_int, vint, vdata),
                     ".*Assertion.*row_offsets != (NULL|__null)*");
        ASSERT_DEATH(mat3.CopyToCSR(vint, null_int, vdata), ".*Assertion.*col != (NULL|__null)*");
        ASSERT_DEATH(mat3.CopyToCSR(vint, vint, null_data), ".*Assertion.*val != (NULL|__null)*");
        ASSERT_DEATH(mat1.CopyFromCOO(null_int, vint, vdata), ".*Assertion.*row != (NULL|__null)*");
        ASSERT_DEATH(mat1.CopyFromCOO(vint, null_int, vdata), ".*Assertion.*col != (NULL|__null)*");
        ASSERT_DEATH(mat1.CopyFromCOO(vint, vint, null_data), ".*Assertion.*val != (NULL|__null)*");
        ASSERT_DEATH(mat1.CopyToCOO(null_int, vint, vdata), ".*Assertion.*row != (NULL|__null)*");
        ASSERT_DEATH(mat1.CopyToCOO(vint, null_int, vdata), ".*Assertion.*col != (NULL|__null)*");
        ASSERT_DEATH(mat1.CopyToCOO(vint, vint, null_data), ".*Assertion.*val != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.CopyFromHostCSR(null_int, vint, vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*row_offset != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.CopyFromHostCSR(vint, null_int, vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*col != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.CopyFromHostCSR(vint, vint, null_data, "", safe_size, safe_size, safe_size),
            ".*Assertion.*val != (NULL|__null)*");
    }

    // CreateFromMat
    {
        LocalMatrix<T>* null_mat = nullptr;
        ASSERT_DEATH(mat1.CreateFromMap(int1, safe_size, safe_size, null_mat),
                     ".*Assertion.*pro != (NULL|__null)*");
    }

    // Apply(Add)
    {
        LocalVector<T>* null_vec = nullptr;
        ASSERT_DEATH(mat1.Apply(vec1, null_vec), ".*Assertion.*out != (NULL|__null)*");
        ASSERT_DEATH(mat1.ApplyAdd(vec1, 1.0, null_vec), ".*Assertion.*out != (NULL|__null)*");
    }

    // Row/Column manipulation
    {
        LocalVector<T>* null_vec = nullptr;
        ASSERT_DEATH(mat1.ExtractColumnVector(0, null_vec), ".*Assertion.*vec != (NULL|__null)*");
        ASSERT_DEATH(mat1.ExtractRowVector(0, null_vec), ".*Assertion.*vec != (NULL|__null)*");
    }

    // AMG
    {
        int                   val;
        LocalVector<bool>*    bool_null_vec  = nullptr;
        LocalVector<int64_t>* int64_null_vec = nullptr;
        ASSERT_DEATH(mat1.AMGGreedyAggregate(0.1, bool_null_vec, &int641, &int641),
                     ".*Assertion.*connections != (NULL|__null)*");
        ASSERT_DEATH(mat1.AMGGreedyAggregate(0.1, &bool1, int64_null_vec, &int641),
                     ".*Assertion.*aggregates != (NULL|__null)*");
        ASSERT_DEATH(mat1.AMGGreedyAggregate(0.1, &bool1, &int641, int64_null_vec),
                     ".*Assertion.*aggregate_root_nodes != (NULL|__null)*");

        LocalMatrix<T>* null_mat = nullptr;
        ASSERT_DEATH(mat1.AMGSmoothedAggregation(0.1, bool1, int641, int641, null_mat),
                     ".*Assertion.*prolong != (NULL|__null)*");
    }

    {
        int               val;
        LocalVector<int>* null_vec = nullptr;
        LocalMatrix<T>*   null_mat = nullptr;
        ASSERT_DEATH(mat1.AMGUnsmoothedAggregation(int641, int641, null_mat),
                     ".*Assertion.*prolong != (NULL|__null)*");
        ASSERT_DEATH(mat1.InitialPairwiseAggregation(0.1, val, null_vec, val, &null_int, val, 0),
                     ".*Assertion.*G != (NULL|__null)*");
        ASSERT_DEATH(mat1.InitialPairwiseAggregation(0.1, val, &int1, val, &vint, val, 0),
                     ".*Assertion.*rG == (NULL|__null)*");
        ASSERT_DEATH(
            mat1.InitialPairwiseAggregation(mat2, 0.1, val, null_vec, val, &null_int, val, 0),
            ".*Assertion.*G != (NULL|__null)*");
        ASSERT_DEATH(mat1.InitialPairwiseAggregation(mat2, 0.1, val, &int1, val, &vint, val, 0),
                     ".*Assertion.*rG == (NULL|__null)*");
        ASSERT_DEATH(mat1.FurtherPairwiseAggregation(0.1, val, null_vec, val, &vint, val, 0),
                     ".*Assertion.*G != (NULL|__null)*");
        ASSERT_DEATH(mat1.FurtherPairwiseAggregation(0.1, val, &int1, val, &null_int, val, 0),
                     ".*Assertion.*rG != (NULL|__null)*");
        ASSERT_DEATH(mat1.FurtherPairwiseAggregation(mat2, 0.1, val, null_vec, val, &vint, val, 0),
                     ".*Assertion.*G != (NULL|__null)*");
        ASSERT_DEATH(mat1.FurtherPairwiseAggregation(mat2, 0.1, val, &int1, val, &null_int, val, 0),
                     ".*Assertion.*rG != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.CoarsenOperator(null_mat, safe_size, safe_size, int1, safe_size, vint, safe_size),
            ".*Assertion.*Ac != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.CoarsenOperator(&mat2, safe_size, safe_size, int1, safe_size, null_int, safe_size),
            ".*Assertion.*rG != (NULL|__null)*");
    }

    // SetDataPtr
    {
        ASSERT_DEATH(
            mat1.SetDataPtrCOO(&null_int, &vint, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*row != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.SetDataPtrCOO(&vint, &null_int, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*col != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.SetDataPtrCOO(&vint, &vint, &null_data, "", safe_size, safe_size, safe_size),
            ".*Assertion.*val != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.SetDataPtrCOO(nullptr, &vint, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*row != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.SetDataPtrCOO(&vint, nullptr, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*col != (NULL|__null)*");
        ASSERT_DEATH(mat1.SetDataPtrCOO(&vint, &vint, nullptr, "", safe_size, safe_size, safe_size),
                     ".*Assertion.*val != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.SetDataPtrCSR(&null_int, &vint, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*row_offset != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.SetDataPtrCSR(&vint, &null_int, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*col != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.SetDataPtrCSR(&vint, &vint, &null_data, "", safe_size, safe_size, safe_size),
            ".*Assertion.*val != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.SetDataPtrCSR(nullptr, &vint, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*row_offset != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.SetDataPtrCSR(&vint, nullptr, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*col != (NULL|__null)*");
        ASSERT_DEATH(mat1.SetDataPtrCSR(&vint, &vint, nullptr, "", safe_size, safe_size, safe_size),
                     ".*Assertion.*val != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.SetDataPtrMCSR(&null_int, &vint, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*row_offset != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.SetDataPtrMCSR(&vint, &null_int, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*col != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.SetDataPtrMCSR(&vint, &vint, &null_data, "", safe_size, safe_size, safe_size),
            ".*Assertion.*val != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.SetDataPtrMCSR(nullptr, &vint, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*row_offset != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.SetDataPtrMCSR(&vint, nullptr, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*col != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.SetDataPtrMCSR(&vint, &vint, nullptr, "", safe_size, safe_size, safe_size),
            ".*Assertion.*val != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.SetDataPtrELL(&null_int, &vdata, "", safe_size, safe_size, safe_size, safe_size),
            ".*Assertion.*col != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.SetDataPtrELL(&vint, &null_data, "", safe_size, safe_size, safe_size, safe_size),
            ".*Assertion.*val != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.SetDataPtrELL(nullptr, &vdata, "", safe_size, safe_size, safe_size, safe_size),
            ".*Assertion.*col != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.SetDataPtrELL(&vint, nullptr, "", safe_size, safe_size, safe_size, safe_size),
            ".*Assertion.*val != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.SetDataPtrDIA(&null_int, &vdata, "", safe_size, safe_size, safe_size, safe_size),
            ".*Assertion.*offset != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.SetDataPtrDIA(&vint, &null_data, "", safe_size, safe_size, safe_size, safe_size),
            ".*Assertion.*val != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.SetDataPtrDIA(nullptr, &vdata, "", safe_size, safe_size, safe_size, safe_size),
            ".*Assertion.*offset != (NULL|__null)*");
        ASSERT_DEATH(
            mat1.SetDataPtrDIA(&vint, nullptr, "", safe_size, safe_size, safe_size, safe_size),
            ".*Assertion.*val != (NULL|__null)*");
        ASSERT_DEATH(mat1.SetDataPtrDENSE(&null_data, "", safe_size, safe_size),
                     ".*Assertion.*val != (NULL|__null)*");
        ASSERT_DEATH(mat1.SetDataPtrDENSE(nullptr, "", safe_size, safe_size),
                     ".*Assertion.*val != (NULL|__null)*");
    }

    // LeaveDataPtr
    {
        int val;
        ASSERT_DEATH(mat1.LeaveDataPtrCOO(&vint, &null_int, &null_data),
                     ".*Assertion.*row == (NULL|__null)*");
        ASSERT_DEATH(mat1.LeaveDataPtrCOO(&null_int, &vint, &null_data),
                     ".*Assertion.*col == (NULL|__null)*");
        ASSERT_DEATH(mat1.LeaveDataPtrCOO(&null_int, &null_int, &vdata),
                     ".*Assertion.*val == (NULL|__null)*");
        ASSERT_DEATH(mat1.LeaveDataPtrCSR(&vint, &null_int, &null_data),
                     ".*Assertion.*row_offset == (NULL|__null)*");
        ASSERT_DEATH(mat1.LeaveDataPtrCSR(&null_int, &vint, &null_data),
                     ".*Assertion.*col == (NULL|__null)*");
        ASSERT_DEATH(mat1.LeaveDataPtrCSR(&null_int, &null_int, &vdata),
                     ".*Assertion.*val == (NULL|__null)*");
        ASSERT_DEATH(mat1.LeaveDataPtrMCSR(&vint, &null_int, &null_data),
                     ".*Assertion.*row_offset == (NULL|__null)*");
        ASSERT_DEATH(mat1.LeaveDataPtrMCSR(&null_int, &vint, &null_data),
                     ".*Assertion.*col == (NULL|__null)*");
        ASSERT_DEATH(mat1.LeaveDataPtrMCSR(&null_int, &null_int, &vdata),
                     ".*Assertion.*val == (NULL|__null)*");
        ASSERT_DEATH(mat1.LeaveDataPtrELL(&null_int, &vdata, val),
                     ".*Assertion.*val == (NULL|__null)*");
        ASSERT_DEATH(mat1.LeaveDataPtrELL(&vint, &null_data, val),
                     ".*Assertion.*col == (NULL|__null)*");
        ASSERT_DEATH(mat1.LeaveDataPtrDIA(&null_int, &vdata, val),
                     ".*Assertion.*val == (NULL|__null)*");
        ASSERT_DEATH(mat1.LeaveDataPtrDIA(&vint, &null_data, val),
                     ".*Assertion.*offset == (NULL|__null)*");
        ASSERT_DEATH(mat1.LeaveDataPtrDENSE(&vdata), ".*Assertion.*val == (NULL|__null)*");
    }

    free_host(&vint);
    free_host(&vdata);

    // Stop rocALUTION
    stop_rocalution();
}

template <typename T>
bool testing_local_matrix_conversions(Arguments argus)
{
    int         size        = argus.size;
    int         blockdim    = argus.blockdim;
    std::string matrix_type = argus.matrix_type;

    // Initialize rocALUTION
    set_device_rocalution(device);
    init_rocalution();

    // Generate A
    int* csr_ptr = NULL;
    int* csr_col = NULL;
    T*   csr_val = NULL;

    int nrow = 0;
    int ncol = 0;
    if(matrix_type == "Laplacian2D")
    {
        nrow = gen_2d_laplacian(size, &csr_ptr, &csr_col, &csr_val);
        ncol = nrow;
    }
    else if(matrix_type == "PermutedIdentity")
    {
        nrow = gen_permuted_identity(size, &csr_ptr, &csr_col, &csr_val);
        ncol = nrow;
    }
    else if(matrix_type == "Random")
    {
        nrow = gen_random(100 * size, 50 * size, 6, &csr_ptr, &csr_col, &csr_val);
        ncol = 50 * size;
    }
    else
    {
        return false;
    }

    int nnz = csr_ptr[nrow];

    LocalMatrix<T> A;
    A.SetDataPtrCSR(&csr_ptr, &csr_col, &csr_val, "A", nnz, nrow, ncol);

    assert(csr_ptr == NULL);
    assert(csr_col == NULL);
    assert(csr_val == NULL);

    bool success = true;

    // Check host conversions
    A.ConvertToCOO();
    success &= A.Check();
    A.ConvertToDIA();
    success &= A.Check();
    A.ConvertToELL();
    success &= A.Check();
    A.ConvertToHYB();
    success &= A.Check();
    A.ConvertToDENSE();
    success &= A.Check();
    A.ConvertToMCSR();
    success &= A.Check();
    A.ConvertToBCSR(blockdim);
    success &= A.Check();
    A.ConvertToCSR();
    success &= A.Check();

    // Check accelerator conversions
    A.MoveToAccelerator();

    A.ConvertToCOO();
    success &= A.Check();
    A.ConvertToDIA();
    success &= A.Check();
    A.ConvertToELL();
    success &= A.Check();
    A.ConvertToHYB();
    success &= A.Check();
    A.ConvertToDENSE();
    success &= A.Check();
    A.ConvertToMCSR();
    success &= A.Check();
    A.ConvertToBCSR(blockdim);
    success &= A.Check();
    A.ConvertToCSR();
    success &= A.Check();

    // Stop rocALUTION platform
    stop_rocalution();

    return success;
}

template <typename T>
bool testing_local_matrix_allocations(Arguments argus)
{
    int size     = argus.size;
    int blockdim = argus.blockdim;

    int m  = size;
    int n  = size;
    int mb = (m + blockdim - 1) / blockdim;
    int nb = (n + blockdim - 1) / blockdim;

    int nnz = 0.05 * m * n;
    if(nnz == 0)
    {
        nnz = m * n;
    }

    int nnzb = 0.01 * mb * nb;
    if(nnzb == 0)
    {
        nnzb = mb * nb;
    }

    // Initialize rocALUTION
    set_device_rocalution(device);
    init_rocalution();

    int ndiag       = 5;
    int ell_max_row = 6;
    int ell_nnz     = ell_max_row * m;
    int coo_nnz     = (nnz - ell_nnz) < 0 ? 0 : nnz - ell_nnz;

    // Testing allocating matrix types
    LocalMatrix<T> A;
    A.AllocateCSR("A", nnz, m, n);

    LocalMatrix<T> B;
    B.AllocateBCSR("B", nnzb, mb, nb, blockdim);

    LocalMatrix<T> C;
    C.AllocateCOO("C", nnz, m, n);

    LocalMatrix<T> D;
    D.AllocateDIA("D", nnz, m, n, ndiag);

    LocalMatrix<T> E;
    E.AllocateMCSR("E", nnz, m, n);

    LocalMatrix<T> F;
    F.AllocateELL("F", ell_nnz, m, n, ell_max_row);

    LocalMatrix<T> G;
    G.AllocateHYB("G", ell_nnz, coo_nnz, ell_max_row, m, n);

    LocalMatrix<T> H;
    H.AllocateDENSE("H", m, n);

    // Stop rocALUTION platform
    stop_rocalution();

    return true;
}

template <typename T>
bool testing_local_matrix_zero(Arguments argus)
{
    int size     = argus.size;
    int blockdim = argus.blockdim;

    int m  = size;
    int n  = size;
    int mb = (m + blockdim - 1) / blockdim;
    int nb = (n + blockdim - 1) / blockdim;

    int nnz = 0.05 * m * n;
    if(nnz == 0)
    {
        nnz = m * n;
    }

    // Initialize rocALUTION
    set_device_rocalution(device);
    init_rocalution();

    // Testing Zeros
    LocalMatrix<T> A;
    A.AllocateCSR("A", nnz, m, n);

    A.Zeros();

    A.Info();

    // Stop rocALUTION platform
    stop_rocalution();

    return true;
}

template <typename T>
bool testing_local_matrix_set_data_ptr(Arguments argus)
{
    int size     = argus.size;
    int blockdim = argus.blockdim;

    int m  = size;
    int n  = size;
    int mb = (m + blockdim - 1) / blockdim;
    int nb = (n + blockdim - 1) / blockdim;

    int nnz = 0.05 * m * n;
    if(nnz == 0)
    {
        nnz = m * n;
    }

    int nnzb = 0.01 * mb * nb;
    if(nnzb == 0)
    {
        nnzb = mb * nb;
    }

    // Initialize rocALUTION
    set_device_rocalution(device);
    init_rocalution();

    int ndiag       = 5;
    int ell_max_row = 6;
    int ell_nnz     = ell_max_row * m;
    int coo_nnz     = (nnz - ell_nnz) < 0 ? 0 : nnz - ell_nnz;

    // Testing allocating matrix types
    {
        LocalMatrix<T> A;
        int*           row_offset = NULL;
        int*           col        = NULL;
        T*             val        = NULL;

        allocate_host(m + 1, &row_offset);
        allocate_host(nnz, &col);
        allocate_host(nnz, &val);

        set_to_zero_host(m + 1, row_offset);
        set_to_zero_host(nnz, col);
        set_to_zero_host(nnz, val);

        A.SetDataPtrCSR(&row_offset, &col, &val, "A", nnz, m, n);
        A.LeaveDataPtrCSR(&row_offset, &col, &val);

        free_host(&row_offset);
        free_host(&col);
        free_host(&val);
    }

    {
        LocalMatrix<T> B;
        int*           row_offset = NULL;
        int*           col        = NULL;
        T*             val        = NULL;

        allocate_host(mb + 1, &row_offset);
        allocate_host(nnzb, &col);
        allocate_host(nnzb, &val);

        set_to_zero_host(mb + 1, row_offset);
        set_to_zero_host(nnzb, col);
        set_to_zero_host(nnzb, val);

        B.SetDataPtrBCSR(&row_offset, &col, &val, "C", nnzb, mb, nb, blockdim);
        B.LeaveDataPtrBCSR(&row_offset, &col, &val, blockdim);

        free_host(&row_offset);
        free_host(&col);
        free_host(&val);
    }

    {
        LocalMatrix<T> C;
        int*           row = NULL;
        int*           col = NULL;
        T*             val = NULL;

        allocate_host(nnz, &row);
        allocate_host(nnz, &col);
        allocate_host(nnz, &val);

        set_to_zero_host(nnz, row);
        set_to_zero_host(nnz, col);
        set_to_zero_host(nnz, val);

        C.SetDataPtrCOO(&row, &col, &val, "C", nnz, m, n);
        C.LeaveDataPtrCOO(&row, &col, &val);

        free_host(&row);
        free_host(&col);
        free_host(&val);
    }

    {
        LocalMatrix<T> E;
        int*           row_offset = NULL;
        int*           col        = NULL;
        T*             val        = NULL;

        allocate_host(m + 1, &row_offset);
        allocate_host(nnz, &col);
        allocate_host(nnz, &val);

        set_to_zero_host(m + 1, row_offset);
        set_to_zero_host(nnz, col);
        set_to_zero_host(nnz, val);

        E.SetDataPtrMCSR(&row_offset, &col, &val, "C", nnz, m, n);
        E.LeaveDataPtrMCSR(&row_offset, &col, &val);

        free_host(&row_offset);
        free_host(&col);
        free_host(&val);
    }

    {
        LocalMatrix<T> F;
        int*           col = NULL;
        T*             val = NULL;

        allocate_host(ell_nnz, &col);
        allocate_host(ell_nnz, &val);

        set_to_zero_host(ell_nnz, col);
        set_to_zero_host(ell_nnz, val);

        F.SetDataPtrELL(&col, &val, "C", ell_nnz, m, n, ell_max_row);
        F.LeaveDataPtrELL(&col, &val, ell_max_row);

        free_host(&col);
        free_host(&val);
    }

    {
        LocalMatrix<T> H;
        T*             val = NULL;

        allocate_host(m * n, &val);

        set_to_zero_host(m * n, val);

        H.SetDataPtrDENSE(&val, "C", m, n);
        H.LeaveDataPtrDENSE(&val);

        free_host(&val);
    }

    // Stop rocALUTION platform
    stop_rocalution();

    return true;
}

template <typename T>
LocalMatrix<T> getTestMatrix()
{
    // Create a simple 2x2 CSR matrix
    LocalMatrix<T> matrix;
    matrix.AllocateCSR("TestMatrix", 4, 2, 2);

    int row_offsets[3] = {0, 2, 4};
    int col_indices[4] = {0, 1, 0, 1};
    T   values[4]      = {1.0, 2.0, 3.0, 4.0};
    matrix.CopyFromCSR(row_offsets, col_indices, values);

    return matrix;
}

template <typename T>
void getTestMatrix(Arguments argus, LocalMatrix<T>& matrix, bool& is_invertible)
{
    int         size        = argus.size;
    int         blockdim    = argus.blockdim;
    std::string matrix_type = argus.matrix_type;

    // Generate A
    int* csr_ptr = NULL;
    int* csr_col = NULL;
    T*   csr_val = NULL;

    int nrow = 0;
    int ncol = 0;
    if(matrix_type == "Laplacian2D")
    {
        nrow = gen_2d_laplacian(size, &csr_ptr, &csr_col, &csr_val);
        ncol = nrow;

        is_invertible = true;
    }
    else if(matrix_type == "PermutedIdentity")
    {
        nrow = gen_permuted_identity(size, &csr_ptr, &csr_col, &csr_val);
        ncol = nrow;

        is_invertible = true;
    }
    else if(matrix_type == "Random")
    {
        nrow = gen_random(100 * size, 50 * size, 6, &csr_ptr, &csr_col, &csr_val);
        ncol = 50 * size;

        is_invertible = false;
    }
    else
    {
        is_invertible = true;

        matrix = getTestMatrix<T>();
        return;
    }

    int nnz = csr_ptr[nrow];

    matrix.SetDataPtrCSR(&csr_ptr, &csr_col, &csr_val, "TestMatrix", nnz, nrow, ncol);
}

template <typename T>
void getTestMatrix(Arguments argus, LocalMatrix<T>& matrix)
{
    bool is_invertible;
    getTestMatrix<T>(argus, matrix, is_invertible);
}

template <typename T>
void getMatrixVal(const LocalMatrix<T>& matrix, T* values)
{
    // Copy the values from the matrix to the provided array
    int64_t m   = matrix.GetM();
    int64_t nnz = matrix.GetNnz();

    int* row_offsets   = new int[m + 1];
    int* col_indices   = new int[nnz];
    T*   matrix_values = new T[nnz];

    matrix.CopyToCSR(row_offsets, col_indices, matrix_values);
    for(int i = 0; i < nnz; ++i)
    {
        values[i] = matrix_values[i];
    }

    delete[] row_offsets;
    delete[] col_indices;
    delete[] matrix_values;
}

template <typename T>
void getMatrixDiagVal(const LocalMatrix<T>& matrix, T* values)
{
    // Copy the values from the matrix to the provided array
    int64_t m   = matrix.GetM();
    int64_t nnz = matrix.GetNnz();

    int* row_offsets   = new int[m + 1];
    int* col_indices   = new int[nnz];
    T*   matrix_values = new T[nnz];

    matrix.CopyToCSR(row_offsets, col_indices, matrix_values);
    for(int row = 0; row < m; ++row)
    {
        int start = row_offsets[row];
        int end   = row_offsets[row + 1];
        for(int i = start; i < end; ++i)
        {
            if(col_indices[i] == row) // Diagonal element
            {
                values[row] = matrix_values[i];
                break; // Only one diagonal element per row
            }
        }
    }

    delete[] row_offsets;
    delete[] col_indices;
    delete[] matrix_values;
}

void checkPermutation(const LocalVector<int>& permutation)
{
    // Check that permutation is a valid permutation of 0..N-1
    std::vector<int> seen(permutation.GetSize(), 0);
    for(int i = 0; i < permutation.GetSize(); ++i)
    {
        int idx = permutation[i];
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, permutation.GetSize());
        seen[idx]++;
    }
    for(int i = 0; i < seen.size(); ++i)
    {
        EXPECT_EQ(seen[i], 1); // Each index appears exactly once
    }
}

template <typename T>
T getTolerance()
{
    // Set tolerance based on the type
    if(std::is_same<T, float>::value)
    {
        return 1e-5f; // Tolerance for float
    }
    else
    {
        return 1e-10; // Default tolerance for other types
    }
}

// Helper to extract dense matrix from LocalMatrix<T>
template <typename T>
std::vector<std::vector<T>> extract_dense_matrix(const LocalMatrix<T>& matrix)
{
    int                         m   = matrix.GetM();
    int                         n   = matrix.GetN();
    int                         nnz = matrix.GetNnz();
    std::vector<std::vector<T>> dense(m, std::vector<T>(n, static_cast<T>(0)));
    std::vector<int>            row_offsets(m + 1);
    std::vector<int>            col_indices(nnz);
    std::vector<T>              values(nnz);

    matrix.CopyToCSR(row_offsets.data(), col_indices.data(), values.data());
    for(int row = 0; row < m; ++row)
    {
        for(int idx = row_offsets[row]; idx < row_offsets[row + 1]; ++idx)
        {
            int col         = col_indices[idx];
            dense[row][col] = values[idx];
        }
    }
    return dense;
}

template <typename T>
void testing_local_allocate()
{
    // Test AllocateCSR
    {
        LocalMatrix<T> mat;
        EXPECT_NO_THROW(mat.AllocateCSR("AllocatedMatrix", 4, 2, 2));
        EXPECT_EQ(mat.GetNnz(), 4);
        EXPECT_EQ(mat.GetM(), 2);
        EXPECT_EQ(mat.GetN(), 2);
    }

    // Test AllocateCOO
    {
        LocalMatrix<T> mat;
        mat.AllocateCOO("AllocatedMatrix", 4, 2, 2);
        EXPECT_EQ(mat.GetNnz(), 4);
        EXPECT_EQ(mat.GetM(), 2);
        EXPECT_EQ(mat.GetN(), 2);
    }

    // Test AllocateBCSR
    {
        LocalMatrix<T> mat;
        int            nnzb = 4, mb = 2, nb = 2, blockdim = 2;
        EXPECT_NO_THROW(mat.AllocateBCSR("BCSR", nnzb, mb, nb, blockdim));
        EXPECT_EQ(mat.GetNnz(), nnzb * blockdim * blockdim);
        EXPECT_EQ(mat.GetM(), mb * blockdim);
        EXPECT_EQ(mat.GetN(), nb * blockdim);
    }

    // Test AllocateDIA
    {
        LocalMatrix<T> mat;
        int            nnz = 6, m = 3, n = 3, ndiag = 2;
        EXPECT_NO_THROW(mat.AllocateDIA("DIA", nnz, m, n, ndiag));
        EXPECT_EQ(mat.GetNnz(), nnz);
        EXPECT_EQ(mat.GetM(), m);
        EXPECT_EQ(mat.GetN(), n);
    }

    // Test AllocateMCSR
    {
        LocalMatrix<T> mat;
        int            nnz = 5, m = 3, n = 3;
        EXPECT_NO_THROW(mat.AllocateMCSR("MCSR", nnz, m, n));
        EXPECT_EQ(mat.GetNnz(), nnz);
        EXPECT_EQ(mat.GetM(), m);
        EXPECT_EQ(mat.GetN(), n);
    }

    // Test AllocateELL
    {
        LocalMatrix<T> mat;
        int            ell_nnz = 6, m = 3, n = 3, ell_max_row = 2;
        EXPECT_NO_THROW(mat.AllocateELL("ELL", ell_nnz, m, n, ell_max_row));
        EXPECT_EQ(mat.GetNnz(), ell_nnz);
        EXPECT_EQ(mat.GetM(), m);
        EXPECT_EQ(mat.GetN(), n);
    }

    // Test AllocateHYB
    {
        LocalMatrix<T> mat;
        int            m = 3, n = 3;
        int            ell_max_row = 2;
        int            ell_nnz     = ell_max_row * m; // 2 * 3 = 6
        int            coo_nnz     = 2;

        EXPECT_NO_THROW(mat.AllocateHYB("HYB", ell_nnz, coo_nnz, ell_max_row, m, n));
        EXPECT_EQ(mat.GetNnz(), ell_nnz + coo_nnz); // 8
        EXPECT_EQ(mat.GetM(), m);
        EXPECT_EQ(mat.GetN(), n);
    }

    // Test AllocateDENSE
    {
        LocalMatrix<T> mat;
        int            m = 3, n = 3;
        EXPECT_NO_THROW(mat.AllocateDENSE("DENSE", m, n));
        EXPECT_EQ(mat.GetNnz(), m * n);
        EXPECT_EQ(mat.GetM(), m);
        EXPECT_EQ(mat.GetN(), n);
    }
}

template <typename T>
void testing_check_with_empty_matrix()
{
    LocalMatrix<T> empty_matrix;
    // Check should pass without any issues
    EXPECT_NO_THROW(empty_matrix.Check());
    // Info should not throw an error
    EXPECT_NO_THROW(empty_matrix.Info());
}

template <typename T>
void testing_local_copy_from_async()
{
    auto           matrix = getTestMatrix<T>();
    LocalMatrix<T> copy_matrix;
    // CopyFromAsync should copy the matrix asynchronously (if supported)
    EXPECT_NO_THROW(copy_matrix.CopyFromAsync(matrix));
    EXPECT_NO_THROW(matrix.Sync());
    EXPECT_EQ(copy_matrix.GetM(), matrix.GetM());
    EXPECT_EQ(copy_matrix.GetN(), matrix.GetN());
    EXPECT_EQ(copy_matrix.GetNnz(), matrix.GetNnz());

    // Compare dense representations
    auto dense_orig = extract_dense_matrix(matrix);
    auto dense_copy = extract_dense_matrix(copy_matrix);

    EXPECT_EQ(dense_orig.size(), dense_copy.size());
    for(size_t i = 0; i < dense_orig.size(); ++i)
    {
        EXPECT_EQ(dense_orig[i].size(), dense_copy[i].size());
        for(size_t j = 0; j < dense_orig[i].size(); ++j)
        {
            EXPECT_EQ(dense_orig[i][j], dense_copy[i][j]);
        }
    }
}

template <typename T>
void testing_local_update_values_csr()
{
    auto matrix = getTestMatrix<T>();

    int64_t nnz = matrix.GetNnz();

    // Use std::vector instead of raw arrays
    std::vector<T> new_values(nnz);
    for(int64_t i = 0; i < nnz; ++i)
    {
        new_values[i] = static_cast<T>(i + 10); // Fill with some values
    }

    // UpdateValuesCSR should update the values in the matrix
    EXPECT_NO_THROW(matrix.UpdateValuesCSR(new_values.data()));

    std::vector<T> check_values(nnz);
    getMatrixVal(matrix, check_values.data());
    for(int64_t i = 0; i < nnz; ++i)
    {
        EXPECT_EQ(check_values[i], new_values[i]);
    }
}

template <typename T>
void testing_local_move_to_accelerator()
{
    auto matrix = getTestMatrix<T>();

    for(int i = 0; i < 10; i++)
    {
        EXPECT_NO_THROW(matrix.MoveToAccelerator());
    }
    for(int i = 0; i < 10; i++)
    {
        EXPECT_NO_THROW(matrix.MoveToHost());
    }
    for(int i = 0; i < 10; i++)
    {
        EXPECT_NO_THROW(matrix.MoveToHost());
        EXPECT_NO_THROW(matrix.MoveToAccelerator());
    }
    EXPECT_EQ(matrix.Check(), true);
}

template <typename T>
void testing_local_move_to_accelerator_async()
{
    auto matrix = getTestMatrix<T>();
    // MoveToAcceleratorAsync should move the matrix asynchronously
    for(int i = 0; i < 10; i++)
    {
        EXPECT_NO_THROW(matrix.MoveToAcceleratorAsync());
    }
    for(int i = 0; i < 10; i++)
    {
        EXPECT_NO_THROW(matrix.MoveToHostAsync());
    }
    for(int i = 0; i < 10; i++)
    {
        EXPECT_NO_THROW(matrix.MoveToHostAsync());
        EXPECT_NO_THROW(matrix.MoveToAcceleratorAsync());
    }
    EXPECT_NO_THROW(matrix.Sync());
    EXPECT_EQ(matrix.Check(), true);
}

template <typename T>
void testing_local_move_to_host_async()
{
    auto matrix = getTestMatrix<T>();
    // MoveToHostAsync should move the matrix asynchronously to host
    EXPECT_NO_THROW(matrix.MoveToHostAsync());
    EXPECT_NO_THROW(matrix.Sync());
    EXPECT_EQ(matrix.GetM(), 2);
    EXPECT_EQ(matrix.GetN(), 2);
}

template <typename T>
void testing_local_clear(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);

    // Clear should remove all data from the matrix
    // This test checks if the Clear operation is valid
    // by checking if the number of non-zero entries (nnz) is zero
    // and the dimensions (m, n) are also zero.

    matrix.Clear();
    EXPECT_EQ(matrix.GetNnz(), 0);
    EXPECT_EQ(matrix.GetM(), 0);
    EXPECT_EQ(matrix.GetN(), 0);
}

template <typename T>
void testing_local_zeros(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);

    // Zeros should set all values in the matrix to zero
    // This test checks if the Zeros operation is valid
    // by checking if all values in the matrix are zero after the operation.

    matrix.Zeros();
    matrix.Check();
    matrix.Info();

    int64_t nnz          = matrix.GetNnz();
    T*      check_values = new T[nnz];
    getMatrixVal(matrix, check_values);
    for(int64_t i = 0; i < nnz; ++i)
    {
        EXPECT_EQ(check_values[i], static_cast<T>(0));
    }
    delete[] check_values;
}

template <typename T>
void testing_local_copy(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);

    // CopyFrom should create a copy of the matrix
    // This test checks if the copy operation is valid
    // by comparing the original matrix with the copied matrix.

    LocalMatrix<T> copy_matrix;
    copy_matrix.CopyFrom(matrix);
    EXPECT_EQ(copy_matrix.GetNnz(), matrix.GetNnz());
    EXPECT_EQ(copy_matrix.GetM(), matrix.GetM());
    EXPECT_EQ(copy_matrix.GetN(), matrix.GetN());

    // Compare dense representations
    auto dense_orig = extract_dense_matrix(matrix);
    auto dense_copy = extract_dense_matrix(copy_matrix);

    EXPECT_EQ(dense_orig.size(), dense_copy.size());
    for(size_t i = 0; i < dense_orig.size(); ++i)
    {
        EXPECT_EQ(dense_orig[i].size(), dense_copy[i].size());
        for(size_t j = 0; j < dense_orig[i].size(); ++j)
        {
            EXPECT_EQ(dense_orig[i][j], dense_copy[i][j]);
        }
    }
}

template <typename T>
void testing_local_scale(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);

    // Scale should multiply all values in the matrix by a scalar
    // This test checks if the scaling operation is valid
    // by comparing the scaled values with the expected values.
    // The expected values are obtained by multiplying the original values
    // by the scaling factor.

    // Save original dense matrix
    auto orig_dense = extract_dense_matrix(matrix);

    // Scale the matrix by 2.0
    matrix.Scale(2.0);

    // Extract new dense matrix
    auto new_dense = extract_dense_matrix(matrix);

    // Compare each value
    int m = matrix.GetM();
    int n = matrix.GetN();
    for(int i = 0; i < m; ++i)
        for(int j = 0; j < n; ++j)
            EXPECT_EQ(new_dense[i][j], orig_dense[i][j] * 2.0);
}

template <typename T>
void testing_local_extract_diagonal(Arguments argus)
{
    if(argus.matrix_type != "Laplacian2D")
    {
        return;
    }

    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);

    // ExtractDiagonal should extract the diagonal of the matrix
    // and store it in a LocalVector
    // This test checks if the diagonal extraction is correct
    // by comparing the extracted diagonal with the expected values.
    // The expected values are obtained by iterating through the matrix
    // and checking the diagonal elements.

    LocalVector<T> diag;
    matrix.ExtractDiagonal(&diag);

    int64_t m            = matrix.GetM();
    T*      check_values = new T[m];
    getMatrixDiagVal(matrix, check_values);

    EXPECT_EQ(diag.GetSize(), m);
    for(int i = 0; i < m; ++i)
    {
        EXPECT_EQ(diag[i], check_values[i]);
    }
    delete[] check_values;
}

template <typename T>
void testing_local_extract_inverse_diagonal(Arguments argus)
{
    if(argus.matrix_type != "Laplacian2D")
    {
        return;
    }

    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);

    // ExtractInverseDiagonal should extract the inverse diagonal of the matrix
    // and store it in a LocalVector
    // This test checks if the inverse diagonal extraction is correct
    // by comparing the extracted inverse diagonal with the expected values.
    // The expected values are obtained by taking the reciprocal of the diagonal elements.

    LocalVector<T> diag;
    matrix.ExtractDiagonal(&diag);
    LocalVector<T> inv_diag;
    matrix.ExtractInverseDiagonal(&inv_diag);
    EXPECT_EQ(inv_diag.GetSize(), diag.GetSize());
    for(int i = 0; i < diag.GetSize(); ++i)
    {
        EXPECT_NEAR(diag[i],
                    1.0 / inv_diag[i],
                    getTolerance<T>() * std::abs(diag[i])); // Allow some tolerance
    }
}

template <typename T>
void testing_local_transpose(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);

    // This test checks if the transpose operation is valid
    // by applying the matrix to two vectors and checking if the results
    // are consistent with the transpose operation.

    LocalVector<T> vec1, vec2, tmp1, tmp2;
    vec1.Allocate("TestVector1", matrix.GetM());
    vec2.Allocate("TestVector2", matrix.GetN());

    tmp1.Allocate("TemporaryVector1", matrix.GetM());
    tmp2.Allocate("TemporaryVector2", matrix.GetN());

    vec1.SetRandomUniform(static_cast<T>(0), static_cast<T>(1));
    vec2.SetRandomUniform(static_cast<T>(0), static_cast<T>(1));

    matrix.Apply(vec2, &tmp1);
    T val1 = vec1.Dot(tmp1);

    matrix.Transpose();

    matrix.Apply(vec1, &tmp2);
    T val2 = vec2.Dot(tmp2);
    EXPECT_NEAR(val1, val2, getTolerance<T>() * std::abs(val1)); // Allow some tolerance
}

template <typename T>
void testing_clone_matrix(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);

    // Clone the matrix
    // This should create a new matrix with the same data
    // and structure as the original matrix

    LocalMatrix<T> clone;
    clone.CloneFrom(matrix);

    EXPECT_EQ(clone.GetNnz(), matrix.GetNnz());
    EXPECT_EQ(clone.GetM(), matrix.GetM());
    EXPECT_EQ(clone.GetN(), matrix.GetN());

    // Compare dense representations
    auto dense_orig  = extract_dense_matrix(matrix);
    auto dense_clone = extract_dense_matrix(clone);

    EXPECT_EQ(dense_orig.size(), dense_clone.size());
    for(size_t i = 0; i < dense_orig.size(); ++i)
    {
        EXPECT_EQ(dense_orig[i].size(), dense_clone[i].size());
        for(size_t j = 0; j < dense_orig[i].size(); ++j)
        {
            EXPECT_EQ(dense_orig[i][j], dense_clone[i][j]);
        }
    }
}

template <typename T>
void testing_invert_matrix(Arguments argus)
{
    LocalMatrix<T> matrix;
    bool           is_invertible;
    getTestMatrix<T>(argus, matrix, is_invertible);

    if(!is_invertible)
    {
        return;
    }

    LocalVector<T> vec1, vec2, tmp1;
    vec1.Allocate("TestVector1", matrix.GetM());
    vec2.Allocate("TestVector2", matrix.GetM());

    tmp1.Allocate("TemporaryVector1", matrix.GetM());

    vec1.SetRandomUniform(static_cast<T>(0), static_cast<T>(1));
    vec2.CopyFrom(vec1);

    matrix.Apply(vec1, &tmp1);

    matrix.Invert();

    matrix.Apply(tmp1, &vec1);

    for(int i = 0; i < vec1.GetSize(); ++i)
    {
        // Check if the result is close to the initial vector
        EXPECT_NEAR(vec1[i], vec2[i], getTolerance<T>() * std::abs(vec2[i]));
    }
}

template <typename T>
void testing_set_data_pointer()
{
    // CSR
    {
        LocalMatrix<T> mat         = getTestMatrix<T>();
        int*           row_offsets = new int[3]{0, 2, 4};
        int*           col_indices = new int[4]{0, 1, 0, 1};
        T*             values      = new T[4]{1.0, 2.0, 3.0, 4.0};

        mat.SetDataPtrCSR(&row_offsets, &col_indices, &values, "TestMatrix", 4, 2, 2);

        EXPECT_EQ(mat.GetNnz(), 4);
        EXPECT_EQ(mat.GetM(), 2);
        EXPECT_EQ(mat.GetN(), 2);

        delete[] row_offsets;
        delete[] col_indices;
        delete[] values;
    }
}

template <typename T>
void testing_leave_data_pointer()
{
    auto matrix      = getTestMatrix<T>();
    int* row_offsets = nullptr;
    int* col_indices = nullptr;
    T*   values      = nullptr;

    matrix.LeaveDataPtrCSR(&row_offsets, &col_indices, &values);

    EXPECT_NE(row_offsets, nullptr);
    EXPECT_NE(col_indices, nullptr);
    EXPECT_NE(values, nullptr);

    EXPECT_EQ(matrix.GetNnz(), 0);
    EXPECT_EQ(matrix.GetM(), 0);
    EXPECT_EQ(matrix.GetN(), 0);

    delete[] row_offsets;
    delete[] col_indices;
    delete[] values;
}

template <typename T>
void testing_maximal_independent_set(Arguments argus)
{
    if(argus.matrix_type != "Laplacian2D")
    {
        return;
    }

    LocalMatrix<T> matrix;
    bool           is_invertible;
    getTestMatrix<T>(argus, matrix, is_invertible);
    LocalVector<int> permutation;
    int              size = 0;

    // Compute the Maximal Independent Set
    matrix.MaximalIndependentSet(size, &permutation);

    // Validate the size of the independent set
    EXPECT_GT(size, 0);
    EXPECT_EQ(permutation.GetSize(), matrix.GetM());

    checkPermutation(permutation);
}

template <typename T>
void testing_zero_block_permutation(Arguments argus)
{
    if(argus.matrix_type != "Laplacian2D")
    {
        return;
    }

    LocalMatrix<T> matrix;
    bool           is_invertible;
    getTestMatrix<T>(argus, matrix, is_invertible);
    LocalVector<int> permutation;
    int              size = 0;

    // Compute the Zero Block Permutation
    matrix.ZeroBlockPermutation(size, &permutation);

    // Validate the size of the permutation
    EXPECT_GT(size, 0);
    EXPECT_EQ(permutation.GetSize(), matrix.GetM());

    checkPermutation(permutation);
}

template <typename T>
void testing_Householder(Arguments argus)
{
    if(argus.matrix_type != "Laplacian2D")
    {
        return;
    }

    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);
    LocalVector<T> vec;
    int            idx = 0;
    vec.Allocate("HouseholderVector", matrix.GetM() - idx);
    T beta = 0.0;

    // Perform Householder transformation
    matrix.Householder(idx, beta, &vec);

    // Validate the result
    EXPECT_GT(beta, 0.0);

    //// Get the first column as a vector
    //LocalVector<T> x;
    //x.Allocate("ExtractedColumn", matrix.GetM());
    //matrix.ExtractColumnVector(0, &x);
    //
    //// Compute v^T x
    //T vTx = 0;
    //for(int i = 0; i < x.GetSize(); ++i)
    //    vTx += x[i] * vec[i];
    //
    //// Compute Hx = x - beta * v * v^T x
    //LocalVector<T> Hx;
    //Hx.Allocate("Hx", x.GetSize());
    //for(int i = 0; i < x.GetSize(); ++i)
    //    Hx[i] = x[i] - beta * vec[i] * vTx;
    //
    //// Check that Hx[1:] are (close to) zero
    //for(int i = 1; i < Hx.GetSize(); ++i)
    //    EXPECT_NEAR(Hx[i], 0.0, 1e-6);
}

template <typename T>
void testing_CMK(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);
    LocalVector<int> permutation;

    // Perform CMK ordering
    matrix.CMK(&permutation);

    // Validate the permutation vector
    EXPECT_EQ(permutation.GetSize(), matrix.GetM());

    checkPermutation(permutation);
}

template <typename T>
void testing_RCMK(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);
    LocalVector<int> permutation;

    // Perform RCMK ordering
    matrix.RCMK(&permutation);

    // Validate the permutation vector
    EXPECT_EQ(permutation.GetSize(), matrix.GetM());

    checkPermutation(permutation);
}

template <typename T>
void testing_connectivity_order(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);
    LocalVector<int> permutation;

    // Perform ConnectivityOrder
    matrix.ConnectivityOrder(&permutation);

    // Validate the permutation vector
    EXPECT_EQ(permutation.GetSize(), matrix.GetM());

    checkPermutation(permutation);
}

template <typename T>
void testing_symbolic_power(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);

    if(matrix.GetM() != matrix.GetN())
    {
        // SymbolicPower is only defined for square matrices
        return;
    }

    // Compute expected symbolic square pattern
    std::set<std::pair<int, int>> expected_pattern;
    {
        int* row_offsets   = new int[matrix.GetM() + 1];
        int* col_indices   = new int[matrix.GetNnz()];
        T*   matrix_values = new T[matrix.GetNnz()];

        matrix.CopyToCSR(row_offsets, col_indices, matrix_values);

        // Save original pattern
        std::set<std::pair<int, int>> original_pattern;
        for(int row = 0; row < matrix.GetM(); ++row)
        {
            for(int idx = row_offsets[row]; idx < row_offsets[row + 1]; ++idx)
                original_pattern.insert({row, col_indices[idx]});
        }

        for(int i = 0; i < matrix.GetM(); ++i)
        {
            for(int k = 0; k < matrix.GetN(); ++k)
            {
                if(original_pattern.count({i, k}))
                {
                    for(int j = 0; j < matrix.GetN(); ++j)
                    {
                        if(original_pattern.count({k, j}))
                            expected_pattern.insert({i, j});
                    }
                }
            }
        }

        delete[] row_offsets;
        delete[] col_indices;
        delete[] matrix_values;
    }

    // Perform SymbolicPower
    matrix.SymbolicPower(2);

    // Extract new pattern
    std::set<std::pair<int, int>> new_pattern;
    {
        int* row_offsets   = new int[matrix.GetM() + 1];
        int* col_indices   = new int[matrix.GetNnz()];
        T*   matrix_values = new T[matrix.GetNnz()];

        matrix.CopyToCSR(row_offsets, col_indices, matrix_values);
        for(int row = 0; row < matrix.GetM(); ++row)
        {
            for(int idx = row_offsets[row]; idx < row_offsets[row + 1]; ++idx)
                new_pattern.insert({row, col_indices[idx]});
        }
        delete[] row_offsets;
        delete[] col_indices;
        delete[] matrix_values;
    }

    // Compare
    EXPECT_EQ(new_pattern, expected_pattern);
}

template <typename T>
void testing_scale_off_diagonal(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);

    // Save the original matrix
    auto orig_dense = extract_dense_matrix(matrix);

    // Scale off-diagonal elements by 2.0
    EXPECT_NO_THROW(matrix.ScaleOffDiagonal(2.0));

    // Extract new dense matrix
    auto new_dense = extract_dense_matrix(matrix);

    int m = matrix.GetM();
    int n = matrix.GetN();

    // Check: diagonal elements unchanged, off-diagonal elements scaled by 2
    for(int i = 0; i < m; ++i)
        for(int j = 0; j < n; ++j)
            if(i == j)
                EXPECT_EQ(new_dense[i][j], orig_dense[i][j]);
            else
                EXPECT_EQ(new_dense[i][j], orig_dense[i][j] * 2.0);
}

template <typename T>
void testing_add_scalar(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);

    int64_t nnz    = matrix.GetNnz();
    T*      values = new T[nnz];
    getMatrixVal(matrix, values);

    // Perform AddScalar
    matrix.AddScalar(1.0);

    // Validate the result
    T* values_new = new T[nnz];
    getMatrixVal(matrix, values_new);
    for(int64_t i = 0; i < nnz; ++i)
    {
        EXPECT_NEAR(values_new[i], values[i] + 1.0, getTolerance<T>() * std::abs(values_new[i]));
    }
    delete[] values;
    delete[] values_new;
}

template <typename T>
void testing_replace_column_vector(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);

    int n = matrix.GetN();
    int m = matrix.GetM();

    // Test replacing each column with a known vector and verify
    for(int col = 0; col < n; ++col)
    {
        LocalVector<T> vec;
        vec.Allocate("ColumnVector", m);
        for(int i = 0; i < m; ++i)
            vec[i] = static_cast<T>(i + 100 * col); // Unique values per column

        // Replace column col
        EXPECT_NO_THROW(matrix.ReplaceColumnVector(col, vec));

        // Extract column col
        LocalVector<T> extracted_vec;
        extracted_vec.Allocate("ExtractedColumn", m);
        EXPECT_NO_THROW(matrix.ExtractColumnVector(col, &extracted_vec));

        // Verify the replacement
        for(int i = 0; i < m; ++i)
            EXPECT_EQ(extracted_vec[i], vec[i]);
    }
}

template <typename T>
void testing_extract_column_vector(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);

    int n = matrix.GetN();
    int m = matrix.GetM();

    // Extract each column and compare with dense representation
    auto dense = extract_dense_matrix(matrix);

    for(int col = 0; col < n; ++col)
    {
        LocalVector<T> vec;
        vec.Allocate("ExtractedColumn", m);

        // Extract column vector
        EXPECT_NO_THROW(matrix.ExtractColumnVector(col, &vec));

        // Validate the result
        for(int row = 0; row < m; ++row)
        {
            EXPECT_EQ(vec[row], dense[row][col]);
        }
    }
}

template <typename T>
void testing_set_data_ptr_DIA()
{
    // Create a LocalMatrix
    LocalMatrix<T> mat;

    // Define DIA matrix parameters
    int64_t nnz      = 12; // Number of non-zero elements
    int64_t nrow     = 6; // Number of rows
    int64_t ncol     = 6; // Number of columns
    int     num_diag = 2; // Number of diagonals

    // Allocate memory for DIA data
    int* offset = new int[num_diag]{0, 1}; // Diagonal offsets
    T*   val    = new T[nnz]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

    // Ensure assertions are valid
    ASSERT_NE(offset, nullptr);
    ASSERT_NE(val, nullptr);
    ASSERT_GT(nnz, 0);
    ASSERT_GT(nrow, 0);
    ASSERT_GT(num_diag, 0);

    // Check nnz consistency
    if(nrow < ncol)
    {
        ASSERT_EQ(nnz, ncol * num_diag);
    }
    else
    {
        ASSERT_EQ(nnz, nrow * num_diag);
    }

    // Set the DIA data pointer
    EXPECT_NO_THROW(mat.SetDataPtrDIA(&offset, &val, "TestMatrix", nnz, nrow, ncol, num_diag));

    // Validate that the matrix has been set correctly
    EXPECT_EQ(mat.GetNnz(), nnz);
    EXPECT_EQ(mat.GetN(), nrow);
    EXPECT_EQ(mat.GetM(), ncol);

    // Validate that the data pointers are now owned by the matrix
    EXPECT_EQ(offset, nullptr);
    EXPECT_EQ(val, nullptr);
}

template <typename T>
void testing_leave_data_ptr_DIA()
{
    // Create a LocalMatrix
    LocalMatrix<T> mat;

    // Define DIA matrix parameters
    int64_t nnz      = 12; // Number of non-zero elements
    int64_t nrow     = 6; // Number of rows
    int64_t ncol     = 6; // Number of columns
    int     num_diag = 2; // Number of diagonals

    // Allocate memory for DIA data
    int* offset = new int[num_diag]{0, 1}; // Diagonal offsets
    T*   val    = new T[nnz]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

    // Set the DIA data pointer
    mat.SetDataPtrDIA(&offset, &val, "TestMatrix", nnz, nrow, ncol, num_diag);

    // Ensure the matrix is in DIA format
    ASSERT_TRUE(mat.GetFormat() == DIA);

    // Prepare pointers to retrieve the data
    int* retrieved_offset   = nullptr;
    T*   retrieved_val      = nullptr;
    int  retrieved_num_diag = 0;

    // Leave the DIA data pointer
    EXPECT_NO_THROW(mat.LeaveDataPtrDIA(&retrieved_offset, &retrieved_val, retrieved_num_diag));

    // Validate that the data pointers are correctly retrieved
    ASSERT_NE(retrieved_offset, nullptr);
    ASSERT_NE(retrieved_val, nullptr);
    EXPECT_EQ(retrieved_num_diag, num_diag);

    // Validate the retrieved data
    for(int i = 0; i < num_diag; ++i)
    {
        EXPECT_EQ(retrieved_offset[i], i); // Offsets: 0, 1
    }
    for(int i = 0; i < nnz; ++i)
    {
        EXPECT_EQ(retrieved_val[i], static_cast<T>(i + 1)); // Values: 1.0, 2.0, ..., 12.0
    }

    EXPECT_EQ(mat.GetNnz(), 0);
    EXPECT_EQ(mat.GetM(), 0);
    EXPECT_EQ(mat.GetN(), 0);

    // Clean up the retrieved pointers
    delete[] retrieved_offset;
    delete[] retrieved_val;
}

template <typename T>
void testing_copy_from_COO()
{
    // Create a LocalMatrix
    LocalMatrix<T> mat;

    // Define COO matrix parameters
    int64_t nnz  = 4; // Number of non-zero elements
    int64_t nrow = 3; // Number of rows
    int64_t ncol = 3; // Number of columns

    // Allocate memory for COO data
    int row[] = {0, 1, 2, 2}; // Row indices
    int col[] = {0, 1, 2, 0}; // Column indices
    T   val[] = {1.0, 2.0, 3.0, 4.0}; // Non-zero values

    // Allocate the matrix in COO format
    mat.AllocateCOO("TestMatrix", nnz, nrow, ncol);

    // Ensure assertions are valid
    ASSERT_NE(row, nullptr);
    ASSERT_NE(col, nullptr);
    ASSERT_NE(val, nullptr);
    ASSERT_EQ(mat.GetFormat(), COO);

    // Copy data from COO
    EXPECT_NO_THROW(mat.CopyFromCOO(row, col, val));

    // Validate the matrix properties
    EXPECT_EQ(mat.GetNnz(), nnz);
    EXPECT_EQ(mat.GetM(), nrow);
    EXPECT_EQ(mat.GetN(), ncol);

    // Validate that the matrix is still in COO format
    EXPECT_EQ(mat.GetFormat(), COO);
}

template <typename T>
void testing_copy_to_COO()
{
    // Create a LocalMatrix
    LocalMatrix<T> mat;

    // Define COO matrix parameters
    int64_t nnz  = 4; // Number of non-zero elements
    int64_t nrow = 3; // Number of rows
    int64_t ncol = 3; // Number of columns

    // Allocate memory for COO data
    int row[] = {0, 1, 2, 2}; // Row indices
    int col[] = {0, 1, 2, 0}; // Column indices
    T   val[] = {1.0, 2.0, 3.0, 4.0}; // Non-zero values

    // Allocate the matrix in COO format and copy data
    mat.AllocateCOO("TestMatrix", nnz, nrow, ncol);
    mat.CopyFromCOO(row, col, val);

    // Ensure assertions are valid
    ASSERT_EQ(mat.GetFormat(), COO);
    ASSERT_EQ(mat.GetNnz(), nnz);
    ASSERT_EQ(mat.GetM(), nrow);
    ASSERT_EQ(mat.GetN(), ncol);

    // Prepare arrays to retrieve the data
    int* retrieved_row = new int[nnz];
    int* retrieved_col = new int[nnz];
    T*   retrieved_val = new T[nnz];

    // Copy data to COO
    EXPECT_NO_THROW(mat.CopyToCOO(retrieved_row, retrieved_col, retrieved_val));

    // Validate the retrieved data
    for(int i = 0; i < nnz; ++i)
    {
        EXPECT_EQ(retrieved_row[i], row[i]);
        EXPECT_EQ(retrieved_col[i], col[i]);
        EXPECT_DOUBLE_EQ(retrieved_val[i], val[i]);
    }

    // Clean up
    delete[] retrieved_row;
    delete[] retrieved_col;
    delete[] retrieved_val;
}

template <typename T>
void testing_copy_from_host_CSR()
{
    // Create a LocalMatrix
    LocalMatrix<double> mat;

    // Define CSR matrix parameters
    int64_t nnz  = 4; // Number of non-zero elements
    int64_t nrow = 3; // Number of rows
    int64_t ncol = 3; // Number of columns

    // Allocate memory for CSR data
    PtrType row_offset[] = {0, 1, 3, 4}; // Row offsets
    int     col[]        = {0, 1, 2, 2}; // Column indices
    double  val[]        = {1.0, 2.0, 3.0, 4.0}; // Non-zero values

    // Ensure assertions are valid
    ASSERT_NE(row_offset, nullptr);
    ASSERT_NE(col, nullptr);
    ASSERT_NE(val, nullptr);
    ASSERT_GE(nnz, 0);
    ASSERT_GE(nrow, 0);
    ASSERT_GE(ncol, 0);

    // Copy data from host CSR
    EXPECT_NO_THROW(mat.CopyFromHostCSR(row_offset, col, val, "TestMatrix", nnz, nrow, ncol));

    // Validate the matrix properties
    EXPECT_EQ(mat.GetNnz(), nnz);
    EXPECT_EQ(mat.GetM(), nrow);
    EXPECT_EQ(mat.GetN(), ncol);

    // Validate that the matrix is in CSR format
    EXPECT_EQ(mat.GetFormat(), CSR);
}

// Helper function to create a sample MTX file
void CreateSampleMTXFile(const std::string& filename)
{
    std::ofstream file(filename);
    ASSERT_TRUE(file.is_open());

    // Write a simple 3x3 matrix in MTX format
    file << "%%MatrixMarket matrix coordinate real general\n";
    file << "% Sample MTX file\n";
    file << "3 3 4\n"; // 3 rows, 3 columns, 4 non-zero elements
    file << "1 1 1.0\n";
    file << "1 2 2.0\n";
    file << "2 3 3.0\n";
    file << "3 1 4.0\n";

    file.close();
}

template <typename T>
void testing_read_file_MTX()
{
    // Create a sample MTX file
    const std::string filename = get_temp_dir() + "test_matrix.mtx";
    CreateSampleMTXFile(filename);

    // Create a LocalMatrix
    LocalMatrix<T> mat;

    // Read the matrix from the MTX file
    EXPECT_NO_THROW(mat.ReadFileMTX(filename));

    // Validate the matrix properties
    EXPECT_EQ(mat.GetM(), 3); // Number of rows
    EXPECT_EQ(mat.GetN(), 3); // Number of columns
    EXPECT_EQ(mat.GetNnz(), 4); // Number of non-zero elements

    // Convert the matrix to CSR format
    EXPECT_NO_THROW(mat.ConvertToCSR());

    // Validate the matrix format
    EXPECT_EQ(mat.GetFormat(), CSR);

    // Prepare arrays to retrieve the data
    PtrType* row_offset = new PtrType[4];
    int*     col        = new int[4];
    T*       val        = new T[4];

    // Copy data to CSR
    EXPECT_NO_THROW(mat.CopyToCSR(row_offset, col, val));

    // Validate the retrieved data
    EXPECT_EQ(row_offset[0], 0);
    EXPECT_EQ(row_offset[1], 2);
    EXPECT_EQ(row_offset[2], 3);
    EXPECT_EQ(row_offset[3], 4);

    EXPECT_EQ(col[0], 0);
    EXPECT_DOUBLE_EQ(val[0], 1.0);
    EXPECT_EQ(col[1], 1);
    EXPECT_DOUBLE_EQ(val[1], 2.0);
    EXPECT_EQ(col[2], 2);
    EXPECT_DOUBLE_EQ(val[2], 3.0);
    EXPECT_EQ(col[3], 0);
    EXPECT_DOUBLE_EQ(val[3], 4.0);

    // Clean up
    delete[] row_offset;
    delete[] col;
    delete[] val;

    // Remove the sample MTX file
    std::remove(filename.c_str());
}

// Helper function to create a test CSR file
void CreateTestCSRFile(const std::string& filename)
{
    std::ofstream file(filename);
    ASSERT_TRUE(file.is_open());

    // Example CSR format for a 3x3 matrix:
    // Matrix:
    // 1 0 2
    // 0 3 0
    // 4 0 5
    //
    // CSR representation:
    // Row pointers: [0, 2, 3, 5]
    // Column indices: [0, 2, 1, 0, 2]
    // Values: [1, 2, 3, 4, 5]

    file << "3 3 5\n"; // 3 rows, 3 columns, 5 non-zero elements
    file << "0 2 3 5\n"; // Row pointers
    file << "0 2 1 0 2\n"; // Column indices
    file << "1 2 3 4 5\n"; // Values

    file.close();
}

// Helper function to read the contents of a file into a string
std::string testing_ReadFileContents(const std::string& filename)
{
    std::ifstream     file(filename);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

template <typename T>
void testing_write_file_MTX()
{
    // Step 1: Create a LocalMatrix in CSR format
    LocalMatrix<T> matrix;
    matrix.AllocateCSR("test", 5, 3, 3); // 3x3 matrix with 5 non-zero elements

    // Fill the matrix with some values
    // Matrix representation:
    // [ 1.0  0.0  2.0 ]
    // [ 0.0  3.0  0.0 ]
    // [ 4.0  0.0  5.0 ]
    int row_offsets[4] = {0, 2, 3, 5};
    int col_indices[5] = {0, 2, 1, 0, 2};
    T   values[5]      = {1.0, 2.0, 3.0, 4.0, 5.0};
    matrix.CopyFromCSR(row_offsets, col_indices, values);

    // Step 2: Write the matrix to a file in MTX format
    const std::string filename = get_temp_dir() + "test_matrix.mtx";
    ASSERT_NO_THROW(matrix.WriteFileMTX(filename));

    // Step 3: Verify the contents of the file
    std::string file_contents = testing_ReadFileContents(filename);

    // Expected MTX file contents
    std::string expected_contents = "%%MatrixMarket matrix coordinate real general\n"
                                    "3 3 5\n" // 3x3 matrix with 5 non-zero elements
                                    "1 1 1\n"
                                    "1 3 2\n"
                                    "2 2 3\n"
                                    "3 1 4\n"
                                    "3 3 5\n";

    // Compare the actual file contents with the expected contents
    EXPECT_EQ(file_contents, expected_contents);

    // Clean up the test file
    std::remove(filename.c_str());
}

template <typename T>
void testing_write_file_RSIO()
{
    // Step 1: Create a LocalMatrix in CSR format
    LocalMatrix<T> matrix;
    matrix.AllocateCSR("test", 5, 3, 3); // 3x3 matrix with 5 non-zero elements

    // Fill the matrix with some values
    // Matrix representation:
    // [ 1.0  0.0  2.0 ]
    // [ 0.0  3.0  0.0 ]
    // [ 4.0  0.0  5.0 ]
    int row_offsets[4] = {0, 2, 3, 5};
    int col_indices[5] = {0, 2, 1, 0, 2};
    T   values[5]      = {1.0, 2.0, 3.0, 4.0, 5.0};
    matrix.CopyFromCSR(row_offsets, col_indices, values);

    // Step 2: Write the matrix to a file in RSIO format
    const std::string filename = get_temp_dir() + "test_matrix.rsio";
    ASSERT_NO_THROW(matrix.WriteFileRSIO(filename));

    // Create a LocalMatrix
    LocalMatrix<T> mat;

    // Read the matrix from the RSIO file
    EXPECT_NO_THROW(mat.ReadFileRSIO(filename));

    // Clean up the test file
    std::remove(filename.c_str());
}

template <typename T>
void testing_local_apply(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);
    LocalVector<T> x, y;
    x.Allocate("x", matrix.GetN());
    y.Allocate("y", matrix.GetM());

    // Fill x with test values
    for(int i = 0; i < x.GetSize(); ++i)
        x[i] = static_cast<T>(i + 1);

    // Compute expected result
    std::vector<T> expected(y.GetSize(), 0);
    int*           row_offsets = new int[matrix.GetM() + 1];
    int*           col_indices = new int[matrix.GetNnz()];
    T*             values      = new T[matrix.GetNnz()];
    matrix.CopyToCSR(row_offsets, col_indices, values);

    for(int row = 0; row < matrix.GetM(); ++row)
    {
        for(int idx = row_offsets[row]; idx < row_offsets[row + 1]; ++idx)
            expected[row] += values[idx] * x[col_indices[idx]];
    }

    // Call Apply
    matrix.Apply(x, &y);

    // Compare
    for(int i = 0; i < y.GetSize(); ++i)
        EXPECT_NEAR(y[i], expected[i], getTolerance<T>() * std::abs(y[i]));

    delete[] row_offsets;
    delete[] col_indices;
    delete[] values;
}

template <typename T>
void testing_local_apply_add(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);
    LocalVector<T> x, y, y_orig, tmp;
    x.Allocate("x", matrix.GetN());
    y.Allocate("y", matrix.GetM());
    y_orig.Allocate("y_orig", matrix.GetM());
    tmp.Allocate("tmp", matrix.GetM());

    // Fill x and y with test values
    for(int i = 0; i < x.GetSize(); ++i)
        x[i] = static_cast<T>(i + 1);
    for(int i = 0; i < y.GetSize(); ++i)
        y[i] = static_cast<T>(2 * i);
    y_orig.CopyFrom(y);

    // Compute matrix * x
    matrix.Apply(x, &tmp);

    // Compute expected result
    std::vector<T> expected(y.GetSize());
    T              scalar = static_cast<T>(3);
    for(int i = 0; i < y.GetSize(); ++i)
        expected[i] = y_orig[i] + scalar * tmp[i];

    // Call ApplyAdd
    matrix.ApplyAdd(x, scalar, &y);

    // Compare
    for(int i = 0; i < y.GetSize(); ++i)
        EXPECT_NEAR(y[i], expected[i], getTolerance<T>() * std::abs(y[i]));
}

template <typename T>
void testing_local_extract_submatrix(Arguments argus)
{
    if(argus.matrix_type != "Laplacian2D")
    {
        return;
    }

    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);
    LocalMatrix<T> submatrix;
    int            row_offset = 0, col_offset = 0, row_size = 2, col_size = 2;
    EXPECT_NO_THROW(
        matrix.ExtractSubMatrix(row_offset, row_size, col_offset, col_size, &submatrix));
    auto orig_dense = extract_dense_matrix(matrix);
    auto sub_dense  = extract_dense_matrix(submatrix);

    for(int i = 0; i < submatrix.GetM(); ++i)
        for(int j = 0; j < submatrix.GetN(); ++j)
            EXPECT_EQ(sub_dense[i][j], orig_dense[row_offset + i][col_offset + j]);
}

template <typename T>
void testing_local_extract_u(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);

    // Extract U with both strictly and non-strictly upper
    LocalMatrix<T> U_strict, U_non_strict;
    EXPECT_NO_THROW(matrix.ExtractU(&U_strict, false));
    EXPECT_NO_THROW(matrix.ExtractU(&U_non_strict, true));

    // Both should have the same dimensions as the original
    EXPECT_EQ(U_strict.GetM(), matrix.GetM());
    EXPECT_EQ(U_strict.GetN(), matrix.GetN());
    EXPECT_EQ(U_non_strict.GetM(), matrix.GetM());
    EXPECT_EQ(U_non_strict.GetN(), matrix.GetN());

    // Compare dense representations
    auto orig_dense       = extract_dense_matrix(matrix);
    auto strict_dense     = extract_dense_matrix(U_strict);
    auto non_strict_dense = extract_dense_matrix(U_non_strict);

    int m = matrix.GetM();
    int n = matrix.GetN();

    // Strictly upper: only elements above diagonal
    for(int i = 0; i < m; ++i)
        for(int j = 0; j < n; ++j)
            if(j > i)
                EXPECT_EQ(strict_dense[i][j], orig_dense[i][j]);
            else
                EXPECT_EQ(strict_dense[i][j], static_cast<T>(0));

    // Non-strictly upper: diagonal and above
    for(int i = 0; i < m; ++i)
        for(int j = 0; j < n; ++j)
            if(j >= i)
                EXPECT_EQ(non_strict_dense[i][j], orig_dense[i][j]);
            else
                EXPECT_EQ(non_strict_dense[i][j], static_cast<T>(0));
}

template <typename T>
void testing_local_extract_l(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);

    // Extract L with both strictly and non-strictly lower
    LocalMatrix<T> L_strict, L_non_strict;
    EXPECT_NO_THROW(matrix.ExtractL(&L_strict, false));
    EXPECT_NO_THROW(matrix.ExtractL(&L_non_strict, true));

    // Both should have the same dimensions as the original
    EXPECT_EQ(L_strict.GetM(), matrix.GetM());
    EXPECT_EQ(L_strict.GetN(), matrix.GetN());
    EXPECT_EQ(L_non_strict.GetM(), matrix.GetM());
    EXPECT_EQ(L_non_strict.GetN(), matrix.GetN());

    // Compare dense representations
    auto orig_dense       = extract_dense_matrix(matrix);
    auto strict_dense     = extract_dense_matrix(L_strict);
    auto non_strict_dense = extract_dense_matrix(L_non_strict);

    int m = matrix.GetM();
    int n = matrix.GetN();

    // Strictly lower: only elements below diagonal
    for(int i = 0; i < m; ++i)
        for(int j = 0; j < n; ++j)
            if(j < i)
                EXPECT_EQ(strict_dense[i][j], orig_dense[i][j]);
            else
                EXPECT_EQ(strict_dense[i][j], static_cast<T>(0));

    // Non-strictly lower: diagonal and below
    for(int i = 0; i < m; ++i)
        for(int j = 0; j < n; ++j)
            if(j <= i)
                EXPECT_EQ(non_strict_dense[i][j], orig_dense[i][j]);
            else
                EXPECT_EQ(non_strict_dense[i][j], static_cast<T>(0));
}

template <typename T>
void testing_local_matrix_add(Arguments argus)
{
    LocalMatrix<T> matrix1;
    getTestMatrix<T>(argus, matrix1);
    LocalMatrix<T> matrix2;
    getTestMatrix<T>(argus, matrix2);

    // Save dense representations before addition
    auto dense1 = extract_dense_matrix(matrix1);
    auto dense2 = extract_dense_matrix(matrix2);

    // Perform matrix addition
    EXPECT_NO_THROW(matrix1.MatrixAdd(matrix2));

    // Extract the result as dense
    auto dense_sum = extract_dense_matrix(matrix1);

    // Check that the result is the element-wise sum
    int m = dense1.size();
    int n = m > 0 ? dense1[0].size() : 0;
    for(int i = 0; i < m; ++i)
        for(int j = 0; j < n; ++j)
            EXPECT_EQ(dense_sum[i][j], dense1[i][j] + dense2[i][j]);
}

template <typename T>
void testing_local_gershgorin(Arguments argus)
{
    if(argus.matrix_type != "Laplacian2D" || argus.matrix_type != "PermutedIdentity")
    {
        return;
    }

    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);
    T lower = 0, upper = 0;
    EXPECT_NO_THROW(matrix.Gershgorin(lower, upper));

    if(argus.matrix_type == "Laplacian2D")
    {
        // For Laplacian2D, the Gershgorin bounds should be [0, 4]
        EXPECT_NEAR(lower, static_cast<T>(0), getTolerance<T>());
        EXPECT_NEAR(upper, static_cast<T>(4), getTolerance<T>());
    }
    else if(argus.matrix_type == "PermutedIdentity")
    {
        // For PermutedIdentity, the Gershgorin bounds should be [1, 1]
        EXPECT_NEAR(lower, static_cast<T>(1), getTolerance<T>());
        EXPECT_NEAR(upper, static_cast<T>(1), getTolerance<T>());
    }
}

template <typename T>
void testing_local_scale_diagonal(Arguments argus)
{
    if(argus.matrix_type != "Laplacian2D")
    {
        return;
    }

    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);

    // Scale the diagonal by a factor of 2.0
    // This should not throw an exception
    // and should correctly scale the diagonal elements

    LocalVector<T> diag_vec;
    matrix.ExtractDiagonal(&diag_vec);

    EXPECT_NO_THROW(matrix.ScaleDiagonal(2.0));

    LocalVector<T> diag_vec_2;
    matrix.ExtractDiagonal(&diag_vec_2);

    // Validate that the diagonal elements are scaled correctly
    for(int i = 0; i < diag_vec.GetSize(); ++i)
    {
        EXPECT_NEAR(diag_vec_2[i], diag_vec[i] * 2.0, getTolerance<T>() * std::abs(diag_vec[i]));
    }
}

template <typename T>
void testing_local_add_scalar_diagonal(Arguments argus)
{
    if(argus.matrix_type != "Laplacian2D")
    {
        return;
    }

    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);

    // Add a scalar value of 1.0 to the diagonal elements
    // This should not throw an exception
    // and should correctly add the scalar to the diagonal elements

    LocalVector<T> diag_vec;
    matrix.ExtractDiagonal(&diag_vec);

    EXPECT_NO_THROW(matrix.AddScalarDiagonal(1.0));

    LocalVector<T> diag_vec_2;
    matrix.ExtractDiagonal(&diag_vec_2);

    // Validate that the diagonal elements are incremented correctly
    for(int i = 0; i < diag_vec.GetSize(); ++i)
    {
        EXPECT_NEAR(diag_vec_2[i], diag_vec[i] + 1.0, getTolerance<T>() * std::abs(diag_vec[i]));
    }
}

template <typename T>
void testing_local_add_scalar_off_diagonal(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);

    // Extract strictly lower and strictly upper matrices
    LocalMatrix<T> L, U;
    matrix.ExtractL(&L, false); // strictly lower
    matrix.ExtractU(&U, false); // strictly upper

    // Add 1.0 to strictly lower and strictly upper matrices
    L.AddScalar(1.0);
    U.AddScalar(1.0);

    // Extract diagonal matrix
    LocalMatrix<T> D;
    D.AllocateCSR("diag", matrix.GetM(), matrix.GetM(), matrix.GetN());
    LocalVector<T> diag_vec;
    matrix.ExtractDiagonal(&diag_vec);
    int              m = matrix.GetM();
    std::vector<int> row_offsets(m + 1), col_indices(m);
    std::vector<T>   values(m);
    for(int i = 0; i < m; ++i)
    {
        row_offsets[i] = i;
        col_indices[i] = i;
        values[i]      = diag_vec[i];
    }
    row_offsets[m] = m;
    D.CopyFromCSR(row_offsets.data(), col_indices.data(), values.data());

    // Now perform AddScalarOffDiagonal on the matrix under test
    EXPECT_NO_THROW(matrix.AddScalarOffDiagonal(1.0));

    // Validate by applying all matrices to a random vector and comparing results
    LocalVector<T> x, y_expected, y_actual, y_L, y_U, y_D;
    int            n = matrix.GetN();
    x.Allocate("x", n);
    y_expected.Allocate("y_expected", matrix.GetM());
    y_actual.Allocate("y_actual", matrix.GetM());
    y_L.Allocate("y_L", matrix.GetM());
    y_U.Allocate("y_U", matrix.GetM());
    y_D.Allocate("y_D", matrix.GetM());
    x.SetRandomUniform(static_cast<T>(0), static_cast<T>(1));

    L.Apply(x, &y_L);
    U.Apply(x, &y_U);
    D.Apply(x, &y_D);

    // y_expected = y_L + y_U + y_D
    for(int i = 0; i < y_expected.GetSize(); ++i)
        y_expected[i] = y_L[i] + y_U[i] + y_D[i];

    matrix.Apply(x, &y_actual);

    for(int i = 0; i < y_actual.GetSize(); ++i)
        EXPECT_NEAR(y_actual[i], y_expected[i], getTolerance<T>() * std::abs(y_expected[i]));
}

template <typename T>
void testing_local_matrix_mult(Arguments argus)
{
    LocalMatrix<T> matrix1;
    getTestMatrix<T>(argus, matrix1);

    if(matrix1.GetN() != matrix1.GetM())
    {
        // If the matrix is not square, we skip the test
        GTEST_SKIP() << "Matrix is not square, skipping matrix multiplication test.";
    }

    LocalMatrix<T> matrix2;
    getTestMatrix<T>(argus, matrix2);
    LocalMatrix<T> matrix3;
    getTestMatrix<T>(argus, matrix3);

    // Perform matrix multiplication
    EXPECT_NO_THROW(matrix1.MatrixMult(matrix2, matrix3));

    // Extract dense representations
    auto dense1 = extract_dense_matrix(matrix1);
    auto dense2 = extract_dense_matrix(matrix2);
    auto dense3 = extract_dense_matrix(matrix3);

    int m = matrix1.GetM();
    int n = matrix1.GetN();
    int k = matrix2.GetN();

    // Check the result: dense1 = dense2 * dense3
    for(int i = 0; i < m; ++i)
        for(int j = 0; j < n; ++j)
        {
            T expected = static_cast<T>(0);
            for(int l = 0; l < k; ++l)
                expected += dense2[i][l] * dense3[l][j];
            EXPECT_NEAR(dense1[i][j], expected, getTolerance<T>() * std::abs(expected));
        }
}

template <typename T>
void testing_local_triple_matrix_product(Arguments argus)
{
    LocalMatrix<T> matrixA, matrixB, matrixC, matrixD;

    getTestMatrix<T>(argus, matrixA);
    getTestMatrix<T>(argus, matrixB);
    getTestMatrix<T>(argus, matrixC);
    getTestMatrix<T>(argus, matrixD);
    // Perform triple matrix product
    // d = A * B * C
    EXPECT_NO_THROW(matrixD.TripleMatrixProduct(matrixA, matrixB, matrixC));

    // Prepare a random input vector of appropriate size (matching matrixD's column count)
    int            n = matrixD.GetN();
    LocalVector<T> x, y1, y2, y3, y_ref;
    x.Allocate("x", n);
    y1.Allocate("y1", matrixC.GetN());
    y2.Allocate("y2", matrixB.GetN());
    y3.Allocate("y3", matrixA.GetN());
    y_ref.Allocate("y_ref", matrixD.GetM());

    // Fill x with random values
    x.SetRandomUniform(static_cast<T>(0), static_cast<T>(1));

    // Reference: y_ref = matrixD * x
    matrixD.Apply(x, &y_ref);

    // Stepwise: y1 = matrixC * x
    matrixC.Apply(x, &y1);
    // y2 = matrixB * y1
    matrixB.Apply(y1, &y2);
    // y3 = matrixA * y2
    matrixA.Apply(y2, &y3);

    // Compare y3 (stepwise) with y_ref (direct)
    ASSERT_EQ(y3.GetSize(), y_ref.GetSize());
    for(int i = 0; i < y_ref.GetSize(); ++i)
        EXPECT_NEAR(y_ref[i], y3[i], getTolerance<T>() * std::abs(y_ref[i]));
}

template <typename T>
void testing_local_diagonal_matrix_mult_r(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);
    int n = matrix.GetN();

    LocalVector<T> diag;
    diag.Allocate("diag", matrix.GetN());
    for(int i = 0; i < diag.GetSize(); ++i)
        diag[i] = static_cast<T>(2);

    // Save original dense matrix
    auto orig_dense = extract_dense_matrix(matrix);

    // Perform right diagonal multiplication
    EXPECT_NO_THROW(matrix.DiagonalMatrixMultR(diag));

    // Extract new dense matrix
    auto new_dense = extract_dense_matrix(matrix);

    // Each column j should be scaled by diag[j]
    int m = matrix.GetM();
    for(int i = 0; i < m; ++i)
        for(int j = 0; j < n; ++j)
            EXPECT_NEAR(new_dense[i][j],
                        orig_dense[i][j] * diag[j],
                        getTolerance<T>() * std::abs(orig_dense[i][j] * diag[j]));
}

template <typename T>
void testing_local_diagonal_matrix_mult(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);
    int m = matrix.GetM();

    // Create a diagonal vector with known values
    LocalVector<T> diag;
    diag.Allocate("diag", m);
    for(int i = 0; i < diag.GetSize(); ++i)
        diag[i] = static_cast<T>(2);

    // Save original dense matrix
    auto orig_dense = extract_dense_matrix(matrix);

    // Perform left diagonal multiplication
    EXPECT_NO_THROW(matrix.DiagonalMatrixMult(diag));

    // Extract new dense matrix
    auto new_dense = extract_dense_matrix(matrix);

    // Each row i should be scaled by diag[i]
    int n = matrix.GetN();
    for(int i = 0; i < m; ++i)
        for(int j = 0; j < n; ++j)
            EXPECT_NEAR(new_dense[i][j],
                        orig_dense[i][j] * diag[i],
                        getTolerance<T>() * std::abs(orig_dense[i][j] * diag[i]));
}

template <typename T>
void testing_local_diagonal_matrix_mult_l(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);
    int m = matrix.GetM();

    // Create a diagonal vector with known values
    LocalVector<T> diag;
    diag.Allocate("diag", m);
    for(int i = 0; i < diag.GetSize(); ++i)
        diag[i] = static_cast<T>(2);

    // Save original dense matrix
    auto orig_dense = extract_dense_matrix(matrix);

    // Perform left diagonal multiplication
    EXPECT_NO_THROW(matrix.DiagonalMatrixMultL(diag));

    // Extract new dense matrix
    auto new_dense = extract_dense_matrix(matrix);

    // Each row i should be scaled by diag[i]
    int n = matrix.GetN();
    for(int i = 0; i < m; ++i)
        for(int j = 0; j < n; ++j)
            EXPECT_NEAR(new_dense[i][j],
                        orig_dense[i][j] * diag[i],
                        getTolerance<T>() * std::abs(orig_dense[i][j] * diag[i]));
}

template <typename T>
void testing_local_compress(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);
    // Optionally, set some small values to test removal
    int64_t nnz    = matrix.GetNnz();
    T*      values = new T[nnz];
    getMatrixVal(matrix, values);
    if(nnz > 1)
        values[1] = static_cast<T>(1e-10); // Set one value below threshold
    matrix.UpdateValuesCSR(values);

    // Compress with threshold 1e-8
    EXPECT_NO_THROW(matrix.Compress(1e-8));

    // Extract values after compression
    int64_t new_nnz     = matrix.GetNnz();
    int     m           = matrix.GetM();
    int     n           = matrix.GetN();
    int*    row_offsets = new int[m + 1];
    int*    col_indices = new int[new_nnz];
    T*      new_values  = new T[new_nnz];
    matrix.CopyToCSR(row_offsets, col_indices, new_values);

    // All remaining off-diagonal values should have abs(value) > 1e-8
    for(int row = 0; row < m; ++row)
    {
        for(int idx = row_offsets[row]; idx < row_offsets[row + 1]; ++idx)
        {
            int col = col_indices[idx];
            if(row != col)
                EXPECT_GT(std::abs(new_values[idx]), 1e-8);
        }
    }

    delete[] values;
    delete[] row_offsets;
    delete[] col_indices;
    delete[] new_values;
}

template <typename T>
void testing_local_replace_row_vector(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);
    const int n = matrix.GetN();

    const int max_row_to_replace = 5 < matrix.GetM() ? 5 : matrix.GetM();

    for(int row_to_replace = 0; row_to_replace < max_row_to_replace; ++row_to_replace)
    {
        // Create a vector with known values
        LocalVector<T> vec;
        vec.Allocate("RowVector", n);
        for(int j = 0; j < n; ++j)
            vec[j] = static_cast<T>(j + 10);

        // Save original dense matrix
        auto orig_dense = extract_dense_matrix(matrix);

        // Replace the row
        EXPECT_NO_THROW(matrix.ReplaceRowVector(row_to_replace, vec));

        // Extract new dense matrix
        auto new_dense = extract_dense_matrix(matrix);

        // Validate the replaced row
        for(int j = 0; j < n; ++j)
            EXPECT_EQ(new_dense[row_to_replace][j], vec[j]);

        // Validate that other rows are unchanged
        for(int i = 0; i < matrix.GetM(); ++i)
            if(i != row_to_replace)
                for(int j = 0; j < n; ++j)
                    EXPECT_EQ(new_dense[i][j], orig_dense[i][j]);
    }
}

template <typename T>
void testing_local_extract_row_vector(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);
    LocalVector<T> vec;
    vec.Allocate("ExtractedRow", matrix.GetN());

    // Extract row vector
    matrix.ExtractRowVector(1, &vec);

    // Validate the result using the dense matrix
    auto dense = extract_dense_matrix(matrix);
    for(int j = 0; j < matrix.GetN(); ++j)
        EXPECT_EQ(vec[j], dense[1][j]);
}

template <typename T>
void testing_local_key(Arguments argus)
{
    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);
    long int row_key = 0, col_key = 0, val_key = 0;

    // Call Key and check it does not throw
    EXPECT_NO_THROW(matrix.Key(row_key, col_key, val_key));

    // Check that the keys are set (for a non-empty matrix, at least one should be nonzero)
    EXPECT_TRUE(row_key != 0 || col_key != 0 || val_key != 0);
}

template <typename T>
void testing_initial_pairwise_aggregation()
{
    auto matrix = getTestMatrix<T>();

    // Prepare required arguments
    double           strength = 0.25;
    int              nrow     = matrix.GetM();
    LocalVector<int> G;
    int*             rG = NULL;
    int              nc = 0, nc2 = 0, nc3 = 0;

    // Allocate G with the correct size
    G.Allocate("G", nrow);

    // Call InitialPairwiseAggregation with all required arguments
    EXPECT_NO_THROW(matrix.InitialPairwiseAggregation(strength, nc, &G, nc2, &rG, nc3, 0));

    // Check that the output vectors have the expected size
    EXPECT_EQ(G.GetSize(), nrow);

    // Optionally, check that the output vectors contain valid indices
    for(int i = 0; i < nrow; ++i)
    {
        EXPECT_GE(G[i], 0);
        EXPECT_GE(rG[i], 0);
    }
}

template <typename T>
void testing_initial_pairwise_aggregation_2()
{
    // Create a test matrix and a coarse matrix
    auto mat        = getTestMatrix<T>();
    auto coarse_mat = getTestMatrix<T>();

    // Prepare required arguments
    double           strength = 0.25;
    int              nrow     = mat.GetM();
    LocalVector<int> G;
    int*             rG = nullptr;
    int              nc = 0, nc2 = 0, nc3 = 0;

    // Allocate G with the correct size
    G.Allocate("G", nrow);

    // Call InitialPairwiseAggregation with all required arguments
    EXPECT_NO_THROW(mat.InitialPairwiseAggregation(coarse_mat, strength, nc, &G, nc2, &rG, nc3, 0));

    // Check that the output vectors have the expected size
    EXPECT_EQ(G.GetSize(), nrow);

    // Optionally, check that the output vectors contain valid indices
    for(int i = 0; i < nrow; ++i)
    {
        EXPECT_GE(G[i], 0);
        EXPECT_GE(rG[i], 0);
    }
}

#endif // TESTING_LOCAL_MATRIX_HPP
