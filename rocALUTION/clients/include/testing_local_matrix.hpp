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

#endif // TESTING_LOCAL_MATRIX_HPP
