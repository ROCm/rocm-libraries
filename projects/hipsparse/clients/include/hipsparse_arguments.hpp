/*! \file */
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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

/*! \file
 *  \brief hipsparse_arguments.hpp provides a class to parse command arguments in both,
 *  clients and gtest.
 */

#pragma once

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "hipsparse_arguments_support.hpp"
#include "hipsparse_datatype2string.hpp"

template <typename T>
static T convert_alpha_beta(double r, double i)
{
    return static_cast<T>(r);
}

struct Arguments
{
    int M;
    int N;
    int K;
    int nnz;
    int block_dim;
    int row_block_dimA;
    int col_block_dimA;
    int row_block_dimB;
    int col_block_dimB;

    int lda;
    int ldb;
    int ldc;

    int batch_count;

    hipsparseIndexType_t index_type_I;
    hipsparseIndexType_t index_type_J;
    hipDataType          compute_type;

    double alpha;
    double alphai;
    double beta;
    double betai;
    double threshold;
    double percentage;

    hipsparseOperation_t transA;
    hipsparseOperation_t transB;
    hipsparseIndexBase_t baseA;
    hipsparseIndexBase_t baseB;
    hipsparseIndexBase_t baseC;
    hipsparseIndexBase_t baseD;

    hipsparseAction_t       action;
    hipsparseHybPartition_t part;
    hipsparseDiagType_t     diag_type;
    hipsparseFillMode_t     fill_mode;
    hipsparseSolvePolicy_t  solve_policy;

    hipsparseDirection_t dirA;
    hipsparseOrder_t     orderA;
    hipsparseOrder_t     orderB;
    hipsparseOrder_t     orderC;
    hipsparseFormat_t    formatA;
    hipsparseFormat_t    formatB;

    int csr2csc_alg;
    int dense2sparse_alg;
    int sparse2dense_alg;
    int sddmm_alg;
    int spgemm_alg;
    int spmm_alg;
    int spmv_alg;
    int spsm_alg;
    int spsv_alg;

    int    numericboost;
    double boosttol;
    double boostval;
    double boostvali;

    int ell_width;
    int permute;
    int gtsv_alg;
    int gpsv_alg;

    int unit_check;
    int timing;
    int iters;

    std::string filename;
    std::string function_name;

    Arguments()
    {
        this->M              = -1;
        this->N              = -1;
        this->K              = -1;
        this->nnz            = -1;
        this->block_dim      = 2;
        this->row_block_dimA = 2;
        this->col_block_dimA = 2;
        this->row_block_dimB = 2;
        this->col_block_dimB = 2;

        this->lda = -1;
        this->ldb = -1;
        this->ldc = -1;

        this->batch_count = -1;

        this->index_type_I = HIPSPARSE_INDEX_32I;
        this->index_type_J = HIPSPARSE_INDEX_32I;
        this->compute_type = HIP_R_32F;

        this->alpha      = 0.0;
        this->alphai     = 0.0;
        this->beta       = 0.0;
        this->betai      = 0.0;
        this->threshold  = 0.0;
        this->percentage = 0.0;

        this->transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
        this->transB = HIPSPARSE_OPERATION_NON_TRANSPOSE;
        this->baseA  = HIPSPARSE_INDEX_BASE_ZERO;
        this->baseB  = HIPSPARSE_INDEX_BASE_ZERO;
        this->baseC  = HIPSPARSE_INDEX_BASE_ZERO;
        this->baseD  = HIPSPARSE_INDEX_BASE_ZERO;

        this->action       = HIPSPARSE_ACTION_NUMERIC;
        this->part         = HIPSPARSE_HYB_PARTITION_AUTO;
        this->diag_type    = HIPSPARSE_DIAG_TYPE_NON_UNIT;
        this->fill_mode    = HIPSPARSE_FILL_MODE_LOWER;
        this->solve_policy = HIPSPARSE_SOLVE_POLICY_NO_LEVEL;

        this->dirA    = HIPSPARSE_DIRECTION_ROW;
        this->orderA  = HIPSPARSE_ORDER_COL;
        this->orderB  = HIPSPARSE_ORDER_COL;
        this->orderC  = HIPSPARSE_ORDER_COL;
        this->formatA = HIPSPARSE_FORMAT_COO;
        this->formatB = HIPSPARSE_FORMAT_COO;

        this->csr2csc_alg      = csr2csc_alg_support::get_default_algorithm();
        this->dense2sparse_alg = dense2sparse_alg_support::get_default_algorithm();
        this->sparse2dense_alg = sparse2dense_alg_support::get_default_algorithm();
        this->sddmm_alg        = sddmm_alg_support::get_default_algorithm();
        this->spgemm_alg       = spgemm_alg_support::get_default_algorithm();
        this->spmm_alg         = spmm_alg_support::get_default_algorithm();
        this->spmv_alg         = spmv_alg_support::get_default_algorithm();
        this->spsm_alg         = spsm_alg_support::get_default_algorithm();
        this->spsv_alg         = spsv_alg_support::get_default_algorithm();

        this->numericboost = 0;
        this->boosttol     = 0.0;
        this->boostval     = 1.0;
        this->boostvali    = 0.0;

        this->ell_width = 0;
        this->permute   = 0;
        this->gtsv_alg  = 0;
        this->gpsv_alg  = 0;

        this->unit_check = 1;
        this->timing     = 0;
        this->iters      = 10;

        this->filename      = "";
        this->function_name = "";
    }

    template <typename T>
    T get_alpha() const
    {
        return convert_alpha_beta<T>(alpha, alphai);
    }

    template <typename T>
    T get_beta() const
    {
        return convert_alpha_beta<T>(beta, betai);
    }

    template <typename T>
    T get_threshold() const
    {
        return threshold;
    }

    template <typename T>
    T get_percentage() const
    {
        return percentage;
    }
};