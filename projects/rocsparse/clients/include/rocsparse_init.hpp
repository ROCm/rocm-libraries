/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef ROCSPARSE_INIT_HPP
#define ROCSPARSE_INIT_HPP

#include "rocsparse_host.hpp"
#include "rocsparse_random.hpp"

#include <fstream>

/* ==================================================================================== */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same value

// Initialize vector with random values
template <typename T>
void rocsparse_init(T*     A,
                    size_t M,
                    size_t N,
                    size_t lda,
                    size_t stride      = 0,
                    size_t batch_count = 1,
                    T      a           = static_cast<T>(0),
                    T      b           = static_cast<T>(1));

// Initialize vector with random integer values
template <typename T>
void rocsparse_init_exact(T*     A,
                          size_t M,
                          size_t N,
                          size_t lda,
                          size_t stride      = 0,
                          size_t batch_count = 1,
                          int    a           = 1,
                          int    b           = 10);

template <typename T>
void rocsparse_init(std::vector<T>& A,
                    size_t          M,
                    size_t          N,
                    size_t          lda,
                    size_t          stride      = 0,
                    size_t          batch_count = 1,
                    T               a           = static_cast<T>(0),
                    T               b           = static_cast<T>(1));

// Initialize vector with random integer values
template <typename T>
void rocsparse_init_exact(std::vector<T>& A,
                          size_t          M,
                          size_t          N,
                          size_t          lda,
                          size_t          stride      = 0,
                          size_t          batch_count = 1,
                          int             a           = 1,
                          int             b           = 10);

// Initializes sparse index vector with nnz entries ranging from start to end
template <typename I>
void rocsparse_init_index(std::vector<I>& x, size_t nnz, size_t start, size_t end);

// Initialize matrix so adjacent entries have alternating sign.
// In gemm if either A or B are initialized with alernating
// sign the reduction sum will be summing positive
// and negative numbers, so it should not get too large.
// This helps reduce floating point inaccuracies for 16bit
// arithmetic where the exponent has only 5 bits, and the
// mantissa 10 bits.
template <typename T>
void rocsparse_init_alternating_sign(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1);

/* ==================================================================================== */
/*! \brief  Initialize an array with random data, with NaN where appropriate */

template <typename T>
void rocsparse_init_nan(T* A, size_t N);

template <typename T>
void rocsparse_init_nan(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1);

/* ==================================================================================== */
/*! \brief  Generate a random sparse matrix in COO format */
template <typename I, typename T>
void rocsparse_init_coo_matrix(std::vector<I>&      row_ind,
                               std::vector<I>&      col_ind,
                               std::vector<T>&      val,
                               I                    M,
                               I                    N,
                               int64_t              nnz,
                               rocsparse_index_base base,
                               bool                 full_rank = false,
                               bool                 to_int    = false);

/* ==================================================================================== */
/*! \brief  Generate 2D 9pt laplacian on unit square in CSR format */
template <typename I, typename J, typename T>
void rocsparse_init_csr_laplace2d(std::vector<I>&      row_ptr,
                                  std::vector<J>&      col_ind,
                                  std::vector<T>&      val,
                                  int32_t              dim_x,
                                  int32_t              dim_y,
                                  J&                   M,
                                  J&                   N,
                                  I&                   nnz,
                                  rocsparse_index_base base);

/* ==================================================================================== */
/*! \brief  Generate 2D 9pt laplacian on unit square in COO format */
template <typename I, typename T>
void rocsparse_init_coo_laplace2d(std::vector<I>&      row_ind,
                                  std::vector<I>&      col_ind,
                                  std::vector<T>&      val,
                                  int32_t              dim_x,
                                  int32_t              dim_y,
                                  I&                   M,
                                  I&                   N,
                                  int64_t&             nnz,
                                  rocsparse_index_base base);

/* ==================================================================================== */
/*! \brief  Generate 2D 9pt laplacian on unit square in GEBSR format */
template <typename I, typename J, typename T>
void rocsparse_init_gebsr_laplace2d(std::vector<I>&      row_ptr,
                                    std::vector<J>&      col_ind,
                                    std::vector<T>&      val,
                                    int32_t              dim_x,
                                    int32_t              dim_y,
                                    J&                   Mb,
                                    J&                   Nb,
                                    I&                   nnzb,
                                    J                    row_block_dim,
                                    J                    col_block_dim,
                                    rocsparse_index_base base);

/* ==================================================================================== */
/*! \brief  Generate 2D 9pt laplacian on unit square in ELL format */
template <typename I, typename T>
void rocsparse_init_ell_laplace2d(std::vector<I>&      col_ind,
                                  std::vector<T>&      val,
                                  int32_t              dim_x,
                                  int32_t              dim_y,
                                  I&                   M,
                                  I&                   N,
                                  I&                   width,
                                  rocsparse_index_base base);

/* ==================================================================================== */
/*! \brief  Generate 3D 27pt laplacian on unit square in CSR format */
template <typename I, typename J, typename T>
void rocsparse_init_csr_laplace3d(std::vector<I>&      row_ptr,
                                  std::vector<J>&      col_ind,
                                  std::vector<T>&      val,
                                  int32_t              dim_x,
                                  int32_t              dim_y,
                                  int32_t              dim_z,
                                  J&                   M,
                                  J&                   N,
                                  I&                   nnz,
                                  rocsparse_index_base base);

/* ==================================================================================== */
/*! \brief  Generate 3D 27pt laplacian on unit square in COO format */
template <typename I, typename T>
void rocsparse_init_coo_laplace3d(std::vector<I>&      row_ind,
                                  std::vector<I>&      col_ind,
                                  std::vector<T>&      val,
                                  int32_t              dim_x,
                                  int32_t              dim_y,
                                  int32_t              dim_z,
                                  I&                   M,
                                  I&                   N,
                                  int64_t&             nnz,
                                  rocsparse_index_base base);

/* ==================================================================================== */
/*! \brief  Generate 3D 27pt laplacian on unit square in GEBSR format */
template <typename I, typename J, typename T>
void rocsparse_init_gebsr_laplace3d(std::vector<I>&      row_ptr,
                                    std::vector<J>&      col_ind,
                                    std::vector<T>&      val,
                                    int32_t              dim_x,
                                    int32_t              dim_y,
                                    int32_t              dim_z,
                                    J&                   Mb,
                                    J&                   Nb,
                                    I&                   nnzb,
                                    J                    row_block_dim,
                                    J                    col_block_dim,
                                    rocsparse_index_base base);

/* ============================================================================================ */
/*! \brief  Read matrix from mtx file in CSR format */
template <typename I, typename J, typename T>
void rocsparse_init_csr_mtx(const char*          filename,
                            std::vector<I>&      csr_row_ptr,
                            std::vector<J>&      csr_col_ind,
                            std::vector<T>&      csr_val,
                            J&                   M,
                            J&                   N,
                            I&                   nnz,
                            rocsparse_index_base base);

/* ============================================================================================ */
/*! \brief  Read matrix from mtx file in COO format */
template <typename I, typename T>
void rocsparse_init_coo_mtx(const char*          filename,
                            std::vector<I>&      coo_row_ind,
                            std::vector<I>&      coo_col_ind,
                            std::vector<T>&      coo_val,
                            I&                   M,
                            I&                   N,
                            int64_t&             nnz,
                            rocsparse_index_base base);

/* ============================================================================================ */
/*! \brief  Read matrix from mtx file in GEBSR format */
template <typename I, typename J, typename T>
void rocsparse_init_gebsr_mtx(const char*          filename,
                              std::vector<I>&      bsr_row_ptr,
                              std::vector<J>&      bsr_col_ind,
                              std::vector<T>&      bsr_val,
                              J&                   Mb,
                              J&                   Nb,
                              I&                   nnzb,
                              J                    row_block_dim,
                              J                    col_block_dim,
                              rocsparse_index_base base);

/* ============================================================================================ */
/*! \brief  Read matrix from smtx file in CSR format */
template <typename I, typename J, typename T>
void rocsparse_init_csr_smtx(const char*          filename,
                             std::vector<I>&      csr_row_ptr,
                             std::vector<J>&      csr_col_ind,
                             std::vector<T>&      csr_val,
                             J&                   M,
                             J&                   N,
                             I&                   nnz,
                             rocsparse_index_base base);

/* ============================================================================================ */
/*! \brief  Read matrix from smtx file in COO format */
template <typename I, typename T>
void rocsparse_init_coo_smtx(const char*          filename,
                             std::vector<I>&      coo_row_ind,
                             std::vector<I>&      coo_col_ind,
                             std::vector<T>&      coo_val,
                             I&                   M,
                             I&                   N,
                             int64_t&             nnz,
                             rocsparse_index_base base);

/* ============================================================================================ */
/*! \brief  Read matrix from smtx file in GEBSR format */
template <typename I, typename J, typename T>
void rocsparse_init_gebsr_smtx(const char*          filename,
                               std::vector<I>&      bsr_row_ptr,
                               std::vector<J>&      bsr_col_ind,
                               std::vector<T>&      bsr_val,
                               J&                   Mb,
                               J&                   Nb,
                               I&                   nnzb,
                               J                    row_block_dim,
                               J                    col_block_dim,
                               rocsparse_index_base base);

/* ============================================================================================ */
/*! \brief  Read matrix from bsmtx file in CSR format */
template <typename I, typename J, typename T>
void rocsparse_init_csr_bsmtx(const char*          filename,
                              std::vector<I>&      csr_row_ptr,
                              std::vector<J>&      csr_col_ind,
                              std::vector<T>&      csr_val,
                              J&                   M,
                              J&                   N,
                              I&                   nnz,
                              rocsparse_index_base base);

/* ============================================================================================ */
/*! \brief  Read matrix from bsmtx file in COO format */
template <typename I, typename T>
void rocsparse_init_coo_bsmtx(const char*          filename,
                              std::vector<I>&      coo_row_ind,
                              std::vector<I>&      coo_col_ind,
                              std::vector<T>&      coo_val,
                              I&                   M,
                              I&                   N,
                              int64_t&             nnz,
                              rocsparse_index_base base);

/* ============================================================================================ */
/*! \brief  Read matrix from bsmtx file in GEBSR format */
template <typename I, typename J, typename T>
void rocsparse_init_gebsr_bsmtx(const char*          filename,
                                std::vector<I>&      bsr_row_ptr,
                                std::vector<J>&      bsr_col_ind,
                                std::vector<T>&      bsr_val,
                                J&                   Mb,
                                J&                   Nb,
                                I&                   nnzb,
                                J                    row_block_dim,
                                J                    col_block_dim,
                                rocsparse_index_base base);

/* ==================================================================================== */
/*! \brief  Read matrix from binary file in rocALUTION format */
template <typename I, typename J, typename T>
void rocsparse_init_csr_rocalution(const char*          filename,
                                   std::vector<I>&      row_ptr,
                                   std::vector<J>&      col_ind,
                                   std::vector<T>&      val,
                                   J&                   M,
                                   J&                   N,
                                   I&                   nnz,
                                   rocsparse_index_base base);

/* ==================================================================================== */
/*! \brief  Read matrix from binary file in rocALUTION format */
template <typename I, typename T>
void rocsparse_init_coo_rocalution(const char*          filename,
                                   std::vector<I>&      row_ind,
                                   std::vector<I>&      col_ind,
                                   std::vector<T>&      val,
                                   I&                   M,
                                   I&                   N,
                                   int64_t&             nnz,
                                   rocsparse_index_base base);

/* ==================================================================================== */
/*! \brief  Read matrix from binary file in rocALUTION format */
template <typename I, typename J, typename T>
void rocsparse_init_gebsr_rocalution(const char*          filename,
                                     std::vector<I>&      row_ptr,
                                     std::vector<J>&      col_ind,
                                     std::vector<T>&      val,
                                     J&                   Mb,
                                     J&                   Nb,
                                     I&                   nnzb,
                                     J                    row_block_dim,
                                     J                    col_block_dim,
                                     rocsparse_index_base base);

/* ==================================================================================== */
/*! \brief  Read matrix from binary file in rocSPARSEIO format */
template <typename I, typename J, typename T>
void rocsparse_init_csr_rocsparseio(const char*          filename,
                                    std::vector<I>&      row_ptr,
                                    std::vector<J>&      col_ind,
                                    std::vector<T>&      val,
                                    J&                   M,
                                    J&                   N,
                                    I&                   nnz,
                                    rocsparse_index_base base);

/* ==================================================================================== */
/*! \brief  Read matrix from binary file in rocSPARSEIO format */
template <typename I, typename T>
void rocsparse_init_coo_rocsparseio(const char*          filename,
                                    std::vector<I>&      row_ind,
                                    std::vector<I>&      col_ind,
                                    std::vector<T>&      val,
                                    I&                   M,
                                    I&                   N,
                                    int64_t&             nnz,
                                    rocsparse_index_base base);

/* ==================================================================================== */
/*! \brief  Read matrix from binary file in rocSPARSEIO format */
template <typename I, typename J, typename T>
void rocsparse_init_gebsr_rocsparseio(const char*          filename,
                                      std::vector<I>&      row_ptr,
                                      std::vector<J>&      col_ind,
                                      std::vector<T>&      val,
                                      rocsparse_direction  dir,
                                      J&                   Mb,
                                      J&                   Nb,
                                      I&                   nnzb,
                                      J                    row_block_dim,
                                      J                    col_block_dim,
                                      rocsparse_index_base base);

/* ==================================================================================== */
/*! \brief  Generate a random sparse matrix in CSR format */
template <typename I, typename J, typename T>
void rocsparse_init_csr_random(std::vector<I>&            row_ptr,
                               std::vector<J>&            col_ind,
                               std::vector<T>&            val,
                               J                          M,
                               J                          N,
                               I&                         nnz,
                               rocsparse_index_base       base,
                               rocsparse_matrix_init_kind init_kind,
                               bool                       full_rank = false,
                               bool                       to_int    = false);

/* ==================================================================================== */
/*! \brief  Generate a random sparse matrix in COO format */
template <typename I, typename T>
void rocsparse_init_coo_random(std::vector<I>&            row_ind,
                               std::vector<I>&            col_ind,
                               std::vector<T>&            val,
                               I                          M,
                               I                          N,
                               int64_t&                   nnz,
                               rocsparse_index_base       base,
                               rocsparse_matrix_init_kind init_kind,
                               bool                       full_rank = false,
                               bool                       to_int    = false);

/* ==================================================================================== */
/*! \brief  Generate a random sparse matrix in GEBSR format */
template <typename I, typename J, typename T>
void rocsparse_init_gebsr_random(std::vector<I>&            row_ptr,
                                 std::vector<J>&            col_ind,
                                 std::vector<T>&            val,
                                 J                          Mb,
                                 J                          Nb,
                                 I&                         nnzb,
                                 J                          row_block_dim,
                                 J                          col_block_dim,
                                 rocsparse_index_base       base,
                                 rocsparse_matrix_init_kind init_kind,
                                 bool                       full_rank = false,
                                 bool                       to_int    = false);

/* ==================================================================================== */
/*! \brief  Generate a tridiagonal sparse matrix in COO format */
template <typename I, typename T>
void rocsparse_init_coo_tridiagonal(std::vector<I>&      row_ind,
                                    std::vector<I>&      col_ind,
                                    std::vector<T>&      val,
                                    I                    M,
                                    I                    N,
                                    int64_t&             nnz,
                                    rocsparse_index_base base,
                                    I                    l,
                                    I                    u);

/* ==================================================================================== */
/*! \brief  Generate a tridiagonal sparse matrix in CSR format */
template <typename I, typename J, typename T>
void rocsparse_init_csr_tridiagonal(std::vector<I>&      row_ptr,
                                    std::vector<J>&      col_ind,
                                    std::vector<T>&      val,
                                    J                    M,
                                    J                    N,
                                    I&                   nnz,
                                    rocsparse_index_base base,
                                    J                    l,
                                    J                    u);

/* ==================================================================================== */
/*! \brief  Generate a tridiagonal sparse matrix in GEBSR format */
template <typename I, typename J, typename T>
void rocsparse_init_gebsr_tridiagonal(std::vector<I>&      row_ptr,
                                      std::vector<J>&      col_ind,
                                      std::vector<T>&      val,
                                      J                    Mb,
                                      J                    Nb,
                                      I&                   nnzb,
                                      J                    row_block_dim,
                                      J                    col_block_dim,
                                      rocsparse_index_base base,
                                      J                    l,
                                      J                    u);

/* ==================================================================================== */
/*! \brief  Generate a penta diagonal sparse matrix in COO format */
template <typename I, typename T>
void rocsparse_init_coo_pentadiagonal(std::vector<I>&      row_ind,
                                      std::vector<I>&      col_ind,
                                      std::vector<T>&      val,
                                      I                    M,
                                      I                    N,
                                      int64_t&             nnz,
                                      rocsparse_index_base base,
                                      I                    ll,
                                      I                    l,
                                      I                    u,
                                      I                    uu);

/* ==================================================================================== */
/*! \brief  Generate a penta diagonal sparse matrix in CSR format */
template <typename I, typename J, typename T>
void rocsparse_init_csr_pentadiagonal(std::vector<I>&      row_ptr,
                                      std::vector<J>&      col_ind,
                                      std::vector<T>&      val,
                                      J                    M,
                                      J                    N,
                                      I&                   nnz,
                                      rocsparse_index_base base,
                                      J                    ll,
                                      J                    l,
                                      J                    u,
                                      J                    uu);

/* ==================================================================================== */
/*! \brief  Generate a penta diagonal sparse matrix in GEBSR format */
template <typename I, typename J, typename T>
void rocsparse_init_gebsr_pentadiagonal(std::vector<I>&      row_ptr,
                                        std::vector<J>&      col_ind,
                                        std::vector<T>&      val,
                                        J                    Mb,
                                        J                    Nb,
                                        I&                   nnzb,
                                        J                    row_block_dim,
                                        J                    col_block_dim,
                                        rocsparse_index_base base,
                                        J                    ll,
                                        J                    l,
                                        J                    u,
                                        J                    uu);

#endif // ROCSPARSE_INIT_HPP
