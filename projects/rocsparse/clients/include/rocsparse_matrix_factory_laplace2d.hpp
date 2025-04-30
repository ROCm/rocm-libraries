/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef ROCSPARSE_MATRIX_FACTORY_LAPLACE2D_HPP
#define ROCSPARSE_MATRIX_FACTORY_LAPLACE2D_HPP

#include "rocsparse_matrix_factory_base.hpp"

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
struct rocsparse_matrix_factory_laplace2d : public rocsparse_matrix_factory_base<T, I, J>
{
private:
    J m_dimx, m_dimy;

public:
    rocsparse_matrix_factory_laplace2d(J dimx, J dimy);

    virtual void init_csr(std::vector<I>&        csr_row_ptr,
                          std::vector<J>&        csr_col_ind,
                          std::vector<T>&        csr_val,
                          J&                     M,
                          J&                     N,
                          I&                     nnz,
                          rocsparse_index_base   base,
                          rocsparse_matrix_type  matrix_type,
                          rocsparse_fill_mode    uplo,
                          rocsparse_storage_mode storage) override;

    virtual void init_coo(std::vector<I>&        coo_row_ind,
                          std::vector<I>&        coo_col_ind,
                          std::vector<T>&        coo_val,
                          I&                     M,
                          I&                     N,
                          int64_t&               nnz,
                          rocsparse_index_base   base,
                          rocsparse_matrix_type  matrix_type,
                          rocsparse_fill_mode    uplo,
                          rocsparse_storage_mode storage) override;

    virtual void init_gebsr(std::vector<I>&        bsr_row_ptr,
                            std::vector<J>&        bsr_col_ind,
                            std::vector<T>&        bsr_val,
                            rocsparse_direction    dirb,
                            J&                     Mb,
                            J&                     Nb,
                            I&                     nnzb,
                            J&                     row_block_dim,
                            J&                     col_block_dim,
                            rocsparse_index_base   base,
                            rocsparse_matrix_type  matrix_type,
                            rocsparse_fill_mode    uplo,
                            rocsparse_storage_mode storage) override;
};

#endif // ROCSPARSE_MATRIX_FACTORY_LAPLACE2D_HPP
