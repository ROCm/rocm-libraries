/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef ROCSPARSE_EXPORTER_HPP
#define ROCSPARSE_EXPORTER_HPP

#include "rocsparse_check.hpp"
#include "rocsparse_traits.hpp"

#include "rocsparse_matrix_coo.hpp"
#include "rocsparse_matrix_csx.hpp"
#include "rocsparse_matrix_dense.hpp"
#include "rocsparse_matrix_gebsx.hpp"
#include "rocsparse_vector.hpp"

template <typename IMPL>
class rocsparse_exporter
{
protected:
    rocsparse_exporter()  = default;
    ~rocsparse_exporter() = default;

public:
    template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
    rocsparse_status write_sparse_csx(rocsparse_direction dir,
                                      J                   m,
                                      J                   n,
                                      I                   nnz,
                                      const I* __restrict__ ptr,
                                      const J* __restrict__ ind,
                                      const T* __restrict__ val,
                                      rocsparse_index_base base)

    {
        return static_cast<IMPL&>(*this).write_sparse_csx(dir, m, n, nnz, ptr, ind, val, base);
    }

    template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
    rocsparse_status write_sparse_gebsx(rocsparse_direction dir,
                                        rocsparse_direction dirb,
                                        J                   mb,
                                        J                   nb,
                                        I                   nnzb,
                                        J                   block_dim_row,
                                        J                   block_dim_column,
                                        const I* __restrict__ ptr,
                                        const J* __restrict__ ind,
                                        const T* __restrict__ val,
                                        rocsparse_index_base base)
    {
        return static_cast<IMPL&>(*this).write_sparse_gebsx(
            dir, dirb, mb, nb, nnzb, block_dim_row, block_dim_column, ptr, ind, val, base);
    }

    template <typename T, typename I = rocsparse_int>
    rocsparse_status write_dense_vector(I size, const T* __restrict__ x, I incx)
    {
        return static_cast<IMPL&>(*this).write_dense_vector(size, x, incx);
    }

    template <typename T, typename I = rocsparse_int>
    rocsparse_status
        write_dense_matrix(rocsparse_order order, I m, I n, const T* __restrict__ x, I ld)
    {
        return static_cast<IMPL&>(*this).write_dense_matrix(order, m, n, x, ld);
    }

    template <typename T, typename I = rocsparse_int>
    rocsparse_status write_sparse_coo(I m,
                                      I n,
                                      I nnz,
                                      const I* __restrict__ row_ind,
                                      const I* __restrict__ col_ind,
                                      const T* __restrict__ val,
                                      rocsparse_index_base base)
    {
        return static_cast<IMPL&>(*this).write_sparse_coo(m, n, nnz, row_ind, col_ind, val, base);
    }

    template <rocsparse_direction DIRECTION, typename T, typename I, typename J>
    rocsparse_status write(const host_csx_matrix<DIRECTION, T, I, J>& that_)
    {
        return this->write_sparse_csx<T, I, J>(
            that_.dir, that_.m, that_.n, that_.nnz, that_.ptr, that_.ind, that_.val, that_.base);
    }

    template <typename T, typename I>
    rocsparse_status write(const host_coo_matrix<T, I>& that_)
    {
        return this->write_sparse_coo<T, I>(
            that_.m, that_.n, that_.nnz, that_.row_ind, that_.col_ind, that_.val, that_.base);
    }

    template <rocsparse_direction DIRECTION, typename T, typename I, typename J>
    rocsparse_status write(const host_gebsx_matrix<DIRECTION, T, I, J>& that_)
    {
        return this->write_sparse_gebsx<T, I, J>(that_.dir,
                                                 that_.block_direction,
                                                 that_.mb,
                                                 that_.nb,
                                                 that_.nnzb,
                                                 that_.row_block_dim,
                                                 that_.col_block_dim,
                                                 that_.ptr,
                                                 that_.ind,
                                                 that_.val,
                                                 that_.base);
    }

    template <typename T, typename I>
    rocsparse_status write(const host_dense_matrix<T, I>& that_)
    {
        return this->write_dense_matrix<T, I>(
            that_.order, that_.m, that_.n, that_.data(), that_.ld);
    }

    template <typename T>
    rocsparse_status write(const host_dense_vector<T>& that_)
    {
        static constexpr size_t one = static_cast<size_t>(1);
        return this->write_dense_vector<T, size_t>(that_.size(), that_.val, one);
    }
};

#endif
