/*! \file */
/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef ROCSPARSE_EXPORTER_ASCII_HPP
#define ROCSPARSE_EXPORTER_ASCII_HPP

#include "rocsparse_exporter.hpp"

class rocsparse_exporter_ascii : public rocsparse_exporter<rocsparse_exporter_ascii>
{
protected:
    std::string m_filename{};

public:
    ~rocsparse_exporter_ascii();
    rocsparse_exporter_ascii(const std::string& filename_);

    template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
    rocsparse_status write_sparse_csx(rocsparse_direction dir,
                                      J                   m,
                                      J                   n,
                                      I                   nnz,
                                      const I* __restrict__ ptr,
                                      const J* __restrict__ ind,
                                      const T* __restrict__ val,
                                      rocsparse_index_base base);

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
                                        rocsparse_index_base base);

    template <typename T, typename I = rocsparse_int>
    rocsparse_status write_sparse_coo(I m,
                                      I n,
                                      I nnz,
                                      const I* __restrict__ row_ind,
                                      const I* __restrict__ col_ind,
                                      const T* __restrict__ val,
                                      rocsparse_index_base base);

    template <typename T, typename I = rocsparse_int>
    rocsparse_status write_dense_vector(I size, const T* __restrict__ x, I incx);

    template <typename T, typename I = rocsparse_int>
    rocsparse_status
        write_dense_matrix(rocsparse_order order, I m, I n, const T* __restrict__ x, I ld);
};

#endif // HEADER
