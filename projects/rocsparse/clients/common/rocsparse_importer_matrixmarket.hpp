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
#ifndef ROCSPARSE_IMPORTER_MATRIXMARKET_HPP
#define ROCSPARSE_IMPORTER_MATRIXMARKET_HPP
#include "rocsparse_importer.hpp"

class rocsparse_importer_matrixmarket : public rocsparse_importer<rocsparse_importer_matrixmarket>
{
protected:
    std::string m_filename;

public:
    rocsparse_importer_matrixmarket(const std::string& filename_);

private:
    FILE*  f;
    size_t m_nnz;
    char   m_data[16];
    int    m_symm;

public:
    template <typename I = rocsparse_int, typename J = rocsparse_int>
    rocsparse_status
        import_sparse_csx(rocsparse_direction* dir, J* m, J* n, I* nnz, rocsparse_index_base* base);
    template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
    rocsparse_status import_sparse_csx(I* ptr, J* ind, T* val);

    template <typename I = rocsparse_int, typename J = rocsparse_int>
    rocsparse_status import_sparse_gebsx(rocsparse_direction*  dir,
                                         rocsparse_direction*  dirb,
                                         J*                    mb,
                                         J*                    nb,
                                         I*                    nnzb,
                                         J*                    block_dim_row,
                                         J*                    block_dim_column,
                                         rocsparse_index_base* base);
    template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
    rocsparse_status import_sparse_gebsx(I* ptr, J* ind, T* val);

    template <typename I = rocsparse_int>
    rocsparse_status import_sparse_coo(I* m, I* n, int64_t* nnz, rocsparse_index_base* base);
    template <typename T, typename I = rocsparse_int>
    rocsparse_status import_sparse_coo(I* row_ind, I* col_ind, T* val);
};

#endif
