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

#include "rocsparse_importer.hpp"

template <typename I,
          typename J,
          typename T,
          typename IMPORTER,
          template <typename...>
          class VECTOR1,
          template <typename...>
          class VECTOR2,
          template <typename...>
          class VECTOR3>
rocsparse_status rocsparse_import_sparse_gebsr(rocsparse_importer<IMPORTER>& importer,
                                               VECTOR1<I>&                   row_ptr,
                                               VECTOR2<J>&                   col_ind,
                                               VECTOR3<T>&                   val,
                                               rocsparse_direction&          dirb,
                                               J&                            Mb,
                                               J&                            Nb,
                                               I&                            nnzb,
                                               J&                            row_block_dim,
                                               J&                            col_block_dim,
                                               rocsparse_index_base          base)
{
    rocsparse_direction  import_dir;
    rocsparse_index_base import_base;
    rocsparse_status     status = importer.import_sparse_gebsx(
        &import_dir, &dirb, &Mb, &Nb, &nnzb, &row_block_dim, &col_block_dim, &import_base);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    if(import_dir != rocsparse_direction_row)
    {
        std::cerr << "expected gebsr matrix, not gebsc. " << std::endl;
        return rocsparse_status_invalid_value;
    }

    row_ptr.resize(Mb + 1);
    col_ind.resize(nnzb);
    val.resize(nnzb * row_block_dim * col_block_dim);

    status = importer.import_sparse_gebsx(row_ptr.data(), col_ind.data(), val.data());
    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocsparse_importer_switch_base(Mb + 1, row_ptr, import_base, base);
    if(status != rocsparse_status_success)
    {
        return status;
    }
    status = rocsparse_importer_switch_base(nnzb, col_ind, import_base, base);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    return rocsparse_status_success;
}

template <typename I,
          typename J,
          typename T,
          typename IMPORTER,
          template <typename...>
          class VECTOR1,
          template <typename...>
          class VECTOR2,
          template <typename...>
          class VECTOR3>
rocsparse_status rocsparse_import_sparse_csr(rocsparse_importer<IMPORTER>& importer,
                                             VECTOR1<I>&                   row_ptr,
                                             VECTOR2<J>&                   col_ind,
                                             VECTOR3<T>&                   val,
                                             J&                            M,
                                             J&                            N,
                                             I&                            nnz,
                                             rocsparse_index_base          base)
{
    rocsparse_direction  dir;
    rocsparse_index_base import_base;
    rocsparse_status     status = importer.import_sparse_csx(&dir, &M, &N, &nnz, &import_base);
    if(status != rocsparse_status_success)
    {
        return status;
    }
    if(dir != rocsparse_direction_row)
    {
        std::cerr << "expected csr matrix " << std::endl;
        return rocsparse_status_invalid_value;
    }

    row_ptr.resize(M + 1);
    col_ind.resize(nnz);
    val.resize(nnz);

    status = importer.import_sparse_csx(row_ptr.data(), col_ind.data(), val.data());
    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocsparse_importer_switch_base(M + 1, row_ptr, import_base, base);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocsparse_importer_switch_base(nnz, col_ind, import_base, base);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    return rocsparse_status_success;
}

template <typename I,
          typename T,
          typename IMPORTER,
          template <typename...>
          class VECTOR1,
          template <typename...>
          class VECTOR2,
          template <typename...>
          class VECTOR3>
rocsparse_status rocsparse_import_sparse_coo(rocsparse_importer<IMPORTER>& importer,
                                             VECTOR1<I>&                   row_ind,
                                             VECTOR2<I>&                   col_ind,
                                             VECTOR3<T>&                   val,
                                             I&                            M,
                                             I&                            N,
                                             int64_t&                      nnz,
                                             rocsparse_index_base          base)
{

    rocsparse_index_base import_base;
    rocsparse_status     status = importer.import_sparse_coo(&M, &N, &nnz, &import_base);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    row_ind.resize(nnz);
    col_ind.resize(nnz);
    val.resize(nnz);

    status = importer.import_sparse_coo(row_ind.data(), col_ind.data(), val.data());
    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocsparse_importer_switch_base(nnz, row_ind, import_base, base);
    if(status != rocsparse_status_success)
    {
        return status;
    }
    status = rocsparse_importer_switch_base(nnz, col_ind, import_base, base);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    return rocsparse_status_success;
}
