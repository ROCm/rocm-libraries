/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_HOST_IO_HPP_
#define ROCALUTION_HOST_IO_HPP_

#include <cstdint>
#include <string>

namespace rocalution
{

    template <typename ValueType>
    bool read_matrix_mtx(int&        nrow,
                         int&        ncol,
                         int64_t&    nnz,
                         int**       row,
                         int**       col,
                         ValueType** val,
                         const char* filename);

    template <typename ValueType>
    bool write_matrix_mtx(int              nrow,
                          int              ncol,
                          int64_t          nnz,
                          const int*       row,
                          const int*       col,
                          const ValueType* val,
                          const char*      filename);

    template <typename ValueType, typename IndexType, typename PointerType>
    bool read_matrix_csr(int64_t&      nrow,
                         int64_t&      ncol,
                         int64_t&      nnz,
                         PointerType** ptr,
                         IndexType**   col,
                         ValueType**   val,
                         const char*   filename);

    template <typename ValueType, typename IndexType, typename PointerType>
    bool write_matrix_csr(int64_t            nrow,
                          int64_t            ncol,
                          int64_t            nnz,
                          const PointerType* ptr,
                          const IndexType*   col,
                          const ValueType*   val,
                          const char*        filename);

    template <typename ValueType, typename IndexType, typename PointerType>
    bool read_matrix_csr_rocsparseio(int64_t&      nrow,
                                     int64_t&      ncol,
                                     int64_t&      nnz,
                                     PointerType** ptr,
                                     IndexType**   col,
                                     ValueType**   val,
                                     const char*   filename);

    template <typename ValueType, typename IndexType, typename PointerType>
    bool write_matrix_csr_rocsparseio(int64_t            nrow,
                                      int64_t            ncol,
                                      int64_t            nnz,
                                      const PointerType* ptr,
                                      const IndexType*   col,
                                      const ValueType*   val,
                                      const char*        filename);

    template <typename ValueType, typename IndexType, typename PointerType>
    bool read_matrix_mcsr_rocsparseio(int64_t&      nrow,
                                      int64_t&      ncol,
                                      int64_t&      nnz,
                                      PointerType** ptr,
                                      IndexType**   col,
                                      ValueType**   val,
                                      const char*   filename);

    template <typename ValueType, typename IndexType, typename PointerType>
    bool write_matrix_mcsr_rocsparseio(int64_t            nrow,
                                       int64_t            ncol,
                                       int64_t            nnz,
                                       const PointerType* ptr,
                                       const IndexType*   col,
                                       const ValueType*   val,
                                       const char*        filename);

    template <typename ValueType, typename IndexType>
    bool read_matrix_coo_rocsparseio(int64_t&    nrow,
                                     int64_t&    ncol,
                                     int64_t&    nnz,
                                     IndexType** row,
                                     IndexType** col,
                                     ValueType** val,
                                     const char* filename);

    template <typename ValueType, typename IndexType>
    bool write_matrix_coo_rocsparseio(int64_t          nrow,
                                      int64_t          ncol,
                                      int64_t          nnz,
                                      const IndexType* row,
                                      const IndexType* col,
                                      const ValueType* val,
                                      const char*      filename);

    template <typename ValueType, typename IndexType>
    bool read_matrix_dia_rocsparseio(int64_t&    nrow,
                                     int64_t&    ncol,
                                     int64_t&    nnz,
                                     int64_t&    ndiag,
                                     IndexType** offset,
                                     ValueType** val,
                                     const char* filename);

    template <typename ValueType, typename IndexType>
    bool write_matrix_dia_rocsparseio(int64_t          nrow,
                                      int64_t          ncol,
                                      int64_t          ndiag,
                                      const IndexType* offset,
                                      const ValueType* val,
                                      const char*      filename);

    template <typename ValueType, typename IndexType>
    bool read_matrix_ell_rocsparseio(int64_t&    nrow,
                                     int64_t&    ncol,
                                     int64_t&    nnz,
                                     int64_t&    width,
                                     IndexType** col,
                                     ValueType** val,
                                     const char* filename);

    template <typename ValueType, typename IndexType>
    bool write_matrix_ell_rocsparseio(int64_t          nrow,
                                      int64_t          ncol,
                                      int64_t          width,
                                      const IndexType* col,
                                      const ValueType* val,
                                      const char*      filename);

    template <typename ValueType, typename IndexType>
    bool read_matrix_hyb_rocsparseio(int64_t&    nrow,
                                     int64_t&    ncol,
                                     int64_t&    nnz,
                                     int64_t&    coo_nnz,
                                     IndexType** coo_row,
                                     IndexType** coo_col,
                                     ValueType** coo_val,
                                     int64_t&    ell_nnz,
                                     int64_t&    ell_width,
                                     IndexType** ell_col,
                                     ValueType** ell_val,
                                     const char* filename);

    template <typename ValueType, typename IndexType>
    bool write_matrix_hyb_rocsparseio(int64_t          nrow,
                                      int64_t          ncol,
                                      int64_t          coo_nnz,
                                      const IndexType* coo_row,
                                      const IndexType* coo_col,
                                      const ValueType* coo_val,
                                      int64_t          ell_width,
                                      const IndexType* ell_col,
                                      const ValueType* ell_val,
                                      const char*      filename);

    template <typename ValueType>
    bool read_matrix_dense_rocsparseio(int64_t&    nrow,
                                       int64_t&    ncol,
                                       ValueType** val,
                                       const char* filename);

    template <typename ValueType>
    bool write_matrix_dense_rocsparseio(int64_t          nrow,
                                        int64_t          ncol,
                                        const ValueType* val,
                                        const char*      filename);

    template <typename ValueType, typename IndexType, typename PointerType>
    bool read_matrix_bcsr_rocsparseio(int64_t&      nrow,
                                      int64_t&      ncol,
                                      int64_t&      nnz,
                                      int64_t&      block_dim,
                                      PointerType** ptr,
                                      IndexType**   col,
                                      ValueType**   val,
                                      const char*   filename);

    template <typename ValueType, typename IndexType, typename PointerType>
    bool write_matrix_bcsr_rocsparseio(int64_t            nrowb,
                                       int64_t            ncolb,
                                       int64_t            nnzb,
                                       int64_t            block_dim,
                                       const PointerType* ptr,
                                       const IndexType*   col,
                                       const ValueType*   val,
                                       const char*        filename);
} // namespace rocalution

#endif // ROCALUTION_HOST_IO_HPP_
