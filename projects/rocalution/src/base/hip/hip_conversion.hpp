/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_HIP_CONVERSION_HPP_
#define ROCALUTION_HIP_CONVERSION_HPP_

#include "../backend_manager.hpp"
#include "../matrix_formats.hpp"

#include <rocblas/rocblas.h>
#include <rocsparse/rocsparse.h>

namespace rocalution
{
    template <typename ValueType, typename IndexType, typename PointerType>
    bool csr_to_coo_hip(const Rocalution_Backend_Descriptor*                backend,
                        int64_t                                             nnz,
                        IndexType                                           nrow,
                        IndexType                                           ncol,
                        const MatrixCSR<ValueType, IndexType, PointerType>& src,
                        MatrixCOO<ValueType, IndexType>*                    dst);

    template <typename ValueType, typename IndexType, typename PointerType>
    bool coo_to_csr_hip(const Rocalution_Backend_Descriptor*          backend,
                        int64_t                                       nnz,
                        IndexType                                     nrow,
                        IndexType                                     ncol,
                        const MatrixCOO<ValueType, IndexType>&        src,
                        MatrixCSR<ValueType, IndexType, PointerType>* dst);

    template <typename ValueType, typename IndexType, typename PointerType>
    bool csr_to_bcsr_hip(const Rocalution_Backend_Descriptor*                backend,
                         int64_t                                             nnz,
                         IndexType                                           nrow,
                         IndexType                                           ncol,
                         const MatrixCSR<ValueType, IndexType, PointerType>& src,
                         const rocsparse_mat_descr                           src_descr,
                         MatrixBCSR<ValueType, IndexType>*                   dst,
                         const rocsparse_mat_descr                           dst_descr);

    template <typename ValueType, typename IndexType, typename PointerType>
    bool bcsr_to_csr_hip(const Rocalution_Backend_Descriptor*          backend,
                         int64_t                                       nnz,
                         IndexType                                     nrow,
                         IndexType                                     ncol,
                         const MatrixBCSR<ValueType, IndexType>&       src,
                         const rocsparse_mat_descr                     src_descr,
                         MatrixCSR<ValueType, IndexType, PointerType>* dst,
                         rocsparse_mat_descr                           dst_descr);

    template <typename ValueType, typename IndexType, typename PointerType>
    bool csr_to_ell_hip(const Rocalution_Backend_Descriptor*                backend,
                        int64_t                                             nnz,
                        IndexType                                           nrow,
                        IndexType                                           ncol,
                        const MatrixCSR<ValueType, IndexType, PointerType>& src,
                        const rocsparse_mat_descr                           src_descr,
                        MatrixELL<ValueType, IndexType>*                    dst,
                        const rocsparse_mat_descr                           dst_descr,
                        int64_t*                                            nnz_ell);

    template <typename ValueType, typename IndexType, typename PointerType>
    bool ell_to_csr_hip(const Rocalution_Backend_Descriptor*          backend,
                        int64_t                                       nnz,
                        IndexType                                     nrow,
                        IndexType                                     ncol,
                        const MatrixELL<ValueType, IndexType>&        src,
                        const rocsparse_mat_descr                     src_descr,
                        MatrixCSR<ValueType, IndexType, PointerType>* dst,
                        const rocsparse_mat_descr                     dst_descr,
                        int64_t*                                      nnz_csr);

    template <typename ValueType, typename IndexType, typename PointerType>
    bool csr_to_dia_hip(const Rocalution_Backend_Descriptor*                backend,
                        int64_t                                             nnz,
                        IndexType                                           nrow,
                        IndexType                                           ncol,
                        const MatrixCSR<ValueType, IndexType, PointerType>& src,
                        MatrixDIA<ValueType, IndexType>*                    dst,
                        int64_t*                                            nnz_dia,
                        IndexType*                                          num_diag);

    template <typename ValueType, typename IndexType, typename PointerType>
    bool csr_to_hyb_hip(const Rocalution_Backend_Descriptor*                backend,
                        int64_t                                             nnz,
                        IndexType                                           nrow,
                        IndexType                                           ncol,
                        const MatrixCSR<ValueType, IndexType, PointerType>& src,
                        MatrixHYB<ValueType, IndexType>*                    dst,
                        int64_t*                                            nnz_hyb,
                        int64_t*                                            nnz_ell,
                        int64_t*                                            nnz_coo);

    template <typename ValueType, typename IndexType, typename PointerType>
    bool csr_to_dense_hip(const Rocalution_Backend_Descriptor*                backend,
                          IndexType                                           nrow,
                          IndexType                                           ncol,
                          const MatrixCSR<ValueType, IndexType, PointerType>& src,
                          const rocsparse_mat_descr                           src_descr,
                          MatrixDENSE<ValueType>*                             dst);

    template <typename ValueType, typename IndexType, typename PointerType>
    bool dense_to_csr_hip(const Rocalution_Backend_Descriptor*          backend,
                          IndexType                                     nrow,
                          IndexType                                     ncol,
                          const MatrixDENSE<ValueType>&                 src,
                          MatrixCSR<ValueType, IndexType, PointerType>* dst,
                          const rocsparse_mat_descr                     dst_descr,
                          int64_t*                                      nnz_csr);

} // namespace rocalution

#endif // ROCALUTION_HIP_CONVERSION_HPP_
