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

#include "host_conversion.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../matrix_formats.hpp"
#include "../matrix_formats_ind.hpp"
#include "rocalution/utils/types.hpp"

#include <complex>
#include <cstdlib>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_set_num_threads(num) ;
#endif

namespace rocalution
{

    template <typename ValueType, typename IndexType, typename PointerType>
    bool csr_to_dense(int                                                 omp_threads,
                      int64_t                                             nnz,
                      IndexType                                           nrow,
                      IndexType                                           ncol,
                      const MatrixCSR<ValueType, IndexType, PointerType>& src,
                      MatrixDENSE<ValueType>*                             dst)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        omp_set_num_threads(omp_threads);

        allocate_host(nrow * ncol, &dst->val);
        set_to_zero_host(nrow * ncol, dst->val);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(IndexType i = 0; i < nrow; ++i)
        {
            for(PointerType j = src.row_offset[i]; j < src.row_offset[i + 1]; ++j)
            {
                dst->val[DENSE_IND(i, src.col[j], nrow, ncol)] = src.val[j];
            }
        }

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool dense_to_csr(int                                           omp_threads,
                      IndexType                                     nrow,
                      IndexType                                     ncol,
                      const MatrixDENSE<ValueType>&                 src,
                      MatrixCSR<ValueType, IndexType, PointerType>* dst,
                      int64_t*                                      nnz)
    {
        assert(nrow > 0);
        assert(ncol > 0);

        omp_set_num_threads(omp_threads);

        allocate_host(nrow + 1, &dst->row_offset);
        set_to_zero_host(nrow + 1, dst->row_offset);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(IndexType i = 0; i < nrow; ++i)
        {
            for(IndexType j = 0; j < ncol; ++j)
            {
                if(src.val[DENSE_IND(i, j, nrow, ncol)] != static_cast<ValueType>(0))
                {
                    dst->row_offset[i] += 1;
                }
            }
        }

        *nnz = 0;
        for(IndexType i = 0; i < nrow; ++i)
        {
            PointerType tmp    = dst->row_offset[i];
            dst->row_offset[i] = *nnz;
            *nnz += tmp;
        }

        assert(*nnz <= std::numeric_limits<int>::max());

        dst->row_offset[nrow] = *nnz;

        allocate_host(*nnz, &dst->col);
        allocate_host(*nnz, &dst->val);

        set_to_zero_host(*nnz, dst->col);
        set_to_zero_host(*nnz, dst->val);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(IndexType i = 0; i < nrow; ++i)
        {
            PointerType ind = dst->row_offset[i];

            for(IndexType j = 0; j < ncol; ++j)
            {
                if(src.val[DENSE_IND(i, j, nrow, ncol)] != static_cast<ValueType>(0))
                {
                    dst->val[ind] = src.val[DENSE_IND(i, j, nrow, ncol)];
                    dst->col[ind] = j;
                    ++ind;
                }
            }
        }

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool csr_to_mcsr(int                                                 omp_threads,
                     int64_t                                             nnz,
                     IndexType                                           nrow,
                     IndexType                                           ncol,
                     const MatrixCSR<ValueType, IndexType, PointerType>& src,
                     MatrixMCSR<ValueType, IndexType>*                   dst)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        // No support for non-squared matrices
        if(nrow != ncol)
        {
            return false;
        }

        omp_set_num_threads(omp_threads);

        // Pre-analysing step to check zero diagonal entries
        IndexType diag_entries = 0;

        for(IndexType i = 0; i < nrow; ++i)
        {
            for(PointerType j = src.row_offset[i]; j < src.row_offset[i + 1]; ++j)
            {
                if(i == src.col[j])
                {
                    ++diag_entries;
                }
            }
        }

        IndexType zero_diag_entries = nrow - diag_entries;

        // MCSR does not support zero diagonal entries (yet)
        if(zero_diag_entries > 0)
        {
            return false;
        }

        allocate_host(nrow + 1, &dst->row_offset);
        allocate_host(nnz, &dst->col);
        allocate_host(nnz, &dst->val);

        set_to_zero_host(nrow + 1, dst->row_offset);
        set_to_zero_host(nnz, dst->col);
        set_to_zero_host(nnz, dst->val);

        assert(nnz <= std::numeric_limits<int>::max());

        for(IndexType ai = 0; ai < nrow + 1; ++ai)
        {
            dst->row_offset[ai] = static_cast<IndexType>(nrow + src.row_offset[ai] - ai);
        }

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(IndexType ai = 0; ai < nrow; ++ai)
        {
            IndexType correction = ai;
            for(PointerType aj = src.row_offset[ai]; aj < src.row_offset[ai + 1]; ++aj)
            {
                if(ai != src.col[aj])
                {
                    PointerType ind = nrow + aj - correction;

                    // non-diag
                    dst->col[ind] = src.col[aj];
                    dst->val[ind] = src.val[aj];
                }
                else
                {
                    // diag
                    dst->val[ai] = src.val[aj];
                    ++correction;
                }
            }
        }

        if(dst->row_offset[nrow] != src.row_offset[nrow])
        {
            return false;
        }

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool mcsr_to_csr(int                                           omp_threads,
                     int64_t                                       nnz,
                     IndexType                                     nrow,
                     IndexType                                     ncol,
                     const MatrixMCSR<ValueType, IndexType>&       src,
                     MatrixCSR<ValueType, IndexType, PointerType>* dst)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        // Only support square matrices
        if(nrow != ncol)
        {
            return false;
        }

        omp_set_num_threads(omp_threads);

        allocate_host(nrow + 1, &dst->row_offset);
        allocate_host(nnz, &dst->col);
        allocate_host(nnz, &dst->val);

        set_to_zero_host(nrow + 1, dst->row_offset);
        set_to_zero_host(nnz, dst->col);
        set_to_zero_host(nnz, dst->val);

        for(IndexType ai = 0; ai < nrow + 1; ++ai)
        {
            dst->row_offset[ai] = src.row_offset[ai] - nrow + ai;
        }

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(IndexType ai = 0; ai < nrow; ++ai)
        {
            for(PointerType aj = src.row_offset[ai]; aj < src.row_offset[ai + 1]; ++aj)
            {
                PointerType ind = aj - nrow + ai;

                // non-diag
                dst->col[ind] = src.col[aj];
                dst->val[ind] = src.val[aj];
            }

            PointerType diag_ind = src.row_offset[ai + 1] - nrow + ai;

            // diag
            dst->val[diag_ind] = src.val[ai];
            dst->col[diag_ind] = ai;
        }

        if(dst->row_offset[nrow] != src.row_offset[nrow])
        {
            return false;
        }

        // Sorting the col (per row)
        // Bubble sort algorithm

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(IndexType i = 0; i < nrow; ++i)
        {
            for(PointerType j = dst->row_offset[i]; j < dst->row_offset[i + 1]; ++j)
            {
                for(PointerType jj = dst->row_offset[i]; jj < dst->row_offset[i + 1] - 1; ++jj)
                {
                    if(dst->col[jj] > dst->col[jj + 1])
                    {
                        // swap elements
                        IndexType ind = dst->col[jj];
                        ValueType val = dst->val[jj];

                        dst->col[jj] = dst->col[jj + 1];
                        dst->val[jj] = dst->val[jj + 1];

                        dst->col[jj + 1] = ind;
                        dst->val[jj + 1] = val;
                    }
                }
            }
        }

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool csr_to_bcsr(int                                                 omp_threads,
                     int64_t                                             nnz,
                     IndexType                                           nrow,
                     IndexType                                           ncol,
                     const MatrixCSR<ValueType, IndexType, PointerType>& src,
                     MatrixBCSR<ValueType, IndexType>*                   dst)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        IndexType blockdim = dst->blockdim;

        assert(blockdim > 1);

        // Matrix dimensions must be a multiple of blockdim
        if((nrow % blockdim) != 0 || (ncol % blockdim) != 0)
        {
            return false;
        }

        // BCSR row blocks
        IndexType mb = (nrow + blockdim - 1) / blockdim;
        IndexType nb = (ncol + blockdim - 1) / blockdim;

        // Allocate memory for row pointers
        allocate_host(mb + 1, &dst->row_offset);

        // Determine BCSR nnz
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            std::vector<bool>      blockcol(nb, false);
            std::vector<IndexType> erase(nb);

#ifdef _OPENMP
#pragma omp for
#endif
            // Loop over blocked rows
            for(IndexType bcsr_i = 0; bcsr_i < mb; ++bcsr_i)
            {
                // CSR row index
                IndexType csr_i = bcsr_i * blockdim;

                // Number of blocks required in the blocked row
                IndexType nblocks = 0;

                // Loop over rows inside the current block
                for(IndexType i = 0; i < blockdim; ++i)
                {
                    // Do not exceed CSR rows
                    if(i >= nrow - csr_i)
                    {
                        break;
                    }

                    PointerType csr_row_begin = src.row_offset[csr_i + i];
                    PointerType csr_row_end   = src.row_offset[csr_i + i + 1];

                    // Loop over CSR columns for each of the rows in the block
                    for(PointerType csr_j = csr_row_begin; csr_j < csr_row_end; ++csr_j)
                    {
                        // Block column index
                        IndexType bcsr_j = src.col[csr_j] / blockdim;

                        // Increment block counter for current blocked row if this column
                        // creates a new block
                        if(blockcol[bcsr_j] == false)
                        {
                            blockcol[bcsr_j] = true;
                            erase[nblocks++] = bcsr_j;
                        }
                    }
                }

                // Store number of blocks of the current blocked row
                dst->row_offset[bcsr_i + 1] = nblocks;

                // Clear block buffer
                for(IndexType i = 0; i < nblocks; ++i)
                {
                    blockcol[erase[i]] = false;
                }
            }
        }

        // Exclusive sum to obtain BCSR row pointers
        dst->row_offset[0] = 0;
        for(IndexType i = 0; i < mb; ++i)
        {
            dst->row_offset[i + 1] += dst->row_offset[i];
        }

        // Extract BCSR nnz
        int64_t nnzb = dst->row_offset[mb];

        // Allocate BCSR structure
        allocate_host(nnzb, &dst->col);
        allocate_host(nnzb * blockdim * blockdim, &dst->val);

        // Initialize values with zero
        set_to_zero_host(nnzb * blockdim * blockdim, dst->val);

        assert(nnz <= std::numeric_limits<int>::max());

        // Fill BCSR structure
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            std::vector<IndexType> blockcol(nb, -1);

#ifdef _OPENMP
#pragma omp for
#endif
            // Loop over blocked rows
            for(IndexType bcsr_i = 0; bcsr_i < mb; ++bcsr_i)
            {
                // CSR row index
                IndexType csr_i = bcsr_i * blockdim;

                // Offset into BCSR row
                PointerType bcsr_row_begin = dst->row_offset[bcsr_i];
                PointerType bcsr_row_end   = dst->row_offset[bcsr_i + 1];
                PointerType bcsr_idx       = bcsr_row_begin;

                // Loop over rows inside the current block
                for(IndexType i = 0; i < blockdim; ++i)
                {
                    // Do not exceed CSR rows
                    if(i >= nrow - csr_i)
                    {
                        break;
                    }

                    PointerType csr_row_begin = src.row_offset[csr_i + i];
                    PointerType csr_row_end   = src.row_offset[csr_i + i + 1];

                    // Loop over CSR columns for each of the rows in the block
                    for(PointerType csr_j = csr_row_begin; csr_j < csr_row_end; ++csr_j)
                    {
                        // CSR column index
                        IndexType csr_col = src.col[csr_j];

                        // Block column index
                        IndexType bcsr_col = csr_col / blockdim;

                        // Intra block column
                        IndexType j = csr_col % blockdim;

                        // Fill the block's column index
                        if(blockcol[bcsr_col] == -1)
                        {
                            // Keep block value offset for filling
                            blockcol[bcsr_col] = static_cast<IndexType>(bcsr_idx);

                            // Write BCSR column index
                            dst->col[bcsr_idx++] = bcsr_col;
                        }

                        // Write BCSR value
                        dst->val[BCSR_IND(blockcol[bcsr_col], i, j, blockdim)] = src.val[csr_j];
                    }
                }

                // Clear block buffer
                for(PointerType i = bcsr_row_begin; i < bcsr_row_end; ++i)
                {
                    blockcol[dst->col[i]] = -1;
                }
            }

            // Sort
#ifdef _OPENMP
#pragma omp for
#endif
            for(IndexType i = 0; i < mb; ++i)
            {
                PointerType row_begin = dst->row_offset[i];
                PointerType row_end   = dst->row_offset[i + 1];

                for(PointerType j = row_begin; j < row_end; ++j)
                {
                    for(PointerType k = row_begin; k < row_end - 1; ++k)
                    {
                        if(dst->col[k] > dst->col[k + 1])
                        {
                            // Swap values
                            for(IndexType b = 0; b < blockdim * blockdim; ++b)
                            {
                                std::swap(dst->val[blockdim * blockdim * k + b],
                                          dst->val[blockdim * blockdim * (k + 1) + b]);
                            }

                            // Swap column index
                            std::swap(dst->col[k], dst->col[k + 1]);
                        }
                    }
                }
            }
        }

        dst->nrowb = mb;
        dst->ncolb = nb;
        dst->nnzb  = nnzb;

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool bcsr_to_csr(int                                           omp_threads,
                     int64_t                                       nnz,
                     IndexType                                     nrow,
                     IndexType                                     ncol,
                     const MatrixBCSR<ValueType, IndexType>&       src,
                     MatrixCSR<ValueType, IndexType, PointerType>* dst)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        allocate_host(nrow + 1, &dst->row_offset);
        allocate_host(nnz, &dst->col);
        allocate_host(nnz, &dst->val);

        dst->row_offset[0] = 0;

        PointerType idx = 0;
        for(IndexType i = 0; i < src.nrowb; ++i)
        {
            for(IndexType r = 0; r < src.blockdim; ++r)
            {
                IndexType row = i * src.blockdim + r;

                for(PointerType k = src.row_offset[i]; k < src.row_offset[i + 1]; ++k)
                {
                    for(IndexType c = 0; c < src.blockdim; ++c)
                    {
                        dst->col[idx] = src.blockdim * src.col[k] + c;
                        dst->val[idx] = src.val[BCSR_IND(k, r, c, src.blockdim)];

                        ++idx;
                    }
                }

                dst->row_offset[row + 1]
                    = dst->row_offset[row]
                      + (src.row_offset[i + 1] - src.row_offset[i]) * src.blockdim;
            }
        }

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool csr_to_coo(int                                                 omp_threads,
                    int64_t                                             nnz,
                    IndexType                                           nrow,
                    IndexType                                           ncol,
                    const MatrixCSR<ValueType, IndexType, PointerType>& src,
                    MatrixCOO<ValueType, IndexType>*                    dst)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        omp_set_num_threads(omp_threads);

        allocate_host(nnz, &dst->row);
        allocate_host(nnz, &dst->col);
        allocate_host(nnz, &dst->val);

        set_to_zero_host(nnz, dst->row);
        set_to_zero_host(nnz, dst->col);
        set_to_zero_host(nnz, dst->val);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(IndexType i = 0; i < nrow; ++i)
        {
            for(PointerType j = src.row_offset[i]; j < src.row_offset[i + 1]; ++j)
            {
                dst->row[j] = i;
            }
        }

        copy_h2h(nnz, src.col, dst->col);
        copy_h2h(nnz, src.val, dst->val);

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool csr_to_ell(int                                                 omp_threads,
                    int64_t                                             nnz,
                    IndexType                                           nrow,
                    IndexType                                           ncol,
                    const MatrixCSR<ValueType, IndexType, PointerType>& src,
                    MatrixELL<ValueType, IndexType>*                    dst,
                    int64_t*                                            nnz_ell)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        omp_set_num_threads(omp_threads);

        dst->max_row = 0;
        for(IndexType i = 0; i < nrow; ++i)
        {
            IndexType max_row = static_cast<IndexType>(src.row_offset[i + 1] - src.row_offset[i]);

            if(max_row > dst->max_row)
            {
                dst->max_row = max_row;
            }
        }

        *nnz_ell = dst->max_row * nrow;

        // Limit ELL size to 5 times CSR nnz
        if(dst->max_row > 5 * (nnz / nrow))
        {
            return false;
        }

        allocate_host(*nnz_ell, &dst->val);
        allocate_host(*nnz_ell, &dst->col);

        set_to_zero_host(*nnz_ell, dst->val);
        set_to_zero_host(*nnz_ell, dst->col);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(IndexType i = 0; i < nrow; ++i)
        {
            IndexType n = 0;

            for(PointerType j = src.row_offset[i]; j < src.row_offset[i + 1]; ++j)
            {
                PointerType ind = ELL_IND(i, n, nrow, dst->max_row);

                dst->val[ind] = src.val[j];
                dst->col[ind] = src.col[j];
                ++n;
            }

            for(PointerType j = src.row_offset[i + 1] - src.row_offset[i]; j < dst->max_row; ++j)
            {
                PointerType ind = ELL_IND(i, n, nrow, dst->max_row);

                dst->val[ind] = static_cast<ValueType>(0);
                dst->col[ind] = static_cast<IndexType>(-1);
                ++n;
            }
        }

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool ell_to_csr(int                                           omp_threads,
                    int64_t                                       nnz,
                    IndexType                                     nrow,
                    IndexType                                     ncol,
                    const MatrixELL<ValueType, IndexType>&        src,
                    MatrixCSR<ValueType, IndexType, PointerType>* dst,
                    int64_t*                                      nnz_csr)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        omp_set_num_threads(omp_threads);

        allocate_host(nrow + 1, &dst->row_offset);
        set_to_zero_host(nrow + 1, dst->row_offset);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(IndexType ai = 0; ai < nrow; ++ai)
        {
            for(IndexType n = 0; n < src.max_row; ++n)
            {
                PointerType aj = ELL_IND(ai, n, nrow, src.max_row);

                if((src.col[aj] >= 0) && (src.col[aj] < ncol))
                {
                    ++dst->row_offset[ai];
                }
            }
        }

        *nnz_csr = 0;
        for(IndexType i = 0; i < nrow; ++i)
        {
            PointerType tmp    = dst->row_offset[i];
            dst->row_offset[i] = *nnz_csr;
            *nnz_csr += tmp;
        }

        assert(*nnz_csr <= std::numeric_limits<int>::max());

        dst->row_offset[nrow] = *nnz_csr;

        allocate_host(*nnz_csr, &dst->col);
        allocate_host(*nnz_csr, &dst->val);

        set_to_zero_host(*nnz_csr, dst->col);
        set_to_zero_host(*nnz_csr, dst->val);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(IndexType ai = 0; ai < nrow; ++ai)
        {
            PointerType ind = dst->row_offset[ai];

            for(IndexType n = 0; n < src.max_row; ++n)
            {
                PointerType aj = ELL_IND(ai, n, nrow, src.max_row);

                if((src.col[aj] >= 0) && (src.col[aj] < ncol))
                {
                    dst->col[ind] = src.col[aj];
                    dst->val[ind] = src.val[aj];
                    ++ind;
                }
            }
        }

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool hyb_to_csr(int                                           omp_threads,
                    int64_t                                       nnz,
                    IndexType                                     nrow,
                    IndexType                                     ncol,
                    int64_t                                       nnz_ell,
                    int64_t                                       nnz_coo,
                    const MatrixHYB<ValueType, IndexType>&        src,
                    MatrixCSR<ValueType, IndexType, PointerType>* dst,
                    int64_t*                                      nnz_csr)
    {
        assert(nnz > 0);
        assert(nnz == nnz_ell + nnz_coo);
        assert(nrow > 0);
        assert(ncol > 0);

        omp_set_num_threads(omp_threads);

        allocate_host(nrow + 1, &dst->row_offset);
        set_to_zero_host(nrow + 1, dst->row_offset);

        IndexType start;
        start = 0;

        // TODO
        //#ifdef _OPENMP
        // #pragma omp parallel for private(start)
        //#endif
        for(IndexType ai = 0; ai < nrow; ++ai)
        {
            // ELL
            for(IndexType n = 0; n < src.ELL.max_row; ++n)
            {
                PointerType aj = ELL_IND(ai, n, nrow, src.ELL.max_row);

                if((src.ELL.col[aj] >= 0) && (src.ELL.col[aj] < ncol))
                {
                    dst->row_offset[ai] += 1;
                }
            }

            // COO
            for(int64_t i = start; i < nnz_coo; ++i)
            {
                if(src.COO.row[i] == ai)
                {
                    dst->row_offset[ai] += 1;
                    ++start;
                }

                if(src.COO.row[i] > ai)
                {
                    break;
                }
            }
        }

        *nnz_csr = 0;
        for(IndexType i = 0; i < nrow; ++i)
        {
            PointerType tmp    = dst->row_offset[i];
            dst->row_offset[i] = *nnz_csr;
            *nnz_csr += tmp;
        }

        assert(*nnz_csr <= std::numeric_limits<int>::max());

        dst->row_offset[nrow] = *nnz_csr;

        allocate_host(*nnz_csr, &dst->col);
        allocate_host(*nnz_csr, &dst->val);

        set_to_zero_host(*nnz_csr, dst->col);
        set_to_zero_host(*nnz_csr, dst->val);

        start = 0;

        // TODO
        //#ifdef _OPENMP
        //#pragma omp parallel for private(start)
        //#endif
        for(IndexType ai = 0; ai < nrow; ++ai)
        {
            PointerType ind = dst->row_offset[ai];

            // ELL
            for(IndexType n = 0; n < src.ELL.max_row; ++n)
            {
                PointerType aj = ELL_IND(ai, n, nrow, src.ELL.max_row);

                if((src.ELL.col[aj] >= 0) && (src.ELL.col[aj] < ncol))
                {
                    dst->col[ind] = src.ELL.col[aj];
                    dst->val[ind] = src.ELL.val[aj];
                    ++ind;
                }
            }

            // COO
            for(int64_t i = start; i < nnz_coo; ++i)
            {
                if(src.COO.row[i] == ai)
                {
                    dst->col[ind] = src.COO.col[i];
                    dst->val[ind] = src.COO.val[i];
                    ++ind;
                    ++start;
                }

                if(src.COO.row[i] > ai)
                {
                    break;
                }
            }
        }

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool coo_to_csr(int                                           omp_threads,
                    int64_t                                       nnz,
                    IndexType                                     nrow,
                    IndexType                                     ncol,
                    const MatrixCOO<ValueType, IndexType>&        src,
                    MatrixCSR<ValueType, IndexType, PointerType>* dst)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        omp_set_num_threads(omp_threads);

        allocate_host(nrow + 1, &dst->row_offset);
        allocate_host(nnz, &dst->col);
        allocate_host(nnz, &dst->val);

        // COO has to be sorted by rows
        for(int64_t i = 1; i < nnz; ++i)
        {
            assert(src.row[i] >= src.row[i - 1]);
        }

        // Initialize row offset with zeros
        set_to_zero_host(nrow + 1, dst->row_offset);

        // Compute nnz entries per row of CSR
        for(int64_t i = 0; i < nnz; ++i)
        {
            ++dst->row_offset[src.row[i] + 1];
        }

        // Do exclusive scan to obtain row ptrs
        for(IndexType i = 0; i < nrow; ++i)
        {
            dst->row_offset[i + 1] += dst->row_offset[i];
        }

        assert(dst->row_offset[nrow] == nnz);

        copy_h2h(nnz, src.col, dst->col);
        copy_h2h(nnz, src.val, dst->val);

// Sorting the col (per row)
// Bubble sort algorithm
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(IndexType i = 0; i < nrow; ++i)
        {
            for(PointerType j = dst->row_offset[i]; j < dst->row_offset[i + 1]; ++j)
            {
                for(PointerType jj = dst->row_offset[i]; jj < dst->row_offset[i + 1] - 1; ++jj)
                {
                    if(dst->col[jj] > dst->col[jj + 1])
                    {
                        // swap elements
                        IndexType ind = dst->col[jj];
                        ValueType val = dst->val[jj];

                        dst->col[jj] = dst->col[jj + 1];
                        dst->val[jj] = dst->val[jj + 1];

                        dst->col[jj + 1] = ind;
                        dst->val[jj + 1] = val;
                    }
                }
            }
        }

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool csr_to_dia(int                                                 omp_threads,
                    int64_t                                             nnz,
                    IndexType                                           nrow,
                    IndexType                                           ncol,
                    const MatrixCSR<ValueType, IndexType, PointerType>& src,
                    MatrixDIA<ValueType, IndexType>*                    dst,
                    int64_t*                                            nnz_dia)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        omp_set_num_threads(omp_threads);

        // Determine number of populated diagonals
        dst->num_diag = 0;

        std::vector<PointerType> diag_idx(nrow + ncol, 0);

        // Loop over rows and increment ndiag counter if diag offset has not been visited yet
        for(IndexType i = 0; i < nrow; ++i)
        {
            for(PointerType j = src.row_offset[i]; j < src.row_offset[i + 1]; ++j)
            {
                // Diagonal offset the current entry belongs to
                PointerType offset = src.col[j] - i + nrow;

                if(!diag_idx[offset])
                {
                    diag_idx[offset] = 1;
                    ++dst->num_diag;
                }
            }
        }

        IndexType size = std::min(nrow, ncol);
        *nnz_dia       = size * dst->num_diag;

        // Conversion fails if DIA nnz exceeds 5 times CSR nnz
        if(dst->num_diag > 5 * (nnz / size))
        {
            return false;
        }

        // Allocate DIA matrix
        allocate_host(dst->num_diag, &dst->offset);
        allocate_host(*nnz_dia, &dst->val);

        set_to_zero_host(*nnz_dia, dst->val);

        assert(nrow * ncol <= std::numeric_limits<int>::max());

        for(PointerType i = 0, d = 0; i < nrow + ncol; ++i)
        {
            // Fill DIA offset, if i-th diagonal is populated
            if(diag_idx[i])
            {
                // Store offset index for reverse index access
                diag_idx[i] = d;
                // Store diagonals offset, where the diagonal is offset 0
                // Left from diagonal offsets are decreasing
                // Right from diagonal offsets are increasing
                dst->offset[d++] = static_cast<IndexType>(i - nrow);
            }
        }

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(IndexType i = 0; i < nrow; ++i)
        {
            for(PointerType j = src.row_offset[i]; j < src.row_offset[i + 1]; ++j)
            {
                // Diagonal offset the current entry belongs to
                PointerType offset                                          = src.col[j] - i + nrow;
                dst->val[DIA_IND(i, diag_idx[offset], nrow, dst->num_diag)] = src.val[j];
            }
        }

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool dia_to_csr(int                                           omp_threads,
                    int64_t                                       nnz,
                    IndexType                                     nrow,
                    IndexType                                     ncol,
                    const MatrixDIA<ValueType, IndexType>&        src,
                    MatrixCSR<ValueType, IndexType, PointerType>* dst,
                    int64_t*                                      nnz_csr)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        omp_set_num_threads(omp_threads);

        // Allocate CSR row pointer array
        allocate_host(nrow + 1, &dst->row_offset);

        // Extract CSR row offsets
        dst->row_offset[0] = 0;
        for(IndexType i = 0; i < nrow; ++i)
        {
            // Initialize row offset of i-th row
            dst->row_offset[i + 1] = dst->row_offset[i];

            for(IndexType n = 0; n < src.num_diag; ++n)
            {
                IndexType j = i + src.offset[n];

                if(j >= 0 && j < ncol)
                {
                    // Exclude padded zeros
                    if(src.val[DIA_IND(i, n, nrow, src.num_diag)] != static_cast<ValueType>(0))
                    {
                        ++dst->row_offset[i + 1];
                    }
                }
            }
        }

        // CSR nnz
        *nnz_csr = dst->row_offset[nrow];

        // Allocate CSR matrix
        allocate_host(*nnz_csr, &dst->col);
        allocate_host(*nnz_csr, &dst->val);

// Fill CSR col and val arrays
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(IndexType i = 0; i < nrow; ++i)
        {
            PointerType idx = dst->row_offset[i];

            for(IndexType n = 0; n < src.num_diag; ++n)
            {
                IndexType j = i + src.offset[n];

                if(j >= 0 && j < ncol)
                {
                    ValueType val = src.val[DIA_IND(i, n, nrow, src.num_diag)];

                    if(val != static_cast<ValueType>(0))
                    {
                        dst->col[idx] = j;
                        dst->val[idx] = val;
                        ++idx;
                    }
                }
            }
        }

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool csr_to_hyb(int                                                 omp_threads,
                    int64_t                                             nnz,
                    IndexType                                           nrow,
                    IndexType                                           ncol,
                    const MatrixCSR<ValueType, IndexType, PointerType>& src,
                    MatrixHYB<ValueType, IndexType>*                    dst,
                    int64_t*                                            nnz_hyb,
                    int64_t*                                            nnz_ell,
                    int64_t*                                            nnz_coo)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        omp_set_num_threads(omp_threads);

        // Determine ELL width by average nnz per row
        if(dst->ELL.max_row == 0)
        {
            int64_t max_row = (nnz - 1) / nrow + 1;

            assert(max_row <= std::numeric_limits<int>::max());

            dst->ELL.max_row = static_cast<IndexType>(max_row);
        }

        // ELL nnz is ELL width times nrow
        *nnz_ell = dst->ELL.max_row * nrow;
        *nnz_coo = 0;

        // Array to hold COO part nnz per row
        PointerType* coo_row_ptr = NULL;
        allocate_host(nrow + 1, &coo_row_ptr);

        // If there is no ELL part, its easy...
        if(*nnz_ell == 0)
        {
            *nnz_coo = nnz;
            // copy rowoffset to coorownnz
        }
        else
        {
// Compute COO nnz per row and COO nnz
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(IndexType i = 0; i < nrow; ++i)
            {
                PointerType row_nnz = src.row_offset[i + 1] - src.row_offset[i] - dst->ELL.max_row;
                coo_row_ptr[i + 1]  = (row_nnz > 0) ? row_nnz : 0;
            }

            // Exclusive scan
            coo_row_ptr[0] = 0;
            for(IndexType i = 0; i < nrow; ++i)
            {
                coo_row_ptr[i + 1] += coo_row_ptr[i];
            }

            *nnz_coo = coo_row_ptr[nrow];
        }

        *nnz_hyb = *nnz_coo + *nnz_ell;

        if(*nnz_hyb <= 0)
        {
            return false;
        }

        // ELL
        if(*nnz_ell > 0)
        {
            allocate_host(*nnz_ell, &dst->ELL.val);
            allocate_host(*nnz_ell, &dst->ELL.col);
        }

        // COO
        if(*nnz_coo > 0)
        {
            allocate_host(*nnz_coo, &dst->COO.row);
            allocate_host(*nnz_coo, &dst->COO.col);
            allocate_host(*nnz_coo, &dst->COO.val);
        }

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(IndexType i = 0; i < nrow; ++i)
        {
            PointerType p         = 0;
            PointerType row_begin = src.row_offset[i];
            PointerType row_end   = src.row_offset[i + 1];
            PointerType coo_idx   = dst->COO.row ? coo_row_ptr[i] : 0;

            // Fill HYB matrix
            for(PointerType j = row_begin; j < row_end; ++j)
            {
                if(p < dst->ELL.max_row)
                {
                    // Fill ELL part
                    PointerType idx   = ELL_IND(i, p++, nrow, dst->ELL.max_row);
                    dst->ELL.col[idx] = src.col[j];
                    dst->ELL.val[idx] = src.val[j];
                }
                else
                {
                    dst->COO.row[coo_idx] = i;
                    dst->COO.col[coo_idx] = src.col[j];
                    dst->COO.val[coo_idx] = src.val[j];
                    ++coo_idx;
                }
            }

            // Pad remaining ELL structure
            for(IndexType j = static_cast<IndexType>(row_end - row_begin); j < dst->ELL.max_row;
                ++j)
            {
                PointerType idx   = ELL_IND(i, p++, nrow, dst->ELL.max_row);
                dst->ELL.col[idx] = -1;
                dst->ELL.val[idx] = static_cast<ValueType>(0);
            }
        }

        free_host(&coo_row_ptr);

        return true;
    }

    template bool csr_to_coo(int                                    omp_threads,
                             int64_t                                nnz,
                             int                                    nrow,
                             int                                    ncol,
                             const MatrixCSR<double, int, PtrType>& src,
                             MatrixCOO<double, int>*                dst);

    template bool csr_to_coo(int                                   omp_threads,
                             int64_t                               nnz,
                             int                                   nrow,
                             int                                   ncol,
                             const MatrixCSR<float, int, PtrType>& src,
                             MatrixCOO<float, int>*                dst);

#ifdef SUPPORT_COMPLEX
    template bool csr_to_coo(int                                                  omp_threads,
                             int64_t                                              nnz,
                             int                                                  nrow,
                             int                                                  ncol,
                             const MatrixCSR<std::complex<double>, int, PtrType>& src,
                             MatrixCOO<std::complex<double>, int>*                dst);

    template bool csr_to_coo(int                                                 omp_threads,
                             int64_t                                             nnz,
                             int                                                 nrow,
                             int                                                 ncol,
                             const MatrixCSR<std::complex<float>, int, PtrType>& src,
                             MatrixCOO<std::complex<float>, int>*                dst);
#endif

    template bool csr_to_coo(int                                 omp_threads,
                             int64_t                             nnz,
                             int                                 nrow,
                             int                                 ncol,
                             const MatrixCSR<int, int, PtrType>& src,
                             MatrixCOO<int, int>*                dst);

    template bool csr_to_mcsr(int                                    omp_threads,
                              int64_t                                nnz,
                              int                                    nrow,
                              int                                    ncol,
                              const MatrixCSR<double, int, PtrType>& src,
                              MatrixMCSR<double, int>*               dst);

    template bool csr_to_mcsr(int                                   omp_threads,
                              int64_t                               nnz,
                              int                                   nrow,
                              int                                   ncol,
                              const MatrixCSR<float, int, PtrType>& src,
                              MatrixMCSR<float, int>*               dst);

#ifdef SUPPORT_COMPLEX
    template bool csr_to_mcsr(int                                                  omp_threads,
                              int64_t                                              nnz,
                              int                                                  nrow,
                              int                                                  ncol,
                              const MatrixCSR<std::complex<double>, int, PtrType>& src,
                              MatrixMCSR<std::complex<double>, int>*               dst);

    template bool csr_to_mcsr(int                                                 omp_threads,
                              int64_t                                             nnz,
                              int                                                 nrow,
                              int                                                 ncol,
                              const MatrixCSR<std::complex<float>, int, PtrType>& src,
                              MatrixMCSR<std::complex<float>, int>*               dst);
#endif

    template bool csr_to_mcsr(int                                 omp_threads,
                              int64_t                             nnz,
                              int                                 nrow,
                              int                                 ncol,
                              const MatrixCSR<int, int, PtrType>& src,
                              MatrixMCSR<int, int>*               dst);

    template bool mcsr_to_csr(int                              omp_threads,
                              int64_t                          nnz,
                              int                              nrow,
                              int                              ncol,
                              const MatrixMCSR<double, int>&   src,
                              MatrixCSR<double, int, PtrType>* dst);

    template bool mcsr_to_csr(int                             omp_threads,
                              int64_t                         nnz,
                              int                             nrow,
                              int                             ncol,
                              const MatrixMCSR<float, int>&   src,
                              MatrixCSR<float, int, PtrType>* dst);

#ifdef SUPPORT_COMPLEX
    template bool mcsr_to_csr(int                                            omp_threads,
                              int64_t                                        nnz,
                              int                                            nrow,
                              int                                            ncol,
                              const MatrixMCSR<std::complex<double>, int>&   src,
                              MatrixCSR<std::complex<double>, int, PtrType>* dst);

    template bool mcsr_to_csr(int                                           omp_threads,
                              int64_t                                       nnz,
                              int                                           nrow,
                              int                                           ncol,
                              const MatrixMCSR<std::complex<float>, int>&   src,
                              MatrixCSR<std::complex<float>, int, PtrType>* dst);
#endif

    template bool mcsr_to_csr(int                           omp_threads,
                              int64_t                       nnz,
                              int                           nrow,
                              int                           ncol,
                              const MatrixMCSR<int, int>&   src,
                              MatrixCSR<int, int, PtrType>* dst);

    template bool csr_to_bcsr(int                                    omp_threads,
                              int64_t                                nnz,
                              int                                    nrow,
                              int                                    ncol,
                              const MatrixCSR<double, int, PtrType>& src,
                              MatrixBCSR<double, int>*               dst);

    template bool csr_to_bcsr(int                                   omp_threads,
                              int64_t                               nnz,
                              int                                   nrow,
                              int                                   ncol,
                              const MatrixCSR<float, int, PtrType>& src,
                              MatrixBCSR<float, int>*               dst);

#ifdef SUPPORT_COMPLEX
    template bool csr_to_bcsr(int                                                  omp_threads,
                              int64_t                                              nnz,
                              int                                                  nrow,
                              int                                                  ncol,
                              const MatrixCSR<std::complex<double>, int, PtrType>& src,
                              MatrixBCSR<std::complex<double>, int>*               dst);

    template bool csr_to_bcsr(int                                                 omp_threads,
                              int64_t                                             nnz,
                              int                                                 nrow,
                              int                                                 ncol,
                              const MatrixCSR<std::complex<float>, int, PtrType>& src,
                              MatrixBCSR<std::complex<float>, int>*               dst);
#endif

    template bool csr_to_bcsr(int                                 omp_threads,
                              int64_t                             nnz,
                              int                                 nrow,
                              int                                 ncol,
                              const MatrixCSR<int, int, PtrType>& src,
                              MatrixBCSR<int, int>*               dst);

    template bool bcsr_to_csr(int                              omp_threads,
                              int64_t                          nnz,
                              int                              nrow,
                              int                              ncol,
                              const MatrixBCSR<double, int>&   src,
                              MatrixCSR<double, int, PtrType>* dst);

    template bool bcsr_to_csr(int                             omp_threads,
                              int64_t                         nnz,
                              int                             nrow,
                              int                             ncol,
                              const MatrixBCSR<float, int>&   src,
                              MatrixCSR<float, int, PtrType>* dst);

#ifdef SUPPORT_COMPLEX
    template bool bcsr_to_csr(int                                            omp_threads,
                              int64_t                                        nnz,
                              int                                            nrow,
                              int                                            ncol,
                              const MatrixBCSR<std::complex<double>, int>&   src,
                              MatrixCSR<std::complex<double>, int, PtrType>* dst);

    template bool bcsr_to_csr(int                                           omp_threads,
                              int64_t                                       nnz,
                              int                                           nrow,
                              int                                           ncol,
                              const MatrixBCSR<std::complex<float>, int>&   src,
                              MatrixCSR<std::complex<float>, int, PtrType>* dst);
#endif

    template bool bcsr_to_csr(int                           omp_threads,
                              int64_t                       nnz,
                              int                           nrow,
                              int                           ncol,
                              const MatrixBCSR<int, int>&   src,
                              MatrixCSR<int, int, PtrType>* dst);

    template bool csr_to_dia(int                                    omp_threads,
                             int64_t                                nnz,
                             int                                    nrow,
                             int                                    ncol,
                             const MatrixCSR<double, int, PtrType>& src,
                             MatrixDIA<double, int>*                dst,
                             int64_t*                               nnz_dia);

    template bool csr_to_dia(int                                   omp_threads,
                             int64_t                               nnz,
                             int                                   nrow,
                             int                                   ncol,
                             const MatrixCSR<float, int, PtrType>& src,
                             MatrixDIA<float, int>*                dst,
                             int64_t*                              nnz_dia);

#ifdef SUPPORT_COMPLEX
    template bool csr_to_dia(int                                                  omp_threads,
                             int64_t                                              nnz,
                             int                                                  nrow,
                             int                                                  ncol,
                             const MatrixCSR<std::complex<double>, int, PtrType>& src,
                             MatrixDIA<std::complex<double>, int>*                dst,
                             int64_t*                                             nnz_dia);

    template bool csr_to_dia(int                                                 omp_threads,
                             int64_t                                             nnz,
                             int                                                 nrow,
                             int                                                 ncol,
                             const MatrixCSR<std::complex<float>, int, PtrType>& src,
                             MatrixDIA<std::complex<float>, int>*                dst,
                             int64_t*                                            nnz_dia);
#endif

    template bool csr_to_dia(int                                 omp_threads,
                             int64_t                             nnz,
                             int                                 nrow,
                             int                                 ncol,
                             const MatrixCSR<int, int, PtrType>& src,
                             MatrixDIA<int, int>*                dst,
                             int64_t*                            nnz_dia);

    template bool csr_to_hyb(int                                    omp_threads,
                             int64_t                                nnz,
                             int                                    nrow,
                             int                                    ncol,
                             const MatrixCSR<double, int, PtrType>& src,
                             MatrixHYB<double, int>*                dst,
                             int64_t*                               nnz_hyb,
                             int64_t*                               nnz_ell,
                             int64_t*                               nnz_coo);

    template bool csr_to_hyb(int                                   omp_threads,
                             int64_t                               nnz,
                             int                                   nrow,
                             int                                   ncol,
                             const MatrixCSR<float, int, PtrType>& src,
                             MatrixHYB<float, int>*                dst,
                             int64_t*                              nnz_hyb,
                             int64_t*                              nnz_ell,
                             int64_t*                              nnz_coo);

#ifdef SUPPORT_COMPLEX
    template bool csr_to_hyb(int                                                  omp_threads,
                             int64_t                                              nnz,
                             int                                                  nrow,
                             int                                                  ncol,
                             const MatrixCSR<std::complex<double>, int, PtrType>& src,
                             MatrixHYB<std::complex<double>, int>*                dst,
                             int64_t*                                             nnz_hyb,
                             int64_t*                                             nnz_ell,
                             int64_t*                                             nnz_coo);

    template bool csr_to_hyb(int                                                 omp_threads,
                             int64_t                                             nnz,
                             int                                                 nrow,
                             int                                                 ncol,
                             const MatrixCSR<std::complex<float>, int, PtrType>& src,
                             MatrixHYB<std::complex<float>, int>*                dst,
                             int64_t*                                            nnz_hyb,
                             int64_t*                                            nnz_ell,
                             int64_t*                                            nnz_coo);
#endif

    template bool csr_to_hyb(int                                 omp_threads,
                             int64_t                             nnz,
                             int                                 nrow,
                             int                                 ncol,
                             const MatrixCSR<int, int, PtrType>& src,
                             MatrixHYB<int, int>*                dst,
                             int64_t*                            nnz_hyb,
                             int64_t*                            nnz_ell,
                             int64_t*                            nnz_coo);

    template bool csr_to_ell(int                                    omp_threads,
                             int64_t                                nnz,
                             int                                    nrow,
                             int                                    ncol,
                             const MatrixCSR<double, int, PtrType>& src,
                             MatrixELL<double, int>*                dst,
                             int64_t*                               nnz_ell);

    template bool csr_to_ell(int                                   omp_threads,
                             int64_t                               nnz,
                             int                                   nrow,
                             int                                   ncol,
                             const MatrixCSR<float, int, PtrType>& src,
                             MatrixELL<float, int>*                dst,
                             int64_t*                              nnz_ell);

#ifdef SUPPORT_COMPLEX
    template bool csr_to_ell(int                                                  omp_threads,
                             int64_t                                              nnz,
                             int                                                  nrow,
                             int                                                  ncol,
                             const MatrixCSR<std::complex<double>, int, PtrType>& src,
                             MatrixELL<std::complex<double>, int>*                dst,
                             int64_t*                                             nnz_ell);

    template bool csr_to_ell(int                                                 omp_threads,
                             int64_t                                             nnz,
                             int                                                 nrow,
                             int                                                 ncol,
                             const MatrixCSR<std::complex<float>, int, PtrType>& src,
                             MatrixELL<std::complex<float>, int>*                dst,
                             int64_t*                                            nnz_ell);
#endif

    template bool csr_to_ell(int                                 omp_threads,
                             int64_t                             nnz,
                             int                                 nrow,
                             int                                 ncol,
                             const MatrixCSR<int, int, PtrType>& src,
                             MatrixELL<int, int>*                dst,
                             int64_t*                            nnz_ell);

    template bool csr_to_dense(int                                    omp_threads,
                               int64_t                                nnz,
                               int                                    nrow,
                               int                                    ncol,
                               const MatrixCSR<double, int, PtrType>& src,
                               MatrixDENSE<double>*                   dst);

    template bool csr_to_dense(int                                   omp_threads,
                               int64_t                               nnz,
                               int                                   nrow,
                               int                                   ncol,
                               const MatrixCSR<float, int, PtrType>& src,
                               MatrixDENSE<float>*                   dst);

#ifdef SUPPORT_COMPLEX
    template bool csr_to_dense(int                                                  omp_threads,
                               int64_t                                              nnz,
                               int                                                  nrow,
                               int                                                  ncol,
                               const MatrixCSR<std::complex<double>, int, PtrType>& src,
                               MatrixDENSE<std::complex<double>>*                   dst);

    template bool csr_to_dense(int                                                 omp_threads,
                               int64_t                                             nnz,
                               int                                                 nrow,
                               int                                                 ncol,
                               const MatrixCSR<std::complex<float>, int, PtrType>& src,
                               MatrixDENSE<std::complex<float>>*                   dst);
#endif

    template bool csr_to_dense(int                                 omp_threads,
                               int64_t                             nnz,
                               int                                 nrow,
                               int                                 ncol,
                               const MatrixCSR<int, int, PtrType>& src,
                               MatrixDENSE<int>*                   dst);

    template bool dense_to_csr(int                              omp_threads,
                               int                              nrow,
                               int                              ncol,
                               const MatrixDENSE<double>&       src,
                               MatrixCSR<double, int, PtrType>* dst,
                               int64_t*                         nnz);

    template bool dense_to_csr(int                             omp_threads,
                               int                             nrow,
                               int                             ncol,
                               const MatrixDENSE<float>&       src,
                               MatrixCSR<float, int, PtrType>* dst,
                               int64_t*                        nnz);

#ifdef SUPPORT_COMPLEX
    template bool dense_to_csr(int                                            omp_threads,
                               int                                            nrow,
                               int                                            ncol,
                               const MatrixDENSE<std::complex<double>>&       src,
                               MatrixCSR<std::complex<double>, int, PtrType>* dst,
                               int64_t*                                       nnz);

    template bool dense_to_csr(int                                           omp_threads,
                               int                                           nrow,
                               int                                           ncol,
                               const MatrixDENSE<std::complex<float>>&       src,
                               MatrixCSR<std::complex<float>, int, PtrType>* dst,
                               int64_t*                                      nnz);
#endif

    template bool dense_to_csr(int                           omp_threads,
                               int                           nrow,
                               int                           ncol,
                               const MatrixDENSE<int>&       src,
                               MatrixCSR<int, int, PtrType>* dst,
                               int64_t*                      nnz);

    template bool dia_to_csr(int                              omp_threads,
                             int64_t                          nnz,
                             int                              nrow,
                             int                              ncol,
                             const MatrixDIA<double, int>&    src,
                             MatrixCSR<double, int, PtrType>* dst,
                             int64_t*                         nnz_csr);

    template bool dia_to_csr(int                             omp_threads,
                             int64_t                         nnz,
                             int                             nrow,
                             int                             ncol,
                             const MatrixDIA<float, int>&    src,
                             MatrixCSR<float, int, PtrType>* dst,
                             int64_t*                        nnz_csr);

#ifdef SUPPORT_COMPLEX
    template bool dia_to_csr(int                                            omp_threads,
                             int64_t                                        nnz,
                             int                                            nrow,
                             int                                            ncol,
                             const MatrixDIA<std::complex<double>, int>&    src,
                             MatrixCSR<std::complex<double>, int, PtrType>* dst,
                             int64_t*                                       nnz_csr);

    template bool dia_to_csr(int                                           omp_threads,
                             int64_t                                       nnz,
                             int                                           nrow,
                             int                                           ncol,
                             const MatrixDIA<std::complex<float>, int>&    src,
                             MatrixCSR<std::complex<float>, int, PtrType>* dst,
                             int64_t*                                      nnz_csr);
#endif

    template bool dia_to_csr(int                           omp_threads,
                             int64_t                       nnz,
                             int                           nrow,
                             int                           ncol,
                             const MatrixDIA<int, int>&    src,
                             MatrixCSR<int, int, PtrType>* dst,
                             int64_t*                      nnz_csr);

    template bool ell_to_csr(int                              omp_threads,
                             int64_t                          nnz,
                             int                              nrow,
                             int                              ncol,
                             const MatrixELL<double, int>&    src,
                             MatrixCSR<double, int, PtrType>* dst,
                             int64_t*                         nnz_csr);

    template bool ell_to_csr(int                             omp_threads,
                             int64_t                         nnz,
                             int                             nrow,
                             int                             ncol,
                             const MatrixELL<float, int>&    src,
                             MatrixCSR<float, int, PtrType>* dst,
                             int64_t*                        nnz_csr);

#ifdef SUPPORT_COMPLEX
    template bool ell_to_csr(int                                            omp_threads,
                             int64_t                                        nnz,
                             int                                            nrow,
                             int                                            ncol,
                             const MatrixELL<std::complex<double>, int>&    src,
                             MatrixCSR<std::complex<double>, int, PtrType>* dst,
                             int64_t*                                       nnz_csr);

    template bool ell_to_csr(int                                           omp_threads,
                             int64_t                                       nnz,
                             int                                           nrow,
                             int                                           ncol,
                             const MatrixELL<std::complex<float>, int>&    src,
                             MatrixCSR<std::complex<float>, int, PtrType>* dst,
                             int64_t*                                      nnz_csr);
#endif

    template bool ell_to_csr(int                           omp_threads,
                             int64_t                       nnz,
                             int                           nrow,
                             int                           ncol,
                             const MatrixELL<int, int>&    src,
                             MatrixCSR<int, int, PtrType>* dst,
                             int64_t*                      nnz_csr);

    template bool coo_to_csr(int                              omp_threads,
                             int64_t                          nnz,
                             int                              nrow,
                             int                              ncol,
                             const MatrixCOO<double, int>&    src,
                             MatrixCSR<double, int, PtrType>* dst);

    template bool coo_to_csr(int                             omp_threads,
                             int64_t                         nnz,
                             int                             nrow,
                             int                             ncol,
                             const MatrixCOO<float, int>&    src,
                             MatrixCSR<float, int, PtrType>* dst);

#ifdef SUPPORT_COMPLEX
    template bool coo_to_csr(int                                            omp_threads,
                             int64_t                                        nnz,
                             int                                            nrow,
                             int                                            ncol,
                             const MatrixCOO<std::complex<double>, int>&    src,
                             MatrixCSR<std::complex<double>, int, PtrType>* dst);

    template bool coo_to_csr(int                                           omp_threads,
                             int64_t                                       nnz,
                             int                                           nrow,
                             int                                           ncol,
                             const MatrixCOO<std::complex<float>, int>&    src,
                             MatrixCSR<std::complex<float>, int, PtrType>* dst);
#endif

    template bool coo_to_csr(int                           omp_threads,
                             int64_t                       nnz,
                             int                           nrow,
                             int                           ncol,
                             const MatrixCOO<int, int>&    src,
                             MatrixCSR<int, int, PtrType>* dst);

    template bool hyb_to_csr(int                              omp_threads,
                             int64_t                          nnz,
                             int                              nrow,
                             int                              ncol,
                             int64_t                          nnz_ell,
                             int64_t                          nnz_coo,
                             const MatrixHYB<double, int>&    src,
                             MatrixCSR<double, int, PtrType>* dst,
                             int64_t*                         nnz_csr);

    template bool hyb_to_csr(int                             omp_threads,
                             int64_t                         nnz,
                             int                             nrow,
                             int                             ncol,
                             int64_t                         nnz_ell,
                             int64_t                         nnz_coo,
                             const MatrixHYB<float, int>&    src,
                             MatrixCSR<float, int, PtrType>* dst,
                             int64_t*                        nnz_csr);

#ifdef SUPPORT_COMPLEX
    template bool hyb_to_csr(int                                            omp_threads,
                             int64_t                                        nnz,
                             int                                            nrow,
                             int                                            ncol,
                             int64_t                                        nnz_ell,
                             int64_t                                        nnz_coo,
                             const MatrixHYB<std::complex<double>, int>&    src,
                             MatrixCSR<std::complex<double>, int, PtrType>* dst,
                             int64_t*                                       nnz_csr);

    template bool hyb_to_csr(int                                           omp_threads,
                             int64_t                                       nnz,
                             int                                           nrow,
                             int                                           ncol,
                             int64_t                                       nnz_ell,
                             int64_t                                       nnz_coo,
                             const MatrixHYB<std::complex<float>, int>&    src,
                             MatrixCSR<std::complex<float>, int, PtrType>* dst,
                             int64_t*                                      nnz_csr);
#endif

    template bool hyb_to_csr(int                           omp_threads,
                             int64_t                       nnz,
                             int                           nrow,
                             int                           ncol,
                             int64_t                       nnz_ell,
                             int64_t                       nnz_coo,
                             const MatrixHYB<int, int>&    src,
                             MatrixCSR<int, int, PtrType>* dst,
                             int64_t*                      nnz_csr);

} // namespace rocalution
