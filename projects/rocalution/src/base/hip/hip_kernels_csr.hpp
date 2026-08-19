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

#ifndef ROCALUTION_HIP_HIP_KERNELS_CSR_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_CSR_HPP_

#include "../matrix_formats_ind.hpp"
#include "hip_atomics.hpp"
#include "hip_unordered_map.hpp"
#include "hip_unordered_set.hpp"
#include "hip_utils.hpp"

#include <hip/hip_runtime.h>

namespace rocalution
{
    template <typename T, typename I, typename J>
    __global__ void kernel_csr_scale_diagonal(I nrow,
                                              const J* __restrict__ row_offset,
                                              const I* __restrict__ col,
                                              T alpha,
                                              T* __restrict__ val)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        for(J aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if(ai == col[aj])
            {
                val[aj] = alpha * val[aj];
            }
        }
    }

    template <typename T, typename I, typename J>
    __global__ void kernel_csr_scale_offdiagonal(I nrow,
                                                 const J* __restrict__ row_offset,
                                                 const I* __restrict__ col,
                                                 T alpha,
                                                 T* __restrict__ val)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        for(J aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if(ai != col[aj])
            {
                val[aj] = alpha * val[aj];
            }
        }
    }

    template <typename T, typename I, typename J>
    __global__ void kernel_csr_add_diagonal(I nrow,
                                            const J* __restrict__ row_offset,
                                            const I* __restrict__ col,
                                            T alpha,
                                            T* __restrict__ val)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        for(J aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if(ai == col[aj])
            {
                val[aj] = val[aj] + alpha;
            }
        }
    }

    template <typename T, typename I, typename J>
    __global__ void kernel_csr_add_offdiagonal(I nrow,
                                               const J* __restrict__ row_offset,
                                               const I* __restrict__ col,
                                               T alpha,
                                               T* __restrict__ val)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        for(J aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if(ai != col[aj])
            {
                val[aj] = val[aj] + alpha;
            }
        }
    }

    template <unsigned int WFSIZE, typename T, typename I, typename J>
    __global__ void kernel_csr_extract_diag(I nrow,
                                            const J* __restrict__ row_offset,
                                            const I* __restrict__ col,
                                            const T* __restrict__ val,
                                            T* __restrict__ vec)
    {
        I tid = threadIdx.x;
        I gid = blockIdx.x * blockDim.x + tid;
        I lid = tid & (WFSIZE - 1);
        I row = gid / WFSIZE;

        if(row >= nrow)
        {
            return;
        }

        J start = row_offset[row];
        J end   = row_offset[row + 1];

        for(J aj = start + lid; aj < end; aj += WFSIZE)
        {
            if(row == col[aj])
            {
                vec[row] = val[aj];
            }
        }
    }

    template <typename T, typename I, typename J>
    __global__ void kernel_csr_extract_inv_diag(I nrow,
                                                const J* __restrict__ row_offset,
                                                const I* __restrict__ col,
                                                const T* __restrict__ val,
                                                T* __restrict__ vec,
                                                int* __restrict__ detect_zero)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        for(J aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if(ai == col[aj])
            {
                if(val[aj] != static_cast<T>(0))
                {
                    vec[ai] = static_cast<T>(1) / val[aj];
                }
                else
                {
                    vec[ai] = static_cast<T>(1);

                    *detect_zero = 1;
                }
            }
        }
    }

    template <typename T, typename I, typename J>
    __global__ void kernel_csr_extract_submatrix_row_nnz(const J* __restrict__ row_offset,
                                                         const I* __restrict__ col,
                                                         const T* __restrict__ val,
                                                         I smrow_offset,
                                                         I smcol_offset,
                                                         I smrow_size,
                                                         I smcol_size,
                                                         J* __restrict__ row_nnz)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= smrow_size)
        {
            return;
        }

        I nnz = 0;
        I ind = ai + smrow_offset;

        for(J aj = row_offset[ind]; aj < row_offset[ind + 1]; ++aj)
        {
            I c = col[aj];

            if((c >= smcol_offset) && (c < smcol_offset + smcol_size))
            {
                ++nnz;
            }
        }

        row_nnz[ai] = nnz;
    }

    template <typename T, typename I, typename J>
    __global__ void kernel_csr_extract_submatrix_copy(const J* __restrict__ row_offset,
                                                      const I* __restrict__ col,
                                                      const T* __restrict__ val,
                                                      I smrow_offset,
                                                      I smcol_offset,
                                                      I smrow_size,
                                                      I smcol_size,
                                                      const J* __restrict__ sm_row_offset,
                                                      I* __restrict__ sm_col,
                                                      T* __restrict__ sm_val)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= smrow_size)
        {
            return;
        }

        J row_nnz = sm_row_offset[ai];
        I ind     = ai + smrow_offset;

        for(J aj = row_offset[ind]; aj < row_offset[ind + 1]; ++aj)
        {
            I c = col[aj];

            if((c >= smcol_offset) && (c < smcol_offset + smcol_size))
            {
                sm_col[row_nnz] = c - smcol_offset;
                sm_val[row_nnz] = val[aj];
                ++row_nnz;
            }
        }
    }

    template <typename T, typename I, typename J>
    __global__ void kernel_csr_diagmatmult_r(I nrow,
                                             const J* __restrict__ row_offset,
                                             const I* __restrict__ col,
                                             const T* __restrict__ diag,
                                             T* __restrict__ val)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        for(J aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            val[aj] = val[aj] * diag[col[aj]];
        }
    }

    template <typename T, typename I, typename J>
    __global__ void kernel_csr_diagmatmult_l(I nrow,
                                             const J* __restrict__ row_offset,
                                             const T* __restrict__ diag,
                                             T* __restrict__ val)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        for(J aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            val[aj] = val[aj] * diag[ai];
        }
    }

    // Calculates the number of non-zero elements per row
    template <typename I, typename J>
    __global__ void
        kernel_calc_row_nnz(I nrow, const J* __restrict__ row_offset, I* __restrict__ row_nnz)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        row_nnz[ai] = row_offset[ai + 1] - row_offset[ai];
    }

    // Performs a permutation on the vector of non-zero elements per row
    //
    // Inputs:   nrow:         number of rows in matrix
    //           row_nnz_src:  original number of non-zero elements per row
    //           perm_vec:     permutation vector
    // Outputs:  row_nnz_dst   permuted number of non-zero elements per row
    template <typename I, typename J>
    __global__ void kernel_permute_row_nnz(I nrow,
                                           const J* __restrict__ row_offset,
                                           const I* __restrict__ perm_vec,
                                           J* __restrict__ row_nnz_dst)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        if(ai == 0)
        {
            row_nnz_dst[0] = 0;
        }

        row_nnz_dst[perm_vec[ai] + 1] = row_offset[ai + 1] - row_offset[ai];
    }

    // Permutes rows
    //
    // Inputs:   nrow:             number of rows in matrix
    //           row_offset:       original row pointer
    //           perm_row_offset:  permuted row pointer
    //           col:              original column indices of elements
    //           data:             original data vector
    //           perm_vec:         permutation vector
    //           row_nnz:          number of non-zero elements per row
    // Outputs:  perm_col:         permuted column indices of elements
    //           perm_data:        permuted data vector
    template <unsigned int WF_SIZE, typename T, typename I, typename J>
    __global__ void kernel_permute_rows(I nrow,
                                        const J* __restrict__ row_offset,
                                        const J* __restrict__ perm_row_offset,
                                        const I* __restrict__ col,
                                        const T* __restrict__ data,
                                        const I* __restrict__ perm_vec,
                                        I* __restrict__ perm_col,
                                        T* __restrict__ perm_data)
    {
        I tid = threadIdx.x;
        I gid = blockIdx.x * blockDim.x + tid;
        I lid = tid & (WF_SIZE - 1);
        I row = gid / WF_SIZE;

        if(row >= nrow)
        {
            return;
        }

        J perm_index = perm_row_offset[perm_vec[row]];
        J prev_index = row_offset[row];
        I num_elems  = row_offset[row + 1] - prev_index;

        for(I i = lid; i < num_elems; i += WF_SIZE)
        {
            perm_data[perm_index + i] = data[prev_index + i];
            perm_col[perm_index + i]  = col[prev_index + i];
        }
    }

    // Permutes columns
    //
    // Inputs:   nrow:             number of rows in matrix
    //           row_offset:       row pointer
    //           perm_vec:         permutation vector
    //           perm_col:         row-permuted column indices of elements
    //           perm_data:        row-permuted data
    // Outputs:  col:              fully permuted column indices of elements
    //           data:             fully permuted data
    template <unsigned int size, typename T, typename I, typename J>
    __launch_bounds__(256) __global__ void kernel_permute_cols(I nrow,
                                                               const J* __restrict__ row_offset,
                                                               const I* __restrict__ perm_vec,
                                                               const I* __restrict__ perm_col,
                                                               const T* __restrict__ perm_data,
                                                               I* __restrict__ col,
                                                               T* __restrict__ data)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        J elem_index = row_offset[ai];
        I num_elems  = row_offset[ai + 1] - elem_index;

        I ccol[size];
        T cval[size];

        col += elem_index;
        data += elem_index;
        perm_col += elem_index;
        perm_data += elem_index;

        for(I i = 0; i < num_elems; ++i)
        {
            ccol[i] = col[i];
            cval[i] = data[i];
        }

        for(I i = 0; i < num_elems; ++i)
        {
            I comp = perm_vec[perm_col[i]];

            I j;
            for(j = i - 1; j >= 0; --j)
            {
                I c = ccol[j];
                if(c > comp)
                {
                    cval[j + 1] = cval[j];
                    ccol[j + 1] = c;
                }
                else
                {
                    break;
                }
            }

            cval[j + 1] = perm_data[i];
            ccol[j + 1] = comp;
        }

        for(I i = 0; i < num_elems; ++i)
        {
            col[i]  = ccol[i];
            data[i] = cval[i];
        }
    }

    // Permutes columns
    //
    // Inputs:   nrow:             number of rows in matrix
    //           row_offset:       row pointer
    //           perm_vec:         permutation vector
    //           perm_col:         row-permuted column indices of elements
    //           perm_data:        row-permuted data
    // Outputs:  col:              fully permuted column indices of elements
    //           data:             fully permuted data
    template <typename T, typename I, typename J>
    __global__ void kernel_permute_cols_fallback(I nrow,
                                                 const J* __restrict__ row_offset,
                                                 const I* __restrict__ perm_vec,
                                                 const I* __restrict__ perm_col,
                                                 const T* __restrict__ perm_data,
                                                 I* __restrict__ col,
                                                 T* __restrict__ data)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        J elem_index = row_offset[ai];
        I num_elems  = row_offset[ai + 1] - elem_index;

        col += elem_index;
        data += elem_index;
        perm_col += elem_index;
        perm_data += elem_index;

        for(I i = 0; i < num_elems; ++i)
        {
            I comp = perm_vec[perm_col[i]];

            I j;
            for(j = i - 1; j >= 0; --j)
            {
                I c = col[j];
                if(c > comp)
                {
                    data[j + 1] = data[j];
                    col[j + 1]  = c;
                }
                else
                {
                    break;
                }
            }

            data[j + 1] = perm_data[i];
            col[j + 1]  = comp;
        }
    }

    // TODO
    // kind of ugly and inefficient ... but works
    template <typename T, typename I, typename J>
    __global__ void kernel_csr_add_csr_same_struct(I nrow,
                                                   const J* __restrict__ out_row_offset,
                                                   const I* __restrict__ out_col,
                                                   const J* __restrict__ in_row_offset,
                                                   const I* __restrict__ in_col,
                                                   const T* __restrict__ in_val,
                                                   T alpha,
                                                   T beta,
                                                   T* __restrict__ out_val)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        J first_col = in_row_offset[ai];

        for(J ajj = out_row_offset[ai]; ajj < out_row_offset[ai + 1]; ++ajj)
        {
            for(J aj = first_col; aj < in_row_offset[ai + 1]; ++aj)
            {
                if(in_col[aj] == out_col[ajj])
                {
                    out_val[ajj] = alpha * out_val[ajj] + beta * in_val[aj];
                    ++first_col;
                    break;
                }
            }
        }
    }

    // Computes the lower triangular part nnz per row
    template <typename I, typename J>
    __global__ void kernel_csr_lower_nnz_per_row(I nrow,
                                                 const J* __restrict__ src_row_offset,
                                                 const I* __restrict__ src_col,
                                                 J* __restrict__ nnz_per_row)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        nnz_per_row[ai] = 0;

        for(J aj = src_row_offset[ai]; aj < src_row_offset[ai + 1]; ++aj)
        {
            if(src_col[aj] <= ai)
            {
                ++nnz_per_row[ai];
            }
        }
    }

    // Computes the upper triangular part nnz per row
    template <typename I, typename J>
    __global__ void kernel_csr_upper_nnz_per_row(I nrow,
                                                 const J* __restrict__ src_row_offset,
                                                 const I* __restrict__ src_col,
                                                 J* __restrict__ nnz_per_row)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        nnz_per_row[ai] = 0;

        for(J aj = src_row_offset[ai]; aj < src_row_offset[ai + 1]; ++aj)
        {
            if(src_col[aj] >= ai)
            {
                ++nnz_per_row[ai];
            }
        }
    }

    // Computes the stricktly lower triangular part nnz per row
    template <typename I, typename J>
    __global__ void kernel_csr_slower_nnz_per_row(I nrow,
                                                  const J* __restrict__ src_row_offset,
                                                  const I* __restrict__ src_col,
                                                  J* __restrict__ nnz_per_row)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        nnz_per_row[ai] = 0;

        for(J aj = src_row_offset[ai]; aj < src_row_offset[ai + 1]; ++aj)
        {
            if(src_col[aj] < ai)
            {
                ++nnz_per_row[ai];
            }
        }
    }

    // Computes the stricktly upper triangular part nnz per row
    template <typename I, typename J>
    __global__ void kernel_csr_supper_nnz_per_row(I nrow,
                                                  const J* __restrict__ src_row_offset,
                                                  const I* __restrict__ src_col,
                                                  J* __restrict__ nnz_per_row)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        nnz_per_row[ai] = 0;

        for(J aj = src_row_offset[ai]; aj < src_row_offset[ai + 1]; ++aj)
        {
            if(src_col[aj] > ai)
            {
                ++nnz_per_row[ai];
            }
        }
    }

    // Extracts lower triangular part for given nnz per row array (partial sums nnz)
    template <typename T, typename I, typename J>
    __global__ void kernel_csr_extract_l_triangular(I nrow,
                                                    const J* __restrict__ src_row_offset,
                                                    const I* __restrict__ src_col,
                                                    const T* __restrict__ src_val,
                                                    J* __restrict__ nnz_per_row,
                                                    I* __restrict__ dst_col,
                                                    T* __restrict__ dst_val)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        J dst_index = nnz_per_row[ai];
        J src_index = src_row_offset[ai];

        for(J aj = 0; aj < nnz_per_row[ai + 1] - nnz_per_row[ai]; ++aj)
        {
            dst_col[dst_index] = src_col[src_index];
            dst_val[dst_index] = src_val[src_index];

            ++dst_index;
            ++src_index;
        }
    }

    // Extracts upper triangular part for given nnz per row array (partial sums nnz)
    template <typename T, typename I, typename J>
    __global__ void kernel_csr_extract_u_triangular(I nrow,
                                                    const J* __restrict__ src_row_offset,
                                                    const I* __restrict__ src_col,
                                                    const T* __restrict__ src_val,
                                                    J* __restrict__ nnz_per_row,
                                                    I* __restrict__ dst_col,
                                                    T* __restrict__ dst_val)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        I num_elements = nnz_per_row[ai + 1] - nnz_per_row[ai];
        J src_index    = src_row_offset[ai + 1] - num_elements;
        J dst_index    = nnz_per_row[ai];

        for(I aj = 0; aj < num_elements; ++aj)
        {
            dst_col[dst_index] = src_col[src_index];
            dst_val[dst_index] = src_val[src_index];

            ++dst_index;
            ++src_index;
        }
    }

    // Compress
    template <typename T, typename I, typename J>
    __global__ void kernel_csr_compress_count_nrow(const J* __restrict__ row_offset,
                                                   const I* __restrict__ col,
                                                   const T* __restrict__ val,
                                                   I      nrow,
                                                   double drop_off,
                                                   I* __restrict__ row_offset_new)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        for(J aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if((hip_abs(val[aj]) > drop_off) || (col[aj] == ai))
            {
                ++row_offset_new[ai];
            }
        }
    }

    // Compress
    template <typename T, typename I, typename J>
    __global__ void kernel_csr_compress_copy(const J* __restrict__ row_offset,
                                             const I* __restrict__ col,
                                             const T* __restrict__ val,
                                             I      nrow,
                                             double drop_off,
                                             const J* __restrict__ row_offset_new,
                                             I* __restrict__ col_new,
                                             T* __restrict__ val_new)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        J ajj = row_offset_new[ai];

        for(J aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if((hip_abs(val[aj]) > drop_off) || (col[aj] == ai))
            {
                col_new[ajj] = col[aj];
                val_new[ajj] = val[aj];
                ++ajj;
            }
        }
    }

    // Extract column vector
    template <typename T, typename I, typename J>
    __global__ void kernel_csr_extract_column_vector(const J* __restrict__ row_offset,
                                                     const I* __restrict__ col,
                                                     const T* __restrict__ val,
                                                     I nrow,
                                                     I idx,
                                                     T* __restrict__ vec)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        vec[ai] = static_cast<T>(0);

        for(J aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if(idx == col[aj])
            {
                vec[ai] = val[aj];
            }
        }
    }

    // Replace column vector - compute new offset
    template <typename T, typename I, typename J>
    __global__ void kernel_csr_replace_column_vector_offset(const J* __restrict__ row_offset,
                                                            const I* __restrict__ col,
                                                            I nrow,
                                                            I idx,
                                                            const T* __restrict__ vec,
                                                            J* __restrict__ offset)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        bool add = true;
        I    val = row_offset[ai + 1] - row_offset[ai];

        for(J aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if(col[aj] == idx)
            {
                add = false;
                break;
            }
        }

        if(add == true && hip_abs(vec[ai]) != 0.0)
        {
            ++val;
        }

        if(add == false && hip_abs(vec[ai]) == 0.0)
        {
            --val;
        }

        offset[ai] = val;
    }

    // Replace column vector - compute new offset
    template <typename T, typename I, typename J>
    __global__ void kernel_csr_replace_column_vector(const J* __restrict__ row_offset,
                                                     const I* __restrict__ col,
                                                     const T* __restrict__ val,
                                                     I nrow,
                                                     I idx,
                                                     const T* __restrict__ vec,
                                                     const J* __restrict__ offset,
                                                     I* __restrict__ new_col,
                                                     T* __restrict__ new_val)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        J aj = row_offset[ai];
        J k  = offset[ai];

        for(; aj < row_offset[ai + 1]; ++aj)
        {
            if(col[aj] < idx)
            {
                new_col[k] = col[aj];
                new_val[k] = val[aj];
                ++k;
            }
            else
            {
                break;
            }
        }

        if(hip_abs(vec[ai]) != 0.0)
        {
            new_col[k] = idx;
            new_val[k] = vec[ai];
            ++k;
            ++aj;
        }

        for(; aj < row_offset[ai + 1]; ++aj)
        {
            if(col[aj] > idx)
            {
                new_col[k] = col[aj];
                new_val[k] = val[aj];
                ++k;
            }
        }
    }

    // Extract row vector
    template <typename T, typename I, typename J>
    __global__ void kernel_csr_extract_row_vector(const J* __restrict__ row_offset,
                                                  const I* __restrict__ col,
                                                  const T* __restrict__ val,
                                                  I row_nnz,
                                                  I idx,
                                                  T* __restrict__ vec)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= row_nnz)
        {
            return;
        }

        J aj         = row_offset[idx] + ai;
        vec[col[aj]] = val[aj];
    }

    template <typename I, typename J>
    __global__ void kernel_csr_boundary_nnz(I       boundary_size,
                                            int64_t nnz,
                                            const I* __restrict__ boundary_index,
                                            const J* __restrict__ csr_row_ptr,
                                            const I* __restrict__ csr_col_ind,
                                            const J* __restrict__ ghost_csr_row_ptr,
                                            const I* __restrict__ ghost_csr_col_ind,
                                            const bool* __restrict__ connections,
                                            J* __restrict__ row_nnz)
    {
        I gid = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(gid >= boundary_size)
        {
            return;
        }

        // This row is a boundary row
        I row = boundary_index[gid];

        J ext_nnz = 0;

        // Extract interior part
        J row_begin = csr_row_ptr[row];
        J row_end   = csr_row_ptr[row + 1];

        // Interior
        for(J j = row_begin; j < row_end; ++j)
        {
            // Check whether this vertex is strongly connected
            if(connections[j] == true)
            {
                ++ext_nnz;
            }
        }

        // Ghost part
        J gst_row_begin = ghost_csr_row_ptr[row];
        J gst_row_end   = ghost_csr_row_ptr[row + 1];

        for(J j = gst_row_begin; j < gst_row_end; ++j)
        {
            // Check whether this vertex is strongly connected
            if(connections[j + nnz] == true)
            {
                ++ext_nnz;
            }
        }

        // Write total number of strongly connected coarse vertices to global memory
        row_nnz[gid] = ext_nnz;
    }

    template <typename I, typename J>
    __global__ void kernel_csr_extract_boundary(I       boundary_size,
                                                int64_t nnz,
                                                int64_t global_column_begin,
                                                const I* __restrict__ boundary_index,
                                                const J* __restrict__ csr_row_ptr,
                                                const I* __restrict__ csr_col_ind,
                                                const J* __restrict__ ghost_csr_row_ptr,
                                                const I* __restrict__ ghost_csr_col_ind,
                                                const bool* __restrict__ connections,
                                                const int64_t* __restrict__ l2g,
                                                const J* __restrict__ bnd_row_ptr,
                                                int64_t* __restrict__ bnd_col_ind)
    {
        I gid = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(gid >= boundary_size)
        {
            return;
        }

        // This row is a boundary row
        I row = boundary_index[gid];

        // Index into boundary array
        J idx = bnd_row_ptr[gid];

        // Extract interior part
        J row_begin = csr_row_ptr[row];
        J row_end   = csr_row_ptr[row + 1];

        // Interior
        for(J j = row_begin; j < row_end; ++j)
        {
            if(connections[j] == true)
            {
                // Get column index
                I col = csr_col_ind[j];

                // Shift column by global column offset, to obtain the global column index
                bnd_col_ind[idx++] = col + global_column_begin;
            }
        }

        // Extract ghost part
        J gst_row_begin = ghost_csr_row_ptr[row];
        J gst_row_end   = ghost_csr_row_ptr[row + 1];

        for(J j = gst_row_begin; j < gst_row_end; ++j)
        {
            // Check whether this vertex is strongly connected
            if(connections[j + nnz] == true)
            {
                // Get column index
                I col = ghost_csr_col_ind[j];

                // Transform local ghost index into global column index
                bnd_col_ind[idx++] = l2g[col];
            }
        }
    }

    __device__ __forceinline__ mis_tuple lexographical_max(mis_tuple* ti, mis_tuple* tj)
    {
        // find lexographical maximum
        if(tj->s > ti->s)
        {
            return *tj;
        }
        else if(tj->s == ti->s)
        {
            if(tj->v > ti->v)
            {
                return *tj;
            }
        }

        return *ti;
    }

    // Note: Remove in next major release
    template <unsigned int WF_SIZE, typename T, typename I, typename J>
    __global__ void kernel_csr_amg_connect(I nrow,
                                           T eps2,
                                           const J* __restrict__ row_offset,
                                           const I* __restrict__ col,
                                           const T* __restrict__ val,
                                           const T* __restrict__ diag,
                                           I* __restrict__ connections)
    {
        I tid = threadIdx.x;
        I gid = blockIdx.x * blockDim.x + tid;
        I lid = tid & (WF_SIZE - 1);
        I row = gid / WF_SIZE;
        if(row >= nrow)
        {
            return;
        }

        T eps_diag_r = eps2 * diag[row];

        J start = row_offset[row];
        J end   = row_offset[row + 1];

        for(J i = start + lid; i < end; i += WF_SIZE)
        {
            I c = col[i];
            T v = val[i];

            connections[i] = (c != row) && (hip_real(v * v) > hip_real(eps_diag_r * diag[c]));
        }
    }

    // Note: Remove in next major release
    __device__ __forceinline__ unsigned int hash(unsigned int x)
    {
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = (x >> 16) ^ x;
        return x;
    }

    // Note: Remove in next major release
    template <typename I, typename J>
    __global__ void kernel_csr_amg_init_mis_tuples(I nrow,
                                                   const J* __restrict__ row_offset,
                                                   const I* __restrict__ connections,
                                                   mis_tuple* __restrict__ tuples)
    {
        I row = blockIdx.x * blockDim.x + threadIdx.x;

        if(row >= nrow)
        {
            return;
        }

        I state = -2;

        J row_start = row_offset[row];
        J row_end   = row_offset[row + 1];

        for(J j = row_start; j < row_end; j++)
        {
            if(connections[j] == 1)
            {
                state = 0;
                break;
            }
        }

        tuples[row].s = state;
        tuples[row].v = hash(row);
        tuples[row].i = row;
    }

    // Note: Remove in next major release
    template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename I, typename J>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_amg_find_max_mis_tuples_shared(I nrow,
                                                       const J* __restrict__ row_offset,
                                                       const I* __restrict__ cols,
                                                       const I* __restrict__ connections,
                                                       const mis_tuple* __restrict__ tuples,
                                                       mis_tuple* __restrict__ max_tuples,
                                                       bool* done)
    {
        I                    tid = threadIdx.x;
        I                    gid = blockIdx.x * BLOCKSIZE + tid;
        I                    lid = tid & (WFSIZE - 1);
        I                    wid = tid / WFSIZE;
        I                    row = gid / WFSIZE;
        __shared__ mis_tuple smax_tuple[BLOCKSIZE];
        if(row >= nrow)
        {
            return;
        }

        smax_tuple[WFSIZE * wid] = max_tuples[row];

        __syncthreads();

        for(I k = 0; k < 2; k++)
        {
            mis_tuple t_max = smax_tuple[WFSIZE * wid];

            J row_start = row_offset[t_max.i];
            J row_end   = row_offset[t_max.i + 1];

            for(J j = row_start + lid; j < row_end; j += WFSIZE)
            {
                if(connections[j] == 1)
                {
                    I         col = cols[j];
                    mis_tuple tj  = tuples[col];

                    t_max = lexographical_max(&tj, &t_max);
                }
            }

            smax_tuple[WFSIZE * wid + lid] = t_max;

            __syncthreads();

            // Finish reducing the intra block lexographical max
            if(WFSIZE >= 64)
            {
                if(lid < 32)
                    smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                        &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 32]);
                __threadfence_block();
            }
            if(WFSIZE >= 32)
            {
                if(lid < 16)
                    smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                        &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 16]);
                __threadfence_block();
            }
            if(WFSIZE >= 16)
            {
                if(lid < 8)
                    smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                        &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 8]);
                __threadfence_block();
            }
            if(WFSIZE >= 8)
            {
                if(lid < 4)
                    smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                        &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 4]);
                __threadfence_block();
            }
            if(WFSIZE >= 4)
            {
                if(lid < 2)
                    smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                        &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 2]);
                __threadfence_block();
            }
            if(WFSIZE >= 2)
            {
                if(lid < 1)
                    smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                        &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 1]);
                __threadfence_block();
            }

            __syncthreads();
        }

        if(lid == 0)
        {
            max_tuples[row] = smax_tuple[WFSIZE * wid];
        }

        if(gid == 0)
        {
            *done = true;
        }
    }

    // Note: Remove in next major release
    template <typename I>
    __global__ void kernel_csr_amg_update_mis_tuples(I nrow,
                                                     const mis_tuple* __restrict__ max_tuples,
                                                     mis_tuple* __restrict__ tuples,
                                                     I* __restrict__ aggregates,
                                                     bool* done)
    {
        I row = blockIdx.x * blockDim.x + threadIdx.x;

        if(row >= nrow)
        {
            return;
        }

        if(tuples[row].s == 0)
        {
            mis_tuple t_max = max_tuples[row];

            if(t_max.i == row)
            {
                tuples[row].s   = 1;
                aggregates[row] = 1;
            }
            else if(t_max.s == 1)
            {
                tuples[row].s   = -1;
                aggregates[row] = 0;
            }
            else
            {
                *done = false;
            }
        }
    }

    // Note: Remove in next major release
    template <typename I, typename J>
    __global__ void kernel_csr_amg_update_aggregates(I nrow,
                                                     const J* __restrict__ row_offset,
                                                     const I* __restrict__ cols,
                                                     const I* __restrict__ connections,
                                                     const mis_tuple* __restrict__ max_tuples,
                                                     mis_tuple* __restrict__ tuples,
                                                     I* __restrict__ aggregates)
    {
        I row = blockIdx.x * blockDim.x + threadIdx.x;

        if(row >= nrow)
        {
            return;
        }

        mis_tuple t = max_tuples[row];

        if(t.s == -1)
        {
            J row_start = row_offset[row];
            J row_end   = row_offset[row + 1];

            for(J j = row_start; j < row_end; j++)
            {
                if(connections[j] == 1)
                {
                    I col = cols[j];

                    if(max_tuples[col].s == 1)
                    {
                        aggregates[row] = aggregates[col];
                        tuples[row].s   = 1;
                        break;
                    }
                }
            }
        }
        else if(t.s == -2)
        {
            aggregates[row] = -2;
        }
    }

    // Note: Remove in next major release
    template <unsigned int BLOCKSIZE,
              unsigned int WFSIZE,
              unsigned int HASHSIZE,
              typename I,
              typename J>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_sa_prolong_nnz(I nrow,
                                       const J* __restrict__ csr_row_ptr,
                                       const I* __restrict__ csr_col_ind,
                                       const I* __restrict__ connections,
                                       const I* __restrict__ aggregates,
                                       J* __restrict__ row_nnz)
    {
        unsigned int lid = threadIdx.x & (WFSIZE - 1);
        unsigned int wid = threadIdx.x / WFSIZE;
        if(warpSize == WFSIZE)
        {
            wid = __builtin_amdgcn_readfirstlane(wid);
        }

        // The row this thread operates on
        I row = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // Shared memory for the unordered set
        __shared__ I stable[BLOCKSIZE / WFSIZE * HASHSIZE];

        // Each wavefront operates on its own set
        unordered_set<I, HASHSIZE, WFSIZE> set(&stable[wid * HASHSIZE]);

        // Row nnz counter
        I nnz = 0;

        // Row entry and exit points
        I row_begin = csr_row_ptr[row];
        I row_end   = csr_row_ptr[row + 1];

        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(I j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Get the column index
            I col = csr_col_ind[j];

            // Get connection state
            I con = connections[j];

            // Get aggregation state for this column
            I agg = aggregates[col];

            // When aggregate is defined, diagonal entries and connected columns will
            // generate a fill in
            if((row == col || con) && agg >= 0)
            {
                // Insert into unordered set, to discard duplicates
                nnz += set.insert(agg);
            }
        }

        // Sum up the row nnz from all lanes
        wf_reduce_sum<WFSIZE>(&nnz);

        if(lid == WFSIZE - 1)
        {
            // Write row nnz back to global memory
            row_nnz[row] = nnz;
        }
    }

    // Note: Remove in next major release
    template <unsigned int BLOCKSIZE,
              unsigned int WFSIZE,
              unsigned int HASHSIZE,
              typename T,
              typename I,
              typename J>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_sa_prolong_fill(I   nrow,
                                        T   relax,
                                        int lumping_strat,
                                        const J* __restrict__ csr_row_ptr,
                                        const I* __restrict__ csr_col_ind,
                                        const T* __restrict__ csr_val,
                                        const I* __restrict__ connections,
                                        const I* __restrict__ aggregates,
                                        const J* __restrict__ csr_row_ptr_P,
                                        I* __restrict__ csr_col_ind_P,
                                        T* __restrict__ csr_val_P)
    {
        unsigned int lid = threadIdx.x & (WFSIZE - 1);
        unsigned int wid = threadIdx.x / WFSIZE;
        if(warpSize == WFSIZE)
        {
            wid = __builtin_amdgcn_readfirstlane(wid);
        }
        // The row this thread operates on
        I row = blockIdx.x * BLOCKSIZE / WFSIZE + wid;
        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // Shared memory for the unordered map
        __shared__ I    stable[(BLOCKSIZE / WFSIZE) * HASHSIZE];
        __shared__ char smem[(BLOCKSIZE / WFSIZE) * HASHSIZE * sizeof(T)];

        T* sdata = reinterpret_cast<T*>(smem);

        // Each wavefront operates on its own map
        unordered_map<I, T, HASHSIZE, WFSIZE> map(&stable[wid * HASHSIZE], &sdata[wid * HASHSIZE]);

        // Row entry and exit points
        J row_begin = csr_row_ptr[row];
        J row_end   = csr_row_ptr[row + 1];

        // Accumulator
        T dia = static_cast<T>(0);
        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(J j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Get the column index
            I col_j = csr_col_ind[j];

            // Get the value
            T val_j = csr_val[j];

            // Add the diagonal
            if(col_j == row)
            {
                dia += val_j;
            }
            else if(connections[j] == false)
            {
                dia = (lumping_strat == 0) ? dia + val_j : dia - val_j;
            }
        }

        // Sum up the accumulator from all lanes
        for(unsigned int i = WFSIZE >> 1; i > 0; i >>= 1)
        {
            dia += hip_shfl_xor(dia, i);
        }

        dia = static_cast<T>(1) / dia;

        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(J j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Get the column index
            I col_j = csr_col_ind[j];

            // Get the value
            T val_j = csr_val[j];

            // Get connection state
            I con = connections[j];

            // Get aggregation state for this column
            I agg = aggregates[col_j];

            // When aggregate is defined, diagonal entries and connected columns will
            // generate a fill in
            if((row == col_j || con) && agg >= 0)
            {
                // Insert or add if already present into unordered map
                T val = (col_j == row) ? static_cast<T>(1) - relax : -relax * dia * val_j;

                map.insert_or_add(agg, val);
            }
        }

        // Access into P
        J aj = csr_row_ptr_P[row];

        // Store key val pairs from map into P
        map.store_sorted(&csr_col_ind_P[aj], &csr_val_P[aj]);
    }

    // Note: Remove in next major release
    template <unsigned int BLOCKSIZE, typename I, typename J>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_unsmoothed_prolong_nnz_per_row(I nrow,
                                                       const I* __restrict__ aggregates,
                                                       J* __restrict__ prolong_row_offset)
    {
        I tid = threadIdx.x;
        I row = blockIdx.x * BLOCKSIZE + tid;

        if(row >= nrow)
        {
            return;
        }

        if(aggregates[row] >= 0)
        {
            prolong_row_offset[row + 1] = 1;
        }
        else
        {
            prolong_row_offset[row + 1] = 0;
        }

        if(row == 0)
        {
            prolong_row_offset[row] = 0;
        }
    }

    // Note: Remove in next major release
    template <unsigned int BLOCKSIZE, typename T, typename I>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_unsmoothed_prolong_fill_simple(I nrow,
                                                       const I* __restrict__ aggregates,
                                                       I* __restrict__ prolong_cols,
                                                       T* __restrict__ prolong_vals)
    {

        I tid = threadIdx.x;
        I row = blockIdx.x * BLOCKSIZE + tid;

        if(row >= nrow)
        {
            return;
        }

        prolong_cols[row] = aggregates[row];
        prolong_vals[row] = static_cast<T>(1);
    }

    // Note: Remove in next major release
    template <unsigned int BLOCKSIZE, typename T, typename I, typename J>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_unsmoothed_prolong_fill(I nrow,
                                                const I* __restrict__ aggregates,
                                                const J* __restrict__ prolong_row_offset,
                                                I* __restrict__ prolong_cols,
                                                T* __restrict__ prolong_vals)
    {

        I tid = threadIdx.x;
        I row = blockIdx.x * BLOCKSIZE + tid;

        if(row >= nrow)
        {
            return;
        }

        I agg = aggregates[row];

        if(agg >= 0)
        {
            J j = prolong_row_offset[row];

            prolong_cols[j] = agg;
            prolong_vals[j] = static_cast<T>(1);
        }
    }

    // AMG Connect
    template <bool GLOBAL, unsigned int WF_SIZE, typename T, typename I, typename J>
    __global__ void kernel_csr_amg_connect(I       nrow,
                                           int64_t nnz,
                                           T       eps2,
                                           const J* __restrict__ row_offset,
                                           const I* __restrict__ col,
                                           const T* __restrict__ val,
                                           const J* __restrict__ gst_row_offset,
                                           const I* __restrict__ gst_col,
                                           const T* __restrict__ gst_val,
                                           const T* __restrict__ diag,
                                           const int64_t* __restrict__ l2g,
                                           bool* __restrict__ connections)
    {
        I tid = threadIdx.x;
        I gid = blockIdx.x * blockDim.x + tid;
        I lid = tid & (WF_SIZE - 1);
        I row = gid / WF_SIZE;

        if(row >= nrow)
        {
            return;
        }

        T eps_diag_r = eps2 * diag[row];

        J row_begin = row_offset[row];
        J row_end   = row_offset[row + 1];

        for(J i = row_begin + lid; i < row_end; i += WF_SIZE)
        {
            I c = col[i];
            T v = val[i];

            connections[i] = (c != row) && (hip_real(v * v) > hip_real(eps_diag_r * diag[c]));
        }

        if(GLOBAL)
        {
            J gst_row_begin = gst_row_offset[row];
            J gst_row_end   = gst_row_offset[row + 1];

            for(J i = gst_row_begin + lid; i < gst_row_end; i += WF_SIZE)
            {
                I c = gst_col[i];
                T v = gst_val[i];

                connections[i + nnz] = (hip_real(v * v) > hip_real(eps_diag_r * diag[c + nrow]));
            }
        }
    }

    // AMG Aggregate
    __device__ __forceinline__ unsigned int hash1(unsigned int x)
    {
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = (x >> 16) ^ x;
        return x / 2;
    }

    template <bool GLOBAL, typename I, typename J>
    __global__ void kernel_csr_amg_init_mis_tuples(int64_t global_column_begin,
                                                   I       nrow,
                                                   int64_t nnz,
                                                   const J* __restrict__ row_offset,
                                                   const J* __restrict__ gst_row_offset,
                                                   const bool* __restrict__ connections,
                                                   int* __restrict__ state,
                                                   int* __restrict__ hash)
    {
        I row = blockIdx.x * blockDim.x + threadIdx.x;

        if(row >= nrow)
        {
            return;
        }

        int s = -2;

        J row_begin = row_offset[row];
        J row_end   = row_offset[row + 1];

        for(J j = row_begin; j < row_end; j++)
        {
            if(connections[j] == true)
            {
                s = 0;
                break;
            }
        }

        if(GLOBAL == true)
        {
            J gst_row_begin = gst_row_offset[row];
            J gst_row_end   = gst_row_offset[row + 1];

            for(J j = gst_row_begin; j < gst_row_end; ++j)
            {
                if(connections[j + nnz] == true)
                {
                    s = 0;
                    break;
                }
            }
        }

        state[row] = s;
        hash[row]  = hash1(row + global_column_begin);
    }

    template <bool GLOBAL, unsigned int BLOCKSIZE, unsigned int WFSIZE, typename I, typename J>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_amg_find_max_node(I       nrow,
                                          int64_t nnz,
                                          int64_t global_column_begin,
                                          int64_t global_column_end,
                                          const J* __restrict__ row_offset,
                                          const I* __restrict__ cols,
                                          const J* __restrict__ gst_row_offset,
                                          const I* __restrict__ gst_cols,
                                          const bool* __restrict__ connections,
                                          const I* __restrict__ state,
                                          const I* __restrict__ hash,
                                          const J* __restrict__ bnd_row_offset,
                                          const int64_t* __restrict__ bnd_cols,
                                          const I* __restrict__ bnd_state,
                                          const I* __restrict__ bnd_hash,
                                          I* __restrict__ max_state,
                                          int64_t* __restrict__ aggregates,
                                          bool* undecided)
    {
        I tid = threadIdx.x;
        I gid = blockIdx.x * BLOCKSIZE + tid;
        I lid = tid & (WFSIZE - 1);
        I wid = tid / WFSIZE;
        I row = gid / WFSIZE;

        __shared__ mis_tuple smax_tuple[BLOCKSIZE];

        if(row >= nrow)
        {
            return;
        }

        smax_tuple[WFSIZE * wid].s = state[row];
        smax_tuple[WFSIZE * wid].v = hash[row];
        smax_tuple[WFSIZE * wid].i = row;

        __syncthreads();

        mis_tuple t_max = smax_tuple[WFSIZE * wid];

        // Find distance one max node
        J row_begin = row_offset[row];
        J row_end   = row_offset[row + 1];

        for(J j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            if(connections[j] == true)
            {
                I col = cols[j];

                mis_tuple tj;
                tj.s = state[col];
                tj.v = hash[col];
                tj.i = col;

                t_max = lexographical_max(&tj, &t_max);
            }
        }

        if(GLOBAL)
        {
            J gst_row_begin = gst_row_offset[row];
            J gst_row_end   = gst_row_offset[row + 1];

            for(J j = gst_row_begin + lid; j < gst_row_end; j += WFSIZE)
            {
                if(connections[j + nnz] == true)
                {
                    I col = gst_cols[j];

                    mis_tuple tj;
                    tj.s = state[nrow + col];
                    tj.v = hash[nrow + col];
                    tj.i = nrow + col;

                    t_max = lexographical_max(&tj, &t_max);
                }
            }
        }

        smax_tuple[WFSIZE * wid + lid] = t_max;

        __syncthreads();

        // Finish reducing the intra block lexographical max
        if(WFSIZE >= 64)
        {
            if(lid < 32)
                smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                    &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 32]);
            __threadfence_block();
        }
        if(WFSIZE >= 32)
        {
            if(lid < 16)
                smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                    &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 16]);
            __threadfence_block();
        }
        if(WFSIZE >= 16)
        {
            if(lid < 8)
                smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                    &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 8]);
            __threadfence_block();
        }
        if(WFSIZE >= 8)
        {
            if(lid < 4)
                smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                    &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 4]);
            __threadfence_block();
        }
        if(WFSIZE >= 4)
        {
            if(lid < 2)
                smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                    &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 2]);
            __threadfence_block();
        }
        if(WFSIZE >= 2)
        {
            if(lid < 1)
                smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                    &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 1]);
            __threadfence_block();
        }

        __syncthreads();
        t_max = smax_tuple[WFSIZE * wid];
        __syncthreads();

        // Find distance two max node
        if(t_max.i < nrow)
        {
            I r = t_max.i;

            row_begin = row_offset[r];
            row_end   = row_offset[r + 1];

            for(J j = row_begin + lid; j < row_end; j += WFSIZE)
            {
                if(connections[j] == true)
                {
                    I col = cols[j];

                    mis_tuple tj;
                    tj.s = state[col];
                    tj.v = hash[col];
                    tj.i = col;

                    t_max = lexographical_max(&tj, &t_max);
                }
            }

            if(GLOBAL)
            {
                J gst_row_begin = gst_row_offset[r];
                J gst_row_end   = gst_row_offset[r + 1];

                for(J j = gst_row_begin + lid; j < gst_row_end; j += WFSIZE)
                {
                    if(connections[j + nnz] == true)
                    {
                        I col = gst_cols[j];

                        mis_tuple tj;
                        tj.s = state[nrow + col];
                        tj.v = hash[nrow + col];
                        tj.i = nrow + col;

                        t_max = lexographical_max(&tj, &t_max);
                    }
                }
            }
        }
        else
        {
            J bnd_row_begin = bnd_row_offset[t_max.i - nrow];
            J bnd_row_end   = bnd_row_offset[t_max.i - nrow + 1];

            for(J j = bnd_row_begin + lid; j < bnd_row_end; j += WFSIZE)
            {
                int64_t gcol = bnd_cols[j];

                // Differentiate between local and ghost column
                I col = -1;
                if(gcol >= global_column_begin && gcol < global_column_end)
                {
                    col = static_cast<I>(gcol - global_column_begin);
                }

                mis_tuple tj;
                tj.s = bnd_state[j];
                tj.v = bnd_hash[j];
                tj.i = col;

                t_max = lexographical_max(&tj, &t_max);
            }
        }

        smax_tuple[WFSIZE * wid + lid] = t_max;

        __syncthreads();

        // Finish reducing the intra block lexographical max
        if(WFSIZE >= 64)
        {
            if(lid < 32)
                smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                    &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 32]);
            __threadfence_block();
        }
        if(WFSIZE >= 32)
        {
            if(lid < 16)
                smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                    &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 16]);
            __threadfence_block();
        }
        if(WFSIZE >= 16)
        {
            if(lid < 8)
                smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                    &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 8]);
            __threadfence_block();
        }
        if(WFSIZE >= 8)
        {
            if(lid < 4)
                smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                    &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 4]);
            __threadfence_block();
        }
        if(WFSIZE >= 4)
        {
            if(lid < 2)
                smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                    &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 2]);
            __threadfence_block();
        }
        if(WFSIZE >= 2)
        {
            if(lid < 1)
                smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                    &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 1]);
            __threadfence_block();
        }

        __syncthreads();
        t_max = smax_tuple[WFSIZE * wid];
        __syncthreads();

        if(lid == 0)
        {
            if(state[row] == 0)
            {
                if(t_max.i == row)
                {
                    max_state[row]  = 1;
                    aggregates[row] = 1;
                }
                else if(t_max.s == 1)
                {
                    max_state[row]  = -1;
                    aggregates[row] = 0;
                }
                else
                {
                    *undecided = true;
                }
            }
        }
    }

    template <bool GLOBAL, typename I, typename J>
    __global__ void kernel_csr_amg_update_aggregates(I       nrow,
                                                     int64_t nnz,
                                                     int64_t global_column_begin,
                                                     const J* __restrict__ row_offset,
                                                     const I* __restrict__ cols,
                                                     const J* __restrict__ gst_row_offset,
                                                     const I* __restrict__ gst_cols,
                                                     const bool* __restrict__ connections,
                                                     const I* __restrict__ state,
                                                     const int64_t* __restrict__ l2g,
                                                     I* __restrict__ max_state,
                                                     int64_t* __restrict__ aggregates,
                                                     int64_t* __restrict__ aggregate_root_nodes)
    {
        I row = blockIdx.x * blockDim.x + threadIdx.x;

        if(row >= nrow)
        {
            return;
        }

        I s = state[row];
        if(s == -1)
        {
            J row_begin = row_offset[row];
            J row_end   = row_offset[row + 1];

            int64_t global_col = -1;
            for(J j = row_begin; j < row_end; j++)
            {
                if(connections[j] == true)
                {
                    I col = cols[j];

                    if(state[col] == 1)
                    {
                        aggregates[row]           = aggregates[col];
                        aggregate_root_nodes[row] = aggregate_root_nodes[col];
                        max_state[row]            = 1;
                        global_col                = global_column_begin + col;
                        break;
                    }
                }
            }
            if(GLOBAL)
            {
                J gst_row_begin = gst_row_offset[row];
                J gst_row_end   = gst_row_offset[row + 1];
                for(J j = gst_row_begin; j < gst_row_end; j++)
                {
                    if(connections[j + nnz] == true)
                    {
                        I col = gst_cols[j];
                        if(state[col + nrow] == 1)
                        {
                            if(global_col == -1 || (global_col >= 0 && l2g[col] < global_col))
                            {
                                aggregates[row]           = aggregates[col + nrow];
                                aggregate_root_nodes[row] = aggregate_root_nodes[col + nrow];
                                max_state[row]            = 1;
                                break;
                            }
                        }
                    }
                }
            }
        }
        else if(s == -2)
        {
            aggregates[row] = -2;
        }
    }

    template <typename I>
    __global__ void
        kernel_csr_amg_initialize_aggregate_nodes(I       nrow,
                                                  I       agg_size,
                                                  int64_t global_column_begin,
                                                  const int64_t* __restrict__ aggregates,
                                                  int64_t* __restrict__ aggregate_root_nodes)
    {
        I row = blockIdx.x * blockDim.x + threadIdx.x;

        if(row >= nrow)
        {
            return;
        }

        aggregate_root_nodes[row] = (aggregates[row] == 1) ? global_column_begin + row : -1;
    }

    // // AMGSmoothedAggregation
    // template <unsigned int BLOCKSIZE,
    //           unsigned int WFSIZE,
    //           unsigned int HASHSIZE,
    //           typename I,
    //           typename J>
    // __launch_bounds__(BLOCKSIZE) __global__
    //     void kernel_csr_sa_prolong_nnz(I nrow,
    //                                    const J* __restrict__ row_offsets,
    //                                    const I* __restrict__ cols,
    //                                    const bool* __restrict__ connections,
    //                                    const int64_t* __restrict__ aggregates,
    //                                    J* __restrict__ pi_row_nnz)
    // {
    //     unsigned int lid = threadIdx.x & (WFSIZE - 1);
    //     unsigned int wid = threadIdx.x / WFSIZE;

    //     if(warpSize == WFSIZE)
    //     {
    //         wid = __builtin_amdgcn_readfirstlane(wid);
    //     }

    //     // The row this thread operates on
    //     I row = blockIdx.x * (BLOCKSIZE / WFSIZE) + wid;

    //     // Do not run out of bounds
    //     if(row >= nrow)
    //     {
    //         return;
    //     }

    //     // Shared memory for the unordered set
    //     __shared__ int64_t int_table[BLOCKSIZE / WFSIZE * HASHSIZE];

    //     // Each wavefront operates on its own set
    //     unordered_set<int64_t, HASHSIZE, WFSIZE> int_set(&int_table[wid * HASHSIZE]);

    //     // Row nnz counter
    //     J pi_nnz = 0;
    //     J pg_nnz = 0;

    //     // Row entry and exit points
    //     J row_begin = row_offsets[row];
    //     J row_end   = row_offsets[row + 1];

    //     // Loop over all columns of the i-th row, whereas each lane processes a column
    //     for(J j = row_begin + lid; j < row_end; j += WFSIZE)
    //     {
    //         // Get the column index
    //         I col = cols[j];

    //         // Get connection state
    //         bool con = connections[j];

    //         // Get aggregation state for this column
    //         int64_t agg = aggregates[col];

    //         // When aggregate is defined, diagonal entries and connected columns will
    //         // generate a fill in
    //         if((row == col || con) && agg >= 0)
    //         {
    //             // Insert into unordered int_set, to discard duplicates
    //             pi_nnz += int_set.insert(agg);
    //         }
    //     }

    //     // Sum up the row nnz from all lanes
    //     wf_reduce_sum<WFSIZE>(&pi_nnz);

    //     if(lid == WFSIZE - 1)
    //     {
    //         // Write row pi_nnz back to global memory
    //         pi_row_nnz[row] = pi_nnz;
    //     }
    // }

    template <bool         GLOBAL,
              unsigned int BLOCKSIZE,
              unsigned int WFSIZE,
              unsigned int HASHSIZE,
              typename I,
              typename J>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_sa_prolong_nnz(I       nrow,
                                       int64_t nnz,
                                       int64_t global_column_begin,
                                       int64_t global_column_end,
                                       const J* __restrict__ row_offsets,
                                       const I* __restrict__ cols,
                                       const J* __restrict__ gst_row_offsets,
                                       const I* __restrict__ gst_cols,
                                       const bool* __restrict__ connections,
                                       const int64_t* __restrict__ aggregates,
                                       const int64_t* __restrict__ aggregate_root_nodes,
                                       int* f2c,
                                       J* __restrict__ pi_row_nnz,
                                       J* __restrict__ pg_row_nnz)
    {
        unsigned int lid = threadIdx.x & (WFSIZE - 1);
        unsigned int wid = threadIdx.x / WFSIZE;

        if(warpSize == WFSIZE)
        {
            wid = __builtin_amdgcn_readfirstlane(wid);
        }

        // The row this thread operates on
        I row = blockIdx.x * (BLOCKSIZE / WFSIZE) + wid;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // Shared memory for the unordered set
        __shared__ int64_t int_table[BLOCKSIZE / WFSIZE * HASHSIZE];
        __shared__ int64_t gst_table[BLOCKSIZE / WFSIZE * HASHSIZE];

        // Each wavefront operates on its own set
        unordered_set<int64_t, HASHSIZE, WFSIZE> int_set(&int_table[wid * HASHSIZE]);
        unordered_set<int64_t, HASHSIZE, WFSIZE> gst_set(&gst_table[wid * HASHSIZE]);

        // Row nnz counter
        J pi_nnz = 0;
        J pg_nnz = 0;

        // Row entry and exit points
        J row_begin = row_offsets[row];
        J row_end   = row_offsets[row + 1];

        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(J j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Get the column index
            I col = cols[j];

            // Get connection state
            bool con = connections[j];

            // Get aggregation state for this column
            int64_t agg = aggregates[col];

            // When aggregate is defined, diagonal entries and connected columns will
            // generate a fill in
            if((row == col || con) && agg >= 0)
            {
                int64_t global_node = aggregate_root_nodes[col];

                if(global_node >= global_column_begin && global_node < global_column_end)
                {
                    // Insert into unordered int_set, to discard duplicates
                    pi_nnz += int_set.insert(global_node);
                    f2c[global_node - global_column_begin] = 1;
                }
                else
                {
                    pg_nnz += gst_set.insert(global_node);
                }
            }
        }
        if(GLOBAL)
        {
            J gst_row_begin = gst_row_offsets[row];
            J gst_row_end   = gst_row_offsets[row + 1];
            for(J j = gst_row_begin + lid; j < gst_row_end; j += WFSIZE)
            {
                // Get the column index
                I col = gst_cols[j];

                // Get connection state
                bool con = connections[j + nnz];

                // Get aggregation state for this column
                int64_t agg = aggregates[col + nrow];

                if(con && agg >= 0)
                {
                    int64_t global_node = aggregate_root_nodes[col + nrow];

                    if(global_node >= global_column_begin && global_node < global_column_end)
                    {
                        // Insert into unordered int_set, to discard duplicates
                        pi_nnz += int_set.insert(global_node);
                        f2c[global_node - global_column_begin] = 1;
                    }
                    else
                    {
                        pg_nnz += gst_set.insert(global_node);
                    }
                }
            }
        }

        // Sum up the row nnz from all lanes
        wf_reduce_sum<WFSIZE>(&pi_nnz);

        if(lid == WFSIZE - 1)
        {
            // Write row pi_nnz back to global memory
            pi_row_nnz[row] = pi_nnz;
        }

        if(GLOBAL)
        {
            wf_reduce_sum<WFSIZE>(&pg_nnz);

            if(lid == WFSIZE - 1)
            {
                // Write row pg_nnz back to global memory
                pg_row_nnz[row] = pg_nnz;
            }
        }
    }

    // template <unsigned int BLOCKSIZE,
    //           unsigned int WFSIZE,
    //           unsigned int HASHSIZE,
    //           typename T,
    //           typename I,
    //           typename J>
    // __launch_bounds__(BLOCKSIZE) __global__
    //     void kernel_csr_sa_prolong_fill(I   nrow,
    //                                     T   relax,
    //                                     int lumping_strat,
    //                                     const J* __restrict__ row_offsets,
    //                                     const I* __restrict__ cols,
    //                                     const T* __restrict__ vals,
    //                                     const bool* __restrict__ connections,
    //                                     const int64_t* __restrict__ aggregates,
    //                                     const J* __restrict__ pi_row_offsets,
    //                                     I* __restrict__ pi_cols,
    //                                     T* __restrict__ pi_vals)
    // {
    //     unsigned int lid = threadIdx.x & (WFSIZE - 1);
    //     unsigned int wid = threadIdx.x / WFSIZE;

    //     if(warpSize == WFSIZE)
    //     {
    //         wid = __builtin_amdgcn_readfirstlane(wid);
    //     }

    //     // The row this thread operates on
    //     I row = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

    //     // Do not run out of bounds
    //     if(row >= nrow)
    //     {
    //         return;
    //     }

    //     // Shared memory for the unordered map
    //     __shared__ int64_t int_table[(BLOCKSIZE / WFSIZE) * HASHSIZE];
    //     __shared__ char    int_mem[(BLOCKSIZE / WFSIZE) * HASHSIZE * sizeof(T)];

    //     T* int_data = reinterpret_cast<T*>(int_mem);

    //     // Each wavefront operates on its own map
    //     unordered_map<int64_t, T, HASHSIZE, WFSIZE>   int_map(&int_table[wid * HASHSIZE],
    //                                                   &int_data[wid * HASHSIZE]);

    //     // Row entry and exit points
    //     J row_begin = row_offsets[row];
    //     J row_end   = row_offsets[row + 1];

    //     // Accumulator
    //     T dia = static_cast<T>(0);

    //     // Loop over all columns of the i-th row, whereas each lane processes a column
    //     for(J j = row_begin + lid; j < row_end; j += WFSIZE)
    //     {
    //         // Get the column index
    //         I col_j = cols[j];

    //         // Get the value
    //         T val_j = vals[j];

    //         // Add the diagonal
    //         if(col_j == row)
    //         {
    //             dia += val_j;
    //         }
    //         else if(connections[j] == false)
    //         {
    //             dia = (lumping_strat == 0) ? dia + val_j : dia - val_j;
    //         }
    //     }

    //     // Sum up the accumulator from all lanes
    //     for(unsigned int i = WFSIZE >> 1; i > 0; i >>= 1)
    //     {
    //         dia += hip_shfl_xor(dia, i);
    //     }

    //     dia = static_cast<T>(1) / dia;

    //     // Loop over all columns of the i-th row, whereas each lane processes a column
    //     for(J j = row_begin + lid; j < row_end; j += WFSIZE)
    //     {
    //         // Get the column index
    //         I col_j = cols[j];

    //         // Get the value
    //         T val_j = vals[j];

    //         // Get connection state
    //         bool con = connections[j];

    //         // Get aggregation state for this column
    //         int64_t agg = aggregates[col_j];

    //         // When aggregate is defined, diagonal entries and connected columns will
    //         // generate a fill in
    //         if((row == col_j || con) && agg >= 0)
    //         {
    //             // Insert or add if already present into unordered map
    //             T val = (col_j == row) ? static_cast<T>(1) - relax : -relax * dia * val_j;

    //             int_map.insert_or_add(agg, val);
    //         }
    //     }

    //     // Access into P
    //     J aj = pi_row_offsets[row];

    //     // Store key val pairs from map into P
    //     int_map.store_sorted(&pi_cols[aj], &pi_vals[aj]);
    // }

    template <bool         GLOBAL,
              unsigned int BLOCKSIZE,
              unsigned int WFSIZE,
              unsigned int HASHSIZE,
              typename T,
              typename I,
              typename J>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_sa_prolong_fill(I       nrow,
                                        int64_t nnz,
                                        int64_t global_column_begin,
                                        int64_t global_column_end,
                                        T       relax,
                                        int     lumping_strat,
                                        const J* __restrict__ row_offsets,
                                        const I* __restrict__ cols,
                                        const T* __restrict__ vals,
                                        const J* __restrict__ gst_row_offsets,
                                        const I* __restrict__ gst_cols,
                                        const T* __restrict__ gst_vals,
                                        const bool* __restrict__ connections,
                                        const int64_t* __restrict__ aggregates,
                                        const int64_t* __restrict__ aggregate_root_nodes,
                                        const int* __restrict__ f2c,
                                        const J* __restrict__ pi_row_offsets,
                                        I* __restrict__ pi_cols,
                                        T* __restrict__ pi_vals,
                                        const J* __restrict__ pg_row_offsets,
                                        I* __restrict__ pg_cols,
                                        T* __restrict__ pg_vals,
                                        int64_t* __restrict__ global_ghost_col)
    {
        unsigned int lid = threadIdx.x & (WFSIZE - 1);
        unsigned int wid = threadIdx.x / WFSIZE;

        if(warpSize == WFSIZE)
        {
            wid = __builtin_amdgcn_readfirstlane(wid);
        }

        // The row this thread operates on
        I row = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // Shared memory for the unordered map
        __shared__ int64_t int_table[(BLOCKSIZE / WFSIZE) * HASHSIZE];
        __shared__ int64_t gst_table[(BLOCKSIZE / WFSIZE) * HASHSIZE];
        __shared__ char    int_mem[(BLOCKSIZE / WFSIZE) * HASHSIZE * sizeof(T)];
        __shared__ char    gst_mem[(BLOCKSIZE / WFSIZE) * HASHSIZE * sizeof(T)];

        T* int_data = reinterpret_cast<T*>(int_mem);
        T* gst_data = reinterpret_cast<T*>(gst_mem);

        // Each wavefront operates on its own map
        unordered_map<int64_t, T, HASHSIZE, WFSIZE> int_map(&int_table[wid * HASHSIZE],
                                                            &int_data[wid * HASHSIZE]);
        unordered_map<int64_t, T, HASHSIZE, WFSIZE> gst_map(&gst_table[wid * HASHSIZE],
                                                            &gst_data[wid * HASHSIZE]);

        // Row entry and exit points
        J row_begin = row_offsets[row];
        J row_end   = row_offsets[row + 1];

        // Accumulator
        T dia = static_cast<T>(0);

        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(J j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Get the column index
            I col_j = cols[j];

            // Get the value
            T val_j = vals[j];

            // Add the diagonal
            if(col_j == row)
            {
                dia += val_j;
            }
            else if(connections[j] == false)
            {
                dia = (lumping_strat == 0) ? dia + val_j : dia - val_j;
            }
        }

        if(GLOBAL)
        {
            // Row entry and exit points
            J gst_row_begin = gst_row_offsets[row];
            J gst_row_end   = gst_row_offsets[row + 1];

            for(J j = gst_row_begin + lid; j < gst_row_end; j += WFSIZE)
            {
                if(connections[j + nnz] == false)
                {
                    if(lumping_strat == 0)
                    {
                        dia += gst_vals[j];
                    }
                    else
                    {
                        dia -= gst_vals[j];
                    }
                }
            }
        }

        // Sum up the accumulator from all lanes
        for(unsigned int i = WFSIZE >> 1; i > 0; i >>= 1)
        {
            dia += hip_shfl_xor(dia, i);
        }

        dia = static_cast<T>(1) / dia;

        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(J j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Get the column index
            I col_j = cols[j];

            // Get the value
            T val_j = vals[j];

            // Get connection state
            bool con = connections[j];

            // Get aggregation state for this column
            int64_t agg = aggregates[col_j];

            // When aggregate is defined, diagonal entries and connected columns will
            // generate a fill in
            if((row == col_j || con) && agg >= 0)
            {
                // Insert or add if already present into unordered map
                T val = (col_j == row) ? static_cast<T>(1) - relax : -relax * dia * val_j;

                int64_t global_node = aggregate_root_nodes[col_j];

                if(global_node >= global_column_begin && global_node < global_column_end)
                {
                    // int_map.insert_or_add(agg, val);
                    // int_map.insert_or_add(f2c[global_node - global_column_begin], val);
                    int_map.insert_or_add(global_node, val);
                }
                else
                {
                    gst_map.insert_or_add(global_node, val);
                }
            }
        }

        if(GLOBAL)
        {
            // Row entry and exit points
            J gst_row_begin = gst_row_offsets[row];
            J gst_row_end   = gst_row_offsets[row + 1];

            for(J j = gst_row_begin + lid; j < gst_row_end; j += WFSIZE)
            {
                // Get the column index
                I col_j = gst_cols[j];

                // Get the value
                T val_j = gst_vals[j];

                // Get connection state
                bool con = connections[j + nnz];

                // Get aggregation state for this column
                int64_t agg = aggregates[col_j + nrow];

                // When aggregate is defined, diagonal entries and connected columns will
                // generate a fill in
                if(con && agg >= 0)
                {
                    // Insert or add if already present into unordered map
                    T val = -relax * dia * val_j;

                    int64_t global_node = aggregate_root_nodes[col_j + nrow];

                    if(global_node >= global_column_begin && global_node < global_column_end)
                    {
                        // int_map.insert_or_add(agg, val);
                        // int_map.insert_or_add(f2c[global_node - global_column_begin], val);
                        int_map.insert_or_add(global_node, val);
                    }
                    else
                    {
                        gst_map.insert_or_add(global_node, val);
                    }
                }
            }

            // Access into P
            J aj = pg_row_offsets[row];

            // Store key val pairs from map into P
            gst_map.store_sorted(&global_ghost_col[aj], &pg_vals[aj]);
        }

        // Access into P
        J aj   = pi_row_offsets[row];
        J ajp1 = pi_row_offsets[row + 1];

        // Store key val pairs from map into P
        //int_map.store_sorted(&pi_cols[aj], &pi_vals[aj]);
        int_map.sort();

        for(J i = aj + lid; i < ajp1; i += WFSIZE)
        {
            pi_cols[i] = f2c[int_map.get_key(i - aj) - global_column_begin];
            pi_vals[i] = int_map.get_val(i - aj);
        }
    }

    // AMGUnsmoothedAggregation
    template <unsigned int BLOCKSIZE, typename I, typename J>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_unsmoothed_prolong_nnz_per_row(I nrow,
                                                       const int64_t* __restrict__ aggregates,
                                                       J* __restrict__ prolong_row_offset)
    {
        I tid = threadIdx.x;
        I row = blockIdx.x * BLOCKSIZE + tid;
        if(row >= nrow)
        {
            return;
        }
        if(aggregates[row] >= 0)
        {
            prolong_row_offset[row + 1] = 1;
        }
        else
        {
            prolong_row_offset[row + 1] = 0;
        }
        if(row == 0)
        {
            prolong_row_offset[row] = 0;
        }
    }

    template <unsigned int BLOCKSIZE, typename T, typename I>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_unsmoothed_prolong_fill_simple(I nrow,
                                                       const int64_t* __restrict__ aggregates,
                                                       I* __restrict__ prolong_cols,
                                                       T* __restrict__ prolong_vals)
    {
        I tid = threadIdx.x;
        I row = blockIdx.x * BLOCKSIZE + tid;
        if(row >= nrow)
        {
            return;
        }
        prolong_cols[row] = aggregates[row];
        prolong_vals[row] = static_cast<T>(1);
    }

    template <unsigned int BLOCKSIZE, typename T, typename I, typename J>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_unsmoothed_prolong_fill(I nrow,
                                                const int64_t* __restrict__ aggregates,
                                                const J* __restrict__ prolong_row_offset,
                                                I* __restrict__ prolong_cols,
                                                T* __restrict__ prolong_vals)
    {
        I tid = threadIdx.x;
        I row = blockIdx.x * BLOCKSIZE + tid;
        if(row >= nrow)
        {
            return;
        }

        int64_t agg = aggregates[row];

        if(agg >= 0)
        {
            J j             = prolong_row_offset[row];
            prolong_cols[j] = agg;
            prolong_vals[j] = static_cast<T>(1);
        }
    }

    template <unsigned int BLOCKSIZE, typename I, typename J>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_ua_prolong_nnz(I       nrow,
                                       int64_t global_column_begin,
                                       int64_t global_column_end,
                                       const int64_t* __restrict__ aggregates,
                                       const int64_t* __restrict__ aggregate_root_nodes,
                                       int* __restrict__ f2c,
                                       J* __restrict__ pi_row_nnz,
                                       J* __restrict__ pg_row_nnz)
    {
        I tid = threadIdx.x;
        I row = blockIdx.x * BLOCKSIZE + tid;

        if(row >= nrow)
        {
            return;
        }

        int64_t agg = aggregates[row];

        if(agg >= 0)
        {
            int64_t global_node = aggregate_root_nodes[row];

            if(global_node >= global_column_begin && global_node < global_column_end)
            {
                pi_row_nnz[row]                        = 1;
                f2c[global_node - global_column_begin] = 1;
            }
            else
            {
                pg_row_nnz[row] = 1;
            }
        }
    }

    template <bool GLOBAL, unsigned int BLOCKSIZE, typename T, typename I, typename J>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_ua_prolong_fill(I       nrow,
                                        int64_t global_column_begin,
                                        int64_t global_column_end,
                                        const int64_t* __restrict__ aggregates,
                                        const int64_t* __restrict__ aggregate_root_nodes,
                                        const int* __restrict__ f2c,
                                        const J* __restrict__ pi_row_offsets,
                                        I* __restrict__ pi_cols,
                                        T* __restrict__ pi_vals,
                                        const J* __restrict__ pg_row_offsets,
                                        I* __restrict__ pg_cols,
                                        T* __restrict__ pg_vals,
                                        int64_t* __restrict__ global_ghost_col)
    {
        I tid = threadIdx.x;
        I row = blockIdx.x * BLOCKSIZE + tid;

        if(row >= nrow)
        {
            return;
        }

        int64_t agg = aggregates[row];

        if(agg >= 0)
        {
            int64_t global_node = aggregate_root_nodes[row];

            if(global_node >= global_column_begin && global_node < global_column_end)
            {
                J pi_row_end = pi_row_offsets[row];

                pi_cols[pi_row_end] = f2c[global_node - global_column_begin];
                pi_vals[pi_row_end] = 1;
            }
            else
            {
                J pg_row_end = pg_row_offsets[row];

                pg_vals[pg_row_end]          = 1;
                global_ghost_col[pg_row_end] = global_node;
            }
        }
    }

    template <typename I, typename J>
    __global__ void kernel_csr_extract_boundary_state(I       boundary_size,
                                                      I       nrow,
                                                      int64_t nnz,
                                                      const I* __restrict__ boundary_index,
                                                      const J* __restrict__ csr_row_ptr,
                                                      const I* __restrict__ csr_col_ind,
                                                      const J* __restrict__ ghost_csr_row_ptr,
                                                      const I* __restrict__ ghost_csr_col_ind,
                                                      const bool* __restrict__ connections,
                                                      const I* __restrict__ max_state,
                                                      const I* __restrict__ hash,
                                                      const J* __restrict__ bnd_row_ptr,
                                                      I* __restrict__ bnd_max_state,
                                                      I* __restrict__ bnd_hash)
    {
        I gid = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(gid >= boundary_size)
        {
            return;
        }

        // This row is a boundary row
        I row = boundary_index[gid];

        // Index into boundary array
        J idx = bnd_row_ptr[gid];

        // Extract interior part
        J row_begin = csr_row_ptr[row];
        J row_end   = csr_row_ptr[row + 1];

        // Interior
        for(J j = row_begin; j < row_end; ++j)
        {
            if(connections[j] == true)
            {
                I col = csr_col_ind[j];

                bnd_max_state[idx] = max_state[col];
                bnd_hash[idx]      = hash[col];

                ++idx;
            }
        }

        // Extract ghost part
        row_begin = ghost_csr_row_ptr[row];
        row_end   = ghost_csr_row_ptr[row + 1];

        for(J j = row_begin; j < row_end; ++j)
        {
            if(connections[j + nnz] == true)
            {
                I col = ghost_csr_col_ind[j];

                bnd_max_state[idx] = max_state[col + nrow];
                bnd_hash[idx]      = hash[col + nrow];

                ++idx;
            }
        }
    }

    template <typename I, typename J>
    __global__ void kernel_csr_merge_interior_ghost_ext_nnz(I       m,
                                                            I       m_ext,
                                                            int64_t nnz_gst,
                                                            const J* __restrict__ int_csr_row_ptr,
                                                            const J* __restrict__ gst_csr_row_ptr,
                                                            const J* __restrict__ ext_csr_row_ptr,
                                                            J* __restrict__ row_nnz)
    {
        // Current row
        I row = blockIdx.x * blockDim.x + threadIdx.x;

        // Count row nnz
        if(row < m)
        {
            int nnz = int_csr_row_ptr[row + 1] - int_csr_row_ptr[row];

            // If ghost part is available
            if(nnz_gst > 0)
            {
                nnz += gst_csr_row_ptr[row + 1] - gst_csr_row_ptr[row];
            }

            row_nnz[row] = nnz;
        }
        else if(row - m < m_ext)
        {
            row_nnz[row] = ext_csr_row_ptr[row - m + 1] - ext_csr_row_ptr[row - m];
        }
    }

    template <typename T, typename I, typename J>
    __global__ void kernel_csr_merge_interior_ghost_nnz(I m,
                                                        I m_ext,
                                                        I n,
                                                        I nnz_gst,
                                                        const J* __restrict__ int_csr_row_ptr,
                                                        const I* __restrict__ int_csr_col_ind,
                                                        const T* __restrict__ int_csr_val,
                                                        const J* __restrict__ gst_csr_row_ptr,
                                                        const I* __restrict__ gst_csr_col_ind,
                                                        const T* __restrict__ gst_csr_val,
                                                        const J* __restrict__ ext_csr_row_ptr,
                                                        const I* __restrict__ ext_csr_col_ind,
                                                        const T* __restrict__ ext_csr_val,
                                                        const I* __restrict__ ext_gst_csr_col_ind,
                                                        const J* __restrict__ merged_csr_row_ptr,
                                                        I* __restrict__ merged_csr_col_ind,
                                                        T* __restrict__ merged_csr_val)
    {
        // Current row
        I row = blockIdx.x * blockDim.x + threadIdx.x;

        // Access into merged structure
        J idx = merged_csr_row_ptr[row];

        if(row < m)
        {
            // Interior
            J row_begin = int_csr_row_ptr[row];
            J row_end   = int_csr_row_ptr[row + 1];

            for(J j = row_begin; j < row_end; ++j)
            {
                merged_csr_col_ind[idx] = int_csr_col_ind[j];
                merged_csr_val[idx]     = int_csr_val[j];

                ++idx;
            }

            // If ghost is available
            if(nnz_gst > 0)
            {
                J row_begin = gst_csr_row_ptr[row];
                J row_end   = gst_csr_row_ptr[row + 1];

                for(J j = row_begin; j < row_end; ++j)
                {
                    merged_csr_col_ind[idx] = ext_gst_csr_col_ind[j] + n;
                    merged_csr_val[idx]     = gst_csr_val[j];

                    ++idx;
                }
            }
        }
        else if(row - m < m_ext)
        {
            J row_begin = ext_csr_row_ptr[row - m];
            J row_end   = ext_csr_row_ptr[row - m + 1];

            for(J j = row_begin; j < row_end; ++j)
            {
                merged_csr_col_ind[idx] = ext_csr_col_ind[j];
                merged_csr_val[idx]     = ext_csr_val[j];

                ++idx;
            }
        }
    }

    // Count the total vertices of this boundary row
    template <typename I, typename J>
    __global__ void kernel_csr_extract_boundary_rows_nnz(I boundary_size,
                                                         const I* __restrict__ boundary_index,
                                                         const J* __restrict__ int_csr_row_ptr,
                                                         const J* __restrict__ gst_csr_row_ptr,
                                                         I* __restrict__ row_nnz)
    {
        I gid = blockIdx.x * blockDim.x + threadIdx.x;
        // Do not run out of bounds
        if(gid >= boundary_size)
        {
            return;
        }
        // Get boundary row
        I row = boundary_index[gid];
        // Write total number of nnz for this boundary row
        row_nnz[gid] = int_csr_row_ptr[row + 1] - int_csr_row_ptr[row] + gst_csr_row_ptr[row + 1]
                       - gst_csr_row_ptr[row];
    }

    template <typename I, typename J, typename K>
    __global__ void kernel_csr_extract_boundary_rows(I boundary_size,
                                                     const I* __restrict__ boundary_index,
                                                     K global_col_offset,
                                                     const J* __restrict__ csr_row_ptr,
                                                     const I* __restrict__ csr_col_ind,
                                                     const J* __restrict__ ghost_csr_row_ptr,
                                                     const I* __restrict__ ghost_csr_col_ind,
                                                     const K* __restrict__ l2g,
                                                     const I* __restrict__ send_row_ptr,
                                                     K* __restrict__ send_col_ind)
    {
        I gid = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(gid >= boundary_size)
        {
            return;
        }

        // This row is a boundary row
        I row = boundary_index[gid];

        // Index into send array
        I send_row = send_row_ptr[gid];

        // Extract interior part
        J row_begin = csr_row_ptr[row];
        J row_end   = csr_row_ptr[row + 1];

        // Interior
        for(J j = row_begin; j < row_end; ++j)
        {
            // Shift column by global column offset, to obtain the global column index
            send_col_ind[send_row] = csr_col_ind[j] + global_col_offset;
            ++send_row;
        }

        // Extract ghost part
        row_begin = ghost_csr_row_ptr[row];
        row_end   = ghost_csr_row_ptr[row + 1];

        for(J j = row_begin; j < row_end; ++j)
        {
            // Map the local ghost column to global column
            send_col_ind[send_row] = l2g[ghost_csr_col_ind[j]];
            ++send_row;
        }
    }

    template <typename T, typename I, typename J, typename K>
    __global__ void kernel_csr_extract_boundary_rows(I boundary_size,
                                                     const I* __restrict__ boundary_index,
                                                     K global_col_offset,
                                                     const J* __restrict__ csr_row_ptr,
                                                     const I* __restrict__ csr_col_ind,
                                                     const T* __restrict__ csr_val,
                                                     const J* __restrict__ ghost_csr_row_ptr,
                                                     const I* __restrict__ ghost_csr_col_ind,
                                                     const T* __restrict__ ghost_csr_val,
                                                     const K* __restrict__ l2g,
                                                     const I* __restrict__ send_row_ptr,
                                                     K* __restrict__ send_col_ind,
                                                     T* __restrict__ send_val)
    {
        I gid = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(gid >= boundary_size)
        {
            return;
        }

        // This row is a boundary row
        I row = boundary_index[gid];

        // Index into send array
        I send_row = send_row_ptr[gid];

        // Extract interior part
        J row_begin = csr_row_ptr[row];
        J row_end   = csr_row_ptr[row + 1];

        // Interior
        for(J j = row_begin; j < row_end; ++j)
        {
            // Shift column by global column offset, to obtain the global column index
            send_col_ind[send_row] = csr_col_ind[j] + global_col_offset;
            send_val[send_row]     = csr_val[j];
            ++send_row;
        }

        // Extract ghost part
        row_begin = ghost_csr_row_ptr[row];
        row_end   = ghost_csr_row_ptr[row + 1];

        for(J j = row_begin; j < row_end; ++j)
        {
            // Map the local ghost column to global column
            send_col_ind[send_row] = l2g[ghost_csr_col_ind[j]];
            send_val[send_row]     = ghost_csr_val[j];
            ++send_row;
        }
    }

    template <typename I, typename J>
    __global__ void kernel_csr_copy_ghost_from_global_nnz(I boundary_size,
                                                          const I* __restrict__ boundary_index,
                                                          const I* __restrict__ csr_row_ptr,
                                                          J* __restrict__ row_nnz)
    {
        I gid = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(gid >= boundary_size)
        {
            return;
        }

        // Get boundary row
        I row = boundary_index[gid];

        // Write total number of nnz for this boundary row
        atomicAdd(row_nnz + row, csr_row_ptr[gid + 1] - csr_row_ptr[gid]);
    }

    template <typename T, typename I, typename J, typename K>
    __global__ void kernel_csr_copy_ghost_from_global(I boundary_size,
                                                      const I* __restrict__ boundary_index,
                                                      const I* __restrict__ csr_row_ptr,
                                                      const K* __restrict__ csr_col_ind,
                                                      const T* __restrict__ csr_val,
                                                      J* __restrict__ ext_row_ptr,
                                                      K* __restrict__ ext_col_ind,
                                                      T* __restrict__ ext_val)
    {
        I gid = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(gid >= boundary_size)
        {
            return;
        }

        // This row is a boundary row
        I row = boundary_index[gid];

        I row_begin = csr_row_ptr[gid];
        I row_end   = csr_row_ptr[gid + 1];

        // Index into extracted matrix, we have to use atomics here, because other
        // blocks might also update this row
        J idx = atomicAdd(ext_row_ptr + row, row_end - row_begin);

        for(I j = row_begin; j < row_end; ++j)
        {
            ext_col_ind[idx] = csr_col_ind[j];
            ext_val[idx]     = csr_val[j];
            ++idx;
        }
    }

    template <typename I, typename J, typename K>
    __global__ void kernel_csr_copy_from_global_nnz(I boundary_size,
                                                    K global_column_begin,
                                                    K global_column_end,
                                                    const I* __restrict__ boundary_index,
                                                    const I* __restrict__ csr_row_ptr,
                                                    const K* __restrict__ csr_col_ind,
                                                    J* __restrict__ int_row_nnz,
                                                    J* __restrict__ gst_row_nnz)
    {
        I gid = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(gid >= boundary_size)
        {
            return;
        }

        // Get boundary row
        I row = boundary_index[gid];

        I int_nnz = 0;
        I gst_nnz = 0;

        I row_begin = csr_row_ptr[gid];
        I row_end   = csr_row_ptr[gid + 1];

        for(I j = row_begin; j < row_end; ++j)
        {
            K col = csr_col_ind[j];

            if(col >= global_column_begin && col < global_column_end)
            {
                // Interior column
                ++int_nnz;
            }
            else
            {
                // Ghost column
                ++gst_nnz;
            }
        }

        // Write total number of nnz for this row
        atomicAdd(int_row_nnz + row, int_nnz);
        atomicAdd(gst_row_nnz + row, gst_nnz);
    }

    template <typename T, typename I, typename J, typename K>
    __global__ void kernel_csr_copy_from_global(I boundary_size,
                                                K global_column_begin,
                                                K global_column_end,
                                                const I* __restrict__ boundary_index,
                                                const I* __restrict__ csr_row_ptr,
                                                const K* __restrict__ csr_col_ind,
                                                const T* __restrict__ csr_val,
                                                J* __restrict__ int_csr_row_ptr,
                                                I* __restrict__ int_csr_col_ind,
                                                T* __restrict__ int_csr_val,
                                                J* __restrict__ gst_csr_row_ptr,
                                                K* __restrict__ gst_csr_col_ind,
                                                T* __restrict__ gst_csr_val)
    {
        I gid = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(gid >= boundary_size)
        {
            return;
        }

        // This row is a boundary row
        I row = boundary_index[gid];

        I row_begin = csr_row_ptr[gid];
        I row_end   = csr_row_ptr[gid + 1];

        // Pre-determine, how many nnz we are going to add for this boundary row
        I int_nnz = 0;
        I gst_nnz = 0;

        for(I j = row_begin; j < row_end; ++j)
        {
            K col = csr_col_ind[j];

            if(col >= global_column_begin && col < global_column_end)
            {
                // Interior column
                ++int_nnz;
            }
            else
            {
                // Ghost column
                ++gst_nnz;
            }
        }

        // Index into extracted matrix, we have to use atomics here, because other
        // blocks might also update this row
        I idx_int = atomicAdd(int_csr_row_ptr + row, int_nnz);
        I idx_gst = atomicAdd(gst_csr_row_ptr + row, gst_nnz);

        for(I j = row_begin; j < row_end; ++j)
        {
            K col = csr_col_ind[j];

            if(col >= global_column_begin && col < global_column_end)
            {
                // Interior column
                int_csr_col_ind[idx_int] = col - global_column_begin;
                int_csr_val[idx_int]     = csr_val[j];

                ++idx_int;
            }
            else
            {
                // Ghost column
                gst_csr_col_ind[idx_gst] = col;
                gst_csr_val[idx_gst]     = csr_val[j];

                ++idx_gst;
            }
        }
    }

    template <typename I>
    __global__ void kernel_csr_update_numbering(I size,
                                                const I* __restrict__ counts,
                                                const I* __restrict__ perm,
                                                I* __restrict__ out)
    {
        I idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(idx >= size)
        {
            return;
        }

        // Entry and exit point from run length encode
        I begin = counts[idx];
        I end   = counts[idx + 1];

        for(I i = begin; i < end; ++i)
        {
            out[perm[i]] = idx;
        }
    }

    template <typename I, typename J>
    __global__ void kernel_csr_split_interior_ghost_nnz(I m,
                                                        const J* __restrict__ csr_row_ptr,
                                                        const I* __restrict__ csr_col_ind,
                                                        J* __restrict__ int_row_nnz,
                                                        J* __restrict__ gst_row_nnz)
    {
        I row = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(row >= m)
        {
            return;
        }

        I int_nnz = 0;
        I gst_nnz = 0;

        J row_begin = csr_row_ptr[row];
        J row_end   = csr_row_ptr[row + 1];

        for(J j = row_begin; j < row_end; ++j)
        {
            I col = csr_col_ind[j];

            if(col >= m)
            {
                // This is a ghost column
                ++gst_nnz;
            }
            else
            {
                // This is an interior column
                ++int_nnz;
            }
        }

        int_row_nnz[row] = int_nnz;
        gst_row_nnz[row] = gst_nnz;
    }

    template <typename T, typename I, typename J>
    __global__ void kernel_csr_split_interior_ghost(I m,
                                                    const J* __restrict__ csr_row_ptr,
                                                    const I* __restrict__ csr_col_ind,
                                                    const T* __restrict__ csr_val,
                                                    const J* __restrict__ int_csr_row_ptr,
                                                    I* __restrict__ int_csr_col_ind,
                                                    T* __restrict__ int_csr_val,
                                                    const J* __restrict__ gst_csr_row_ptr,
                                                    I* __restrict__ gst_csr_col_ind,
                                                    T* __restrict__ gst_csr_val)
    {
        I row = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(row >= m)
        {
            return;
        }

        J row_begin = csr_row_ptr[row];
        J row_end   = csr_row_ptr[row + 1];

        J idx_int = int_csr_row_ptr[row];
        J idx_gst = gst_csr_row_ptr[row];

        for(J j = row_begin; j < row_end; ++j)
        {
            I col = csr_col_ind[j];

            if(col >= m)
            {
                // This is a ghost column
                gst_csr_col_ind[idx_gst] = col - m;
                gst_csr_val[idx_gst]     = csr_val[j];

                ++idx_gst;
            }
            else
            {
                // This is an interior column
                int_csr_col_ind[idx_int] = col;
                int_csr_val[idx_int]     = csr_val[j];

                ++idx_int;
            }
        }
    }

    template <typename I, typename K>
    __global__ void kernel_csr_extract_global_column_indices(I ncol,
                                                             I nnz,
                                                             K global_offset,
                                                             const I* __restrict__ csr_col_ind,
                                                             const K* __restrict__ l2g,
                                                             K* __restrict__ global_col)
    {
        I gid = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(gid >= nnz)
        {
            return;
        }

        int local_col = csr_col_ind[gid];

        if(local_col >= ncol)
        {
            // This is an ext column, map to global
            global_col[gid] = l2g[local_col - ncol];
        }
        else
        {
            // This is a local column, shift by offset
            global_col[gid] = local_col + global_offset;
        }
    }

    template <typename I, typename K>
    __global__ void kernel_csr_renumber_global_to_local_count(I nnz,
                                                              const K* __restrict__ global_sorted,
                                                              I* __restrict__ local_col)
    {
        I gid = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(gid >= nnz)
        {
            return;
        }

        // First column entry cannot be a duplicate
        if(gid == 0)
        {
            local_col[0] = 1;

            return;
        }

        // Get global column
        K global_col = global_sorted[gid];
        K prev_col   = global_sorted[gid - 1];

        // Compare column indices
        if(global_col == prev_col)
        {
            // Same index as previous column
            local_col[gid] = 0;
        }
        else
        {
            // New local column index
            local_col[gid] = 1;
        }
    }

    template <typename I>
    __global__ void kernel_csr_renumber_global_to_local_fill(I nnz,
                                                             const I* __restrict__ local_col,
                                                             const I* __restrict__ perm,
                                                             I* __restrict__ csr_col_ind)
    {
        I gid = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(gid >= nnz)
        {
            return;
        }

        // Back permutation into matrix
        csr_col_ind[perm[gid]] = local_col[gid] - 1;
    }

    template <typename I, typename K>
    __global__ void kernel_csr_local_to_global(I nnz,
                                               const K* __restrict__ l2g,
                                               const I* __restrict__ local,
                                               K* __restrict__ global)
    {
        I gid = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(gid >= nnz)
        {
            return;
        }

        // Convert local to global
        global[gid] = l2g[local[gid]];
    }

    template <typename I, typename K>
    __global__ void kernel_csr_ghost_columns_nnz(int64_t size,
                                                 K       global_column_begin,
                                                 K       global_column_end,
                                                 const K* __restrict__ in,
                                                 I* __restrict__ nnz)
    {
        I gid = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(gid >= size)
        {
            return;
        }

        // Get global column id
        K col = in[gid];

        // Count this column if its not interior
        nnz[gid] = (col < global_column_begin || col >= global_column_end) ? 1 : 0;
    }

    template <typename I, typename K>
    __global__ void kernel_csr_ghost_columns_fill(int64_t nnz,
                                                  K       global_column_begin,
                                                  K       global_column_end,
                                                  const K* __restrict__ in,
                                                  const I* __restrict__ nnz_ptr,
                                                  K* __restrict__ out)
    {
        I gid = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(gid >= nnz)
        {
            return;
        }

        // Get global column id
        K col = in[gid];

        // Count this column if its not interior
        if(col < global_column_begin || col >= global_column_end)
        {
            out[nnz_ptr[gid]] = col;
        }
    }

    template <typename I, typename K>
    __global__ void kernel_csr_column_id_transfer(int64_t nnz,
                                                  I       ncol,
                                                  K       global_column_begin,
                                                  K       global_column_end,
                                                  const K* __restrict__ global,
                                                  const I* __restrict__ cmb,
                                                  const I* __restrict__ nnz_ptr,
                                                  I* __restrict__ out)
    {
        I gid = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(gid >= nnz)
        {
            return;
        }

        K global_col = global[gid];

        // Filter external source to match given column id interval
        if(global_col >= global_column_begin && global_col < global_column_end)
        {
            // Map column into interval
            out[gid] = global_col - global_column_begin;
        }
        else
        {
            out[gid] = cmb[nnz_ptr[gid]] + ncol;
        }
    }

    // Count the number of nnz per row of two matrices
    template <typename I, typename J>
    __global__ void kernel_csr_combined_row_nnz(I m,
                                                const J* __restrict__ csr_row_ptr_A,
                                                const J* __restrict__ csr_row_ptr_B,
                                                J* __restrict__ row_nnz)
    {
        I row = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(row >= m)
        {
            return;
        }

        // Write combined number of row nnz for this row
        row_nnz[row] = csr_row_ptr_A[row + 1] - csr_row_ptr_A[row] + csr_row_ptr_B[row + 1]
                       - csr_row_ptr_B[row];
    }

    template <bool         GLOBAL,
              unsigned int BLOCKSIZE,
              unsigned int WFSIZE,
              unsigned int HASHSIZE,
              typename I,
              typename J,
              typename K>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_compress_add_nnz(I m,
                                         const J* __restrict__ csr_row_ptr_A,
                                         const I* __restrict__ csr_col_ind_A,
                                         const J* __restrict__ csr_row_ptr_B,
                                         const K* __restrict__ csr_col_ind_B,
                                         const K* __restrict__ l2g,
                                         J* __restrict__ row_nnz)
    {
        unsigned int lid = threadIdx.x & (WFSIZE - 1);
        unsigned int wid = threadIdx.x / WFSIZE;

        // The row this thread operates on
        I row = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= m)
        {
            return;
        }

        // Row nnz counter
        I nnz = 0;

        // Shared memory for the unordered set
        __shared__ K sdata[BLOCKSIZE / WFSIZE * HASHSIZE];

        // Each wavefront operates on its own set
        unordered_set<K, HASHSIZE, WFSIZE> set(&sdata[wid * HASHSIZE]);

        // Row entry and exit points of A
        J row_begin_A = csr_row_ptr_A[row];
        J row_end_A   = csr_row_ptr_A[row + 1];

        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(J j = row_begin_A + lid; j < row_end_A; j += WFSIZE)
        {
            // Get the column index
            K col_j = GLOBAL ? l2g[csr_col_ind_A[j]] : csr_col_ind_A[j];

            // Add column to the set
            nnz += set.insert(col_j);
        }

        // Row entry and exit points of B
        J row_begin_B = csr_row_ptr_B[row];
        J row_end_B   = csr_row_ptr_B[row + 1];

        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(J j = row_begin_B + lid; j < row_end_B; j += WFSIZE)
        {
            // Add column to the set
            nnz += set.insert(csr_col_ind_B[j]);
        }

        // Sum up the row nnz from all lanes
        wf_reduce_sum<WFSIZE>(&nnz);

        // Last lane in wavefront writes row nnz back to global memory
        if(lid == WFSIZE - 1)
        {
            row_nnz[row] = nnz;
        }
    }

    template <bool         GLOBAL,
              unsigned int BLOCKSIZE,
              unsigned int WFSIZE,
              unsigned int HASHSIZE,
              typename T,
              typename I,
              typename J,
              typename K>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_compress_add_fill(I m,
                                          const J* __restrict__ csr_row_ptr_A,
                                          const I* __restrict__ csr_col_ind_A,
                                          const T* __restrict__ csr_val_A,
                                          const J* __restrict__ csr_row_ptr_B,
                                          const K* __restrict__ csr_col_ind_B,
                                          const T* __restrict__ csr_val_B,
                                          const K* __restrict__ l2g,
                                          const J* __restrict__ csr_row_ptr_C,
                                          K* __restrict__ csr_col_ind_C,
                                          T* __restrict__ csr_val_C)
    {
        unsigned int lid = threadIdx.x & (WFSIZE - 1);
        unsigned int wid = threadIdx.x / WFSIZE;

        // The row this thread operates on
        I row = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= m)
        {
            return;
        }

        // Shared memory for the map
        extern __shared__ char smem[];

        K* stable = reinterpret_cast<K*>(smem);
        T* sdata  = reinterpret_cast<T*>(stable + BLOCKSIZE / WFSIZE * HASHSIZE);

        // Each wavefront operates on its own map
        unordered_map<K, T, HASHSIZE, WFSIZE> map(&stable[wid * HASHSIZE], &sdata[wid * HASHSIZE]);

        // Row entry and exit points of A
        J row_begin_A = csr_row_ptr_A[row];
        J row_end_A   = csr_row_ptr_A[row + 1];

        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(J j = row_begin_A + lid; j < row_end_A; j += WFSIZE)
        {
            // Get the column index and value
            K col_j = GLOBAL ? l2g[csr_col_ind_A[j]] : csr_col_ind_A[j];

            // Accumulate
            map.insert_or_add(col_j, csr_val_A[j]);
        }

        // Row entry and exit points of B
        J row_begin_B = csr_row_ptr_B[row];
        J row_end_B   = csr_row_ptr_B[row + 1];

        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(J j = row_begin_B + lid; j < row_end_B; j += WFSIZE)
        {
            // Accumulate
            map.insert_or_add(csr_col_ind_B[j], csr_val_B[j]);
        }

        // Finally, extract the numerical values from the hash map and fill C such
        // that the resulting matrix is sorted by columns
        for(unsigned int i = lid; i < HASHSIZE; i += WFSIZE)
        {
            // Get column from map to fill into C
            K col = map.get_key(i);

            // Skip, if table is empty
            if(col == map.empty_key())
            {
                continue;
            }

            // Get index into C
            J idx = csr_row_ptr_C[row];

            // Hash table index counter
            unsigned int cnt = 0;

            // Go through the hash table, until we reach its end
            while(cnt < HASHSIZE)
            {
                // We are searching for the right place in C to
                // insert the i-th hash table entry.
                // If the i-th hash table column entry is greater then the current one,
                // we need to leave a slot to its left.
                if(col > map.get_key(cnt))
                {
                    ++idx;
                }

                // Process next hash table entry
                ++cnt;
            }

            // Add hash table entry into P
            csr_col_ind_C[idx] = col;
            csr_val_C[idx]     = map.get_val(i);
        }
    }

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_KERNELS_CSR_HPP_
