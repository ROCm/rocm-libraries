/*************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Copyright (c) 2012-, King Abdullah University of Science and Technology
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * -- Innovative Computing Laboratory
 * -- Electrical Engineering and Computer Science Department
 * -- University of Tennessee
 * -- (C) Copyright (C) 2009-2020

 * Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
  * Neither the name of the University of Tennessee, Knoxville nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

 * This software is provided by the copyright holders and contributors
 * ``as is'' and any express or implied warranties, including, but not
 * limited to, the implied warranties of merchantability and fitness for
 * a particular purpose are disclaimed. In no event shall the copyright
 * holders or contributors be liable for any direct, indirect, incidental,
 * special, exemplary, or consequential damages (including, but not
 * limited to, procurement of substitute goods or services; loss of use,
 * data, or profits; or business interruption) however caused and on any
 * theory of liability, whether in contract, strict liability, or tort
 * (including negligence or otherwise) arising in any way out of the use
 * of this software, even if advised of the possibility of such damage.
 **************************************************************************/

#include "rocblas_hemv_symv_kernels.hpp"
#include "check_numerics_matrix.hpp"
#include "check_numerics_vector.hpp"
#include "device_macros.hpp"
#include "int64_helpers.hpp"
#include "rocblas_hemv_symv.hpp"
#include "rocblas_level2_threshold.hpp"

constexpr int rocblas_hemv_DIM_X()
{
    return 64;
}

template <typename To>
ROCBLAS_INTERNAL_EXPORT_NOINLINE size_t
    rocblas_internal_hemv_symv_kernel_workspace_size(rocblas_int n, rocblas_int batch_count)
{
    // No support for int64_t n-sizes yet in hemv/symv
    auto blocks = (n - 1) / rocblas_hemv_DIM_X() + 1;
    return sizeof(To) * blocks * n * batch_count;
}

/** helper for complex support */
template <typename T>
ROCBLAS_KERNEL_ILF void hemv_zero_imaginary(T&)
{
}

template <typename T>
ROCBLAS_KERNEL_ILF void hemv_zero_imaginary(rocblas_complex_num<T>& a)
{
    a.imag(0);
}

// treats sA as 16x64 block
#define sA16(i_, j_) (sA[(i_)][(j_)]) // i.e., sA[ (i_)*(NB_X+3) + (j_) ]

// treats sA as 32x32 block
#define sA32(i_, j_) (sA[0][(i_) + bank_shift * (j_)])

/***************************************************************************/ /**
    Upper case, compute block multiply, workspace = A*x, for any size n:

                [ (A11*x1 + A12*x2 + A13*x3)     ---                 ---    ]   [ A11    A12    A13 ]   [ x1 ]
    workspace = [ (A12^H*x1)                   (A22*x2 + A23*x3)     ---    ] = [ A12^H  A22    A23 ] * [ x2 ]
                [ (A13^H*x1)                   (A23^H*x2)          (A33*x3) ]   [ A13^H  A23^H  A33 ]   [ x3 ]

    The order is different from the lower case, because
    the upper case processes a block row from the diagonal to the right, whereas
    the lower case processes a block row from the diagonal to the left.

    Uses a 64x4 thread block.
    For     diagonal tiles, covers a 64x64 tile using three 32x32 tiles (plus one gets transposed).
    For off-diagonal tiles, covers a 64x64 tile using four  64x16 tiles.
    In both cases, each thread multiplies 4 elements.

    For rows past the bottom of the matrix, the A pointer is adjusted to be the
    last valid row of A, which multiple threads will read.
    Extra rows are ignored when saving results to workspace.
    Columns past the right edge are explicitly ignored when loading.
    x values past the bottom are set to zero, thus, extra columns are zeroed
    when multiplying.
*******************************************************************************/
template <bool IS_HEMV,
          int  NB_X,
          int  bank_shift,
          int  half_NB_X,
          int  quarter_NB_X,
          typename T_index,
          typename T>
ROCBLAS_KERNEL_ILF void rocblas_hemvn_kernel_upper_calc(int n,
                                                        T   alpha,
                                                        const T* __restrict__ A,
                                                        T_index lda,
                                                        const T* __restrict__ x,
                                                        T_index incx,
                                                        T* __restrict__ workspace,
                                                        uint32_t batch)
{
    if(!alpha)
        return;
    // 64x4 thread block
    const int tx      = threadIdx.x;
    const int ty      = threadIdx.y;
    const int blk     = blockIdx.x;
    const int blk_ind = NB_X * blk;
    const int td      = NB_X * ty + tx;

    // 32x8 thread block
    const int tx2 = td % half_NB_X;
    const int ty2 = td / half_NB_X;

    // If this blk has fewer than NB_X rows, partial is the number of valid rows,
    // so tx = 0, ..., partial-1 are valid rows, and tx >= partial are invalid.
    // Else, partial == 0.
    int partial = (blk == gridDim.x - 1 ? (n % NB_X) : 0);

    T psum, psum_t;
    T total = T{0};

    // sA is used as a 32x32 block, sA32(i,j),
    // and as a 16x64 block, sA16(i,j), in different parts of the code.
    // sA must be at least half_NB_X*bank_shift = 32x33 = 1056;
    // quarter_NB_X*(NB_X + 2) = 16*(64 + 2) = 1056
    __shared__ T sA[quarter_NB_X][NB_X + 3]; // only needs +2, use +3 to avoid bank conflict
    __shared__ T sx_blk[NB_X]; // for x[ blk ]
    __shared__ T sx_jj[NB_X]; // for x[ jj ], which cycles over all blocks right of diag

    T rA[4];
    T psums_t[4];

    // --------------------
    // load 64x1 block x(blk_ind + 0:63) into sx_blk
    x += (blk_ind + tx) * incx; // x is x(blk_ind + tx)
    if(ty == 0)
    {
        if(partial == 0 || tx < partial)
        {
            sx_blk[tx] = x[0];
        }
        else
        {
            sx_blk[tx] = T{0};
        }
    }

    // offset blocks * cols * batch
    workspace += size_t(gridDim.x) * n * batch; // workspace is workspace(0, 0, batch_count)

    // --------------------
    // move to block row
    workspace += blk * size_t(n); // workspace is workspace(0, blk)

    A += blk_ind; // A is A(blk_ind, 0)
    A += ty2 * lda + tx2; // A is A(blk_ind + tx2, ty2)

    // move to 32x32 diag block
    A += blk_ind * lda; // A is A(blk_ind + tx2, blk_ind + ty2)

    // load 32x32 diag block A(blk_ind + 0:31, blk_ind + 0:31) into sA,
    // as four 32x8 sections one after another:
    // columns 0:7, then 8:15, then 16:23, then 24:31
    if(partial)
    {
        if(tx2 >= partial)
        {
            A = A - tx2
                + (partial
                   - 1); // A is A(blk_ind + partial-1, blk_ind + ty2), the bottom-most valid row
        }
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            if(ty2 + j < partial)
            {
                sA32(tx2, ty2 + j) = A[j * lda];
            }
            else
            {
                sA32(tx2, ty2 + j) = T{0};
            }
        }
        if(tx2 >= partial)
        {
            A = A + tx2 - (partial - 1); // A is A(blk_ind + tx2, blk_ind + ty2)
        }
    }
    else
    {
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            sA32(tx2, ty2 + j) = A[j * lda];
        }
    }
    __syncthreads();

// symmetrize 32x32 diag block, copying upper to lower triangle,
// as four 32x8 sections in parallel:
// columns 0,4,8,12,16,20,24,28; then 1,5,...,29; then 2,6,...,30, then 3,7,...,31
#pragma unroll
    for(int j = ty2 * 4; j < ty2 * 4 + 4; j++)
    {
        if(j > tx2)
        {
            sA32(j, tx2) = IS_HEMV ? conj(sA32(tx2, j)) : sA32(tx2, j);
        }
        //The main diagonal of matrix A should be real
        else if(j == tx2 && IS_HEMV)
        {
            hemv_zero_imaginary(sA32(tx2, j));
        }
    }
    __syncthreads();

    // multiply 32x32 diag block * x
    // each thread does partial row sA(tx2, ty2*4 : ty2*4 + 3)
    psum = T{0};
#pragma unroll
    for(int j = 0; j < 4; j++)
    {
        psum += sA32(tx2, ty2 * 4 + j) * sx_blk[ty2 * 4 + j];
    }
    __syncthreads();

    // store partial row sums
    sA32(ty2, tx2) = psum;
    __syncthreads();

    // sum up partial row sums, so thread (tx2,0) has total for row (blk_ind + tx2)
    if(ty2 == 0)
    {
        total = sA32(0, tx2) + sA32(1, tx2) + sA32(2, tx2) + sA32(3, tx2) + sA32(4, tx2)
                + sA32(5, tx2) + sA32(6, tx2) + sA32(7, tx2);
    }
    __syncthreads();

    // --------------------
    // move to next 32x32 diag block, then repeat steps from first diag block
    A += half_NB_X + half_NB_X * lda; // A is A(blk_ind + NB/2 + tx2, blk_ind + NB/2 + ty2)

    // load 32x32 diag block A[block + 0:31, block + 0:31] into sA
    if(partial)
    {
        if(tx2 + half_NB_X >= partial)
        {
            A = A - (tx2 + half_NB_X) + (partial - 1);
        }
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            if(ty2 + j + half_NB_X < partial)
            {
                sA32(tx2, ty2 + j) = A[j * lda];
            }
            else
            {
                sA32(tx2, ty2 + j) = T{0};
            }
        }
        if(tx2 + half_NB_X >= partial)
        {
            A = A + (tx2 + half_NB_X) - (partial - 1);
        }
    }
    else
    {
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            sA32(tx2, ty2 + j) = A[j * lda];
        }
    }
    __syncthreads();

// symmetrize 32x32 diag block, copying upper to lower triangle
#pragma unroll
    for(int j = ty2 * 4; j < ty2 * 4 + 4; j++)
    {
        if(j > tx2)
        {
            sA32(j, tx2) = IS_HEMV ? conj(sA32(tx2, j)) : sA32(tx2, j);
        }
        //The main diagonal of matrix A should be real
        else if(j == tx2 && IS_HEMV)
        {
            hemv_zero_imaginary(sA32(tx2, j));
        }
    }
    __syncthreads();

    // multiply 32x32 diag block * x
    psum = T{0};
#pragma unroll
    for(int j = 0; j < 4; j++)
    {
        psum += sA32(tx2, ty2 * 4 + j) * sx_blk[half_NB_X + ty2 * 4 + j];
    }
    __syncthreads();

    // store partial row sums
    sA32(ty2, tx2) = psum;
    __syncthreads();

    // sum up partial row sums, so thread (tx2,1) has total for row (blk_ind + NB/2 + tx2)
    if(ty2 == 1)
    {
        total = sA32(0, tx2) + sA32(1, tx2) + sA32(2, tx2) + sA32(3, tx2) + sA32(4, tx2)
                + sA32(5, tx2) + sA32(6, tx2) + sA32(7, tx2);
    }
    __syncthreads();

    // --------------------
    // move to off-diag 32x32 block
    A -= half_NB_X; // A is A(blk_ind + tx2, blk_ind + NB/2 + ty2)

    // load 32x32 block of A into sA,
    // as four 32x8 sections one after another:
    // columns 0:7, then 8:15, then 16:23, then 24:31
    if(partial)
    {
        if(tx2 >= partial)
        {
            A = A - (tx2) + (partial - 1);
        }
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            if(ty2 + j + half_NB_X < partial)
            {
                sA32(tx2, ty2 + j) = A[j * lda];
            }
            else
            {
                sA32(tx2, ty2 + j) = T{0};
            }
        }
        if(tx2 >= partial)
        {
            A = A + (tx2) - (partial - 1);
        }
    }
    else
    {
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            sA32(tx2, ty2 + j) = A[j * lda];
        }
    }
    __syncthreads();

    // multiply 32x32 block (below diag)
    psum = T{0};
#pragma unroll
    for(int j = 0; j < 4; j++)
    {
        psum += (IS_HEMV ? conj(sA32(ty2 + j * 8, tx2)) : sA32(ty2 + j * 8, tx2))
                * sx_blk[j * 8 + ty2];
    }
    //__syncthreads();  // no sync needed here

    // multiply transposed 32x32 block (above diag)
    psum_t = T{0};
#pragma unroll
    for(int j = 0; j < 4; j++)
    {
        psum_t += sA32(tx2, ty2 * 4 + j) * sx_blk[half_NB_X + ty2 * 4 + j];
    }
    __syncthreads();

    // store partial sums for non-transposed 32x32 block
    sA32(ty2, tx2) = psum;
    __syncthreads();

    // sum up partial row sums, so thread (tx2,1) has total for row (blk_ind + NB/2 + tx2)
    if(ty2 == 1)
    {
        total = total + sA32(0, tx2) + sA32(1, tx2) + sA32(2, tx2) + sA32(3, tx2) + sA32(4, tx2)
                + sA32(5, tx2) + sA32(6, tx2) + sA32(7, tx2);
    }
    __syncthreads();

    // store partial sums for transposed 32x32 block
    sA32(ty2, tx2) = psum_t;
    __syncthreads();

    // sum up partial row sums, so thread (tx2,0) has total for row (blk_ind + tx2)
    if(ty2 == 0)
    {
        total = total + sA32(0, tx2) + sA32(1, tx2) + sA32(2, tx2) + sA32(3, tx2) + sA32(4, tx2)
                + sA32(5, tx2) + sA32(6, tx2) + sA32(7, tx2);
    }
    __syncthreads();

    // --------------------
    // move to next 64x64 block right of diag in block row, and
    // switch thread offset from (tx2,ty2) 32x8 block to (tx,ty) 64x4 block
    A += half_NB_X * lda; // A is A(blk_ind + tx2, blk_ind + NB_X + ty2 )
    A -= ty2 * lda + tx2; // A is A(blk_ind,       blk_ind + NB_X       )
    A += 4 * ty * lda + tx; // A is A(blk_ind + tx,  blk_ind        + 4*ty)

    // Unlike lower case, don't adjust A here for partial # of rows.
    // Since block is right of diagonal, it must have all NB rows,
    // but can have < NB columns, dealt with when loading below.

    x -= blk_ind * incx; // x is x(tx)

    // 16x16 thread block
    const int tx4 = td % quarter_NB_X;
    const int ty4 = td / quarter_NB_X;

    // cycle over blocks jj right of diagonal, in block row blk
    for(int jj = blk + 1; jj < gridDim.x; ++jj)
    {
        partial = (jj == gridDim.x - 1 ? (n % NB_X) : 0);

        // load 64x1 block x(jj_ind + 0:63) into sx_jj
        if(ty == 0)
        {
            if(partial == 0 || tx < partial)
            {
                sx_jj[tx] = x[jj * NB_X * incx];
            }
            else
            {
                sx_jj[tx] = T{0};
            }
        }
        __syncthreads();

        for(int k = 0; k < 4; k++)
        {
            // load 64x16 block of A into rA, 4 elements per thread,
            // as four 64x4 sections in parallel:
            // columns 0,4,8,12; then 1,5,9,13; then 2,6,10,14; then 3,7,11,15
            if(partial)
            {
#pragma unroll
                for(int j = 0; j < 4; j++)
                {
                    if(4 * ty + j + k * quarter_NB_X < partial)
                    {
                        rA[j] = A[j * lda];
                    }
                    else
                    {
                        rA[j] = T{0};
                    }
                }
            }
            else
            {
#pragma unroll
                for(int j = 0; j < 4; j++)
                {
                    rA[j] = A[j * lda];
                }
            }

// 1) multiply 64x16 block A_{blk,jj} * x_jj
//    each thread does partial row rA(tx + 16*k, ty*4 + 16*k : ty*4 + 3 + 16*k)
// 2) multiply 16x64 block A_{blk,jj} * x_blk,
//    storing each product Aji*xi to sA(j,i)
#pragma unroll
            for(int j = 0; j < 4; j++)
            {
                total
                    += rA[j] * sx_jj[quarter_NB_X * k + ty * 4 + j]; // y_blk = A_{blk,jj}   * x_jj
                sA16(ty * 4 + j, tx)
                    = (IS_HEMV ? conj(rA[j]) : rA[j]) * sx_blk[tx]; // y_jj  = A_{blk,jj}^H * x_blk
            }
            __syncthreads();

            // do partial row sums for transposed 16x64 result
            // use 16x16 thread grid (tx4, ty4) instead of 64x4 (tx, ty)
            // sum sixteen 16x4 sections in parallel:
            // columns 0,4,8,...,60; then 1,5,...,61; then 2,6,...,62; then 3,7,...,63
            psum_t = T{0};
#pragma unroll
            for(int j = 0; j < 4; j++)
            {
                psum_t += sA16(tx4, ty4 * 4 + j);
            }
            __syncthreads();

            // store partial row sums of transposed result, y_jj (locally)
            psums_t[k] = psum_t;

            // move right to next 64x16 block
            A += lda * quarter_NB_X; // A is A(blk_ind + tx, jj*NB_X + (k+1)*NB_X/4 + 4*ty)
        }
// already at next 64x64 block
// A is A(blk_ind + tx, (jj+1)*NB_x + 4*ty)

// store partial row sums of transposed result, y_jj
#pragma unroll
        for(int k = 0; k < 4; k++)
        {
            sA16(tx4, ty4 + quarter_NB_X * k) = psums_t[k];
        }
        __syncthreads();

        // sum up partial row sums of transposed result, y_jj, and store final total to workspace
        // thread (tx4,ty4) where ty4 < 4 sums row tx4 + ty4*16
        if(ty4 < 4 && (partial == 0 || tx4 + ty4 * quarter_NB_X < partial))
        {
            int ty4_nb4 = ty4 * quarter_NB_X;
            psum_t      = sA16(tx4, 0 + ty4_nb4) + sA16(tx4, 1 + ty4_nb4) + sA16(tx4, 2 + ty4_nb4)
                     + sA16(tx4, 3 + ty4_nb4) + sA16(tx4, 4 + ty4_nb4) + sA16(tx4, 5 + ty4_nb4)
                     + sA16(tx4, 6 + ty4_nb4) + sA16(tx4, 7 + ty4_nb4) + sA16(tx4, 8 + ty4_nb4)
                     + sA16(tx4, 9 + ty4_nb4) + sA16(tx4, 10 + ty4_nb4) + sA16(tx4, 11 + ty4_nb4)
                     + sA16(tx4, 12 + ty4_nb4) + sA16(tx4, 13 + ty4_nb4) + sA16(tx4, 14 + ty4_nb4)
                     + sA16(tx4, 15 + ty4_nb4);
            workspace[jj * NB_X + tx4 + ty4_nb4]
                = psum_t; // store at workspace( jj*NB_X + tx4 + ty4*16, blk )
        }
        __syncthreads();
    }

    // store row sums
    sA16(ty, tx) = total;
    __syncthreads();

    partial = (blk == gridDim.x - 1 ? (n % NB_X) : 0);

    // sum up final total, y_blk, for row tx
    if(ty == 0 && (partial == 0 || tx < partial))
    {
        total                      = sA16(0, tx) + sA16(1, tx) + sA16(2, tx) + sA16(3, tx);
        workspace[blk * NB_X + tx] = total; // store at workspace( blk*NB_X + tx, blk )
    }
}
// end rocblas_hemvn_kernel_upper_calc

/*****************************************************************************
    Upper case, sum up final results
    Each block sums one block row; each thread sums one row.

    On input (for 3 blocks):
                [ (A11*x1 + A12*x2 + A13*x3)     ---                 ---    ]
    workspace = [ (A12^H*x1)                   (A22*x2 + A23*x3)     ---    ]
                [ (A13^H*x1)                   (A23^H*x2)          (A33*x3) ]

    On output:
              [ (A11*x1 + A12*x2 + A13*x3)         ]
    y = alpha*[ (A12^H*x1) + (A22*x2 + A23*x3)     ] + beta*y
              [ (A13^H*x1) + (A23^H*x2) + (A33*x3) ]
*******************************************************************************/
template <rocblas_int NB_X, typename T_index, typename U, typename TPtr, typename W>
ROCBLAS_KERNEL(NB_X)
rocblas_hemvn_kernel_upper_block_sum(rocblas_int    n,
                                     U              alpha_device_host,
                                     rocblas_stride stride_alpha,
                                     U              beta_device_host,
                                     rocblas_stride stride_beta,
                                     TPtr __restrict__ ya,
                                     rocblas_stride shifty,
                                     T_index        incy,
                                     rocblas_stride stridey,
                                     W*             workspace,
                                     rocblas_int    batch_count)
{
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
        W* saved_workspace_address
            = workspace; //Saving the address of the workspace memory to assign it after the computation
#endif

        auto alpha = load_scalar(alpha_device_host, batch, stride_alpha);
        auto beta  = load_scalar(beta_device_host, batch, stride_beta);

        if(!alpha && beta == 1)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        auto* y = load_ptr_batch(ya, batch, shifty, stridey);

        int tx      = threadIdx.x;
        int blk     = blockIdx.x;
        int blk_ind = blk * NB_X;
        int ind     = blk_ind + tx;
        if(!alpha)
        {
            if(ind < n)
                y[ind * incy] = beta ? beta * y[ind * incy] : 0;

#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        // offset blocks * cols * batch
        workspace += size_t(gridDim.x) * n * batch; // workspace is workspace(0, 0, batch_count)

        // Don't write outside [0, ..., n)
        if(ind < n)
        {
            workspace += ind;
            W Ax = W{0};
            for(int j = 0; j <= blk; ++j)
            {
                Ax += workspace[0];
                workspace += n;
            }
            y[ind * incy] = beta ? beta * y[ind * incy] + alpha * Ax : alpha * Ax;
        }
#if DEVICE_GRID_YZ_16BIT
        workspace
            = saved_workspace_address; // Assigning back the saved address of the workspace memory to do the next batch computation
    }
#endif
}
// end hemvn_kernel_upper_block_sum_calc

/***************************************************************************/ /**
    Lower case, compute block multiply, workspace = A*x, for any size n:

                [ (A11*x1)   (A21^H*x2)          (A31^H*x3)                 ]   [ A11  A21^H  A31^H ]   [ x1 ]
    workspace = [   ---      (A21*x1 + A22*x2)   (A32^H*x3)                 ] = [ A21  A22    A32^H ] * [ x2 ]
                [   ---        ---               (A31*x1 + A32*x2 + A33*x3) ]   [ A31  A32    A33   ]   [ x3 ]

    Uses a 64x4 thread block.
    For     diagonal tiles, covers a 64x64 tile using three 32x32 tiles (plus one gets transposed).
    For off-diagonal tiles, covers a 64x64 tile using four  64x16 tiles.
    In both cases, each thread multiplies 4 elements.

    For rows past the bottom of the matrix, the A pointer is adjusted to be the
    last valid row of A, which multiple threads will read.
    Extra rows are ignored when saving results to workspace.
    Columns past the right edge are explicitly ignored when loading.
    x values past the bottom are set to zero, thus, extra columns are zeroed
    when multiplying.

    Previously:
                [ (A11*x1)       ---                                          ]
    workspace = [ (A21^H*x2)   (A21*x1 + A22*x2)     ---                      ]
                [ (A31^H*x3)   (A32^H*x3)          (A31*x1 + A32*x2 + A33*x3) ]
    which doesn't workspace as well because that has dimension blocks*NB by blocks,
    where blocks*NB >= n, and it can be that blocks*NB > lda, so it won't fit in
    lda*blocks space. This is why it used to need lwork = lda*(blocks + 1).
*******************************************************************************/

template <bool        IS_HEMV,
          rocblas_int NB_X,
          rocblas_int bank_shift,
          rocblas_int half_NB_X,
          rocblas_int quarter_NB_X,
          typename T_index,
          typename T>
ROCBLAS_KERNEL_ILF void rocblas_hemvn_kernel_lower_calc(rocblas_int n,
                                                        T           alpha,
                                                        const T* __restrict__ A,
                                                        T_index lda,
                                                        const T* __restrict__ x,
                                                        T_index incx,
                                                        T* __restrict__ workspace,
                                                        uint32_t batch)
{
    if(!alpha)
        return;
    // 64x4 thread block
    const int tx      = threadIdx.x;
    const int ty      = threadIdx.y;
    const int blk     = blockIdx.x;
    const int blk_ind = NB_X * blk;
    const int td      = NB_X * ty + tx;

    // 32x8 thread block
    const int tx2 = td % half_NB_X;
    const int ty2 = td / half_NB_X;

    // If this blk has fewer than NB_X rows, partial is the number of valid rows,
    // so tx = 0, ..., partial-1 are valid rows, and tx >= partial are invalid.
    // Else, partial == 0.
    const int partial = (blk == gridDim.x - 1 ? (n % NB_X) : 0);

    T psum, psum_t;
    T total = T{0};

    // sA is used as a 32x32 block, sA32(i,j),
    // and as a 16x64 block, sA16(i,j), in different parts of the code.
    // sA must be at least half_NB_X*bank_shift = 32x33 = 1056;
    // quarter_NB_X*(NB_X + 2) = 16*(64 + 2) = 1056
    __shared__ T sA[quarter_NB_X][NB_X + 3]; // only needs +2, use +3 to avoid bank conflict
    __shared__ T sx_blk[NB_X]; // for x[ blk ]
    __shared__ T sx_jj[NB_X]; // for x[ jj ], which cycles over all blocks left of diag

    T rA[4];
    T psums_t[4];

    // --------------------
    // load 64x1 block x(blk_ind + 0:63) into sx_blk
    x += (blk_ind + tx) * incx; // x is x(blk_ind + tx)
    if(ty == 0)
    {
        if(partial == 0 || tx < partial)
        {
            sx_blk[tx] = x[0];
        }
        else
        {
            sx_blk[tx] = T{0};
        }
    }

    // --------------------
    // offset blocks * cols * batch
    workspace += size_t(gridDim.x) * n * batch; // workspace is workspace(0, 0, batch_count)

    // move to block row
    workspace += blk * n; // workspace is workspace(0, blk)

    A += blk_ind; // A is A(blk_ind, 0)
    A += ty2 * lda + tx2; // A is A(blk_ind + tx2, ty2)

    // move to 32x32 diag block
    A += blk_ind * lda; // A is A(blk_ind + tx2, blk_ind + ty2)

    // load 32x32 diag block A(blk_ind + 0:31, blk_ind + 0:31) into sA,
    // as four 32x8 sections one after another:
    // columns 0:7, then 8:15, then 16:23, then 24:31
    if(partial)
    {
        if(tx2 >= partial)
        {
            A = A - tx2
                + (partial
                   - 1); // A is A(blk_ind + partial-1, blk_ind + ty2), the bottom-most valid row
        }
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            if(ty2 + j < partial)
            {
                sA32(tx2, ty2 + j) = A[j * lda];
            }
            else
            {
                sA32(tx2, ty2 + j) = T{0};
            }
        }
        if(tx2 >= partial)
        {
            A = A + tx2 - (partial - 1); // A is A(blk_ind + tx2, blk_ind + ty2)
        }
    }
    else
    {
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            sA32(tx2, ty2 + j) = A[j * lda];
        }
    }
    __syncthreads();

    // symmetrize 32x32 diag block, copying lower to upper triangle,
    // as four 32x8 sections in parallel:
    // columns 0,4,8,12,16,20,24,28; then 1,5,...,29; then 2,6,...,30, then 3,7,...,31
    for(int j = ty2 * 4; j < ty2 * 4 + 4; j++)
    {
        if(j < tx2)
        {
            sA32(j, tx2) = IS_HEMV ? conj(sA32(tx2, j)) : sA32(tx2, j);
        }
        //The main diagonal of matrix A should be real
        else if(j == tx2 && IS_HEMV)
        {
            hemv_zero_imaginary(sA32(tx2, j));
        }
    }
    __syncthreads();

    // multiply 32x32 diag block * x
    // each thread does partial row sA(tx2, ty2*4 : ty2*4 + 3)
    psum = T{0};
#pragma unroll
    for(int j = 0; j < 4; j++)
    {
        psum += sA32(tx2, ty2 * 4 + j) * sx_blk[ty2 * 4 + j];
    }
    __syncthreads();

    // store partial row sums
    sA32(ty2, tx2) = psum;
    __syncthreads();

    // sum up partial row sums, so thread (tx2,0) has total for row (blk_ind + tx2)
    if(ty2 == 0)
    {
        total = sA32(0, tx2) + sA32(1, tx2) + sA32(2, tx2) + sA32(3, tx2) + sA32(4, tx2)
                + sA32(5, tx2) + sA32(6, tx2) + sA32(7, tx2);
    }
    __syncthreads();

    // --------------------
    // move to next 32x32 diag block, then repeat steps from first diag block
    A += half_NB_X + half_NB_X * lda; // A is A(blk_ind + NB/2 + tx2, blk_ind + NB/2 + ty2)

    // load 32x32 diag block A[block + 0:31, block + 0:31] into sA
    if(partial)
    {
        if(tx2 + half_NB_X >= partial)
        {
            A = A - (tx2 + half_NB_X) + (partial - 1);
        }
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            if(ty2 + j + half_NB_X < partial)
            {
                sA32(tx2, ty2 + j) = A[j * lda];
            }
            else
            {
                sA32(tx2, ty2 + j) = T{0};
            }
        }
        if(tx2 + half_NB_X >= partial)
        {
            A = A + (tx2 + half_NB_X) - (partial - 1);
        }
    }
    else
    {
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            sA32(tx2, ty2 + j) = A[j * lda];
        }
    }
    __syncthreads();

    // symmetrize 32x32 diag block, copying lower to upper triangle
    for(int j = ty2 * 4; j < ty2 * 4 + 4; j++)
    {
        if(j < tx2)
        {
            sA32(j, tx2) = IS_HEMV ? conj(sA32(tx2, j)) : sA32(tx2, j);
        }
        //The main diagonal of matrix A should be real
        else if(j == tx2 && IS_HEMV)
        {
            hemv_zero_imaginary(sA32(tx2, j));
        }
    }
    __syncthreads();

    // multiply 32x32 diag block * x
    psum = T{0};
#pragma unroll
    for(int j = 0; j < 4; j++)
    {
        psum += sA32(tx2, ty2 * 4 + j) * sx_blk[half_NB_X + ty2 * 4 + j];
    }
    __syncthreads();

    // store partial row sums
    sA32(ty2, tx2) = psum;
    __syncthreads();

    // sum up partial row sums, so thread (tx2,1) has total for row (blk_ind + NB/2 + tx2)
    if(ty2 == 1)
    {
        total = sA32(0, tx2) + sA32(1, tx2) + sA32(2, tx2) + sA32(3, tx2) + sA32(4, tx2)
                + sA32(5, tx2) + sA32(6, tx2) + sA32(7, tx2);
    }
    __syncthreads();

    // --------------------
    // move to off-diag 32x32 block
    A -= half_NB_X * lda; // A is A(blk_ind + NB/2 + tx2, blk_ind + ty2)

    // load 32x32 block of A into sA,
    // as four 32x8 sections one after another:
    // columns 0:7, then 8:15, then 16:23, then 24:31
    if(partial)
    {
        if(tx2 + half_NB_X >= partial)
        {
            A = A - (tx2 + half_NB_X) + (partial - 1);
        }
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            if(ty2 + j < partial)
            {
                sA32(tx2, ty2 + j) = A[j * lda];
            }
            else
            {
                sA32(tx2, ty2 + j) = T{0};
            }
        }
        if(tx2 + half_NB_X >= partial)
        {
            A = A + (tx2 + half_NB_X) - (partial - 1);
        }
    }
    else
    {
#pragma unroll
        for(int j = 0; j < half_NB_X; j += 8)
        {
            sA32(tx2, ty2 + j) = A[j * lda];
        }
    }
    __syncthreads();

    // multiply 32x32 block (below diag)
    psum = T{0};
#pragma unroll
    for(int j = 0; j < 4; j++)
    {
        psum += sA32(tx2, ty2 + j * 8) * sx_blk[j * 8 + ty2];
    }
    //__syncthreads();  // no sync needed here

    // multiply transposed 32x32 block (above diag)
    psum_t = T{0};
#pragma unroll
    for(int j = 0; j < 4; j++)
    {
        psum_t += (IS_HEMV ? conj(sA32(ty2 * 4 + j, tx2)) : sA32(ty2 * 4 + j, tx2))
                  * sx_blk[half_NB_X + ty2 * 4 + j];
    }
    __syncthreads();

    // store partial sums for non-transposed 32x32 block
    sA32(ty2, tx2) = psum;
    __syncthreads();

    // sum up partial row sums, so thread (tx2,1) has total for row (blk_ind + NB/2 + tx2)
    if(ty2 == 1)
    {
        total = total + sA32(0, tx2) + sA32(1, tx2) + sA32(2, tx2) + sA32(3, tx2) + sA32(4, tx2)
                + sA32(5, tx2) + sA32(6, tx2) + sA32(7, tx2);
    }
    __syncthreads();

    // store partial sums for transposed 32x32 block
    sA32(ty2, tx2) = psum_t;
    __syncthreads();

    // sum up partial row sums, so thread (tx2,0) has total for row (blk_ind + tx2)
    if(ty2 == 0)
    {
        total = total + sA32(0, tx2) + sA32(1, tx2) + sA32(2, tx2) + sA32(3, tx2) + sA32(4, tx2)
                + sA32(5, tx2) + sA32(6, tx2) + sA32(7, tx2);
    }
    __syncthreads();

    // --------------------
    // move to leftmost 64x64 block in block row, and
    // switch thread offset from (tx2,ty2) 32x8 block to (tx,ty) 64x4 block
    A -= half_NB_X; // A is A(blk_ind + tx2, blk_ind + ty2)
    A -= blk_ind * lda; // A is A(blk_ind + tx2,           ty2)
    A -= ty2 * lda + tx2; // A is A(blk_ind, 0)
    A += 4 * ty * lda + tx; // A is A(blk_ind + tx, 4*ty)

    if(partial && tx >= partial)
    {
        A = A - tx + (partial - 1); // A is A(blk_ind + partial-1, 4*ty), the bottom-most valid row
    }

    x -= blk_ind * incx; // x is x(tx)

    // 16x16 thread block
    const int tx4 = td % quarter_NB_X;
    const int ty4 = td / quarter_NB_X;

    // cycle over blocks jj left of diagonal, in block row blk
    for(int jj = 0; jj < blk; ++jj)
    {
        // load 64x1 block x(jj_ind + 0:63) into sx_jj
        // since this block is left of diagonal, x must have all NB rows
        if(ty == 0)
        {
            sx_jj[tx] = x[jj * NB_X * incx];
        }
        __syncthreads();

        for(int k = 0; k < 4; k++)
        {
// load 64x16 block of A into rA, 4 elements per thread,
// as four 64x4 sections in parallel:
// columns 0,4,8,12; then 1,5,9,13; then 2,6,10,14; then 3,7,11,15
// since this block is left of diagonal, it has all NB columns,
// and block of x must have all NB rows.
#pragma unroll
            for(int j = 0; j < 4; j++)
            {
                rA[j] = A[j * lda];
            }

// 1) multiply 64x16 block A_{blk,jj} * x_jj
//    each thread does partial row rA(tx + 16*k, ty*4 + 16*k : ty*4 + 3 + 16*k)
// 2) multiply transposed 16x64 block A_{blk,jj}^H * x_blk,
//    storing each product Aji*xi to sA(j,i)
#pragma unroll
            for(int j = 0; j < 4; j++)
            {
                total
                    += rA[j] * sx_jj[quarter_NB_X * k + ty * 4 + j]; // y_blk = A_{blk,jj}   * x_jj
                sA16(ty * 4 + j, tx)
                    = (IS_HEMV ? conj(rA[j]) : rA[j]) * sx_blk[tx]; // y_jj  = A_{blk,jj}^H * x_blk
            }
            __syncthreads();

            // do partial row sums for transposed 16x64 result
            // use 16x16 thread grid (tx4, ty4) instead of 64x4 (tx, ty)
            // sum sixteen 16x4 sections in parallel:
            // columns 0,4,8,...,60; then 1,5,...,61; then 2,6,...,62; then 3,7,...,63
            psum_t = T{0};
#pragma unroll
            for(int j = 0; j < 4; j++)
            {
                psum_t += sA16(tx4, ty4 * 4 + j);
            }
            __syncthreads();

            // store partial row sums of transposed result, y_jj (locally)
            psums_t[k] = psum_t;

            // move right to next 64x16 block
            A += lda
                 * quarter_NB_X; // A is A(blk_ind + tx#, jj*NB_x + (k+1)*NB_X/4 + 4*ty), # tx or partial
        }
// already at next 64x64 block
// A is A(blk_ind + tx#, (jj+1)*NB_x + 4*ty), # tx or partial

// store partial row sums of transposed result, y_jj
#pragma unroll
        for(int k = 0; k < 4; k++)
        {
            sA16(tx4, ty4 + quarter_NB_X * k) = psums_t[k];
        }
        __syncthreads();

        // sum up partial row sums of transposed result, y_jj, and store final total to workspace
        // thread (tx4,ty4) where ty4 < 4 sums row tx4 + ty4*16
        // since this is the transposed block above the diagonal, it must have all NB rows
        if(ty4 < 4)
        {
            int ty4_nb4 = ty4 * quarter_NB_X;
            psum_t      = sA16(tx4, 0 + ty4_nb4) + sA16(tx4, 1 + ty4_nb4) + sA16(tx4, 2 + ty4_nb4)
                     + sA16(tx4, 3 + ty4_nb4) + sA16(tx4, 4 + ty4_nb4) + sA16(tx4, 5 + ty4_nb4)
                     + sA16(tx4, 6 + ty4_nb4) + sA16(tx4, 7 + ty4_nb4) + sA16(tx4, 8 + ty4_nb4)
                     + sA16(tx4, 9 + ty4_nb4) + sA16(tx4, 10 + ty4_nb4) + sA16(tx4, 11 + ty4_nb4)
                     + sA16(tx4, 12 + ty4_nb4) + sA16(tx4, 13 + ty4_nb4) + sA16(tx4, 14 + ty4_nb4)
                     + sA16(tx4, 15 + ty4_nb4);
            workspace[jj * NB_X + tx4 + ty4_nb4]
                = psum_t; // store at workspace( jj*NB_X + tx4 + ty4*16, blk )
        }
        __syncthreads();
    }

    // store row sums
    sA16(ty, tx) = total;
    __syncthreads();

    // sum up final total, y_blk, for row tx
    if(ty == 0 && (partial == 0 || tx < partial))
    {
        total                      = sA16(0, tx) + sA16(1, tx) + sA16(2, tx) + sA16(3, tx);
        workspace[blk * NB_X + tx] = total; // store at workspace( blk*NB_X + tx, blk )
    }
}
// end rocblas_hemvn_kernel_lower_calc

/*****************************************************************************
    UPLO = rocblas_fill_lower, sum up final results
    Each block sums one block row; each thread sums one row.

    On input (for 3 blocks):
                [ (A11*x1)   (A21^H*x2)          (A31^H*x3)                 ]
    workspace = [   ---      (A21*x1 + A22*x2)   (A32^H*x3)                 ]
                [   ---        ---               (A31*x1 + A32*x2 + A33*x3) ]


    On output:
              [ (A11*x1) + (A21^H*x2) + (A31^H*x3) ]
    y = alpha*[ (A21*x1 + A22*x2)     + (A32^H*x3) ] + beta*y
              [ (A21*x1 + A22*x2 + A33*x3)         ]
*******************************************************************************/
template <rocblas_int NB_X, typename T_index, typename U, typename TPtr, typename W>
ROCBLAS_KERNEL(NB_X)
rocblas_hemvn_kernel_lower_block_sum(rocblas_int    n,
                                     U              alpha_device_host,
                                     rocblas_stride stride_alpha,
                                     U              beta_device_host,
                                     rocblas_stride stride_beta,
                                     TPtr __restrict__ ya,
                                     rocblas_stride shifty,
                                     T_index        incy,
                                     rocblas_stride stridey,
                                     W* __restrict__ workspace,
                                     rocblas_int batch_count)
{
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
        W* saved_workspace_address
            = workspace; //Saving the address of the workspace memory to assign it after the computation
#endif
        auto alpha = load_scalar(alpha_device_host, batch, stride_alpha);
        auto beta  = load_scalar(beta_device_host, batch, stride_beta);

        if(!alpha && beta == 1)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        auto* y = load_ptr_batch(ya, batch, shifty, stridey);

        int tx      = threadIdx.x;
        int blk     = blockIdx.x;
        int blk_ind = blk * NB_X;
        int ind     = blk_ind + tx;
        int blocks  = gridDim.x;

        if(!alpha)
        {
            if(ind < n)
                y[ind * incy] = beta ? beta * y[ind * incy] : 0;

#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        // offset blocks * cols * batch
        workspace += size_t(gridDim.x) * n * batch; // workspace is workspace(0, 0, batch_count)

        // Don't write outside [0, ..., n)
        if(ind < n)
        {
            workspace += ind + blk * n;
            W Ax = W{0};
            for(int j = blk; j < blocks; ++j)
            {
                Ax += workspace[0];
                workspace += n;
            }
            y[ind * incy] = beta ? beta * y[ind * incy] + alpha * Ax : alpha * Ax;
        }
#if DEVICE_GRID_YZ_16BIT
        workspace
            = saved_workspace_address; // Assigning back the saved address of the workspace memory to do the next batch computation
    }
#endif
}
// end hemvn_kernel_lower_block_sum_calc

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T>
ROCBLAS_KERNEL_ILF void
    rocblas_symv_kernel_upper_double_buffered_diagonal_calc(rocblas_int n,
                                                            T           alpha,
                                                            const T* __restrict__ A,
                                                            int64_t lda,
                                                            const T* __restrict__ x,
                                                            int64_t incx,
                                                            T       beta,
                                                            T* __restrict__ y,
                                                            int64_t incy)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int td = (DIM_X * ty) + tx;

    T res  = T(0);
    T yold = T(0);

    __shared__ T la[DIM_X * DIM_X];
    __shared__ T buff[DIM_X];
    __shared__ T accum[DIM_X * (2 * DIM_Y)];

    // Advance 'y'
    y += (bx * DIM_X) * incy;

    // Early return when alpha == 0
    if(!alpha)
    {
        if(ty == 0)
            y[incy * tx] *= beta;
        return;
    }

    // Advance 'A' to start of diagonal blocks first
    A += DIM_X * bx * (lda + 1);

    // Advance 'A' to start row for each thread inside the diagonal block
    A += ty * lda + tx;

    // Advance 'x'
    x += (bx * DIM_X) * incx;

    if(ty == 0)
    {
        // skip beta * y when beta == 0
        if(beta)
            yold = beta * y[incy * tx];

        buff[tx] = x[incx * tx];
    }

    // load first chunk
    if(tx < (DIM_X / 2))
    {
#pragma unroll
        for(int j = 0; j < (DIM_X / 2); j += DIM_Y)
            la[td + j * DIM_X] = A[j * lda];
    }

    // Advance to second chunk
    A += (DIM_X / 2) * lda;

// load second chunk first
#pragma unroll
    for(int j = 0; j < (DIM_X / 2); j += DIM_Y)
        la[DIM_X * ((DIM_X / 2) + j + ty) + tx] = A[j * lda];

    __syncthreads();

// mirror second chunk
#pragma unroll
    for(int j = 0; j < (DIM_X / 2); j += DIM_Y)
        if(abs(tx - ty) > (j + (DIM_X / 2)))
            la[DIM_X * ((DIM_X / 2) + j + ty) + tx] = la[DIM_X * tx + (DIM_X / 2) + j + ty];

    // mirror first chunk
    if(ty <= tx)
        la[td] = la[tx * DIM_X + ty];

#pragma unroll
    for(int j = DIM_Y; j < (DIM_X / 2); j += DIM_Y)
        if(abs(tx - ty) > j)
            la[tx + (ty + j) * DIM_X] = la[ty + j + tx * DIM_X];

    __syncthreads();

// compute first chunk
#pragma unroll
    for(int j = 0; j < (DIM_X / 2); j += DIM_Y)
        res += la[(ty + j) * DIM_X + tx] * buff[j + ty];

// compute second chunk
#pragma unroll
    for(int j = (DIM_X / 2); j < 2 * (DIM_X / 2); j += DIM_Y)
        res += la[(ty + j) * DIM_X + tx] * buff[j + ty];

    accum[td] = res;
    __syncthreads();

    if(ty == 0)
    {
        res = T(0);
#pragma unroll
        for(int j = 0; j < DIM_Y; j++)
            res += accum[j * DIM_X + tx];

        res *= alpha;

        if(beta)
            res += yold;

        y[tx * incy] = res;
    }
}

template <rocblas_int DIM_X, rocblas_int DIM_Y, rocblas_int elements_per_thread, typename T>
ROCBLAS_KERNEL_ILF void
    rocblas_symv_kernel_upper_double_buffered_non_diagonal_calc(rocblas_int n,
                                                                T           alpha,
                                                                const T* __restrict__ A,
                                                                int64_t  lda,
                                                                const T* x,
                                                                int64_t  incx,
                                                                T*       y,
                                                                int64_t  incy)
{
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int td  = (DIM_X * ty) + tx;
    const int tx_ = td % (DIM_X / 2);
    const int ty_ = td / (DIM_X / 2);
    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;

    // compute how many matrix blocks to be processed
    int count = bx / gridDim.y;

    T res_1_ = T(0);
    T res_2_ = T(0);
    T x1, x2;

    const T* xcopy;
    T*       ycopy;

    __shared__ T la[DIM_X * (DIM_X / 2)];
    __shared__ T accum[DIM_X * (2 * DIM_Y)];
    __shared__ T xbuff[DIM_X];

    T A_reg_upper[elements_per_thread];
    T A_reg_lower[elements_per_thread];
    T treg[elements_per_thread] = {T(0)};

    // Advance 'A' to the start a non-diagonal block
    A += DIM_X * bx * lda;

    // divide the work among y-direction of the grid
    A += (by * count) * DIM_X;

    xcopy = x + (bx * DIM_X) * incx;
    x += (by * count * DIM_X) * incx;

    if(bx == 0)
        return;

    if(ty == 0)
        xbuff[tx] = xcopy[tx * incx];

    // Advance 'y'
    ycopy = y;
    y += (bx * DIM_X) * incy;
    ycopy += (by * count * DIM_X) * incy;

    if(by == gridDim.y - 1)
        count += bx % gridDim.y;

    if(count == 0)
        return;

    size_t j = ty_ * elements_per_thread * lda + tx_;

    __syncthreads();

// prefetch upper
#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
        A_reg_upper[k] = A[j + k * lda];

    x1 = x[incx * tx_];

    //#pragma unroll
    for(int Vblocks = 0; Vblocks < count; Vblocks++)
    {
        res_1_ = T(0);
        res_2_ = T(0);

        x2 = x[incx * (tx_ + (DIM_X / 2))];

// prefetch lower
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
            A_reg_lower[k] = A[(DIM_X / 2) + j + k * lda];

// compute upper
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
        {
            res_1_ += A_reg_upper[k] * xbuff[ty_ * elements_per_thread + k];
            treg[k] += A_reg_upper[k] * x1;
        }

        // Advance to next block in A
        A += DIM_X;
        x += DIM_X * incx;

        if(Vblocks != count - 1)
        {
// prefetch upper of next block
#pragma unroll
            for(int k = 0; k < elements_per_thread; k++)
                A_reg_upper[k] = A[j + k * lda];

            x1 = x[incx * tx_];
        }

#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
        {
            res_2_ += A_reg_lower[k] * xbuff[ty_ * elements_per_thread + k];
            treg[k] += A_reg_lower[k] * x2;
        }

        // Horizontal block should be stored in global memory
        __syncthreads();
        accum[ty_ * DIM_X + tx_]               = res_1_;
        accum[ty_ * DIM_X + tx_ + (DIM_X / 2)] = res_2_;
        __syncthreads();
        if(ty == 0)
        {
            res_1_ = T(0);
#pragma unroll
            for(int k = 0; k < (2 * DIM_Y); k++)
                res_1_ += accum[k * DIM_X + tx];

            // use atomics
            atomicAdd(&ycopy[tx * incy], res_1_ * alpha);
            ycopy += DIM_X * incy;
        }
    } // end of for loop on blocks

#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
        la[(ty_ * elements_per_thread + k) * (DIM_X / 2) + tx_] = treg[k];

    __syncthreads(); // important

    if(ty == 0)
    {
        treg[0] = T(0);
#pragma unroll
        for(int k = tx; k < tx + (DIM_X / 2); k++)
            treg[0] += la[tx * (DIM_X / 2) + (k % (DIM_X / 2))];

        // use atomics
        atomicAdd(&y[tx * incy], treg[0] * alpha);
    }
}

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T>
ROCBLAS_KERNEL_ILF void
    rocblas_symv_kernel_upper_double_buffered_diagonal_generic_calc(rocblas_int n,
                                                                    T           alpha,
                                                                    const T* __restrict__ A,
                                                                    int64_t lda,
                                                                    const T* __restrict__ x,
                                                                    int64_t incx,
                                                                    T       beta,
                                                                    T* __restrict__ y,
                                                                    int64_t           incy,
                                                                    const rocblas_int n_mod_DIM_X)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int td = (DIM_X * ty) + tx;

    T res  = T(0);
    T yold = T(0);

    __shared__ T la_shared[DIM_X * DIM_X];
    __shared__ T x_buff_shared[DIM_X];
    __shared__ T accum_shared[DIM_X * (2 * DIM_Y)];

    // Advance 'y'
    y += (bx * DIM_X) * incy;

    // Early return when alpha == 0
    if(!alpha)
    {
        if(ty == 0 && (tx < n_mod_DIM_X || bx < gridDim.x - 1))
            y[tx * incy] *= beta;

        return;
    }

    // Advance 'A' to start of diagonal blocks first
    A += DIM_X * bx * (lda + 1);

    // Advance 'A' to start row for each thread inside the diagonal block
    A += ty * lda + tx;

    // Advance 'x'
    x += (bx * DIM_X) * incx;

    // load part of vector 'x'
    if(ty == 0 && (tx < n_mod_DIM_X || bx < gridDim.x - 1))
    {
        x_buff_shared[tx] = x[tx * incx];
        if(beta)
            yold = beta * y[tx * incy];
    }

    // init shmem (last TB only)
    if(bx == gridDim.x - 1)
    {
#pragma unroll
        for(int j = 0; j < DIM_X; j += DIM_Y)
            la_shared[j * DIM_X + td] = T(0);

        if(ty == 0 && tx >= n_mod_DIM_X)
            x_buff_shared[tx] = T(0);
    }

    // load a bock of data
    if(bx == gridDim.x - 1)
    {
        if(tx < n_mod_DIM_X)
        {
            int j;
            for(j = 0; j < n_mod_DIM_X / DIM_Y; j++)
                la_shared[(j * DIM_Y) * DIM_X + td] = A[(j * DIM_Y) * lda];

            if(ty < (n_mod_DIM_X % DIM_Y))
                la_shared[(j * DIM_Y) * DIM_X + td] = A[(j * DIM_Y) * lda];
        }
    }
    else
    {
#pragma unroll
        for(int j = 0; j < DIM_X; j += DIM_Y)
            la_shared[j * DIM_X + td] = A[j * lda];
    }
    // end of reading a diagonal block of data

    __syncthreads();

// mirror second chunk
#pragma unroll
    for(int j = 0; j < (DIM_X / 2); j += DIM_Y)
        if(abs(tx - ty) > (j + (DIM_X / 2)))
            la_shared[DIM_X * ((DIM_X / 2) + j + ty) + tx]
                = la_shared[DIM_X * tx + (DIM_X / 2) + j + ty];

    // mirror elements in first chunk
    if(ty <= tx)
        la_shared[td] = la_shared[tx * DIM_X + ty];

#pragma unroll
    for(int j = DIM_Y; j < (DIM_X / 2); j += DIM_Y)
        if(abs(tx - ty) > j)
            la_shared[tx + (ty + j) * DIM_X] = la_shared[ty + j + tx * DIM_X];

    __syncthreads();

// compute first chunk
#pragma unroll
    for(int j = 0; j < (DIM_X / 2); j += DIM_Y)
        res += la_shared[(ty + j) * DIM_X + tx] * x_buff_shared[j + ty];

// compute second chunk
#pragma unroll
    for(int j = (DIM_X / 2); j < 2 * (DIM_X / 2); j += DIM_Y)
        res += la_shared[(ty + j) * DIM_X + tx] * x_buff_shared[j + ty];

    accum_shared[td] = res;
    __syncthreads();

    if(ty == 0)
    {
        res = T(0);
#pragma unroll
        for(int j = 0; j < DIM_Y; j++)
            res += accum_shared[j * DIM_X + tx];

        res *= alpha;

        if(beta)
            res += yold;

        if(tx < n_mod_DIM_X || bx < gridDim.x - 1)
            y[tx * incy] = res;
    }
}

template <rocblas_int DIM_X,
          rocblas_int DIM_Y,
          rocblas_int elements_per_thread,
          rocblas_int irregular_part,
          typename T>
ROCBLAS_KERNEL_ILF void rocblas_symv_kernel_upper_double_buffered_non_diagonal_generic_calc(
    rocblas_int n,
    T           alpha,
    const T* __restrict__ A,
    int64_t           lda,
    const T*          x,
    int64_t           incx,
    T*                y,
    int64_t           incy,
    const rocblas_int n_mod_DIM_X)
{
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int td  = (DIM_X * ty) + tx;
    const int tx_ = td % (DIM_X / 2);
    const int ty_ = td / (DIM_X / 2);
    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;

    // compute how many matrix blocks to be processed
    int count = bx / gridDim.y;

    T res_1_ = T(0);
    T res_2_ = T(0);
    T x1, x2;

    const T* xcopy;
    T*       ycopy;

    __shared__ T la_shared[DIM_X * (DIM_X / 2)];
    __shared__ T accum_shared[DIM_X * (2 * DIM_Y)];
    __shared__ T x_buff_shared[DIM_X];

    T A_reg_upper[elements_per_thread] = {T(0)};
    T A_reg_lower[elements_per_thread] = {T(0)};
    T treg[elements_per_thread]        = {T(0)};

    // Advance 'A' to the start a non-diagonal block
    A += DIM_X * bx * lda;
    // divide the work among y-direction of the grid
    A += (by * count) * DIM_X;

    // Advance 'x'
    xcopy = x + (bx * DIM_X) * incx;
    x += (by * count * DIM_X) * incx;

    // Advance 'y'
    ycopy = y;
    y += (bx * DIM_X) * incy;
    ycopy += (by * count * DIM_X) * incy;

    if(bx == 0)
        return;

    if(by == gridDim.y - 1)
        count += bx % gridDim.y;
    if(count == 0)
        return;

    // useful for last thread block only
    const int num_active_thread_cols = n_mod_DIM_X / elements_per_thread;

    // cache part of 'x' needed for computing res_1_ and res_2_
    if(bx == gridDim.x - 1) //last TB
    {
        if(ty == 0)
        {
            if(tx < n_mod_DIM_X)
                x_buff_shared[tx] = xcopy[tx * incx];
            else
                x_buff_shared[tx] = T(0);
        }
        // init shmem arrays to zeros
        accum_shared[ty_ * DIM_X + tx_]               = T(0);
        accum_shared[ty_ * DIM_X + tx_ + (DIM_X / 2)] = T(0);
        for(int k = 0; k < elements_per_thread; k++)
            la_shared[(ty_ * elements_per_thread + k) * (DIM_X / 2) + tx_] = T(0);
    }
    else // not the last TB
    {
        if(ty == 0)
            x_buff_shared[tx] = xcopy[tx * incx];
    }

    __syncthreads();

    const size_t j = ty_ * elements_per_thread * lda + tx_;

    // prefetch upper
    if(bx == gridDim.x - 1) // last TB "irregular"
    {
        if(ty_ < num_active_thread_cols)
        {
#pragma unroll
            for(int k = 0; k < elements_per_thread; k++)
                A_reg_upper[k] = A[j + k * lda];
        }
        else if(ty_ == num_active_thread_cols)
        {
#pragma unroll
            for(int k = 0; k < irregular_part; k++)
                A_reg_upper[k] = A[j + k * lda];
        }
    }
    else // not last TB
    {
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
            A_reg_upper[k] = A[j + k * lda];
    }

    x1 = x[tx_ * incx];

    for(int Vblocks = 0; Vblocks < count; Vblocks++)
    {
        res_1_ = T(0);
        res_2_ = T(0);

        x2 = x[(tx_ + (DIM_X / 2)) * incx];

        // prefetch lower
        if(bx == gridDim.x - 1)
        {
            if(ty_ < num_active_thread_cols)
            {
#pragma unroll
                for(int k = 0; k < elements_per_thread; k++)
                    A_reg_lower[k] = A[(DIM_X / 2) + j + k * lda];
            }
            else if(ty_ == num_active_thread_cols)
            {
#pragma unroll
                for(int k = 0; k < irregular_part; k++)
                    A_reg_lower[k] = A[(DIM_X / 2) + j + k * lda];
            }
        }
        else
        {
#pragma unroll
            for(int k = 0; k < elements_per_thread; k++)
                A_reg_lower[k] = A[(DIM_X / 2) + j + k * lda];
        } // end of prefetch lower

// compute upper
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
        {
            res_1_ += A_reg_upper[k] * x_buff_shared[ty_ * elements_per_thread + k];
            treg[k] += A_reg_upper[k] * x1;
        }

        // Advance to next block
        A += DIM_X;
        x += DIM_X * incx;

        // prefetch upper of next block
        if(Vblocks != count - 1)
        {
            if(bx == gridDim.x - 1)
            {
                if(ty_ < num_active_thread_cols)
                {
#pragma unroll
                    for(int k = 0; k < elements_per_thread; k++)
                        A_reg_upper[k] = A[j + k * lda];
                }
                else if(ty_ == num_active_thread_cols)
                {
#pragma unroll
                    for(int k = 0; k < irregular_part; k++)
                        A_reg_upper[k] = A[j + k * lda];
                }
            }
            else // not last TB
            {
#pragma unroll
                for(int k = 0; k < elements_per_thread; k++)
                    A_reg_upper[k] = A[j + k * lda];
            }
            x1 = x[tx_ * incx];
        }

#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
        {
            res_2_ += A_reg_lower[k] * x_buff_shared[ty_ * elements_per_thread + k];
            treg[k] += A_reg_lower[k] * x2;
        }

        // Horizontal block should be stored in global memory
        __syncthreads();
        accum_shared[ty_ * DIM_X + tx_]               = res_1_;
        accum_shared[ty_ * DIM_X + tx_ + (DIM_X / 2)] = res_2_;
        __syncthreads();

        if(ty == 0)
        {
            res_1_ = T(0);
#pragma unroll
            for(int k = 0; k < (2 * DIM_Y); k++)
                res_1_ += accum_shared[k * DIM_X + tx];

            // use atomics
            atomicAdd(&ycopy[tx * incy], res_1_ * alpha);
            ycopy += DIM_X * incy;
        }
    } // end of for loop on blocks

#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
        la_shared[(ty_ * elements_per_thread + k) * (DIM_X / 2) + tx_] = treg[k];

    __syncthreads();

    if(ty == 0)
    {
        treg[0] = T(0);
#pragma unroll
        for(int j = tx; j < tx + (DIM_X / 2); j++)
            treg[0] += la_shared[tx * (DIM_X / 2) + (j % (DIM_X / 2))];

        // use atomics
        if(tx < n_mod_DIM_X || bx < gridDim.x - 1)
            atomicAdd(&y[tx * incy], treg[0] * alpha);
    }
}

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T>
ROCBLAS_KERNEL_ILF void
    rocblas_symv_kernel_lower_double_buffered_diagonal_calc(rocblas_int n,
                                                            T           alpha,
                                                            const T* __restrict__ A,
                                                            int64_t lda,
                                                            const T* __restrict__ x,
                                                            int64_t incx,
                                                            T       beta,
                                                            T* __restrict__ y,
                                                            int64_t incy)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int td = (DIM_X * ty) + tx;
    const int bx = blockIdx.x;

    T res  = T(0);
    T yold = T(0);

    __shared__ T la[DIM_X * DIM_X];
    __shared__ T buff[DIM_X];
    __shared__ T accum[DIM_X * (2 * DIM_Y)];

    // Advance 'y'
    y += (bx * DIM_X) * incy;

    // Early return when alpha == 0
    if(!alpha)
    {
        if(ty == 0)
            y[tx * incy] *= beta;
        return;
    }

    // Advance 'A' to start of diagonal blocks first
    A += DIM_X * bx * (lda + 1);

    // Advance 'A' to start row for each thread inside the diagonal block
    A += ty * lda + tx;

    // Advance 'x'
    x += (bx * DIM_X) * incx;

    if(ty == 0)
    {
        // skip beta * y when beta == 0
        if(beta)
            yold = beta * y[tx * incy];
        buff[tx] = x[tx * incx];
    }

// load first chunk
#pragma unroll
    for(int k = 0; k < (DIM_X / 2); k += DIM_Y)
        la[td + k * DIM_X] = A[k * lda];

    // Advance to second chunk
    A += (DIM_X / 2) * lda;

    // load second chunk
    if(tx
       >= (DIM_X
           / 2)) // even warps will load un-necessary elements in the 2nd chunck of diagonal block
    {
#pragma unroll
        for(int k = 0; k < (DIM_X / 2); k += DIM_Y)
            la[DIM_X * ((DIM_X / 2) + k + ty) + tx] = A[k * lda];
    }

    __syncthreads();

    // mirror necessary elements in first chunk
    if(ty > tx)
        la[td] = la[tx * DIM_X + ty];

#pragma unroll
    for(int k = DIM_Y; k < (DIM_X / 2); k += DIM_Y)
        if(abs(tx - ty) < k)
            la[tx + (ty + k) * DIM_X] = la[ty + k + tx * DIM_X];

// mirror second chunk
#pragma unroll
    for(int k = 0; k < (DIM_X / 2); k += DIM_Y)
        if(abs(tx - ty) < (k + (DIM_X / 2)))
            la[DIM_X * ((DIM_X / 2) + k + ty) + tx] = la[DIM_X * tx + (DIM_X / 2) + k + ty];

    __syncthreads();

// compute first chunk
#pragma unroll
    for(int k = 0; k < (DIM_X / 2); k += DIM_Y)
        res += la[(ty + k) * DIM_X + tx] * buff[k + ty];

// compute second chunk
#pragma unroll
    for(int k = (DIM_X / 2); k < 2 * (DIM_X / 2); k += DIM_Y)
        res += la[(ty + k) * DIM_X + tx] * buff[k + ty];

    accum[td] = res;

    __syncthreads();

    if(ty == 0)
    {
        res = T(0);
#pragma unroll
        for(int k = 0; k < DIM_Y; k++)
            res += accum[k * DIM_X + tx];
        res *= alpha;

        if(beta)
            res += yold;

        y[tx * incy] = res;
    }
}

template <rocblas_int DIM_X, rocblas_int DIM_Y, rocblas_int elements_per_thread, typename T>
ROCBLAS_KERNEL_ILF void
    rocblas_symv_kernel_lower_double_buffered_non_diagonal_calc(rocblas_int n,
                                                                T           alpha,
                                                                const T* __restrict__ A,
                                                                int64_t  lda,
                                                                const T* x,
                                                                int64_t  incx,
                                                                T*       y,
                                                                int64_t  incy)
{
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;
    const int td  = (DIM_X * ty) + tx;
    const int tx_ = td % (DIM_X / 2);
    const int ty_ = td / (DIM_X / 2);
    const T*  xcopy;
    T*        ycopy;

    // compute how many matrix blocks to be processed
    int count = (gridDim.x - bx - 1) / gridDim.y;

    T A_reg_upper[elements_per_thread];
    T A_reg_lower[elements_per_thread];
    T treg[elements_per_thread] = {T(0)};

    __shared__ T la[DIM_X * (DIM_X / 2)];
    __shared__ T accum[DIM_X * (2 * DIM_Y)];
    __shared__ T xbuff[DIM_X];

    if(bx == gridDim.x - 1)
        return;
    {
        // Advance 'A' to start of diagonal blocks first
        A += DIM_X * bx * (lda + 1);

        // divide work among the y-direction of the grid
        A += (by * count) * DIM_X;

        // Advance 'x'
        x += (bx * DIM_X) * incx;
        xcopy = x;
        x += (by * count * DIM_X) * incx;

        if(ty == 0)
            xbuff[tx] = xcopy[tx * incx];

        // Advance 'y'
        y += (bx * DIM_X) * incy;
        ycopy = y;
        ycopy += (by * count * DIM_X) * incy;
    }

    if(by == gridDim.y - 1)
        count += (gridDim.x - bx - 1) % gridDim.y;

    if(count == 0)
        return;

    T res_1_ = T(0);
    T res_2_ = T(0);
    T x1     = T(0);
    T x2     = T(0);

    const size_t j = ty_ * elements_per_thread * lda + tx_;

    A += DIM_X;
    x += DIM_X * incx;

    __syncthreads();

// read upper
#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
        A_reg_upper[k] = A[j + k * lda];

    for(int Vblocks = 0; Vblocks < count; Vblocks++)
    {

        res_1_ = T(0);
        res_2_ = T(0);

        x1 = x[incx * tx_];
        x2 = x[incx * (tx_ + (DIM_X / 2))];

// read lower
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
            A_reg_lower[k] = A[(DIM_X / 2) + j + k * lda];

// compute upper
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
        {
            res_1_ += A_reg_upper[k] * xbuff[ty_ * elements_per_thread + k];
            treg[k] += A_reg_upper[k] * x1;
        }

        A += DIM_X;
        x += DIM_X * incx;

        // read upper from next block
        if(Vblocks != count - 1)
        {
#pragma unroll
            for(int k = 0; k < elements_per_thread; k++)
                A_reg_upper[k] = A[j + k * lda];
        }

// compute lower
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
        {
            res_2_ += A_reg_lower[k] * xbuff[ty_ * elements_per_thread + k];
            treg[k] += A_reg_lower[k] * x2;
        }

        // Horizontal block should be stored in global memory
        __syncthreads();
        accum[ty_ * DIM_X + tx_]               = res_1_;
        accum[ty_ * DIM_X + tx_ + (DIM_X / 2)] = res_2_;
        __syncthreads();
        if(ty == 0)
        {
            ycopy += DIM_X * incy;
            res_1_ = T(0);
#pragma unroll
            for(int k = 0; k < (2 * DIM_Y); k++)
                res_1_ += accum[k * DIM_X + tx];

            // use atomics
            atomicAdd(&ycopy[tx * incy], res_1_ * alpha);
        }
    }

// reduction of treg
#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
        la[(ty_ * elements_per_thread + k) * (DIM_X / 2) + tx_] = treg[k];

    __syncthreads();

    if(bx != gridDim.x - 1)
    {
        if(ty == 0)
        {
            treg[0] = T(0); // as a temporary accumulator
#pragma unroll
            for(int k = tx; k < tx + (DIM_X / 2); k++)
                treg[0] += la[tx * (DIM_X / 2) + (k % (DIM_X / 2))];

            // use atomics
            atomicAdd(&y[tx * incy], treg[0] * alpha);
        }
    }
}

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T>
ROCBLAS_KERNEL_ILF void
    rocblas_symv_kernel_lower_double_buffered_diagonal_generic_calc(rocblas_int n,
                                                                    T           alpha,
                                                                    const T* __restrict__ A,
                                                                    int64_t lda,
                                                                    const T* __restrict__ x,
                                                                    int64_t incx,
                                                                    T       beta,
                                                                    T* __restrict__ y,
                                                                    int64_t           incy,
                                                                    const rocblas_int n_mod_DIM_X)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int td = (DIM_X * ty) + tx;

    T res  = T(0);
    T yold = T(0);

    __shared__ T la_shared[DIM_X * DIM_X];
    __shared__ T x_buff_shared[DIM_X];
    __shared__ T accum_shared[DIM_X * (2 * DIM_Y)];

    // Advance y
    y += (bx * DIM_X) * incy;

    // Early return when alpha == 0
    if(!alpha)
    {
        if(ty == 0 && (tx < n_mod_DIM_X || bx < gridDim.x - 1))
            y[tx * incy] *= beta;

        return;
    }

    // Advance 'A' to start of diagonal blocks first
    A += DIM_X * bx * (lda + 1);

    // Advance 'A' to start row for each thread inside the diagonal block
    A += ty * lda + tx;

    // Advance x
    x += (bx * DIM_X) * incx;

    // load part of vector x
    if(bx == gridDim.x - 1)
    {
        if(ty == 0)
        {
            if(tx < n_mod_DIM_X)
            {
                x_buff_shared[tx] = x[tx * incx];

                // skip beta * y when beta == 0
                if(beta)
                    yold = beta * y[tx * incy];
            }
            else
            {
                x_buff_shared[tx] = T(0);
                yold              = T(0);
            }
        }
    }
    else
    {
        if(ty == 0)
        {
            x_buff_shared[tx] = x[incx * tx];

            // skip beta * y when beta == 0
            if(beta)
                yold = beta * y[tx * incy];
        }
    } // end of load part of vector x

    // init shmem (last TB only)
    if(bx == gridDim.x - 1)
    {
#pragma unroll
        for(int j = 0; j < DIM_X; j += DIM_Y)
            la_shared[j * DIM_X + td] = T(0);
    }

    // load a block of data
    if(bx == gridDim.x - 1)
    {
        // These threads should not read any useful data
        if(tx < n_mod_DIM_X)
        {
            int j;
            for(j = 0; j < n_mod_DIM_X / DIM_Y; j++)
                la_shared[(j * DIM_Y) * DIM_X + td] = A[(j * DIM_Y) * lda];

            if(ty < (n_mod_DIM_X % DIM_Y))
                la_shared[(j * DIM_Y) * DIM_X + td] = A[(j * DIM_Y) * lda];
        }
    }
    else
    {
#pragma unroll
        for(int j = 0; j < DIM_X; j += DIM_Y)
            la_shared[j * DIM_X + td] = A[j * lda];
    }
    // end of reading a diagonal block of data

    __syncthreads();

    // mirror necessary elements in first chunk
    if(ty > tx)
        la_shared[td] = la_shared[tx * DIM_X + ty];

#pragma unroll
    for(int j = DIM_Y; j < (DIM_X / 2); j += DIM_Y)
        if(abs(tx - ty) < j)
            la_shared[tx + (ty + j) * DIM_X] = la_shared[ty + j + tx * DIM_X];

// mirror second chunk
#pragma unroll
    for(int j = 0; j < (DIM_X / 2); j += DIM_Y)
        if(abs(tx - ty) < (j + (DIM_X / 2)))
            la_shared[DIM_X * ((DIM_X / 2) + j + ty) + tx]
                = la_shared[DIM_X * tx + (DIM_X / 2) + j + ty];

    __syncthreads();

// compute first chunk
#pragma unroll
    for(int j = 0; j < (DIM_X / 2); j += DIM_Y)
        res += la_shared[(ty + j) * DIM_X + tx] * x_buff_shared[j + ty];

// compute second chunk
#pragma unroll
    for(int j = (DIM_X / 2); j < 2 * (DIM_X / 2); j += DIM_Y)
        res += la_shared[(ty + j) * DIM_X + tx] * x_buff_shared[j + ty];

    accum_shared[td] = res;
    __syncthreads();
    if(ty == 0)
    {
        res = T(0);
#pragma unroll
        for(int j = 0; j < DIM_Y; j++)
            res += accum_shared[j * DIM_X + tx];

        res *= alpha;

        if(beta)
            res += yold;

        if(bx == gridDim.x - 1)
        {
            if(tx < n_mod_DIM_X)
                y[tx * incy] = res;
        }
        else
        {
            y[tx * incy] = res;
        }
    }
}

template <rocblas_int DIM_X, rocblas_int DIM_Y, rocblas_int elements_per_thread, typename T>
ROCBLAS_KERNEL_ILF void rocblas_symv_kernel_lower_double_buffered_non_diagonal_generic_calc(
    rocblas_int n,
    T           alpha,
    const T* __restrict__ A,
    int64_t           lda,
    const T*          x,
    int64_t           incx,
    T*                y,
    int64_t           incy,
    const rocblas_int n_mod_DIM_X)
{
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;
    const int td  = (DIM_X * ty) + tx;
    const int tx_ = td % (DIM_X / 2);
    const int ty_ = td / (DIM_X / 2);
    const T*  xcopy;
    T*        ycopy;

    int count = (gridDim.x - bx - 1 - 1) / gridDim.y; // -1 for the generic block at the bottom

    T A_reg_upper[elements_per_thread];
    T A_reg_lower[elements_per_thread];
    T treg[elements_per_thread] = {T(0)};

    T res_1_ = T(0);
    T res_2_ = T(0);
    T x1     = T(0);
    T x2     = T(0);

    __shared__ T la_shared[DIM_X * (DIM_X / 2)];
    __shared__ T accum_shared[DIM_X * (2 * DIM_Y)];
    __shared__ T x_buff_shared[DIM_X];

    if(bx == gridDim.x - 1)
        return;

    // Advance 'A' to start of diagonal blocks first
    A += DIM_X * bx * (lda + 1);

    // divide work among the y-direction of the grid
    A += (by * count) * DIM_X;

    // Advance 'x'
    x += (bx * DIM_X) * incx;
    xcopy = x;
    x += (by * count * DIM_X) * incx;

    if(ty == 0)
        x_buff_shared[tx] = xcopy[tx * incx];

    //Advance 'y'
    y += (bx * DIM_X) * incy;
    ycopy = y;
    ycopy += (by * count * DIM_X) * incy;

    if(by == gridDim.y - 1)
        count += ((gridDim.x - bx - 1 - 1) % gridDim.y); // -1 for the generic block at the bottom

    if(by != gridDim.y - 1)
    {
        if(count == 0)
            return;
    }

    size_t j = ty_ * elements_per_thread * lda + tx_;

    __syncthreads();

    A += DIM_X;
    x += DIM_X * incx;

    if(bx < gridDim.x - 2) // to prevent out of bound access
    {
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
            A_reg_upper[k] = A[j + k * lda];
        x1 = x[tx_ * incx];
    }

    A -= DIM_X;
    x -= DIM_X * incx;

    for(int Vblocks = 0; Vblocks < count; Vblocks++)
    {
        A += DIM_X;
        x += DIM_X * incx;

        res_1_ = T(0);
        res_2_ = T(0);

        x2 = x[(tx_ + (DIM_X / 2)) * incx];

#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
            A_reg_lower[k] = A[(DIM_X / 2) + j + k * lda];

#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
        {
            res_1_ += A_reg_upper[k] * x_buff_shared[ty_ * elements_per_thread + k];
            treg[k] += A_reg_upper[k] * x1;
        }

        A += DIM_X;
        x += DIM_X * incx;

        if(Vblocks != count - 1)
        {
#pragma unroll
            for(int k = 0; k < elements_per_thread; k++)
                A_reg_upper[k] = A[j + k * lda];
            x1 = x[tx_ * incx];
        }

        A -= DIM_X;
        x -= DIM_X * incx;

#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
        {
            res_2_ += A_reg_lower[k] * x_buff_shared[ty_ * elements_per_thread + k];
            treg[k] += A_reg_lower[k] * x2;
        }

        // Horizontal block should be stored in global memory
        __syncthreads();
        accum_shared[ty_ * DIM_X + tx_]               = res_1_;
        accum_shared[ty_ * DIM_X + tx_ + (DIM_X / 2)] = res_2_;
        __syncthreads();
        if(ty == 0)
        {
            ycopy += DIM_X * incy;
            res_1_ = T(0);
#pragma unroll
            for(int k = 0; k < (2 * DIM_Y); k++)
                res_1_ += accum_shared[k * DIM_X + tx];

            // use atomics
            atomicAdd(&ycopy[tx * incy], res_1_ * alpha);
        }
    } // end of for loop on blocks

    //////////////////////////////////////////////////
    // last irregular tile
    if(by == gridDim.y - 1)
    {
        res_1_ = T(0);
        res_2_ = T(0);

        A += DIM_X;
        x += DIM_X * incx;

#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
        {
            A_reg_upper[k] = T(0);
            A_reg_lower[k] = T(0);
        }

        if(tx_ < n_mod_DIM_X)
        {
#pragma unroll
            for(int k = 0; k < elements_per_thread; k++)
                A_reg_upper[k] = A[j + k * lda];

            x1 = x[tx_ * incx];
        }

        if((tx_ + (DIM_X / 2)) < n_mod_DIM_X)
        {
#pragma unroll
            for(int k = 0; k < elements_per_thread; k++)
                A_reg_lower[k] = A[(DIM_X / 2) + j + k * lda];

            x2 = x[(tx_ + (DIM_X / 2)) * incx];
        }

#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
        {
            res_1_ += A_reg_upper[k] * x_buff_shared[ty_ * elements_per_thread + k];
            treg[k] += A_reg_upper[k] * x1;
        }

#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
        {
            res_2_ += A_reg_lower[k] * x_buff_shared[ty_ * elements_per_thread + k];
            treg[k] += A_reg_lower[k] * x2;
        }

        // Horizontal block reduction
        __syncthreads();
        accum_shared[ty_ * DIM_X + tx_]               = res_1_;
        accum_shared[ty_ * DIM_X + tx_ + (DIM_X / 2)] = res_2_;
        __syncthreads();
        if(ty == 0)
        {
            ycopy += DIM_X * incy;
            res_1_ = T(0);
#pragma unroll
            for(int k = 0; k < (2 * DIM_Y); k++)
                res_1_ += accum_shared[k * DIM_X + tx];

            // use atomics
            if(tx < n_mod_DIM_X)
                atomicAdd(&ycopy[tx * incy], res_1_ * alpha);
        }
    }

#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
        la_shared[(ty_ * elements_per_thread + k) * (DIM_X / 2) + tx_] = treg[k];

    __syncthreads(); // important

    if(ty == 0)
    {
        treg[0] = T(0); // tmp accumulator
#pragma unroll
        for(int k = tx; k < tx + (DIM_X / 2); k++)
            treg[0] += la_shared[tx * (DIM_X / 2) + (k % (DIM_X / 2))];

        atomicAdd(&y[tx * incy], treg[0] * alpha);
    }
}

template <bool        IS_HEMV,
          rocblas_int NB_X,
          rocblas_int NB_Y,
          rocblas_int bank_shift,
          rocblas_int half_NB_X,
          rocblas_int quarter_NB_X,
          typename T_index,
          typename U,
          typename V,
          typename W>
ROCBLAS_KERNEL(NB_X* NB_Y)
rocblas_hemvn_kernel_upper(rocblas_int    n,
                           U              alpha_device_host,
                           rocblas_stride stride_alpha,
                           V              Aa,
                           rocblas_stride shifta,
                           T_index        lda,
                           rocblas_stride strideA,
                           V              xa,
                           rocblas_stride shiftx,
                           T_index        incx,
                           rocblas_stride stridex,
                           U              beta_device_host,
                           rocblas_stride stride_beta,
                           W              workspace,
                           rocblas_int    batch_count)
{
    rocblas_int num_threads = blockDim.x * blockDim.y * blockDim.z;

    if(NB_X * NB_Y != num_threads)
        return;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto alpha = load_scalar(alpha_device_host, batch, stride_alpha);
        auto beta  = load_scalar(beta_device_host, batch, stride_beta);

        if(!alpha && beta == 1)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        const auto* A = cond_load_ptr_batch(alpha, Aa, batch, shifta, strideA);
        const auto* x = cond_load_ptr_batch(alpha, xa, batch, shiftx, stridex);

        rocblas_hemvn_kernel_upper_calc<IS_HEMV,
                                        NB_X,
                                        bank_shift,
                                        half_NB_X,
                                        quarter_NB_X,
                                        T_index>(n, alpha, A, lda, x, incx, workspace, batch);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <bool        IS_HEMV,
          rocblas_int NB_X,
          rocblas_int NB_Y,
          rocblas_int bank_shift,
          rocblas_int half_NB_X,
          rocblas_int quarter_NB_X,
          typename T_lda,
          typename U,
          typename V,
          typename W>
ROCBLAS_KERNEL(NB_X* NB_Y)
rocblas_hemvn_kernel_lower(rocblas_int    n,
                           U              alpha_device_host,
                           rocblas_stride stride_alpha,
                           V              Aa,
                           rocblas_stride shifta,
                           T_lda          lda,
                           rocblas_stride strideA,
                           V              xa,
                           rocblas_stride shiftx,
                           T_lda          incx,
                           rocblas_stride stridex,
                           U              beta_device_host,
                           rocblas_stride stride_beta,
                           W              workspace,
                           rocblas_int    batch_count)
{
    rocblas_int num_threads = blockDim.x * blockDim.y * blockDim.z;

    if(NB_X * NB_Y != num_threads)
        return;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto alpha = load_scalar(alpha_device_host, batch, stride_alpha);
        auto beta  = load_scalar(beta_device_host, batch, stride_beta);

        if(!alpha && beta == 1)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        const auto* A = cond_load_ptr_batch(alpha, Aa, batch, shifta, strideA);
        const auto* x = cond_load_ptr_batch(alpha, xa, batch, shiftx, stridex);

        rocblas_hemvn_kernel_lower_calc<IS_HEMV, NB_X, bank_shift, half_NB_X, quarter_NB_X, T_lda>(
            n, alpha, A, lda, x, incx, workspace, batch);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename TStruct, typename V, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_symv_kernel_upper_double_buffered_diagonal(bool           host_ptr_mode,
                                                   rocblas_int    n,
                                                   TStruct        alpha_device_host,
                                                   rocblas_stride stride_alpha,
                                                   V              Aa,
                                                   rocblas_stride shifta,
                                                   int64_t        lda,
                                                   rocblas_stride strideA,
                                                   V              xa,
                                                   rocblas_stride shiftx,
                                                   int64_t        incx,
                                                   rocblas_stride stridex,
                                                   TStruct        beta_device_host,
                                                   rocblas_stride stride_beta,
                                                   TPtr __restrict__ ya,
                                                   rocblas_stride shifty,
                                                   int64_t        incy,
                                                   rocblas_stride stridey,
                                                   rocblas_int    batch_count)
{
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif
        const auto alpha = host_ptr_mode ? alpha_device_host.value
                                         : load_scalar(alpha_device_host.ptr, batch, stride_alpha);
        const auto beta  = host_ptr_mode ? beta_device_host.value
                                         : load_scalar(beta_device_host.ptr, batch, stride_beta);

        if(!alpha && beta == 1)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        const auto* A = cond_load_ptr_batch(alpha, Aa, batch, shifta, strideA);
        const auto* x = cond_load_ptr_batch(alpha, xa, batch, shiftx, stridex);
        auto*       y = load_ptr_batch(ya, batch, shifty, stridey);

        rocblas_symv_kernel_upper_double_buffered_diagonal_calc<DIM_X, DIM_Y>(
            n, alpha, A, lda, x, incx, beta, y, incy);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <rocblas_int DIM_X,
          rocblas_int DIM_Y,
          rocblas_int elements_per_thread,
          typename TStruct,
          typename V,
          typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_symv_kernel_upper_double_buffered_non_diagonal(bool           host_ptr_mode,
                                                       rocblas_int    n,
                                                       TStruct        alpha_device_host,
                                                       rocblas_stride stride_alpha,
                                                       V              Aa,
                                                       rocblas_stride shifta,
                                                       int64_t        lda,
                                                       rocblas_stride strideA,
                                                       V              xa,
                                                       rocblas_stride shiftx,
                                                       int64_t        incx,
                                                       rocblas_stride stridex,
                                                       TPtr __restrict__ ya,
                                                       rocblas_stride shifty,
                                                       int64_t        incy,
                                                       rocblas_stride stridey,
                                                       rocblas_int    batch_count)
{
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        const auto alpha = host_ptr_mode ? alpha_device_host.value
                                         : load_scalar(alpha_device_host.ptr, batch, stride_alpha);

        if(!alpha)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        const auto* A = cond_load_ptr_batch(alpha, Aa, batch, shifta, strideA);
        const auto* x = cond_load_ptr_batch(alpha, xa, batch, shiftx, stridex);
        auto*       y = load_ptr_batch(ya, batch, shifty, stridey);

        rocblas_symv_kernel_upper_double_buffered_non_diagonal_calc<DIM_X,
                                                                    DIM_Y,
                                                                    elements_per_thread>(
            n, alpha, A, lda, x, incx, y, incy);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename TStruct, typename V, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_symv_kernel_upper_double_buffered_diagonal_generic(bool           host_ptr_mode,
                                                           rocblas_int    n,
                                                           TStruct        alpha_device_host,
                                                           rocblas_stride stride_alpha,
                                                           V              Aa,
                                                           rocblas_stride shifta,
                                                           int64_t        lda,
                                                           rocblas_stride strideA,
                                                           V              xa,
                                                           rocblas_stride shiftx,
                                                           int64_t        incx,
                                                           rocblas_stride stridex,
                                                           TStruct        beta_device_host,
                                                           rocblas_stride stride_beta,
                                                           TPtr __restrict__ ya,
                                                           rocblas_stride    shifty,
                                                           int64_t           incy,
                                                           rocblas_stride    stridey,
                                                           const rocblas_int mod,
                                                           rocblas_int       batch_count)
{
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        const auto alpha = host_ptr_mode ? alpha_device_host.value
                                         : load_scalar(alpha_device_host.ptr, batch, stride_alpha);
        const auto beta  = host_ptr_mode ? beta_device_host.value
                                         : load_scalar(beta_device_host.ptr, batch, stride_beta);

        if(!alpha && beta == 1)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        const auto* A = cond_load_ptr_batch(alpha, Aa, batch, shifta, strideA);
        const auto* x = cond_load_ptr_batch(alpha, xa, batch, shiftx, stridex);
        auto*       y = load_ptr_batch(ya, batch, shifty, stridey);

        rocblas_symv_kernel_upper_double_buffered_diagonal_generic_calc<DIM_X, DIM_Y>(
            n, alpha, A, lda, x, incx, beta, y, incy, mod);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <rocblas_int DIM_X,
          rocblas_int DIM_Y,
          rocblas_int elements_per_thread,
          rocblas_int irregular_part,
          typename TStruct,
          typename V,
          typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_symv_kernel_upper_double_buffered_non_diagonal_generic(bool           host_ptr_mode,
                                                               rocblas_int    n,
                                                               TStruct        alpha_device_host,
                                                               rocblas_stride stride_alpha,
                                                               V              Aa,
                                                               rocblas_stride shifta,
                                                               int64_t        lda,
                                                               rocblas_stride strideA,
                                                               V              xa,
                                                               rocblas_stride shiftx,
                                                               int64_t        incx,
                                                               rocblas_stride stridex,
                                                               TPtr __restrict__ ya,
                                                               rocblas_stride    shifty,
                                                               int64_t           incy,
                                                               rocblas_stride    stridey,
                                                               const rocblas_int mod,
                                                               rocblas_int       batch_count)
{
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif
        const auto alpha = host_ptr_mode ? alpha_device_host.value
                                         : load_scalar(alpha_device_host.ptr, batch, stride_alpha);

        if(!alpha)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        const auto* A = cond_load_ptr_batch(alpha, Aa, batch, shifta, strideA);
        const auto* x = cond_load_ptr_batch(alpha, xa, batch, shiftx, stridex);
        auto*       y = load_ptr_batch(ya, batch, shifty, stridey);

        rocblas_symv_kernel_upper_double_buffered_non_diagonal_generic_calc<DIM_X,
                                                                            DIM_Y,
                                                                            elements_per_thread,
                                                                            irregular_part>(
            n, alpha, A, lda, x, incx, y, incy, mod);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename TStruct, typename V, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_symv_kernel_lower_double_buffered_diagonal(bool           host_ptr_mode,
                                                   rocblas_int    n,
                                                   TStruct        alpha_device_host,
                                                   rocblas_stride stride_alpha,
                                                   V              Aa,
                                                   rocblas_stride shifta,
                                                   int64_t        lda,
                                                   rocblas_stride strideA,
                                                   V              xa,
                                                   rocblas_stride shiftx,
                                                   int64_t        incx,
                                                   rocblas_stride stridex,
                                                   TStruct        beta_device_host,
                                                   rocblas_stride stride_beta,
                                                   TPtr __restrict__ ya,
                                                   rocblas_stride shifty,
                                                   int64_t        incy,
                                                   rocblas_stride stridey,
                                                   rocblas_int    batch_count)
{
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        const auto alpha = host_ptr_mode ? alpha_device_host.value
                                         : load_scalar(alpha_device_host.ptr, batch, stride_alpha);
        const auto beta  = host_ptr_mode ? beta_device_host.value
                                         : load_scalar(beta_device_host.ptr, batch, stride_beta);

        if(!alpha && beta == 1)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        const auto* A = cond_load_ptr_batch(alpha, Aa, batch, shifta, strideA);
        const auto* x = cond_load_ptr_batch(alpha, xa, batch, shiftx, stridex);
        auto*       y = load_ptr_batch(ya, batch, shifty, stridey);

        rocblas_symv_kernel_lower_double_buffered_diagonal_calc<DIM_X, DIM_Y>(
            n, alpha, A, lda, x, incx, beta, y, incy);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <rocblas_int DIM_X,
          rocblas_int DIM_Y,
          rocblas_int elements_per_thread,
          typename TStruct,
          typename V,
          typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_symv_kernel_lower_double_buffered_non_diagonal(bool           host_ptr_mode,
                                                       rocblas_int    n,
                                                       TStruct        alpha_device_host,
                                                       rocblas_stride stride_alpha,
                                                       V              Aa,
                                                       rocblas_stride shifta,
                                                       int64_t        lda,
                                                       rocblas_stride strideA,
                                                       V              xa,
                                                       rocblas_stride shiftx,
                                                       int64_t        incx,
                                                       rocblas_stride stridex,
                                                       TPtr __restrict__ ya,
                                                       rocblas_stride shifty,
                                                       int64_t        incy,
                                                       rocblas_stride stridey,
                                                       rocblas_int    batch_count)
{
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        const auto alpha = host_ptr_mode ? alpha_device_host.value
                                         : load_scalar(alpha_device_host.ptr, batch, stride_alpha);

        if(!alpha)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        const auto* A = cond_load_ptr_batch(alpha, Aa, batch, shifta, strideA);
        const auto* x = cond_load_ptr_batch(alpha, xa, batch, shiftx, stridex);
        auto*       y = load_ptr_batch(ya, batch, shifty, stridey);

        rocblas_symv_kernel_lower_double_buffered_non_diagonal_calc<DIM_X,
                                                                    DIM_Y,
                                                                    elements_per_thread>(
            n, alpha, A, lda, x, incx, y, incy);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename TStruct, typename V, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_symv_kernel_lower_double_buffered_diagonal_generic(bool           host_ptr_mode,
                                                           rocblas_int    n,
                                                           TStruct        alpha_device_host,
                                                           rocblas_stride stride_alpha,
                                                           V              Aa,
                                                           rocblas_stride shifta,
                                                           int64_t        lda,
                                                           rocblas_stride strideA,
                                                           V              xa,
                                                           rocblas_stride shiftx,
                                                           int64_t        incx,
                                                           rocblas_stride stridex,
                                                           TStruct        beta_device_host,
                                                           rocblas_stride stride_beta,
                                                           TPtr __restrict__ ya,
                                                           rocblas_stride    shifty,
                                                           int64_t           incy,
                                                           rocblas_stride    stridey,
                                                           const rocblas_int mod,
                                                           rocblas_int       batch_count)
{
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif
        const auto alpha = host_ptr_mode ? alpha_device_host.value
                                         : load_scalar(alpha_device_host.ptr, batch, stride_alpha);
        const auto beta  = host_ptr_mode ? beta_device_host.value
                                         : load_scalar(beta_device_host.ptr, batch, stride_beta);

        if(!alpha && beta == 1)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        const auto* A = cond_load_ptr_batch(alpha, Aa, batch, shifta, strideA);
        const auto* x = cond_load_ptr_batch(alpha, xa, batch, shiftx, stridex);
        auto*       y = load_ptr_batch(ya, batch, shifty, stridey);

        rocblas_symv_kernel_lower_double_buffered_diagonal_generic_calc<DIM_X, DIM_Y>(
            n, alpha, A, lda, x, incx, beta, y, incy, mod);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <rocblas_int DIM_X,
          rocblas_int DIM_Y,
          rocblas_int elements_per_thread,
          typename TStruct,
          typename V,
          typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_symv_kernel_lower_double_buffered_non_diagonal_generic(bool           host_ptr_mode,
                                                               rocblas_int    n,
                                                               TStruct        alpha_device_host,
                                                               rocblas_stride stride_alpha,
                                                               V              Aa,
                                                               rocblas_stride shifta,
                                                               int64_t        lda,
                                                               rocblas_stride strideA,
                                                               V              xa,
                                                               rocblas_stride shiftx,
                                                               int64_t        incx,
                                                               rocblas_stride stridex,
                                                               TPtr __restrict__ ya,
                                                               rocblas_stride    shifty,
                                                               int64_t           incy,
                                                               rocblas_stride    stridey,
                                                               const rocblas_int mod,
                                                               rocblas_int       batch_count)
{
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif
        const auto alpha = host_ptr_mode ? alpha_device_host.value
                                         : load_scalar(alpha_device_host.ptr, batch, stride_alpha);

        if(!alpha)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        const auto* A = cond_load_ptr_batch(alpha, Aa, batch, shifta, strideA);
        const auto* x = cond_load_ptr_batch(alpha, xa, batch, shiftx, stridex);
        auto*       y = load_ptr_batch(ya, batch, shifty, stridey);

        rocblas_symv_kernel_lower_double_buffered_non_diagonal_generic_calc<DIM_X,
                                                                            DIM_Y,
                                                                            elements_per_thread>(
            n, alpha, A, lda, x, incx, y, incy, mod);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

/**
  *  V is either: const T* OR const T* const*
  *  W is either:       T* OR       T* const*
  *  Note stride_alpha and stride_beta are only used AND only tested by rocSOLVER
  *  These strided scalar fetches are only supported for device_ptr mode
  */
template <bool IS_HEMV, typename T, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_symv_hemv_launcher(rocblas_handle handle,
                                                   rocblas_fill   uplo,
                                                   int            n,
                                                   TScal          alpha,
                                                   rocblas_stride stride_alpha,
                                                   TConstPtr      A,
                                                   rocblas_stride offseta,
                                                   int64_t        lda,
                                                   rocblas_stride strideA,
                                                   TConstPtr      x,
                                                   rocblas_stride offsetx,
                                                   int64_t        incx,
                                                   rocblas_stride stridex,
                                                   TScal          beta,
                                                   rocblas_stride stride_beta,
                                                   TPtr           y,
                                                   rocblas_stride offsety,
                                                   int64_t        incy,
                                                   rocblas_stride stridey,
                                                   int            batch_count,
                                                   T*             workspace)
{
    //quick return
    if(!n || !batch_count)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->get_stream();

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    auto shiftx = incx < 0 ? offsetx - incx * (n - 1) : offsetx;
    auto shifty = incy < 0 ? offsety - incy * (n - 1) : offsety;

    //Identifying the precision to have an appropriate optimization
    static constexpr bool is_float
        = std::is_same_v<TPtr, rocblas_float*> || std::is_same_v<TPtr, rocblas_float* const*>;

    static constexpr bool is_double
        = std::is_same_v<TPtr, double*> || std::is_same_v<TPtr, double* const*>;

    bool i64_indices = size_t(n) * lda > c_i32_max || size_t(n) * std::abs(incx) > c_i32_max
                       || size_t(n) * std::abs(incy) > c_i32_max;

    const bool is_atomics_allowed = handle->atomics_mode == rocblas_atomics_allowed ? true : false;

    //Identifying the specific architecture to have an appropriate optimization
    bool is_gfx90a = handle->getArch() == 910 ? true : false;
    bool is_gfx908 = handle->getArch() == 908 ? true : false;

    int batches = handle->getBatchGridDim((int)batch_count);

    static constexpr int HEMV_DIM_X         = rocblas_hemv_DIM_X();
    static constexpr int HEMV_DIM_Y         = 4;
    static constexpr int bank_shift         = 33;
    static constexpr int half_HEMV_DIM_X    = 32;
    static constexpr int quarter_HEMV_DIM_X = 16;
    rocblas_int          blocks             = (n - 1) / (HEMV_DIM_X) + 1;

    dim3 hemv_grid(blocks, 1, batches);
    dim3 hemv_threads(HEMV_DIM_X, HEMV_DIM_Y);
    dim3 hemv_threads_sum(HEMV_DIM_X);

#define hemv_kernel_KARGS(alpha_, beta_)                                                           \
    hemv_grid, hemv_threads, 0, rocblas_stream, n, alpha_, stride_alpha, A, offseta, lda, strideA, \
        x, shiftx, incx, stridex, beta_, stride_beta, workspace, batch_count

#define hemv_kernel_sum_KARGS(alpha_, beta_)                                                     \
    hemv_grid, hemv_threads_sum, 0, rocblas_stream, n, alpha_, stride_alpha, beta_, stride_beta, \
        y, shifty, incy, stridey, workspace, batch_count

    if(uplo == rocblas_fill_upper)
    {
        if(is_atomics_allowed
           && ((is_gfx90a
                && ((is_float && n < ssymv_U_gfx908_gfx90a_higher_threshold)
                    || (is_double && n < dsymv_U_gfx90a_higher_threshold)))
               || (is_gfx908
                   && ((is_float && n < ssymv_U_gfx908_gfx90a_higher_threshold)
                       || (is_double
                           && (((n % 32 == 0) && n < dsymv_U_gfx908_higher_threshold)
                               || ((n % 32 != 0)
                                   && (n < dsymv_U_gfx908_generic_higher_threshold
                                       || n > dsymv_U_gfx908_generic_lower_threshold))))))))
        {
            bool host_ptr_mode = handle->pointer_mode == rocblas_pointer_mode_host;
            rocblas_internal_val_ptr<T> alpha_device_host(host_ptr_mode, alpha);
            rocblas_internal_val_ptr<T> beta_device_host(host_ptr_mode, beta);

            static constexpr rocblas_int DIM_X   = 32;
            static constexpr rocblas_int block_y = 8;
            const rocblas_int            mod     = n % DIM_X;

            if constexpr(is_float || is_double)
            {
                if(mod == 0)
                {
                    //The following symv_kernel_upper_double_buffered is only valid for the multiples of DIM_X
                    static constexpr rocblas_int DIM_Y               = 4;
                    static constexpr rocblas_int elements_per_thread = (DIM_X / (2 * DIM_Y));
                    const int                    block_x             = n / DIM_X;

                    dim3 threads(DIM_X, DIM_Y);
                    dim3 grid(block_x, 1, batches);
                    dim3 grid_(block_x, block_y, batches);

                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_symv_kernel_upper_double_buffered_diagonal<DIM_X, DIM_Y>),
                        grid,
                        threads,
                        0,
                        rocblas_stream,
                        host_ptr_mode,
                        n,
                        alpha_device_host,
                        stride_alpha,
                        A,
                        offseta,
                        lda,
                        strideA,
                        x,
                        shiftx,
                        incx,
                        stridex,
                        beta_device_host,
                        stride_beta,
                        y,
                        shifty,
                        incy,
                        stridey,
                        batch_count);

                    ROCBLAS_LAUNCH_KERNEL((rocblas_symv_kernel_upper_double_buffered_non_diagonal<
                                              DIM_X,
                                              DIM_Y,
                                              elements_per_thread>),
                                          grid_,
                                          threads,
                                          0,
                                          rocblas_stream,
                                          host_ptr_mode,
                                          n,
                                          alpha_device_host,
                                          stride_alpha,
                                          A,
                                          offseta,
                                          lda,
                                          strideA,
                                          x,
                                          shiftx,
                                          incx,
                                          stridex,
                                          y,
                                          shifty,
                                          incy,
                                          stridey,
                                          batch_count);
                }
                else
                {
                    static constexpr rocblas_int DIM_Y               = 8;
                    static constexpr rocblas_int elements_per_thread = (DIM_X / (2 * DIM_Y));
                    const rocblas_int            irregular_part      = mod % elements_per_thread;
                    const rocblas_int            block_x             = n / DIM_X + (mod != 0);

                    dim3 threads(DIM_X, DIM_Y);
                    dim3 grid(block_x, 1, batches);
                    dim3 grid_(block_x, block_y, batches);

                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_symv_kernel_upper_double_buffered_diagonal_generic<DIM_X, DIM_Y>),
                        grid,
                        threads,
                        0,
                        rocblas_stream,
                        host_ptr_mode,
                        n,
                        alpha_device_host,
                        stride_alpha,
                        A,
                        offseta,
                        lda,
                        strideA,
                        x,
                        shiftx,
                        incx,
                        stridex,
                        beta_device_host,
                        stride_beta,
                        y,
                        shifty,
                        incy,
                        stridey,
                        mod,
                        batch_count);

#define symvu_KARGS                                                                          \
    grid_, threads, 0, rocblas_stream, host_ptr_mode, n, alpha_device_host, stride_alpha, A, \
        offseta, lda, strideA, x, shiftx, incx, stridex, y, shifty, incy, stridey, mod,      \
        batch_count
                    if(irregular_part == 0)
                    {
                        ROCBLAS_LAUNCH_KERNEL(
                            (rocblas_symv_kernel_upper_double_buffered_non_diagonal_generic<
                                DIM_X,
                                DIM_Y,
                                elements_per_thread,
                                0>),
                            symvu_KARGS);
                    }
                    else if(irregular_part == 1)
                    {
                        ROCBLAS_LAUNCH_KERNEL(
                            (rocblas_symv_kernel_upper_double_buffered_non_diagonal_generic<
                                DIM_X,
                                DIM_Y,
                                elements_per_thread,
                                1>),
                            symvu_KARGS);
                    }
#undef symvu_KARGS
                }
            }
        }
        else
        {
            if(handle->pointer_mode == rocblas_pointer_mode_device)
            {
                if(i64_indices)
                {
                    ROCBLAS_LAUNCH_KERNEL((rocblas_hemvn_kernel_upper<IS_HEMV,
                                                                      HEMV_DIM_X,
                                                                      HEMV_DIM_Y,
                                                                      bank_shift,
                                                                      half_HEMV_DIM_X,
                                                                      quarter_HEMV_DIM_X,
                                                                      int64_t>),
                                          hemv_kernel_KARGS(alpha, beta));

                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_hemvn_kernel_upper_block_sum<HEMV_DIM_X, int64_t>),
                        hemv_kernel_sum_KARGS(alpha, beta));
                }
                else
                {
                    ROCBLAS_LAUNCH_KERNEL((rocblas_hemvn_kernel_upper<IS_HEMV,
                                                                      HEMV_DIM_X,
                                                                      HEMV_DIM_Y,
                                                                      bank_shift,
                                                                      half_HEMV_DIM_X,
                                                                      quarter_HEMV_DIM_X,
                                                                      rocblas_int>),
                                          hemv_kernel_KARGS(alpha, beta));

                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_hemvn_kernel_upper_block_sum<HEMV_DIM_X, rocblas_int>),
                        hemv_kernel_sum_KARGS(alpha, beta));
                }
            }
            else
            {
                if(!*alpha && *beta == 1)
                    return rocblas_status_success;
                if(i64_indices)
                {
                    ROCBLAS_LAUNCH_KERNEL((rocblas_hemvn_kernel_upper<IS_HEMV,
                                                                      HEMV_DIM_X,
                                                                      HEMV_DIM_Y,
                                                                      bank_shift,
                                                                      half_HEMV_DIM_X,
                                                                      quarter_HEMV_DIM_X,
                                                                      int64_t>),
                                          hemv_kernel_KARGS(*alpha, *beta));

                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_hemvn_kernel_upper_block_sum<HEMV_DIM_X, int64_t>),
                        hemv_kernel_sum_KARGS(*alpha, *beta));
                }
                else
                {
                    ROCBLAS_LAUNCH_KERNEL((rocblas_hemvn_kernel_upper<IS_HEMV,
                                                                      HEMV_DIM_X,
                                                                      HEMV_DIM_Y,
                                                                      bank_shift,
                                                                      half_HEMV_DIM_X,
                                                                      quarter_HEMV_DIM_X,
                                                                      rocblas_int>),
                                          hemv_kernel_KARGS(*alpha, *beta));

                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_hemvn_kernel_upper_block_sum<HEMV_DIM_X, rocblas_int>),
                        hemv_kernel_sum_KARGS(*alpha, *beta));
                }
            }
        }
    }
    else
    {
        if(is_atomics_allowed
           && ((is_gfx908 && (is_float || is_double))
               || (is_gfx90a
                   && (((is_float && n < ssymv_L_gfx90a_higher_threshold)
                        || ((n % 32 == 0 && is_double && n < dsymv_L_gfx90a_higher_threshold)
                            || (n % 32 != 0 && is_double
                                && n < dsymv_L_gfx90a_generic_higher_threshold)))))))
        {
            //The following symv_kernel_upper_double_buffered is only valid for the multiples of DIM_X
            static constexpr rocblas_int DIM_X               = 32;
            static constexpr rocblas_int DIM_Y               = 4;
            static constexpr rocblas_int elements_per_thread = (DIM_X / (2 * DIM_Y));
            static constexpr rocblas_int block_y             = 8;
            const rocblas_int            mod                 = n % DIM_X;
            const rocblas_int            block_x             = n / DIM_X + (mod != 0);

            dim3 threads(DIM_X, DIM_Y);
            dim3 grid(block_x, 1, batches);
            dim3 grid_(block_x, block_y, batches);

            bool host_ptr_mode = handle->pointer_mode == rocblas_pointer_mode_host;
            rocblas_internal_val_ptr<T> alpha_device_host(host_ptr_mode, alpha);
            rocblas_internal_val_ptr<T> beta_device_host(host_ptr_mode, beta);

            if constexpr(is_float || is_double)
            {
                if(mod == 0)
                {
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_symv_kernel_lower_double_buffered_diagonal<DIM_X, DIM_Y>),
                        grid,
                        threads,
                        0,
                        rocblas_stream,
                        host_ptr_mode,
                        n,
                        alpha_device_host,
                        stride_alpha,
                        A,
                        offseta,
                        lda,
                        strideA,
                        x,
                        shiftx,
                        incx,
                        stridex,
                        beta_device_host,
                        stride_beta,
                        y,
                        shifty,
                        incy,
                        stridey,
                        batch_count);

                    ROCBLAS_LAUNCH_KERNEL((rocblas_symv_kernel_lower_double_buffered_non_diagonal<
                                              DIM_X,
                                              DIM_Y,
                                              elements_per_thread>),
                                          grid_,
                                          threads,
                                          0,
                                          rocblas_stream,
                                          host_ptr_mode,
                                          n,
                                          alpha_device_host,
                                          stride_alpha,
                                          A,
                                          offseta,
                                          lda,
                                          strideA,
                                          x,
                                          shiftx,
                                          incx,
                                          stridex,
                                          y,
                                          shifty,
                                          incy,
                                          stridey,
                                          batch_count);
                }
                else
                {
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_symv_kernel_lower_double_buffered_diagonal_generic<DIM_X, DIM_Y>),
                        grid,
                        threads,
                        0,
                        rocblas_stream,
                        host_ptr_mode,
                        n,
                        alpha_device_host,
                        stride_alpha,
                        A,
                        offseta,
                        lda,
                        strideA,
                        x,
                        shiftx,
                        incx,
                        stridex,
                        beta_device_host,
                        stride_beta,
                        y,
                        shifty,
                        incy,
                        stridey,
                        mod,
                        batch_count);

                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_symv_kernel_lower_double_buffered_non_diagonal_generic<
                            DIM_X,
                            DIM_Y,
                            elements_per_thread>),
                        grid_,
                        threads,
                        0,
                        rocblas_stream,
                        host_ptr_mode,
                        n,
                        alpha_device_host,
                        stride_alpha,
                        A,
                        offseta,
                        lda,
                        strideA,
                        x,
                        shiftx,
                        incx,
                        stridex,
                        y,
                        shifty,
                        incy,
                        stridey,
                        mod,
                        batch_count);
                }
            }
        }
        else
        {
            if(handle->pointer_mode == rocblas_pointer_mode_device)
            {
                if(i64_indices)
                {
                    ROCBLAS_LAUNCH_KERNEL((rocblas_hemvn_kernel_lower<IS_HEMV,
                                                                      HEMV_DIM_X,
                                                                      HEMV_DIM_Y,
                                                                      bank_shift,
                                                                      half_HEMV_DIM_X,
                                                                      quarter_HEMV_DIM_X,
                                                                      int64_t>),
                                          hemv_kernel_KARGS(alpha, beta));

                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_hemvn_kernel_lower_block_sum<HEMV_DIM_X, int64_t>),
                        hemv_kernel_sum_KARGS(alpha, beta));
                }
                else
                {
                    ROCBLAS_LAUNCH_KERNEL((rocblas_hemvn_kernel_lower<IS_HEMV,
                                                                      HEMV_DIM_X,
                                                                      HEMV_DIM_Y,
                                                                      bank_shift,
                                                                      half_HEMV_DIM_X,
                                                                      quarter_HEMV_DIM_X,
                                                                      rocblas_int>),
                                          hemv_kernel_KARGS(alpha, beta));

                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_hemvn_kernel_lower_block_sum<HEMV_DIM_X, rocblas_int>),
                        hemv_kernel_sum_KARGS(alpha, beta));
                }
            }
            else
            {
                if(!*alpha && *beta == 1)
                    return rocblas_status_success;
                if(i64_indices)
                {
                    ROCBLAS_LAUNCH_KERNEL((rocblas_hemvn_kernel_lower<IS_HEMV,
                                                                      HEMV_DIM_X,
                                                                      HEMV_DIM_Y,
                                                                      bank_shift,
                                                                      half_HEMV_DIM_X,
                                                                      quarter_HEMV_DIM_X,
                                                                      int64_t>),
                                          hemv_kernel_KARGS(*alpha, *beta));

                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_hemvn_kernel_lower_block_sum<HEMV_DIM_X, int64_t>),
                        hemv_kernel_sum_KARGS(*alpha, *beta));
                }
                else
                {
                    ROCBLAS_LAUNCH_KERNEL((rocblas_hemvn_kernel_lower<IS_HEMV,
                                                                      HEMV_DIM_X,
                                                                      HEMV_DIM_Y,
                                                                      bank_shift,
                                                                      half_HEMV_DIM_X,
                                                                      quarter_HEMV_DIM_X,
                                                                      rocblas_int>),
                                          hemv_kernel_KARGS(*alpha, *beta));

                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_hemvn_kernel_lower_block_sum<HEMV_DIM_X, rocblas_int>),
                        hemv_kernel_sum_KARGS(*alpha, *beta));
                }
            }
        }
    }
    return rocblas_status_success;
}

/**
  *  Note stride_alpha and stride_beta are only used AND only tested by rocSOLVER
  *  These strided scalar fetches are only supported for device_ptr mode
  */
template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_symv_template(rocblas_handle handle,
                                   rocblas_fill   uplo,
                                   rocblas_int    n,
                                   const T*       alpha,
                                   rocblas_stride stride_alpha,
                                   const T*       A,
                                   rocblas_stride offseta,
                                   rocblas_int    lda,
                                   rocblas_stride strideA,
                                   const T*       x,
                                   rocblas_stride offsetx,
                                   rocblas_int    incx,
                                   rocblas_stride stridex,
                                   const T*       beta,
                                   rocblas_stride stride_beta,
                                   T*             y,
                                   rocblas_stride offsety,
                                   rocblas_int    incy,
                                   rocblas_stride stridey,
                                   rocblas_int    batch_count,
                                   T*             workspace)
{
    // flag to check whether the kernel function being called is for hemv or symv
    // For hemv, IS_HEMV = true and for SYMV, IS_HEMV = false
    static constexpr bool IS_HEMV = false;

    /* SYMV and HEMV are nearly identical BLAS functions with the following changes
        1. In HEMV, the imaginary part of the main diagonal in the matrix `A` of is assumed to be zero. But, for SYMV both real and imaginary part is considered
        2. If matrix 'A' is a Hermitian matrix then A = A^H, where A^H is the conjugate transpose of matrix 'A', therefore the `conj()` helper function is used
        3. If matrix 'A' is a Symmetric matrix then A = A^T, Where A^T is the transpose of matrix 'A', therefore the `conj()` helper function is not used*/

    rocblas_status status = rocblas_internal_symv_hemv_launcher<IS_HEMV>(handle,
                                                                         uplo,
                                                                         n,
                                                                         alpha,
                                                                         stride_alpha,
                                                                         A,
                                                                         offseta,
                                                                         lda,
                                                                         strideA,
                                                                         x,
                                                                         offsetx,
                                                                         incx,
                                                                         stridex,
                                                                         beta,
                                                                         stride_beta,
                                                                         y,
                                                                         offsety,
                                                                         incy,
                                                                         stridey,
                                                                         batch_count,
                                                                         workspace);
    return status;
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_symv_batched_template(rocblas_handle  handle,
                                           rocblas_fill    uplo,
                                           rocblas_int     n,
                                           const T*        alpha,
                                           rocblas_stride  stride_alpha,
                                           const T* const* A,
                                           rocblas_stride  offseta,
                                           rocblas_int     lda,
                                           rocblas_stride  strideA,
                                           const T* const* x,
                                           rocblas_stride  offsetx,
                                           rocblas_int     incx,
                                           rocblas_stride  stridex,
                                           const T*        beta,
                                           rocblas_stride  stride_beta,
                                           T* const*       y,
                                           rocblas_stride  offsety,
                                           rocblas_int     incy,
                                           rocblas_stride  stridey,
                                           rocblas_int     batch_count,
                                           T*              workspace)
{
    static constexpr bool IS_HEMV = false;
    return rocblas_internal_symv_hemv_launcher<IS_HEMV>(handle,
                                                        uplo,
                                                        n,
                                                        alpha,
                                                        stride_alpha,
                                                        A,
                                                        offseta,
                                                        lda,
                                                        strideA,
                                                        x,
                                                        offsetx,
                                                        incx,
                                                        stridex,
                                                        beta,
                                                        stride_beta,
                                                        y,
                                                        offsety,
                                                        incy,
                                                        stridey,
                                                        batch_count,
                                                        workspace);
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_hemv_template(rocblas_handle handle,
                                   rocblas_fill   uplo,
                                   rocblas_int    n,
                                   const T*       alpha,
                                   rocblas_stride stride_alpha,
                                   const T*       A,
                                   rocblas_stride offseta,
                                   rocblas_int    lda,
                                   rocblas_stride strideA,
                                   const T*       x,
                                   rocblas_stride offsetx,
                                   rocblas_int    incx,
                                   rocblas_stride stridex,
                                   const T*       beta,
                                   rocblas_stride stride_beta,
                                   T*             y,
                                   rocblas_stride offsety,
                                   rocblas_int    incy,
                                   rocblas_stride stridey,
                                   rocblas_int    batch_count,
                                   T*             workspace)
{
    static constexpr bool IS_HEMV = true;
    return rocblas_internal_symv_hemv_launcher<IS_HEMV>(handle,
                                                        uplo,
                                                        n,
                                                        alpha,
                                                        stride_alpha,
                                                        A,
                                                        offseta,
                                                        lda,
                                                        strideA,
                                                        x,
                                                        offsetx,
                                                        incx,
                                                        stridex,
                                                        beta,
                                                        stride_beta,
                                                        y,
                                                        offsety,
                                                        incy,
                                                        stridey,
                                                        batch_count,
                                                        workspace);
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_hemv_batched_template(rocblas_handle  handle,
                                           rocblas_fill    uplo,
                                           rocblas_int     n,
                                           const T*        alpha,
                                           rocblas_stride  stride_alpha,
                                           const T* const* A,
                                           rocblas_stride  offseta,
                                           rocblas_int     lda,
                                           rocblas_stride  strideA,
                                           const T* const* x,
                                           rocblas_stride  offsetx,
                                           rocblas_int     incx,
                                           rocblas_stride  stridex,
                                           const T*        beta,
                                           rocblas_stride  stride_beta,
                                           T* const*       y,
                                           rocblas_stride  offsety,
                                           rocblas_int     incy,
                                           rocblas_stride  stridey,
                                           rocblas_int     batch_count,
                                           T*              workspace)
{
    static constexpr bool IS_HEMV = true;
    return rocblas_internal_symv_hemv_launcher<IS_HEMV>(handle,
                                                        uplo,
                                                        n,
                                                        alpha,
                                                        stride_alpha,
                                                        A,
                                                        offseta,
                                                        lda,
                                                        strideA,
                                                        x,
                                                        offsetx,
                                                        incx,
                                                        stridex,
                                                        beta,
                                                        stride_beta,
                                                        y,
                                                        offsety,
                                                        incy,
                                                        stridey,
                                                        batch_count,
                                                        workspace);
}

template <typename T, typename U>
rocblas_status rocblas_hemv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_fill   uplo,
                                           int64_t        n,
                                           T              A,
                                           rocblas_stride offset_a,
                                           int64_t        lda,
                                           rocblas_stride stride_a,
                                           T              x,
                                           rocblas_stride offset_x,
                                           int64_t        inc_x,
                                           rocblas_stride stride_x,
                                           U              y,
                                           rocblas_stride offset_y,
                                           int64_t        inc_y,
                                           rocblas_stride stride_y,
                                           int64_t        batch_count,
                                           const int      check_numerics,
                                           bool           is_input)
{
    rocblas_status check_numerics_status = rocblas_status_success;

    if(is_input)
    {
        check_numerics_status
            = rocblas_internal_check_numerics_matrix_template(function_name,
                                                              handle,
                                                              rocblas_operation_none,
                                                              uplo,
                                                              rocblas_client_hermitian_matrix,
                                                              n,
                                                              n,
                                                              A,
                                                              offset_a,
                                                              lda,
                                                              stride_a,
                                                              batch_count,
                                                              check_numerics,
                                                              is_input);

        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;

        check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                                handle,
                                                                                n,
                                                                                x,
                                                                                offset_x,
                                                                                inc_x,
                                                                                stride_x,
                                                                                batch_count,
                                                                                check_numerics,
                                                                                is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;
    }

    check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                            handle,
                                                                            n,
                                                                            y,
                                                                            offset_y,
                                                                            inc_y,
                                                                            stride_y,
                                                                            batch_count,
                                                                            check_numerics,
                                                                            is_input);

    return check_numerics_status;
}

template <typename T, typename U>
rocblas_status rocblas_symv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_fill   uplo,
                                           int64_t        n,
                                           T              A,
                                           rocblas_stride offset_a,
                                           int64_t        lda,
                                           rocblas_stride stride_a,
                                           T              x,
                                           rocblas_stride offset_x,
                                           int64_t        inc_x,
                                           rocblas_stride stride_x,
                                           U              y,
                                           rocblas_stride offset_y,
                                           int64_t        inc_y,
                                           rocblas_stride stride_y,
                                           int64_t        batch_count,
                                           const int      check_numerics,
                                           bool           is_input)
{
    rocblas_status check_numerics_status
        = rocblas_internal_check_numerics_matrix_template(function_name,
                                                          handle,
                                                          rocblas_operation_none,
                                                          uplo,
                                                          rocblas_client_symmetric_matrix,
                                                          n,
                                                          n,
                                                          A,
                                                          offset_a,
                                                          lda,
                                                          stride_a,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);

    if(check_numerics_status != rocblas_status_success)
        return check_numerics_status;

    check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                            handle,
                                                                            n,
                                                                            x,
                                                                            offset_x,
                                                                            inc_x,
                                                                            stride_x,
                                                                            batch_count,
                                                                            check_numerics,
                                                                            is_input);
    if(check_numerics_status != rocblas_status_success)
        return check_numerics_status;

    check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                            handle,
                                                                            n,
                                                                            y,
                                                                            offset_y,
                                                                            inc_y,
                                                                            stride_y,
                                                                            batch_count,
                                                                            check_numerics,
                                                                            is_input);

    return check_numerics_status;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *hemv*.cpp and *symv*.cpp

// clang-format off

#define INSTANTIATE_HEMV_WORKSPACE(To_)                                                                \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE size_t rocblas_internal_hemv_symv_kernel_workspace_size<To_> \
                        (rocblas_int n, rocblas_int batch_count);

INSTANTIATE_HEMV_WORKSPACE(float)
INSTANTIATE_HEMV_WORKSPACE(double)
INSTANTIATE_HEMV_WORKSPACE(rocblas_float_complex)
INSTANTIATE_HEMV_WORKSPACE(rocblas_double_complex)

#undef INSTANTIATE_HEMV_WORKSPACE

#ifdef INSTANTIATE_HEMV_TEMPLATE
#error INSTANTIATE_HEMV_TEMPLATE already defined
#endif

#define INSTANTIATE_HEMV_TEMPLATE(T_)                                                       \
    template rocblas_status rocblas_internal_hemv_template<T_>(rocblas_handle handle,       \
                                                               rocblas_fill   uplo,         \
                                                               int            n,            \
                                                               const T_*      alpha,        \
                                                               rocblas_stride stride_alpha, \
                                                               const T_*      A,            \
                                                               rocblas_stride offsetA,      \
                                                               int            lda,          \
                                                               rocblas_stride strideA,      \
                                                               const T_*      x,            \
                                                               rocblas_stride offsetx,      \
                                                               int            incx,         \
                                                               rocblas_stride stridex,      \
                                                               const T_*      beta,         \
                                                               rocblas_stride stride_beta,  \
                                                               T_*            y,            \
                                                               rocblas_stride offsety,      \
                                                               int            incy,         \
                                                               rocblas_stride stridey,      \
                                                               int            batch_count,  \
                                                               T_*            workspace);

INSTANTIATE_HEMV_TEMPLATE(rocblas_float_complex)
INSTANTIATE_HEMV_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_HEMV_TEMPLATE

#ifdef INSTANTIATE_HEMV_BATCHED_TEMPLATE
#error INSTANTIATE_HEMV_BATCHED_TEMPLATE already defined
#endif

#define INSTANTIATE_HEMV_BATCHED_TEMPLATE(T_)                           \
    template rocblas_status rocblas_internal_hemv_batched_template<T_>( \
        rocblas_handle   handle,                                        \
        rocblas_fill     uplo,                                          \
        int              n,                                             \
        const T_*        alpha,                                         \
        rocblas_stride   stride_alpha,                                  \
        const T_* const* A,                                             \
        rocblas_stride   offsetA,                                       \
        int              lda,                                           \
        rocblas_stride   strideA,                                       \
        const T_* const* x,                                             \
        rocblas_stride   offsetx,                                       \
        int              incx,                                          \
        rocblas_stride   stridex,                                       \
        const T_*        beta,                                          \
        rocblas_stride   stride_beta,                                   \
        T_* const*       y,                                             \
        rocblas_stride   offsety,                                       \
        int              incy,                                          \
        rocblas_stride   stridey,                                       \
        int              batch_count,                                   \
        T_*              workspace);

INSTANTIATE_HEMV_BATCHED_TEMPLATE(rocblas_float_complex)
INSTANTIATE_HEMV_BATCHED_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_HEMV_BATCHED_TEMPLATE

#ifdef INSTANTIATE_SYMV_TEMPLATE
#error INSTANTIATE_SYMV_TEMPLATE already defined
#endif

#define INSTANTIATE_SYMV_TEMPLATE(T_)                                                       \
    template rocblas_status rocblas_internal_symv_template<T_>(rocblas_handle handle,       \
                                                               rocblas_fill   uplo,         \
                                                               int            n,            \
                                                               const T_*      alpha,        \
                                                               rocblas_stride stride_alpha, \
                                                               const T_*      A,            \
                                                               rocblas_stride offsetA,      \
                                                               int            lda,          \
                                                               rocblas_stride strideA,      \
                                                               const T_*      x,            \
                                                               rocblas_stride offsetx,      \
                                                               int            incx,         \
                                                               rocblas_stride stridex,      \
                                                               const T_*      beta,         \
                                                               rocblas_stride stride_beta,  \
                                                               T_*            y,            \
                                                               rocblas_stride offsety,      \
                                                               int            incy,         \
                                                               rocblas_stride stridey,      \
                                                               int            batch_count,  \
                                                               T_*            workspace);

INSTANTIATE_SYMV_TEMPLATE(float)
INSTANTIATE_SYMV_TEMPLATE(double)
INSTANTIATE_SYMV_TEMPLATE(rocblas_float_complex)
INSTANTIATE_SYMV_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_SYMV_TEMPLATE

#ifdef INSTANTIATE_SYMV_BATCHED_TEMPLATE
#error INSTANTIATE_SYMV_BATCHED_TEMPLATE already defined
#endif

#define INSTANTIATE_SYMV_BATCHED_TEMPLATE(T_)                           \
    template rocblas_status rocblas_internal_symv_batched_template<T_>( \
        rocblas_handle   handle,                                        \
        rocblas_fill     uplo,                                          \
        int              n,                                             \
        const T_*        alpha,                                         \
        rocblas_stride   stride_alpha,                                  \
        const T_* const* A,                                             \
        rocblas_stride   offsetA,                                       \
        int              lda,                                           \
        rocblas_stride   strideA,                                       \
        const T_* const* x,                                             \
        rocblas_stride   offsetx,                                       \
        int              incx,                                          \
        rocblas_stride   stridex,                                       \
        const T_*        beta,                                          \
        rocblas_stride   stride_beta,                                   \
        T_* const*       y,                                             \
        rocblas_stride   offsety,                                       \
        int              incy,                                          \
        rocblas_stride   stridey,                                       \
        int              batch_count,                                   \
        T_*              workspace);

INSTANTIATE_SYMV_BATCHED_TEMPLATE(float)
INSTANTIATE_SYMV_BATCHED_TEMPLATE(double)
INSTANTIATE_SYMV_BATCHED_TEMPLATE(rocblas_float_complex)
INSTANTIATE_SYMV_BATCHED_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_SYMV_BATCHED_TEMPLATE_64

#undef INSTANTIATE_SYMV_LAUNCHER

#ifdef INSTANTIATE_HEMV_NUMERICS
#error INSTANTIATE_HEMV_NUMERICS already defined
#endif

#define INSTANTIATE_HEMV_NUMERICS(T_, U_)                                 \
template rocblas_status rocblas_hemv_check_numerics<T_, U_>               \
                                          (const char*    function_name,  \
                                           rocblas_handle handle,         \
                                           rocblas_fill   uplo,           \
                                           int64_t        n,              \
                                           T_             A,              \
                                           rocblas_stride    offset_a,    \
                                           int64_t        lda,            \
                                           rocblas_stride stride_a,       \
                                           T_             x,              \
                                           rocblas_stride    offset_x,    \
                                           int64_t        inc_x,          \
                                           rocblas_stride stride_x,       \
                                           U_             y,              \
                                           rocblas_stride    offset_y,    \
                                           int64_t        inc_y,          \
                                           rocblas_stride stride_y,       \
                                           int64_t        batch_count,    \
                                           const int      check_numerics, \
                                           bool           is_input);

INSTANTIATE_HEMV_NUMERICS(rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_HEMV_NUMERICS(rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_HEMV_NUMERICS(rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_HEMV_NUMERICS(rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_HEMV_NUMERICS

#ifdef INSTANTIATE_SYMV_NUMERICS
#error INSTANTIATE_SYMV_NUMERICS already defined
#endif

#define INSTANTIATE_SYMV_NUMERICS(T_, U_)                                 \
template rocblas_status rocblas_symv_check_numerics<T_, U_>               \
                                          (const char*    function_name,  \
                                           rocblas_handle handle,         \
                                           rocblas_fill   uplo,           \
                                           int64_t        n,              \
                                           T_             A,              \
                                           rocblas_stride offset_a,       \
                                           int64_t        lda,            \
                                           rocblas_stride stride_a,       \
                                           T_             x,              \
                                           rocblas_stride offset_x,       \
                                           int64_t        inc_x,          \
                                           rocblas_stride stride_x,       \
                                           U_             y,              \
                                           rocblas_stride offset_y,       \
                                           int64_t        inc_y,          \
                                           rocblas_stride stride_y,       \
                                           int64_t        batch_count,    \
                                           const int      check_numerics, \
                                           bool           is_input);

INSTANTIATE_SYMV_NUMERICS(float const*, float*)
INSTANTIATE_SYMV_NUMERICS(double const*, double*)
INSTANTIATE_SYMV_NUMERICS(rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_SYMV_NUMERICS(rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_SYMV_NUMERICS(float const* const*, float* const*)
INSTANTIATE_SYMV_NUMERICS(double const* const*, double* const*)
INSTANTIATE_SYMV_NUMERICS(rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_SYMV_NUMERICS(rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_SYMV_NUMERICS

// clang-format on
