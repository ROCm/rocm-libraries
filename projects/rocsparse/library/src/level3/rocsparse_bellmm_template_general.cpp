/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "bellmm_device_general.h"
#include "rocsparse_float16.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    //
    // Compile-time launch geometry for the general blocked-ELL SpMM kernel.
    //
    // The kernel maps one thread to one output element C[C_row, n]:
    // hipThreadIdx_x selects the row inside the A block (stepping by
    // BELL_BLOCK_DIM to cover block dimensions larger than the thread block)
    // and hipThreadIdx_y selects the dense column (tiled by BLK_SIZE_Y). There
    // is no cross-thread reduction, so the kernel is correct for any square
    // TILE x TILE workgroup: rows beyond block_dim and columns beyond N are
    // simply skipped by the in-kernel guards.
    //
    // Historically TILE was the fixed magic number 32 (=> 32x32 = 1024 threads).
    // That is a poor fit for small ELL block dimensions: a block_dim of 8 still
    // launches a 1024-thread (32-wave on wave32) workgroup in which every thread
    // with hipThreadIdx_x >= 8 (i.e. 3/4 of the x-lanes) early-outs, wasting wave
    // slots and capping occupancy of this memory-bound kernel.
    //
    // We derive TILE from hardware properties instead of hard-coding it:
    //   * t_max   = largest power-of-two whose square still fits in
    //               maxThreadsPerBlock (== 32 on every arch rocSPARSE targets,
    //               since 32*32 = 1024). This reproduces the historical tile
    //               exactly and is what wave64 (CDNA) keeps unconditionally.
    //   * min_tile= smallest power-of-two whose square covers at least one full
    //               wavefront (== 8 on wave32: 8*8 = 64 = 2 wavefronts), so the
    //               shrunk workgroup is always a whole number of wavefronts.
    // On wave32 we then pick the smallest power-of-two tile that still covers
    // block_dim, clamped to [min_tile, t_max]. wave64 and anything we cannot
    // validate here fall back to t_max, i.e. the historical 32-wide tile, so
    // CDNA behaviour is unchanged by construction.
    //
    static rocsparse_int bellmm_general_tile_size(rocsparse_handle handle, int64_t block_dim)
    {
        rocsparse_int t_max = 1;
        while((t_max * 2) * (t_max * 2) <= handle->properties.maxThreadsPerBlock)
        {
            t_max *= 2;
        }

        // Gate the shrink on the only wavefront width we can validate here
        // (wave32 / RDNA). On wave64 keep the historical full tile.
        if(handle->wavefront_size != 32)
        {
            return t_max;
        }

        rocsparse_int min_tile = 1;
        while(min_tile * min_tile < handle->wavefront_size)
        {
            min_tile *= 2;
        }

        rocsparse_int t = min_tile;
        while(t < block_dim && t < t_max)
        {
            t *= 2;
        }
        return t;
    }

    template <rocsparse_int BELL_BLOCK_DIM,
              rocsparse_int BLK_SIZE_Y,
              typename T,
              typename I,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_KERNEL(BELL_BLOCK_DIM* BLK_SIZE_Y)
    void bellmm_general_blockdim_kernel(rocsparse_operation trans_A,
                                        rocsparse_operation trans_B,
                                        I                   Mb,
                                        I                   N,
                                        ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                        I bell_cols,
                                        I bell_block_dim,
                                        const I* __restrict__ bell_col_ind,
                                        const A* __restrict__ bell_val,
                                        const B* __restrict__ dense_B,
                                        int64_t         ldb,
                                        rocsparse_order order_B,
                                        ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                        C* __restrict__ dense_C,
                                        int64_t              ldc,
                                        rocsparse_order      order_C,
                                        rocsparse_index_base idx_base,
                                        bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);

        if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
        {
            return;
        }

        rocsparse::bellmm_general_blockdim_device<BELL_BLOCK_DIM, BLK_SIZE_Y, T>(trans_A,
                                                                                 trans_B,
                                                                                 Mb,
                                                                                 N,
                                                                                 alpha,
                                                                                 bell_cols,
                                                                                 bell_block_dim,
                                                                                 bell_col_ind,
                                                                                 bell_val,
                                                                                 dense_B,
                                                                                 ldb,
                                                                                 order_B,
                                                                                 beta,
                                                                                 dense_C,
                                                                                 ldc,
                                                                                 order_C,
                                                                                 idx_base);
    }

    template <typename T, typename I, typename A, typename B, typename C>
    rocsparse_status bellmm_template_general(rocsparse_handle          handle,
                                             rocsparse_operation       trans_A,
                                             rocsparse_operation       trans_B,
                                             I                         mb,
                                             I                         n,
                                             I                         kb,
                                             I                         bell_cols,
                                             I                         bell_block_dim,
                                             const T*                  alpha,
                                             const rocsparse_mat_descr descr,
                                             const I*                  bell_col_ind,
                                             const A*                  bell_val,
                                             const B*                  dense_B,
                                             int64_t                   ldb,
                                             rocsparse_order           order_B,
                                             const T*                  beta,
                                             C*                        dense_C,
                                             int64_t                   ldc,
                                             rocsparse_order           order_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        hipStream_t stream = handle->stream;

        if(trans_A != rocsparse_operation_none)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_not_implemented,
                "This function is designed for trans_A = rocsparse_operation_none.");
        }

        // Device-derived, wave-relative square tile edge (see
        // rocsparse::bellmm_general_tile_size). On wave64 this is the historical
        // 32; on wave32 it shrinks to fit small block dimensions.
        const rocsparse_int tile = rocsparse::bellmm_general_tile_size(handle, bell_block_dim);

#define ROCSPARSE_LAUNCH_BELLMM_GENERAL(TILE)                               \
    {                                                                       \
        dim3 bellmm_blocks((mb - 1) / 1 + 1, (n - 1) / (TILE) + 1);         \
        dim3 bellmm_threads((TILE), (TILE), 1);                             \
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                 \
            (rocsparse::bellmm_general_blockdim_kernel<(TILE), (TILE), T>), \
            bellmm_blocks,                                                  \
            bellmm_threads,                                                 \
            0,                                                              \
            stream,                                                         \
            trans_A,                                                        \
            trans_B,                                                        \
            mb,                                                             \
            n,                                                              \
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha),               \
            bell_cols,                                                      \
            bell_block_dim,                                                 \
            bell_col_ind,                                                   \
            bell_val,                                                       \
            dense_B,                                                        \
            ldb,                                                            \
            order_B,                                                        \
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta),                \
            dense_C,                                                        \
            ldc,                                                            \
            order_C,                                                        \
            descr->base,                                                    \
            handle->pointer_mode == rocsparse_pointer_mode_host);           \
    }

        switch(tile)
        {
        case 8:
            ROCSPARSE_LAUNCH_BELLMM_GENERAL(8);
            break;
        case 16:
            ROCSPARSE_LAUNCH_BELLMM_GENERAL(16);
            break;
        default:
            ROCSPARSE_LAUNCH_BELLMM_GENERAL(32);
            break;
        }

#undef ROCSPARSE_LAUNCH_BELLMM_GENERAL

        return rocsparse_status_success;
    }
}

#define INSTANTIATE(TTYPE, ITYPE, ATYPE, BTYPE, CTYPE)                                               \
    template rocsparse_status rocsparse::bellmm_template_general(rocsparse_handle    handle,         \
                                                                 rocsparse_operation trans_A,        \
                                                                 rocsparse_operation trans_B,        \
                                                                 ITYPE               mb,             \
                                                                 ITYPE               n,              \
                                                                 ITYPE               kb,             \
                                                                 ITYPE               bell_cols,      \
                                                                 ITYPE               bell_block_dim, \
                                                                 const TTYPE*        alpha,          \
                                                                 const rocsparse_mat_descr descr,    \
                                                                 const ITYPE*    bell_col_ind,       \
                                                                 const ATYPE*    bell_val,           \
                                                                 const BTYPE*    dense_B,            \
                                                                 int64_t         ldb,                \
                                                                 rocsparse_order order_B,            \
                                                                 const TTYPE*    beta,               \
                                                                 CTYPE*          dense_C,            \
                                                                 int64_t         ldc,                \
                                                                 rocsparse_order order_C)

// Uniform precisions
INSTANTIATE(float, int32_t, float, float, float);
INSTANTIATE(float, int64_t, float, float, float);
INSTANTIATE(double, int32_t, double, double, double);
INSTANTIATE(double, int64_t, double, double, double);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);

// Mixed precisions
INSTANTIATE(int32_t, int32_t, int32_t, int32_t, int32_t);
INSTANTIATE(int32_t, int64_t, int32_t, int32_t, int32_t);
INSTANTIATE(float, int32_t, _Float16, _Float16, float);
INSTANTIATE(float, int64_t, _Float16, _Float16, float);
INSTANTIATE(float, int32_t, _Float16, _Float16, _Float16);
INSTANTIATE(float, int64_t, _Float16, _Float16, _Float16);
INSTANTIATE(float, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16);
INSTANTIATE(float, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16);
INSTANTIATE(int32_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE(int32_t, int64_t, int8_t, int8_t, int32_t);
INSTANTIATE(float, int32_t, int8_t, int8_t, float);
INSTANTIATE(float, int64_t, int8_t, int8_t, float);

#undef INSTANTIATE
