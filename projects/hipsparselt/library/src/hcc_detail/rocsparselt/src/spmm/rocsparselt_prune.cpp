/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "definitions.h"
#include "handle.h"
#include "rocsparselt.h"
#include "rocsparselt_spmm_utils.hpp"
#include "status.h"
#include "utility.hpp"

#include "hipsparselt_ostream.hpp"
#include <hip/hip_fp8.h>
#include <hip/hip_runtime_api.h>

template <typename Ti, int SG0I, int SG1J, int TT0I, int TT1J>
__global__ void prune_check_kernel(const Ti* in,
                                   int*      out,
                                   int64_t   m,
                                   int64_t   n,
                                   int64_t   stride1,
                                   int64_t   stride2,
                                   int       num_batches,
                                   int64_t   batch_stride,
                                   int64_t   sizes)
{
    constexpr unsigned int MT0I = SG0I * TT0I;
    constexpr unsigned int MT1J = SG1J * TT1J;
    using c_type = std::conditional_t<std::is_same<__hip_fp8_e4m3, Ti>::value || std::is_same<__hip_fp8_e5m2, Ti>::value, float, Ti>;
    const c_type ZERO_C = static_cast<c_type>(0.0f);

    unsigned int serial = hc_get_workitem_id(0);
    unsigned int sg0I   = serial % SG0I;
    unsigned int sg1J   = serial / SG0I;

    int64_t stride = sg0I * stride1 + sg1J * TT1J * stride2;

    unsigned int wg0I    = hc_get_group_id(0);
    unsigned int wg1J    = hc_get_group_id(1);
    unsigned int batchId = hc_get_group_id(2);

    if((MT1J * wg1J + sg1J * TT1J) >= n || (MT0I * wg0I + sg0I * TT0I) >= m)
        return;

    int64_t wg_stride = MT1J * wg1J * stride2 + MT0I * wg0I * stride1;
    int64_t b_stride  = batchId * batch_stride;

    int64_t globalReadOffset = b_stride + wg_stride + stride;

    for(int i = 0; i < TT0I; i++)
    {
        for(int j = 0; j < TT1J; j += 4)
        {
            if(*out)
                return;

            int64_t offset = i * stride1 + j * stride2;
            int     nz     = 0;

#pragma unroll
            for(int k = 0; k < 4; k++)
            {
                int64_t pos = globalReadOffset + offset + k * stride2;
                if(pos < sizes)
                {
                    if(static_cast<c_type>(in[pos]) != ZERO_C)
                    {
                        nz++;
                    }
                }
            }
            if(nz > 2)
            {
                *out = 1;
                return;
            }
        }
    }
}

template <typename Ti, typename Tc>
__host__ __device__ inline Tc norm1(Ti a, Ti b)
{
    Tc ac = static_cast<Tc>(a);
    Tc bc = static_cast<Tc>(b);
    return static_cast<Tc>(abs(ac) + abs(bc));
}

template <typename T>
__host__ __device__ inline T sum(T a, T b, T c, T d, T e, T f, T g, T h)
{
    return a + b + c + d + e + f + g + h;
}

template <typename T>
__host__ __device__ inline T acc_sum8(T* v, uint8_t* p, int v_offset, int p_offset)
{
    return sum<T>(v[v_offset + p[p_offset]],
                  v[v_offset + p[p_offset + 1]],
                  v[v_offset + 1 * 4 + p[p_offset + 2]],
                  v[v_offset + 1 * 4 + p[p_offset + 3]],
                  v[v_offset + 2 * 4 + p[p_offset + 4]],
                  v[v_offset + 2 * 4 + p[p_offset + 5]],
                  v[v_offset + 3 * 4 + p[p_offset + 6]],
                  v[v_offset + 3 * 4 + p[p_offset + 7]]);
}

template <typename T, bool InPlace>
__host__ __device__ inline void prune_if(bool prune, T* a, T b)
{
    if constexpr(InPlace)
    {
        if(prune) 
            *a = static_cast<T>(0.0f);
    }
    else
    {
        *a = prune ? static_cast<T>(0.0f) : b;
    }
}

template <typename Ti, typename Tc, int SG0I, int SG1J, int TT0I, int TT1J, bool InPlace>
__global__ void prune_strip_kernel(const Ti* in,
                                   Ti*       out,
                                   int64_t   m,
                                   int64_t   n,
                                   int64_t   stride1,
                                   int64_t   stride2,
                                   int       num_batches,
                                   int64_t   batch_stride,
                                   int64_t   sizes)
{
    constexpr unsigned int MT0I = SG0I * TT0I;
    constexpr unsigned int MT1J = SG1J * TT1J;

    unsigned int serial = hc_get_workitem_id(0);
    unsigned int sg0I   = serial % SG0I;
    unsigned int sg1J   = serial / SG0I;
    int64_t      stride = sg0I * TT0I * stride1 + sg1J * TT1J * stride2;

    unsigned int wg0I    = hc_get_group_id(0);
    unsigned int wg1J    = hc_get_group_id(1);
    unsigned int batchId = hc_get_group_id(2);

    if((MT1J * wg1J + sg1J * TT1J) >= n || (MT0I * wg0I + sg0I * TT0I) >= m)
        return;

    int64_t wg_stride = MT1J * wg1J * stride2 + MT0I * wg0I * stride1;
    int64_t b_stride  = batchId * batch_stride;

    int64_t globalReadOffset = b_stride + wg_stride + stride;

    for(int i = 0; i < TT0I; i++)
    {
        for(int j = 0; j < TT1J; j += 4)
        {
            int64_t offset = globalReadOffset + i * stride1 + j * stride2;
            Ti      values[4];
#pragma unroll 4
            for(int k = 0; k < 4; k++)
            {
                int64_t pos    = offset + k * stride2;
                bool    update = pos >= sizes;
                values[k]      = update ? static_cast<Ti>(0.0f) : in[pos];
            }

            auto max_norm1 = static_cast<Tc>(-1.0);
            int  pos_a = 0, pos_b = 0;

#pragma unroll 4
            for(int a = 0; a < 4; a++)
            {
                for(int b = a + 1; b < 4; b++)
                {
                    auto norm1_v = norm1<Ti, Tc>(values[a], values[b]);
                    bool update  = norm1_v > max_norm1;
                    pos_a        = update ? a : pos_a;
                    pos_b        = update ? b : pos_b;
                    max_norm1    = update ? norm1_v : max_norm1;
                }
            }

#pragma unroll 4
            for(int k = 0; k < 4; k++)
            {
                int64_t pos = offset + k * stride2;
                prune_if<Ti, InPlace>(k != pos_a && k != pos_b, &out[pos], values[k]);
            }
        }
    }
}

template <typename Ti,
          typename Tc,
          int  SG0I,
          int  SG1J,
          int  TT0I,
          int  TT1J,
          int  PATTERNS_COUNT,
          int  THREADS_PER_SG,
          int  PATTERNS_PER_THREAD,
          bool InPlace>
__global__
    __launch_bounds__(SG0I* SG1J* THREADS_PER_SG) void prune_tile_kernel(const Ti* in,
                                                                         Ti*       out,
                                                                         int64_t   m,
                                                                         int64_t   n,
                                                                         int64_t   stride1,
                                                                         int64_t   stride2,
                                                                         int       num_batches,
                                                                         int64_t   batch_stride,
                                                                         int64_t   sizes)
{
    constexpr int  PAD = 0;
    __shared__ Tc  value_abs[(16 + PAD) * SG0I * SG1J];
    __shared__ Tc  norm_res[THREADS_PER_SG * SG0I * SG1J];
    __shared__ int norm_idx[THREADS_PER_SG * SG0I * SG1J];

    // 90 patterns, that pick 2 elements from each row and column from a 4x4 tile, total pick 8 elements.
    // the first pattern: 0, 2, 0, 2, 1, 3, 1, 3 => COL#(ROW#,ROW#) = 0(0,2), 1(0,2), 2(1,3), 3(1,3)
    __constant__ static uint8_t pos_patterns[90 * 4 * 2] = {
        0, 2, 0, 2, 1, 3, 1, 3, 0, 2, 0, 3, 1, 3, 1, 2, 0, 2, 0, 3, 1, 2, 1, 3, 0, 2, 0, 1, 1, 3,
        2, 3, 0, 2, 0, 1, 2, 3, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 3, 1, 2, 0, 2, 1, 3,
        0, 1, 2, 3, 0, 2, 1, 3, 1, 3, 0, 2, 0, 2, 1, 3, 1, 2, 0, 3, 0, 2, 1, 3, 2, 3, 0, 1, 0, 2,
        1, 2, 0, 3, 1, 3, 0, 2, 1, 2, 1, 3, 0, 3, 0, 2, 2, 3, 0, 1, 1, 3, 0, 2, 2, 3, 1, 3, 0, 1,
        0, 3, 0, 2, 1, 3, 1, 2, 0, 3, 0, 2, 1, 2, 1, 3, 0, 3, 0, 3, 1, 2, 1, 2, 0, 3, 0, 1, 1, 2,
        2, 3, 0, 3, 0, 1, 2, 3, 1, 2, 0, 3, 1, 3, 0, 2, 1, 2, 0, 3, 1, 3, 1, 2, 0, 2, 0, 3, 1, 2,
        0, 2, 1, 3, 0, 3, 1, 2, 0, 3, 1, 2, 0, 3, 1, 2, 0, 1, 2, 3, 0, 3, 1, 2, 1, 3, 0, 2, 0, 3,
        1, 2, 1, 2, 0, 3, 0, 3, 1, 2, 2, 3, 0, 1, 0, 3, 2, 3, 0, 1, 1, 2, 0, 3, 2, 3, 1, 2, 0, 1,
        0, 1, 0, 2, 1, 3, 2, 3, 0, 1, 0, 2, 2, 3, 1, 3, 0, 1, 0, 3, 1, 2, 2, 3, 0, 1, 0, 3, 2, 3,
        1, 2, 0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 1, 3, 0, 2, 2, 3, 0, 1, 1, 3, 2, 3, 0, 2, 0, 1, 1, 2,
        0, 3, 2, 3, 0, 1, 1, 2, 2, 3, 0, 3, 0, 1, 2, 3, 0, 2, 1, 3, 0, 1, 2, 3, 0, 3, 1, 2, 0, 1,
        2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 3, 0, 2, 0, 1, 2, 3, 1, 2, 0, 3, 0, 1, 2, 3, 2, 3, 0, 1,
        1, 3, 0, 2, 0, 2, 1, 3, 1, 3, 0, 2, 0, 3, 1, 2, 1, 3, 0, 2, 0, 1, 2, 3, 1, 3, 0, 2, 1, 3,
        0, 2, 1, 3, 0, 2, 1, 2, 0, 3, 1, 3, 0, 2, 2, 3, 0, 1, 1, 3, 0, 3, 0, 2, 1, 2, 1, 3, 0, 3,
        1, 2, 0, 2, 1, 3, 0, 1, 0, 2, 2, 3, 1, 3, 0, 1, 2, 3, 0, 2, 1, 3, 1, 3, 0, 2, 0, 2, 1, 3,
        1, 2, 0, 2, 0, 3, 1, 3, 1, 2, 0, 3, 0, 2, 1, 3, 2, 3, 0, 2, 0, 1, 1, 3, 2, 3, 0, 1, 0, 2,
        1, 2, 0, 2, 0, 3, 1, 3, 1, 2, 0, 2, 1, 3, 0, 3, 1, 2, 0, 3, 0, 2, 1, 3, 1, 2, 0, 3, 0, 3,
        1, 2, 1, 2, 0, 3, 0, 1, 2, 3, 1, 2, 0, 3, 1, 3, 0, 2, 1, 2, 0, 3, 1, 2, 0, 3, 1, 2, 0, 3,
        2, 3, 0, 1, 1, 2, 0, 1, 0, 3, 2, 3, 1, 2, 0, 1, 2, 3, 0, 3, 1, 2, 1, 3, 0, 2, 0, 3, 1, 2,
        1, 3, 0, 3, 0, 2, 1, 2, 1, 2, 0, 3, 0, 3, 1, 2, 2, 3, 0, 3, 0, 1, 1, 2, 2, 3, 0, 1, 0, 3,
        2, 3, 0, 2, 0, 1, 1, 3, 2, 3, 0, 2, 1, 3, 0, 1, 2, 3, 0, 3, 0, 1, 1, 2, 2, 3, 0, 3, 1, 2,
        0, 1, 2, 3, 0, 1, 0, 2, 1, 3, 2, 3, 0, 1, 0, 3, 1, 2, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3, 0, 1,
        1, 3, 0, 2, 2, 3, 0, 1, 1, 2, 0, 3, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 3, 0, 2, 0, 1, 2, 3,
        1, 3, 0, 1, 0, 2, 2, 3, 1, 2, 0, 3, 0, 1, 2, 3, 1, 2, 0, 1, 0, 3, 2, 3, 2, 3, 0, 1, 0, 1,
    };

    constexpr unsigned int MT0I = SG0I * TT0I;
    constexpr unsigned int MT1J = SG1J * TT1J;

    const unsigned int serial
        = hc_get_workitem_id(0); //total has SG0I * SG1J * (THREADS_PER_SG) threads.
    const unsigned int serial_g  = serial / THREADS_PER_SG; //work idx of SG
    const unsigned int serial_gt = serial % THREADS_PER_SG; //thread idx in SG
    const unsigned int sg0I      = serial_g % SG0I;
    const unsigned int sg1J      = serial_g / SG0I;
    const int64_t      stride    = sg0I * TT0I * stride1 + sg1J * TT1J * stride2;

    const unsigned int wg0I    = hc_get_group_id(0);
    const unsigned int wg1J    = hc_get_group_id(1);
    const unsigned int batchId = hc_get_group_id(2);

    const int64_t wg_pos_x = MT0I * wg0I + sg0I * TT0I;
    const int64_t wg_pos_y = MT1J * wg1J + sg1J * TT1J;
    if((wg_pos_y) >= n || (wg_pos_x) >= m)
        return;

    const int64_t wg_stride      = MT1J * wg1J * stride2 + MT0I * wg0I * stride1;
    const int64_t b_stride       = batchId * batch_stride;
    int           serial_gt_     = serial_gt > 15 ? 15 : serial_gt;
    const int     x              = (serial_gt_) % 4;
    const int     y              = (serial_gt_) / 4;
    const int     value_offset   = serial_g * (16 + PAD);
    const int     c_value_offset = value_offset + (serial_gt_);

    const int norm_res_offset   = serial_g * THREADS_PER_SG;
    const int c_norm_res_offset = norm_res_offset + serial_gt;
    int64_t   globalReadOffset  = b_stride + wg_stride + stride + x * stride1 + y * stride2;

    int64_t pos;
    for(int j = 0; j < TT1J; j += 4)
    {
        pos = globalReadOffset;
        for(int i = 0; i < TT0I; i += 4)
        {
            Ti c_value = static_cast<Ti>(0.0f);
            // read 4x4 from the in matrix.
            {
                if((wg_pos_x + i + x) < m && (wg_pos_y + j + y) < n)
                {
                    c_value = in[pos];
                }
                value_abs[c_value_offset] = abs(static_cast<Tc>(c_value));
            }

            {
                int  offset             = min(serial_gt, PATTERNS_COUNT - 1);
                auto pos_pattern_offset = offset << 3;
                Tc   max_norm           = static_cast<Tc>(-1.f);
                int  max_norm_idx_      = 0;

                __syncthreads(); //wait until value_abs[] ready

                // caculate norm1 result for each pattern
#pragma unroll PATTERNS_PER_THREAD
                for(int k = 0; k < PATTERNS_PER_THREAD; k++)
                {
                    Tc tmp_norm = acc_sum8(
                        &value_abs[0], &pos_patterns[0], value_offset, pos_pattern_offset);
                    bool update   = max_norm < tmp_norm;
                    max_norm      = update ? tmp_norm : max_norm;
                    max_norm_idx_ = update ? pos_pattern_offset : max_norm_idx_;
                    offset += THREADS_PER_SG;
                    offset             = min(offset, PATTERNS_COUNT - 1);
                    pos_pattern_offset = offset << 3;
                }

                norm_res[c_norm_res_offset] = max_norm;
                norm_idx[c_norm_res_offset] = max_norm_idx_;
                __syncthreads(); //wait until norm_res[], norm_idx[] ready
            }

// find the pattern who has the largest norm1 value
#pragma unroll 5 //log2(THREADS_PER_SG)
            for(int tidxs = THREADS_PER_SG >> 1; tidxs > 0; tidxs >>= 1)
            {
                Tc   a                      = norm_res[c_norm_res_offset];
                Tc   b                      = norm_res[c_norm_res_offset + tidxs];
                int  b_idx                  = norm_idx[c_norm_res_offset + tidxs];
                bool update                 = serial_gt < tidxs && a < b;
                norm_res[c_norm_res_offset] = update ? b : norm_res[c_norm_res_offset];
                norm_idx[c_norm_res_offset] = update ? b_idx : norm_idx[c_norm_res_offset];
                __syncthreads(); //wait until norm_res[], norm_idx[] ready
            };

            // write 4x4 to the out matrix.
            {
                if((wg_pos_x + i + x) < m && (wg_pos_y + j + y) < n)
                {
                    prune_if<Ti, InPlace>(pos_patterns[norm_idx[norm_res_offset] + y * 2] != x
                                              && pos_patterns[norm_idx[norm_res_offset] + y * 2 + 1]
                                                     != x,
                                          &out[pos],
                                          c_value);
                }
            }
            pos += (stride1 << 2);
            __syncthreads(); // wait until norm_res[], norm_idx[] use completed.
        }
        globalReadOffset += (stride2 << 2);
    }
}

void get_prune_matrix_size(bool                    is_sparse_a,
                           rocsparselt_operation   op,
                           _rocsparselt_mat_descr* _sparseMatDescr,
                           int64_t&                m,
                           int64_t&                n,
                           int64_t&                stride0,
                           int64_t&                stride1)
{
    if(is_sparse_a)
    {
        m       = op == rocsparselt_operation_transpose ? _sparseMatDescr->n : _sparseMatDescr->m;
        n       = op == rocsparselt_operation_transpose ? _sparseMatDescr->m : _sparseMatDescr->n;
        stride0 = (op == rocsparselt_operation_transpose) ? _sparseMatDescr->ld : 1;
        stride1 = (op == rocsparselt_operation_transpose) ? 1 : _sparseMatDescr->ld;
    }
    else
    {
        m       = op == rocsparselt_operation_transpose ? _sparseMatDescr->m : _sparseMatDescr->n;
        n       = op == rocsparselt_operation_transpose ? _sparseMatDescr->n : _sparseMatDescr->m;
        stride0 = (op == rocsparselt_operation_transpose) ? 1 : _sparseMatDescr->ld;
        stride1 = (op == rocsparselt_operation_transpose) ? _sparseMatDescr->ld : 1;
    }
    if(_sparseMatDescr->order == rocsparselt_order_row)
    {
        std::swap(stride0, stride1);
    }
}
template <typename Ti, typename Tc>
rocsparselt_status rocsparselt_smfmac_prune_template(const _rocsparselt_handle* handle,
                                                     int64_t                    m,
                                                     int64_t                    n,
                                                     int64_t                    stride0,
                                                     int64_t                    stride1,
                                                     int                        num_batches,
                                                     int64_t                    batch_stride,
                                                     rocsparselt_order          order,
                                                     const Ti*                  d_in,
                                                     Ti*                        d_out,
                                                     rocsparselt_prune_alg      pruneAlg,
                                                     hipStream_t                stream)
{
    if(pruneAlg == rocsparselt_prune_smfmac_strip)
    {
        constexpr int SG0I    = 16;
        constexpr int SG1J    = 4;
        constexpr int TT0I    = 1;
        constexpr int TT1J    = 4;
        constexpr int MT0I    = SG0I * TT0I;
        constexpr int MT1J    = SG1J * TT1J;
        int           block_x = m / MT0I + (m % MT0I > 0 ? 1 : 0);
        int           block_y = n / MT1J + (n % MT1J > 0 ? 1 : 0);

        void (*func)(const Ti* in,
                     Ti*       out,
                     int64_t   m,
                     int64_t   n,
                     int64_t   stride1,
                     int64_t   stride2,
                     int       num_batches,
                     int64_t   batch_stride,
                     int64_t   sizes);
        if(d_in == d_out)
            func = prune_strip_kernel<Ti, Tc, SG0I, SG1J, TT0I, TT1J, true>;
        else
            func = prune_strip_kernel<Ti, Tc, SG0I, SG1J, TT0I, TT1J, false>;
        hipLaunchKernelGGL(func, /* compute kernel*/
                           dim3(block_x, block_y, num_batches),
                           dim3(SG0I * SG1J),
                           0 /*dynamic shared*/,
                           stream,
                           d_in,
                           d_out,
                           m,
                           n,
                           stride0,
                           stride1,
                           num_batches,
                           batch_stride,
                           num_batches * batch_stride);
        return rocsparselt_status_success;
    }
    else if(pruneAlg == rocsparselt_prune_smfmac_tile)
    {
        constexpr int SG0I           = 4;
        constexpr int SG1J           = 4;
        constexpr int TT0I           = 4;
        constexpr int TT1J           = 4;
        constexpr int MT0I           = SG0I * TT0I;
        constexpr int MT1J           = SG1J * TT1J;
        constexpr int PATTERNS_COUNT = 90; // 90 pre-gernated pattens.
        constexpr int THREADS_PER_SG = 32; // fix at 16
        constexpr int PATTERNS_PER_THREAD
            = PATTERNS_COUNT / THREADS_PER_SG + (PATTERNS_COUNT % THREADS_PER_SG != 0 ? 1 : 0);

        int block_x = m / MT0I + (m % MT0I > 0 ? 1 : 0);
        int block_y = n / MT1J + (n % MT1J > 0 ? 1 : 0);

        void (*func)(const Ti* in,
                     Ti*       out,
                     int64_t   m,
                     int64_t   n,
                     int64_t   stride1,
                     int64_t   stride2,
                     int       num_batches,
                     int64_t   batch_stride,
                     int64_t   sizes);
        if(d_in == d_out)
            func = prune_tile_kernel<Ti,
                                     Tc,
                                     SG0I,
                                     SG1J,
                                     TT0I,
                                     TT1J,
                                     PATTERNS_COUNT,
                                     THREADS_PER_SG,
                                     PATTERNS_PER_THREAD,
                                     true>;
        else
            func = prune_tile_kernel<Ti,
                                     Tc,
                                     SG0I,
                                     SG1J,
                                     TT0I,
                                     TT1J,
                                     PATTERNS_COUNT,
                                     THREADS_PER_SG,
                                     PATTERNS_PER_THREAD,
                                     false>;
        hipLaunchKernelGGL(func, /* compute kernel*/
                           dim3(block_x, block_y, num_batches),
                           dim3(SG0I * SG1J * THREADS_PER_SG),
                           0 /*dynamic shared*/,
                           stream,
                           d_in,
                           d_out,
                           m,
                           n,
                           stride0,
                           stride1,
                           num_batches,
                           batch_stride,
                           num_batches * batch_stride);
        return rocsparselt_status_success;
    }
    return rocsparselt_status_not_implemented;
}

template <typename Ti>
rocsparselt_status rocsparselt_smfmac_prune_check_template(const _rocsparselt_handle* handle,
                                                           int64_t                    m,
                                                           int64_t                    n,
                                                           int64_t                    stride0,
                                                           int64_t                    stride1,
                                                           int                        num_batches,
                                                           int64_t                    batch_stride,
                                                           rocsparselt_order          order,
                                                           const Ti*                  d_in,
                                                           int*                       d_out,
                                                           hipStream_t                stream)
{
    constexpr int SG0I = 16;
    constexpr int SG1J = 4;
    constexpr int TT0I = 1;
    constexpr int TT1J = 4;
    constexpr int MT0I = SG0I * TT0I;
    constexpr int MT1J = SG1J * TT1J;

    int block_x = m / MT0I + (m % MT0I > 0 ? 1 : 0);
    int block_y = n / MT1J + (n % MT1J > 0 ? 1 : 0);

    RETURN_IF_HIP_ERROR(hipMemsetAsync(d_out, 0, sizeof(int), stream));
    hipLaunchKernelGGL((prune_check_kernel<Ti, SG0I, SG1J, TT0I, TT1J>), /* compute kernel*/
                       dim3(block_x, block_y, num_batches),
                       dim3(SG0I * SG1J),
                       0 /*dynamic shared*/,
                       stream,
                       d_in,
                       d_out,
                       m,
                       n,
                       stride0,
                       stride1,
                       num_batches,
                       batch_stride,
                       num_batches * batch_stride);
    return rocsparselt_status_success;
}

#ifdef __cplusplus
extern "C" {
#endif

rocsparselt_status rocsparselt_smfmac_prune_impl(const _rocsparselt_handle*    handle,
                                                 const _rocsparselt_mat_descr* matrix,
                                                 int64_t                       m,
                                                 int64_t                       n,
                                                 int64_t                       stride0,
                                                 int64_t                       stride1,
                                                 int64_t                       ld,
                                                 const void*                   d_in,
                                                 void*                         d_out,
                                                 rocsparselt_prune_alg         pruneAlg,
                                                 hipStream_t                   stream)
{

    rocsparselt_order order = matrix->order;
    hipDataType       type  = matrix->type;

    int     num_batches  = matrix->num_batches;
    int64_t batch_stride = matrix->batch_stride;

    //set the number of batches to 1 since in the broadcast case, we only care about contents in first batch.
    if(batch_stride == 0) //boardcast case.
    {
        num_batches  = 1;
        batch_stride = matrix->order == rocsparselt_order_column ? matrix->n * ld : matrix->m * ld;
    }

#define PRUNE_PARAMS(T)                                               \
    handle, m, n, stride0, stride1, num_batches, batch_stride, order, \
        reinterpret_cast<const T*>(d_in), reinterpret_cast<T*>(d_out), pruneAlg, stream

    switch(type)
    {
    case HIP_R_16F:
        return rocsparselt_smfmac_prune_template<__half, float>(PRUNE_PARAMS(__half));
    case HIP_R_16BF:
        return rocsparselt_smfmac_prune_template<hip_bfloat16, float>(PRUNE_PARAMS(hip_bfloat16));
    case HIP_R_8I:
        return rocsparselt_smfmac_prune_template<int8_t, float>(PRUNE_PARAMS(int8_t));
#if HIP_FP8_TYPE_OCP
    case HIP_R_8F_E4M3:
        return rocsparselt_smfmac_prune_template<__hip_fp8_e4m3, float>(
            PRUNE_PARAMS(__hip_fp8_e4m3));
    case HIP_R_8F_E5M2:
        return rocsparselt_smfmac_prune_template<__hip_fp8_e5m2, float>(
            PRUNE_PARAMS(__hip_fp8_e5m2));
#endif
    default:
        log_error(handle,
                  "rocsparselt_smfmac_prune",
                  "datatype",
                  hip_datatype_to_string(type),
                  "is not supported");
        return rocsparselt_status_not_implemented;
    }
}

rocsparselt_status rocsparselt_smfmac_prune_check_impl(const _rocsparselt_handle*    handle,
                                                       const _rocsparselt_mat_descr* matrix,
                                                       int64_t                       m,
                                                       int64_t                       n,
                                                       int64_t                       stride0,
                                                       int64_t                       stride1,
                                                       int64_t                       ld,
                                                       const void*                   d_in,
                                                       int*                          d_out,
                                                       hipStream_t                   stream,
                                                       bool sparse_b = false)
{
    rocsparselt_order order = matrix->order;
    hipDataType       type  = matrix->type;

    int     num_batches  = matrix->num_batches;
    int64_t batch_stride = matrix->batch_stride;

    //set the number of batches to 1 since in the broadcast case, we only care about contents in first batch.
    if(batch_stride == 0) //boardcast case.
    {
        num_batches  = 1;
        batch_stride = matrix->n * ld;
    }

#define PRUNE_CHECK_PARAMS(T)                                         \
    handle, m, n, stride0, stride1, num_batches, batch_stride, order, \
        reinterpret_cast<const T*>(d_in), d_out, stream

    switch(type)
    {
    case HIP_R_16F:
        return rocsparselt_smfmac_prune_check_template<__half>(PRUNE_CHECK_PARAMS(__half));
    case HIP_R_16BF:
        return rocsparselt_smfmac_prune_check_template<hip_bfloat16>(
            PRUNE_CHECK_PARAMS(hip_bfloat16));
    case HIP_R_8I:
        return rocsparselt_smfmac_prune_check_template<int8_t>(PRUNE_CHECK_PARAMS(int8_t));
#if HIP_FP8_TYPE_OCP
    case HIP_R_8F_E4M3:
        return rocsparselt_smfmac_prune_check_template<__hip_fp8_e4m3>(
            PRUNE_CHECK_PARAMS(__hip_fp8_e4m3));
    case HIP_R_8F_E5M2:
        return rocsparselt_smfmac_prune_check_template<__hip_fp8_e5m2>(
            PRUNE_CHECK_PARAMS(__hip_fp8_e5m2));
#endif
    default:
        log_error(handle,
                  "rocsparselt_smfmac_prune_check",
                  "datatype",
                  hip_datatype_to_string(type),
                  "is not supported");
        return rocsparselt_status_not_implemented;
    }
}

/********************************************************************************
 * \brief prunes a dense matrix according to the specified algorithm.
 *******************************************************************************/
rocsparselt_status rocsparselt_smfmac_prune(const rocsparselt_handle*       handle,
                                            const rocsparselt_matmul_descr* matmulDescr,
                                            const void*                     d_in,
                                            void*                           d_out,
                                            rocsparselt_prune_alg           pruneAlg,
                                            hipStream_t                     stream)

{
    // Check if handle is valid
    if(handle == nullptr)
    {
        hipsparselt_cerr << "handle is a NULL pointer" << std::endl;
        return rocsparselt_status_invalid_handle;
    }
    auto _handle = reinterpret_cast<const _rocsparselt_handle*>(handle);
    if(!check_is_init_handle(_handle))
    {
        hipsparselt_cerr << "handle did not initialized or already destroyed" << std::endl;
        return rocsparselt_status_invalid_handle;
    }

    if(matmulDescr == nullptr)
    {
        log_error(_handle, __func__, "matmulDescr is a NULL pointer");
        return rocsparselt_status_invalid_handle;
    }
    auto _matmulDescr = reinterpret_cast<const _rocsparselt_matmul_descr*>(matmulDescr);
    if(!check_is_init_matmul_descr(_matmulDescr))
    {
        log_error(_handle, __func__, "matmulDescr did not initialized or already destroyed");
        return rocsparselt_status_invalid_handle;
    }

    // Check if pointer is valid
    if(d_in == nullptr)
    {
        log_error(_handle, __func__, "d_in is a NULL pointer");
        return rocsparselt_status_invalid_pointer;
    }

    if(d_out == nullptr)
    {
        log_error(_handle, __func__, "d_out is a NULL pointer");
        return rocsparselt_status_invalid_pointer;
    }

    // Check if prune alg is valid
    if(pruneAlg != rocsparselt_prune_smfmac_strip && pruneAlg != rocsparselt_prune_smfmac_tile)
    {
        log_error(_handle, __func__, "pruneAlg", pruneAlg, "is not supported");
        return rocsparselt_status_not_implemented;
    }

    rocsparselt_operation op = _matmulDescr->is_sparse_a ? _matmulDescr->op_A : _matmulDescr->op_B;
    _rocsparselt_mat_descr* _sparseMatDescr
        = _matmulDescr->is_sparse_a ? _matmulDescr->matrix_A : _matmulDescr->matrix_B;
    int64_t m, n, stride0, stride1;
    int64_t ld = _sparseMatDescr->ld;
    get_prune_matrix_size(_matmulDescr->is_sparse_a, op, _sparseMatDescr, m, n, stride0, stride1);

    log_api(_handle,
            __func__,
            "matmulDescr[in]",
            *_matmulDescr,
            "d_in[in]",
            d_in,
            "d_out[out]",
            d_out,
            "pruneAlg[in]",
            pruneAlg,
            "stream[in]",
            stream);
    return rocsparselt_smfmac_prune_impl(
        _handle, _sparseMatDescr, m, n, stride0, stride1, ld, d_in, d_out, pruneAlg, stream);
}

/********************************************************************************
 * \brief prunes a dense matrix according to the specified algorithm.
 *******************************************************************************/
rocsparselt_status rocsparselt_smfmac_prune2(const rocsparselt_handle*    handle,
                                             const rocsparselt_mat_descr* sparseMatDescr,
                                             int                          isSparseA,
                                             rocsparselt_operation        op,
                                             const void*                  d_in,
                                             void*                        d_out,
                                             rocsparselt_prune_alg        pruneAlg,
                                             hipStream_t                  stream)
{
    // Check if handle is valid
    if(handle == nullptr)
    {
        hipsparselt_cerr << "handle is a NULL pointer" << std::endl;
        return rocsparselt_status_invalid_handle;
    }
    auto _handle = reinterpret_cast<const _rocsparselt_handle*>(handle);
    if(!check_is_init_handle(_handle))
    {
        hipsparselt_cerr << "handle did not initialized or already destroyed" << std::endl;
        return rocsparselt_status_invalid_handle;
    }

    if(sparseMatDescr == nullptr)
    {
        log_error(_handle, __func__, "sparseMatDescr is a NULL pointer");
        return rocsparselt_status_invalid_handle;
    }
    auto _sparseMatDescr = reinterpret_cast<_rocsparselt_mat_descr*>(
        const_cast<rocsparselt_mat_descr*>(sparseMatDescr));
    if(!check_is_init_mat_descr(_sparseMatDescr))
    {
        log_error(_handle, __func__, "sparseMatDescr did not initialized or already destroyed");
        return rocsparselt_status_invalid_handle;
    }

    if(op != rocsparselt_operation_none && op != rocsparselt_operation_transpose)
    {
        log_error(_handle, __func__, "op is invalid");
        return rocsparselt_status_invalid_value;
    }

    // Check if pointer is valid
    if(d_in == nullptr)
    {
        log_error(_handle, __func__, "d_in is a NULL pointer");
        return rocsparselt_status_invalid_pointer;
    }

    if(d_out == nullptr)
    {
        log_error(_handle, __func__, "d_out is a NULL pointer");
        return rocsparselt_status_invalid_pointer;
    }

    // Check if prune alg is valid
    if(pruneAlg != rocsparselt_prune_smfmac_strip && pruneAlg != rocsparselt_prune_smfmac_tile)
    {
        log_error(_handle, __func__, "pruneAlg", pruneAlg, "is not supported");
        return rocsparselt_status_not_implemented;
    }

    // Check if matrix A is a structured matrix
    if(_sparseMatDescr->m_type != rocsparselt_matrix_type_structured)
    {
        log_error(_handle, __func__, "Matrix is not a structured matrix");
        return rocsparselt_status_not_implemented;
    }

    initSparseMatrixLayout(op, sparseMatDescr, isSparseA);

    log_api(_handle,
            __func__,
            "sparseMatDescr[in]",
            *_sparseMatDescr,
            "isSparseA[in]",
            isSparseA,
            "op[in]",
            rocsparselt_operation_to_string(op),
            "d_in[in]",
            d_in,
            "d_out[out]",
            d_out,
            "pruneAlg[in]",
            pruneAlg,
            "stream[in]",
            stream);

    int64_t m, n, stride0, stride1;
    int64_t ld = _sparseMatDescr->ld;
    get_prune_matrix_size(isSparseA, op, _sparseMatDescr, m, n, stride0, stride1);

    return rocsparselt_smfmac_prune_impl(
        _handle, _sparseMatDescr, m, n, stride0, stride1, ld, d_in, d_out, pruneAlg, stream);
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparselt_status rocsparselt_smfmac_prune_check(const rocsparselt_handle*       handle,
                                                  const rocsparselt_matmul_descr* matmulDescr,
                                                  const void*                     d_in,
                                                  int*                            d_out,
                                                  hipStream_t                     stream)
{
    // Check if handle is valid
    if(handle == nullptr)
    {
        hipsparselt_cerr << "handle is a NULL pointer" << std::endl;
        return rocsparselt_status_invalid_handle;
    }
    auto _handle = reinterpret_cast<const _rocsparselt_handle*>(handle);
    if(!check_is_init_handle(_handle))
    {
        hipsparselt_cerr << "handle did not initialized or already destroyed" << std::endl;
        return rocsparselt_status_invalid_handle;
    }

    if(matmulDescr == nullptr)
    {
        log_error(_handle, __func__, "matmulDescr is a NULL pointer");
        return rocsparselt_status_invalid_handle;
    }
    auto _matmulDescr = reinterpret_cast<const _rocsparselt_matmul_descr*>(matmulDescr);
    if(!check_is_init_matmul_descr(_matmulDescr))
    {
        log_error(_handle, __func__, "matmulDescr did not initialized or already destroyed");
        return rocsparselt_status_invalid_handle;
    }

    // Check if pointer is valid
    if(d_in == nullptr)
    {
        log_error(_handle, __func__, "d_in is a NULL pointer");
        return rocsparselt_status_invalid_pointer;
    }

    if(d_out == nullptr)
    {
        log_error(_handle, __func__, "d_out is a NULL pointer");
        return rocsparselt_status_invalid_pointer;
    }

    rocsparselt_operation op = _matmulDescr->is_sparse_a ? _matmulDescr->op_A : _matmulDescr->op_B;
    _rocsparselt_mat_descr* _sparseMatDescr
        = _matmulDescr->is_sparse_a ? _matmulDescr->matrix_A : _matmulDescr->matrix_B;
    int64_t m, n, stride0, stride1;
    int64_t ld = _sparseMatDescr->ld;
    get_prune_matrix_size(_matmulDescr->is_sparse_a, op, _sparseMatDescr, m, n, stride0, stride1);

    log_api(_handle,
            __func__,
            "matmulDescr[in]",
            *_matmulDescr,
            "d_in[in]",
            d_in,
            "d_out[out]",
            d_out,
            "stream[in]",
            stream);
    return rocsparselt_smfmac_prune_check_impl(
        _handle, _sparseMatDescr, m, n, stride0, stride1, ld, d_in, d_out, stream);
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparselt_status rocsparselt_smfmac_prune_check2(const rocsparselt_handle*    handle,
                                                   const rocsparselt_mat_descr* sparseMatDescr,
                                                   int                          isSparseA,
                                                   rocsparselt_operation        op,
                                                   const void*                  d_in,
                                                   int*                         d_out,
                                                   hipStream_t                  stream)
{
    // Check if handle is valid
    if(handle == nullptr)
    {
        hipsparselt_cerr << "handle is a NULL pointer" << std::endl;
        return rocsparselt_status_invalid_handle;
    }
    auto _handle = reinterpret_cast<const _rocsparselt_handle*>(handle);
    if(!check_is_init_handle(_handle))
    {
        hipsparselt_cerr << "handle did not initialized or already destroyed" << std::endl;
        return rocsparselt_status_invalid_handle;
    }

    if(sparseMatDescr == nullptr)
    {
        log_error(_handle, __func__, "sparseMatDescr is a NULL pointer");
        return rocsparselt_status_invalid_handle;
    }
    auto _sparseMatDescr = reinterpret_cast<_rocsparselt_mat_descr*>(
        const_cast<rocsparselt_mat_descr*>(sparseMatDescr));
    if(!check_is_init_mat_descr(_sparseMatDescr))
    {
        log_error(_handle, __func__, "sparseMatDescr did not initialized or already destroyed");
        return rocsparselt_status_invalid_handle;
    }

    if(op != rocsparselt_operation_none && op != rocsparselt_operation_transpose)
    {
        log_error(_handle, __func__, "op is invalid");
        return rocsparselt_status_invalid_value;
    }

    // Check if pointer is valid
    if(d_in == nullptr)
    {
        log_error(_handle, __func__, "d_in is a NULL pointer");
        return rocsparselt_status_invalid_pointer;
    }

    if(d_out == nullptr)
    {
        log_error(_handle, __func__, "d_out is a NULL pointer");
        return rocsparselt_status_invalid_pointer;
    }

    // Check if matrix A is a structured matrix
    if(_sparseMatDescr->m_type != rocsparselt_matrix_type_structured)
    {
        log_error(_handle, __func__, "Matrix is not a structured matrix");
        return rocsparselt_status_not_implemented;
    }

    initSparseMatrixLayout(op, sparseMatDescr, isSparseA);

    log_api(_handle,
            __func__,
            "sparseMatDescr[in]",
            *_sparseMatDescr,
            "isSparseA[in]",
            isSparseA,
            "op[in]",
            rocsparselt_operation_to_string(op),
            "d_in[in]",
            d_in,
            "d_out[out]",
            d_out,
            "stream[in]",
            stream);

    int64_t m, n, stride0, stride1;
    int64_t ld = _sparseMatDescr->ld;
    get_prune_matrix_size(isSparseA, op, _sparseMatDescr, m, n, stride0, stride1);

    return rocsparselt_smfmac_prune_check_impl(
        _handle, _sparseMatDescr, m, n, stride0, stride1, ld, d_in, d_out, stream);
}

#ifdef __cplusplus
}
#endif
