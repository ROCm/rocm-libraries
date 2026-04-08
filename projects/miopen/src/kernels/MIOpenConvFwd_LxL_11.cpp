// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// HIP port of the MIOpenConvFwd_LxL_11 OpenCL kernel.
// Implements a tiled, LDS-cached forward convolution optimised for large
// filters with stride > 1 (canonical use-case: 11x11 kernel, stride 4).
//
// All algorithm parameters are injected at compile time via -D defines
// (same macro names as the original OpenCL kernel).

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

__device__ uint iDiv_legacy(uint v, uint d)
{
    uint r = (uint)((float)v * (1.0f / (float)d) + 0.00001f);
    return r;
}

__device__ uint iMod(uint v, uint u, uint d) { return v - __mul24(u, d); }

// float_types.h uses FLOAT / FLOAT_ACCUM for HIP and _FLOAT / _FLOAT_ACCUM for OpenCL.
// This kernel uses the _FLOAT / _FLOAT_ACCUM names throughout; alias them for HIP.
#ifdef __HIP_PLATFORM_AMD__
#ifndef _FLOAT
#define _FLOAT FLOAT
#endif
#define _FLOAT_ACCUM FLOAT_ACCUM
#endif

// ---------------------------------------------------------------------------
// Compile-time derived constants (mirror the OpenCL kernel header block)
// ---------------------------------------------------------------------------

#ifndef MLO_N_FILTER_SPLITS1
#define MLO_N_FILTER_SPLITS1 ((MLO_FILTER_SIZE1 + MLO_FILTER_STRIDE1 - 1) / MLO_FILTER_STRIDE1)
#endif
#ifndef MLO_N_FILTER_SPLITS0
#define MLO_N_FILTER_SPLITS0 ((MLO_FILTER_SIZE0 + MLO_FILTER_STRIDE0 - 1) / MLO_FILTER_STRIDE0)
#endif
#ifndef MLO_OUT_PIX_TILE0
#define MLO_OUT_PIX_TILE0 MLO_N_FILTER_SPLITS0
#endif

// Output extent in the vertical direction handled by one workgroup (pass 1).
#ifndef MLO_OUT_EXTENT1
#define MLO_PROCESSING_WIDTH ((MLO_OUT_WIDTH + MLO_OUT_PIX_TILE0 - 1) / MLO_OUT_PIX_TILE0)
#define MLO_OUT_EXTENT1 (MLO_GRP_SZ / MLO_PROCESSING_WIDTH)
#endif

#define MLO_WEI_LCL_WIDTH MLO_FILTER_SIZE0
#define MLO_WEI_EXTENT1 MLO_N_FILTER_SPLITS1
#define MLO_WEI_SZ (MLO_WEI_EXTENT1 * MLO_WEI_LCL_WIDTH)

#ifndef MLO_WEI_LCL_SZ
#define MLO_WEI_LCL_SZ (MLO_WEI_SZ * MLO_N_LCL_OUT_MAPS * MLO_N_LCL_IN_MAPS)
#endif

#ifndef MLO_IN_LCL_HEIGHT
#define MLO_IN_LCL_HEIGHT (MLO_OUT_EXTENT1 + MLO_N_FILTER_SPLITS1 - 1)
#endif

#define MLO_N_IN_HORIZ_PIX_READS (MLO_IN_WIDTH)
#ifndef MLO_N_IN_HORIZ_READS
#define MLO_N_IN_HORIZ_READS ((MLO_N_IN_HORIZ_PIX_READS + MLO_READ_UNIT - 1) / MLO_READ_UNIT)
#endif

#define MLO_IN_N_PIXS_OFF \
    (MLO_N_IN_HORIZ_PIX_READS - (MLO_N_IN_HORIZ_PIX_READS / MLO_READ_UNIT) * MLO_READ_UNIT)

#define MLO_IN_LCL_WIDTH (MLO_N_IN_HORIZ_READS * MLO_READ_UNIT + 2 * MLO_FILTER_PAD0)
#define MLO_IN_LCL_SZ (MLO_IN_LCL_WIDTH * MLO_IN_LCL_HEIGHT)
#define MLO_TOTAL_IN_LCL_SZ (MLO_N_LCL_BATCHS * MLO_IN_LCL_SZ * MLO_N_LCL_IN_MAPS)

#ifndef MLO_LCL_MEM_SZ
#define MLO_LCL_MEM_SZ (MLO_WEI_LCL_SZ + MLO_TOTAL_IN_LCL_SZ)
#endif

// ---------------------------------------------------------------------------
// Wave / thread-ID helpers
// ---------------------------------------------------------------------------

#if defined(__AMDGCN__)
#define uniform(x) __builtin_amdgcn_readfirstlane(x)
#else
#define uniform(x) (x)
#endif

__device__ uint getWaveId()
{
#if defined(__AMDGCN__)
    return __builtin_amdgcn_readfirstlane((uint)(threadIdx.x >> MLO_LG2_WAVE_SZ));
#else
    return (uint)(threadIdx.x >> MLO_LG2_WAVE_SZ);
#endif
}

__device__ uint getWaveLocalId() { return (uint)(threadIdx.x & ((1u << MLO_LG2_WAVE_SZ) - 1u)); }

// ---------------------------------------------------------------------------
// fetchWeights – cooperative load of weight rows into shared memory
// ---------------------------------------------------------------------------

__device__ void fetchWeights(uint c,
                             uint k_idx,
                             uint f_s,
                             uint lcl_id,
                             uint wei_read,
                             uint gbl_wei_off,
                             _FLOAT* __restrict__ wei_mem,
                             const _FLOAT* __restrict__ weights)
{
    for(uint w = lcl_id; w < (wei_read / MLO_FILTER_SIZE0) * MLO_N_LCL_OUT_MAPS; w += MLO_GRP_SZ)
    {
        uint k = iDiv_legacy(w, (wei_read / MLO_FILTER_SIZE0));
        uint j = iMod(w, k, (wei_read / MLO_FILTER_SIZE0));
        int wei_off =
            ((j * MLO_FILTER_STRIDE1 + f_s) < MLO_FILTER_SIZE1 && k_idx + k < MLO_N_OUTPUTS)
                ? (int)(gbl_wei_off + k * MLO_WEI_BATCH_STRIDE + c * MLO_WEI_CHANNEL_STRIDE +
                        (j * MLO_FILTER_STRIDE1 + f_s) * MLO_FILTER_SIZE0)
                : 0;
        const _FLOAT* wei_p = &weights[wei_off];

        for(uint i = 0; i < MLO_FILTER_SIZE0; ++i)
        {
            wei_mem[k * MLO_WEI_SZ + j * MLO_WEI_LCL_WIDTH + i] = wei_p[i];
        }
    }
}

// ---------------------------------------------------------------------------
// fetchData (pass 1) – cooperative load of input scanlines into shared memory
// ---------------------------------------------------------------------------

__device__ void fetchData(uint f_s,
                          uint lcl_id,
                          uint lcl_scan,
                          uint n_reads,
                          int in_y,
                          uint gbl_in_scan_off,
                          _FLOAT* __restrict__ bot_mem,
                          const _FLOAT* __restrict__ bot)
{
    _FLOAT in_rd_data[MLO_READ_UNIT];

    for(uint p4 = lcl_id, c_scan = 0; p4 < MLO_N_IN_HORIZ_READS * n_reads * MLO_N_LCL_BATCHS;
        p4 += MLO_GRP_SZ)
    {
        uint b  = 0;
        uint t0 = p4;
#if MLO_N_LCL_BATCHS > 1
        b  = iDiv_legacy(p4, MLO_N_IN_HORIZ_READS * n_reads);
        t0 = iMod(p4, b, MLO_N_IN_HORIZ_READS * n_reads);
#endif
#if MLO_N_IN_HORIZ_READS & (MLO_N_IN_HORIZ_READS - 1)
        c_scan      = iDiv_legacy(t0, MLO_N_IN_HORIZ_READS);
        uint c_pix4 = iMod(t0, c_scan, MLO_N_IN_HORIZ_READS);
#else
        c_scan      = t0 / MLO_N_IN_HORIZ_READS;
        uint c_pix4 = t0 & (MLO_N_IN_HORIZ_READS - 1);
#endif
        int in_scan = (int)((c_scan + lcl_scan) * MLO_FILTER_STRIDE1 + f_s);

        for(uint i = 0; i < MLO_READ_UNIT; ++i)
            in_rd_data[i] = (_FLOAT)0;

        if(0 <= in_y + in_scan && in_y + in_scan < MLO_IN_HEIGHT)
        {
            int gbl_off = (int)(gbl_in_scan_off + b * MLO_IN_BATCH_STRIDE) +
                          in_scan * (int)MLO_IN_STRIDE + (int)(c_pix4 * MLO_READ_UNIT);
            const _FLOAT* bot_p = &bot[gbl_off];

#if MLO_IN_N_PIXS_OFF > 0
            if(c_pix4 == MLO_N_IN_HORIZ_READS - 1)
            {
                for(uint i = 0; i < MLO_IN_N_PIXS_OFF; ++i)
                    in_rd_data[i] = bot_p[i];
            }
            else
#endif
            {
                for(uint i = 0; i < MLO_READ_UNIT; ++i)
                    in_rd_data[i] = bot_p[i];
            }
        }

        int lcl_off = (int)((lcl_scan + c_scan) * MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 +
                            c_pix4 * MLO_READ_UNIT);
        for(uint i = 0; i < MLO_READ_UNIT; ++i)
            bot_mem[lcl_off + i] = in_rd_data[i];
    }
}

// ---------------------------------------------------------------------------
// Convolve (pass 1) – compute partial dot-products from shared memory
// ---------------------------------------------------------------------------

__device__ void Convolve(uint ex_row,
                         uint ex_pix,
                         uint l,
                         uint m,
                         uint wei_h,
                         uint bot_h,
                         const _FLOAT* __restrict__ wei_mem,
                         const _FLOAT* __restrict__ bot_mem,
                         _FLOAT_ACCUM* pvt_accum)
{
    _FLOAT wei_vals[MLO_N_LCL_OUT_MAPS * MLO_N_FILTER_SPLITS0];
    _FLOAT in_vals[MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 1];

    for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
    {
        for(uint i = 0; i < wei_h; ++i)
        {
            wei_vals[k * MLO_N_FILTER_SPLITS0 + i] =
                wei_mem[k * MLO_WEI_SZ + m * MLO_WEI_LCL_WIDTH + i * MLO_FILTER_STRIDE0 + l];
        }
    }

    for(uint i = 0; i < bot_h; ++i)
    {
        in_vals[i] = bot_mem[(ex_row + m) * MLO_IN_LCL_WIDTH + ex_pix * MLO_FILTER_STRIDE0 +
                             i * MLO_FILTER_STRIDE0 + l];
    }

    for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
    {
        for(uint n = 0; n < MLO_OUT_PIX_TILE0; ++n)
        {
            for(uint i = 0; i < wei_h; ++i)
            {
                pvt_accum[k * MLO_OUT_PIX_TILE0 + n] +=
                    CVT_FLOAT2ACCUM(wei_vals[k * MLO_N_FILTER_SPLITS0 + i]) *
                    CVT_FLOAT2ACCUM(in_vals[n + i]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MIOpenCvFwd11x11 – pass-1 forward kernel
// ---------------------------------------------------------------------------

#define MLO_ACCUM_SZ \
    (MLO_OUT_PIX_TILE1 * MLO_OUT_PIX_TILE0 * MLO_N_LCL_OUT_MAPS * MLO_N_LCL_IN_MAPS)

extern "C" __global__ __launch_bounds__(
    MLO_GRP_SZ0* MLO_GRP_SZ1* MLO_GRP_SZ2) void MIOpenCvFwd11x11(const _FLOAT* __restrict__ bot,
                                                                 const _FLOAT* __restrict__ weights,
#if MLO_CONV_BIAS == 1
                                                                 const _FLOAT* __restrict__ bias,
#endif
                                                                 _FLOAT* __restrict__ top,
                                                                 _FLOAT /*padding_val*/)
{
    __shared__ _FLOAT lcl_mem[MLO_LCL_MEM_SZ];
    _FLOAT* bot_mem = lcl_mem;
    _FLOAT* wei_mem = lcl_mem + MLO_TOTAL_IN_LCL_SZ;

    uint lcl_id = threadIdx.x;

    uint ob     = blockIdx.x;
    uint k_idx  = blockIdx.y * (MLO_N_LCL_OUT_MAPS);
    uint ib_idx = blockIdx.z * MLO_N_LCL_BATCHS;
    uint ib     = ib_idx;

    int gbl_in_off   = (int)(ib * MLO_IN_BATCH_STRIDE);
    uint gbl_wei_off = k_idx * MLO_WEI_BATCH_STRIDE;
    uint out_y       = ob * MLO_OUT_EXTENT1;
    int in_y         = (int)(out_y * MLO_FILTER_STRIDE1) - (int)MLO_FILTER_PAD1;
    gbl_in_off += in_y * (int)MLO_IN_STRIDE;

    _FLOAT_ACCUM pvt_accum[MLO_ACCUM_SZ];

    // Zero shared memory
    for(uint i = lcl_id; i < MLO_LCL_MEM_SZ; i += MLO_GRP_SZ)
        lcl_mem[i] = (_FLOAT)0;

// Derive per-thread output tile position
#if MLO_PROCESSING_WIDTH & (MLO_PROCESSING_WIDTH - 1)
    uint ex_row = iDiv_legacy(lcl_id, MLO_PROCESSING_WIDTH);
    uint ex_col = iMod(lcl_id, ex_row, MLO_PROCESSING_WIDTH);
#else
    uint ex_row = lcl_id / MLO_PROCESSING_WIDTH;
    uint ex_col = lcl_id & (MLO_PROCESSING_WIDTH - 1);
#if MLO_PROCESSING_WIDTH >= 64
    ex_row = uniform(ex_row);
#endif
#endif
    uint ex_pix = ex_col * MLO_OUT_PIX_TILE0;

    for(uint b = 0; b < MLO_N_BATCH_LOOPS; ++b, gbl_in_off += (int)MLO_IN_BATCH_STRIDE)
    {
        int gbl_in_scan_off0 = gbl_in_off;

        for(uint i = 0; i < MLO_ACCUM_SZ; ++i)
            pvt_accum[i] = CVT_FLOAT2ACCUM((_FLOAT)0);

#ifdef __AMDGCN__
#pragma unroll 4
#endif
        for(uint c = 0, gbl_in_scan_off = (uint)gbl_in_scan_off0; c < MLO_N_INPUTS;
            ++c, gbl_in_scan_off += MLO_IN_CHANNEL_STRIDE)
        {
            // ---- Filter-stride sub-loops (all but the last f_s) ----
            uint f_s = 0;
            for(; f_s < MLO_FILTER_STRIDE1 - 1; ++f_s)
            {
                __syncthreads();
                fetchWeights(c, k_idx, f_s, lcl_id, MLO_WEI_SZ, gbl_wei_off, wei_mem, weights);
                fetchData(f_s, lcl_id, 0, MLO_IN_LCL_HEIGHT, in_y, gbl_in_scan_off, bot_mem, bot);
                __syncthreads();

#pragma unroll
                for(uint m = 0; m < MLO_N_FILTER_SPLITS1; ++m)
                {
                    uint l;
                    for(l = 0; l < MLO_FILTER_STRIDE0 - 1; ++l)
                    {
                        Convolve(ex_row,
                                 ex_pix,
                                 l,
                                 m,
                                 MLO_N_FILTER_SPLITS0,
                                 MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 1,
                                 wei_mem,
                                 bot_mem,
                                 pvt_accum);
                    }
                    Convolve(ex_row,
                             ex_pix,
                             l,
                             m,
                             MLO_N_FILTER_SPLITS0 - 1,
                             MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 2,
                             wei_mem,
                             bot_mem,
                             pvt_accum);
                }
            }

            // ---- Last f_s (one fewer input row to read) ----
            {
                __syncthreads();

#define MLO_WEI_READ ((MLO_N_FILTER_SPLITS1 - 1) * MLO_WEI_LCL_WIDTH)
                fetchWeights(c, k_idx, f_s, lcl_id, MLO_WEI_READ, gbl_wei_off, wei_mem, weights);
                fetchData(
                    f_s, lcl_id, 0, MLO_IN_LCL_HEIGHT - 1, in_y, gbl_in_scan_off, bot_mem, bot);
                __syncthreads();

#pragma unroll
                for(uint m = 0; m < MLO_N_FILTER_SPLITS1 - 1; ++m)
                {
                    uint l;
                    for(l = 0; l < MLO_FILTER_STRIDE0 - 1; ++l)
                    {
                        Convolve(ex_row,
                                 ex_pix,
                                 l,
                                 m,
                                 MLO_N_FILTER_SPLITS0,
                                 MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 1,
                                 wei_mem,
                                 bot_mem,
                                 pvt_accum);
                    }
                    Convolve(ex_row,
                             ex_pix,
                             l,
                             m,
                             MLO_N_FILTER_SPLITS0 - 1,
                             MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 2,
                             wei_mem,
                             bot_mem,
                             pvt_accum);
                }
#undef MLO_WEI_READ
            }
        } // c

        // Write output tile
        for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
        {
            uint out_off = (ib + b) * MLO_OUT_BATCH_STRIDE + (k_idx + k) * MLO_OUT_CHANNEL_STRIDE +
                           (out_y + ex_row) * MLO_OUT_STRIDE + ex_pix;
            _FLOAT* top_p = &top[out_off];
            for(uint i = 0; i < MLO_OUT_PIX_TILE0; ++i)
            {
                if((k_idx + k) < MLO_N_OUTPUTS && ex_row < MLO_OUT_EXTENT1 &&
                   (out_y + ex_row) < MLO_OUT_HEIGHT && ex_pix + i < MLO_OUT_WIDTH)
                {
                    top_p[i] = CVT_ACCUM2FLOAT(pvt_accum[k * MLO_OUT_PIX_TILE0 + i]);
                }
            }
        }
    } // b
}

#undef MLO_ACCUM_SZ

// ===========================================================================
// Pass-2 kernel (handles the remainder rows when MLO_LAST_OUT_EXTENT1 > 0)
// ===========================================================================

// Redefine LDS sizing macros for the smaller extent used in pass 2.
#undef MLO_LCL_MEM_SZ
#undef MLO_TOTAL_IN_LCL_SZ
#undef MLO_IN_LCL_SZ
#undef MLO_IN_LCL_HEIGHT
#undef MLO_OUT_EXTENT1
#undef MLO_N_LCL_BATCHS

#define MLO_N_LCL_BATCHS MLO_N_LCL_BATCHS_PASS2
#define MLO_OUT_EXTENT1 (MLO_LAST_OUT_EXTENT1)
#define MLO_IN_LCL_HEIGHT (MLO_OUT_EXTENT1 + MLO_N_FILTER_SPLITS1 - 1)
#define MLO_IN_LCL_SZ (MLO_IN_LCL_WIDTH * MLO_IN_LCL_HEIGHT)
#define MLO_TOTAL_IN_LCL_SZ (MLO_N_LCL_BATCHS * MLO_IN_LCL_SZ * MLO_N_LCL_IN_MAPS)
#define MLO_LCL_MEM_SZ (MLO_WEI_LCL_SZ + MLO_TOTAL_IN_LCL_SZ)

// ---------------------------------------------------------------------------
// fetchData2 – pass-2 version that handles multiple batches per workgroup
// ---------------------------------------------------------------------------

__device__ void fetchData2(uint ib,
                           uint f_s,
                           uint lcl_id,
                           uint lcl_scan,
                           uint n_reads,
                           int in_y,
                           int gbl_in_scan_off,
                           _FLOAT* __restrict__ bot_mem,
                           const _FLOAT* __restrict__ bot)
{
    _FLOAT in_rd_data[MLO_READ_UNIT];

    for(uint p4 = lcl_id, c_scan = 0; p4 < MLO_N_IN_HORIZ_READS * n_reads * MLO_N_LCL_BATCHS;
        p4 += MLO_GRP_SZ)
    {
        uint b  = 0;
        uint t0 = p4;
#if MLO_N_LCL_BATCHS > 1
        b  = iDiv_legacy(p4, MLO_N_IN_HORIZ_READS * n_reads);
        t0 = iMod(p4, b, MLO_N_IN_HORIZ_READS * n_reads);
#endif
#if MLO_N_IN_HORIZ_READS & (MLO_N_IN_HORIZ_READS - 1)
        c_scan      = iDiv_legacy(t0, MLO_N_IN_HORIZ_READS);
        uint c_pix4 = iMod(t0, c_scan, MLO_N_IN_HORIZ_READS);
#else
        c_scan      = t0 / MLO_N_IN_HORIZ_READS;
        uint c_pix4 = t0 & (MLO_N_IN_HORIZ_READS - 1);
#endif
        int in_scan = (int)((c_scan + lcl_scan) * MLO_FILTER_STRIDE1 + f_s);

        for(uint i = 0; i < MLO_READ_UNIT; ++i)
            in_rd_data[i] = (_FLOAT)0;

        if(0 <= in_y + in_scan && in_y + in_scan < MLO_IN_HEIGHT && b < MLO_N_LCL_BATCHS &&
           (ib + b) < MLO_BATCH_SZ)
        {
            int gbl_off = gbl_in_scan_off + (int)(b * MLO_IN_BATCH_STRIDE) +
                          in_scan * (int)MLO_IN_STRIDE + (int)(c_pix4 * MLO_READ_UNIT);
            const _FLOAT* bot_p = &bot[gbl_off];

#if MLO_IN_N_PIXS_OFF > 0
            if(c_pix4 == MLO_N_IN_HORIZ_READS - 1)
            {
                for(uint i = 0; i < MLO_IN_N_PIXS_OFF; ++i)
                    in_rd_data[i] = bot_p[i];
            }
            else
#endif
            {
                for(uint i = 0; i < MLO_READ_UNIT; ++i)
                    in_rd_data[i] = bot_p[i];
            }
        }

        if(b < MLO_N_LCL_BATCHS)
        {
            int lcl_off = (int)(b * MLO_IN_LCL_SZ + (lcl_scan + c_scan) * MLO_IN_LCL_WIDTH +
                                MLO_FILTER_PAD0 + c_pix4 * MLO_READ_UNIT);
            for(uint i = 0; i < MLO_READ_UNIT; ++i)
                bot_mem[lcl_off + i] = in_rd_data[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Convolve2 – pass-2 convolution helper (indexes into per-batch LDS slice)
// ---------------------------------------------------------------------------

__device__ void Convolve2(uint b,
                          uint ex_row,
                          uint ex_pix,
                          uint l,
                          uint m,
                          uint wei_h,
                          uint bot_h,
                          const _FLOAT* __restrict__ wei_mem,
                          const _FLOAT* __restrict__ bot_mem,
                          _FLOAT_ACCUM* pvt_accum)
{
    _FLOAT wei_vals[MLO_N_LCL_OUT_MAPS * MLO_N_FILTER_SPLITS0];
    _FLOAT in_vals[MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 1];

    for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
    {
        for(uint i = 0; i < wei_h; ++i)
        {
            wei_vals[k * MLO_N_FILTER_SPLITS0 + i] =
                wei_mem[k * MLO_WEI_SZ + m * MLO_WEI_LCL_WIDTH + i * MLO_FILTER_STRIDE0 + l];
        }
    }

    for(uint i = 0; i < bot_h; ++i)
    {
        in_vals[i] = bot_mem[b * MLO_IN_LCL_SZ + (ex_row + m) * MLO_IN_LCL_WIDTH +
                             ex_pix * MLO_FILTER_STRIDE0 + i * MLO_FILTER_STRIDE0 + l];
    }

    for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
    {
        for(uint n = 0; n < MLO_OUT_PIX_TILE0; ++n)
        {
            for(uint i = 0; i < wei_h; ++i)
            {
                pvt_accum[k * MLO_OUT_PIX_TILE0 + n] +=
                    CVT_FLOAT2ACCUM(wei_vals[k * MLO_N_FILTER_SPLITS0 + i]) *
                    CVT_FLOAT2ACCUM(in_vals[n + i]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MIOpenCvFwd11x11_2 – pass-2 kernel for remainder output rows
// ---------------------------------------------------------------------------

#define MLO_ACCUM_SZ \
    (MLO_OUT_PIX_TILE1 * MLO_OUT_PIX_TILE0 * MLO_N_LCL_OUT_MAPS * MLO_N_LCL_IN_MAPS)

extern "C" __global__
__launch_bounds__(MLO_GRP_SZ0* MLO_GRP_SZ1* MLO_GRP_SZ2) void MIOpenCvFwd11x11_2(
    const _FLOAT* __restrict__ bot,
    const _FLOAT* __restrict__ weights,
#if MLO_CONV_BIAS == 1
    const _FLOAT* __restrict__ bias,
#endif
    _FLOAT* __restrict__ top,
    _FLOAT /*padding_val*/)
{
    __shared__ _FLOAT lcl_mem[MLO_LCL_MEM_SZ];
    _FLOAT* bot_mem = lcl_mem;
    _FLOAT* wei_mem = lcl_mem + MLO_TOTAL_IN_LCL_SZ;

    uint lcl_id = threadIdx.x;

    uint k_idx  = blockIdx.y * (MLO_N_LCL_OUT_MAPS);
    uint ib_idx = blockIdx.z * MLO_N_LCL_BATCHS;
    uint ib     = ib_idx;

    int gbl_in_off   = (int)(ib * MLO_IN_BATCH_STRIDE);
    uint gbl_wei_off = k_idx * MLO_WEI_BATCH_STRIDE;

    // Pass 2 processes the last MLO_LAST_OUT_EXTENT1 rows
    int out_y = (int)(MLO_OUT_HEIGHT - MLO_LAST_OUT_EXTENT1);
    int in_y  = out_y * (int)MLO_FILTER_STRIDE1 - (int)MLO_FILTER_PAD1;
    gbl_in_off += in_y * (int)MLO_IN_STRIDE;

    _FLOAT_ACCUM pvt_accum[MLO_ACCUM_SZ];

    // Zero shared memory
    for(uint i = lcl_id; i < MLO_LCL_MEM_SZ; i += MLO_GRP_SZ)
        lcl_mem[i] = (_FLOAT)0;

// Decompose thread ID into (batch, row, col) within the tile
#if(MLO_PROCESSING_WIDTH * MLO_LAST_OUT_EXTENT1) & (MLO_PROCESSING_WIDTH * MLO_LAST_OUT_EXTENT1 - 1)
    uint bb = iDiv_legacy(lcl_id, (MLO_PROCESSING_WIDTH * MLO_LAST_OUT_EXTENT1));
    uint t0 = iMod(lcl_id, bb, (MLO_PROCESSING_WIDTH * MLO_LAST_OUT_EXTENT1));
#elif(MLO_PROCESSING_WIDTH * MLO_LAST_OUT_EXTENT1) != 0
    uint bb = lcl_id / (MLO_PROCESSING_WIDTH * MLO_LAST_OUT_EXTENT1);
    uint t0 = lcl_id & ((MLO_PROCESSING_WIDTH * MLO_LAST_OUT_EXTENT1) - 1);
#if(MLO_PROCESSING_WIDTH * MLO_LAST_OUT_EXTENT1) >= 64
    bb = uniform(bb);
#endif
#else
    uint bb = lcl_id;
    uint t0 = 0;
#endif

#if MLO_PROCESSING_WIDTH & (MLO_PROCESSING_WIDTH - 1)
    uint ex_row = iDiv_legacy(t0, MLO_PROCESSING_WIDTH);
    uint ex_col = iMod(t0, ex_row, MLO_PROCESSING_WIDTH);
#else
    uint ex_row = t0 / MLO_PROCESSING_WIDTH;
    uint ex_col = t0 & (MLO_PROCESSING_WIDTH - 1);
#endif
    uint ex_pix = ex_col * MLO_OUT_PIX_TILE0;

    for(uint b = 0; b < MLO_N_BATCH_LOOPS; ++b, gbl_in_off += (int)MLO_IN_BATCH_STRIDE)
    {
        int gbl_in_scan_off0 = gbl_in_off;

        for(uint i = 0; i < MLO_ACCUM_SZ; ++i)
            pvt_accum[i] = CVT_FLOAT2ACCUM((_FLOAT)0);

#ifdef __AMDGCN__
#pragma unroll 4
#endif
        for(uint c = 0, gbl_in_scan_off = (uint)gbl_in_scan_off0; c < MLO_N_INPUTS;
            ++c, gbl_in_scan_off += MLO_IN_CHANNEL_STRIDE)
        {
            uint f_s = 0;
            for(; f_s < MLO_FILTER_STRIDE1 - 1; ++f_s)
            {
                __syncthreads();
                fetchWeights(c, k_idx, f_s, lcl_id, MLO_WEI_SZ, gbl_wei_off, wei_mem, weights);
                fetchData2((ib + b),
                           f_s,
                           lcl_id,
                           0,
                           MLO_IN_LCL_HEIGHT,
                           in_y,
                           gbl_in_scan_off,
                           bot_mem,
                           bot);
                __syncthreads();

#pragma unroll
                for(uint m = 0; m < MLO_N_FILTER_SPLITS1; ++m)
                {
                    uint l;
                    for(l = 0; l < MLO_FILTER_STRIDE0 - 1; ++l)
                    {
                        Convolve2(bb,
                                  ex_row,
                                  ex_pix,
                                  l,
                                  m,
                                  MLO_N_FILTER_SPLITS0,
                                  MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 1,
                                  wei_mem,
                                  bot_mem,
                                  pvt_accum);
                    }
                    Convolve2(bb,
                              ex_row,
                              ex_pix,
                              l,
                              m,
                              MLO_N_FILTER_SPLITS0 - 1,
                              MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 2,
                              wei_mem,
                              bot_mem,
                              pvt_accum);
                }
            }

            // Last f_s
            {
                __syncthreads();

#define MLO_WEI_READ ((MLO_N_FILTER_SPLITS1 - 1) * MLO_WEI_LCL_WIDTH)
                fetchWeights(c, k_idx, f_s, lcl_id, MLO_WEI_READ, gbl_wei_off, wei_mem, weights);
                fetchData2((ib + b),
                           f_s,
                           lcl_id,
                           0,
                           MLO_IN_LCL_HEIGHT - 1,
                           in_y,
                           gbl_in_scan_off,
                           bot_mem,
                           bot);
                __syncthreads();

#pragma unroll
                for(uint m = 0; m < MLO_N_FILTER_SPLITS1 - 1; ++m)
                {
                    uint l;
                    for(l = 0; l < MLO_FILTER_STRIDE0 - 1; ++l)
                    {
                        Convolve2(bb,
                                  ex_row,
                                  ex_pix,
                                  l,
                                  m,
                                  MLO_N_FILTER_SPLITS0,
                                  MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 1,
                                  wei_mem,
                                  bot_mem,
                                  pvt_accum);
                    }
                    Convolve2(bb,
                              ex_row,
                              ex_pix,
                              l,
                              m,
                              MLO_N_FILTER_SPLITS0 - 1,
                              MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 2,
                              wei_mem,
                              bot_mem,
                              pvt_accum);
                }
#undef MLO_WEI_READ
            }
        } // c

        // Write output tile
        for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
        {
            uint out_off = (ib + bb + b) * MLO_OUT_BATCH_STRIDE +
                           (k_idx + k) * MLO_OUT_CHANNEL_STRIDE +
                           (out_y + ex_row) * MLO_OUT_STRIDE + ex_pix;
            _FLOAT* top_p = &top[out_off];
            for(uint i = 0; i < MLO_OUT_PIX_TILE0; ++i)
            {
                if((ib + bb + b) < MLO_BATCH_SZ && bb < MLO_N_LCL_BATCHS &&
                   (k_idx + k) < MLO_N_OUTPUTS && ex_row < MLO_LAST_OUT_EXTENT1 &&
                   (out_y + ex_row) < MLO_OUT_HEIGHT && ex_pix + i < MLO_OUT_WIDTH)
                {
                    top_p[i] = CVT_ACCUM2FLOAT(pvt_accum[k * MLO_OUT_PIX_TILE0 + i]);
                }
            }
        }
    } // b
}

#undef MLO_ACCUM_SZ
