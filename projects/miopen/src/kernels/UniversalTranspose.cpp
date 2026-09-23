// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#endif

#include "float_types.h"
#include "miopen_cstdint.hpp"

#ifndef INDEX_TYPE
#define INDEX_TYPE uint64_t
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef FLOAT
#define FLOAT float
#endif

using index_t = INDEX_TYPE;

extern "C" __global__
__launch_bounds__(BLOCK_SIZE) void UniversalTranspose(const FLOAT* __restrict__ in,
                                                      FLOAT* __restrict__ out,
                                                      index_t lens_n,
                                                      index_t lens_c,
                                                      index_t lens_d,
                                                      index_t lens_h,
                                                      index_t lens_w,
                                                      index_t in_strides_n,
                                                      index_t in_strides_c,
                                                      index_t in_strides_d,
                                                      index_t in_strides_h,
                                                      index_t in_strides_w,
                                                      index_t out_strides_n,
                                                      index_t out_strides_c,
                                                      index_t out_strides_d,
                                                      index_t out_strides_h,
                                                      index_t out_strides_w)
{
    const index_t global_size = static_cast<index_t>(gridDim.x) * static_cast<index_t>(blockDim.x);
    const index_t global_id =
        static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) + threadIdx.x;

    const index_t lens_wh    = lens_w * lens_h;
    const index_t lens_whd   = lens_wh * lens_d;
    const index_t lens_whdc  = lens_whd * lens_c;
    const index_t lens_whdcn = lens_whdc * lens_n;

    for(index_t id = global_id; id < lens_whdcn; id += global_size)
    {
        const index_t n     = id / lens_whdc;
        const index_t rem_n = id - n * lens_whdc;
        const index_t c     = rem_n / lens_whd;
        const index_t rem_c = rem_n - c * lens_whd;
        const index_t d     = rem_c / lens_wh;
        const index_t rem_d = rem_c - d * lens_wh;
        const index_t h     = rem_d / lens_w;
        const index_t w     = rem_d - h * lens_w;

        const index_t in_id = n * in_strides_n + c * in_strides_c + d * in_strides_d +
                              h * in_strides_h + w * in_strides_w;

        const index_t out_id = n * out_strides_n + c * out_strides_c + d * out_strides_d +
                               h * out_strides_h + w * out_strides_w;

        out[out_id] = in[in_id];
    }
}

#ifndef TILE_SIZE_X
#define TILE_SIZE_X 16
#endif

#ifndef TILE_SIZE_Y
#define TILE_SIZE_Y 16
#endif

extern "C" __global__
__launch_bounds__(BLOCK_SIZE) void TiledTranspose(const FLOAT* __restrict__ in,
                                                  FLOAT* __restrict__ out,
                                                  index_t lens_n,
                                                  index_t lens_c,
                                                  index_t lens_d,
                                                  index_t lens_h,
                                                  index_t lens_w,
                                                  index_t in_strides_n,
                                                  index_t in_strides_c,
                                                  index_t in_strides_d,
                                                  index_t in_strides_h,
                                                  index_t in_strides_w,
                                                  index_t out_strides_n,
                                                  index_t out_strides_c,
                                                  index_t out_strides_d,
                                                  index_t out_strides_h,
                                                  index_t out_strides_w)
{
    __shared__ FLOAT tile[TILE_SIZE_Y][TILE_SIZE_X + 1];

    constexpr index_t TILE_C    = TILE_SIZE_Y;
    constexpr index_t TILE_W    = TILE_SIZE_X;
    constexpr index_t TILE_AREA = TILE_C * TILE_W;

    const index_t tiles_c     = (lens_c + TILE_C - 1) / TILE_C;
    const index_t tiles_w     = (lens_w + TILE_W - 1) / TILE_W;
    const index_t tiles_cw    = tiles_c * tiles_w;
    const index_t tiles_hcw   = lens_h * tiles_cw;
    const index_t tiles_dhcw  = lens_d * tiles_hcw;
    const index_t total_tiles = lens_n * tiles_dhcw;

    for(index_t tile_id = blockIdx.x; tile_id < total_tiles; tile_id += gridDim.x)
    {
        const index_t n          = tile_id / tiles_dhcw;
        const index_t rem_n      = tile_id - n * tiles_dhcw;
        const index_t d          = rem_n / tiles_hcw;
        const index_t rem_d      = rem_n - d * tiles_hcw;
        const index_t h          = rem_d / tiles_cw;
        const index_t rem_h      = rem_d - h * tiles_cw;
        const index_t tile_c_idx = rem_h / tiles_w;
        const index_t tile_w_idx = rem_h - tile_c_idx * tiles_w;

        const index_t c_base = tile_c_idx * TILE_C;
        const index_t w_base = tile_w_idx * TILE_W;

        const index_t tile_c_size = (c_base + TILE_C <= lens_c) ? TILE_C : (lens_c - c_base);
        const index_t tile_w_size = (w_base + TILE_W <= lens_w) ? TILE_W : (lens_w - w_base);

        const index_t in_base  = n * in_strides_n + d * in_strides_d + h * in_strides_h;
        const index_t out_base = n * out_strides_n + d * out_strides_d + h * out_strides_h;

        for(index_t local_idx = threadIdx.x; local_idx < TILE_AREA; local_idx += BLOCK_SIZE)
        {
            const index_t local_c = local_idx / TILE_W;
            const index_t local_w = local_idx - local_c * TILE_W;

            if(local_c < tile_c_size && local_w < tile_w_size)
            {
                const index_t c        = c_base + local_c;
                const index_t w        = w_base + local_w;
                const index_t in_idx   = in_base + c * in_strides_c + w * in_strides_w;
                tile[local_c][local_w] = in[in_idx];
            }
        }

        __syncthreads();

        for(index_t local_idx = threadIdx.x; local_idx < TILE_AREA; local_idx += BLOCK_SIZE)
        {
            const index_t local_w = local_idx / TILE_C;
            const index_t local_c = local_idx - local_w * TILE_C;

            if(local_c < tile_c_size && local_w < tile_w_size)
            {
                const index_t c       = c_base + local_c;
                const index_t w       = w_base + local_w;
                const index_t out_idx = out_base + c * out_strides_c + w * out_strides_w;
                out[out_idx]          = tile[local_c][local_w];
                __syncthreads(); // Ensure all threads finish reading before next iteration
                                 // overwrites tile
            }
        }
    }
}

#ifndef VECTOR_SIZE
#define VECTOR_SIZE 4
#endif

#if VECTOR_SIZE == 4
#if MIOPEN_USE_FP32 == 1
using vec_t = float4;
#elif MIOPEN_USE_FP16 == 1
using vec_t = _Float16 __attribute__((ext_vector_type(4)));
#elif MIOPEN_USE_BFP16 == 1
using vec_t = ushort4;
#else
using vec_t = float4;
#endif
#elif VECTOR_SIZE == 2
#if MIOPEN_USE_FP32 == 1
using vec_t = float2;
#elif MIOPEN_USE_FP16 == 1
using vec_t = _Float16 __attribute__((ext_vector_type(2)));
#elif MIOPEN_USE_BFP16 == 1
using vec_t = ushort2;
#else
using vec_t = float2;
#endif
#else // VECTOR_SIZE == 1
using vec_t = FLOAT;
#endif

constexpr index_t kVectorSize = VECTOR_SIZE;
constexpr index_t kBlockSize  = BLOCK_SIZE;

template <typename VecType, typename ScalarType>
__device__ __forceinline__ VecType load_vector(const ScalarType* ptr)
{
    return *reinterpret_cast<const VecType*>(__builtin_assume_aligned(ptr, sizeof(VecType)));
}

template <typename VecType, typename ScalarType>
__device__ __forceinline__ void store_vector(ScalarType* ptr, VecType val)
{
    *reinterpret_cast<VecType*>(__builtin_assume_aligned(ptr, sizeof(VecType))) = val;
}

extern "C" __global__
__launch_bounds__(BLOCK_SIZE) void VectorizedTranspose(const FLOAT* __restrict__ in,
                                                       FLOAT* __restrict__ out,
                                                       index_t lens_n,
                                                       index_t lens_c,
                                                       index_t lens_d,
                                                       index_t lens_h,
                                                       index_t lens_w,
                                                       index_t in_strides_n,
                                                       index_t in_strides_c,
                                                       index_t in_strides_d,
                                                       index_t in_strides_h,
                                                       index_t in_strides_w,
                                                       index_t out_strides_n,
                                                       index_t out_strides_c,
                                                       index_t out_strides_d,
                                                       index_t out_strides_h,
                                                       index_t out_strides_w,
                                                       bool vectorize_in,
                                                       bool vectorize_out)
{
    const index_t global_size = static_cast<index_t>(gridDim.x) * static_cast<index_t>(blockDim.x);
    const index_t global_id =
        static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) + threadIdx.x;

    const index_t lens_wh    = lens_w * lens_h;
    const index_t lens_whd   = lens_wh * lens_d;
    const index_t lens_whdc  = lens_whd * lens_c;
    const index_t lens_whdcn = lens_whdc * lens_n;

    if(vectorize_in && vectorize_out)
    {
        const index_t lens_w_vec    = lens_w / kVectorSize;
        const index_t lens_wh_vec   = lens_w_vec * lens_h;
        const index_t lens_whd_vec  = lens_wh_vec * lens_d;
        const index_t lens_whdc_vec = lens_whd_vec * lens_c;
        const index_t total_vec     = lens_whdc_vec * lens_n;

        for(index_t id = global_id; id < total_vec; id += global_size)
        {
            const index_t n     = id / lens_whdc_vec;
            const index_t rem_n = id - n * lens_whdc_vec;
            const index_t c     = rem_n / lens_whd_vec;
            const index_t rem_c = rem_n - c * lens_whd_vec;
            const index_t d     = rem_c / lens_wh_vec;
            const index_t rem_d = rem_c - d * lens_wh_vec;
            const index_t h     = rem_d / lens_w_vec;
            const index_t w_vec = rem_d - h * lens_w_vec;

            const index_t w = w_vec * kVectorSize;

            const index_t in_id =
                n * in_strides_n + c * in_strides_c + d * in_strides_d + h * in_strides_h + w;

            const index_t out_id =
                n * out_strides_n + c * out_strides_c + d * out_strides_d + h * out_strides_h + w;

            vec_t val = load_vector<vec_t, FLOAT>(&in[in_id]);
            store_vector<vec_t, FLOAT>(&out[out_id], val);
        }
    }
    else if(vectorize_in)
    {
        const index_t lens_w_vec    = lens_w / kVectorSize;
        const index_t lens_wh_vec   = lens_w_vec * lens_h;
        const index_t lens_whd_vec  = lens_wh_vec * lens_d;
        const index_t lens_whdc_vec = lens_whd_vec * lens_c;
        const index_t total_vec     = lens_whdc_vec * lens_n;

        for(index_t id = global_id; id < total_vec; id += global_size)
        {
            const index_t n     = id / lens_whdc_vec;
            const index_t rem_n = id - n * lens_whdc_vec;
            const index_t c     = rem_n / lens_whd_vec;
            const index_t rem_c = rem_n - c * lens_whd_vec;
            const index_t d     = rem_c / lens_wh_vec;
            const index_t rem_d = rem_c - d * lens_wh_vec;
            const index_t h     = rem_d / lens_w_vec;
            const index_t w_vec = rem_d - h * lens_w_vec;

            const index_t w = w_vec * kVectorSize;

            const index_t in_id =
                n * in_strides_n + c * in_strides_c + d * in_strides_d + h * in_strides_h + w;

            vec_t val = load_vector<vec_t, FLOAT>(&in[in_id]);

            const FLOAT* val_ptr = reinterpret_cast<const FLOAT*>(&val);
#pragma unroll
            for(index_t i = 0; i < kVectorSize; ++i)
            {
                const index_t out_id = n * out_strides_n + c * out_strides_c + d * out_strides_d +
                                       h * out_strides_h + (w + i) * out_strides_w;
                out[out_id] = val_ptr[i];
            }
        }
    }
    else if(vectorize_out)
    {
        const index_t lens_w_vec    = lens_w / kVectorSize;
        const index_t lens_wh_vec   = lens_w_vec * lens_h;
        const index_t lens_whd_vec  = lens_wh_vec * lens_d;
        const index_t lens_whdc_vec = lens_whd_vec * lens_c;
        const index_t total_vec     = lens_whdc_vec * lens_n;

        for(index_t id = global_id; id < total_vec; id += global_size)
        {
            const index_t n     = id / lens_whdc_vec;
            const index_t rem_n = id - n * lens_whdc_vec;
            const index_t c     = rem_n / lens_whd_vec;
            const index_t rem_c = rem_n - c * lens_whd_vec;
            const index_t d     = rem_c / lens_wh_vec;
            const index_t rem_d = rem_c - d * lens_wh_vec;
            const index_t h     = rem_d / lens_w_vec;
            const index_t w_vec = rem_d - h * lens_w_vec;

            const index_t w = w_vec * kVectorSize;

            vec_t val;
            FLOAT* val_ptr = reinterpret_cast<FLOAT*>(&val);
#pragma unroll
            for(index_t i = 0; i < kVectorSize; ++i)
            {
                const index_t in_id = n * in_strides_n + c * in_strides_c + d * in_strides_d +
                                      h * in_strides_h + (w + i) * in_strides_w;
                val_ptr[i] = in[in_id];
            }

            const index_t out_id =
                n * out_strides_n + c * out_strides_c + d * out_strides_d + h * out_strides_h + w;

            store_vector<vec_t, FLOAT>(&out[out_id], val);
        }
    }
    else
    {
        for(index_t id = global_id; id < lens_whdcn; id += global_size)
        {
            const index_t n     = id / lens_whdc;
            const index_t rem_n = id - n * lens_whdc;
            const index_t c     = rem_n / lens_whd;
            const index_t rem_c = rem_n - c * lens_whd;
            const index_t d     = rem_c / lens_wh;
            const index_t rem_d = rem_c - d * lens_wh;
            const index_t h     = rem_d / lens_w;
            const index_t w     = rem_d - h * lens_w;

            const index_t in_id = n * in_strides_n + c * in_strides_c + d * in_strides_d +
                                  h * in_strides_h + w * in_strides_w;

            const index_t out_id = n * out_strides_n + c * out_strides_c + d * out_strides_d +
                                   h * out_strides_h + w * out_strides_w;

            out[out_id] = in[in_id];
        }
    }
}
