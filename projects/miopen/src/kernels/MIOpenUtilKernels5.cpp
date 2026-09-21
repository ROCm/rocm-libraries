// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

#ifndef MIOPEN_DEF_USE_ALPHA
#define MIOPEN_DEF_USE_ALPHA 0
#endif
#ifndef MIOPEN_DEF_USE_BETA
#define MIOPEN_DEF_USE_BETA 0
#endif

#define UNUSED __attribute__((unused))

static inline __device__ unsigned int
GetInOff(const unsigned int p_blck, const unsigned int n, const unsigned int c)
{
#if MIOPEN_DEF_TRANS

    const unsigned int in_off =
#if MIOPEN_DEF_FORWARD
        p_blck * MIOPEN_DEF_RD_BLCK + c * MIOPEN_DEF_HW + n * MIOPEN_DEF_CHW * MIOPEN_DEF_VEC_SIZE
#else
        p_blck * MIOPEN_DEF_WR_BLCK + n * MIOPEN_DEF_HW * MIOPEN_DEF_VEC_SIZE +
        c * MIOPEN_DEF_NHW_OUT
#endif
        ;

#else

    const unsigned int in_off =
#if MIOPEN_DEF_FORWARD
        p_blck * MIOPEN_DEF_RD_BLCK + c * MIOPEN_DEF_HW * MIOPEN_DEF_VEC_SIZE + n * MIOPEN_DEF_CHW
#else
        p_blck * MIOPEN_DEF_WR_BLCK + c * MIOPEN_DEF_HW * MIOPEN_DEF_VEC_SIZE +
        n * MIOPEN_DEF_CHW_OUT
#endif
        ;

#endif // end of #if MIOPEN_DEF_TRANS

    return in_off;
}

static inline __device__ unsigned int
GetOutOff(const unsigned int p_blck, const unsigned int n, const unsigned int c)
{
#if MIOPEN_DEF_TRANS

    const unsigned int out_off =
#if MIOPEN_DEF_FORWARD
        p_blck * MIOPEN_DEF_WR_BLCK + n * MIOPEN_DEF_HW * MIOPEN_DEF_VEC_SIZE +
        c * MIOPEN_DEF_NHW_OUT
#else
        p_blck * MIOPEN_DEF_RD_BLCK + c * MIOPEN_DEF_HW + n * MIOPEN_DEF_CHW * MIOPEN_DEF_VEC_SIZE
#endif
        ;

#else

    const unsigned int out_off =
#if MIOPEN_DEF_FORWARD
        p_blck * MIOPEN_DEF_WR_BLCK + c * MIOPEN_DEF_HW * MIOPEN_DEF_VEC_SIZE +
        n * MIOPEN_DEF_CHW_OUT
#else
        p_blck * MIOPEN_DEF_RD_BLCK + c * MIOPEN_DEF_HW * MIOPEN_DEF_VEC_SIZE + n * MIOPEN_DEF_CHW
#endif
        ;

#endif // end of #if MIOPEN_DEF_TRANS

    return out_off;
}

static inline __device__ void LoadData(const unsigned int in_off,
#if !(MIOPEN_DEF_FORWARD && MIOPEN_DEF_TRANS)
                                       UNUSED
#endif
                                       const unsigned int n,
#if !(MIOPEN_DEF_FORWARD && !MIOPEN_DEF_TRANS)
                                       UNUSED
#endif
                                       const unsigned int c,
                                       const MIOPEN_DEF_DATA_TYPE* __restrict__ in,
                                       MIOPEN_DEF_DATA_TYPE* in_buf)
{
#if MIOPEN_DEF_FORWARD
#pragma unroll
    for(int v = 0; v < MIOPEN_DEF_VEC_SIZE; v++)
    {
        auto dst = reinterpret_cast<MIOPEN_DEF_READ_TYPE*>(in_buf + MIOPEN_DEF_RD_BLCK * v);
#if MIOPEN_DEF_TRANS
        *dst =
            ((n * MIOPEN_DEF_VEC_SIZE + v) < MIOPEN_DEF_N)
                ? *reinterpret_cast<const MIOPEN_DEF_READ_TYPE*>(in + in_off + MIOPEN_DEF_CHW * v)
                : (MIOPEN_DEF_READ_TYPE)0;
#else
        *dst = ((c * MIOPEN_DEF_VEC_SIZE + v) < MIOPEN_DEF_C)
                   ? *reinterpret_cast<const MIOPEN_DEF_READ_TYPE*>(in + in_off + MIOPEN_DEF_HW * v)
                   : (MIOPEN_DEF_READ_TYPE)0;
#endif
    }
#else
    *reinterpret_cast<MIOPEN_DEF_WRITE_TYPE*>(in_buf) =
        *reinterpret_cast<const MIOPEN_DEF_WRITE_TYPE*>(in + in_off);
#endif
}

static inline __device__ void LocalTrans(MIOPEN_DEF_DATA_TYPE* in_buf,
                                         MIOPEN_DEF_DATA_TYPE* out_buf)
{
    for(int i = 0; i < MIOPEN_DEF_RD_BLCK; i++)
    {
#pragma unroll
        for(int v = 0; v < MIOPEN_DEF_VEC_SIZE; v++)
        {
#if MIOPEN_DEF_FORWARD
            out_buf[i * MIOPEN_DEF_VEC_SIZE + v] = in_buf[v * MIOPEN_DEF_RD_BLCK + i];
#else
            out_buf[v * MIOPEN_DEF_RD_BLCK + i] = in_buf[i * MIOPEN_DEF_VEC_SIZE + v];
#endif
        }
    }
}

static inline __device__ void WriteData(const unsigned int out_off,
#if !(!MIOPEN_DEF_FORWARD && MIOPEN_DEF_TRANS)
                                        UNUSED
#endif
                                        const unsigned int n,
#if !(!MIOPEN_DEF_FORWARD && !MIOPEN_DEF_TRANS)
                                        UNUSED
#endif
                                        const unsigned int c,
                                        MIOPEN_DEF_DATA_TYPE* out,
                                        const MIOPEN_DEF_DATA_TYPE* out_buf)
{
#if MIOPEN_DEF_FORWARD
    *reinterpret_cast<MIOPEN_DEF_WRITE_TYPE*>(out + out_off) =
        *reinterpret_cast<const MIOPEN_DEF_WRITE_TYPE*>(out_buf);
#else
#pragma unroll
    for(int v = 0; v < MIOPEN_DEF_VEC_SIZE; v++)
    {
        auto src = reinterpret_cast<const MIOPEN_DEF_READ_TYPE*>(out_buf + MIOPEN_DEF_RD_BLCK * v);
#if MIOPEN_DEF_TRANS
        if((n * MIOPEN_DEF_VEC_SIZE + v) < MIOPEN_DEF_N)
            *reinterpret_cast<MIOPEN_DEF_READ_TYPE*>(out + out_off + MIOPEN_DEF_CHW * v) = *src;
#else
        if((c * MIOPEN_DEF_VEC_SIZE + v) < MIOPEN_DEF_C)
            *reinterpret_cast<MIOPEN_DEF_READ_TYPE*>(out + out_off + MIOPEN_DEF_HW * v) = *src;
#endif
    }
#endif
}

static inline __device__ void GlobalTrans(const unsigned int in_off,
                                          const unsigned int out_off,
                                          const unsigned int p_blck,
#if !MIOPEN_DEF_TRANS
                                          UNUSED
#endif
                                          const unsigned int n,
#if MIOPEN_DEF_TRANS
                                          UNUSED
#endif
                                          const unsigned int c,
                                          const MIOPEN_DEF_DATA_TYPE* __restrict__ in,
                                          MIOPEN_DEF_DATA_TYPE* __restrict__ out)
{
    int HW_tail = MIOPEN_DEF_HW - p_blck * MIOPEN_DEF_RD_BLCK;

    for(int i = 0; i < HW_tail; i++)
    {
#pragma unroll
        for(int v = 0; v < MIOPEN_DEF_VEC_SIZE; v++)
        {
#if MIOPEN_DEF_FORWARD

#if MIOPEN_DEF_TRANS
            out[out_off + i * MIOPEN_DEF_VEC_SIZE + v] =
                ((n * MIOPEN_DEF_VEC_SIZE + v) < MIOPEN_DEF_N) ? in[in_off + MIOPEN_DEF_CHW * v + i]
                                                               : (MIOPEN_DEF_DATA_TYPE)0;
#else
            out[out_off + i * MIOPEN_DEF_VEC_SIZE + v] =
                ((c * MIOPEN_DEF_VEC_SIZE + v) < MIOPEN_DEF_C) ? in[in_off + MIOPEN_DEF_HW * v + i]
                                                               : (MIOPEN_DEF_DATA_TYPE)0;
#endif

#else

#if MIOPEN_DEF_TRANS
            if((n * MIOPEN_DEF_VEC_SIZE + v) < MIOPEN_DEF_N)
                out[out_off + MIOPEN_DEF_CHW * v + i] = in[in_off + i * MIOPEN_DEF_VEC_SIZE + v];
#else
            if((c * MIOPEN_DEF_VEC_SIZE + v) < MIOPEN_DEF_C)
                out[out_off + MIOPEN_DEF_HW * v + i] = in[in_off + i * MIOPEN_DEF_VEC_SIZE + v];
#endif

#endif
        }
    }
}

extern "C" __global__ void TransposeNCHW2Vec(const MIOPEN_DEF_DATA_TYPE* __restrict__ in,
                                             MIOPEN_DEF_DATA_TYPE* __restrict__ out,
#if !MIOPEN_DEF_USE_ALPHA
                                             UNUSED
#endif
                                             const float alpha,
#if !MIOPEN_DEF_USE_BETA
                                             UNUSED
#endif
                                             const float beta)
{
    const unsigned int c_p_blck = blockIdx.x * blockDim.x + threadIdx.x;
    if(c_p_blck >= MIOPEN_DEF_MAP_RD)
        return;

    const unsigned int c      = c_p_blck / MIOPEN_DEF_HW_RD;
    const unsigned int p_blck = c_p_blck - c * MIOPEN_DEF_HW_RD;

    MIOPEN_DEF_DATA_TYPE in_buf[MIOPEN_DEF_RD_BLCK * MIOPEN_DEF_VEC_SIZE];
    MIOPEN_DEF_DATA_TYPE out_buf[MIOPEN_DEF_RD_BLCK * MIOPEN_DEF_VEC_SIZE];

#if MIOPEN_DEF_IS_2D_WG
    const unsigned int n = blockIdx.y * blockDim.y + threadIdx.y;
#else
    for(unsigned int n = 0; n < MIOPEN_DEF_GD_1; n++)
#endif
    {
        unsigned int in_off  = GetInOff(p_blck, n, c);
        unsigned int out_off = GetOutOff(p_blck, n, c);

#if MIOPEN_DEF_IS_HW_ODD
        if(p_blck < MIOPEN_DEF_HW_RD - 1)
#endif
        {
            LoadData(in_off, n, c, in, in_buf);
            LocalTrans(in_buf, out_buf);
            WriteData(out_off, n, c, out, out_buf);
        }
#if MIOPEN_DEF_IS_HW_ODD
        else
        {
            GlobalTrans(in_off, out_off, p_blck, n, c, in, out);
        }
#endif

        // TODO: support y=alpha*x+beta*y
    }

#if MIOPEN_DEF_USE_ALPHA
    (void)alpha;
#endif
#if MIOPEN_DEF_USE_BETA
    (void)beta;
#endif
}
