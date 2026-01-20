// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

#ifndef USE_ALPHA
#define USE_ALPHA 0
#endif
#ifndef USE_BETA
#define USE_BETA 0
#endif

#define UNUSED __attribute__((unused))

static inline __device__ unsigned int
GetInOff(const unsigned int p_blck, const unsigned int n, const unsigned int c)
{
#if TRANS

    const unsigned int in_off =
#if FORWARD
        p_blck * RD_BLCK + c * HW + n * CHW * VEC_SIZE
#else
        p_blck * WR_BLCK + n * HW * VEC_SIZE + c * NHW_OUT
#endif
        ;

#else

    const unsigned int in_off =
#if FORWARD
        p_blck * RD_BLCK + c * HW * VEC_SIZE + n * CHW
#else
        p_blck * WR_BLCK + c * HW * VEC_SIZE + n * CHW_OUT
#endif
        ;

#endif // end of #if TRANS

    return in_off;
}

static inline __device__ unsigned int
GetOutOff(const unsigned int p_blck, const unsigned int n, const unsigned int c)
{
#if TRANS

    const unsigned int out_off =
#if FORWARD
        p_blck * WR_BLCK + n * HW * VEC_SIZE + c * NHW_OUT
#else
        p_blck * RD_BLCK + c * HW + n * CHW * VEC_SIZE
#endif
        ;

#else

    const unsigned int out_off =
#if FORWARD
        p_blck * WR_BLCK + c * HW * VEC_SIZE + n * CHW_OUT
#else
        p_blck * RD_BLCK + c * HW * VEC_SIZE + n * CHW
#endif
        ;

#endif // end of #if TRANS

    return out_off;
}

static inline __device__ void LoadData(const unsigned int in_off,
#if !(FORWARD && TRANS)
                                       UNUSED
#endif
                                       const unsigned int n,
#if !(FORWARD && !TRANS)
                                       UNUSED
#endif
                                       const unsigned int c,
                                       const DATA_TYPE* __restrict__ in,
                                       DATA_TYPE* in_buf)
{
#if FORWARD
#pragma unroll
    for(int v = 0; v < VEC_SIZE; v++)
    {
        auto dst = reinterpret_cast<READ_TYPE*>(in_buf + RD_BLCK * v);
#if TRANS
        *dst = ((n * VEC_SIZE + v) < N) ? *reinterpret_cast<const READ_TYPE*>(in + in_off + CHW * v)
                                        : (READ_TYPE)0;
#else
        *dst = ((c * VEC_SIZE + v) < C) ? *reinterpret_cast<const READ_TYPE*>(in + in_off + HW * v)
                                        : (READ_TYPE)0;
#endif
    }
#else
    *reinterpret_cast<WRITE_TYPE*>(in_buf) = *reinterpret_cast<const WRITE_TYPE*>(in + in_off);
#endif
}

static inline __device__ void LocalTrans(DATA_TYPE* in_buf, DATA_TYPE* out_buf)
{
    for(int i = 0; i < RD_BLCK; i++)
    {
#pragma unroll
        for(int v = 0; v < VEC_SIZE; v++)
        {
#if FORWARD
            out_buf[i * VEC_SIZE + v] = in_buf[v * RD_BLCK + i];
#else
            out_buf[v * RD_BLCK + i] = in_buf[i * VEC_SIZE + v];
#endif
        }
    }
}

static inline __device__ void WriteData(const unsigned int out_off,
#if !(!FORWARD && TRANS)
                                        UNUSED
#endif
                                        const unsigned int n,
#if !(!FORWARD && !TRANS)
                                        UNUSED
#endif
                                        const unsigned int c,
                                        DATA_TYPE* out,
                                        const DATA_TYPE* out_buf)
{
#if FORWARD
    *reinterpret_cast<WRITE_TYPE*>(out + out_off) = *reinterpret_cast<const WRITE_TYPE*>(out_buf);
#else
#pragma unroll
    for(int v = 0; v < VEC_SIZE; v++)
    {
        auto src = reinterpret_cast<const READ_TYPE*>(out_buf + RD_BLCK * v);
#if TRANS
        if((n * VEC_SIZE + v) < N)
            *reinterpret_cast<READ_TYPE*>(out + out_off + CHW * v) = *src;
#else
        if((c * VEC_SIZE + v) < C)
            *reinterpret_cast<READ_TYPE*>(out + out_off + HW * v) = *src;
#endif
    }
#endif
}

static inline __device__ void GlobalTrans(const unsigned int in_off,
                                          const unsigned int out_off,
                                          const unsigned int p_blck,
#if !TRANS
                                          UNUSED
#endif
                                          const unsigned int n,
#if TRANS
                                          UNUSED
#endif
                                          const unsigned int c,
                                          const DATA_TYPE* __restrict__ in,
                                          DATA_TYPE* __restrict__ out)
{
    int HW_tail = HW - p_blck * RD_BLCK;

    for(int i = 0; i < HW_tail; i++)
    {
#pragma unroll
        for(int v = 0; v < VEC_SIZE; v++)
        {
#if FORWARD

#if TRANS
            out[out_off + i * VEC_SIZE + v] =
                ((n * VEC_SIZE + v) < N) ? in[in_off + CHW * v + i] : (DATA_TYPE)0;
#else
            out[out_off + i * VEC_SIZE + v] =
                ((c * VEC_SIZE + v) < C) ? in[in_off + HW * v + i] : (DATA_TYPE)0;
#endif

#else

#if TRANS
            if((n * VEC_SIZE + v) < N)
                out[out_off + CHW * v + i] = in[in_off + i * VEC_SIZE + v];
#else
            if((c * VEC_SIZE + v) < C)
                out[out_off + HW * v + i] = in[in_off + i * VEC_SIZE + v];
#endif

#endif
        }
    }
}

extern "C" __global__ void TransposeNCHW2Vec(const DATA_TYPE* __restrict__ in,
                                             DATA_TYPE* __restrict__ out,
#if !USE_ALPHA
                                             UNUSED
#endif
                                             const float alpha,
#if !USE_BETA
                                             UNUSED
#endif
                                             const float beta)
{
    const unsigned int c_p_blck = blockIdx.x * blockDim.x + threadIdx.x;
    if(c_p_blck >= MAP_RD)
        return;

    const unsigned int c      = c_p_blck / HW_RD;
    const unsigned int p_blck = c_p_blck - c * HW_RD;

    DATA_TYPE in_buf[RD_BLCK * VEC_SIZE];
    DATA_TYPE out_buf[RD_BLCK * VEC_SIZE];

#if IS_2D_WG
    const unsigned int n = blockIdx.y * blockDim.y + threadIdx.y;
#else
    for(unsigned int n = 0; n < GD_1; n++)
#endif
    {
        unsigned int in_off  = GetInOff(p_blck, n, c);
        unsigned int out_off = GetOutOff(p_blck, n, c);

#if IS_HW_ODD
        if(p_blck < HW_RD - 1)
#endif
        {
            LoadData(in_off, n, c, in, in_buf);
            LocalTrans(in_buf, out_buf);
            WriteData(out_off, n, c, out, out_buf);
        }
#if IS_HW_ODD
        else
        {
            GlobalTrans(in_off, out_off, p_blck, n, c, in, out);
        }
#endif

        // TODO: support y=alpha*x+beta*y
    }

#if USE_ALPHA
    (void)alpha;
#endif
#if USE_BETA
    (void)beta;
#endif
}
