// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "miopen_type_traits.hpp"
#include "miopen_cstdint.hpp"

#include "activation_functions.hpp"

using T_VEC2 = std::conditional<std::is_same<DATA_TYPE, half>::value, ushort2, float2>::type;
using T_VEC4 = std::conditional<std::is_same<DATA_TYPE, half>::value, ushort4, float4>::type;
using T_VEC =
    std::conditional<READ_BLOCK == 4,
                     T_VEC4,
                     typename std::conditional<READ_BLOCK == 2, T_VEC2, DATA_TYPE>::type>::type;

template <typename T>
__forceinline__ __device__ void lstmfwdhiddenupdate(const T* __restrict__ cx,
                                                    T* __restrict__ reservespace,
                                                    const int hy_h,
                                                    const int hy_stride,
                                                    const int64_t cx_offset,
                                                    const int64_t i_offset,
                                                    const int64_t f_offset,
                                                    const int64_t o_offset,
                                                    const int64_t c_offset,
                                                    const int64_t cell_offset,
                                                    const int64_t cell_offset_pre,
                                                    const int64_t activ_cell_offset,
                                                    const int64_t hidden_offset,
                                                    const int cur_batch,
                                                    const int use_batch)
{
    const int total_items = max(cur_batch * hy_h / READ_BLOCK, 1);
    const T activ_param   = 1;

    T s_dat[READ_BLOCK];

    T i_dat[READ_BLOCK];
    T f_dat[READ_BLOCK];
    T o_dat[READ_BLOCK];
    T c_dat[READ_BLOCK];

    T cx_dat[READ_BLOCK];

    for(int gid = blockIdx.x * LOCAL_SIZE + threadIdx.x; gid < total_items; gid += GLOBAL_SIZE)
    {
        int b_idx   = (gid * READ_BLOCK) / hy_h;
        int h_idx   = (gid * READ_BLOCK) % hy_h;
        int rsv_idx = b_idx * hy_stride + h_idx;

        *reinterpret_cast<T_VEC*>(s_dat) =
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + i_offset]);
        ActivationFunction_Sigmoid(i_dat, s_dat, activ_param, activ_param, activ_param);

        *reinterpret_cast<T_VEC*>(s_dat) =
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + f_offset]);
        ActivationFunction_Sigmoid(f_dat, s_dat, activ_param, activ_param, activ_param);

        *reinterpret_cast<T_VEC*>(s_dat) =
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + o_offset]);
        ActivationFunction_Sigmoid(o_dat, s_dat, activ_param, activ_param, activ_param);

        *reinterpret_cast<T_VEC*>(s_dat) =
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + c_offset]);
        ActivationFunction_TanH(c_dat, s_dat, activ_param, activ_param, activ_param);

        if constexpr(IS_SEQ_BEGIN)
        {
            if constexpr(USE_CX)
            {
                *reinterpret_cast<T_VEC*>(cx_dat) =
                    *reinterpret_cast<const T_VEC*>(&cx[gid * READ_BLOCK + cx_offset]);
            }
            else
            {
#pragma unroll
                for(T& value : cx_dat)
                {
                    value = CVT_FP32_2FLOAT(0.0f);
                }
            }
        }
        else if(b_idx < use_batch)
        {
            *reinterpret_cast<T_VEC*>(cx_dat) =
                *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + cell_offset_pre]);
        }
        else if constexpr(DIRECTION == 1 && USE_CX)
        {
            *reinterpret_cast<T_VEC*>(cx_dat) =
                *reinterpret_cast<const T_VEC*>(&cx[gid * READ_BLOCK + cx_offset]);
        }
        else
        {
#pragma unroll
            for(T& value : cx_dat)
            {
                value = CVT_FP32_2FLOAT(0.0f);
            }
        }

#pragma unroll
        for(int i = 0; i < READ_BLOCK; ++i)
        {
            s_dat[i] = i_dat[i] * c_dat[i] + f_dat[i] * cx_dat[i];
        }
        ActivationFunction_TanH(cx_dat, s_dat, activ_param, activ_param, activ_param);

        if constexpr(!INFERENCE_MODE)
        {
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + i_offset]) =
                *reinterpret_cast<T_VEC*>(i_dat);
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + f_offset]) =
                *reinterpret_cast<T_VEC*>(f_dat);
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + o_offset]) =
                *reinterpret_cast<T_VEC*>(o_dat);
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + c_offset]) =
                *reinterpret_cast<T_VEC*>(c_dat);
        }

        *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + cell_offset]) =
            *reinterpret_cast<T_VEC*>(s_dat); // Ct

        if constexpr(!INFERENCE_MODE)
        {
            *reinterpret_cast<T_VEC*>(
                &reservespace[b_idx * hy_stride / 6 + h_idx + activ_cell_offset]) =
                *reinterpret_cast<T_VEC*>(cx_dat);
        }

#pragma unroll
        for(int i = 0; i < READ_BLOCK; ++i)
        {
            s_dat[i] = o_dat[i] * cx_dat[i];
        }

        *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + hidden_offset]) =
            *reinterpret_cast<T_VEC*>(s_dat); // Ht
    }
}

template <typename T>
__forceinline__ __device__ void lstmbwdhiddenupdate(const T* __restrict__ cx,
                                                    const T* __restrict__ dcy,
                                                    T* __restrict__ reservespace,
                                                    T* __restrict__ workspace,
                                                    const int hy_h,
                                                    const int hy_stride,
                                                    const int64_t cx_offset,
                                                    const int64_t dcy_offset,
                                                    const int64_t i_offset,
                                                    const int64_t f_offset,
                                                    const int64_t o_offset,
                                                    const int64_t c_offset,
                                                    const int64_t activ_cell_offset,
                                                    const int64_t cell_offset_pre,
                                                    const int64_t di_offset,
                                                    const int64_t df_offset,
                                                    const int64_t do_offset,
                                                    const int64_t dc_offset,
                                                    const int64_t dcell_offset,
                                                    const int64_t dcell_offset_pre,
                                                    const int64_t dhidden_offset,
                                                    const int64_t f_offset_pre,
                                                    const int cur_batch,
                                                    const int use_batch,
                                                    const int use_batch2)
{
    const int total_items = max(cur_batch * hy_h / READ_BLOCK, 1);
    const T activ_param   = 1;

    T dh_dat[READ_BLOCK];

    T s_dat[READ_BLOCK];

    T i_dat[READ_BLOCK];
    T f_dat[READ_BLOCK];
    T o_dat[READ_BLOCK];
    T c_dat[READ_BLOCK];

    T di_dat[READ_BLOCK];
    T df_dat[READ_BLOCK];
    T do_dat[READ_BLOCK];
    T dc_dat[READ_BLOCK];

    T cx_dat[READ_BLOCK];
    T dcx_dat[READ_BLOCK];

    for(int gid = blockIdx.x * LOCAL_SIZE + threadIdx.x; gid < total_items; gid += GLOBAL_SIZE)
    {
        int b_idx   = (gid * READ_BLOCK) / hy_h;
        int h_idx   = (gid * READ_BLOCK) % hy_h;
        int rsv_idx = b_idx * hy_stride + h_idx;

        *reinterpret_cast<T_VEC*>(dh_dat) =
            *reinterpret_cast<T_VEC*>(&workspace[rsv_idx + dhidden_offset]);
        *reinterpret_cast<T_VEC*>(o_dat) =
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + o_offset]);
        *reinterpret_cast<T_VEC*>(i_dat) =
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + i_offset]);
        *reinterpret_cast<T_VEC*>(c_dat) =
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + c_offset]);

#pragma unroll
        for(int i = 0; i < READ_BLOCK; ++i)
        {
            s_dat[i] = dh_dat[i] * o_dat[i];
        }

        *reinterpret_cast<T_VEC*>(cx_dat) = *reinterpret_cast<T_VEC*>(
            &reservespace[b_idx * hy_stride / 6 + h_idx + activ_cell_offset]);

        ActivationFunction_TanH_Diff(
            dcx_dat, s_dat, cx_dat, cx_dat, activ_param, activ_param, activ_param, activ_param);

        if constexpr(IS_SEQ_END)
        {
            if constexpr(USE_DCY)
            {
                *reinterpret_cast<T_VEC*>(s_dat) =
                    *reinterpret_cast<const T_VEC*>(&dcy[gid * READ_BLOCK + dcy_offset]);

#pragma unroll
                for(int i = 0; i < READ_BLOCK; ++i)
                {
                    dcx_dat[i] += s_dat[i];
                }
            }
        }
        else if(b_idx < use_batch)
        {
            *reinterpret_cast<T_VEC*>(s_dat) =
                *reinterpret_cast<const T_VEC*>(&workspace[rsv_idx + dcell_offset_pre]);
            *reinterpret_cast<T_VEC*>(f_dat) =
                *reinterpret_cast<const T_VEC*>(&reservespace[rsv_idx + f_offset_pre]);

#pragma unroll
            for(int i = 0; i < READ_BLOCK; ++i)
            {
                dcx_dat[i] += s_dat[i] * f_dat[i];
            }
        }
        else if constexpr(DIRECTION == 0 && USE_DCY)
        {
            *reinterpret_cast<T_VEC*>(s_dat) =
                *reinterpret_cast<const T_VEC*>(&dcy[gid * READ_BLOCK + dcy_offset]);

#pragma unroll
            for(int i = 0; i < READ_BLOCK; ++i)
            {
                dcx_dat[i] += s_dat[i];
            }
        }

        if constexpr(IS_SEQ_BEGIN)
        {
            if constexpr(USE_CX)
            {
                *reinterpret_cast<T_VEC*>(df_dat) =
                    *reinterpret_cast<const T_VEC*>(&cx[gid * READ_BLOCK + cx_offset]);

#pragma unroll
                for(int i = 0; i < READ_BLOCK; ++i)
                {
                    df_dat[i] *= dcx_dat[i];
                }
            }
            else
            {
#pragma unroll
                for(T& value : df_dat)
                {
                    value = CVT_FP32_2FLOAT(0.0f);
                }
            }
        }
        else if(b_idx < use_batch2)
        {
            *reinterpret_cast<T_VEC*>(df_dat) =
                *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + cell_offset_pre]);

#pragma unroll
            for(int i = 0; i < READ_BLOCK; ++i)
            {
                df_dat[i] *= dcx_dat[i];
            }
        }
        else if constexpr(DIRECTION == 1 && USE_CX)
        {
            *reinterpret_cast<T_VEC*>(df_dat) =
                *reinterpret_cast<const T_VEC*>(&cx[gid * READ_BLOCK + cx_offset]);

#pragma unroll
            for(int i = 0; i < READ_BLOCK; ++i)
            {
                df_dat[i] *= dcx_dat[i];
            }
        }
        else
        {
#pragma unroll
            for(T& value : df_dat)
            {
                value = CVT_FP32_2FLOAT(0.0f);
            }
        }

        *reinterpret_cast<T_VEC*>(f_dat) =
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + f_offset]);
        ActivationFunction_Sigmoid_Diff(
            s_dat, df_dat, f_dat, f_dat, activ_param, activ_param, activ_param, activ_param);
        *reinterpret_cast<T_VEC*>(&workspace[rsv_idx + df_offset]) =
            *reinterpret_cast<T_VEC*>(s_dat);

#pragma unroll
        for(int i = 0; i < READ_BLOCK; ++i)
        {
            di_dat[i] = c_dat[i] * dcx_dat[i];
        }
        ActivationFunction_Sigmoid_Diff(
            s_dat, di_dat, i_dat, i_dat, activ_param, activ_param, activ_param, activ_param);
        *reinterpret_cast<T_VEC*>(&workspace[rsv_idx + di_offset]) =
            *reinterpret_cast<T_VEC*>(s_dat);

#pragma unroll
        for(int i = 0; i < READ_BLOCK; ++i)
        {
            do_dat[i] = cx_dat[i] * dh_dat[i];
        }
        ActivationFunction_Sigmoid_Diff(
            s_dat, do_dat, o_dat, o_dat, activ_param, activ_param, activ_param, activ_param);
        *reinterpret_cast<T_VEC*>(&workspace[rsv_idx + do_offset]) =
            *reinterpret_cast<T_VEC*>(s_dat);

#pragma unroll
        for(int i = 0; i < READ_BLOCK; ++i)
        {
            dc_dat[i] = i_dat[i] * dcx_dat[i];
        }
        ActivationFunction_TanH_Diff(
            s_dat, dc_dat, c_dat, c_dat, activ_param, activ_param, activ_param, activ_param);
        *reinterpret_cast<T_VEC*>(&workspace[rsv_idx + dc_offset]) =
            *reinterpret_cast<T_VEC*>(s_dat);

        *reinterpret_cast<T_VEC*>(&workspace[rsv_idx + dcell_offset]) =
            *reinterpret_cast<T_VEC*>(dcx_dat);
    }
}

extern "C" __global__ void LSTMFwdHiddenUpdate(const DATA_TYPE* __restrict__ cx,
                                               DATA_TYPE* __restrict__ reservespace,
                                               const int hy_h,
                                               const int hy_stride,
                                               const int64_t cx_offset,
                                               const int64_t i_offset,
                                               const int64_t f_offset,
                                               const int64_t o_offset,
                                               const int64_t c_offset,
                                               const int64_t cell_offset,
                                               const int64_t cell_offset_pre,
                                               const int64_t activ_cell_offset,
                                               const int64_t hidden_offset,
                                               const int cur_batch,
                                               const int use_batch)
{
    lstmfwdhiddenupdate<DATA_TYPE>(cx,
                                   reservespace,
                                   hy_h,
                                   hy_stride,
                                   cx_offset,
                                   i_offset,
                                   f_offset,
                                   o_offset,
                                   c_offset,
                                   cell_offset,
                                   cell_offset_pre,
                                   activ_cell_offset,
                                   hidden_offset,
                                   cur_batch,
                                   use_batch);
}

extern "C" __global__ void LSTMBwdHiddenUpdate(const DATA_TYPE* __restrict__ cx,
                                               const DATA_TYPE* __restrict__ dcy,
                                               DATA_TYPE* __restrict__ reservespace,
                                               DATA_TYPE* __restrict__ workspace,
                                               const int hy_h,
                                               const int hy_stride,
                                               const int64_t cx_offset,
                                               const int64_t dcy_offset,
                                               const int64_t i_offset,
                                               const int64_t f_offset,
                                               const int64_t o_offset,
                                               const int64_t c_offset,
                                               const int64_t activ_cell_offset,
                                               const int64_t cell_offset_pre,
                                               const int64_t di_offset,
                                               const int64_t df_offset,
                                               const int64_t do_offset,
                                               const int64_t dc_offset,
                                               const int64_t dcell_offset,
                                               const int64_t dcell_offset_pre,
                                               const int64_t dhidden_offset,
                                               const int64_t f_offset_pre,
                                               const int cur_batch,
                                               const int use_batch,
                                               const int use_batch2)
{
    lstmbwdhiddenupdate<DATA_TYPE>(cx,
                                   dcy,
                                   reservespace,
                                   workspace,
                                   hy_h,
                                   hy_stride,
                                   cx_offset,
                                   dcy_offset,
                                   i_offset,
                                   f_offset,
                                   o_offset,
                                   c_offset,
                                   activ_cell_offset,
                                   cell_offset_pre,
                                   di_offset,
                                   df_offset,
                                   do_offset,
                                   dc_offset,
                                   dcell_offset,
                                   dcell_offset_pre,
                                   dhidden_offset,
                                   f_offset_pre,
                                   cur_batch,
                                   use_batch,
                                   use_batch2);
}
