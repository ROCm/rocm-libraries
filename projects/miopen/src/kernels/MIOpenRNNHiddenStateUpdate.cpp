// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "miopen_type_traits.hpp"

#include "activation_functions.hpp"

using T_VEC2 = std::conditional<std::is_same<DATA_TYPE, half>::value, ushort2, float2>::type;
using T_VEC4 = std::conditional<std::is_same<DATA_TYPE, half>::value, ushort4, float4>::type;
using T_VEC =
    std::conditional<READ_BLOCK == 4,
                     T_VEC4,
                     typename std::conditional<READ_BLOCK == 2, T_VEC2, DATA_TYPE>::type>::type;

template <typename T>
__forceinline__ __device__ void tvec_to_accumvec(FLOAT_ACCUM data[READ_BLOCK])
{
    if constexpr(!std::is_same<T, FLOAT_ACCUM>::value)
    {
        for(int i = READ_BLOCK - 1; i >= 0; --i)
        {
            data[i] = CVT_FLOAT2ACCUM(reinterpret_cast<T*>(data)[i]);
        }
    }
}

template <typename T>
__forceinline__ __device__ void accumvec_to_tvec(FLOAT_ACCUM data[READ_BLOCK])
{
    if constexpr(!std::is_same<T, FLOAT_ACCUM>::value)
    {
        for(int i = 0; i < READ_BLOCK; ++i)
        {
            reinterpret_cast<T*>(data)[i] = CVT_ACCUM2FLOAT(data[i]);
        }
    }
}

template <typename T>
__forceinline__ __device__ void lstmfwdhiddenupdate(const T* __restrict__ cx,
                                                    T* __restrict__ reservespace,
                                                    const int hy_h,
                                                    const int hy_stride,
                                                    const long cx_offset,
                                                    const long i_offset,
                                                    const long f_offset,
                                                    const long o_offset,
                                                    const long c_offset,
                                                    const long cell_offset,
                                                    const long cell_offset_pre,
                                                    const long activ_cell_offset,
                                                    const long hidden_offset,
                                                    const bool use_cx,
                                                    const bool is_seq_begin,
                                                    const int direction,
                                                    const int cur_batch,
                                                    const int use_batch)
{
    const int total_items         = max(cur_batch * hy_h / READ_BLOCK, 1);
    const FLOAT_ACCUM activ_param = 1;

    FLOAT_ACCUM s_dat[READ_BLOCK];

    FLOAT_ACCUM i_dat[READ_BLOCK];
    FLOAT_ACCUM f_dat[READ_BLOCK];
    FLOAT_ACCUM o_dat[READ_BLOCK];
    FLOAT_ACCUM c_dat[READ_BLOCK];

    FLOAT_ACCUM cx_dat[READ_BLOCK];

    for(int gid = blockIdx.x * LOCAL_SIZE + threadIdx.x; gid < total_items; gid += LOCAL_SIZE)
    {
        int b_idx   = (gid * READ_BLOCK) / hy_h;
        int h_idx   = (gid * READ_BLOCK) % hy_h;
        int rsv_idx = b_idx * hy_stride + h_idx;

        *reinterpret_cast<T_VEC*>(s_dat) =
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + i_offset]);
        tvec_to_accumvec<T>(s_dat);
        ActivationFunction_Sigmoid(i_dat, s_dat, activ_param, activ_param, activ_param);

        *reinterpret_cast<T_VEC*>(s_dat) =
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + f_offset]);
        tvec_to_accumvec<T>(s_dat);
        ActivationFunction_Sigmoid(f_dat, s_dat, activ_param, activ_param, activ_param);

        *reinterpret_cast<T_VEC*>(s_dat) =
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + o_offset]);
        tvec_to_accumvec<T>(s_dat);
        ActivationFunction_Sigmoid(o_dat, s_dat, activ_param, activ_param, activ_param);

        *reinterpret_cast<T_VEC*>(s_dat) =
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + c_offset]);
        tvec_to_accumvec<T>(s_dat);
        ActivationFunction_TanH(c_dat, s_dat, activ_param, activ_param, activ_param);

        if(is_seq_begin)
        {
            if(use_cx)
            {
                *reinterpret_cast<T_VEC*>(cx_dat) =
                    *reinterpret_cast<const T_VEC*>(&cx[gid * READ_BLOCK + cx_offset]);
                tvec_to_accumvec<T>(cx_dat);
            }
            else
            {
                for(FLOAT_ACCUM& value : cx_dat)
                {
                    value = FLOAT_ACCUM{0};
                }
            }
        }
        else if(b_idx < use_batch)
        {
            *reinterpret_cast<T_VEC*>(cx_dat) =
                *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + cell_offset_pre]);
            tvec_to_accumvec<T>(cx_dat);
        }
        else if(direction == 1 && use_cx)
        {
            *reinterpret_cast<T_VEC*>(cx_dat) =
                *reinterpret_cast<const T_VEC*>(&cx[gid * READ_BLOCK + cx_offset]);
            tvec_to_accumvec<T>(cx_dat);
        }
        else
        {
            for(FLOAT_ACCUM& value : cx_dat)
            {
                value = FLOAT_ACCUM{0};
            }
        }

        for(int i = 0; i < READ_BLOCK; ++i)
        {
            s_dat[i] = i_dat[i] * c_dat[i] + f_dat[i] * cx_dat[i];
        }
        ActivationFunction_TanH(cx_dat, s_dat, activ_param, activ_param, activ_param);

        if constexpr(!INFERENCE_MODE)
        {
            accumvec_to_tvec<T>(i_dat);
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + i_offset]) =
                *reinterpret_cast<T_VEC*>(i_dat);
            accumvec_to_tvec<T>(f_dat);
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + f_offset]) =
                *reinterpret_cast<T_VEC*>(f_dat);
            accumvec_to_tvec<T>(o_dat);
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + o_offset]) =
                *reinterpret_cast<T_VEC*>(o_dat);
            accumvec_to_tvec<T>(c_dat);
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + c_offset]) =
                *reinterpret_cast<T_VEC*>(c_dat);
        }

        accumvec_to_tvec<T>(s_dat);
        *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + cell_offset]) =
            *reinterpret_cast<T_VEC*>(s_dat); // Ct

        if constexpr(!INFERENCE_MODE)
        {
            accumvec_to_tvec<T>(cx_dat);
            *reinterpret_cast<T_VEC*>(
                &reservespace[b_idx * hy_stride / 6 + h_idx + activ_cell_offset]) =
                *reinterpret_cast<T_VEC*>(cx_dat);
        }

        for(int i = 0; i < READ_BLOCK; ++i)
        {
            s_dat[i] = o_dat[i] * cx_dat[i];
        }

        accumvec_to_tvec<T>(s_dat);
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
                                                    const long cx_offset,
                                                    const long dcy_offset,
                                                    const long i_offset,
                                                    const long f_offset,
                                                    const long o_offset,
                                                    const long c_offset,
                                                    const long activ_cell_offset,
                                                    const long cell_offset_pre,
                                                    const long di_offset,
                                                    const long df_offset,
                                                    const long do_offset,
                                                    const long dc_offset,
                                                    const long dcell_offset,
                                                    const long dcell_offset_pre,
                                                    const long dhidden_offset,
                                                    const long f_offset_pre,
                                                    const bool use_cx,
                                                    const bool use_dcy,
                                                    const bool is_seq_begin,
                                                    const bool is_seq_end,
                                                    const int direction,
                                                    const int cur_batch,
                                                    const int use_batch,
                                                    const int use_batch2)
{
    const int total_items         = max(cur_batch * hy_h / READ_BLOCK, 1);
    const FLOAT_ACCUM activ_param = 1;

    FLOAT_ACCUM dh_dat[READ_BLOCK];

    FLOAT_ACCUM s_dat[READ_BLOCK];

    FLOAT_ACCUM i_dat[READ_BLOCK];
    FLOAT_ACCUM f_dat[READ_BLOCK];
    FLOAT_ACCUM o_dat[READ_BLOCK];
    FLOAT_ACCUM c_dat[READ_BLOCK];

    FLOAT_ACCUM di_dat[READ_BLOCK];
    FLOAT_ACCUM df_dat[READ_BLOCK];
    FLOAT_ACCUM do_dat[READ_BLOCK];
    FLOAT_ACCUM dc_dat[READ_BLOCK];

    FLOAT_ACCUM cx_dat[READ_BLOCK];
    FLOAT_ACCUM dcx_dat[READ_BLOCK];

    for(int gid = blockIdx.x * LOCAL_SIZE + threadIdx.x; gid < total_items; gid += LOCAL_SIZE)
    {
        int b_idx   = (gid * READ_BLOCK) / hy_h;
        int h_idx   = (gid * READ_BLOCK) % hy_h;
        int rsv_idx = b_idx * hy_stride + h_idx;

        *reinterpret_cast<T_VEC*>(dh_dat) =
            *reinterpret_cast<T_VEC*>(&workspace[rsv_idx + dhidden_offset]);
        tvec_to_accumvec<T>(dh_dat);
        *reinterpret_cast<T_VEC*>(o_dat) =
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + o_offset]);
        tvec_to_accumvec<T>(o_dat);
        *reinterpret_cast<T_VEC*>(i_dat) =
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + i_offset]);
        tvec_to_accumvec<T>(i_dat);
        *reinterpret_cast<T_VEC*>(c_dat) =
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + c_offset]);
        tvec_to_accumvec<T>(c_dat);

        for(int i = 0; i < READ_BLOCK; ++i)
        {
            s_dat[i] = dh_dat[i] * o_dat[i];
        }

        *reinterpret_cast<T_VEC*>(cx_dat) = *reinterpret_cast<T_VEC*>(
            &reservespace[b_idx * hy_stride / 6 + h_idx + activ_cell_offset]);
        tvec_to_accumvec<T>(cx_dat);

        ActivationFunction_TanH_Diff(
            dcx_dat, s_dat, cx_dat, cx_dat, activ_param, activ_param, activ_param, activ_param);

        accumvec_to_tvec<T>(dh_dat);
        *reinterpret_cast<T_VEC*>(&workspace[rsv_idx + dcell_offset]) =
            *reinterpret_cast<T_VEC*>(dh_dat);
        accumvec_to_tvec<T>(o_dat);
        *reinterpret_cast<T_VEC*>(&workspace[rsv_idx + dhidden_offset]) =
            *reinterpret_cast<T_VEC*>(o_dat);

        if(is_seq_end)
        {
            if(use_dcy)
            {
                *reinterpret_cast<T_VEC*>(s_dat) =
                    *reinterpret_cast<const T_VEC*>(&dcy[gid * READ_BLOCK + dcy_offset]);
                tvec_to_accumvec<T>(s_dat);

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
            tvec_to_accumvec<T>(s_dat);
            *reinterpret_cast<T_VEC*>(f_dat) =
                *reinterpret_cast<const T_VEC*>(&reservespace[rsv_idx + f_offset_pre]);
            tvec_to_accumvec<T>(f_dat);

            for(int i = 0; i < READ_BLOCK; ++i)
            {
                dcx_dat[i] += s_dat[i] * f_dat[i];
            }
        }
        else if(direction == 0 && use_dcy)
        {
            *reinterpret_cast<T_VEC*>(s_dat) =
                *reinterpret_cast<const T_VEC*>(&dcy[gid * READ_BLOCK + dcy_offset]);
            tvec_to_accumvec<T>(s_dat);

            for(int i = 0; i < READ_BLOCK; ++i)
            {
                dcx_dat[i] += s_dat[i];
            }
        }

        if(is_seq_begin)
        {
            if(use_cx)
            {
                *reinterpret_cast<T_VEC*>(df_dat) =
                    *reinterpret_cast<const T_VEC*>(&cx[gid * READ_BLOCK + cx_offset]);
                tvec_to_accumvec<T>(df_dat);

                for(int i = 0; i < READ_BLOCK; ++i)
                {
                    df_dat[i] *= dcx_dat[i];
                }
            }
            else
            {
                for(FLOAT_ACCUM& value : df_dat)
                {
                    value = FLOAT_ACCUM{0};
                }
            }
        }
        else if(b_idx < use_batch2)
        {
            *reinterpret_cast<T_VEC*>(df_dat) =
                *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + cell_offset_pre]);
            tvec_to_accumvec<T>(df_dat);

            for(int i = 0; i < READ_BLOCK; ++i)
            {
                df_dat[i] *= dcx_dat[i];
            }
        }
        else if(direction == 1 && use_cx)
        {
            *reinterpret_cast<T_VEC*>(df_dat) =
                *reinterpret_cast<const T_VEC*>(&cx[gid * READ_BLOCK + cx_offset]);
            tvec_to_accumvec<T>(df_dat);

            for(int i = 0; i < READ_BLOCK; ++i)
            {
                df_dat[i] *= dcx_dat[i];
            }
        }
        else
        {
            for(FLOAT_ACCUM& value : df_dat)
            {
                value = FLOAT_ACCUM{0};
            }
        }

        *reinterpret_cast<T_VEC*>(f_dat) =
            *reinterpret_cast<T_VEC*>(&reservespace[rsv_idx + f_offset]);
        tvec_to_accumvec<T>(f_dat);
        ActivationFunction_Sigmoid_Diff(
            s_dat, df_dat, f_dat, f_dat, activ_param, activ_param, activ_param, activ_param);
        accumvec_to_tvec<T>(s_dat);
        *reinterpret_cast<T_VEC*>(&workspace[rsv_idx + df_offset]) =
            *reinterpret_cast<T_VEC*>(s_dat);

        for(int i = 0; i < READ_BLOCK; ++i)
        {
            di_dat[i] = c_dat[i] * dcx_dat[i];
        }
        ActivationFunction_Sigmoid_Diff(
            s_dat, di_dat, i_dat, i_dat, activ_param, activ_param, activ_param, activ_param);
        accumvec_to_tvec<T>(s_dat);
        *reinterpret_cast<T_VEC*>(&workspace[rsv_idx + di_offset]) =
            *reinterpret_cast<T_VEC*>(s_dat);

        for(int i = 0; i < READ_BLOCK; ++i)
        {
            do_dat[i] = cx_dat[i] * dh_dat[i];
        }
        ActivationFunction_Sigmoid_Diff(
            s_dat, do_dat, o_dat, o_dat, activ_param, activ_param, activ_param, activ_param);
        accumvec_to_tvec<T>(s_dat);
        *reinterpret_cast<T_VEC*>(&workspace[rsv_idx + do_offset]) =
            *reinterpret_cast<T_VEC*>(s_dat);

        for(int i = 0; i < READ_BLOCK; ++i)
        {
            dc_dat[i] = i_dat[i] * dcx_dat[i];
        }
        ActivationFunction_TanH_Diff(
            s_dat, dc_dat, c_dat, c_dat, activ_param, activ_param, activ_param, activ_param);
        accumvec_to_tvec<T>(s_dat);
        *reinterpret_cast<T_VEC*>(&workspace[rsv_idx + dc_offset]) =
            *reinterpret_cast<T_VEC*>(s_dat);

        accumvec_to_tvec<T>(dcx_dat);
        *reinterpret_cast<T_VEC*>(&workspace[rsv_idx + dcell_offset]) =
            *reinterpret_cast<T_VEC*>(dcx_dat);
    }
}

extern "C" __global__ void LSTMFwdHiddenUpdate(const DATA_TYPE* __restrict__ cx,
                                               DATA_TYPE* __restrict__ reservespace,
                                               const int hy_h,
                                               const int hy_stride,
                                               const long cx_offset,
                                               const long i_offset,
                                               const long f_offset,
                                               const long o_offset,
                                               const long c_offset,
                                               const long cell_offset,
                                               const long cell_offset_pre,
                                               const long activ_cell_offset,
                                               const long hidden_offset,
                                               const bool use_cx,
                                               const bool is_seq_begin,
                                               const int direction,
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
                                   use_cx,
                                   is_seq_begin,
                                   direction,
                                   cur_batch,
                                   use_batch);
}

extern "C" __global__ void LSTMBwdHiddenUpdate(const DATA_TYPE* __restrict__ cx,
                                               const DATA_TYPE* __restrict__ dcy,
                                               DATA_TYPE* __restrict__ reservespace,
                                               DATA_TYPE* __restrict__ workspace,
                                               const int hy_h,
                                               const int hy_stride,
                                               const long cx_offset,
                                               const long dcy_offset,
                                               const long i_offset,
                                               const long f_offset,
                                               const long o_offset,
                                               const long c_offset,
                                               const long activ_cell_offset,
                                               const long cell_offset_pre,
                                               const long di_offset,
                                               const long df_offset,
                                               const long do_offset,
                                               const long dc_offset,
                                               const long dcell_offset,
                                               const long dcell_offset_pre,
                                               const long dhidden_offset,
                                               const long f_offset_pre,
                                               const bool use_cx,
                                               const bool use_dcy,
                                               const bool is_seq_begin,
                                               const bool is_seq_end,
                                               const int direction,
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
                                   use_cx,
                                   use_dcy,
                                   is_seq_begin,
                                   is_seq_end,
                                   direction,
                                   cur_batch,
                                   use_batch,
                                   use_batch2);
}
