/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#endif

template <typename T>
__device__ T miopenAdd(T a, T b)
{
    return a + b;
}

template <typename T>
__device__ T miopenMul(T a, T b)
{
    return a * b;
}

template <typename T>
__device__ T miopenMax(T a, T b)
{
    return ((a > b) ? a : b);
}

template <typename T>
__device__ T miopenMin(T a, T b)
{
    return ((a < b) ? a : b);
}

#ifdef USE_1D_TENSOR_GENERIC
// N
extern "C" __global__ void Op1dTensorGeneric(const MIOPEN_TYPE* a,
                                             const MIOPEN_TYPE* b,
                                             MIOPEN_TYPE* c,
                                             const uint64_t Aoffset,
                                             const uint64_t Boffset,
                                             const uint64_t Coffset,
                                             const DIM_TYPE a_nstride,
                                             const DIM_TYPE b_nstride,
                                             const DIM_TYPE c_nstride,
                                             const MIOPEN_TYPE alpha0,
                                             const MIOPEN_TYPE alpha1,
                                             const MIOPEN_TYPE beta,
                                             const DIM_TYPE total_work,
                                             const bool use_beta)
{
    const MIOPEN_TYPE* a_off = a + Aoffset;
    const MIOPEN_TYPE* b_off = b + Boffset;
    MIOPEN_TYPE* c_off       = c + Coffset;

    const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
    auto a_ptr     = a_off + gid * a_nstride;
    auto b_ptr     = b_off + gid * b_nstride;
    auto c_ptr     = c_off + gid * c_nstride;

    const auto step   = gridDim.x * blockDim.x;
    const auto a_step = step * a_nstride;
    const auto b_step = step * b_nstride;
    const auto c_step = step * c_nstride;

    const auto c_end = c_off + total_work * c_nstride;
    while(c_ptr < c_end)
    {
        const auto res = MIOPEN_TENSOR_OP(a_ptr[0] * alpha0, b_ptr[0] * alpha1);
        c_ptr[0]       = use_beta ? c_ptr[0] * beta + res : res;

        a_ptr += a_step;
        b_ptr += b_step;
        c_ptr += c_step;
    }
}

#endif

#ifdef USE_2D_TENSOR_GENERIC
// NC
extern "C" __global__ void Op2dTensorGeneric(const MIOPEN_TYPE* a,
                                             const MIOPEN_TYPE* b,
                                             MIOPEN_TYPE* c,
                                             const uint64_t Aoffset,
                                             const uint64_t Boffset,
                                             const uint64_t Coffset,
                                             const DIM_TYPE b_c,
                                             const DIM_TYPE c_c,
                                             const DIM_TYPE a_nstride,
                                             const DIM_TYPE a_cstride,
                                             const DIM_TYPE b_nstride,
                                             const DIM_TYPE b_cstride,
                                             const DIM_TYPE c_nstride,
                                             const DIM_TYPE c_cstride,
                                             const MIOPEN_TYPE alpha0,
                                             const MIOPEN_TYPE alpha1,
                                             const MIOPEN_TYPE beta,
                                             const DIM_TYPE total_work,
                                             const bool use_beta)
{
    const MIOPEN_TYPE* a_off = a + Aoffset;
    const MIOPEN_TYPE* b_off = b + Boffset;
    MIOPEN_TYPE* c_off       = c + Coffset;

    auto gid          = blockIdx.x * blockDim.x + threadIdx.x;
    const auto* a_ptr = a_off + (gid / c_c) * a_nstride + (gid % c_c) * a_cstride;
    auto* c_ptr       = c_off + (gid / c_c) * c_nstride + (gid % c_c) * c_cstride;

    const auto step   = gridDim.x * blockDim.x;
    const auto a_step = (step / c_c) * a_nstride + (step % c_c) * a_cstride;
    const auto c_step = (step / c_c) * c_nstride + (step % c_c) * c_cstride;

    const auto c_end = c_off + total_work * c_nstride;
    while(c_ptr < c_end)
    {
        const auto* b_ptr = b_off;
        if(b_nstride != 0)
            b_ptr += (gid / b_c) * b_nstride;

        if(b_cstride != 0)
            b_ptr += (gid % b_c) * b_cstride;

        auto b_val = *b_ptr;
        auto a_val = *a_ptr;
        auto c_val = use_beta ? *c_ptr : static_cast<MIOPEN_TYPE>(0);
        *c_ptr     = MIOPEN_TENSOR_OP(b_val * alpha1, a_val * alpha0) + c_val * beta;

        a_ptr += a_step;
        c_ptr += c_step;
        gid += step;
    }
}

#endif

#ifdef USE_3D_TENSOR_GENERIC
// NCH
extern "C" __global__ void Op3dTensorGeneric(const MIOPEN_TYPE* a,
                                             const MIOPEN_TYPE* b,
                                             MIOPEN_TYPE* c,
                                             const uint64_t Aoffset,
                                             const uint64_t Boffset,
                                             const uint64_t Coffset,
                                             const uint32_t b_c,
                                             const uint32_t b_h,
                                             const uint32_t c_c,
                                             const uint32_t c_h,
                                             const uint32_t a_nstride,
                                             const uint32_t a_cstride,
                                             const uint32_t a_hstride,
                                             const uint32_t b_nstride,
                                             const uint32_t b_cstride,
                                             const uint32_t b_hstride,
                                             const uint32_t c_nstride,
                                             const uint32_t c_cstride,
                                             const uint32_t c_hstride,
                                             const MIOPEN_TYPE alpha0,
                                             const MIOPEN_TYPE alpha1,
                                             const MIOPEN_TYPE beta,
                                             const uint32_t total_work,
                                             const bool use_beta)
{
    const MIOPEN_TYPE* a_off = a + Aoffset;
    const MIOPEN_TYPE* b_off = b + Boffset;
    MIOPEN_TYPE* c_off       = c + Coffset;

    auto gid          = blockIdx.x * blockDim.x + threadIdx.x;
    const auto* a_ptr = a_off + (gid / (c_c * c_h)) * a_nstride +
                        ((gid % (c_c * c_h)) / c_h) * a_cstride +
                        ((gid % (c_c * c_h)) % c_h) * a_hstride;
    auto* c_ptr = c_off + (gid / (c_c * c_h)) * c_nstride +
                  ((gid % (c_c * c_h)) / c_h) * c_cstride + ((gid % (c_c * c_h)) % c_h) * c_hstride;

    const auto step   = gridDim.x * blockDim.x;
    const auto a_step = (step / (c_c * c_h)) * a_nstride +
                        ((step % (c_c * c_h)) / c_h) * a_cstride +
                        ((step % (c_c * c_h)) % c_h) * a_hstride;

    const auto c_step = (step / (c_c * c_h)) * c_nstride +
                        ((step % (c_c * c_h)) / c_h) * c_cstride +
                        ((step % (c_c * c_h)) % c_h) * c_hstride;

    const auto c_end = c_off + total_work * c_nstride;
    while(c_ptr < c_end)
    {
        const auto* b_ptr = b_off;
        if(b_nstride != 0)
            b_ptr += (gid / (b_c * b_h)) * b_nstride;

        if(b_cstride != 0)
            b_ptr += ((gid % (b_c * b_h)) / b_h) * b_cstride;

        if(b_hstride != 0)
            b_ptr += ((gid % (b_c * b_h)) % b_h) * b_hstride;

        auto b_val = *b_ptr;
        auto a_val = *a_ptr;
        auto c_val = use_beta ? *c_ptr : static_cast<MIOPEN_TYPE>(0);
        *c_ptr     = MIOPEN_TENSOR_OP(b_val * alpha1, a_val * alpha0) + c_val * beta;

        a_ptr += a_step;
        c_ptr += c_step;
        gid += step;
    }
}

#endif

#ifdef USE_4D_TENSOR_GENERIC
// NCHW
extern "C" __global__ void Op4dTensorGeneric(MIOPEN_TYPE* a,
                                             const int a_nstride,
                                             const int a_cstride,
                                             const int a_hstride,
                                             MIOPEN_TYPE* b,
                                             const int b_c,
                                             const int b_h,
                                             const int b_w,
                                             const int b_nstride,
                                             const int b_cstride,
                                             const int b_hstride,
                                             MIOPEN_TYPE* c,
                                             const int c_c,
                                             const int c_h,
                                             const int c_w,
                                             const int c_nstride,
                                             const int c_cstride,
                                             const int c_hstride,
                                             const MIOPEN_TYPE alpha0,
                                             const MIOPEN_TYPE alpha1,
                                             const MIOPEN_TYPE beta,
                                             const unsigned int bitmap,
                                             const int work_per_wg,
                                             const long Aoffset,
                                             const long Boffset,
                                             const long Coffset,
                                             const int num_wg)
{
    int gid = blockIdx.x;

    MIOPEN_TYPE* a_off = a + Aoffset;
    MIOPEN_TYPE* b_off = b + Boffset;
    MIOPEN_TYPE* c_off = c + Coffset;

// MIOPEN_TYPE operand = b[gid + Boffset];
// num_wg: the number of workgroups should be launched
// MAX_NUM_WG: the maximum number of workgroups actually launched
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
    if(beta == static_cast<MIOPEN_TYPE>(0))
#pragma clang diagnostic pop
    {
        for(; gid < num_wg; gid += MAX_NUM_WG)
        {
            int lid = threadIdx.x;

            int o_h_div = (bitmap & (1 << 0)) ? 1 : c_w;
            int o_c_div = o_h_div * ((bitmap & (1 << 1)) ? 1 : c_h);
            int o_n_div = o_c_div * ((bitmap & (1 << 2)) ? 1 : c_c);

            int o_w_gid_off = gid % b_w;
            int o_h_gid_off = (gid / b_w) % b_h;
            int o_c_gid_off = (gid / b_w / b_h) % b_c;
            int o_n_gid_off = (gid / b_w / b_h) / b_c;

            int bindex = o_n_gid_off * b_nstride + o_c_gid_off * b_cstride +
                         o_h_gid_off * b_hstride + o_w_gid_off;
            MIOPEN_TYPE operand = b_off[bindex] * alpha1;

            while(lid < work_per_wg)
            {
                int o_w = (bitmap & (1 << 0)) ? o_w_gid_off : lid % c_w;
                int o_h = (bitmap & (1 << 1)) ? o_h_gid_off : (lid / o_h_div) % c_h;
                int o_c = (bitmap & (1 << 2)) ? o_c_gid_off : (lid / o_c_div) % c_c;
                int o_n = (bitmap & (1 << 3)) ? o_n_gid_off : lid / o_n_div;

                int aindex    = o_n * a_nstride + o_c * a_cstride + o_h * a_hstride + o_w;
                int cindex    = o_n * c_nstride + o_c * c_cstride + o_h * c_hstride + o_w;
                c_off[cindex] = MIOPEN_TENSOR_OP(a_off[aindex] * alpha0, operand);

                lid += blockDim.x;
            }
        }
    }
    else
    {
        for(; gid < num_wg; gid += MAX_NUM_WG)
        {
            int lid = threadIdx.x;

            int o_h_div = (bitmap & (1 << 0)) ? 1 : c_w;
            int o_c_div = o_h_div * ((bitmap & (1 << 1)) ? 1 : c_h);
            int o_n_div = o_c_div * ((bitmap & (1 << 2)) ? 1 : c_c);

            int o_w_gid_off = gid % b_w;
            int o_h_gid_off = (gid / b_w) % b_h;
            int o_c_gid_off = (gid / b_w / b_h) % b_c;
            int o_n_gid_off = (gid / b_w / b_h) / b_c;

            int bindex = o_n_gid_off * b_nstride + o_c_gid_off * b_cstride +
                         o_h_gid_off * b_hstride + o_w_gid_off;
            MIOPEN_TYPE operand = b_off[bindex] * alpha1;

            while(lid < work_per_wg)
            {
                int o_w = (bitmap & (1 << 0)) ? o_w_gid_off : lid % c_w;
                int o_h = (bitmap & (1 << 1)) ? o_h_gid_off : (lid / o_h_div) % c_h;
                int o_c = (bitmap & (1 << 2)) ? o_c_gid_off : (lid / o_c_div) % c_c;
                int o_n = (bitmap & (1 << 3)) ? o_n_gid_off : lid / o_n_div;

                int aindex = o_n * a_nstride + o_c * a_cstride + o_h * a_hstride + o_w;
                int cindex = o_n * c_nstride + o_c * c_cstride + o_h * c_hstride + o_w;
                c_off[cindex] =
                    MIOPEN_TENSOR_OP(a_off[aindex] * alpha0, operand) + beta * c_off[cindex];

                lid += blockDim.x;
            }
        }
    }
}

#endif
