/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define TWO 2
#define FOUR 4
#define EIGHT 8

#define UNUSED __attribute__((__unused__))

#define MIOPEN_NRN_GROUP_SZ2 1

#include "vector_types.hpp"
#include <hip/hip_fp16.h>
#include "activation_functions.hpp"

#ifdef LITE

/**********************************************************************************************
**********************************************************************************************/

// N - batch size
// C - # of maps
// H - map height
// W - map width
// TENS_LEN = (N*C*H*W);
// RD_BLCK = (TENS_LEN%4==0) ? 4 : (TENS_LEN%3==0)? 3 : (TENS_LEN%2==0)? 2 : 1;
// READ_TYPE = (RD_BLCK==4) ? "float4" : (RD_BLCK == 3) ? "float3" : (RD_BLC==2) ? "float2" :
// "float";
// local size = (256, 1, 1)
// global size = ((TENS_LEN/RD_BLCK), 1, 1)

extern "C" __global__ void MIOpenActiveFwdLite(const FP_TYPE* bot,
                                               FP_TYPE* top,
                                               FP_TYPE gamma,
                                               FP_TYPE beta,
                                               FP_TYPE alpha,
                                               const long bot_offset,
                                               const long top_offset)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned int index = tid * MIOPEN_READ_UNIT;

    FP_TYPE data[MIOPEN_READ_UNIT];
    FP_TYPE response[MIOPEN_READ_UNIT];

    *((MIOPEN_READ_TYPE*)data) = *((const MIOPEN_READ_TYPE*)(bot + bot_offset + index));

    ActivationFunction(response, data, gamma, beta, alpha);

    *((MIOPEN_READ_TYPE*)(top + top_offset + index)) = *((MIOPEN_READ_TYPE*)response);
}

/**********************************************************************************************
**********************************************************************************************/

extern "C" __global__ void MIOpenActiveFwd2DLite(const FP_TYPE* bot,
                                                 FP_TYPE* top,
                                                 FP_TYPE gamma,
                                                 FP_TYPE beta,
                                                 FP_TYPE alpha,
                                                 const long bot_offset,
                                                 const long top_offset,
                                                 const uint bot_stride,
                                                 const uint top_stride)
{
    const unsigned int x_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y    = blockIdx.y * blockDim.y + threadIdx.y;

    uint bot_index = y * bot_stride + x_id * MIOPEN_READ_UNIT;
    uint top_index = y * top_stride + x_id * MIOPEN_READ_UNIT;

    FP_TYPE data[MIOPEN_READ_UNIT];
    FP_TYPE response[MIOPEN_READ_UNIT];

    *((MIOPEN_READ_TYPE*)data) = *((const MIOPEN_READ_TYPE*)(bot + bot_offset + bot_index));

    ActivationFunction(response, data, gamma, beta, alpha);

    *((MIOPEN_READ_TYPE*)(top + top_offset + top_index)) = *((MIOPEN_READ_TYPE*)response);
}

#else

/***************************************************************************************************************/
__launch_bounds__(
    MIOPEN_NRN_GROUP_SZ0* MIOPEN_NRN_GROUP_SZ1* MIOPEN_NRN_GROUP_SZ2) extern "C" __global__
    void MIOpenNeuronFwd(const FP_TYPE* bot,
                         FP_TYPE* top,
                         FP_TYPE gamma,
                         FP_TYPE beta,
                         FP_TYPE alpha,
                         const long xOffset,
                         const long yOffset)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; // channel x

#if MIOPEN_N_OUT_STRIDE > MIOPEN_OUT_BLOCK_SZ
    int n_out_stride     = MIOPEN_N_OUT_STRIDE;
    int c_out            = MIOPEN_C_OUT;
    int h_out            = MIOPEN_H_OUT;
    int w_out            = MIOPEN_W_OUT;
#endif
#if MIOPEN_N_IN_STRIDE > MIOPEN_IN_BLOCK_SZ
    int n_in_stride      = MIOPEN_N_IN_STRIDE;
    int c_in             = MIOPEN_C_IN;
    int h_in             = MIOPEN_H_IN;
    int w_in             = MIOPEN_W_IN;
#endif

    FP_TYPE data[MIOPEN_READ_UNIT];
    FP_TYPE response[MIOPEN_READ_UNIT];
#if MIOPEN_N_PIXS_OFF > 0
    if(x == MIOPEN_MAP_SZ_ALIGNED - 1)
    {
        int i = 0;
        for(; i < MIOPEN_N_PIXS_OFF; ++i)
        {
#if MIOPEN_N_IN_STRIDE > MIOPEN_IN_BLOCK_SZ
            if(n_in_stride > c_in * h_in * w_in && c_in != 0 && h_in != 0 && w_in != 0)
            {
                int loc, n_loc, c_loc, h_loc, w_loc;
                loc   = x * MIOPEN_READ_UNIT + i;
                n_loc = loc / (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN);
                c_loc =
                    (loc % (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN)) / (MIOPEN_H_IN * MIOPEN_W_IN);
                h_loc = ((loc % (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN)) %
                         (MIOPEN_H_IN * MIOPEN_W_IN)) /
                        MIOPEN_W_IN;
                w_loc = ((loc % (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN)) %
                         (MIOPEN_H_IN * MIOPEN_W_IN)) %
                        MIOPEN_W_IN;

                data[i] = bot[xOffset + n_loc * MIOPEN_N_IN_STRIDE + c_loc * MIOPEN_C_IN_STRIDE +
                              h_loc * MIOPEN_H_IN_STRIDE + w_loc * MIOPEN_W_IN_STRIDE];
            }
            else
#endif
            {
                data[i] = bot[xOffset + x * MIOPEN_READ_UNIT + i];
            }
        }
        for(; i < MIOPEN_READ_UNIT; ++i)
        {
            data[i] = (FP_TYPE)1.f;
        }
    }
    else
#endif
    {
        for(int i = 0; i < MIOPEN_READ_UNIT; ++i)
        {
#if MIOPEN_N_IN_STRIDE > MIOPEN_IN_BLOCK_SZ
            if(n_in_stride > c_in * h_in * w_in && c_in != 0 && h_in != 0 && w_in != 0)
            {
                int loc, n_loc, c_loc, h_loc, w_loc;
                loc   = x * MIOPEN_READ_UNIT + i;
                n_loc = loc / (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN);
                c_loc =
                    (loc % (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN)) / (MIOPEN_H_IN * MIOPEN_W_IN);
                h_loc = ((loc % (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN)) %
                         (MIOPEN_H_IN * MIOPEN_W_IN)) /
                        MIOPEN_W_IN;
                w_loc = ((loc % (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN)) %
                         (MIOPEN_H_IN * MIOPEN_W_IN)) %
                        MIOPEN_W_IN;

                data[i] = bot[xOffset + n_loc * MIOPEN_N_IN_STRIDE + c_loc * MIOPEN_C_IN_STRIDE +
                              h_loc * MIOPEN_H_IN_STRIDE + w_loc * MIOPEN_W_IN_STRIDE];
            }
            else
#endif
            {
                data[i] = bot[xOffset + x * MIOPEN_READ_UNIT + i];
            }
        }
    }
    ActivationFunction(response, data, gamma, beta, alpha);

#if MIOPEN_N_PIXS_OFF > 0
    if(x == MIOPEN_MAP_SZ_ALIGNED - 1)
    {
        int i = 0;
        for(; i < MIOPEN_N_PIXS_OFF; ++i)
        {
#if MIOPEN_N_OUT_STRIDE > MIOPEN_OUT_BLOCK_SZ
            if(n_out_stride > c_out * h_out * w_out && c_out != 0 && h_out != 0 && w_out != 0)
            {
                int loc, n_loc, c_loc, h_loc, w_loc;
                loc   = x * MIOPEN_READ_UNIT + i;
                n_loc = loc / (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT);
                c_loc = (loc % (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT)) /
                        (MIOPEN_H_OUT * MIOPEN_W_OUT);
                h_loc = ((loc % (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT)) %
                         (MIOPEN_H_OUT * MIOPEN_W_OUT)) /
                        MIOPEN_W_OUT;
                w_loc = ((loc % (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT)) %
                         (MIOPEN_H_OUT * MIOPEN_W_OUT)) %
                        MIOPEN_W_OUT;

                top[yOffset + n_loc * MIOPEN_N_OUT_STRIDE + c_loc * MIOPEN_C_OUT_STRIDE +
                    h_loc * MIOPEN_H_OUT_STRIDE + w_loc * MIOPEN_W_OUT_STRIDE] = response[i];
            }
            else
#endif
            {
                top[yOffset + x * MIOPEN_READ_UNIT + i] = response[i];
            }
        }
    }
    else
#endif
    {
        for(int i = 0; i < MIOPEN_READ_UNIT; ++i)
        {
#if MIOPEN_N_OUT_STRIDE > MIOPEN_OUT_BLOCK_SZ
            if(n_out_stride > c_out * h_out * w_out && c_out != 0 && h_out != 0 && w_out != 0)
            {
                int loc, n_loc, c_loc, h_loc, w_loc;
                loc   = x * MIOPEN_READ_UNIT + i;
                n_loc = loc / (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT);
                c_loc = (loc % (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT)) /
                        (MIOPEN_H_OUT * MIOPEN_W_OUT);
                h_loc = ((loc % (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT)) %
                         (MIOPEN_H_OUT * MIOPEN_W_OUT)) /
                        MIOPEN_W_OUT;
                w_loc = ((loc % (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT)) %
                         (MIOPEN_H_OUT * MIOPEN_W_OUT)) %
                        MIOPEN_W_OUT;

                top[yOffset + n_loc * MIOPEN_N_OUT_STRIDE + c_loc * MIOPEN_C_OUT_STRIDE +
                    h_loc * MIOPEN_H_OUT_STRIDE + w_loc * MIOPEN_W_OUT_STRIDE] = response[i];
            }
            else
#endif
            {
                top[yOffset + x * MIOPEN_READ_UNIT + i] = response[i];
            }
        }
    }
}

#endif // #ifdef LITE
