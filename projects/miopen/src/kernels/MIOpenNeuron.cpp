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
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint index = tid * MIOPEN_READ_UNIT;

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
    uint x_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint y    = blockIdx.y * blockDim.y + threadIdx.y;

    uint bot_index = y * bot_stride + x_id * MIOPEN_READ_UNIT;
    uint top_index = y * top_stride + x_id * MIOPEN_READ_UNIT;

    FP_TYPE data[MIOPEN_READ_UNIT];
    FP_TYPE response[MIOPEN_READ_UNIT];

    *((MIOPEN_READ_TYPE*)data) = *((const MIOPEN_READ_TYPE*)(bot + bot_offset + bot_index));

    ActivationFunction(response, data, gamma, beta, alpha);

    *((MIOPEN_READ_TYPE*)(top + top_offset + top_index)) = *((MIOPEN_READ_TYPE*)response);
}

#endif // #ifdef LITE
