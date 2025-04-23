/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "hip_tensor_executors.hpp"
#include "rpp_hip_math.hpp"

__global__ void down_mixing_hip_tensor(float *srcPtr,
                                       uint srcStride,
                                       float *dstPtr,
                                       uint dstStride,
                                       int2 *srcDimsTensor)

{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int srcLength = srcDimsTensor[id_z].x;
    int channels = srcDimsTensor[id_z].y;

    if (id_x >= srcLength)
        return;

    float outVal = 0.0f;
    uint srcIdx = id_z * srcStride + id_x * channels;
    int i = 0;
    int alignedChannels = (channels / 8) * 8;

    // do 8 pixel load till alignedChannels value
    if (alignedChannels)
    {
        d_float8 outVal_f8;
        outVal_f8.f4[0] = static_cast<float4>(0.0f);
        outVal_f8.f4[1] = outVal_f8.f4[0];
        for(; i < alignedChannels; i += 8, srcIdx += 8)
        {
            d_float8 src_f8;
            rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
            rpp_hip_math_add8(&outVal_f8, &src_f8, &outVal_f8);
        }
        outVal_f8.f4[0] += outVal_f8.f4[1];
        outVal += (outVal_f8.f1[0] + outVal_f8.f1[1] + outVal_f8.f1[2] + outVal_f8.f1[3]);
    }
    // process remaining channels
    for(; i < channels; i++, srcIdx++)
        outVal += srcPtr[srcIdx];
    outVal *= (1.f / channels);

    uint dstIdx = id_z * dstStride + id_x;
    dstPtr[dstIdx] = outVal;
}

RppStatus hip_exec_down_mixing_tensor(Rpp32f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32s *srcDimsTensor,
                                      bool normalizeWeights,
                                      rpp::Handle& handle)
{
    Rpp32s globalThreads_x = dstDescPtr->strides.nStride;
    Rpp32s globalThreads_y = 1;
    Rpp32s globalThreads_z = dstDescPtr->n;

    hipLaunchKernelGGL(down_mixing_hip_tensor,
                       dim3(ceil((Rpp32f)globalThreads_x/LOCAL_THREADS_X_1DIM), ceil((Rpp32f)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((Rpp32f)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                       dim3(LOCAL_THREADS_X_1DIM, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       srcDescPtr->strides.nStride,
                       dstPtr,
                       dstDescPtr->strides.nStride,
                       reinterpret_cast<int2 *>(srcDimsTensor));

    return RPP_SUCCESS;
}
