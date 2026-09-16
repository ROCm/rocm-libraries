/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

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

__global__ void audio_tensor_add_scalar_hip(float* srcPtr1, uint srcStride, float* dstPtr,
                                            uint dstStride, int* srcLengthTensor,
                                            float* scalarTensor) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= srcLengthTensor[id_z]) return;

    uint srcIdx = (id_z * srcStride) + id_x;
    uint dstIdx = (id_z * dstStride) + id_x;
    float scalarValue = scalarTensor[id_z];
    float4 scalarValue_f4 = MAKE_FLOAT4(scalarValue);

    d_float8 src_f8, dst_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr1 + srcIdx, &src_f8);
    dst_f8.f4[0] = src_f8.f4[0] + scalarValue_f4;
    dst_f8.f4[1] = src_f8.f4[1] + scalarValue_f4;
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

RppStatus hip_exec_audio_tensor_add_tensor(Rpp32f* srcPtr1, Rpp32f* srcPtr2, RpptDescPtr srcDescPtr,
                                           Rpp32f* dstPtr, RpptDescPtr dstDescPtr,
                                           Rpp32s* srcLengthTensor, rpp::Handle& handle) {
    Rpp32s globalThreads_x = (srcDescPtr->strides.nStride + 7) >> 3;
    Rpp32s globalThreads_y = 1;
    Rpp32s globalThreads_z = srcDescPtr->n;

    // Scalar per batch broadcasting: srcPtr2 has shape (batchSize, 1)
    hipLaunchKernelGGL(audio_tensor_add_scalar_hip,
                       dim3(ceil((Rpp32f)globalThreads_x / LOCAL_THREADS_X_1DIM),
                            ceil((Rpp32f)globalThreads_y / LOCAL_THREADS_Y_1DIM),
                            ceil((Rpp32f)globalThreads_z / LOCAL_THREADS_Z_1DIM)),
                       dim3(LOCAL_THREADS_X_1DIM, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM), 0,
                       handle.GetStream(), srcPtr1, srcDescPtr->strides.nStride, dstPtr,
                       dstDescPtr->strides.nStride, srcLengthTensor, srcPtr2);

    return RPP_SUCCESS;
}
