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

// -------------------- Set 1 - helper kernels --------------------
__device__ void log1p_hip_compute(Rpp16s *srcPtr, d_float8 *src_f8, d_float8 *dst_f8)
{
    // Apply absolute value to each element
    for(int i = 0; i < 8; i++)
        src_f8->f1[i] =  fabsf(src_f8->f1[i]);

    rpp_hip_math_add8_const(src_f8, src_f8, (float4)1);
    rpp_hip_math_log1p(src_f8, dst_f8);
}

// -------------------- Set 2 - log1p kernels --------------------
__global__ void log1p_1d_hip_tensor(Rpp16s *srcPtr,
                                    uint srcStrides,
                                    Rpp32f *dstPtr,
                                    uint dstStrides,
                                    uint *roiTensor)
{
    uint id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8; // width
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;       // batchsize

    uint *roi = &roiTensor[id_z * 2];
    uint beginX = roi[0];
    uint width = roi[1];

    if (id_x >= width)
        return;

    uint srcIdx = (id_z * srcStrides) + id_x + beginX;
    uint dstIdx = (id_z * dstStrides) + id_x;

    d_float8 src_f8, dst_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    log1p_hip_compute(srcPtr, &src_f8, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

__global__ void log1p_2d_hip_tensor(Rpp16s *srcPtr,
                                    uint2 srcStridesNH,
                                    Rpp32f *dstPtr,
                                    uint2 dstStridesNH,
                                    uint *roiTensor)
{
    uint id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8; // width
    uint id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;       // height
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;       // batchsize

    uint *roi = &roiTensor[id_z * 4];
    uint beginY = roi[0];
    uint beginX = roi[1];
    uint height = roi[2];
    uint width = roi[3];

    if (id_x >= width || id_y >= height)
        return;

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + beginY) * srcStridesNH.y) + id_x + beginX;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;

    d_float8 src_f8, dst_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    log1p_hip_compute(srcPtr, &src_f8, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

__global__ void log1p_nd_hip_tensor(Rpp16s *srcPtr,
                                    uint *srcStrides,
                                    uint *srcDims,
                                    uint numDims,
                                    Rpp32f *dstPtr,
                                    uint *dstStrides,
                                    Rpp32u *roiTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // batchsize

    if(id_x >= srcStrides[0])
        return;

    uint *roi = roiTensor + id_z * numDims * 2;
    uint *begin = roi;
    uint *length = &roi[numDims];
    uint dstIdx = (id_z * *dstStrides++);
    uint srcIdx = (id_z * *srcStrides++);
    uint coords[RPPT_MAX_DIMS];

    for (int i = 0; i < numDims; i++)
    {
        coords[i] = (id_x / srcStrides[i]) % srcDims[i];
        if(coords[i] >= length[i])
            return;
    }

    for (int i = 0; i < numDims; i++)
    {
        dstIdx += (coords[i] * dstStrides[i]);
        srcIdx += (begin[i] + (coords[i] * srcStrides[i]));
    }

    d_float8 src_f8, dst_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    log1p_hip_compute(srcPtr, &src_f8, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

// -------------------- Set 3 - kernel executor --------------------
RppStatus hip_exec_log1p_i16_f32_tensor(Rpp16s *srcPtr,
                                        RpptGenericDescPtr srcGenericDescPtr,
                                        Rpp32f *dstPtr,
                                        RpptGenericDescPtr dstGenericDescPtr,
                                        uint *roiTensor,
                                        rpp::Handle& handle)
{
    Rpp32u numDims = srcGenericDescPtr->numDims - 1; // exclude batchsize from input dims
    int globalThreads_x, globalThreads_y, globalThreads_z;
    // based on number of dimensions call the corresponding kernel
    if (numDims == 1)
    {
        // NW
        globalThreads_x = (dstGenericDescPtr->dims[1] + 7) >> 3;
        globalThreads_y = 1;
        globalThreads_z = dstGenericDescPtr->dims[0];

        hipLaunchKernelGGL(log1p_1d_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           srcGenericDescPtr->strides[0],
                           dstPtr,
                           dstGenericDescPtr->strides[0],
                           roiTensor);
    }
    else if (numDims == 2)
    {
        // NHW
        globalThreads_x = (dstGenericDescPtr->dims[2] + 7) >> 3;
        globalThreads_y = dstGenericDescPtr->dims[1];
        globalThreads_z = dstGenericDescPtr->dims[0];

        hipLaunchKernelGGL(log1p_2d_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcGenericDescPtr->strides[0], srcGenericDescPtr->strides[1]),
                           dstPtr,
                           make_uint2(dstGenericDescPtr->strides[0], dstGenericDescPtr->strides[1]),
                           roiTensor);
    }
    else
    {
        // interpret the input as 1D tensor
        globalThreads_x = (dstGenericDescPtr->strides[0] + 7) >> 3;
        globalThreads_y = 1;
        globalThreads_z = dstGenericDescPtr->dims[0];

        hipLaunchKernelGGL(log1p_nd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/1024), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                           dim3(1024, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           srcGenericDescPtr->strides,
                           srcGenericDescPtr->dims + 1,
                           srcGenericDescPtr->numDims - 1,
                           dstPtr,
                           dstGenericDescPtr->strides,
                           roiTensor);
    }

    return RPP_SUCCESS;
}
