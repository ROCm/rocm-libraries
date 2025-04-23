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
template <typename T>
__device__ void log_hip_compute(T *srcPtr, d_float8 *src_f8, d_float8 *dst_f8)
{
    if constexpr (std::is_same<T, schar>::value)
        rpp_hip_math_add8_const(src_f8, src_f8, (float4)128);

    rpp_hip_math_log(src_f8, dst_f8);
}

// -------------------- Set 2 - log kernels --------------------
template <typename T, typename U>
__global__ void log_1d_hip_tensor(T *srcPtr,
                                  uint srcStrides,
                                  U *dstPtr,
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
    log_hip_compute(srcPtr, &src_f8, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

template <typename T, typename U>
__global__ void log_2d_hip_tensor(T *srcPtr,
                                  uint2 srcStridesNH,
                                  U *dstPtr,
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
    log_hip_compute(srcPtr, &src_f8, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

template <typename T, typename U>
__global__ void log_3d_hip_tensor(T *srcPtr,
                                  uint2 srcStridesDH,
                                  U *dstPtr,
                                  uint2 dstStridesDH,
                                  uint *roiTensor)
{
    uint id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; // lengthX
    uint id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y; // lengthY
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // lengthZ

    uint *roi = roiTensor;
    uint beginZ = roi[0];
    uint beginY = roi[1];
    uint beginX = roi[2];
    uint lengthZ = roi[3];
    uint lengthY = roi[4];
    uint lengthX = roi[5];

    if (id_x >= lengthX || id_y >= lengthY || id_z >= lengthZ)
        return;

    uint srcIdx = ((id_z + beginZ) * srcStridesDH.x) + ((id_y + beginY) * srcStridesDH.y) + id_x + beginX;
    uint dstIdx = (id_z * dstStridesDH.x) + (id_y * dstStridesDH.y) + id_x;

    d_float8 src_f8, dst_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    log_hip_compute(srcPtr, &src_f8, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

template <typename T, typename U>
__global__ void log_nd_hip_tensor(T *srcPtr,
                                  uint *srcStrides,
                                  uint *srcDims,
                                  uint numDims,
                                  U *dstPtr,
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
    log_hip_compute(srcPtr, &src_f8, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

// -------------------- Set 3 - executor kernels --------------------
template <typename T, typename U>
RppStatus hip_exec_log_generic_tensor(T *srcPtr,
                                      RpptGenericDescPtr srcGenericDescPtr,
                                      U *dstPtr,
                                      RpptGenericDescPtr dstGenericDescPtr,
                                      uint *roiTensor,
                                      rpp::Handle& handle)
{
    Rpp32u numDims = srcGenericDescPtr->numDims - 1; // exclude batchsize from input dims
    // based on number of dimensions call the corresponding kernel
    if (numDims == 1)
    {
        // NW
        int globalThreads_x = dstGenericDescPtr->dims[1];
        int globalThreads_y = 1;
        int globalThreads_z = dstGenericDescPtr->dims[0];

        hipLaunchKernelGGL(log_1d_hip_tensor,
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
        int globalThreads_x = dstGenericDescPtr->dims[2];
        int globalThreads_y = dstGenericDescPtr->dims[1];
        int globalThreads_z = dstGenericDescPtr->dims[0];

        hipLaunchKernelGGL(log_2d_hip_tensor,
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
    else if (numDims == 3)
    {
        // NDHW
        int globalThreads_x = dstGenericDescPtr->dims[3];
        int globalThreads_y = dstGenericDescPtr->dims[2];
        int globalThreads_z = dstGenericDescPtr->dims[1];

        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            hipLaunchKernelGGL(log_3d_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr + (batchCount * srcGenericDescPtr->strides[0]),
                               make_uint2(srcGenericDescPtr->strides[1], srcGenericDescPtr->strides[2]),
                               dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                               make_uint2(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2]),
                               &roiTensor[batchCount * 6]);
        }
    }
    else
    {
        // interpret the input as 1D tensor
        int globalThreads_x = (dstGenericDescPtr->strides[0] + 7) >> 3;
        int globalThreads_y = 1;
        int globalThreads_z = dstGenericDescPtr->dims[0];

        hipLaunchKernelGGL(log_nd_hip_tensor,
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

template RppStatus hip_exec_log_generic_tensor<Rpp8u, Rpp32f>(Rpp8u*,
                                                              RpptGenericDescPtr,
                                                              Rpp32f*,
                                                              RpptGenericDescPtr,
                                                              uint*,
                                                              rpp::Handle&);

template RppStatus hip_exec_log_generic_tensor<half, half>(half*,
                                                           RpptGenericDescPtr,
                                                           half*,
                                                           RpptGenericDescPtr,
                                                           uint*,
                                                           rpp::Handle&);

template RppStatus hip_exec_log_generic_tensor<Rpp32f, Rpp32f>(Rpp32f*,
                                                               RpptGenericDescPtr,
                                                               Rpp32f*,
                                                               RpptGenericDescPtr,
                                                               uint*,
                                                               rpp::Handle&);

template RppStatus hip_exec_log_generic_tensor<Rpp8s, Rpp32f>(Rpp8s*,
                                                              RpptGenericDescPtr,
                                                              Rpp32f*,
                                                              RpptGenericDescPtr,
                                                              uint*,
                                                              rpp::Handle&);
