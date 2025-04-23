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
#include "kernel_dims.hpp"

// -------------------- Set 0 - Reduction Stage 2 --------------------
template <typename T>
__global__ void tensor_mean_grid_result_hip(T *srcPtr,
                                            uint xBufferLength,
                                            float *dstPtr,
                                            RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialSum_smem[1024];                         // 8192 floats of src reduced to 1024 in a 1024 x 1 thread block
    partialSum_smem[hipThreadIdx_x] = 0.0f;                         // initialization of Shared to 0 using all 1024 x 1 threads

    if (id_x >= xBufferLength)
        return;

    int xDiff = xBufferLength - (xBufferLength & ~7);               // difference between bufferLength and alignedLength, where alignedLength = bufferLength & ~7
    uint srcIdx = (id_z * xBufferLength) + id_x;

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);   // load 8 pixels to local memory

    if (id_x + 8 > xBufferLength)
        for(int i = xDiff; i < 8; i++)
            src_f8.f1[i] = 0.0f;                                    // local memory reset of invalid values (from the vectorized global load) to 0.0f

    src_f8.f4[0] += src_f8.f4[1];                                   // perform small work of vectorized float4 addition
    partialSum_smem[hipThreadIdx_x] += (src_f8.f1[0] +
                                        src_f8.f1[1] +
                                        src_f8.f1[2] +
                                        src_f8.f1[3]);              // perform small work of reducing float4s to float using 1024 x 1 threads and store in Shared
    __syncthreads();                                                // syncthreads after Shared load

    // Reduction of 1024 floats on 1024 threads per block in x dimension
    for (int threadMax = 512; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialSum_smem[hipThreadIdx_x] += partialSum_smem[hipThreadIdx_x + threadMax];
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        int totalElements = roiTensorPtrSrc[hipBlockIdx_z].xywhROI.roiHeight * roiTensorPtrSrc[hipBlockIdx_z].xywhROI.roiWidth;
        dstPtr[hipBlockIdx_z] = partialSum_smem[0] / totalElements;
    }
}

template <typename T>
__global__ void tensor_mean_grid_3channel_result_hip(T *srcPtr,
                                                     uint xBufferLength,
                                                     float *dstPtr,
                                                     RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialRSum_smem[1024];                                     // 8192 floats of src reduced to 1024 in a 1024 x 1 thread block
    __shared__ float partialGSum_smem[1024];
    __shared__ float partialBSum_smem[1024];
    partialRSum_smem[hipThreadIdx_x] = 0.0f;                                     // initialization of Shared to 0 using all 1024 x 1 threads
    partialGSum_smem[hipThreadIdx_x] = 0.0f;
    partialBSum_smem[hipThreadIdx_x] = 0.0f;

    if (id_x >= xBufferLength)
        return;

    int xDiff = xBufferLength - (xBufferLength & ~7);                            // difference between bufferLength and alignedLength, where alignedLength = bufferLength & ~7
    uint srcIdx = ((id_z * xBufferLength) + id_x) * 3;

    d_float24 src_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);   // load 24 pixels to local memory
    if (id_x + 8 > xBufferLength)                                                // local memory reset of invalid values (from the vectorized global load) to 0.0f
    {
        for(int i = xDiff; i < 8; i++)
        {
            src_f24.f8[0].f1[i] = 0.0f;
            src_f24.f8[1].f1[i] = 0.0f;
            src_f24.f8[2].f1[i] = 0.0f;
        }
    }
    src_f24.f8[0].f4[0] += src_f24.f8[0].f4[1];                                  // perform small work of vectorized float4 addition
    src_f24.f8[1].f4[0] += src_f24.f8[1].f4[1];
    src_f24.f8[2].f4[0] += src_f24.f8[2].f4[1];
    partialRSum_smem[hipThreadIdx_x] = (src_f24.f8[0].f1[0] +
                                        src_f24.f8[0].f1[1] +
                                        src_f24.f8[0].f1[2] +
                                        src_f24.f8[0].f1[3]);                    // perform small work of reducing R float4s to float using 1024 threads and store in Shared
    partialGSum_smem[hipThreadIdx_x] = (src_f24.f8[1].f1[0] +
                                        src_f24.f8[1].f1[1] +
                                        src_f24.f8[1].f1[2] +
                                        src_f24.f8[1].f1[3]);                    // perform small work of reducing G float4s to float using 1024 threads and store in Shared
    partialBSum_smem[hipThreadIdx_x] = (src_f24.f8[2].f1[0] +
                                        src_f24.f8[2].f1[1] +
                                        src_f24.f8[2].f1[2] +
                                        src_f24.f8[2].f1[3]);                    // perform small work of reducing B float4s to float using 1024 threads and store in Shared

    __syncthreads();                                                             // syncthreads after Shared load

    // Reduction of 1024 floats on 1024 threads per block in x dimension
    for (int threadMax = 512; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRSum_smem[hipThreadIdx_x] += partialRSum_smem[hipThreadIdx_x + threadMax];
            partialGSum_smem[hipThreadIdx_x] += partialGSum_smem[hipThreadIdx_x + threadMax];
            partialBSum_smem[hipThreadIdx_x] += partialBSum_smem[hipThreadIdx_x + threadMax];
        }
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        int totalElements = roiTensorPtrSrc[hipBlockIdx_z].xywhROI.roiHeight * roiTensorPtrSrc[hipBlockIdx_z].xywhROI.roiWidth;
        float sum = partialRSum_smem[0] + partialGSum_smem[0] + partialBSum_smem[0];
        int idx = hipBlockIdx_z * 4;
        dstPtr[idx] = partialRSum_smem[0] / totalElements;
        dstPtr[idx + 1] = partialGSum_smem[0] / totalElements;
        dstPtr[idx + 2] = partialBSum_smem[0] / totalElements;
        dstPtr[idx + 3] = sum  / (totalElements * 3);
    }
}

// -------------------- Set 1 - Kernel Executors --------------------

template <typename T, typename U>
RppStatus hip_exec_tensor_mean(T *srcPtr,
                               RpptDescPtr srcDescPtr,
                               Rpp32f *tensorMeanArr,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (srcDescPtr->w + 7) >> 3;
    int globalThreads_y = srcDescPtr->h;
    int globalThreads_z = srcDescPtr->n;
    int gridDim_x = (int) ceil((float)globalThreads_x/LOCAL_THREADS_X);
    int gridDim_y = (int) ceil((float)globalThreads_y/LOCAL_THREADS_Y);
    int gridDim_z = (int) ceil((float)globalThreads_z/LOCAL_THREADS_Z);

    if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u tensorPartialSumArrLength = gridDim_x * gridDim_y * gridDim_z;
        U *tensorPartialSumArr;
        tensorPartialSumArr = reinterpret_cast<U*>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
        hipMemsetAsync(tensorPartialSumArr, 0, tensorPartialSumArrLength * sizeof(U), handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pln1_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           tensorPartialSumArr,
                           roiTensorPtrSrc);
        hipLaunchKernelGGL(tensor_mean_grid_result_hip,
                           dim3(1, 1, gridDim_z),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           tensorPartialSumArr,
                           gridDim_x * gridDim_y,
                           tensorMeanArr,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u tensorPartialSumArrLength = gridDim_x * gridDim_y * gridDim_z * 3;
        U *tensorPartialSumArr;
        tensorPartialSumArr = reinterpret_cast<U*>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
        hipMemsetAsync(tensorPartialSumArr, 0, tensorPartialSumArrLength * sizeof(U), handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pln3_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           tensorPartialSumArr,
                           roiTensorPtrSrc);
        hipLaunchKernelGGL(tensor_mean_grid_3channel_result_hip,
                           dim3(1, 1, gridDim_z),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           tensorPartialSumArr,
                           gridDim_x * gridDim_y,
                           tensorMeanArr,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
    {
        Rpp32u tensorPartialSumArrLength = gridDim_x * gridDim_y * gridDim_z * 3;
        U *tensorPartialSumArr;
        tensorPartialSumArr = reinterpret_cast<U*>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
        hipMemsetAsync(tensorPartialSumArr, 0, tensorPartialSumArrLength * sizeof(U), handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pkd3_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           tensorPartialSumArr,
                           roiTensorPtrSrc);
        hipLaunchKernelGGL(tensor_mean_grid_3channel_result_hip,
                           dim3(1, 1, gridDim_z),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           tensorPartialSumArr,
                           gridDim_x * gridDim_y,
                           tensorMeanArr,
                           roiTensorPtrSrc);
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_tensor_mean<Rpp8u, Rpp32u>(Rpp8u*,
                                                       RpptDescPtr,
                                                       Rpp32f*,
                                                       RpptROIPtr,
                                                       RpptRoiType,
                                                       rpp::Handle&);

template RppStatus hip_exec_tensor_mean<half, float>(half*,
                                                     RpptDescPtr,
                                                     Rpp32f*,
                                                     RpptROIPtr,
                                                     RpptRoiType,
                                                     rpp::Handle&);

template RppStatus hip_exec_tensor_mean<Rpp32f, float>(Rpp32f*,
                                                       RpptDescPtr,
                                                       Rpp32f*,
                                                       RpptROIPtr,
                                                       RpptRoiType,
                                                       rpp::Handle&);

template RppStatus hip_exec_tensor_mean<Rpp8s, Rpp32s>(Rpp8s*,
                                                       RpptDescPtr,
                                                       Rpp32f*,
                                                       RpptROIPtr,
                                                       RpptRoiType,
                                                       rpp::Handle&);
