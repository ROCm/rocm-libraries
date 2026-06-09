/*
MIT License

Copyright (c) 2019 - 2026 Advanced Micro Devices, Inc.

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

#define THREADS_PER_BLOCK_X 8                               // Threads per block in X
#define THREADS_PER_BLOCK_X_SHIFT 3                         // Shift value for dividing by 8
#define THREADS_PER_BLOCK_X_MASK (THREADS_PER_BLOCK_X - 1)

// -------------------- Set 0 - random_erase main kernels --------------------
template <typename T>
__global__ void random_erase_pkd_hip_tensor(T *dstPtr,
                                            uint2 dstStridesNH,
                                            RpptRoiLtrb *anchorBoxInfoTensor,
                                            T *noiseBuffer,
                                            int maxWidth,
                                            int maxHeight)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // Clamp anchor box coordinates to image dimensions
    int ltX = max(0, min(anchorBoxInfoTensor[id_z].lt.x, maxWidth - 1));
    int rbX = max(0, min(anchorBoxInfoTensor[id_z].rb.x, maxWidth - 1));
    int ltY = max(0, min(anchorBoxInfoTensor[id_z].lt.y, maxHeight - 1));
    int rbY = max(0, min(anchorBoxInfoTensor[id_z].rb.y, maxHeight - 1));

    if (id_x < ltX || id_x > rbX || id_y < ltY || id_y > rbY)
        return;

    uint noiseIdx = (((id_y + id_z) % RANDOM_ERASE_NOISE_BUFFER_SIDE) * RANDOM_ERASE_NOISE_BUFFER_SIDE + (id_x % RANDOM_ERASE_NOISE_BUFFER_SIDE)) * 3;
    uint dstIdx = id_z * dstStridesNH.x + id_y * dstStridesNH.y + id_x * 3;

    dstPtr[dstIdx]     = noiseBuffer[noiseIdx];
    dstPtr[dstIdx + 1] = noiseBuffer[noiseIdx + 1];
    dstPtr[dstIdx + 2] = noiseBuffer[noiseIdx + 2];
}

template <typename T>
__global__ void random_erase_pln_hip_tensor(T *dstPtr,
                                            uint3 dstStridesNCH,
                                            RpptRoiLtrb *anchorBoxInfoTensor,
                                            T *noiseBuffer,
                                            int maxWidth,
                                            int maxHeight)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // Clamp anchor box coordinates to image dimensions
    int ltX = max(0, min(anchorBoxInfoTensor[id_z].lt.x, maxWidth - 1));
    int rbX = max(0, min(anchorBoxInfoTensor[id_z].rb.x, maxWidth - 1));
    int ltY = max(0, min(anchorBoxInfoTensor[id_z].lt.y, maxHeight - 1));
    int rbY = max(0, min(anchorBoxInfoTensor[id_z].rb.y, maxHeight - 1));

    if (id_x < ltX || id_x > rbX || id_y < ltY || id_y > rbY)
        return;

    uint noiseIdx = (((id_y + id_z) % RANDOM_ERASE_NOISE_BUFFER_SIDE) * RANDOM_ERASE_NOISE_BUFFER_SIDE + (id_x % RANDOM_ERASE_NOISE_BUFFER_SIDE));
    uint dstIdx = id_z * dstStridesNCH.x + id_y * dstStridesNCH.z + id_x;

    dstPtr[dstIdx] = noiseBuffer[noiseIdx];
}

template <typename T>
__global__ void random_erase_pln3_hip_tensor(T *dstPtr,
                                             uint3 dstStridesNCH,
                                             RpptRoiLtrb *anchorBoxInfoTensor,
                                             T *noiseBuffer,
                                             int maxWidth,
                                             int maxHeight)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // Clamp anchor box coordinates to image dimensions
    int ltX = max(0, min(anchorBoxInfoTensor[id_z].lt.x, maxWidth - 1));
    int rbX = max(0, min(anchorBoxInfoTensor[id_z].rb.x, maxWidth - 1));
    int ltY = max(0, min(anchorBoxInfoTensor[id_z].lt.y, maxHeight - 1));
    int rbY = max(0, min(anchorBoxInfoTensor[id_z].rb.y, maxHeight - 1));

    if (id_x < ltX || id_x > rbX || id_y < ltY || id_y > rbY)
        return;

    uint noiseIdx = (((id_y + id_z) % RANDOM_ERASE_NOISE_BUFFER_SIDE) * RANDOM_ERASE_NOISE_BUFFER_SIDE + (id_x % RANDOM_ERASE_NOISE_BUFFER_SIDE)) * 3;
    uint dstIdx = id_z * dstStridesNCH.x + id_y * dstStridesNCH.z + id_x;

    dstPtr[dstIdx]                       = noiseBuffer[noiseIdx];
    dstPtr[dstIdx + dstStridesNCH.y]     = noiseBuffer[noiseIdx + 1];
    dstPtr[dstIdx + 2 * dstStridesNCH.y] = noiseBuffer[noiseIdx + 2];
}

// -------------------- Set 1 - Kernel Executors --------------------
template <typename T>
RppStatus hip_exec_random_erase_tensor(T *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       T *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       RpptRoiLtrb *anchorBoxInfoTensor,
                                       T *noiseBuffer,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_conversion_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = dstDescPtr->w;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if (dstDescPtr->layout == RpptLayout::NHWC)
    {
        // if src layout is NHWC, copy src to dst
        if (srcDescPtr->layout == RpptLayout::NHWC)
        {
            RPP_HIP_RETURN_IF_ERROR(hipMemcpyAsync(dstPtr, srcPtr, static_cast<size_t>(srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T)), hipMemcpyDeviceToDevice, handle.GetStream()));
            RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));
        }
        // if src layout is NCHW, convert src from NCHW to NHWC
        else if (srcDescPtr->layout == RpptLayout::NCHW)
        {
            globalThreads_x = (dstDescPtr->w + THREADS_PER_BLOCK_X_MASK) >> THREADS_PER_BLOCK_X_SHIFT;
            hipLaunchKernelGGL(convert_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc);
            globalThreads_x = dstDescPtr->w;
            RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));
        }

        if (srcDescPtr->dataType == RpptDataType::U8)
        {
            hipLaunchKernelGGL(random_erase_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor,
                               noiseBuffer,
                               dstDescPtr->w,
                               dstDescPtr->h);
        }
        else if (srcDescPtr->dataType == RpptDataType::F16)
        {
            hipLaunchKernelGGL(random_erase_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor,
                               noiseBuffer,
                               dstDescPtr->w,
                               dstDescPtr->h);
        }
        else if (srcDescPtr->dataType == RpptDataType::F32)
        {
            hipLaunchKernelGGL(random_erase_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor,
                               noiseBuffer,
                               dstDescPtr->w,
                               dstDescPtr->h);
        }
        else if (srcDescPtr->dataType == RpptDataType::I8)
        {
            hipLaunchKernelGGL(random_erase_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor,
                               noiseBuffer,
                               dstDescPtr->w,
                               dstDescPtr->h);
        }
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW) && dstDescPtr->c == 1)
    {
        RPP_HIP_RETURN_IF_ERROR(hipMemcpyAsync(dstPtr, srcPtr, static_cast<size_t>(srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T)), hipMemcpyDeviceToDevice, handle.GetStream()));
        RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));
        hipLaunchKernelGGL(random_erase_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           anchorBoxInfoTensor,
                           noiseBuffer,
                           dstDescPtr->w,
                           dstDescPtr->h);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW) && dstDescPtr->c == 3)
    {
        RPP_HIP_RETURN_IF_ERROR(hipMemcpyAsync(dstPtr, srcPtr, static_cast<size_t>(srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T)), hipMemcpyDeviceToDevice, handle.GetStream()));
        RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));
        hipLaunchKernelGGL(random_erase_pln3_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           anchorBoxInfoTensor,
                           noiseBuffer,
                           dstDescPtr->w,
                           dstDescPtr->h);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            globalThreads_x = (dstDescPtr->w + THREADS_PER_BLOCK_X_MASK) >> THREADS_PER_BLOCK_X_SHIFT;
            hipLaunchKernelGGL(convert_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc);
            RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));
            globalThreads_x = dstDescPtr->w;
            hipLaunchKernelGGL(random_erase_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor,
                               noiseBuffer,
                               dstDescPtr->w,
                               dstDescPtr->h);
        }
    }
    else
        return RPP_ERROR_NOT_IMPLEMENTED;

    return RPP_SUCCESS;
}

template RppStatus hip_exec_random_erase_tensor<Rpp8u>(Rpp8u*,
                                                       RpptDescPtr,
                                                       Rpp8u*,
                                                       RpptDescPtr,
                                                       RpptRoiLtrb*,
                                                       Rpp8u*,
                                                       RpptROIPtr,
                                                       RpptRoiType,
                                                       rpp::Handle&);

template RppStatus hip_exec_random_erase_tensor<half>(half*,
                                                     RpptDescPtr,
                                                     half*,
                                                     RpptDescPtr,
                                                     RpptRoiLtrb*,
                                                     half*,
                                                     RpptROIPtr,
                                                     RpptRoiType,
                                                     rpp::Handle&);

template RppStatus hip_exec_random_erase_tensor<Rpp32f>(Rpp32f*,
                                                         RpptDescPtr,
                                                         Rpp32f*,
                                                         RpptDescPtr,
                                                         RpptRoiLtrb*,
                                                         Rpp32f*,
                                                         RpptROIPtr,
                                                         RpptRoiType,
                                                         rpp::Handle&);

template RppStatus hip_exec_random_erase_tensor<Rpp8s>(Rpp8s*,
                                                       RpptDescPtr,
                                                       Rpp8s*,
                                                       RpptDescPtr,
                                                       RpptRoiLtrb*,
                                                       Rpp8s*,
                                                       RpptROIPtr,
                                                       RpptRoiType,
                                                       rpp::Handle&);
