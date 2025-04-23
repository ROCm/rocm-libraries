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

// -------------------- Set 0 - Erase main kernels --------------------
template <typename T, typename U>
__global__ void erase_pkd_hip_tensor(T *dstPtr,
                                     uint2 dstStridesNH,
                                     RpptRoiLtrb *anchorBoxInfoTensor,
                                     U *colorsTensor,
                                     Rpp32u *numBoxesTensor,
                                     RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    Rpp32u numBoxes = numBoxesTensor[id_z];
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    // check if the co-ordinates is within any user defined box
    for (int i = 0; i < numBoxes; i++)
    {
        int temp = (id_z * numBoxes) + i;
        if (id_x >= anchorBoxInfoTensor[temp].lt.x && id_x <= anchorBoxInfoTensor[temp].rb.x && id_y >= anchorBoxInfoTensor[temp].lt.y && id_y <= anchorBoxInfoTensor[temp].rb.y)
        {
            *reinterpret_cast<U *>(dstPtr + dstIdx) = static_cast<U>(colorsTensor[temp]);
            break;
        }
    }
}

template <typename T>
__global__ void erase_pln_hip_tensor(T *dstPtr,
                                     uint3 dstStridesNCH,
                                     RpptRoiLtrb *anchorBoxInfoTensor,
                                     T *colorsTensor,
                                     Rpp32u *numBoxesTensor,
                                     RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    Rpp32u numBoxes = numBoxesTensor[id_z];
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    // check if the co-ordinates is within any user defined box
    for (int i = 0; i < numBoxes; i++)
    {
        int temp = (id_z * numBoxes) + i;
        if (id_x >= anchorBoxInfoTensor[temp].lt.x && id_x <= anchorBoxInfoTensor[temp].rb.x && id_y >= anchorBoxInfoTensor[temp].lt.y && id_y <= anchorBoxInfoTensor[temp].rb.y)
        {
            *static_cast<T *>((dstPtr + dstIdx)) = colorsTensor[temp];
            break;
        }
    }
}

template <typename T>
__global__ void erase_pln3_hip_tensor(T *dstPtr,
                                      uint3 dstStridesNCH,
                                      RpptRoiLtrb *anchorBoxInfoTensor,
                                      T *colorsTensor,
                                      Rpp32u *numBoxesTensor,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    Rpp32u numBoxes = numBoxesTensor[id_z];
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    // check if the co-ordinates is within any user defined box
    for (int i = 0; i < numBoxes; i++)
    {
        int temp = (id_z * numBoxes) + i;
        if (id_x >= anchorBoxInfoTensor[temp].lt.x && id_x <= anchorBoxInfoTensor[temp].rb.x && id_y >= anchorBoxInfoTensor[temp].lt.y && id_y <= anchorBoxInfoTensor[temp].rb.y)
        {
            temp *= 3;
            *static_cast<T *>(dstPtr + dstIdx) = colorsTensor[temp];
            dstIdx += dstStridesNCH.y;
            *static_cast<T *>(dstPtr + dstIdx) = colorsTensor[temp + 1];
            dstIdx += dstStridesNCH.y;
            *static_cast<T *>(dstPtr + dstIdx) = colorsTensor[temp + 2];
            break;
        }
    }
}

// -------------------- Set 1 - Kernel Executors --------------------
template <typename T, typename U>
RppStatus hip_exec_erase_tensor(T *srcPtr,
                                RpptDescPtr srcDescPtr,
                                T *dstPtr,
                                RpptDescPtr dstDescPtr,
                                RpptRoiLtrb *anchorBoxInfoTensor,
                                U *colorsTensor,
                                Rpp32u *numBoxesTensor,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = dstDescPtr->w;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if (dstDescPtr->layout == RpptLayout::NHWC)
    {
        // if src layout is NHWC, copy src to dst
        if (srcDescPtr->layout == RpptLayout::NHWC)
        {
            hipMemcpyAsync(dstPtr, srcPtr, static_cast<size_t>(srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T)), hipMemcpyDeviceToDevice, handle.GetStream());
            hipStreamSynchronize(handle.GetStream());
        }
        // if src layout is NCHW, convert src from NCHW to NHWC
        else if (srcDescPtr->layout == RpptLayout::NCHW)
        {
            globalThreads_x = (dstDescPtr->w + 7) >> 3;
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
            hipStreamSynchronize(handle.GetStream());
        }

        if (srcDescPtr->dataType == RpptDataType::U8)
        {
            hipLaunchKernelGGL(erase_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor,
                               reinterpret_cast<uchar3*>(colorsTensor),
                               numBoxesTensor,
                               roiTensorPtrSrc);
        }
        else if (srcDescPtr->dataType == RpptDataType::F16)
        {
            hipLaunchKernelGGL(erase_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor,
                               reinterpret_cast<d_half3_s*>(colorsTensor),
                               numBoxesTensor,
                               roiTensorPtrSrc);
        }
        else if (srcDescPtr->dataType == RpptDataType::F32)
        {
            hipLaunchKernelGGL(erase_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor,
                               reinterpret_cast<float3*>(colorsTensor),
                               numBoxesTensor,
                               roiTensorPtrSrc);
        }
        else if (srcDescPtr->dataType == RpptDataType::I8)
        {
            hipLaunchKernelGGL(erase_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor,
                               reinterpret_cast<d_schar3_s*>(colorsTensor),
                               numBoxesTensor,
                               roiTensorPtrSrc);
        }
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW) && dstDescPtr->c == 1)
    {
        hipMemcpyAsync(dstPtr, srcPtr, static_cast<size_t>(srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T)), hipMemcpyDeviceToDevice, handle.GetStream());
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(erase_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           anchorBoxInfoTensor,
                           colorsTensor,
                           numBoxesTensor,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW) && dstDescPtr->c == 3)
    {
        hipMemcpyAsync(dstPtr, srcPtr, static_cast<size_t>(srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T)), hipMemcpyDeviceToDevice, handle.GetStream());
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(erase_pln3_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           anchorBoxInfoTensor,
                           colorsTensor,
                           numBoxesTensor,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            globalThreads_x = (dstDescPtr->w + 7) >> 3;
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
            hipStreamSynchronize(handle.GetStream());
            globalThreads_x = dstDescPtr->w;
            hipLaunchKernelGGL(erase_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor,
                               colorsTensor,
                               numBoxesTensor,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_erase_tensor<Rpp8u, Rpp8u>(Rpp8u*,
                                                       RpptDescPtr,
                                                       Rpp8u*,
                                                       RpptDescPtr,
                                                       RpptRoiLtrb*,
                                                       Rpp8u*,
                                                       Rpp32u*,
                                                       RpptROIPtr,
                                                       RpptRoiType,
                                                       rpp::Handle&);

template RppStatus hip_exec_erase_tensor<half, half>(half*,
                                                     RpptDescPtr,
                                                     half*,
                                                     RpptDescPtr,
                                                     RpptRoiLtrb*,
                                                     half*,
                                                     Rpp32u*,
                                                     RpptROIPtr,
                                                     RpptRoiType,
                                                     rpp::Handle&);

template RppStatus hip_exec_erase_tensor<Rpp32f, Rpp32f>(Rpp32f*,
                                                         RpptDescPtr,
                                                         Rpp32f*,
                                                         RpptDescPtr,
                                                         RpptRoiLtrb*,
                                                         Rpp32f*,
                                                         Rpp32u*,
                                                         RpptROIPtr,
                                                         RpptRoiType,
                                                         rpp::Handle&);

template RppStatus hip_exec_erase_tensor<Rpp8s, Rpp8s>(Rpp8s*,
                                                       RpptDescPtr,
                                                       Rpp8s*,
                                                       RpptDescPtr,
                                                       RpptRoiLtrb*,
                                                       Rpp8s*,
                                                       Rpp32u*,
                                                       RpptROIPtr,
                                                       RpptRoiType,
                                                       rpp::Handle&);
