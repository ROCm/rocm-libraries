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
__global__ void erase_pkd_hip_tensor(T* dstPtr, uint2 dstStridesNH,
                                     RpptRoiLtrb* anchorBoxInfoTensor, U* colorsTensor,
                                     Rpp32u* numBoxesTensor, RpptROIPtr roiTensorPtrSrc) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) ||
        (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    Rpp32u numBoxes = numBoxesTensor[id_z];
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    // Anchor boxes are in image space; convert ROI-local thread coordinates to image space
    // before comparing against them
    int imgX = id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int imgY = id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y;

    // check if the co-ordinates is within any user defined box
    for (int i = 0; i < numBoxes; i++) {
        int temp = (id_z * numBoxes) + i;
        if (imgX >= anchorBoxInfoTensor[temp].lt.x && imgX <= anchorBoxInfoTensor[temp].rb.x &&
            imgY >= anchorBoxInfoTensor[temp].lt.y && imgY <= anchorBoxInfoTensor[temp].rb.y) {
            *reinterpret_cast<U*>(dstPtr + dstIdx) = static_cast<U>(colorsTensor[temp]);
            break;
        }
    }
}

template <typename T>
__global__ void erase_pln_hip_tensor(T* dstPtr, uint3 dstStridesNCH,
                                     RpptRoiLtrb* anchorBoxInfoTensor, T* colorsTensor,
                                     Rpp32u* numBoxesTensor, RpptROIPtr roiTensorPtrSrc) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) ||
        (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    Rpp32u numBoxes = numBoxesTensor[id_z];
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    // Anchor boxes are in image space; convert ROI-local thread coordinates to image space
    // before comparing against them
    int imgX = id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int imgY = id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y;

    // check if the co-ordinates is within any user defined box
    for (int i = 0; i < numBoxes; i++) {
        int temp = (id_z * numBoxes) + i;
        if (imgX >= anchorBoxInfoTensor[temp].lt.x && imgX <= anchorBoxInfoTensor[temp].rb.x &&
            imgY >= anchorBoxInfoTensor[temp].lt.y && imgY <= anchorBoxInfoTensor[temp].rb.y) {
            *static_cast<T*>((dstPtr + dstIdx)) = colorsTensor[temp];
            break;
        }
    }
}

template <typename T>
__global__ void erase_pln3_hip_tensor(T* dstPtr, uint3 dstStridesNCH,
                                      RpptRoiLtrb* anchorBoxInfoTensor, T* colorsTensor,
                                      Rpp32u* numBoxesTensor, RpptROIPtr roiTensorPtrSrc) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) ||
        (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    Rpp32u numBoxes = numBoxesTensor[id_z];
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    // Anchor boxes are in image space; convert ROI-local thread coordinates to image space
    // before comparing against them
    int imgX = id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int imgY = id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y;

    // check if the co-ordinates is within any user defined box
    for (int i = 0; i < numBoxes; i++) {
        int temp = (id_z * numBoxes) + i;
        if (imgX >= anchorBoxInfoTensor[temp].lt.x && imgX <= anchorBoxInfoTensor[temp].rb.x &&
            imgY >= anchorBoxInfoTensor[temp].lt.y && imgY <= anchorBoxInfoTensor[temp].rb.y) {
            temp *= 3;
            *static_cast<T*>(dstPtr + dstIdx) = colorsTensor[temp];
            dstIdx += dstStridesNCH.y;
            *static_cast<T*>(dstPtr + dstIdx) = colorsTensor[temp + 1];
            dstIdx += dstStridesNCH.y;
            *static_cast<T*>(dstPtr + dstIdx) = colorsTensor[temp + 2];
            break;
        }
    }
}

// ROI-aware shift-copy kernels: read source at the ROI offset but write destination packed at
// the origin, so the box-application kernels above (which operate on the packed-origin frame)
// see correctly copied background pixels under a partial ROI
template <typename T>
__global__ void erase_shift_copy_pkd_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                                uint2 dstStridesNH, RpptROIPtr roiTensorPtrSrc) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) ||
        (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesNH.x) +
                  ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) +
                  (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    dstPtr[dstIdx] = srcPtr[srcIdx];
    dstPtr[dstIdx + 1] = srcPtr[srcIdx + 1];
    dstPtr[dstIdx + 2] = srcPtr[srcIdx + 2];
}

template <typename T>
__global__ void erase_shift_copy_pln_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                                uint3 dstStridesNCH, RpptROIPtr roiTensorPtrSrc) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) ||
        (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesNCH.x) +
                  ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) +
                  (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    dstPtr[dstIdx] = srcPtr[srcIdx];
}

template <typename T>
__global__ void erase_shift_copy_pln3_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                                 uint3 dstStridesNCH, RpptROIPtr roiTensorPtrSrc) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) ||
        (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesNCH.x) +
                  ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) +
                  (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    dstPtr[dstIdx] = srcPtr[srcIdx];
    srcIdx += srcStridesNCH.y;
    dstIdx += dstStridesNCH.y;
    dstPtr[dstIdx] = srcPtr[srcIdx];
    srcIdx += srcStridesNCH.y;
    dstIdx += dstStridesNCH.y;
    dstPtr[dstIdx] = srcPtr[srcIdx];
}

// -------------------- Set 1 - Kernel Executors --------------------
template <typename T, typename U>
RppStatus hip_exec_erase_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                RpptDescPtr dstDescPtr, RpptRoiLtrb* anchorBoxInfoTensor,
                                U* colorsTensor, Rpp32u* numBoxesTensor, RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType, rpp::Handle& handle) {
    if (roiType == RpptRoiType::LTRB) hip_exec_roi_conversion_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = dstDescPtr->w;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if (dstDescPtr->layout == RpptLayout::NHWC) {
        // if src layout is NHWC, copy src to dst, shifted by the ROI offset so the packed-origin
        // destination frame matches what the box-application kernel expects
        if (srcDescPtr->layout == RpptLayout::NHWC) {
            hipLaunchKernelGGL(
                erase_shift_copy_pkd_hip_tensor,
                dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                     ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                     ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(),
                srcPtr, make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                dstPtr, make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
            RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));
        }
        // if src layout is NCHW, convert src from NCHW to NHWC
        else if (srcDescPtr->layout == RpptLayout::NCHW) {
            globalThreads_x = (dstDescPtr->w + 7) >> 3;
            hipLaunchKernelGGL(convert_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                    ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                          srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
            globalThreads_x = dstDescPtr->w;
            RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));
        }

        if (srcDescPtr->dataType == RpptDataType::U8) {
            hipLaunchKernelGGL(erase_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                    ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor, reinterpret_cast<uchar3*>(colorsTensor),
                               numBoxesTensor, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        } else if (srcDescPtr->dataType == RpptDataType::F16) {
            hipLaunchKernelGGL(erase_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                    ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor, reinterpret_cast<d_half3_s*>(colorsTensor),
                               numBoxesTensor, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        } else if (srcDescPtr->dataType == RpptDataType::F32) {
            hipLaunchKernelGGL(erase_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                    ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor, reinterpret_cast<float3*>(colorsTensor),
                               numBoxesTensor, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        } else if (srcDescPtr->dataType == RpptDataType::I8) {
            hipLaunchKernelGGL(erase_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                    ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor, reinterpret_cast<d_schar3_s*>(colorsTensor),
                               numBoxesTensor, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        }
    } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NCHW) && dstDescPtr->c == 1) {
        hipLaunchKernelGGL(erase_shift_copy_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                           handle.GetStream(), srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                      srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                      dstDescPtr->strides.hStride),
                           roiTensorPtrSrc);
        HIP_CHECK_LAUNCH_RETURN();
        RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));
        hipLaunchKernelGGL(erase_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                           handle.GetStream(), dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                      dstDescPtr->strides.hStride),
                           anchorBoxInfoTensor, colorsTensor, numBoxesTensor, roiTensorPtrSrc);
        HIP_CHECK_LAUNCH_RETURN();
    } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NCHW) && dstDescPtr->c == 3) {
        hipLaunchKernelGGL(erase_shift_copy_pln3_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                           handle.GetStream(), srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                      srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                      dstDescPtr->strides.hStride),
                           roiTensorPtrSrc);
        HIP_CHECK_LAUNCH_RETURN();
        RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));
        hipLaunchKernelGGL(erase_pln3_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                           handle.GetStream(), dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                      dstDescPtr->strides.hStride),
                           anchorBoxInfoTensor, colorsTensor, numBoxesTensor, roiTensorPtrSrc);
        HIP_CHECK_LAUNCH_RETURN();
    } else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3)) {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW)) {
            globalThreads_x = (dstDescPtr->w + 7) >> 3;
            hipLaunchKernelGGL(convert_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                    ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                          dstDescPtr->strides.hStride),
                               roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
            RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));
            globalThreads_x = dstDescPtr->w;
            hipLaunchKernelGGL(erase_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                    ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                          dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor, colorsTensor, numBoxesTensor, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_erase_tensor<Rpp8u, Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*, RpptDescPtr,
                                                       RpptRoiLtrb*, Rpp8u*, Rpp32u*, RpptROIPtr,
                                                       RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_erase_tensor<half, half>(half*, RpptDescPtr, half*, RpptDescPtr,
                                                     RpptRoiLtrb*, half*, Rpp32u*, RpptROIPtr,
                                                     RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_erase_tensor<Rpp32f, Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*, RpptDescPtr,
                                                         RpptRoiLtrb*, Rpp32f*, Rpp32u*, RpptROIPtr,
                                                         RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_erase_tensor<Rpp8s, Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*, RpptDescPtr,
                                                       RpptRoiLtrb*, Rpp8s*, Rpp32u*, RpptROIPtr,
                                                       RpptRoiType, rpp::Handle&);
