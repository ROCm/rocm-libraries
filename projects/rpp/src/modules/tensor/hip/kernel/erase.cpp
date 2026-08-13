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
// srcPtr/srcStridesNH is the pass-through source for pixels outside every anchor box. Pass
// srcPtr == nullptr when the caller has already placed the correct pass-through pixel in dstPtr
// (e.g. via a preceding layout-conversion kernel), so this kernel only needs to punch in the boxes.
template <typename T, typename U>
__global__ void erase_pkd_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr, uint2 dstStridesNH,
                                     RpptRoiLtrb* anchorBoxInfoTensor, U* colorsTensor,
                                     Rpp32u* numBoxesTensor, RpptROIPtr roiTensorPtrSrc) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) ||
        (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    // Anchor boxes are in absolute image coordinates; also used to locate the pass-through
    // source pixel (source is read at the ROI's absolute offset, dst stays packed at the origin)
    int img_x = id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int img_y = id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y;

    Rpp32u numBoxes = numBoxesTensor[id_z];
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    // check if the co-ordinates is within any user defined box
    bool erased = false;
    for (int i = 0; i < numBoxes; i++) {
        int temp = (id_z * numBoxes) + i;
        if (img_x >= anchorBoxInfoTensor[temp].lt.x && img_x <= anchorBoxInfoTensor[temp].rb.x &&
            img_y >= anchorBoxInfoTensor[temp].lt.y && img_y <= anchorBoxInfoTensor[temp].rb.y) {
            *reinterpret_cast<U*>(dstPtr + dstIdx) = static_cast<U>(colorsTensor[temp]);
            erased = true;
            break;
        }
    }

    if (!erased && srcPtr != nullptr) {
        uint srcIdx = (id_z * srcStridesNH.x) + (img_y * srcStridesNH.y) + img_x * 3;
        *reinterpret_cast<U*>(dstPtr + dstIdx) = *reinterpret_cast<U*>(srcPtr + srcIdx);
    }
}

template <typename T>
__global__ void erase_pln_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr, uint3 dstStridesNCH,
                                     RpptRoiLtrb* anchorBoxInfoTensor, T* colorsTensor,
                                     Rpp32u* numBoxesTensor, RpptROIPtr roiTensorPtrSrc) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) ||
        (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    // Anchor boxes are in absolute image coordinates; also used to locate the pass-through
    // source pixel (source is read at the ROI's absolute offset, dst stays packed at the origin)
    int img_x = id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int img_y = id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y;

    Rpp32u numBoxes = numBoxesTensor[id_z];
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    // check if the co-ordinates is within any user defined box
    bool erased = false;
    for (int i = 0; i < numBoxes; i++) {
        int temp = (id_z * numBoxes) + i;
        if (img_x >= anchorBoxInfoTensor[temp].lt.x && img_x <= anchorBoxInfoTensor[temp].rb.x &&
            img_y >= anchorBoxInfoTensor[temp].lt.y && img_y <= anchorBoxInfoTensor[temp].rb.y) {
            *static_cast<T*>((dstPtr + dstIdx)) = colorsTensor[temp];
            erased = true;
            break;
        }
    }

    if (!erased && srcPtr != nullptr) {
        uint srcIdx = (id_z * srcStridesNCH.x) + (img_y * srcStridesNCH.z) + img_x;
        *static_cast<T*>(dstPtr + dstIdx) = *static_cast<T*>(srcPtr + srcIdx);
    }
}

template <typename T>
__global__ void erase_pln3_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                      uint3 dstStridesNCH, RpptRoiLtrb* anchorBoxInfoTensor,
                                      T* colorsTensor, Rpp32u* numBoxesTensor,
                                      RpptROIPtr roiTensorPtrSrc) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) ||
        (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    // Anchor boxes are in absolute image coordinates; also used to locate the pass-through
    // source pixel (source is read at the ROI's absolute offset, dst stays packed at the origin)
    int img_x = id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int img_y = id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y;

    Rpp32u numBoxes = numBoxesTensor[id_z];
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    // check if the co-ordinates is within any user defined box
    bool erased = false;
    for (int i = 0; i < numBoxes; i++) {
        int temp = (id_z * numBoxes) + i;
        if (img_x >= anchorBoxInfoTensor[temp].lt.x && img_x <= anchorBoxInfoTensor[temp].rb.x &&
            img_y >= anchorBoxInfoTensor[temp].lt.y && img_y <= anchorBoxInfoTensor[temp].rb.y) {
            int temp3 = temp * 3;
            *static_cast<T*>(dstPtr + dstIdx) = colorsTensor[temp3];
            *static_cast<T*>(dstPtr + dstIdx + dstStridesNCH.y) = colorsTensor[temp3 + 1];
            *static_cast<T*>(dstPtr + dstIdx + 2 * dstStridesNCH.y) = colorsTensor[temp3 + 2];
            erased = true;
            break;
        }
    }

    if (!erased && srcPtr != nullptr) {
        uint srcIdx = (id_z * srcStridesNCH.x) + (img_y * srcStridesNCH.z) + img_x;
        *static_cast<T*>(dstPtr + dstIdx) = *static_cast<T*>(srcPtr + srcIdx);
        *static_cast<T*>(dstPtr + dstIdx + dstStridesNCH.y) =
            *static_cast<T*>(srcPtr + srcIdx + srcStridesNCH.y);
        *static_cast<T*>(dstPtr + dstIdx + 2 * dstStridesNCH.y) =
            *static_cast<T*>(srcPtr + srcIdx + 2 * srcStridesNCH.y);
    }
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
        // Pass-through source for pixels outside every anchor box, read at the ROI's absolute
        // offset. Left null when a preceding conversion kernel already placed the correct
        // pass-through pixel in dstPtr, so the erase kernel below only needs to punch in boxes.
        T* passthroughSrcPtr = nullptr;
        uint2 passthroughSrcStridesNH = make_uint2(0, 0);

        // if src layout is NHWC, erase kernel copies src to dst per-pixel (absolute source frame)
        if (srcDescPtr->layout == RpptLayout::NHWC) {
            passthroughSrcPtr = srcPtr;
            passthroughSrcStridesNH =
                make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride);
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
                               handle.GetStream(), passthroughSrcPtr, passthroughSrcStridesNH,
                               dstPtr,
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
                               handle.GetStream(), passthroughSrcPtr, passthroughSrcStridesNH,
                               dstPtr,
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
                               handle.GetStream(), passthroughSrcPtr, passthroughSrcStridesNH,
                               dstPtr,
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
                               handle.GetStream(), passthroughSrcPtr, passthroughSrcStridesNH,
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor, reinterpret_cast<d_schar3_s*>(colorsTensor),
                               numBoxesTensor, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        }
    } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NCHW) && dstDescPtr->c == 1) {
        hipLaunchKernelGGL(erase_pln_hip_tensor,
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
                           anchorBoxInfoTensor, colorsTensor, numBoxesTensor, roiTensorPtrSrc);
        HIP_CHECK_LAUNCH_RETURN();
    } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NCHW) && dstDescPtr->c == 3) {
        hipLaunchKernelGGL(erase_pln3_hip_tensor,
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
            // dstPtr already holds the correct pass-through pixel (written by the convert
            // kernel above in the absolute-source/packed-destination frame); pass a null
            // source so this kernel only punches in the erase boxes.
            hipLaunchKernelGGL(erase_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                    ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), static_cast<T*>(nullptr), make_uint3(0, 0, 0),
                               dstPtr,
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
