/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

// -------------------- Set 0 - Coarse dropout main kernels --------------------
// srcPtr/srcStridesNH is the pass-through source for pixels outside every anchor box. Pass
// srcPtr == nullptr when the caller has already placed the correct pass-through pixel in dstPtr
// (e.g. via a preceding layout-conversion kernel), so this kernel only needs to punch in boxes.
template <typename T>
__global__ void coarse_dropout_pkd_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                              uint2 dstStridesNH, RpptRoiLtrb* anchorBoxInfoTensor,
                                              Rpp32u* numBoxesTensor, RpptROIPtr roiTensorPtrSrc,
                                              int maxBoxesPerImage) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) ||
        (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    // Clamp numBoxes to maxBoxesPerImage to prevent buffer overflow
    Rpp32u numBoxes = min(numBoxesTensor[id_z], static_cast<Rpp32u>(maxBoxesPerImage));
    int boxStartOffset = id_z * maxBoxesPerImage;

    // Get ROI origin for coordinate conversion from image space to ROI-local space
    int roiX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiY = roiTensorPtrSrc[id_z].xywhROI.xy.y;

    // Convert thread coordinates from ROI-local to image space
    int img_x = id_x + roiX;
    int img_y = id_y + roiY;

    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    bool dropped = false;
    for (int i = 0; i < numBoxes; i++) {
        int boxIdx = boxStartOffset + i;
        // Compare against anchor boxes in image space
        if (img_x >= anchorBoxInfoTensor[boxIdx].lt.x &&
            img_x <= anchorBoxInfoTensor[boxIdx].rb.x &&
            img_y >= anchorBoxInfoTensor[boxIdx].lt.y &&
            img_y <= anchorBoxInfoTensor[boxIdx].rb.y) {
            dstPtr[dstIdx] = (std::is_same<T, Rpp8s>::value) ? -128 : 0;
            dstPtr[dstIdx + 1] = (std::is_same<T, Rpp8s>::value) ? -128 : 0;
            dstPtr[dstIdx + 2] = (std::is_same<T, Rpp8s>::value) ? -128 : 0;
            dropped = true;
            break;
        }
    }

    if (!dropped && srcPtr != nullptr) {
        uint srcIdx = (id_z * srcStridesNH.x) + (img_y * srcStridesNH.y) + img_x * 3;
        dstPtr[dstIdx] = srcPtr[srcIdx];
        dstPtr[dstIdx + 1] = srcPtr[srcIdx + 1];
        dstPtr[dstIdx + 2] = srcPtr[srcIdx + 2];
    }
}

template <typename T>
__global__ void coarse_dropout_pln_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                              uint3 dstStridesNCH, RpptRoiLtrb* anchorBoxInfoTensor,
                                              Rpp32u* numBoxesTensor, RpptROIPtr roiTensorPtrSrc,
                                              int maxBoxesPerImage) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) ||
        (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    // Clamp numBoxes to maxBoxesPerImage to prevent buffer overflow
    Rpp32u numBoxes = min(numBoxesTensor[id_z], static_cast<Rpp32u>(maxBoxesPerImage));
    int boxStartOffset = id_z * maxBoxesPerImage;

    // Get ROI origin for coordinate conversion from image space to ROI-local space
    int roiX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiY = roiTensorPtrSrc[id_z].xywhROI.xy.y;

    // Convert thread coordinates from ROI-local to image space
    int img_x = id_x + roiX;
    int img_y = id_y + roiY;

    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    bool dropped = false;
    for (int i = 0; i < numBoxes; i++) {
        int boxIdx = boxStartOffset + i;
        // Compare against anchor boxes in image space
        if (img_x >= anchorBoxInfoTensor[boxIdx].lt.x &&
            img_x <= anchorBoxInfoTensor[boxIdx].rb.x &&
            img_y >= anchorBoxInfoTensor[boxIdx].lt.y &&
            img_y <= anchorBoxInfoTensor[boxIdx].rb.y) {
            dstPtr[dstIdx] = (std::is_same<T, Rpp8s>::value) ? -128 : 0;
            dropped = true;
            break;
        }
    }

    if (!dropped && srcPtr != nullptr) {
        uint srcIdx = (id_z * srcStridesNCH.x) + (img_y * srcStridesNCH.z) + img_x;
        dstPtr[dstIdx] = srcPtr[srcIdx];
    }
}

template <typename T>
__global__ void coarse_dropout_pln3_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                               uint3 dstStridesNCH,
                                               RpptRoiLtrb* anchorBoxInfoTensor,
                                               Rpp32u* numBoxesTensor, RpptROIPtr roiTensorPtrSrc,
                                               int maxBoxesPerImage) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) ||
        (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    // Clamp numBoxes to maxBoxesPerImage to prevent buffer overflow
    Rpp32u numBoxes = min(numBoxesTensor[id_z], static_cast<Rpp32u>(maxBoxesPerImage));
    int boxStartOffset = id_z * maxBoxesPerImage;

    // Get ROI origin for coordinate conversion from image space to ROI-local space
    int roiX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiY = roiTensorPtrSrc[id_z].xywhROI.xy.y;

    // Convert thread coordinates from ROI-local to image space
    int img_x = id_x + roiX;
    int img_y = id_y + roiY;

    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    bool dropped = false;
    for (int i = 0; i < numBoxes; i++) {
        int boxIdx = boxStartOffset + i;
        // Compare against anchor boxes in image space
        if (img_x >= anchorBoxInfoTensor[boxIdx].lt.x &&
            img_x <= anchorBoxInfoTensor[boxIdx].rb.x &&
            img_y >= anchorBoxInfoTensor[boxIdx].lt.y &&
            img_y <= anchorBoxInfoTensor[boxIdx].rb.y) {
            dstPtr[dstIdx] = (std::is_same<T, Rpp8s>::value) ? -128 : 0;
            dstPtr[dstIdx + dstStridesNCH.y] = (std::is_same<T, Rpp8s>::value) ? -128 : 0;
            dstPtr[dstIdx + 2 * dstStridesNCH.y] = (std::is_same<T, Rpp8s>::value) ? -128 : 0;
            dropped = true;
            break;
        }
    }

    if (!dropped && srcPtr != nullptr) {
        uint srcIdx = (id_z * srcStridesNCH.x) + (img_y * srcStridesNCH.z) + img_x;
        dstPtr[dstIdx] = srcPtr[srcIdx];
        dstPtr[dstIdx + dstStridesNCH.y] = srcPtr[srcIdx + srcStridesNCH.y];
        dstPtr[dstIdx + 2 * dstStridesNCH.y] = srcPtr[srcIdx + 2 * srcStridesNCH.y];
    }
}

// -------------------- Set 1 - Kernel Executors --------------------
template <typename T>
RppStatus hip_exec_coarse_dropout_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                         RpptDescPtr dstDescPtr, RpptRoiLtrb* anchorBoxInfoTensor,
                                         Rpp32u* numBoxesTensor, Rpp32u maxBoxesPerImage,
                                         RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                         rpp::Handle& handle) {
    if (roiType == RpptRoiType::LTRB) hip_exec_roi_conversion_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = dstDescPtr->w;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if (dstDescPtr->layout == RpptLayout::NHWC) {
        // NHWC layout only supports 3-channel (RGB) images
        if (dstDescPtr->c != 3) {
            return RPP_ERROR_NOT_IMPLEMENTED;
        }

        // Pass-through source for pixels outside every anchor box, read at the ROI's absolute
        // offset. Left null when a preceding conversion kernel already placed the correct
        // pass-through pixel in dstPtr, so the dropout kernel below only needs to punch in boxes.
        T* passthroughSrcPtr = nullptr;
        uint2 passthroughSrcStridesNH = make_uint2(0, 0);

        // if src layout is NHWC, dropout kernel copies src to dst per-pixel (absolute source frame)
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
            globalThreads_x = dstDescPtr->w;
            RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));
        }

        hipLaunchKernelGGL(coarse_dropout_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                           handle.GetStream(), passthroughSrcPtr, passthroughSrcStridesNH, dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           anchorBoxInfoTensor, numBoxesTensor, roiTensorPtrSrc, maxBoxesPerImage);
    } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NCHW) && dstDescPtr->c == 1) {
        hipLaunchKernelGGL(coarse_dropout_pln_hip_tensor,
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
                           anchorBoxInfoTensor, numBoxesTensor, roiTensorPtrSrc, maxBoxesPerImage);
    } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NCHW) && dstDescPtr->c == 3) {
        hipLaunchKernelGGL(coarse_dropout_pln3_hip_tensor,
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
                           anchorBoxInfoTensor, numBoxesTensor, roiTensorPtrSrc, maxBoxesPerImage);
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
            RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));
            globalThreads_x = dstDescPtr->w;
            // dstPtr already holds the correct pass-through pixel (written by the convert
            // kernel above in the absolute-source/packed-destination frame); pass a null
            // source so this kernel only punches in the dropout boxes.
            hipLaunchKernelGGL(
                coarse_dropout_pln3_hip_tensor,
                dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                     ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                     ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(),
                static_cast<T*>(nullptr), make_uint3(0, 0, 0), dstPtr,
                make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                           dstDescPtr->strides.hStride),
                anchorBoxInfoTensor, numBoxesTensor, roiTensorPtrSrc, maxBoxesPerImage);
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_coarse_dropout_tensor<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*, RpptDescPtr,
                                                         RpptRoiLtrb*, Rpp32u*, Rpp32u, RpptROIPtr,
                                                         RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_coarse_dropout_tensor<half>(half*, RpptDescPtr, half*, RpptDescPtr,
                                                        RpptRoiLtrb*, Rpp32u*, Rpp32u, RpptROIPtr,
                                                        RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_coarse_dropout_tensor<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*,
                                                          RpptDescPtr, RpptRoiLtrb*, Rpp32u*,
                                                          Rpp32u, RpptROIPtr, RpptRoiType,
                                                          rpp::Handle&);

template RppStatus hip_exec_coarse_dropout_tensor<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*, RpptDescPtr,
                                                         RpptRoiLtrb*, Rpp32u*, Rpp32u, RpptROIPtr,
                                                         RpptRoiType, rpp::Handle&);
