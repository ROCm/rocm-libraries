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
template <typename T>
__global__ void coarse_dropout_pkd_hip_tensor(T* dstPtr, uint2 dstStridesNH,
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

    for (int i = 0; i < numBoxes; i++) {
        int boxIdx = boxStartOffset + i;
        // Compare against anchor boxes in image space
        if (img_x >= anchorBoxInfoTensor[boxIdx].lt.x &&
            img_x <= anchorBoxInfoTensor[boxIdx].rb.x &&
            img_y >= anchorBoxInfoTensor[boxIdx].lt.y &&
            img_y <= anchorBoxInfoTensor[boxIdx].rb.y) {
            uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
            dstPtr[dstIdx] = (std::is_same<T, Rpp8s>::value) ? -128 : 0;
            dstPtr[dstIdx + 1] = (std::is_same<T, Rpp8s>::value) ? -128 : 0;
            dstPtr[dstIdx + 2] = (std::is_same<T, Rpp8s>::value) ? -128 : 0;
            break;
        }
    }
}

template <typename T>
__global__ void coarse_dropout_pln_hip_tensor(T* dstPtr, uint3 dstStridesNCH,
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

    for (int i = 0; i < numBoxes; i++) {
        int boxIdx = boxStartOffset + i;
        // Compare against anchor boxes in image space
        if (img_x >= anchorBoxInfoTensor[boxIdx].lt.x &&
            img_x <= anchorBoxInfoTensor[boxIdx].rb.x &&
            img_y >= anchorBoxInfoTensor[boxIdx].lt.y &&
            img_y <= anchorBoxInfoTensor[boxIdx].rb.y) {
            uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
            dstPtr[dstIdx] = (std::is_same<T, Rpp8s>::value) ? -128 : 0;
            break;
        }
    }
}

template <typename T>
__global__ void coarse_dropout_pln3_hip_tensor(T* dstPtr, uint3 dstStridesNCH,
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

    for (int i = 0; i < numBoxes; i++) {
        int boxIdx = boxStartOffset + i;
        // Compare against anchor boxes in image space
        if (img_x >= anchorBoxInfoTensor[boxIdx].lt.x &&
            img_x <= anchorBoxInfoTensor[boxIdx].rb.x &&
            img_y >= anchorBoxInfoTensor[boxIdx].lt.y &&
            img_y <= anchorBoxInfoTensor[boxIdx].rb.y) {
            uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
            dstPtr[dstIdx] = (std::is_same<T, Rpp8s>::value) ? -128 : 0;
            dstPtr[dstIdx + dstStridesNCH.y] = (std::is_same<T, Rpp8s>::value) ? -128 : 0;
            dstPtr[dstIdx + 2 * dstStridesNCH.y] = (std::is_same<T, Rpp8s>::value) ? -128 : 0;
            break;
        }
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

        // if src layout is NHWC, copy src to dst
        if (srcDescPtr->layout == RpptLayout::NHWC) {
            RPP_HIP_RETURN_IF_ERROR(hipMemcpyAsync(
                dstPtr, srcPtr,
                static_cast<size_t>(srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T)),
                hipMemcpyDeviceToDevice, handle.GetStream()));
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
            globalThreads_x = dstDescPtr->w;
            RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));
        }

        hipLaunchKernelGGL(coarse_dropout_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                           handle.GetStream(), dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           anchorBoxInfoTensor, numBoxesTensor, roiTensorPtrSrc, maxBoxesPerImage);
    } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NCHW) && dstDescPtr->c == 1) {
        RPP_HIP_RETURN_IF_ERROR(hipMemcpyAsync(
            dstPtr, srcPtr,
            static_cast<size_t>(srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T)),
            hipMemcpyDeviceToDevice, handle.GetStream()));
        RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));
        hipLaunchKernelGGL(coarse_dropout_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                           handle.GetStream(), dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                      dstDescPtr->strides.hStride),
                           anchorBoxInfoTensor, numBoxesTensor, roiTensorPtrSrc, maxBoxesPerImage);
    } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NCHW) && dstDescPtr->c == 3) {
        RPP_HIP_RETURN_IF_ERROR(hipMemcpyAsync(
            dstPtr, srcPtr,
            static_cast<size_t>(srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T)),
            hipMemcpyDeviceToDevice, handle.GetStream()));
        RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));
        hipLaunchKernelGGL(coarse_dropout_pln3_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                           handle.GetStream(), dstPtr,
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
            hipLaunchKernelGGL(coarse_dropout_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                    ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                          dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor, numBoxesTensor, roiTensorPtrSrc,
                               maxBoxesPerImage);
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
