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

#include "rppt_tensor_geometric_augmentations.h"

#include "host_tensor_executors.hpp"
#include "rppdefs.h"
#include "rppt_validate.hpp"

#ifdef GPU_SUPPORT
#include "hip_tensor_executors.hpp"
#endif  // GPU_SUPPORT

#if __APPLE__
#define sincosf __sincosf
#endif

/******************** crop ********************/

RppStatus rppt_crop(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                    RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                    rppHandle_t rppHandle, RppBackend executionBackend) {
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcDescPtr->n != dstDescPtr->n) return RPP_ERROR_INVALID_ARGUMENTS;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

        if (srcDescPtr->n == 1 && dstDescPtr->n == 1) {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                crop_u8_u8_host_single_image(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    roiTensorPtrSrc, roiType, layoutParams, rpp::deref(rppHandle));
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                crop_f16_f16_host_single_image(
                    (Rpp16f*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes), srcDescPtr,
                    (Rpp16f*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), dstDescPtr,
                    roiTensorPtrSrc, roiType, layoutParams, rpp::deref(rppHandle));
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                crop_f32_f32_host_single_image(
                    (Rpp32f*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes), srcDescPtr,
                    (Rpp32f*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), dstDescPtr,
                    roiTensorPtrSrc, roiType, layoutParams, rpp::deref(rppHandle));
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                crop_i8_i8_host_single_image(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    roiTensorPtrSrc, roiType, layoutParams, rpp::deref(rppHandle));
            } else
                return RPP_ERROR_NOT_IMPLEMENTED;
        } else {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                crop_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                crop_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) +
                                                                   srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) +
                                                                   dstDescPtr->offsetInBytes),
                                         dstDescPtr, roiTensorPtrSrc, roiType, layoutParams,
                                         handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                crop_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                                                   srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                                                   dstDescPtr->offsetInBytes),
                                         dstDescPtr, roiTensorPtrSrc, roiType, layoutParams,
                                         handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                crop_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else
                return RPP_ERROR_NOT_IMPLEMENTED;
        }

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if (srcDescPtr->n == 1 && dstDescPtr->n == 1) {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                hip_exec_crop_single_image(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                hip_exec_crop_single_image(
                    (half*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes), srcDescPtr,
                    (half*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), dstDescPtr,
                    roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                hip_exec_crop_single_image(
                    (Rpp32f*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes), srcDescPtr,
                    (Rpp32f*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), dstDescPtr,
                    roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                hip_exec_crop_single_image(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
            } else
                return RPP_ERROR_NOT_IMPLEMENTED;
        } else {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                hip_exec_crop_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                     dstDescPtr, roiTensorPtrSrc, roiType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                hip_exec_crop_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) +
                                                             srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) +
                                                             dstDescPtr->offsetInBytes),
                                     dstDescPtr, roiTensorPtrSrc, roiType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                hip_exec_crop_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                                               srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                                               dstDescPtr->offsetInBytes),
                                     dstDescPtr, roiTensorPtrSrc, roiType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                hip_exec_crop_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                     dstDescPtr, roiTensorPtrSrc, roiType, handle);
            } else
                return RPP_ERROR_NOT_IMPLEMENTED;
        }

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** crop_and_patch ********************/

RppStatus rppt_crop_and_patch(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptDescPtr srcDescPtr,
                              RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrDst,
                              RpptROIPtr cropRoi, RpptROIPtr patchRoi, RpptRoiType roiType,
                              rppHandle_t rppHandle, RppBackend executionBackend) {
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            crop_and_patch_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                roiTensorPtrDst, cropRoi, patchRoi, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            crop_and_patch_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, roiTensorPtrDst, cropRoi, patchRoi, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            crop_and_patch_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, roiTensorPtrDst, cropRoi, patchRoi, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            crop_and_patch_i8_i8_host_tensor(
                static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                roiTensorPtrDst, cropRoi, patchRoi, roiType, layoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_crop_and_patch_tensor(
                static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                roiTensorPtrDst, cropRoi, patchRoi, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_crop_and_patch_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, roiTensorPtrDst, cropRoi, patchRoi, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_crop_and_patch_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, roiTensorPtrDst, cropRoi, patchRoi, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_crop_and_patch_tensor(
                static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                roiTensorPtrDst, cropRoi, patchRoi, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** crop_mirror_normalize ********************/

RppStatus rppt_crop_mirror_normalize(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32f* offsetTensor,
                                     Rpp32f* multiplierTensor, Rpp32u* mirrorTensor,
                                     RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                     rppHandle_t rppHandle, RppBackend executionBackend) {
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            crop_mirror_normalize_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, offsetTensor,
                multiplierTensor, mirrorTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            crop_mirror_normalize_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, offsetTensor, multiplierTensor, mirrorTensor, roiTensorPtrSrc, roiType,
                layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            crop_mirror_normalize_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, offsetTensor, multiplierTensor, mirrorTensor, roiTensorPtrSrc, roiType,
                layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            crop_mirror_normalize_i8_i8_host_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, offsetTensor,
                multiplierTensor, mirrorTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::U8) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            crop_mirror_normalize_u8_f32_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, offsetTensor, multiplierTensor, mirrorTensor, roiTensorPtrSrc, roiType,
                layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::U8) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            crop_mirror_normalize_u8_f16_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, offsetTensor, multiplierTensor, mirrorTensor, roiTensorPtrSrc, roiType,
                layoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_crop_mirror_normalize_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, offsetTensor,
                multiplierTensor, mirrorTensor, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_crop_mirror_normalize_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, offsetTensor, multiplierTensor, mirrorTensor, roiTensorPtrSrc, roiType,
                handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_crop_mirror_normalize_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, offsetTensor, multiplierTensor, mirrorTensor, roiTensorPtrSrc, roiType,
                handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_crop_mirror_normalize_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, offsetTensor,
                multiplierTensor, mirrorTensor, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::U8) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_crop_mirror_normalize_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, offsetTensor, multiplierTensor, mirrorTensor, roiTensorPtrSrc, roiType,
                handle);
        } else if ((srcDescPtr->dataType == RpptDataType::U8) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_crop_mirror_normalize_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, offsetTensor, multiplierTensor, mirrorTensor, roiTensorPtrSrc, roiType,
                handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** warp_affine ********************/

RppStatus rppt_warp_affine(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                           RpptDescPtr dstDescPtr, Rpp32f* affineTensor,
                           RpptInterpolationType interpolationType, RpptROIPtr roiTensorPtrSrc,
                           RpptRoiType roiType, rppHandle_t rppHandle,
                           RppBackend executionBackend) {
    if ((interpolationType != RpptInterpolationType::BILINEAR) &&
        (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR))
        return RPP_ERROR_NOT_IMPLEMENTED;
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if (interpolationType == RpptInterpolationType::NEAREST_NEIGHBOR) {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                warp_affine_nn_u8_u8_host_tensor(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    affineTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                warp_affine_nn_f16_f16_host_tensor(
                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, affineTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                warp_affine_nn_f32_f32_host_tensor(
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, affineTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                warp_affine_nn_i8_i8_host_tensor(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    affineTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            }
        } else if (interpolationType == RpptInterpolationType::BILINEAR) {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                warp_affine_bilinear_u8_u8_host_tensor(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    affineTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                warp_affine_bilinear_f16_f16_host_tensor(
                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, affineTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                warp_affine_bilinear_f32_f32_host_tensor(
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, affineTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                warp_affine_bilinear_i8_i8_host_tensor(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    affineTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            }
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_warp_affine_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, affineTensor,
                interpolationType, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_warp_affine_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_warp_affine_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_warp_affine_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, affineTensor,
                interpolationType, roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** flip ********************/

RppStatus rppt_flip(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                    RpptDescPtr dstDescPtr, Rpp32u* horizontalTensor, Rpp32u* verticalTensor,
                    RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle,
                    RppBackend executionBackend) {
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcDescPtr->n != dstDescPtr->n) return RPP_ERROR_INVALID_ARGUMENTS;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if (srcDescPtr->n == 1 && dstDescPtr->n == 1) {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                flip_u8_u8_host_single_image(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    horizontalTensor, verticalTensor, roiTensorPtrSrc, roiType, layoutParams,
                    rpp::deref(rppHandle));
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                flip_f16_f16_host_single_image(
                    (Rpp16f*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes), srcDescPtr,
                    (Rpp16f*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), dstDescPtr,
                    horizontalTensor, verticalTensor, roiTensorPtrSrc, roiType, layoutParams,
                    rpp::deref(rppHandle));
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                flip_f32_f32_host_single_image(
                    (Rpp32f*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes), srcDescPtr,
                    (Rpp32f*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), dstDescPtr,
                    horizontalTensor, verticalTensor, roiTensorPtrSrc, roiType, layoutParams,
                    rpp::deref(rppHandle));
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                flip_i8_i8_host_single_image(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    horizontalTensor, verticalTensor, roiTensorPtrSrc, roiType, layoutParams,
                    rpp::deref(rppHandle));
            } else
                return RPP_ERROR_NOT_IMPLEMENTED;
        } else {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                flip_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr, horizontalTensor, verticalTensor,
                                       roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                flip_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) +
                                                                   srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) +
                                                                   dstDescPtr->offsetInBytes),
                                         dstDescPtr, horizontalTensor, verticalTensor,
                                         roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                flip_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                                                   srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                                                   dstDescPtr->offsetInBytes),
                                         dstDescPtr, horizontalTensor, verticalTensor,
                                         roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                flip_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr, horizontalTensor, verticalTensor,
                                       roiTensorPtrSrc, roiType, layoutParams, handle);
            } else
                return RPP_ERROR_NOT_IMPLEMENTED;
        }

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if (srcDescPtr->n == 1 && dstDescPtr->n == 1) {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                hip_exec_flip_single_image(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                           srcDescPtr,
                                           static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                           dstDescPtr, horizontalTensor, verticalTensor,
                                           roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                hip_exec_flip_single_image(
                    (half*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes), srcDescPtr,
                    (half*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), dstDescPtr,
                    horizontalTensor, verticalTensor, roiTensorPtrSrc, roiType,
                    rpp::deref(rppHandle));
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                hip_exec_flip_single_image(
                    (Rpp32f*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes), srcDescPtr,
                    (Rpp32f*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), dstDescPtr,
                    horizontalTensor, verticalTensor, roiTensorPtrSrc, roiType,
                    rpp::deref(rppHandle));
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                hip_exec_flip_single_image(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                           srcDescPtr,
                                           static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                           dstDescPtr, horizontalTensor, verticalTensor,
                                           roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
            } else
                return RPP_ERROR_NOT_IMPLEMENTED;
        } else {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                hip_exec_flip_tensor(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    horizontalTensor, verticalTensor, roiTensorPtrSrc, roiType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                hip_exec_flip_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) +
                                                             srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) +
                                                             dstDescPtr->offsetInBytes),
                                     dstDescPtr, horizontalTensor, verticalTensor, roiTensorPtrSrc,
                                     roiType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                hip_exec_flip_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                                               srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                                               dstDescPtr->offsetInBytes),
                                     dstDescPtr, horizontalTensor, verticalTensor, roiTensorPtrSrc,
                                     roiType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                hip_exec_flip_tensor(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    horizontalTensor, verticalTensor, roiTensorPtrSrc, roiType, handle);
            } else
                return RPP_ERROR_NOT_IMPLEMENTED;
        }

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** resize ********************/

RppStatus rppt_resize(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                      RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSizes,
                      RpptInterpolationType interpolationType, RpptROIPtr roiTensorPtrSrc,
                      RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend) {
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcDescPtr->n != dstDescPtr->n) return RPP_ERROR_INVALID_ARGUMENTS;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams srcLayoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if (interpolationType == RpptInterpolationType::NEAREST_NEIGHBOR) {
            if (srcDescPtr->n == 1 && dstDescPtr->n == 1) {
                if ((srcDescPtr->dataType == RpptDataType::U8) &&
                    (dstDescPtr->dataType == RpptDataType::U8)) {
                    resize_nn_u8_u8_host_single_image(
                        static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                        static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                        dstImgSizes, roiTensorPtrSrc, roiType, srcLayoutParams,
                        rpp::deref(rppHandle));
                } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                           (dstDescPtr->dataType == RpptDataType::F16)) {
                    resize_nn_f16_f16_host_single_image(
                        (Rpp16f*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                        srcDescPtr,
                        (Rpp16f*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                        dstDescPtr, dstImgSizes, roiTensorPtrSrc, roiType, srcLayoutParams,
                        rpp::deref(rppHandle));
                } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                           (dstDescPtr->dataType == RpptDataType::F32)) {
                    resize_nn_f32_f32_host_single_image(
                        (Rpp32f*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                        srcDescPtr,
                        (Rpp32f*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                        dstDescPtr, dstImgSizes, roiTensorPtrSrc, roiType, srcLayoutParams,
                        rpp::deref(rppHandle));
                } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                           (dstDescPtr->dataType == RpptDataType::I8)) {
                    resize_nn_i8_i8_host_single_image(
                        static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                        static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                        dstImgSizes, roiTensorPtrSrc, roiType, srcLayoutParams,
                        rpp::deref(rppHandle));
                } else
                    return RPP_ERROR_NOT_IMPLEMENTED;
            } else {
                if ((srcDescPtr->dataType == RpptDataType::U8) &&
                    (dstDescPtr->dataType == RpptDataType::U8)) {
                    resize_nn_u8_u8_host_tensor(
                        static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                        static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                        dstImgSizes, roiTensorPtrSrc, roiType, srcLayoutParams, handle);
                } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                           (dstDescPtr->dataType == RpptDataType::F16)) {
                    resize_nn_f16_f16_host_tensor(
                        reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) +
                                                  srcDescPtr->offsetInBytes),
                        srcDescPtr,
                        reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) +
                                                  dstDescPtr->offsetInBytes),
                        dstDescPtr, dstImgSizes, roiTensorPtrSrc, roiType, srcLayoutParams, handle);
                } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                           (dstDescPtr->dataType == RpptDataType::F32)) {
                    resize_nn_f32_f32_host_tensor(
                        reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                                  srcDescPtr->offsetInBytes),
                        srcDescPtr,
                        reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                                  dstDescPtr->offsetInBytes),
                        dstDescPtr, dstImgSizes, roiTensorPtrSrc, roiType, srcLayoutParams, handle);
                } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                           (dstDescPtr->dataType == RpptDataType::I8)) {
                    resize_nn_i8_i8_host_tensor(
                        static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                        static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                        dstImgSizes, roiTensorPtrSrc, roiType, srcLayoutParams, handle);
                } else
                    return RPP_ERROR_NOT_IMPLEMENTED;
            }
        } else if (interpolationType == RpptInterpolationType::BILINEAR) {
            if (srcDescPtr->n == 1 && dstDescPtr->n == 1) {
                if ((srcDescPtr->dataType == RpptDataType::U8) &&
                    (dstDescPtr->dataType == RpptDataType::U8)) {
                    resize_bilinear_u8_u8_host_single_image(
                        static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                        static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                        dstImgSizes, roiTensorPtrSrc, roiType, srcLayoutParams,
                        rpp::deref(rppHandle));
                } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                           (dstDescPtr->dataType == RpptDataType::F16)) {
                    resize_bilinear_f16_f16_host_single_image(
                        (Rpp16f*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                        srcDescPtr,
                        (Rpp16f*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                        dstDescPtr, dstImgSizes, roiTensorPtrSrc, roiType, srcLayoutParams,
                        rpp::deref(rppHandle));
                } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                           (dstDescPtr->dataType == RpptDataType::F32)) {
                    resize_bilinear_f32_f32_host_single_image(
                        (Rpp32f*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                        srcDescPtr,
                        (Rpp32f*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                        dstDescPtr, dstImgSizes, roiTensorPtrSrc, roiType, srcLayoutParams,
                        rpp::deref(rppHandle));
                } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                           (dstDescPtr->dataType == RpptDataType::I8)) {
                    resize_bilinear_i8_i8_host_single_image(
                        static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                        static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                        dstImgSizes, roiTensorPtrSrc, roiType, srcLayoutParams,
                        rpp::deref(rppHandle));
                } else
                    return RPP_ERROR_NOT_IMPLEMENTED;
            } else {
                if ((srcDescPtr->dataType == RpptDataType::U8) &&
                    (dstDescPtr->dataType == RpptDataType::U8)) {
                    resize_bilinear_u8_u8_host_tensor(
                        static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                        static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                        dstImgSizes, roiTensorPtrSrc, roiType, srcLayoutParams, handle);
                } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                           (dstDescPtr->dataType == RpptDataType::F16)) {
                    resize_bilinear_f16_f16_host_tensor(
                        reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) +
                                                  srcDescPtr->offsetInBytes),
                        srcDescPtr,
                        reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) +
                                                  dstDescPtr->offsetInBytes),
                        dstDescPtr, dstImgSizes, roiTensorPtrSrc, roiType, srcLayoutParams, handle);
                } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                           (dstDescPtr->dataType == RpptDataType::F32)) {
                    resize_bilinear_f32_f32_host_tensor(
                        reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                                  srcDescPtr->offsetInBytes),
                        srcDescPtr,
                        reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                                  dstDescPtr->offsetInBytes),
                        dstDescPtr, dstImgSizes, roiTensorPtrSrc, roiType, srcLayoutParams, handle);
                } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                           (dstDescPtr->dataType == RpptDataType::I8)) {
                    resize_bilinear_i8_i8_host_tensor(
                        static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                        static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                        dstImgSizes, roiTensorPtrSrc, roiType, srcLayoutParams, handle);
                } else
                    return RPP_ERROR_NOT_IMPLEMENTED;
            }
        } else {
            RpptDesc tempDesc;
            tempDesc = *srcDescPtr;
            RpptDescPtr tempDescPtr = &tempDesc;
            tempDescPtr->h = dstDescPtr->h;
            tempDescPtr->strides.nStride = srcDescPtr->w * dstDescPtr->h * srcDescPtr->c;

            // The channel stride changes with the change in the height for PLN images
            if (srcDescPtr->layout == RpptLayout::NCHW)
                tempDescPtr->strides.cStride = srcDescPtr->w * dstDescPtr->h;

            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                resize_separable_host_tensor(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    handle.GetInitHandle()->mem.mcpu.scratchBufferHost, tempDescPtr, dstImgSizes,
                    roiTensorPtrSrc, roiType, srcLayoutParams, interpolationType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                resize_separable_host_tensor(
                    static_cast<Rpp32f*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp32f*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    handle.GetInitHandle()->mem.mcpu.scratchBufferHost, tempDescPtr, dstImgSizes,
                    roiTensorPtrSrc, roiType, srcLayoutParams, interpolationType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                resize_separable_host_tensor(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    handle.GetInitHandle()->mem.mcpu.scratchBufferHost, tempDescPtr, dstImgSizes,
                    roiTensorPtrSrc, roiType, srcLayoutParams, interpolationType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                resize_separable_host_tensor(
                    static_cast<Rpp16f*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp16f*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    handle.GetInitHandle()->mem.mcpu.scratchBufferHost, tempDescPtr, dstImgSizes,
                    roiTensorPtrSrc, roiType, srcLayoutParams, interpolationType, handle);
            } else
                return RPP_ERROR_NOT_IMPLEMENTED;
        }

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if (srcDescPtr->n == 1 && dstDescPtr->n == 1) {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                hip_exec_resize_single_image(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    dstImgSizes, interpolationType, roiTensorPtrSrc, roiType,
                    rpp::deref(rppHandle));
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                hip_exec_resize_single_image(
                    (half*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes), srcDescPtr,
                    (half*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), dstDescPtr,
                    dstImgSizes, interpolationType, roiTensorPtrSrc, roiType,
                    rpp::deref(rppHandle));
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                hip_exec_resize_single_image(
                    (Rpp32f*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes), srcDescPtr,
                    (Rpp32f*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), dstDescPtr,
                    dstImgSizes, interpolationType, roiTensorPtrSrc, roiType,
                    rpp::deref(rppHandle));
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                hip_exec_resize_single_image(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    dstImgSizes, interpolationType, roiTensorPtrSrc, roiType,
                    rpp::deref(rppHandle));
            } else
                return RPP_ERROR_NOT_IMPLEMENTED;
        } else {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                hip_exec_resize_tensor(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    dstImgSizes, interpolationType, roiTensorPtrSrc, roiType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                hip_exec_resize_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) +
                                                               srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) +
                                                               dstDescPtr->offsetInBytes),
                                       dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc,
                                       roiType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                hip_exec_resize_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                                                 srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                                                 dstDescPtr->offsetInBytes),
                                       dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc,
                                       roiType, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                hip_exec_resize_tensor(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    dstImgSizes, interpolationType, roiTensorPtrSrc, roiType, handle);
            } else
                return RPP_ERROR_NOT_IMPLEMENTED;
        }

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** resize_mirror_normalize ********************/

RppStatus rppt_resize_mirror_normalize(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                                       RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSizes,
                                       RpptInterpolationType interpolationType, Rpp32f* meanTensor,
                                       Rpp32f* stdDevTensor, Rpp32u* mirrorTensor,
                                       RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                       rppHandle_t rppHandle, RppBackend executionBackend) {
    if (interpolationType != RpptInterpolationType::BILINEAR) return RPP_ERROR_NOT_IMPLEMENTED;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams srcLayoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            resize_mirror_normalize_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, dstImgSizes,
                meanTensor, stdDevTensor, mirrorTensor, roiTensorPtrSrc, roiType, srcLayoutParams,
                handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            resize_mirror_normalize_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, dstImgSizes, meanTensor, stdDevTensor, mirrorTensor, roiTensorPtrSrc,
                roiType, srcLayoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            resize_mirror_normalize_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, dstImgSizes, meanTensor, stdDevTensor, mirrorTensor, roiTensorPtrSrc,
                roiType, srcLayoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            resize_mirror_normalize_i8_i8_host_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, dstImgSizes,
                meanTensor, stdDevTensor, mirrorTensor, roiTensorPtrSrc, roiType, srcLayoutParams,
                handle);
        } else if ((srcDescPtr->dataType == RpptDataType::U8) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            resize_mirror_normalize_u8_f32_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, dstImgSizes, meanTensor, stdDevTensor, mirrorTensor, roiTensorPtrSrc,
                roiType, srcLayoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::U8) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            resize_mirror_normalize_u8_f16_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, dstImgSizes, meanTensor, stdDevTensor, mirrorTensor, roiTensorPtrSrc,
                roiType, srcLayoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_resize_mirror_normalize_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, dstImgSizes,
                interpolationType, meanTensor, stdDevTensor, mirrorTensor, roiTensorPtrSrc, roiType,
                handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_resize_mirror_normalize_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, dstImgSizes, interpolationType, meanTensor, stdDevTensor, mirrorTensor,
                roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_resize_mirror_normalize_tensor(
                (Rpp32f*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes), srcDescPtr,
                (Rpp32f*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), dstDescPtr,
                dstImgSizes, interpolationType, meanTensor, stdDevTensor, mirrorTensor,
                roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_resize_mirror_normalize_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, dstImgSizes,
                interpolationType, meanTensor, stdDevTensor, mirrorTensor, roiTensorPtrSrc, roiType,
                handle);
        } else if ((srcDescPtr->dataType == RpptDataType::U8) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_resize_mirror_normalize_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                (Rpp32f*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), dstDescPtr,
                dstImgSizes, interpolationType, meanTensor, stdDevTensor, mirrorTensor,
                roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::U8) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_resize_mirror_normalize_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                (half*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), dstDescPtr,
                dstImgSizes, interpolationType, meanTensor, stdDevTensor, mirrorTensor,
                roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** resize_crop_mirror ********************/

RppStatus rppt_resize_crop_mirror(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                                  RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSizes,
                                  RpptInterpolationType interpolationType, Rpp32u* mirrorTensor,
                                  RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                  rppHandle_t rppHandle, RppBackend executionBackend) {
    if (interpolationType != RpptInterpolationType::BILINEAR) return RPP_ERROR_NOT_IMPLEMENTED;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams srcLayoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            resize_crop_mirror_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, dstImgSizes,
                mirrorTensor, roiTensorPtrSrc, roiType, srcLayoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            resize_crop_mirror_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, dstImgSizes, mirrorTensor, roiTensorPtrSrc, roiType, srcLayoutParams,
                handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            resize_crop_mirror_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, dstImgSizes, mirrorTensor, roiTensorPtrSrc, roiType, srcLayoutParams,
                handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            resize_crop_mirror_i8_i8_host_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, dstImgSizes,
                mirrorTensor, roiTensorPtrSrc, roiType, srcLayoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_resize_crop_mirror_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, dstImgSizes,
                mirrorTensor, interpolationType, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_resize_crop_mirror_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, dstImgSizes, mirrorTensor, interpolationType, roiTensorPtrSrc, roiType,
                handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_resize_crop_mirror_tensor(
                (Rpp32f*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes), srcDescPtr,
                (Rpp32f*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), dstDescPtr,
                dstImgSizes, mirrorTensor, interpolationType, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_resize_crop_mirror_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, dstImgSizes,
                mirrorTensor, interpolationType, roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** rotate ********************/

RppStatus rppt_rotate(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                      RpptDescPtr dstDescPtr, Rpp32f* angle,
                      RpptInterpolationType interpolationType, RpptROIPtr roiTensorPtrSrc,
                      RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend) {
    if ((interpolationType != RpptInterpolationType::BILINEAR) &&
        (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR))
        return RPP_ERROR_NOT_IMPLEMENTED;
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

        // Compute affine transformation matrix from rotate angle
        Rpp32f* affineTensor = handle.GetInitHandle()->mem.mcpu.scratchBufferHost;
        for (int idx = 0; idx < srcDescPtr->n; idx++) {
            Rpp32f angleInRad = angle[idx] * PI_OVER_180;
            Rpp32f alpha, beta;
            sincosf(angleInRad, &beta, &alpha);
            ((Rpp32f6*)affineTensor)[idx] = {alpha, -beta, 0, beta, alpha, 0};
        }

        if (interpolationType == RpptInterpolationType::NEAREST_NEIGHBOR) {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                warp_affine_nn_u8_u8_host_tensor(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    affineTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                warp_affine_nn_f16_f16_host_tensor(
                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, affineTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                warp_affine_nn_f32_f32_host_tensor(
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, affineTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                warp_affine_nn_i8_i8_host_tensor(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    affineTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            }
        } else if (interpolationType == RpptInterpolationType::BILINEAR) {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                warp_affine_bilinear_u8_u8_host_tensor(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    affineTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                warp_affine_bilinear_f16_f16_host_tensor(
                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, affineTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                warp_affine_bilinear_f32_f32_host_tensor(
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, affineTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                warp_affine_bilinear_i8_i8_host_tensor(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    affineTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            }
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        // Compute affine transformation matrix from rotate angle
        Rpp32f* affineTensor = handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem;
        for (int idx = 0; idx < srcDescPtr->n; idx++) {
            Rpp32f angleInRad = angle[idx] * PI_OVER_180;
            Rpp32f alpha, beta;
            sincosf(angleInRad, &beta, &alpha);
            ((Rpp32f6*)affineTensor)[idx] = {alpha, -beta, 0, beta, alpha, 0};
        }

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_warp_affine_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, affineTensor,
                interpolationType, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_warp_affine_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_warp_affine_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_warp_affine_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, affineTensor,
                interpolationType, roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** phase ********************/

RppStatus rppt_phase(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                     RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                     rppHandle_t rppHandle, RppBackend executionBackend) {
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            phase_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                    static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            phase_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            phase_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            phase_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                    static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_phase_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                  static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                  srcDescPtr,
                                  static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                  dstDescPtr, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_phase_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_phase_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_phase_tensor(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                  static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                  srcDescPtr,
                                  static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                  dstDescPtr, roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** slice ********************/

RppStatus rppt_slice(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr,
                     RpptGenericDescPtr dstGenericDescPtr, Rpp32s* anchorTensor,
                     Rpp32s* shapeTensor, RppPtr_t fillValue, bool enablePadding, Rpp32u* roiTensor,
                     rppHandle_t rppHandle, RppBackend executionBackend) {
    if ((srcGenericDescPtr->dataType != RpptDataType::F32) &&
        (srcGenericDescPtr->dataType != RpptDataType::U8))
        return RPP_ERROR_INVALID_SRC_DATATYPE;
    if ((dstGenericDescPtr->dataType != RpptDataType::F32) &&
        (dstGenericDescPtr->dataType != RpptDataType::U8))
        return RPP_ERROR_INVALID_DST_DATATYPE;
    if (srcGenericDescPtr->dataType != dstGenericDescPtr->dataType)
        return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_LAYOUT_MISMATCH;
    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams;
        if ((srcGenericDescPtr->layout == RpptLayout::NCDHW) &&
            (dstGenericDescPtr->layout == RpptLayout::NCDHW))
            layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[1]);
        else if ((srcGenericDescPtr->layout == RpptLayout::NDHWC) &&
                 (dstGenericDescPtr->layout == RpptLayout::NDHWC))
            layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[4]);
        else if ((srcGenericDescPtr->layout == RpptLayout::NCHW) &&
                 (dstGenericDescPtr->layout == RpptLayout::NCHW))
            layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[1]);
        else if ((srcGenericDescPtr->layout == RpptLayout::NHWC) &&
                 (dstGenericDescPtr->layout == RpptLayout::NHWC))
            layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[3]);

        if ((srcGenericDescPtr->dataType == RpptDataType::F32) &&
            (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            slice_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                                        srcGenericDescPtr->offsetInBytes),
                              srcGenericDescPtr,
                              reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                                        dstGenericDescPtr->offsetInBytes),
                              dstGenericDescPtr, anchorTensor, shapeTensor,
                              static_cast<Rpp32f*>(fillValue), enablePadding, roiTensor,
                              layoutParams, handle);
        } else if ((srcGenericDescPtr->dataType == RpptDataType::U8) &&
                   (dstGenericDescPtr->dataType == RpptDataType::U8)) {
            slice_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes, srcGenericDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                anchorTensor, shapeTensor, static_cast<Rpp8u*>(fillValue), enablePadding, roiTensor,
                layoutParams, handle);
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcGenericDescPtr->dataType == RpptDataType::F32) &&
            (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_slice_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                                            srcGenericDescPtr->offsetInBytes),
                                  srcGenericDescPtr,
                                  reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                                            dstGenericDescPtr->offsetInBytes),
                                  dstGenericDescPtr, anchorTensor, shapeTensor,
                                  static_cast<Rpp32f*>(fillValue), enablePadding, roiTensor,
                                  handle);
        } else if ((srcGenericDescPtr->dataType == RpptDataType::U8) &&
                   (dstGenericDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_slice_tensor(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                  srcGenericDescPtr,
                                  static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes,
                                  dstGenericDescPtr, anchorTensor, shapeTensor,
                                  static_cast<Rpp8u*>(fillValue), enablePadding, roiTensor, handle);
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** flip_voxel ********************/

RppStatus rppt_flip_voxel(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr,
                          RpptGenericDescPtr dstGenericDescPtr, Rpp32u* horizontalTensor,
                          Rpp32u* verticalTensor, Rpp32u* depthTensor,
                          RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType,
                          rppHandle_t rppHandle, RppBackend executionBackend) {
    if ((srcGenericDescPtr->dataType != RpptDataType::F32) &&
        (srcGenericDescPtr->dataType != RpptDataType::U8))
        return RPP_ERROR_INVALID_SRC_DATATYPE;
    if ((dstGenericDescPtr->dataType != RpptDataType::F32) &&
        (dstGenericDescPtr->dataType != RpptDataType::U8))
        return RPP_ERROR_INVALID_DST_DATATYPE;
    if (srcGenericDescPtr->dataType != dstGenericDescPtr->dataType)
        return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcGenericDescPtr->layout != RpptLayout::NCDHW) &&
        (srcGenericDescPtr->layout != RpptLayout::NDHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstGenericDescPtr->layout != RpptLayout::NCDHW) &&
        (dstGenericDescPtr->layout != RpptLayout::NDHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_INVALID_ARGUMENTS;
    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams;
        if ((srcGenericDescPtr->layout == RpptLayout::NCDHW) &&
            (dstGenericDescPtr->layout == RpptLayout::NCDHW))
            layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[1]);
        else if ((srcGenericDescPtr->layout == RpptLayout::NDHWC) &&
                 (dstGenericDescPtr->layout == RpptLayout::NDHWC))
            layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[4]);

        if ((srcGenericDescPtr->dataType == RpptDataType::F32) &&
            (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            flip_voxel_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                          srcGenericDescPtr->offsetInBytes),
                srcGenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, horizontalTensor, verticalTensor, depthTensor, roiGenericPtrSrc,
                roiType, layoutParams, handle);
        } else if ((srcGenericDescPtr->dataType == RpptDataType::U8) &&
                   (dstGenericDescPtr->dataType == RpptDataType::U8)) {
            flip_voxel_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes, srcGenericDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                horizontalTensor, verticalTensor, depthTensor, roiGenericPtrSrc, roiType,
                layoutParams, handle);
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcGenericDescPtr->dataType == RpptDataType::F32) &&
            (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_flip_voxel_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                                                 srcGenericDescPtr->offsetInBytes),
                                       srcGenericDescPtr,
                                       reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                                                 dstGenericDescPtr->offsetInBytes),
                                       dstGenericDescPtr, roiGenericPtrSrc, horizontalTensor,
                                       verticalTensor, depthTensor, roiType, handle);
        } else if ((srcGenericDescPtr->dataType == RpptDataType::U8) &&
                   (dstGenericDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_flip_voxel_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes, srcGenericDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                roiGenericPtrSrc, horizontalTensor, verticalTensor, depthTensor, roiType, handle);
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** remap ********************/

RppStatus rppt_remap(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                     RpptDescPtr dstDescPtr, Rpp32f* rowRemapTable, Rpp32f* colRemapTable,
                     RpptDescPtr tableDescPtr, RpptInterpolationType interpolationType,
                     RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle,
                     RppBackend executionBackend) {
    if (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR &&
        interpolationType != RpptInterpolationType::BILINEAR)
        return RPP_ERROR_NOT_IMPLEMENTED;
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if (interpolationType == RpptInterpolationType::NEAREST_NEIGHBOR) {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                remap_nn_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                           srcDescPtr,
                                           static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                           dstDescPtr, rowRemapTable, colRemapTable, tableDescPtr,
                                           roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                remap_nn_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) +
                                                                       srcDescPtr->offsetInBytes),
                                             srcDescPtr,
                                             reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) +
                                                                       dstDescPtr->offsetInBytes),
                                             dstDescPtr, rowRemapTable, colRemapTable, tableDescPtr,
                                             roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                remap_nn_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                                                       srcDescPtr->offsetInBytes),
                                             srcDescPtr,
                                             reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                                                       dstDescPtr->offsetInBytes),
                                             dstDescPtr, rowRemapTable, colRemapTable, tableDescPtr,
                                             roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                remap_nn_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                           srcDescPtr,
                                           static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                           dstDescPtr, rowRemapTable, colRemapTable, tableDescPtr,
                                           roiTensorPtrSrc, roiType, layoutParams, handle);
            }
        } else if (interpolationType == RpptInterpolationType::BILINEAR) {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                remap_bilinear_u8_u8_host_tensor(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    rowRemapTable, colRemapTable, tableDescPtr, roiTensorPtrSrc, roiType,
                    layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                remap_bilinear_f16_f16_host_tensor(
                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, rowRemapTable, colRemapTable, tableDescPtr, roiTensorPtrSrc,
                    roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                remap_bilinear_f32_f32_host_tensor(
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, rowRemapTable, colRemapTable, tableDescPtr, roiTensorPtrSrc,
                    roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                remap_bilinear_i8_i8_host_tensor(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    rowRemapTable, colRemapTable, tableDescPtr, roiTensorPtrSrc, roiType,
                    layoutParams, handle);
            }
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_remap_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, rowRemapTable,
                colRemapTable, tableDescPtr, interpolationType, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_remap_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, rowRemapTable, colRemapTable, tableDescPtr, interpolationType,
                roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_remap_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, rowRemapTable, colRemapTable, tableDescPtr, interpolationType,
                roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_remap_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, rowRemapTable,
                colRemapTable, tableDescPtr, interpolationType, roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** lens_correction ********************/

RppStatus rppt_lens_correction(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr, Rpp32f* rowRemapTable, Rpp32f* colRemapTable,
                               RpptDescPtr tableDescPtr, Rpp32f* cameraMatrixTensor,
                               Rpp32f* distortionCoeffsTensor, RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType, rppHandle_t rppHandle,
                               RppBackend executionBackend) {
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        compute_lens_correction_remap_tables_host_tensor(
            srcDescPtr, rowRemapTable, colRemapTable, tableDescPtr, cameraMatrixTensor,
            distortionCoeffsTensor, roiTensorPtrSrc, handle);

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            remap_bilinear_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, rowRemapTable,
                colRemapTable, tableDescPtr, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            remap_bilinear_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, rowRemapTable, colRemapTable, tableDescPtr, roiTensorPtrSrc, roiType,
                layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            remap_bilinear_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, rowRemapTable, colRemapTable, tableDescPtr, roiTensorPtrSrc, roiType,
                layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            remap_bilinear_i8_i8_host_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, rowRemapTable,
                colRemapTable, tableDescPtr, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        hip_exec_lens_correction_tensor(dstDescPtr, rowRemapTable, colRemapTable, tableDescPtr,
                                        cameraMatrixTensor, distortionCoeffsTensor, roiTensorPtrSrc,
                                        roiType, handle);

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_remap_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, rowRemapTable,
                colRemapTable, tableDescPtr, RpptInterpolationType::BILINEAR, roiTensorPtrSrc,
                roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_remap_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, rowRemapTable, colRemapTable, tableDescPtr,
                RpptInterpolationType::BILINEAR, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_remap_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, rowRemapTable, colRemapTable, tableDescPtr,
                RpptInterpolationType::BILINEAR, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_remap_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, rowRemapTable,
                colRemapTable, tableDescPtr, RpptInterpolationType::BILINEAR, roiTensorPtrSrc,
                roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** transpose ********************/

RppStatus rppt_transpose(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr,
                         RpptGenericDescPtr dstGenericDescPtr, Rpp32u* permTensor,
                         Rpp32u* roiTensor, rppHandle_t rppHandle, RppBackend executionBackend) {
    if (srcGenericDescPtr->dataType != dstGenericDescPtr->dataType)
        return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        if ((srcGenericDescPtr->dataType == RpptDataType::U8) &&
            (dstGenericDescPtr->dataType == RpptDataType::U8)) {
            transpose_generic_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes, srcGenericDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                permTensor, roiTensor, handle);
        } else if ((srcGenericDescPtr->dataType == RpptDataType::F16) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F16)) {
            transpose_generic_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) +
                                          srcGenericDescPtr->offsetInBytes),
                srcGenericDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, permTensor, roiTensor, handle);
        } else if ((srcGenericDescPtr->dataType == RpptDataType::F32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            transpose_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                          srcGenericDescPtr->offsetInBytes),
                srcGenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, permTensor, roiTensor, handle);
        } else if ((srcGenericDescPtr->dataType == RpptDataType::I8) &&
                   (dstGenericDescPtr->dataType == RpptDataType::I8)) {
            transpose_generic_host_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcGenericDescPtr->offsetInBytes, srcGenericDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                permTensor, roiTensor, handle);
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcGenericDescPtr->dataType == RpptDataType::U8) &&
            (dstGenericDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_transpose_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes, srcGenericDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                permTensor, roiTensor, handle);
        } else if ((srcGenericDescPtr->dataType == RpptDataType::F16) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_transpose_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) +
                                                              srcGenericDescPtr->offsetInBytes),
                                      srcGenericDescPtr,
                                      reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) +
                                                              dstGenericDescPtr->offsetInBytes),
                                      dstGenericDescPtr, permTensor, roiTensor, handle);
        } else if ((srcGenericDescPtr->dataType == RpptDataType::F32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_transpose_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                                                srcGenericDescPtr->offsetInBytes),
                                      srcGenericDescPtr,
                                      reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                                                dstGenericDescPtr->offsetInBytes),
                                      dstGenericDescPtr, permTensor, roiTensor, handle);
        } else if ((srcGenericDescPtr->dataType == RpptDataType::I8) &&
                   (dstGenericDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_transpose_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcGenericDescPtr->offsetInBytes, srcGenericDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                permTensor, roiTensor, handle);
        }

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** warp_perspective ********************/

RppStatus rppt_warp_perspective(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                                RpptDescPtr dstDescPtr, Rpp32f* perspectiveTensor,
                                RpptInterpolationType interpolationType, RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType, rppHandle_t rppHandle,
                                RppBackend executionBackend) {
    if ((interpolationType != RpptInterpolationType::BILINEAR) &&
        (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR))
        return RPP_ERROR_NOT_IMPLEMENTED;
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if (interpolationType == RpptInterpolationType::NEAREST_NEIGHBOR) {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                warp_perspective_nn_u8_u8_host_tensor(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    perspectiveTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                warp_perspective_nn_f32_f32_host_tensor(
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, perspectiveTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                warp_perspective_nn_i8_i8_host_tensor(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    perspectiveTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                warp_perspective_nn_f16_f16_host_tensor(
                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, perspectiveTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            }
        } else if (interpolationType == RpptInterpolationType::BILINEAR) {
            if ((srcDescPtr->dataType == RpptDataType::U8) &&
                (dstDescPtr->dataType == RpptDataType::U8)) {
                warp_perspective_bilinear_u8_u8_host_tensor(
                    static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    perspectiveTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                       (dstDescPtr->dataType == RpptDataType::F32)) {
                warp_perspective_bilinear_f32_f32_host_tensor(
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, perspectiveTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                       (dstDescPtr->dataType == RpptDataType::I8)) {
                warp_perspective_bilinear_i8_i8_host_tensor(
                    static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                    perspectiveTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                       (dstDescPtr->dataType == RpptDataType::F16)) {
                warp_perspective_bilinear_f16_f16_host_tensor(
                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) +
                                              srcDescPtr->offsetInBytes),
                    srcDescPtr,
                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) +
                                              dstDescPtr->offsetInBytes),
                    dstDescPtr, perspectiveTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
            }
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_warp_perspective_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                perspectiveTensor, interpolationType, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_warp_perspective_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, perspectiveTensor, interpolationType, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_warp_perspective_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, perspectiveTensor, interpolationType, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_warp_perspective_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                perspectiveTensor, interpolationType, roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** jpeg_compression_distortion ********************/

RppStatus rppt_jpeg_compression_distortion(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                                           RpptDescPtr dstDescPtr, Rpp32s* qualityTensor,
                                           RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                           rppHandle_t rppHandle, RppBackend executionBackend) {
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            jpeg_compression_distortion_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, qualityTensor,
                roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            jpeg_compression_distortion_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, qualityTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            jpeg_compression_distortion_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, qualityTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            jpeg_compression_distortion_i8_i8_host_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, qualityTensor,
                roiTensorPtrSrc, roiType, layoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_jpeg_compression_distortion(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, qualityTensor,
                roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_jpeg_compression_distortion(
                reinterpret_cast<half*>((static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes)),
                srcDescPtr,
                reinterpret_cast<half*>((static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes)),
                dstDescPtr, qualityTensor, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_jpeg_compression_distortion(
                reinterpret_cast<Rpp32f*>(
                    (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes)),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(
                    (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes)),
                dstDescPtr, qualityTensor, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_jpeg_compression_distortion(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, qualityTensor,
                roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** concat ********************/

RppStatus rppt_concat(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptGenericDescPtr srcPtr1GenericDescPtr,
                      RpptGenericDescPtr srcPtr2GenericDescPtr, RppPtr_t dstPtr,
                      RpptGenericDescPtr dstGenericDescPtr, Rpp32u axisMask, Rpp32u* roiTensorSrc1,
                      Rpp32u* roiTensorSrc2, rppHandle_t rppHandle, RppBackend executionBackend) {
    Rpp32u tensorDim =
        srcPtr1GenericDescPtr->numDims - 1;  // Ignoring batchSize here to get tensor dimensions.
    if (srcPtr1GenericDescPtr->numDims != srcPtr2GenericDescPtr->numDims)
        return RPP_ERROR_INVALID_SRC_DIMS;
    if (srcPtr1GenericDescPtr->layout != dstGenericDescPtr->layout)
        return RPP_ERROR_LAYOUT_MISMATCH;
    if (axisMask >= srcPtr1GenericDescPtr->numDims) return RPP_ERROR_INVALID_AXIS;
    for (int i = 0; i < tensorDim; i++)
        if ((i != axisMask) && (srcPtr1GenericDescPtr->dims[i] != srcPtr2GenericDescPtr->dims[i]))
            return RPP_ERROR_INVALID_DIM_LENGTHS;

    if ((srcPtr1GenericDescPtr->dataType != srcPtr2GenericDescPtr->dataType) ||
        (srcPtr1GenericDescPtr->dataType != dstGenericDescPtr->dataType))
        return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams;
        if (tensorDim == 3 && (srcPtr1GenericDescPtr->layout == RpptLayout::NHWC))
            layoutParams =
                get_layout_params(srcPtr1GenericDescPtr->layout, srcPtr1GenericDescPtr->dims[3]);
        else if ((srcPtr1GenericDescPtr->layout == RpptLayout::NCDHW) &&
                 (dstGenericDescPtr->layout == RpptLayout::NCDHW))
            layoutParams =
                get_layout_params(srcPtr1GenericDescPtr->layout, srcPtr1GenericDescPtr->dims[1]);
        else if ((srcPtr1GenericDescPtr->layout == RpptLayout::NDHWC) &&
                 (dstGenericDescPtr->layout == RpptLayout::NDHWC))
            layoutParams =
                get_layout_params(srcPtr1GenericDescPtr->layout, srcPtr1GenericDescPtr->dims[4]);
        else if (tensorDim == 2 && (srcPtr1GenericDescPtr->layout == RpptLayout::NHWC))
            layoutParams =
                get_layout_params(srcPtr1GenericDescPtr->layout, srcPtr1GenericDescPtr->dims[2]);

        if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U8) &&
            (dstGenericDescPtr->dataType == RpptDataType::U8)) {
            concat_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8u*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                axisMask, roiTensorSrc1, roiTensorSrc2, layoutParams, handle);
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::F16) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F16)) {
            concat_generic_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, axisMask, roiTensorSrc1, roiTensorSrc2, layoutParams, handle);
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::F32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            concat_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, axisMask, roiTensorSrc1, roiTensorSrc2, layoutParams, handle);
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I8) &&
                   (dstGenericDescPtr->dataType == RpptDataType::I8)) {
            concat_generic_host_tensor(
                static_cast<Rpp8s*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8s*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                axisMask, roiTensorSrc1, roiTensorSrc2, layoutParams, handle);
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U8) &&
            (dstGenericDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_concat_tensor(
                static_cast<Rpp8u*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr,
                static_cast<Rpp8u*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr2GenericDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                axisMask, roiTensorSrc1, roiTensorSrc2, handle);
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::F16) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_concat_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr1) +
                                                           srcPtr1GenericDescPtr->offsetInBytes),
                                   srcPtr1GenericDescPtr,
                                   reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr2) +
                                                           srcPtr2GenericDescPtr->offsetInBytes),
                                   srcPtr2GenericDescPtr,
                                   reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) +
                                                           dstGenericDescPtr->offsetInBytes),
                                   dstGenericDescPtr, axisMask, roiTensorSrc1, roiTensorSrc2,
                                   handle);
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::F32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_concat_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) +
                                                             srcPtr1GenericDescPtr->offsetInBytes),
                                   srcPtr1GenericDescPtr,
                                   reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) +
                                                             srcPtr2GenericDescPtr->offsetInBytes),
                                   srcPtr2GenericDescPtr,
                                   reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                                             dstGenericDescPtr->offsetInBytes),
                                   dstGenericDescPtr, axisMask, roiTensorSrc1, roiTensorSrc2,
                                   handle);
        } else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I8) &&
                   (dstGenericDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_concat_tensor(
                static_cast<Rpp8s*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr,
                static_cast<Rpp8s*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr2GenericDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                axisMask, roiTensorSrc1, roiTensorSrc2, handle);
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** fisheye ********************/

RppStatus rppt_fisheye(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                       RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                       rppHandle_t rppHandle, RppBackend executionBackend) {
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            fisheye_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                roiTensorPtrSrc, roiType, layoutParams, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            fisheye_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, roiTensorPtrSrc, roiType, layoutParams, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            fisheye_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, roiTensorPtrSrc, roiType, layoutParams, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            fisheye_i8_i8_host_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                roiTensorPtrSrc, roiType, layoutParams, rpp::deref(rppHandle));
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_fisheye_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr, roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_fisheye_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_fisheye_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_fisheye_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr, roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}
