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

#include "rppdefs.h"
#include "rppi_validate.hpp"
#include "rppt_tensor_geometric_augmentations.h"
#include "host_tensor_executors.hpp"

#ifdef HIP_COMPILE
#include "hip_tensor_executors.hpp"
#endif // HIP_COMPILE

#if __APPLE__
#define sincosf __sincosf
#endif

/******************** crop ********************/

RppStatus rppt_crop_host(RppPtr_t srcPtr,
                         RpptDescPtr srcDescPtr,
                         RppPtr_t dstPtr,
                         RpptDescPtr dstDescPtr,
                         RpptROIPtr roiTensorPtrSrc,
                         RpptRoiType roiType,
                         rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        crop_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               roiTensorPtrSrc,
                               roiType,
                               layoutParams,
                               rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        crop_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 roiTensorPtrSrc,
                                 roiType,
                                 layoutParams,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        crop_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 roiTensorPtrSrc,
                                 roiType,
                                 layoutParams,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        crop_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               roiTensorPtrSrc,
                               roiType,
                               layoutParams,
                               rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** crop_and_patch ********************/

RppStatus rppt_crop_and_patch_host(RppPtr_t srcPtr1,
                                   RppPtr_t srcPtr2,
                                   RpptDescPtr srcDescPtr,
                                   RppPtr_t dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RpptROIPtr roiTensorPtrDst,
                                   RpptROIPtr cropRoi,
                                   RpptROIPtr patchRoi,
                                   RpptRoiType roiType,
                                   rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        crop_and_patch_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                         static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         roiTensorPtrDst,
                                         cropRoi,
                                         patchRoi,
                                         roiType,
                                         layoutParams,
                                         rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        crop_and_patch_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                           (Rpp16f*) (static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                           srcDescPtr,
                                           (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                           dstDescPtr,
                                           roiTensorPtrDst,
                                           cropRoi,
                                           patchRoi,
                                           roiType,
                                           layoutParams,
                                           rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        crop_and_patch_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                           (Rpp32f*) (static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                           srcDescPtr,
                                           (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                           dstDescPtr,
                                           roiTensorPtrDst,
                                           cropRoi,
                                           patchRoi,
                                           roiType,
                                           layoutParams,
                                           rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        crop_and_patch_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                         static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         roiTensorPtrDst,
                                         cropRoi,
                                         patchRoi,
                                         roiType,
                                         layoutParams,
                                         rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** crop mirror normalize ********************/

RppStatus rppt_crop_mirror_normalize_host(RppPtr_t srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          RppPtr_t dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          Rpp32f *offsetTensor,
                                          Rpp32f *multiplierTensor,
                                          Rpp32u *mirrorTensor,
                                          RpptROIPtr roiTensorPtrSrc,
                                          RpptRoiType roiType,
                                          rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        crop_mirror_normalize_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                dstDescPtr,
                                                offsetTensor,
                                                multiplierTensor,
                                                mirrorTensor,
                                                roiTensorPtrSrc,
                                                roiType,
                                                layoutParams,
                                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        crop_mirror_normalize_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                  srcDescPtr,
                                                  (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                  dstDescPtr,
                                                  offsetTensor,
                                                  multiplierTensor,
                                                  mirrorTensor,
                                                  roiTensorPtrSrc,
                                                  roiType,
                                                  layoutParams,
                                                  rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        crop_mirror_normalize_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                  srcDescPtr,
                                                  (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                  dstDescPtr,
                                                  offsetTensor,
                                                  multiplierTensor,
                                                  mirrorTensor,
                                                  roiTensorPtrSrc,
                                                  roiType,
                                                  layoutParams,
                                                  rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        crop_mirror_normalize_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                dstDescPtr,
                                                offsetTensor,
                                                multiplierTensor,
                                                mirrorTensor,
                                                roiTensorPtrSrc,
                                                roiType,
                                                layoutParams,
                                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        crop_mirror_normalize_u8_f32_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                 srcDescPtr,
                                                 (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                 dstDescPtr,
                                                 offsetTensor,
                                                 multiplierTensor,
                                                 mirrorTensor,
                                                 roiTensorPtrSrc,
                                                 roiType,
                                                 layoutParams,
                                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        crop_mirror_normalize_u8_f16_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                 srcDescPtr,
                                                 (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                 dstDescPtr,
                                                 offsetTensor,
                                                 multiplierTensor,
                                                 mirrorTensor,
                                                 roiTensorPtrSrc,
                                                 roiType,
                                                 layoutParams,
                                                 rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** warp_affine ********************/

RppStatus rppt_warp_affine_host(RppPtr_t srcPtr,
                                RpptDescPtr srcDescPtr,
                                RppPtr_t dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32f *affineTensor,
                                RpptInterpolationType interpolationType,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                rppHandle_t rppHandle)
{
    if ((interpolationType != RpptInterpolationType::BILINEAR) && (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR))
        return RPP_ERROR_NOT_IMPLEMENTED;

    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if(interpolationType == RpptInterpolationType::NEAREST_NEIGHBOR)
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            warp_affine_nn_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                             srcDescPtr,
                                             static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                             dstDescPtr,
                                             affineTensor,
                                             roiTensorPtrSrc,
                                             roiType,
                                             layoutParams,
                                             rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            warp_affine_nn_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                               srcDescPtr,
                                               (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                               dstDescPtr,
                                               affineTensor,
                                               roiTensorPtrSrc,
                                               roiType,
                                               layoutParams,
                                               rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            warp_affine_nn_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                               srcDescPtr,
                                               (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                               dstDescPtr,
                                               affineTensor,
                                               roiTensorPtrSrc,
                                               roiType,
                                               layoutParams,
                                               rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            warp_affine_nn_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                             srcDescPtr,
                                             static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                             dstDescPtr,
                                             affineTensor,
                                             roiTensorPtrSrc,
                                             roiType,
                                             layoutParams,
                                             rpp::deref(rppHandle));
        }
    }
    else if(interpolationType == RpptInterpolationType::BILINEAR)
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            warp_affine_bilinear_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                   srcDescPtr,
                                                   static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                   dstDescPtr,
                                                   affineTensor,
                                                   roiTensorPtrSrc,
                                                   roiType,
                                                   layoutParams,
                                                   rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            warp_affine_bilinear_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                     srcDescPtr,
                                                     (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                     dstDescPtr,
                                                     affineTensor,
                                                     roiTensorPtrSrc,
                                                     roiType,
                                                     layoutParams,
                                                     rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            warp_affine_bilinear_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                     srcDescPtr,
                                                     (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                     dstDescPtr,
                                                     affineTensor,
                                                     roiTensorPtrSrc,
                                                     roiType,
                                                     layoutParams,
                                                     rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            warp_affine_bilinear_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                   srcDescPtr,
                                                   static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                   dstDescPtr,
                                                   affineTensor,
                                                   roiTensorPtrSrc,
                                                   roiType,
                                                   layoutParams,
                                                   rpp::deref(rppHandle));
        }
    }

    return RPP_SUCCESS;
}

/******************** flip ********************/

RppStatus rppt_flip_host(RppPtr_t srcPtr,
                         RpptDescPtr srcDescPtr,
                         RppPtr_t dstPtr,
                         RpptDescPtr dstDescPtr,
                         Rpp32u *horizontalTensor,
                         Rpp32u *verticalTensor,
                         RpptROIPtr roiTensorPtrSrc,
                         RpptRoiType roiType,
                         rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        flip_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               horizontalTensor,
                               verticalTensor,
                               roiTensorPtrSrc,
                               roiType,
                               layoutParams,
                               rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        flip_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 horizontalTensor,
                                 verticalTensor,
                                 roiTensorPtrSrc,
                                 roiType,
                                 layoutParams,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        flip_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 horizontalTensor,
                                 verticalTensor,
                                 roiTensorPtrSrc,
                                 roiType,
                                 layoutParams,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        flip_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               horizontalTensor,
                               verticalTensor,
                               roiTensorPtrSrc,
                               roiType,
                               layoutParams,
                               rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** resize ********************/

RppStatus rppt_resize_host(RppPtr_t srcPtr,
                           RpptDescPtr srcDescPtr,
                           RppPtr_t dstPtr,
                           RpptDescPtr dstDescPtr,
                           RpptImagePatchPtr dstImgSizes,
                           RpptInterpolationType interpolationType,
                           RpptROIPtr roiTensorPtrSrc,
                           RpptRoiType roiType,
                           rppHandle_t rppHandle)
{
    RppLayoutParams srcLayoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if(interpolationType == RpptInterpolationType::NEAREST_NEIGHBOR)
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            resize_nn_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr,
                                        dstImgSizes,
                                        roiTensorPtrSrc,
                                        roiType,
                                        srcLayoutParams,
                                        rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            resize_nn_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                          srcDescPtr,
                                          (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                          dstDescPtr,
                                          dstImgSizes,
                                          roiTensorPtrSrc,
                                          roiType,
                                          srcLayoutParams,
                                          rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            resize_nn_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                          srcDescPtr,
                                          (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                          dstDescPtr,
                                          dstImgSizes,
                                          roiTensorPtrSrc,
                                          roiType,
                                          srcLayoutParams,
                                          rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            resize_nn_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr,
                                        dstImgSizes,
                                        roiTensorPtrSrc,
                                        roiType,
                                        srcLayoutParams,
                                        rpp::deref(rppHandle));
        }
    }
    else if(interpolationType == RpptInterpolationType::BILINEAR)
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            resize_bilinear_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                              srcDescPtr,
                                              static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                              dstDescPtr,
                                              dstImgSizes,
                                              roiTensorPtrSrc,
                                              roiType,
                                              srcLayoutParams,
                                              rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            resize_bilinear_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                srcDescPtr,
                                                (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                dstImgSizes,
                                                roiTensorPtrSrc,
                                                roiType,
                                                srcLayoutParams,
                                                rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            resize_bilinear_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                srcDescPtr,
                                                (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                dstImgSizes,
                                                roiTensorPtrSrc,
                                                roiType,
                                                srcLayoutParams,
                                                rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            resize_bilinear_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                               srcDescPtr,
                                               static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                               dstDescPtr,
                                               dstImgSizes,
                                               roiTensorPtrSrc,
                                               roiType,
                                               srcLayoutParams,
                                               rpp::deref(rppHandle));
        }
    }
    else
    {
        RpptDesc tempDesc;
        tempDesc = *srcDescPtr;
        RpptDescPtr tempDescPtr = &tempDesc;
        tempDescPtr->h = dstDescPtr->h;
        tempDescPtr->strides.nStride = srcDescPtr->w * dstDescPtr->h * srcDescPtr->c;

        // The channel stride changes with the change in the height for PLN images
        if(srcDescPtr->layout == RpptLayout::NCHW)
            tempDescPtr->strides.cStride = srcDescPtr->w * dstDescPtr->h;

        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            resize_separable_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.scratchBufferHost,
                                         tempDescPtr,
                                         dstImgSizes,
                                         roiTensorPtrSrc,
                                         roiType,
                                         srcLayoutParams,
                                         interpolationType,
                                         rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            resize_separable_host_tensor(static_cast<Rpp32f*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp32f*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.scratchBufferHost,
                                         tempDescPtr,
                                         dstImgSizes,
                                         roiTensorPtrSrc,
                                         roiType,
                                         srcLayoutParams,
                                         interpolationType,
                                         rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            resize_separable_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.scratchBufferHost,
                                         tempDescPtr,
                                         dstImgSizes,
                                         roiTensorPtrSrc,
                                         roiType,
                                         srcLayoutParams,
                                         interpolationType,
                                         rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            resize_separable_host_tensor(static_cast<Rpp16f*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp16f*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.scratchBufferHost,
                                         tempDescPtr,
                                         dstImgSizes,
                                         roiTensorPtrSrc,
                                         roiType,
                                         srcLayoutParams,
                                         interpolationType,
                                         rpp::deref(rppHandle));
        }
    }

    return RPP_SUCCESS;
}

/******************** resize_mirror_normalize ********************/

RppStatus rppt_resize_mirror_normalize_host(RppPtr_t srcPtr,
                                            RpptDescPtr srcDescPtr,
                                            RppPtr_t dstPtr,
                                            RpptDescPtr dstDescPtr,
                                            RpptImagePatchPtr dstImgSizes,
                                            RpptInterpolationType interpolationType,
                                            Rpp32f *meanTensor,
                                            Rpp32f *stdDevTensor,
                                            Rpp32u *mirrorTensor,
                                            RpptROIPtr roiTensorPtrSrc,
                                            RpptRoiType roiType,
                                            rppHandle_t rppHandle)
{
    RppLayoutParams srcLayoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if (interpolationType != RpptInterpolationType::BILINEAR)
        return RPP_ERROR_NOT_IMPLEMENTED;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        resize_mirror_normalize_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                  srcDescPtr,
                                                  static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                  dstDescPtr,
                                                  dstImgSizes,
                                                  meanTensor,
                                                  stdDevTensor,
                                                  mirrorTensor,
                                                  roiTensorPtrSrc,
                                                  roiType,
                                                  srcLayoutParams,
                                                  rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        resize_mirror_normalize_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                    srcDescPtr,
                                                    (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                    dstDescPtr,
                                                    dstImgSizes,
                                                    meanTensor,
                                                    stdDevTensor,
                                                    mirrorTensor,
                                                    roiTensorPtrSrc,
                                                    roiType,
                                                    srcLayoutParams,
                                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        resize_mirror_normalize_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                    srcDescPtr,
                                                    (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                    dstDescPtr,
                                                    dstImgSizes,
                                                    meanTensor,
                                                    stdDevTensor,
                                                    mirrorTensor,
                                                    roiTensorPtrSrc,
                                                    roiType,
                                                    srcLayoutParams,
                                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        resize_mirror_normalize_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                  srcDescPtr,
                                                  static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                  dstDescPtr,
                                                  dstImgSizes,
                                                  meanTensor,
                                                  stdDevTensor,
                                                  mirrorTensor,
                                                  roiTensorPtrSrc,
                                                  roiType,
                                                  srcLayoutParams,
                                                  rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        resize_mirror_normalize_u8_f32_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                   srcDescPtr,
                                                   (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                   dstDescPtr,
                                                   dstImgSizes,
                                                   meanTensor,
                                                   stdDevTensor,
                                                   mirrorTensor,
                                                   roiTensorPtrSrc,
                                                   roiType,
                                                   srcLayoutParams,
                                                   rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        resize_mirror_normalize_u8_f16_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                   srcDescPtr,
                                                   (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                   dstDescPtr,
                                                   dstImgSizes,
                                                   meanTensor,
                                                   stdDevTensor,
                                                   mirrorTensor,
                                                   roiTensorPtrSrc,
                                                   roiType,
                                                   srcLayoutParams,
                                                   rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

RppStatus rppt_resize_crop_mirror_host(RppPtr_t srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       RppPtr_t dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       RpptImagePatchPtr dstImgSizes,
                                       RpptInterpolationType interpolationType,
                                       Rpp32u *mirrorTensor,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       rppHandle_t rppHandle)
{
    RppLayoutParams srcLayoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if (interpolationType != RpptInterpolationType::BILINEAR)
        return RPP_ERROR_NOT_IMPLEMENTED;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        resize_crop_mirror_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                             srcDescPtr,
                                             static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                             dstDescPtr,
                                             dstImgSizes,
                                             mirrorTensor,
                                             roiTensorPtrSrc,
                                             roiType,
                                             srcLayoutParams,
                                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        resize_crop_mirror_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                               srcDescPtr,
                                               (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                               dstDescPtr,
                                               dstImgSizes,
                                               mirrorTensor,
                                               roiTensorPtrSrc,
                                               roiType,
                                               srcLayoutParams,
                                               rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        resize_crop_mirror_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                               srcDescPtr,
                                               (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                               dstDescPtr,
                                               dstImgSizes,
                                               mirrorTensor,
                                               roiTensorPtrSrc,
                                               roiType,
                                               srcLayoutParams,
                                               rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        resize_crop_mirror_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                             srcDescPtr,
                                             static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                             dstDescPtr,
                                             dstImgSizes,
                                             mirrorTensor,
                                             roiTensorPtrSrc,
                                             roiType,
                                             srcLayoutParams,
                                             rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** rotate ********************/

RppStatus rppt_rotate_host(RppPtr_t srcPtr,
                           RpptDescPtr srcDescPtr,
                           RppPtr_t dstPtr,
                           RpptDescPtr dstDescPtr,
                           Rpp32f *angle,
                           RpptInterpolationType interpolationType,
                           RpptROIPtr roiTensorPtrSrc,
                           RpptRoiType roiType,
                           rppHandle_t rppHandle)
{
    if ((interpolationType != RpptInterpolationType::BILINEAR) && (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR))
        return RPP_ERROR_NOT_IMPLEMENTED;

    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    // Compute affine transformation matrix from rotate angle
    Rpp32f *affineTensor = rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.scratchBufferHost;
    for(int idx = 0; idx < srcDescPtr->n; idx++)
    {
        Rpp32f angleInRad = angle[idx] * PI_OVER_180;
        Rpp32f alpha, beta;
        sincosf(angleInRad, &beta, &alpha);
        ((Rpp32f6 *)affineTensor)[idx] = {alpha, -beta, 0, beta, alpha, 0};
    }

    if(interpolationType == RpptInterpolationType::NEAREST_NEIGHBOR)
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            warp_affine_nn_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                             srcDescPtr,
                                             static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                             dstDescPtr,
                                             affineTensor,
                                             roiTensorPtrSrc,
                                             roiType,
                                             layoutParams,
                                             rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            warp_affine_nn_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                               srcDescPtr,
                                               (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                               dstDescPtr,
                                               affineTensor,
                                               roiTensorPtrSrc,
                                               roiType,
                                               layoutParams,
                                               rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            warp_affine_nn_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                               srcDescPtr,
                                               (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                               dstDescPtr,
                                               affineTensor,
                                               roiTensorPtrSrc,
                                               roiType,
                                               layoutParams,
                                               rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            warp_affine_nn_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                             srcDescPtr,
                                             static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                             dstDescPtr,
                                             affineTensor,
                                             roiTensorPtrSrc,
                                             roiType,
                                             layoutParams,
                                             rpp::deref(rppHandle));
        }
    }
    else if(interpolationType == RpptInterpolationType::BILINEAR)
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            warp_affine_bilinear_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                   srcDescPtr,
                                                   static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                   dstDescPtr,
                                                   affineTensor,
                                                   roiTensorPtrSrc,
                                                   roiType,
                                                   layoutParams,
                                                   rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            warp_affine_bilinear_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                     srcDescPtr,
                                                     (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                     dstDescPtr,
                                                     affineTensor,
                                                     roiTensorPtrSrc,
                                                     roiType,
                                                     layoutParams,
                                                     rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            warp_affine_bilinear_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                     srcDescPtr,
                                                     (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                     dstDescPtr,
                                                     affineTensor,
                                                     roiTensorPtrSrc,
                                                     roiType,
                                                     layoutParams,
                                                     rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            warp_affine_bilinear_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                   srcDescPtr,
                                                   static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                   dstDescPtr,
                                                   affineTensor,
                                                   roiTensorPtrSrc,
                                                   roiType,
                                                   layoutParams,
                                                   rpp::deref(rppHandle));
        }
    }

    return RPP_SUCCESS;
}

/******************** phase ********************/

RppStatus rppt_phase_host(RppPtr_t srcPtr1,
                          RppPtr_t srcPtr2,
                          RpptDescPtr srcDescPtr,
                          RppPtr_t dstPtr,
                          RpptDescPtr dstDescPtr,
                          RpptROIPtr roiTensorPtrSrc,
                          RpptRoiType roiType,
                          rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout == RpptLayout::NCDHW) || (srcDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout == RpptLayout::NCDHW) || (dstDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        phase_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                dstDescPtr,
                                roiTensorPtrSrc,
                                roiType,
                                layoutParams,
                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        phase_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                  reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                  srcDescPtr,
                                  reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                  dstDescPtr,
                                  roiTensorPtrSrc,
                                  roiType,
                                  layoutParams,
                                  rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        phase_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                  reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                  srcDescPtr,
                                  reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                  dstDescPtr,
                                  roiTensorPtrSrc,
                                  roiType,
                                  layoutParams,
                                  rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        phase_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                dstDescPtr,
                                roiTensorPtrSrc,
                                roiType,
                                layoutParams,
                                rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** slice ********************/

RppStatus rppt_slice_host(RppPtr_t srcPtr,
                          RpptGenericDescPtr srcGenericDescPtr,
                          RppPtr_t dstPtr,
                          RpptGenericDescPtr dstGenericDescPtr,
                          Rpp32s *anchorTensor,
                          Rpp32s *shapeTensor,
                          RppPtr_t fillValue,
                          bool enablePadding,
                          Rpp32u *roiTensor,
                          rppHandle_t rppHandle)
{
    if ((srcGenericDescPtr->dataType != RpptDataType::F32) && (srcGenericDescPtr->dataType != RpptDataType::U8)) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if ((dstGenericDescPtr->dataType != RpptDataType::F32) && (dstGenericDescPtr->dataType != RpptDataType::U8)) return RPP_ERROR_INVALID_DST_DATATYPE;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_LAYOUT_MISMATCH;

    RppLayoutParams layoutParams;
    if ((srcGenericDescPtr->layout == RpptLayout::NCDHW) && (dstGenericDescPtr->layout == RpptLayout::NCDHW))
        layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[1]);
    else if ((srcGenericDescPtr->layout == RpptLayout::NDHWC) && (dstGenericDescPtr->layout == RpptLayout::NDHWC))
        layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[4]);
    else if ((srcGenericDescPtr->layout == RpptLayout::NCHW) && (dstGenericDescPtr->layout == RpptLayout::NCHW))
        layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[1]);
    else if ((srcGenericDescPtr->layout == RpptLayout::NHWC) && (dstGenericDescPtr->layout == RpptLayout::NHWC))
        layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[3]);

    if ((srcGenericDescPtr->dataType == RpptDataType::F32) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        slice_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                          srcGenericDescPtr,
                          reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                          dstGenericDescPtr,
                          anchorTensor,
                          shapeTensor,
                          static_cast<Rpp32f *>(fillValue),
                          enablePadding,
                          roiTensor,
                          layoutParams,
                          rpp::deref(rppHandle));
    }
    else if ((srcGenericDescPtr->dataType == RpptDataType::U8) && (dstGenericDescPtr->dataType == RpptDataType::U8))
    {
        slice_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                          srcGenericDescPtr,
                          static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes,
                          dstGenericDescPtr,
                          anchorTensor,
                          shapeTensor,
                          static_cast<Rpp8u *>(fillValue),
                          enablePadding,
                          roiTensor,
                          layoutParams,
                          rpp::deref(rppHandle));
    }
    return RPP_SUCCESS;
}

/******************** flip_voxel ********************/

RppStatus rppt_flip_voxel_host(RppPtr_t srcPtr,
                               RpptGenericDescPtr srcGenericDescPtr,
                               RppPtr_t dstPtr,
                               RpptGenericDescPtr dstGenericDescPtr,
                               Rpp32u *horizontalTensor,
                               Rpp32u *verticalTensor,
                               Rpp32u *depthTensor,
                               RpptROI3DPtr roiGenericPtrSrc,
                               RpptRoi3DType roiType,
                               rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams;
    if ((srcGenericDescPtr->layout == RpptLayout::NCDHW) && (dstGenericDescPtr->layout == RpptLayout::NCDHW))
        layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[1]);
    else if ((srcGenericDescPtr->layout == RpptLayout::NDHWC) && (dstGenericDescPtr->layout == RpptLayout::NDHWC))
        layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[4]);

    if ((srcGenericDescPtr->dataType != RpptDataType::F32) && (srcGenericDescPtr->dataType != RpptDataType::U8)) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if ((dstGenericDescPtr->dataType != RpptDataType::F32) && (dstGenericDescPtr->dataType != RpptDataType::U8)) return RPP_ERROR_INVALID_DST_DATATYPE;
    if ((srcGenericDescPtr->layout != RpptLayout::NCDHW) && (srcGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstGenericDescPtr->layout != RpptLayout::NCDHW) && (dstGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_INVALID_ARGUMENTS;

    if ((srcGenericDescPtr->dataType == RpptDataType::F32) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        flip_voxel_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                       srcGenericDescPtr,
                                       reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                       dstGenericDescPtr,
                                       horizontalTensor,
                                       verticalTensor,
                                       depthTensor,
                                       roiGenericPtrSrc,
                                       roiType,
                                       layoutParams,
                                       rpp::deref(rppHandle));
    }
    else if ((srcGenericDescPtr->dataType == RpptDataType::U8) && (dstGenericDescPtr->dataType == RpptDataType::U8))
    {
        flip_voxel_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                     srcGenericDescPtr,
                                     static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes,
                                     dstGenericDescPtr,
                                     horizontalTensor,
                                     verticalTensor,
                                     depthTensor,
                                     roiGenericPtrSrc,
                                     roiType,
                                     layoutParams,
                                     rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** remap ********************/

RppStatus rppt_remap_host(RppPtr_t srcPtr,
                          RpptDescPtr srcDescPtr,
                          RppPtr_t dstPtr,
                          RpptDescPtr dstDescPtr,
                          Rpp32f *rowRemapTable,
                          Rpp32f *colRemapTable,
                          RpptDescPtr tableDescPtr,
                          RpptInterpolationType interpolationType,
                          RpptROIPtr roiTensorPtrSrc,
                          RpptRoiType roiType,
                          rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR && interpolationType != RpptInterpolationType::BILINEAR)
        return RPP_ERROR_NOT_IMPLEMENTED;

    if(interpolationType == RpptInterpolationType::NEAREST_NEIGHBOR)
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            remap_nn_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       rowRemapTable,
                                       colRemapTable,
                                       tableDescPtr,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams,
                                       rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            remap_nn_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                         dstDescPtr,
                                         rowRemapTable,
                                         colRemapTable,
                                         tableDescPtr,
                                         roiTensorPtrSrc,
                                         roiType,
                                         layoutParams,
                                         rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            remap_nn_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                         dstDescPtr,
                                         rowRemapTable,
                                         colRemapTable,
                                         tableDescPtr,
                                         roiTensorPtrSrc,
                                         roiType,
                                         layoutParams,
                                         rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            remap_nn_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       rowRemapTable,
                                       colRemapTable,
                                       tableDescPtr,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams,
                                       rpp::deref(rppHandle));
        }
    }
    else if(interpolationType == RpptInterpolationType::BILINEAR)
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            remap_bilinear_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                             srcDescPtr,
                                             static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                             dstDescPtr,
                                             rowRemapTable,
                                             colRemapTable,
                                             tableDescPtr,
                                             roiTensorPtrSrc,
                                             roiType,
                                             layoutParams,
                                             rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            remap_bilinear_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                               srcDescPtr,
                                               reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                               dstDescPtr,
                                               rowRemapTable,
                                               colRemapTable,
                                               tableDescPtr,
                                               roiTensorPtrSrc,
                                               roiType,
                                               layoutParams,
                                               rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            remap_bilinear_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                               srcDescPtr,
                                               reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                               dstDescPtr,
                                               rowRemapTable,
                                               colRemapTable,
                                               tableDescPtr,
                                               roiTensorPtrSrc,
                                               roiType,
                                               layoutParams,
                                               rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            remap_bilinear_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                             srcDescPtr,
                                             static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                             dstDescPtr,
                                             rowRemapTable,
                                             colRemapTable,
                                             tableDescPtr,
                                             roiTensorPtrSrc,
                                             roiType,
                                             layoutParams,
                                             rpp::deref(rppHandle));
        }
    }

    return RPP_SUCCESS;
}

/******************** lens_correction ********************/

RppStatus rppt_lens_correction_host(RppPtr_t srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    RppPtr_t dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32f *rowRemapTable,
                                    Rpp32f *colRemapTable,
                                    RpptDescPtr tableDescPtr,
                                    Rpp32f *cameraMatrixTensor,
                                    Rpp32f *distortionCoeffsTensor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
    compute_lens_correction_remap_tables_host_tensor(srcDescPtr,
                                                     rowRemapTable,
                                                     colRemapTable,
                                                     tableDescPtr,
                                                     cameraMatrixTensor,
                                                     distortionCoeffsTensor,
                                                     roiTensorPtrSrc,
                                                     rpp::deref(rppHandle));

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        remap_bilinear_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         rowRemapTable,
                                         colRemapTable,
                                         tableDescPtr,
                                         roiTensorPtrSrc,
                                         roiType,
                                         layoutParams,
                                         rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        remap_bilinear_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                           srcDescPtr,
                                           reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                           dstDescPtr,
                                           rowRemapTable,
                                           colRemapTable,
                                           tableDescPtr,
                                           roiTensorPtrSrc,
                                           roiType,
                                           layoutParams,
                                           rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        remap_bilinear_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                           srcDescPtr,
                                           reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                           dstDescPtr,
                                           rowRemapTable,
                                           colRemapTable,
                                           tableDescPtr,
                                           roiTensorPtrSrc,
                                           roiType,
                                           layoutParams,
                                           rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        remap_bilinear_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         rowRemapTable,
                                         colRemapTable,
                                         tableDescPtr,
                                         roiTensorPtrSrc,
                                         roiType,
                                         layoutParams,
                                         rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** transpose ********************/

RppStatus rppt_transpose_host(RppPtr_t srcPtr,
                              RpptGenericDescPtr srcGenericDescPtr,
                              RppPtr_t dstPtr,
                              RpptGenericDescPtr dstGenericDescPtr,
                              Rpp32u *permTensor,
                              Rpp32u *roiTensor,
                              rppHandle_t rppHandle)
{
    if ((srcGenericDescPtr->dataType == RpptDataType::U8) && (dstGenericDescPtr->dataType == RpptDataType::U8))
    {
        transpose_generic_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                      srcGenericDescPtr,
                                      static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes,
                                      dstGenericDescPtr,
                                      permTensor,
                                      roiTensor,
                                      rpp::deref(rppHandle));
    }
    else if ((srcGenericDescPtr->dataType == RpptDataType::F16) && (dstGenericDescPtr->dataType == RpptDataType::F16))
    {
        transpose_generic_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                      srcGenericDescPtr,
                                      reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                      dstGenericDescPtr,
                                      permTensor,
                                      roiTensor,
                                      rpp::deref(rppHandle));
    }
    else if ((srcGenericDescPtr->dataType == RpptDataType::F32) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        transpose_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                      srcGenericDescPtr,
                                      reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                      dstGenericDescPtr,
                                      permTensor,
                                      roiTensor,
                                      rpp::deref(rppHandle));
    }
    else if ((srcGenericDescPtr->dataType == RpptDataType::I8) && (dstGenericDescPtr->dataType == RpptDataType::I8))
    {
        transpose_generic_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                      srcGenericDescPtr,
                                      static_cast<Rpp8s*>(dstPtr) + dstGenericDescPtr->offsetInBytes,
                                      dstGenericDescPtr,
                                      permTensor,
                                      roiTensor,
                                      rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

RppStatus rppt_warp_perspective_host(RppPtr_t srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     RppPtr_t dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32f *perspectiveTensor,
                                     RpptInterpolationType interpolationType,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     rppHandle_t rppHandle)
{
    if ((interpolationType != RpptInterpolationType::BILINEAR) && (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR))
        return RPP_ERROR_NOT_IMPLEMENTED;

    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if(interpolationType == RpptInterpolationType::NEAREST_NEIGHBOR)
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            warp_perspective_nn_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                  srcDescPtr,
                                                  static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                  dstDescPtr,
                                                  perspectiveTensor,
                                                  roiTensorPtrSrc,
                                                  roiType,
                                                  layoutParams,
                                                  rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            warp_perspective_nn_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                    srcDescPtr,
                                                    reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                    dstDescPtr,
                                                    perspectiveTensor,
                                                    roiTensorPtrSrc,
                                                    roiType,
                                                    layoutParams,
                                                    rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            warp_perspective_nn_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                  srcDescPtr,
                                                  static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                  dstDescPtr,
                                                  perspectiveTensor,
                                                  roiTensorPtrSrc,
                                                  roiType,
                                                  layoutParams,
                                                  rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            warp_perspective_nn_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                    srcDescPtr,
                                                    reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                    dstDescPtr,
                                                    perspectiveTensor,
                                                    roiTensorPtrSrc,
                                                    roiType,
                                                    layoutParams,
                                                    rpp::deref(rppHandle));
        }
    }
    else if(interpolationType == RpptInterpolationType::BILINEAR)
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            warp_perspective_bilinear_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                        srcDescPtr,
                                                        static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                        dstDescPtr,
                                                        perspectiveTensor,
                                                        roiTensorPtrSrc,
                                                        roiType,
                                                        layoutParams,
                                                        rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            warp_perspective_bilinear_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                          srcDescPtr,
                                                          reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                          dstDescPtr,
                                                          perspectiveTensor,
                                                          roiTensorPtrSrc,
                                                          roiType,
                                                          layoutParams,
                                                          rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            warp_perspective_bilinear_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                        srcDescPtr,
                                                        static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                        dstDescPtr,
                                                        perspectiveTensor,
                                                        roiTensorPtrSrc,
                                                        roiType,
                                                        layoutParams,
                                                        rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            warp_perspective_bilinear_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                          srcDescPtr,
                                                          reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                          dstDescPtr,
                                                          perspectiveTensor,
                                                          roiTensorPtrSrc,
                                                          roiType,
                                                          layoutParams,
                                                          rpp::deref(rppHandle));
        }
    }
    return RPP_SUCCESS;
}

/******************** concat_ND ********************/

RppStatus rppt_concat_host(RppPtr_t srcPtr1,
                           RppPtr_t srcPtr2,
                           RpptGenericDescPtr srcPtr1GenericDescPtr,
                           RpptGenericDescPtr srcPtr2GenericDescPtr,
                           RppPtr_t dstPtr,
                           RpptGenericDescPtr dstGenericDescPtr,
                           Rpp32u axisMask,
                           Rpp32u *roiTensorSrc1,
                           Rpp32u *roiTensorSrc2,
                           rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams;
    Rpp32u tensorDim = srcPtr1GenericDescPtr->numDims - 1;  // Ignoring batchSize here to get tensor dimensions.
    if (tensorDim == 3 && (srcPtr1GenericDescPtr->layout == RpptLayout::NHWC))
        layoutParams = get_layout_params(srcPtr1GenericDescPtr->layout, srcPtr1GenericDescPtr->dims[3]);
    else if ((srcPtr1GenericDescPtr->layout == RpptLayout::NCDHW) && (dstGenericDescPtr->layout == RpptLayout::NCDHW))
        layoutParams = get_layout_params(srcPtr1GenericDescPtr->layout, srcPtr1GenericDescPtr->dims[1]);
    else if ((srcPtr1GenericDescPtr->layout == RpptLayout::NDHWC) && (dstGenericDescPtr->layout == RpptLayout::NDHWC))
        layoutParams = get_layout_params(srcPtr1GenericDescPtr->layout, srcPtr1GenericDescPtr->dims[4]);
    else if(tensorDim == 2 && (srcPtr1GenericDescPtr->layout == RpptLayout::NHWC))
        layoutParams = get_layout_params(srcPtr1GenericDescPtr->layout, srcPtr1GenericDescPtr->dims[2]);

    if(srcPtr1GenericDescPtr->numDims != srcPtr2GenericDescPtr->numDims)
        return RPP_ERROR_INVALID_SRC_DIMS;
    if (srcPtr1GenericDescPtr->layout != dstGenericDescPtr->layout)
        return RPP_ERROR_LAYOUT_MISMATCH;
    if (axisMask >= srcPtr1GenericDescPtr->numDims)
        return RPP_ERROR_INVALID_AXIS;
    for(int i = 0 ;i < tensorDim ; i++)
    {
        if((i != axisMask) && (srcPtr1GenericDescPtr->dims[i] != srcPtr2GenericDescPtr->dims[i]))
            return RPP_ERROR_INVALID_DIM_LENGTHS;
    }
    if ((srcPtr1GenericDescPtr->dataType != srcPtr2GenericDescPtr->dataType) || (srcPtr1GenericDescPtr->dataType != dstGenericDescPtr->dataType))
        return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;

    if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U8) && (dstGenericDescPtr->dataType == RpptDataType::U8))
    {
        concat_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                                 static_cast<Rpp8u*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                                 srcPtr1GenericDescPtr,
                                 srcPtr2GenericDescPtr,
                                 static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes,
                                 dstGenericDescPtr,
                                 axisMask,
                                 roiTensorSrc1,
                                 roiTensorSrc2,
                                 layoutParams,
                                 rpp::deref(rppHandle));
    }
    else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::F16) && (dstGenericDescPtr->dataType == RpptDataType::F16))
    {
        concat_generic_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes),
                                   reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes),
                                   srcPtr1GenericDescPtr,
                                   srcPtr2GenericDescPtr,
                                   reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                   dstGenericDescPtr,
                                   axisMask,
                                   roiTensorSrc1,
                                   roiTensorSrc2,
                                   layoutParams,
                                   rpp::deref(rppHandle));
    }
    else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::F32) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        concat_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes),
                                   reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes),
                                   srcPtr1GenericDescPtr,
                                   srcPtr2GenericDescPtr,
                                   reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                   dstGenericDescPtr,
                                   axisMask,
                                   roiTensorSrc1,
                                   roiTensorSrc2,
                                   layoutParams,
                                   rpp::deref(rppHandle));
    }

    else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I8) && (dstGenericDescPtr->dataType == RpptDataType::I8))
    {
        concat_generic_host_tensor(static_cast<Rpp8s*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                                   static_cast<Rpp8s*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                                   srcPtr1GenericDescPtr,
                                   srcPtr2GenericDescPtr,
                                   static_cast<Rpp8s*>(dstPtr) + dstGenericDescPtr->offsetInBytes,
                                   dstGenericDescPtr,
                                   axisMask,
                                   roiTensorSrc1,
                                   roiTensorSrc2,
                                   layoutParams,
                                   rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}


/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** crop ********************/

RppStatus rppt_crop_gpu(RppPtr_t srcPtr,
                        RpptDescPtr srcDescPtr,
                        RppPtr_t dstPtr,
                        RpptDescPtr dstDescPtr,
                        RpptROIPtr roiTensorPtrSrc,
                        RpptRoiType roiType,
                        rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_crop_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                             srcDescPtr,
                             static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_crop_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                             srcDescPtr,
                             (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_crop_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                             srcDescPtr,
                             (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_crop_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                             srcDescPtr,
                             static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** crop mirror normalize ********************/

RppStatus rppt_crop_mirror_normalize_gpu(RppPtr_t srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         RppPtr_t dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *offsetTensor,
                                         Rpp32f *multiplierTensor,
                                         Rpp32u *mirrorTensor,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    Rpp32u paramIndex = 0;
    if(srcDescPtr->c == 3)
    {
        copy_param_float3(offsetTensor, rpp::deref(rppHandle), paramIndex++);
        copy_param_float3(multiplierTensor, rpp::deref(rppHandle), paramIndex++);
    }
    else if(srcDescPtr->c == 1)
    {
        copy_param_float(offsetTensor, rpp::deref(rppHandle), paramIndex++);
        copy_param_float(multiplierTensor, rpp::deref(rppHandle), paramIndex++);
    }
    copy_param_uint(mirrorTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_crop_mirror_normalize_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                              srcDescPtr,
                                              static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                              dstDescPtr,
                                              roiTensorPtrSrc,
                                              roiType,
                                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_crop_mirror_normalize_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                              srcDescPtr,
                                              (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                              dstDescPtr,
                                              roiTensorPtrSrc,
                                              roiType,
                                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_crop_mirror_normalize_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                              srcDescPtr,
                                              (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                              dstDescPtr,
                                              roiTensorPtrSrc,
                                              roiType,
                                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_crop_mirror_normalize_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                              srcDescPtr,
                                              static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                              dstDescPtr,
                                              roiTensorPtrSrc,
                                              roiType,
                                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_crop_mirror_normalize_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                              srcDescPtr,
                                              (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                              dstDescPtr,
                                              roiTensorPtrSrc,
                                              roiType,
                                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_crop_mirror_normalize_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                              srcDescPtr,
                                              (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                              dstDescPtr,
                                              roiTensorPtrSrc,
                                              roiType,
                                              rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** warp_affine ********************/

RppStatus rppt_warp_affine_gpu(RppPtr_t srcPtr,
                               RpptDescPtr srcDescPtr,
                               RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr,
                               Rpp32f *affineTensor,
                               RpptInterpolationType interpolationType,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((interpolationType != RpptInterpolationType::BILINEAR) && (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR))
        return RPP_ERROR_NOT_IMPLEMENTED;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_warp_affine_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr,
                                    affineTensor,
                                    interpolationType,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_warp_affine_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                    dstDescPtr,
                                    affineTensor,
                                    interpolationType,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_warp_affine_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                    dstDescPtr,
                                    affineTensor,
                                    interpolationType,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_warp_affine_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr,
                                    affineTensor,
                                    interpolationType,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

RppStatus rppt_warp_perspective_gpu(RppPtr_t srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    RppPtr_t dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32f *perspectiveTensor,
                                    RpptInterpolationType interpolationType,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((interpolationType != RpptInterpolationType::BILINEAR) && (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR))
        return RPP_ERROR_NOT_IMPLEMENTED;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_warp_perspective_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         perspectiveTensor,
                                         interpolationType,
                                         roiTensorPtrSrc,
                                         roiType,
                                         rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_warp_perspective_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                         dstDescPtr,
                                         perspectiveTensor,
                                         interpolationType,
                                         roiTensorPtrSrc,
                                         roiType,
                                         rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_warp_perspective_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                         dstDescPtr,
                                         perspectiveTensor,
                                         interpolationType,
                                         roiTensorPtrSrc,
                                         roiType,
                                         rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_warp_perspective_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         perspectiveTensor,
                                         interpolationType,
                                         roiTensorPtrSrc,
                                         roiType,
                                         rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** flip ********************/

RppStatus rppt_flip_gpu(RppPtr_t srcPtr,
                        RpptDescPtr srcDescPtr,
                        RppPtr_t dstPtr,
                        RpptDescPtr dstDescPtr,
                        Rpp32u *horizontalTensor,
                        Rpp32u *verticalTensor,
                        RpptROIPtr roiTensorPtrSrc,
                        RpptRoiType roiType,
                        rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    Rpp32u paramIndex = 0;
    copy_param_uint(horizontalTensor, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(verticalTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_flip_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                             srcDescPtr,
                             static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_flip_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                             srcDescPtr,
                             (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_flip_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                             srcDescPtr,
                             (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_flip_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                             srcDescPtr,
                             static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** resize_mirror_normalize ********************/

RppStatus rppt_resize_mirror_normalize_gpu(RppPtr_t srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           RppPtr_t dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           RpptImagePatchPtr dstImgSizes,
                                           RpptInterpolationType interpolationType,
                                           Rpp32f *meanTensor,
                                           Rpp32f *stdDevTensor,
                                           Rpp32u *mirrorTensor,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (interpolationType != RpptInterpolationType::BILINEAR)
        return RPP_ERROR_NOT_IMPLEMENTED;

    Rpp32u paramIndex = 0;
    if(srcDescPtr->c == 3)
    {
        copy_param_float3(meanTensor, rpp::deref(rppHandle), paramIndex++);
        copy_param_float3(stdDevTensor, rpp::deref(rppHandle), paramIndex++);
    }
    else if(srcDescPtr->c == 1)
    {
        copy_param_float(meanTensor, rpp::deref(rppHandle), paramIndex++);
        copy_param_float(stdDevTensor, rpp::deref(rppHandle), paramIndex++);
    }
    copy_param_uint(mirrorTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_resize_mirror_normalize_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                dstDescPtr,
                                                dstImgSizes,
                                                interpolationType,
                                                roiTensorPtrSrc,
                                                roiType,
                                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_resize_mirror_normalize_tensor((half*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                srcDescPtr,
                                                (half*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                dstImgSizes,
                                                interpolationType,
                                                roiTensorPtrSrc,
                                                roiType,
                                                rpp::deref(rppHandle));
    }

    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_resize_mirror_normalize_tensor((Rpp32f*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                srcDescPtr,
                                                (Rpp32f*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                dstImgSizes,
                                                interpolationType,
                                                roiTensorPtrSrc,
                                                roiType,
                                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_resize_mirror_normalize_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                dstDescPtr,
                                                dstImgSizes,
                                                interpolationType,
                                                roiTensorPtrSrc,
                                                roiType,
                                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_resize_mirror_normalize_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                (Rpp32f *)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                dstImgSizes,
                                                interpolationType,
                                                roiTensorPtrSrc,
                                                roiType,
                                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_resize_mirror_normalize_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                (half *)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                dstImgSizes,
                                                interpolationType,
                                                roiTensorPtrSrc,
                                                roiType,
                                                rpp::deref(rppHandle));
    }

return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** resize ********************/

RppStatus rppt_resize_gpu(RppPtr_t srcPtr,
                          RpptDescPtr srcDescPtr,
                          RppPtr_t dstPtr,
                          RpptDescPtr dstDescPtr,
                          RpptImagePatchPtr dstImgSizes,
                          RpptInterpolationType interpolationType,
                          RpptROIPtr roiTensorPtrSrc,
                          RpptRoiType roiType,
                          rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_resize_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               dstImgSizes,
                               interpolationType,
                               roiTensorPtrSrc,
                               roiType,
                               rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_resize_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                               srcDescPtr,
                               (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                               dstDescPtr,
                               dstImgSizes,
                               interpolationType,
                               roiTensorPtrSrc,
                               roiType,
                               rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_resize_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                               srcDescPtr,
                               (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                               dstDescPtr,
                               dstImgSizes,
                               interpolationType,
                               roiTensorPtrSrc,
                               roiType,
                               rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_resize_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               dstImgSizes,
                               interpolationType,
                               roiTensorPtrSrc,
                               roiType,
                               rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** resize_crop_mirror ********************/

RppStatus rppt_resize_crop_mirror_gpu(RppPtr_t srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      RppPtr_t dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptImagePatchPtr dstImgSizes,
                                      RpptInterpolationType interpolationType,
                                      Rpp32u *mirrorTensor,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (interpolationType != RpptInterpolationType::BILINEAR)
        return RPP_ERROR_NOT_IMPLEMENTED;

    copy_param_uint(mirrorTensor, rpp::deref(rppHandle), 0);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_resize_crop_mirror_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                           srcDescPtr,
                                           static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                           dstDescPtr,
                                           dstImgSizes,
                                           interpolationType,
                                           roiTensorPtrSrc,
                                           roiType,
                                           rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_resize_crop_mirror_tensor((half*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                           srcDescPtr,
                                           (half*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                           dstDescPtr,
                                           dstImgSizes,
                                           interpolationType,
                                           roiTensorPtrSrc,
                                           roiType,
                                           rpp::deref(rppHandle));
    }

    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_resize_crop_mirror_tensor((Rpp32f*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                           srcDescPtr,
                                           (Rpp32f*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                           dstDescPtr,
                                           dstImgSizes,
                                           interpolationType,
                                           roiTensorPtrSrc,
                                           roiType,
                                           rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_resize_crop_mirror_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                           srcDescPtr,
                                           static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                           dstDescPtr,
                                           dstImgSizes,
                                           interpolationType,
                                           roiTensorPtrSrc,
                                           roiType,
                                           rpp::deref(rppHandle));
}

return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** rotate ********************/

RppStatus rppt_rotate_gpu(RppPtr_t srcPtr,
                          RpptDescPtr srcDescPtr,
                          RppPtr_t dstPtr,
                          RpptDescPtr dstDescPtr,
                          Rpp32f *angle,
                          RpptInterpolationType interpolationType,
                          RpptROIPtr roiTensorPtrSrc,
                          RpptRoiType roiType,
                          rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((interpolationType != RpptInterpolationType::BILINEAR) && (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR))
        return RPP_ERROR_NOT_IMPLEMENTED;

    // Compute affine transformation matrix from rotate angle
    Rpp32f *affineTensor = rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.scratchBufferHost;
    for(int idx = 0; idx < srcDescPtr->n; idx++)
    {
        Rpp32f angleInRad = angle[idx] * PI_OVER_180;
        Rpp32f alpha, beta;
        sincosf(angleInRad, &beta, &alpha);
        ((Rpp32f6 *)affineTensor)[idx] = {alpha, -beta, 0, beta, alpha, 0};
    }

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_warp_affine_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr,
                                    affineTensor,
                                    interpolationType,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_warp_affine_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                    dstDescPtr,
                                    affineTensor,
                                    interpolationType,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_warp_affine_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                    dstDescPtr,
                                    affineTensor,
                                    interpolationType,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_warp_affine_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr,
                                    affineTensor,
                                    interpolationType,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** phase ********************/

RppStatus rppt_phase_gpu(RppPtr_t srcPtr1,
                         RppPtr_t srcPtr2,
                         RpptDescPtr srcDescPtr,
                         RppPtr_t dstPtr,
                         RpptDescPtr dstDescPtr,
                         RpptROIPtr roiTensorPtrSrc,
                         RpptRoiType roiType,
                         rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout == RpptLayout::NCDHW) || (srcDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout == RpptLayout::NCDHW) || (dstDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_phase_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                              static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                              srcDescPtr,
                              static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                              dstDescPtr,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_phase_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                              reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                              srcDescPtr,
                              reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                              dstDescPtr,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_phase_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                              reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                              srcDescPtr,
                              reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                              dstDescPtr,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_phase_tensor(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                              static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes,
                              srcDescPtr,
                              static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                              dstDescPtr,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** slice ********************/

RppStatus rppt_slice_gpu(RppPtr_t srcPtr,
                         RpptGenericDescPtr srcGenericDescPtr,
                         RppPtr_t dstPtr,
                         RpptGenericDescPtr dstGenericDescPtr,
                         Rpp32s *anchorTensor,
                         Rpp32s *shapeTensor,
                         RppPtr_t fillValue,
                         bool enablePadding,
                         Rpp32u *roiTensor,
                         rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((srcGenericDescPtr->dataType != RpptDataType::F32) && (srcGenericDescPtr->dataType != RpptDataType::U8)) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if ((dstGenericDescPtr->dataType != RpptDataType::F32) && (dstGenericDescPtr->dataType != RpptDataType::U8)) return RPP_ERROR_INVALID_DST_DATATYPE;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_LAYOUT_MISMATCH;

    if ((srcGenericDescPtr->dataType == RpptDataType::F32) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_slice_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                              srcGenericDescPtr,
                              reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                              dstGenericDescPtr,
                              anchorTensor,
                              shapeTensor,
                              static_cast<Rpp32f *>(fillValue),
                              enablePadding,
                              roiTensor,
                              rpp::deref(rppHandle));
    }
    else if ((srcGenericDescPtr->dataType == RpptDataType::U8) && (dstGenericDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_slice_tensor(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                              srcGenericDescPtr,
                              static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes,
                              dstGenericDescPtr,
                              anchorTensor,
                              shapeTensor,
                              static_cast<Rpp8u *>(fillValue),
                              enablePadding,
                              roiTensor,
                              rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** crop_and_patch ********************/

RppStatus rppt_crop_and_patch_gpu(RppPtr_t srcPtr1,
                                  RppPtr_t srcPtr2,
                                  RpptDescPtr srcDescPtr,
                                  RppPtr_t dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptROIPtr cropTensorPtr,
                                  RpptROIPtr patchTensorPtr,
                                  RpptRoiType roiType,
                                  rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_crop_and_patch_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                       static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       roiTensorPtrSrc,
                                       cropTensorPtr,
                                       patchTensorPtr,
                                       roiType,
                                       rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_crop_and_patch_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr1)) + srcDescPtr->offsetInBytes,
                                       reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr2)) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr)) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       roiTensorPtrSrc,
                                       cropTensorPtr,
                                       patchTensorPtr,
                                       roiType,
                                       rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_crop_and_patch_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1)) + srcDescPtr->offsetInBytes,
                                       reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2)) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr)) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       roiTensorPtrSrc,
                                       cropTensorPtr,
                                       patchTensorPtr,
                                       roiType,
                                       rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_crop_and_patch_tensor(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                       static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       roiTensorPtrSrc,
                                       cropTensorPtr,
                                       patchTensorPtr,
                                       roiType,
                                       rpp::deref(rppHandle));
    }
    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** flip_voxel ********************/

RppStatus rppt_flip_voxel_gpu(RppPtr_t srcPtr,
                              RpptGenericDescPtr srcGenericDescPtr,
                              RppPtr_t dstPtr,
                              RpptGenericDescPtr dstGenericDescPtr,
                              Rpp32u *horizontalTensor,
                              Rpp32u *verticalTensor,
                              Rpp32u *depthTensor,
                              RpptROI3DPtr roiGenericPtrSrc,
                              RpptRoi3DType roiType,
                              rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((srcGenericDescPtr->layout != RpptLayout::NCDHW) && (srcGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstGenericDescPtr->layout != RpptLayout::NCDHW) && (dstGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_INVALID_ARGUMENTS;
    if ((srcGenericDescPtr->dataType != RpptDataType::F32) && (srcGenericDescPtr->dataType != RpptDataType::U8)) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if ((dstGenericDescPtr->dataType != RpptDataType::F32) && (dstGenericDescPtr->dataType != RpptDataType::U8)) return RPP_ERROR_INVALID_DST_DATATYPE;

    if ((srcGenericDescPtr->dataType == RpptDataType::F32) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_flip_voxel_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                   srcGenericDescPtr,
                                   reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                   dstGenericDescPtr,
                                   roiGenericPtrSrc,
                                   horizontalTensor,
                                   verticalTensor,
                                   depthTensor,
                                   roiType,
                                   rpp::deref(rppHandle));
    }
    else if ((srcGenericDescPtr->dataType == RpptDataType::U8) && (dstGenericDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_flip_voxel_tensor(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                   srcGenericDescPtr,
                                   static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes,
                                   dstGenericDescPtr,
                                   roiGenericPtrSrc,
                                   horizontalTensor,
                                   verticalTensor,
                                   depthTensor,
                                   roiType,
                                   rpp::deref(rppHandle));
    }
    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** remap ********************/

RppStatus rppt_remap_gpu(RppPtr_t srcPtr,
                         RpptDescPtr srcDescPtr,
                         RppPtr_t dstPtr,
                         RpptDescPtr dstDescPtr,
                         Rpp32f *rowRemapTable,
                         Rpp32f *colRemapTable,
                         RpptDescPtr tableDescPtr,
                         RpptInterpolationType interpolationType,
                         RpptROIPtr roiTensorPtrSrc,
                         RpptRoiType roiType,
                         rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR && interpolationType != RpptInterpolationType::BILINEAR)
        return RPP_ERROR_NOT_IMPLEMENTED;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_remap_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                              srcDescPtr,
                              static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                              dstDescPtr,
                              rowRemapTable,
                              colRemapTable,
                              tableDescPtr,
                              interpolationType,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_remap_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                              srcDescPtr,
                              reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                              dstDescPtr,
                              rowRemapTable,
                              colRemapTable,
                              tableDescPtr,
                              interpolationType,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_remap_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                              srcDescPtr,
                              reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                              dstDescPtr,
                              rowRemapTable,
                              colRemapTable,
                              tableDescPtr,
                              interpolationType,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_remap_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                              srcDescPtr,
                              static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                              dstDescPtr,
                              rowRemapTable,
                              colRemapTable,
                              tableDescPtr,
                              interpolationType,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** lens_correction ********************/

RppStatus rppt_lens_correction_gpu(RppPtr_t srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   RppPtr_t dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32f *rowRemapTable,
                                   Rpp32f *colRemapTable,
                                   RpptDescPtr tableDescPtr,
                                   Rpp32f *cameraMatrixTensor,
                                   Rpp32f *distortionCoeffsTensor,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    hip_exec_lens_correction_tensor(dstDescPtr,
                                    rowRemapTable,
                                    colRemapTable,
                                    tableDescPtr,
                                    cameraMatrixTensor,
                                    distortionCoeffsTensor,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_remap_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                              srcDescPtr,
                              static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                              dstDescPtr,
                              rowRemapTable,
                              colRemapTable,
                              tableDescPtr,
                              RpptInterpolationType::BILINEAR,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_remap_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                              srcDescPtr,
                              reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                              dstDescPtr,
                              rowRemapTable,
                              colRemapTable,
                              tableDescPtr,
                              RpptInterpolationType::BILINEAR,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_remap_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                              srcDescPtr,
                              reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                              dstDescPtr,
                              rowRemapTable,
                              colRemapTable,
                              tableDescPtr,
                              RpptInterpolationType::BILINEAR,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_remap_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                              srcDescPtr,
                              static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                              dstDescPtr,
                              rowRemapTable,
                              colRemapTable,
                              tableDescPtr,
                              RpptInterpolationType::BILINEAR,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** transpose ********************/

RppStatus rppt_transpose_gpu(RppPtr_t srcPtr,
                             RpptGenericDescPtr srcGenericDescPtr,
                             RppPtr_t dstPtr,
                             RpptGenericDescPtr dstGenericDescPtr,
                             Rpp32u *permTensor,
                             Rpp32u *roiTensor,
                             rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((srcGenericDescPtr->dataType == RpptDataType::U8) && (dstGenericDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_transpose_tensor(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                  srcGenericDescPtr,
                                  static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes,
                                  dstGenericDescPtr,
                                  permTensor,
                                  roiTensor,
                                  rpp::deref(rppHandle));
    }
    else if ((srcGenericDescPtr->dataType == RpptDataType::F16) && (dstGenericDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_transpose_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                  srcGenericDescPtr,
                                  reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                  dstGenericDescPtr,
                                  permTensor,
                                  roiTensor,
                                  rpp::deref(rppHandle));
    }
    else if ((srcGenericDescPtr->dataType == RpptDataType::F32) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_transpose_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                  srcGenericDescPtr,
                                  reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                  dstGenericDescPtr,
                                  permTensor,
                                  roiTensor,
                                  rpp::deref(rppHandle));
    }
    else if ((srcGenericDescPtr->dataType == RpptDataType::I8) && (dstGenericDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_transpose_tensor(static_cast<Rpp8s*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                  srcGenericDescPtr,
                                  static_cast<Rpp8s*>(dstPtr) + dstGenericDescPtr->offsetInBytes,
                                  dstGenericDescPtr,
                                  permTensor,
                                  roiTensor,
                                  rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** concat ********************/

RppStatus rppt_concat_gpu(RppPtr_t srcPtr1,
                          RppPtr_t srcPtr2,
                          RpptGenericDescPtr srcPtr1GenericDescPtr,
                          RpptGenericDescPtr srcPtr2GenericDescPtr,
                          RppPtr_t dstPtr,
                          RpptGenericDescPtr dstGenericDescPtr,
                          Rpp32u axis,
                          Rpp32u *roiTensorSrc1,
                          Rpp32u *roiTensorSrc2,
                          rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE

    Rpp32u tensorDim = srcPtr1GenericDescPtr->numDims - 1;  // Ignoring batchSize here to get tensor dimensions.
    if(srcPtr1GenericDescPtr->numDims != srcPtr2GenericDescPtr->numDims)
        return RPP_ERROR_INVALID_SRC_DIMS;
    if (srcPtr1GenericDescPtr->layout != dstGenericDescPtr->layout)
        return RPP_ERROR_LAYOUT_MISMATCH;
    if (axis >= srcPtr1GenericDescPtr->numDims)
        return RPP_ERROR_INVALID_AXIS;
    for(int i = 0;i < tensorDim ; i++)
    {
        if((i != axis) && (srcPtr1GenericDescPtr->dims[i] != srcPtr2GenericDescPtr->dims[i]))
            return RPP_ERROR_INVALID_DIM_LENGTHS;
    }
    if((srcPtr1GenericDescPtr->dataType != srcPtr2GenericDescPtr->dataType) || (srcPtr1GenericDescPtr->dataType != dstGenericDescPtr->dataType))
        return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;

    if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U8) && (dstGenericDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_concat_tensor(static_cast<Rpp8u*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                               srcPtr1GenericDescPtr,
                               static_cast<Rpp8u*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                               srcPtr2GenericDescPtr,
                               static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes,
                               dstGenericDescPtr,
                               axis,
                               roiTensorSrc1,
                               rpp::deref(rppHandle));
    }
    else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::F16) && (dstGenericDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_concat_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes),
                               srcPtr1GenericDescPtr,
                               reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes),
                               srcPtr2GenericDescPtr,
                               reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                               dstGenericDescPtr,
                               axis,
                               roiTensorSrc1,
                               rpp::deref(rppHandle));
    }
    else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::F32) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_concat_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes),
                               srcPtr1GenericDescPtr,
                               reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes),
                               srcPtr2GenericDescPtr,
                               reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                               dstGenericDescPtr,
                               axis,
                               roiTensorSrc1,
                               rpp::deref(rppHandle));
    }
    else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::I8) && (dstGenericDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_concat_tensor(static_cast<Rpp8s*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                               srcPtr1GenericDescPtr,
                               static_cast<Rpp8s*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                               srcPtr2GenericDescPtr,
                               static_cast<Rpp8s*>(dstPtr) + dstGenericDescPtr->offsetInBytes,
                               dstGenericDescPtr,
                               axis,
                               roiTensorSrc1,
                               rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}


#endif // GPU_SUPPORT