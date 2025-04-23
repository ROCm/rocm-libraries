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
#include "rppt_tensor_color_augmentations.h"
#include "host_tensor_executors.hpp"

#ifdef HIP_COMPILE
#include "hip_tensor_executors.hpp"
#endif // HIP_COMPILE

/******************** brightness ********************/

RppStatus rppt_brightness_host(RppPtr_t srcPtr,
                               RpptDescPtr srcDescPtr,
                               RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr,
                               Rpp32f *alphaTensor,
                               Rpp32f *betaTensor,
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
        brightness_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                     dstDescPtr,
                                     alphaTensor,
                                     betaTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams,
                                     rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        brightness_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                       dstDescPtr,
                                       alphaTensor,
                                       betaTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams,
                                       rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        brightness_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                       dstDescPtr,
                                       alphaTensor,
                                       betaTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams,
                                       rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        brightness_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                     dstDescPtr,
                                     alphaTensor,
                                     betaTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams,
                                     rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** gamma_correction ********************/

RppStatus rppt_gamma_correction_host(RppPtr_t srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     RppPtr_t dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32f *gammaTensor,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        gamma_correction_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                           srcDescPtr,
                                           static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                           dstDescPtr,
                                           gammaTensor,
                                           roiTensorPtrSrc,
                                           roiType,
                                           layoutParams,
                                           rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        gamma_correction_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                             srcDescPtr,
                                             (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                             dstDescPtr,
                                             gammaTensor,
                                             roiTensorPtrSrc,
                                             roiType,
                                             layoutParams,
                                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        gamma_correction_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                             srcDescPtr,
                                             (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                             dstDescPtr,
                                             gammaTensor,
                                             roiTensorPtrSrc,
                                             roiType,
                                             layoutParams,
                                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        gamma_correction_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                           srcDescPtr,
                                           static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                           dstDescPtr,
                                           gammaTensor,
                                           roiTensorPtrSrc,
                                           roiType,
                                           layoutParams,
                                           rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** blend ********************/

RppStatus rppt_blend_host(RppPtr_t srcPtr1,
                          RppPtr_t srcPtr2,
                          RpptDescPtr srcDescPtr,
                          RppPtr_t dstPtr,
                          RpptDescPtr dstDescPtr,
                          Rpp32f *alphaTensor,
                          RpptROIPtr roiTensorPtrSrc,
                          RpptRoiType roiType,
                          rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        blend_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                dstDescPtr,
                                alphaTensor,
                                roiTensorPtrSrc,
                                roiType,
                                layoutParams,
                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        blend_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                  (Rpp16f*) (static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                  srcDescPtr,
                                  (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                  dstDescPtr,
                                  alphaTensor,
                                  roiTensorPtrSrc,
                                  roiType,
                                  layoutParams,
                                  rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        blend_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                  (Rpp32f*) (static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                  srcDescPtr,
                                  (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                  dstDescPtr,
                                  alphaTensor,
                                  roiTensorPtrSrc,
                                  roiType,
                                  layoutParams,
                                  rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        blend_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                dstDescPtr,
                                alphaTensor,
                                roiTensorPtrSrc,
                                roiType,
                                layoutParams,
                                rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** color_twist ********************/

RppStatus rppt_color_twist_host(RppPtr_t srcPtr,
                                RpptDescPtr srcDescPtr,
                                RppPtr_t dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32f *brightnessTensor,
                                Rpp32f *contrastTensor,
                                Rpp32f *hueTensor,
                                Rpp32f *saturationTensor,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        color_twist_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                      srcDescPtr,
                                      static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                      dstDescPtr,
                                      brightnessTensor,
                                      contrastTensor,
                                      hueTensor,
                                      saturationTensor,
                                      roiTensorPtrSrc,
                                      roiType,
                                      layoutParams,
                                      rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        color_twist_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                        srcDescPtr,
                                        (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                        dstDescPtr,
                                        brightnessTensor,
                                        contrastTensor,
                                        hueTensor,
                                        saturationTensor,
                                        roiTensorPtrSrc,
                                        roiType,
                                        layoutParams,
                                        rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        color_twist_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                        srcDescPtr,
                                        (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                        dstDescPtr,
                                        brightnessTensor,
                                        contrastTensor,
                                        hueTensor,
                                        saturationTensor,
                                        roiTensorPtrSrc,
                                        roiType,
                                        layoutParams,
                                        rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        color_twist_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                      srcDescPtr,
                                      static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                      dstDescPtr,
                                      brightnessTensor,
                                      contrastTensor,
                                      hueTensor,
                                      saturationTensor,
                                      roiTensorPtrSrc,
                                      roiType,
                                      layoutParams,
                                      rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** color_jitter ********************/

RppStatus rppt_color_jitter_host(RppPtr_t srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 RppPtr_t dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp32f *brightnessTensor,
                                 Rpp32f *contrastTensor,
                                 Rpp32f *hueTensor,
                                 Rpp32f *saturationTensor,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        color_jitter_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       brightnessTensor,
                                       contrastTensor,
                                       hueTensor,
                                       saturationTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams,
                                       rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        color_jitter_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                         dstDescPtr,
                                         brightnessTensor,
                                         contrastTensor,
                                         hueTensor,
                                         saturationTensor,
                                         roiTensorPtrSrc,
                                         roiType,
                                         layoutParams,
                                         rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        color_jitter_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                         dstDescPtr,
                                         brightnessTensor,
                                         contrastTensor,
                                         hueTensor,
                                         saturationTensor,
                                         roiTensorPtrSrc,
                                         roiType,
                                         layoutParams,
                                         rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        color_jitter_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       brightnessTensor,
                                       contrastTensor,
                                       hueTensor,
                                       saturationTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams,
                                       rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** color_cast ********************/

RppStatus rppt_color_cast_host(RppPtr_t srcPtr,
                               RpptDescPtr srcDescPtr,
                               RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr,
                               RpptRGB *rgbTensor,
                               Rpp32f *alphaTensor,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rppHandle_t rppHandle)
{
    if (srcDescPtr->c != 3)
    {
        return RPP_ERROR_INVALID_CHANNELS;
    }

    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        color_cast_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                     dstDescPtr,
                                     rgbTensor,
                                     alphaTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams,
                                     rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        color_cast_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                       dstDescPtr,
                                       rgbTensor,
                                       alphaTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams,
                                       rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        color_cast_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                       dstDescPtr,
                                       rgbTensor,
                                       alphaTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams,
                                       rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        color_cast_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                     dstDescPtr,
                                     rgbTensor,
                                     alphaTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams,
                                     rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** exposure ********************/

RppStatus rppt_exposure_host(RppPtr_t srcPtr,
                             RpptDescPtr srcDescPtr,
                             RppPtr_t dstPtr,
                             RpptDescPtr dstDescPtr,
                             Rpp32f *exposureFactorTensor,
                             RpptROIPtr roiTensorPtrSrc,
                             RpptRoiType roiType,
                             rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        exposure_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr,
                                   exposureFactorTensor,
                                   roiTensorPtrSrc,
                                   roiType,
                                   layoutParams,
                                   rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        exposure_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                     dstDescPtr,
                                     exposureFactorTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams,
                                     rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        exposure_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                     dstDescPtr,
                                     exposureFactorTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams,
                                     rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        exposure_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr,
                                   exposureFactorTensor,
                                   roiTensorPtrSrc,
                                   roiType,
                                   layoutParams,
                                   rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** contrast ********************/

RppStatus rppt_contrast_host(RppPtr_t srcPtr,
                             RpptDescPtr srcDescPtr,
                             RppPtr_t dstPtr,
                             RpptDescPtr dstDescPtr,
                             Rpp32f *contrastFactorTensor,
                             Rpp32f *contrastCenterTensor,
                             RpptROIPtr roiTensorPtrSrc,
                             RpptRoiType roiType,
                             rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        contrast_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr,
                                   contrastFactorTensor,
                                   contrastCenterTensor,
                                   roiTensorPtrSrc,
                                   roiType,
                                   layoutParams,
                                   rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        contrast_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                     dstDescPtr,
                                     contrastFactorTensor,
                                     contrastCenterTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams,
                                     rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        contrast_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                     dstDescPtr,
                                     contrastFactorTensor,
                                     contrastCenterTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams,
                                     rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        contrast_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr,
                                   contrastFactorTensor,
                                   contrastCenterTensor,
                                   roiTensorPtrSrc,
                                   roiType,
                                   layoutParams,
                                   rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** lut ********************/

RppStatus rppt_lut_host(RppPtr_t srcPtr,
                        RpptDescPtr srcDescPtr,
                        RppPtr_t dstPtr,
                        RpptDescPtr dstDescPtr,
                        RppPtr_t lutPtr,
                        RpptROIPtr roiTensorPtrSrc,
                        RpptRoiType roiType,
                        rppHandle_t rppHandle)
{
    if (srcDescPtr->dataType != RpptDataType::U8 && srcDescPtr->dataType != RpptDataType::I8)
        return RPP_ERROR_INVALID_SRC_DATATYPE;

    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        lut_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                              srcDescPtr,
                              static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                              dstDescPtr,
                              static_cast<Rpp8u*>(lutPtr),
                              roiTensorPtrSrc,
                              roiType,
                              layoutParams);
    }
    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        lut_u8_f16_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp16f*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               static_cast<Rpp16f*>(lutPtr),
                               roiTensorPtrSrc,
                               roiType,
                               layoutParams);
    }
    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        lut_u8_f32_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp32f*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               static_cast<Rpp32f*>(lutPtr),
                               roiTensorPtrSrc,
                               roiType,
                               layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        lut_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                              srcDescPtr,
                              static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                              dstDescPtr,
                              static_cast<Rpp8s*>(lutPtr),
                              roiTensorPtrSrc,
                              roiType,
                              layoutParams);
    }

    return RPP_SUCCESS;
}

/******************** color_temperature ********************/

RppStatus rppt_color_temperature_host(RppPtr_t srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      RppPtr_t dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32s *adjustmentValueTensor,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      rppHandle_t rppHandle)
{
    if (srcDescPtr->c != 3)
    {
        return RPP_ERROR_INVALID_CHANNELS;
    }

    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        color_temperature_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                            srcDescPtr,
                                            static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                            dstDescPtr,
                                            adjustmentValueTensor,
                                            roiTensorPtrSrc,
                                            roiType,
                                            layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        color_temperature_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                              srcDescPtr,
                                              reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                              dstDescPtr,
                                              adjustmentValueTensor,
                                              roiTensorPtrSrc,
                                              roiType,
                                              layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        color_temperature_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                              srcDescPtr,
                                              reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                              dstDescPtr,
                                              adjustmentValueTensor,
                                              roiTensorPtrSrc,
                                              roiType,
                                              layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        color_temperature_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                            srcDescPtr,
                                            static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                            dstDescPtr,
                                            adjustmentValueTensor,
                                            roiTensorPtrSrc,
                                            roiType,
                                            layoutParams);
    }

    return RPP_SUCCESS;
}

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** brightness ********************/

RppStatus rppt_brightness_gpu(RppPtr_t srcPtr,
                              RpptDescPtr srcDescPtr,
                              RppPtr_t dstPtr,
                              RpptDescPtr dstDescPtr,
                              Rpp32f *alphaTensor,
                              Rpp32f *betaTensor,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    Rpp32u paramIndex = 0;
    copy_param_float(alphaTensor, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(betaTensor, rpp::deref(rppHandle), paramIndex++);
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout == RpptLayout::NCDHW) || (srcDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout == RpptLayout::NCDHW) || (dstDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_brightness_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr,
                                   roiTensorPtrSrc,
                                   roiType,
                                   rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_brightness_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                   srcDescPtr,
                                   (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                   dstDescPtr,
                                   roiTensorPtrSrc,
                                   roiType,
                                   rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_brightness_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                   srcDescPtr,
                                   (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                   dstDescPtr,
                                   roiTensorPtrSrc,
                                   roiType,
                                   rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_brightness_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
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

/******************** gamma_correction ********************/

RppStatus rppt_gamma_correction_gpu(RppPtr_t srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    RppPtr_t dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32f *gammaTensor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    Rpp32u paramIndex = 0;
    copy_param_float(gammaTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_gamma_correction_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         roiTensorPtrSrc,
                                         roiType,
                                         rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_gamma_correction_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                         dstDescPtr,
                                         roiTensorPtrSrc,
                                         roiType,
                                         rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_gamma_correction_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                         dstDescPtr,
                                         roiTensorPtrSrc,
                                         roiType,
                                         rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_gamma_correction_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
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

/******************** blend ********************/

RppStatus rppt_blend_gpu(RppPtr_t srcPtr1,
                         RppPtr_t srcPtr2,
                         RpptDescPtr srcDescPtr,
                         RppPtr_t dstPtr,
                         RpptDescPtr dstDescPtr,
                         Rpp32f *alphaTensor,
                         RpptROIPtr roiTensorPtrSrc,
                         RpptRoiType roiType,
                         rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    Rpp32u paramIndex = 0;
    copy_param_float(alphaTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_blend_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
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
        hip_exec_blend_tensor((half*) (static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                              (half*) (static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                              srcDescPtr,
                              (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                              dstDescPtr,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_blend_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                              (Rpp32f*) (static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                              srcDescPtr,
                              (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                              dstDescPtr,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_blend_tensor(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
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

/******************** color_twist ********************/

RppStatus rppt_color_twist_gpu(RppPtr_t srcPtr,
                               RpptDescPtr srcDescPtr,
                               RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr,
                               Rpp32f *brightnessTensor,
                               Rpp32f *contrastTensor,
                               Rpp32f *hueTensor,
                               Rpp32f *saturationTensor,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcDescPtr->c != 3)
    {
        return RPP_ERROR_INVALID_CHANNELS;
    }

    Rpp32u paramIndex = 0;
    copy_param_float(brightnessTensor, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(contrastTensor, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(hueTensor, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(saturationTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_color_twist_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_color_twist_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                    dstDescPtr,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_color_twist_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                    dstDescPtr,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_color_twist_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
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

/******************** color_cast ********************/

RppStatus rppt_color_cast_gpu(RppPtr_t srcPtr,
                              RpptDescPtr srcDescPtr,
                              RppPtr_t dstPtr,
                              RpptDescPtr dstDescPtr,
                              RpptRGB *rgbTensor,
                              Rpp32f *alphaTensor,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcDescPtr->c != 3)
    {
        return RPP_ERROR_INVALID_CHANNELS;
    }

    Rpp32u paramIndex = 0;
    copy_param_float(alphaTensor, rpp::deref(rppHandle), paramIndex++);
    copy_param_RpptRGB(rgbTensor, rpp::deref(rppHandle));

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_color_cast_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr,
                                   roiTensorPtrSrc,
                                   roiType,
                                   rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_color_cast_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                   srcDescPtr,
                                   (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                   dstDescPtr,
                                   roiTensorPtrSrc,
                                   roiType,
                                   rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_color_cast_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                   srcDescPtr,
                                   (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                   dstDescPtr,
                                   roiTensorPtrSrc,
                                   roiType,
                                   rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_color_cast_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
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

/******************** exposure ********************/

RppStatus rppt_exposure_gpu(RppPtr_t srcPtr,
                            RpptDescPtr srcDescPtr,
                            RppPtr_t dstPtr,
                            RpptDescPtr dstDescPtr,
                            Rpp32f *exposureFactorTensor,
                            RpptROIPtr roiTensorPtrSrc,
                            RpptRoiType roiType,
                            rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    Rpp32u paramIndex = 0;
    copy_param_float(exposureFactorTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_exposure_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                 srcDescPtr,
                                 static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                 dstDescPtr,
                                 roiTensorPtrSrc,
                                 roiType,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_exposure_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 roiTensorPtrSrc,
                                 roiType,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_exposure_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 roiTensorPtrSrc,
                                 roiType,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_exposure_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
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

/******************** contrast ********************/

RppStatus rppt_contrast_gpu(RppPtr_t srcPtr,
                            RpptDescPtr srcDescPtr,
                            RppPtr_t dstPtr,
                            RpptDescPtr dstDescPtr,
                            Rpp32f *contrastFactorTensor,
                            Rpp32f *contrastCenterTensor,
                            RpptROIPtr roiTensorPtrSrc,
                            RpptRoiType roiType,
                            rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    Rpp32u paramIndex = 0;
    copy_param_float(contrastFactorTensor, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(contrastCenterTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_contrast_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                 srcDescPtr,
                                 static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                 dstDescPtr,
                                 roiTensorPtrSrc,
                                 roiType,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_contrast_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 roiTensorPtrSrc,
                                 roiType,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_contrast_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 roiTensorPtrSrc,
                                 roiType,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_contrast_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
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

/******************** lut ********************/

RppStatus rppt_lut_gpu(RppPtr_t srcPtr,
                       RpptDescPtr srcDescPtr,
                       RppPtr_t dstPtr,
                       RpptDescPtr dstDescPtr,
                       RppPtr_t lutPtr,
                       RpptROIPtr roiTensorPtrSrc,
                       RpptRoiType roiType,
                       rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcDescPtr->dataType != RpptDataType::U8 && srcDescPtr->dataType != RpptDataType::I8)
        return RPP_ERROR_INVALID_SRC_DATATYPE;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_lut_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                            srcDescPtr,
                            static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                            dstDescPtr,
                            static_cast<Rpp8u*>(lutPtr),
                            roiTensorPtrSrc,
                            roiType,
                            rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_lut_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                            srcDescPtr,
                            (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                            dstDescPtr,
                            static_cast<half*>(lutPtr),
                            roiTensorPtrSrc,
                            roiType,
                            rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_lut_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                            srcDescPtr,
                            (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                            dstDescPtr,
                            static_cast<Rpp32f*>(lutPtr),
                            roiTensorPtrSrc,
                            roiType,
                            rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_lut_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                            srcDescPtr,
                            static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                            dstDescPtr,
                            static_cast<Rpp8s*>(lutPtr),
                            roiTensorPtrSrc,
                            roiType,
                            rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** color_temperature ********************/

RppStatus rppt_color_temperature_gpu(RppPtr_t srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     RppPtr_t dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32s *adjustmentValueTensor,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcDescPtr->c != 3)
    {
        return RPP_ERROR_INVALID_CHANNELS;
    }

    Rpp32u paramIndex = 0;
    copy_param_int(adjustmentValueTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_color_temperature_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                          srcDescPtr,
                                          static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                          dstDescPtr,
                                          roiTensorPtrSrc,
                                          roiType,
                                          rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_color_temperature_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                          srcDescPtr,
                                          reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                          dstDescPtr,
                                          roiTensorPtrSrc,
                                          roiType,
                                          rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_color_temperature_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                          srcDescPtr,
                                          reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                          dstDescPtr,
                                          roiTensorPtrSrc,
                                          roiType,
                                          rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_color_temperature_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
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

#endif // GPU_SUPPORT
