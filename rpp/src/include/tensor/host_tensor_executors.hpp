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

#ifndef HOST_TENSOR_EXECUTORS_HPP
#define HOST_TENSOR_EXECUTORS_HPP

#include "rppdefs.h"
#include "rpp_cpu_common.hpp"

/**************************************** ARITHMETIC OPERATIONS ****************************************/

// -------------------- add_scalar --------------------

RppStatus add_scalar_f32_f32_host_tensor(Rpp32f *srcPtr,
                                         RpptGenericDescPtr srcGenericDescPtr,
                                         Rpp32f *dstPtr,
                                         RpptGenericDescPtr dstGenericDescPtr,
                                         Rpp32f *addTensor,
                                         RpptROI3DPtr roiGenericPtrSrc,
                                         RpptRoi3DType roiType,
                                         RppLayoutParams layoutParams,
                                         rpp::Handle& handle);

// -------------------- fused_multiply_add_scalar --------------------

RppStatus fused_multiply_add_scalar_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                        RpptGenericDescPtr srcGenericDescPtr,
                                                        Rpp32f *dstPtr,
                                                        RpptGenericDescPtr dstGenericDescPtr,
                                                        Rpp32f *mulTensor,
                                                        Rpp32f *addTensor,
                                                        RpptROI3DPtr roiGenericPtrSrc,
                                                        RpptRoi3DType roiType,
                                                        RppLayoutParams layoutParams,
                                                        rpp::Handle& handle);

// -------------------- log_generic --------------------

RppStatus log_generic_host_tensor(Rpp8u *srcPtr,
                                  RpptGenericDescPtr srcGenericDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptGenericDescPtr dstGenericDescPtr,
                                  Rpp32u *roiTensor,
                                  rpp::Handle& handle);

RppStatus log_generic_host_tensor(Rpp8s *srcPtr,
                                  RpptGenericDescPtr srcGenericDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptGenericDescPtr dstGenericDescPtr,
                                  Rpp32u *roiTensor,
                                  rpp::Handle& handle);

RppStatus log_generic_host_tensor(Rpp32f *srcPtr,
                                  RpptGenericDescPtr srcGenericDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptGenericDescPtr dstGenericDescPtr,
                                  Rpp32u *roiTensor,
                                  rpp::Handle& handle);

RppStatus log_generic_host_tensor(Rpp16f *srcPtr,
                                  RpptGenericDescPtr srcGenericDescPtr,
                                  Rpp16f *dstPtr,
                                  RpptGenericDescPtr dstGenericDescPtr,
                                  Rpp32u *roiTensor,
                                  rpp::Handle& handle);

// -------------------- log1p --------------------

RppStatus log1p_i16_f32_host_tensor(Rpp16s *srcPtr,
                                    RpptGenericDescPtr srcGenericDescPtr,
                                    Rpp32f *dstPtr,
                                    RpptGenericDescPtr dstGenericDescPtr,
                                    Rpp32u *roiTensor,
                                    rpp::Handle& handle);

// -------------------- magnitude --------------------

RppStatus magnitude_u8_u8_host_tensor(Rpp8u *srcPtr1,
                                      Rpp8u *srcPtr2,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8u *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams layoutParams,
                                      rpp::Handle& handle);

RppStatus magnitude_f32_f32_host_tensor(Rpp32f *srcPtr1,
                                        Rpp32f *srcPtr2,
                                        RpptDescPtr srcDescPtr,
                                        Rpp32f *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle);

RppStatus magnitude_f16_f16_host_tensor(Rpp16f *srcPtr1,
                                        Rpp16f *srcPtr2,
                                        RpptDescPtr srcDescPtr,
                                        Rpp16f *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle);

RppStatus magnitude_i8_i8_host_tensor(Rpp8s *srcPtr1,
                                      Rpp8s *srcPtr2,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8s *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams layoutParams,
                                      rpp::Handle& handle);

// -------------------- multiply_scalar --------------------

RppStatus multiply_scalar_f32_f32_host_tensor(Rpp32f *srcPtr,
                                              RpptGenericDescPtr srcGenericDescPtr,
                                              Rpp32f *dstPtr,
                                              RpptGenericDescPtr dstGenericDescPtr,
                                              Rpp32f *mulTensor,
                                              RpptROI3DPtr roiGenericPtrSrc,
                                              RpptRoi3DType roiType,
                                              RppLayoutParams layoutParams,
                                              rpp::Handle& handle);

// -------------------- subtract_scalar --------------------

RppStatus subtract_scalar_f32_f32_host_tensor(Rpp32f *srcPtr,
                                              RpptGenericDescPtr srcGenericDescPtr,
                                              Rpp32f *dstPtr,
                                              RpptGenericDescPtr dstGenericDescPtr,
                                              Rpp32f *subtractTensor,
                                              RpptROI3DPtr roiGenericPtrSrc,
                                              RpptRoi3DType roiType,
                                              RppLayoutParams layoutParams,
                                              rpp::Handle& handle);

/**************************************** AUDIO AUGMENTATIONS ****************************************/

#ifdef AUDIO_SUPPORT

// -------------------- down_mixing --------------------

RppStatus down_mixing_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32s *srcDimsTensor,
                                  bool normalizeWeights,
                                  rpp::Handle& handle);

// -------------------- mel_filter_bank --------------------

RppStatus mel_filter_bank_host_tensor(Rpp32f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32s *srcDimsTensor,
                                      Rpp32f maxFreqVal,    // check unused
                                      Rpp32f minFreqVal,
                                      RpptMelScaleFormula melFormula,
                                      Rpp32s numFilter,
                                      Rpp32f sampleRate,
                                      bool normalize,
                                      rpp::Handle& handle);

// -------------------- non_silent_region_detection --------------------

RppStatus non_silent_region_detection_host_tensor(Rpp32f *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp32s *srcLengthTensor,
                                                  Rpp32s *detectedIndexTensor,
                                                  Rpp32s *detectionLengthTensor,
                                                  Rpp32f cutOffDB,
                                                  Rpp32s windowLength,
                                                  Rpp32f referencePower,
                                                  Rpp32s resetInterval,
                                                  rpp::Handle& handle);

// -------------------- pre_emphasis_filter --------------------

RppStatus pre_emphasis_filter_host_tensor(Rpp32f *srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          Rpp32f *dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          Rpp32s *srcLengthTensor,
                                          Rpp32f *coeffTensor,
                                          Rpp32u borderType,
                                          rpp::Handle& handle);

// -------------------- resample --------------------

RppStatus resample_host_tensor(Rpp32f *srcPtr,
                               RpptDescPtr srcDescPtr,
                               Rpp32f *dstPtr,
                               RpptDescPtr dstDescPtr,
                               Rpp32f *inRateTensor,
                               Rpp32f *outRateTensor,
                               Rpp32s *srcDimsTensor,
                               RpptResamplingWindow &window,
                               rpp::Handle& handle);

// -------------------- spectrogram --------------------

RppStatus spectrogram_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32s *srcLengthTensor,
                                  bool centerWindows,
                                  bool reflectPadding,
                                  Rpp32f *windowFunction,
                                  Rpp32s nfft,
                                  Rpp32s power,
                                  Rpp32s windowLength,
                                  Rpp32s windowStep,
                                  rpp::Handle& handle);

// -------------------- to_decibels --------------------

RppStatus to_decibels_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  RpptImagePatchPtr srcDims,
                                  Rpp32f cutOffDB,
                                  Rpp32f multiplier,
                                  Rpp32f referenceMagnitude,
                                  rpp::Handle& handle);

#endif // AUDIO_SUPPORT

/**************************************** BITWISE OPERATIONS ****************************************/

// -------------------- bitwise_and --------------------

RppStatus bitwise_and_u8_u8_host_tensor(Rpp8u *srcPtr1,
                                        Rpp8u *srcPtr2,
                                        RpptDescPtr srcDescPtr,
                                        Rpp8u *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle);

// -------------------- bitwise_not --------------------

RppStatus bitwise_not_u8_u8_host_tensor(Rpp8u *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp8u *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& Handle);

// -------------------- bitwise_or --------------------

RppStatus bitwise_or_u8_u8_host_tensor(Rpp8u *srcPtr1,
                                       Rpp8u *srcPtr2,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8u *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& Handle);

// -------------------- bitwise_xor --------------------

RppStatus bitwise_xor_u8_u8_host_tensor(Rpp8u *srcPtr1,
                                        Rpp8u *srcPtr2,
                                        RpptDescPtr srcDescPtr,
                                        Rpp8u *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& Handle);

/**************************************** COLOR AUGMENTATIONS ****************************************/

// -------------------- brightness --------------------

RppStatus brightness_u8_u8_host_tensor(Rpp8u *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8u *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *alphaTensor,
                                       Rpp32f *betaTensor,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

RppStatus brightness_f32_f32_host_tensor(Rpp32f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp32f *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *alphaTensor,
                                         Rpp32f *betaTensor,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams layoutParams,
                                         rpp::Handle& handle);

RppStatus brightness_f16_f16_host_tensor(Rpp16f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp16f *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *alphaTensor,
                                         Rpp32f *betaTensor,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams layoutParams,
                                         rpp::Handle& handle);

RppStatus brightness_i8_i8_host_tensor(Rpp8s *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8s *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *alphaTensor,
                                       Rpp32f *betaTensor,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

// -------------------- blend --------------------

RppStatus blend_u8_u8_host_tensor(Rpp8u *srcPtr1,
                                  Rpp8u *srcPtr2,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8u *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32f *alphaTensor,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle);

RppStatus blend_f32_f32_host_tensor(Rpp32f *srcPtr1,
                                    Rpp32f *srcPtr2,
                                    RpptDescPtr srcDescPtr,
                                    Rpp32f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32f *alphaTensor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

RppStatus blend_f16_f16_host_tensor(Rpp16f *srcPtr1,
                                    Rpp16f *srcPtr2,
                                    RpptDescPtr srcDescPtr,
                                    Rpp16f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32f *alphaTensor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

RppStatus blend_i8_i8_host_tensor(Rpp8s *srcPtr1,
                                  Rpp8s *srcPtr2,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8s *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32f *alphaTensor,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle);

// -------------------- color_cast --------------------

RppStatus color_cast_u8_u8_host_tensor(Rpp8u *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8u *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       RpptRGB *rgbTensor,
                                       Rpp32f *alphaTensor,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

RppStatus color_cast_f32_f32_host_tensor(Rpp32f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp32f *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         RpptRGB *rgbTensor,
                                         Rpp32f *alphaTensor,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams layoutParams,
                                         rpp::Handle& handle);

RppStatus color_cast_f16_f16_host_tensor(Rpp16f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp16f *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         RpptRGB *rgbTensor,
                                         Rpp32f *alphaTensor,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams layoutParams,
                                         rpp::Handle& handle);

RppStatus color_cast_i8_i8_host_tensor(Rpp8s *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8s *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       RpptRGB *rgbTensor,
                                       Rpp32f *alphaTensor,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

// -------------------- color_temperature --------------------

RppStatus color_temperature_u8_u8_host_tensor(Rpp8u *srcPtr,
                                              RpptDescPtr srcDescPtr,
                                              Rpp8u *dstPtr,
                                              RpptDescPtr dstDescPtr,
                                              Rpp32s *adjustmentValueTensor,
                                              RpptROIPtr roiTensorPtrSrc,
                                              RpptRoiType roiType,
                                              RppLayoutParams layoutParams);

RppStatus color_temperature_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                RpptDescPtr srcDescPtr,
                                                Rpp32f *dstPtr,
                                                RpptDescPtr dstDescPtr,
                                                Rpp32s *adjustmentValueTensor,
                                                RpptROIPtr roiTensorPtrSrc,
                                                RpptRoiType roiType,
                                                RppLayoutParams layoutParams);

RppStatus color_temperature_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                RpptDescPtr srcDescPtr,
                                                Rpp16f *dstPtr,
                                                RpptDescPtr dstDescPtr,
                                                Rpp32s *adjustmentValueTensor,
                                                RpptROIPtr roiTensorPtrSrc,
                                                RpptRoiType roiType,
                                                RppLayoutParams layoutParams);

RppStatus color_temperature_i8_i8_host_tensor(Rpp8s *srcPtr,
                                              RpptDescPtr srcDescPtr,
                                              Rpp8s *dstPtr,
                                              RpptDescPtr dstDescPtr,
                                              Rpp32s *adjustmentValueTensor,
                                              RpptROIPtr roiTensorPtrSrc,
                                              RpptRoiType roiType,
                                              RppLayoutParams layoutParams);

// -------------------- color_twist --------------------

RppStatus color_twist_u8_u8_host_tensor(Rpp8u *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp8u *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        Rpp32f *brightnessTensor,
                                        Rpp32f *contrastTensor,
                                        Rpp32f *hueTensor,
                                        Rpp32f *saturationTensor,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle);

RppStatus color_twist_f32_f32_host_tensor(Rpp32f *srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          Rpp32f *dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          Rpp32f *brightnessTensor,
                                          Rpp32f *contrastTensor,
                                          Rpp32f *hueTensor,
                                          Rpp32f *saturationTensor,
                                          RpptROIPtr roiTensorPtrSrc,
                                          RpptRoiType roiType,
                                          RppLayoutParams layoutParams,
                                          rpp::Handle& handle);

RppStatus color_twist_f16_f16_host_tensor(Rpp16f *srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          Rpp16f *dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          Rpp32f *brightnessTensor,
                                          Rpp32f *contrastTensor,
                                          Rpp32f *hueTensor,
                                          Rpp32f *saturationTensor,
                                          RpptROIPtr roiTensorPtrSrc,
                                          RpptRoiType roiType,
                                          RppLayoutParams layoutParams,
                                          rpp::Handle& handle);

RppStatus color_twist_i8_i8_host_tensor(Rpp8s *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp8s *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        Rpp32f *brightnessTensor,
                                        Rpp32f *contrastTensor,
                                        Rpp32f *hueTensor,
                                        Rpp32f *saturationTensor,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle);

// -------------------- color_jitter --------------------

RppStatus color_jitter_u8_u8_host_tensor(Rpp8u *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp8u *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *brightnessTensor,
                                         Rpp32f *contrastTensor,
                                         Rpp32f *hueTensor,
                                         Rpp32f *saturationTensor,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams layoutParams,
                                         rpp::Handle& handle);

RppStatus color_jitter_f32_f32_host_tensor(Rpp32f *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp32f *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *brightnessTensor,
                                           Rpp32f *contrastTensor,
                                           Rpp32f *hueTensor,
                                           Rpp32f *saturationTensor,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams layoutParams,
                                           rpp::Handle& handle);

RppStatus color_jitter_f16_f16_host_tensor(Rpp16f *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp16f *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *brightnessTensor,
                                           Rpp32f *contrastTensor,
                                           Rpp32f *hueTensor,
                                           Rpp32f *saturationTensor,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams layoutParams,
                                           rpp::Handle& handle);

RppStatus color_jitter_i8_i8_host_tensor(Rpp8s *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp8s *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *brightnessTensor,
                                         Rpp32f *contrastTensor,
                                         Rpp32f *hueTensor,
                                         Rpp32f *saturationTensor,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams layoutParams,
                                         rpp::Handle& handle);

// -------------------- contrast --------------------

RppStatus contrast_u8_u8_host_tensor(Rpp8u *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp8u *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32f *contrastFactorTensor,
                                     Rpp32f *contrastCenterTensor,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);

RppStatus contrast_f32_f32_host_tensor(Rpp32f *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp32f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *contrastFactorTensor,
                                       Rpp32f *contrastCenterTensor,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

RppStatus contrast_f16_f16_host_tensor(Rpp16f *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp16f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *contrastFactorTensor,
                                       Rpp32f *contrastCenterTensor,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

RppStatus contrast_i8_i8_host_tensor(Rpp8s *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp8s *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32f *contrastFactorTensor,
                                     Rpp32f *contrastCenterTensor,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);

// -------------------- exposure --------------------

RppStatus exposure_u8_u8_host_tensor(Rpp8u *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp8u *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32f *exposureFactorTensor,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);

RppStatus exposure_f32_f32_host_tensor(Rpp32f *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp32f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *exposureFactorTensor,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

RppStatus exposure_f16_f16_host_tensor(Rpp16f *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp16f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *exposureFactorTensor,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

RppStatus exposure_i8_i8_host_tensor(Rpp8s *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp8s *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32f *exposureFactorTensor,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);

// -------------------- gamma_correction --------------------

RppStatus gamma_correction_u8_u8_host_tensor(Rpp8u *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp8u *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *gammaTensor,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams,
                                             rpp::Handle& handle);

RppStatus gamma_correction_f32_f32_host_tensor(Rpp32f *srcPtr,
                                               RpptDescPtr srcDescPtr,
                                               Rpp32f *dstPtr,
                                               RpptDescPtr dstDescPtr,
                                               Rpp32f *gammaTensor,
                                               RpptROIPtr roiTensorPtrSrc,
                                               RpptRoiType roiType,
                                               RppLayoutParams layoutParams,
                                               rpp::Handle& handle);

RppStatus gamma_correction_f16_f16_host_tensor(Rpp16f *srcPtr,
                                               RpptDescPtr srcDescPtr,
                                               Rpp16f *dstPtr,
                                               RpptDescPtr dstDescPtr,
                                               Rpp32f *gammaTensor,
                                               RpptROIPtr roiTensorPtrSrc,
                                               RpptRoiType roiType,
                                               RppLayoutParams layoutParams,
                                               rpp::Handle& handle);

RppStatus gamma_correction_i8_i8_host_tensor(Rpp8s *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp8s *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *gammaTensor,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams,
                                             rpp::Handle& handle);

// -------------------- lut --------------------

RppStatus lut_u8_u8_host_tensor(Rpp8u *srcPtr,
                                RpptDescPtr srcDescPtr,
                                Rpp8u *dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp8u *lutPtr,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                RppLayoutParams layoutParams);

RppStatus lut_u8_f16_host_tensor(Rpp8u *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp16f *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp16f *lutPtr,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams);

RppStatus lut_u8_f32_host_tensor(Rpp8u *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp32f *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp32f *lutPtr,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams);

RppStatus lut_i8_i8_host_tensor(Rpp8s *srcPtr,
                                RpptDescPtr srcDescPtr,
                                Rpp8s *dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp8s *lutPtr,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                RppLayoutParams layoutParams);


/**************************************** DATA EXCHANGE OPERATIONS ****************************************/

// -------------------- copy --------------------

RppStatus copy_u8_u8_host_tensor(Rpp8u *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8u *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 RppLayoutParams layoutParams,
                                 rpp::Handle& handle);

RppStatus copy_f32_f32_host_tensor(Rpp32f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp32f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

RppStatus copy_f16_f16_host_tensor(Rpp16f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp16f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

RppStatus copy_i8_i8_host_tensor(Rpp8s *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8s *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 RppLayoutParams layoutParams,
                                 rpp::Handle& handle);

// -------------------- swap_channels --------------------

RppStatus swap_channels_u8_u8_host_tensor(Rpp8u *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp8u *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

RppStatus swap_channels_f32_f32_host_tensor(Rpp32f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RppLayoutParams layoutParams,
                                      rpp::Handle& handle);

RppStatus swap_channels_f16_f16_host_tensor(Rpp16f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp16f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RppLayoutParams layoutParams,
                                      rpp::Handle& handle);

RppStatus swap_channels_i8_i8_host_tensor(Rpp8s *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp8s *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

// -------------------- color_to_greyscale --------------------

RppStatus color_to_greyscale_u8_u8_host_tensor(Rpp8u *srcPtr,
                                               RpptDescPtr srcDescPtr,
                                               Rpp8u *dstPtr,
                                               RpptDescPtr dstDescPtr,
                                               Rpp32f *channelWeights,
                                               RppLayoutParams layoutParams,
                                               rpp::Handle& handle);

RppStatus color_to_greyscale_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                 RpptDescPtr srcDescPtr,
                                                 Rpp32f *dstPtr,
                                                 RpptDescPtr dstDescPtr,
                                                 Rpp32f *channelWeights,
                                                 RppLayoutParams layoutParams,
                                                 rpp::Handle& handle);

RppStatus color_to_greyscale_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                 RpptDescPtr srcDescPtr,
                                                 Rpp16f *dstPtr,
                                                 RpptDescPtr dstDescPtr,
                                                 Rpp32f *channelWeights,
                                                 RppLayoutParams layoutParams,
                                                 rpp::Handle& handle);

RppStatus color_to_greyscale_i8_i8_host_tensor(Rpp8s *srcPtr,
                                               RpptDescPtr srcDescPtr,
                                               Rpp8s *dstPtr,
                                               RpptDescPtr dstDescPtr,
                                               Rpp32f *channelWeights,
                                               RppLayoutParams layoutParams,
                                               rpp::Handle& handle);

/**************************************** EFFECTS AUGMENTATIONS ****************************************/

// -------------------- gridmask --------------------

RppStatus gridmask_u8_u8_host_tensor(Rpp8u *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp8u *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32u tileWidth,
                                     Rpp32f gridRatio,
                                     Rpp32f gridAngle,
                                     RpptUintVector2D translateVector,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);

RppStatus gridmask_f32_f32_host_tensor(Rpp32f *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp32f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32u tileWidth,
                                       Rpp32f gridRatio,
                                       Rpp32f gridAngle,
                                       RpptUintVector2D translateVector,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

RppStatus gridmask_f16_f16_host_tensor(Rpp16f *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp16f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32u tileWidth,
                                       Rpp32f gridRatio,
                                       Rpp32f gridAngle,
                                       RpptUintVector2D translateVector,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

RppStatus gridmask_i8_i8_host_tensor(Rpp8s *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp8s *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32u tileWidth,
                                     Rpp32f gridRatio,
                                     Rpp32f gridAngle,
                                     RpptUintVector2D translateVector,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);

// -------------------- spatter --------------------

RppStatus spatter_u8_u8_host_tensor(Rpp8u *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp8u *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    RpptRGB spatterColor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

RppStatus spatter_f32_f32_host_tensor(Rpp32f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptRGB spatterColor,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams layoutParams,
                                      rpp::Handle& handle);

RppStatus spatter_f16_f16_host_tensor(Rpp16f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp16f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptRGB spatterColor,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams layoutParams,
                                      rpp::Handle& handle);

RppStatus spatter_i8_i8_host_tensor(Rpp8s *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp8s *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    RpptRGB spatterColor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

// -------------------- noise_salt_and_pepper --------------------

RppStatus salt_and_pepper_noise_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp8u *dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  Rpp32f *noiseProbabilityTensor,
                                                  Rpp32f *saltProbabilityTensor,
                                                  Rpp32f *saltValueTensor,
                                                  Rpp32f *pepperValueTensor,
                                                  RpptXorwowState *xorwowInitialStatePtr,
                                                  RpptROIPtr roiTensorPtrSrc,
                                                  RpptRoiType roiType,
                                                  RppLayoutParams layoutParams,
                                                  rpp::Handle& handle);

RppStatus salt_and_pepper_noise_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                    RpptDescPtr srcDescPtr,
                                                    Rpp32f *dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    Rpp32f *noiseProbabilityTensor,
                                                    Rpp32f *saltProbabilityTensor,
                                                    Rpp32f *saltValueTensor,
                                                    Rpp32f *pepperValueTensor,
                                                    RpptXorwowState *xorwowInitialStatePtr,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptRoiType roiType,
                                                    RppLayoutParams layoutParams,
                                                    rpp::Handle& handle);

RppStatus salt_and_pepper_noise_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                    RpptDescPtr srcDescPtr,
                                                    Rpp16f *dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    Rpp32f *noiseProbabilityTensor,
                                                    Rpp32f *saltProbabilityTensor,
                                                    Rpp32f *saltValueTensor,
                                                    Rpp32f *pepperValueTensor,
                                                    RpptXorwowState *xorwowInitialStatePtr,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptRoiType roiType,
                                                    RppLayoutParams layoutParams,
                                                    rpp::Handle& handle);

RppStatus salt_and_pepper_noise_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp8s *dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  Rpp32f *noiseProbabilityTensor,
                                                  Rpp32f *saltProbabilityTensor,
                                                  Rpp32f *saltValueTensor,
                                                  Rpp32f *pepperValueTensor,
                                                  RpptXorwowState *xorwowInitialStatePtr,
                                                  RpptROIPtr roiTensorPtrSrc,
                                                  RpptRoiType roiType,
                                                  RppLayoutParams layoutParams,
                                                  rpp::Handle& handle);

// -------------------- noise_shot --------------------

RppStatus shot_noise_u8_u8_host_tensor(Rpp8u *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8u *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *shotNoiseFactorTensor,
                                       RpptXorwowState *xorwowInitialStatePtr,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

RppStatus shot_noise_f32_f32_host_tensor(Rpp32f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp32f *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *shotNoiseFactorTensor,
                                         RpptXorwowState *xorwowInitialStatePtr,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams layoutParams,
                                         rpp::Handle& handle);

RppStatus shot_noise_f16_f16_host_tensor(Rpp16f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp16f *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *shotNoiseFactorTensor,
                                         RpptXorwowState *xorwowInitialStatePtr,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams layoutParams,
                                         rpp::Handle& handle);

RppStatus shot_noise_i8_i8_host_tensor(Rpp8s *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8s *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *shotNoiseFactorTensor,
                                       RpptXorwowState *xorwowInitialStatePtr,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

// -------------------- noise_gaussian --------------------

RppStatus gaussian_noise_u8_u8_host_tensor(Rpp8u *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8u *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *meanTensor,
                                           Rpp32f *stdDevTensor,
                                           RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams layoutParams,
                                           rpp::Handle& handle);

RppStatus gaussian_noise_f32_f32_host_tensor(Rpp32f *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp32f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *meanTensor,
                                             Rpp32f *stdDevTensor,
                                             RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams,
                                             rpp::Handle& handle);

RppStatus gaussian_noise_f16_f16_host_tensor(Rpp16f *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp16f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *meanTensor,
                                             Rpp32f *stdDevTensor,
                                             RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams,
                                             rpp::Handle& handle);

RppStatus gaussian_noise_i8_i8_host_tensor(Rpp8s *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8s *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *meanTensor,
                                           Rpp32f *stdDevTensor,
                                           RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams layoutParams,
                                           rpp::Handle& handle);

RppStatus gaussian_noise_voxel_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                 RpptGenericDescPtr srcGenericDescPtr,
                                                 Rpp8u *dstPtr,
                                                 RpptGenericDescPtr dstGenericDescPtr,
                                                 Rpp32f *meanTensor,
                                                 Rpp32f *stdDevTensor,
                                                 RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                                 RpptROI3DPtr roiGenericPtrSrc,
                                                 RpptRoi3DType roiType,
                                                 RppLayoutParams layoutParams,
                                                 rpp::Handle& handle);

RppStatus gaussian_noise_voxel_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                   RpptGenericDescPtr srcGenericDescPtr,
                                                   Rpp32f *dstPtr,
                                                   RpptGenericDescPtr dstGenericDescPtr,
                                                   Rpp32f *meanTensor,
                                                   Rpp32f *stdDevTensor,
                                                   RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                                   RpptROI3DPtr roiGenericPtrSrc,
                                                   RpptRoi3DType roiType,
                                                   RppLayoutParams layoutParams,
                                                   rpp::Handle& handle);

// -------------------- non_linear_blend --------------------

RppStatus non_linear_blend_u8_u8_host_tensor(Rpp8u *srcPtr1,
                                             Rpp8u *srcPtr2,
                                             RpptDescPtr srcDescPtr,
                                             Rpp8u *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *stdDevTensor,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams,
                                             rpp::Handle &handle);

RppStatus non_linear_blend_f32_f32_host_tensor(Rpp32f *srcPtr1,
                                               Rpp32f *srcPtr2,
                                               RpptDescPtr srcDescPtr,
                                               Rpp32f *dstPtr,
                                               RpptDescPtr dstDescPtr,
                                               Rpp32f *stdDevTensor,
                                               RpptROIPtr roiTensorPtrSrc,
                                               RpptRoiType roiType,
                                               RppLayoutParams layoutParams,
                                               rpp::Handle &handle);

RppStatus non_linear_blend_i8_i8_host_tensor(Rpp8s *srcPtr1,
                                             Rpp8s *srcPtr2,
                                             RpptDescPtr srcDescPtr,
                                             Rpp8s *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *stdDevTensor,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams,
                                             rpp::Handle &handle);

RppStatus non_linear_blend_f16_f16_host_tensor(Rpp16f *srcPtr1,
                                               Rpp16f *srcPtr2,
                                               RpptDescPtr srcDescPtr,
                                               Rpp16f *dstPtr,
                                               RpptDescPtr dstDescPtr,
                                               Rpp32f *stdDevTensor,
                                               RpptROIPtr roiTensorPtrSrc,
                                               RpptRoiType roiType,
                                               RppLayoutParams layoutParams,
                                               rpp::Handle &handle);

// -------------------- jitter --------------------

RppStatus jitter_u8_u8_host_tensor(Rpp8u *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp8u *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32u *kernelSizeTensor,
                                   RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

RppStatus jitter_f32_f32_host_tensor(Rpp32f *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp32f *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32u *kernelSizeTensor,
                                     RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);

RppStatus jitter_f16_f16_host_tensor(Rpp16f *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp16f *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32u *kernelSizeTensor,
                                     RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);

RppStatus jitter_i8_i8_host_tensor(Rpp8s *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp8s *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32u *kernelSizeTensor,
                                   RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

// -------------------- glitch --------------------

RppStatus glitch_u8_u8_host_tensor(Rpp8u *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp8u *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RpptChannelOffsets *rgbOffsets,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

RppStatus glitch_f32_f32_host_tensor(Rpp32f *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp32f *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptChannelOffsets *rgbOffsets,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);

RppStatus glitch_f16_f16_host_tensor(Rpp16f *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp16f *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptChannelOffsets *rgbOffsets,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);

RppStatus glitch_i8_i8_host_tensor(Rpp8s *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp8s *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RpptChannelOffsets *rgbOffsets,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

// -------------------- water --------------------

RppStatus water_u8_u8_host_tensor(Rpp8u *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8u *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32f *amplitudeXTensor,
                                  Rpp32f *amplitudeYTensor,
                                  Rpp32f *frequencyXTensor,
                                  Rpp32f *frequencyYTensor,
                                  Rpp32f *phaseXTensor,
                                  Rpp32f *phaseYTensor,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle);

RppStatus water_f32_f32_host_tensor(Rpp32f *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp32f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32f *amplitudeXTensor,
                                    Rpp32f *amplitudeYTensor,
                                    Rpp32f *frequencyXTensor,
                                    Rpp32f *frequencyYTensor,
                                    Rpp32f *phaseXTensor,
                                    Rpp32f *phaseYTensor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

RppStatus water_f16_f16_host_tensor(Rpp16f *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp16f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32f *amplitudeXTensor,
                                    Rpp32f *amplitudeYTensor,
                                    Rpp32f *frequencyXTensor,
                                    Rpp32f *frequencyYTensor,
                                    Rpp32f *phaseXTensor,
                                    Rpp32f *phaseYTensor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

RppStatus water_i8_i8_host_tensor(Rpp8s *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8s *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32f *amplitudeXTensor,
                                  Rpp32f *amplitudeYTensor,
                                  Rpp32f *frequencyXTensor,
                                  Rpp32f *frequencyYTensor,
                                  Rpp32f *phaseXTensor,
                                  Rpp32f *phaseYTensor,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle);

// -------------------- ricap --------------------

RppStatus ricap_u8_u8_host_tensor(Rpp8u *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8u *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32u *permutedIndices,
                                  RpptROIPtr roiPtrInputCropRegion,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle);

RppStatus ricap_f32_f32_host_tensor(Rpp32f *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp32f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32u *permutedIndices,
                                    RpptROIPtr roiPtrInputCropRegion,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

RppStatus ricap_f16_f16_host_tensor(Rpp16f *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp16f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32u *permutedIndices,
                                    RpptROIPtr roiPtrInputCropRegion,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

RppStatus ricap_i8_i8_host_tensor(Rpp8s *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8s *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32u *permutedIndices,
                                  RpptROIPtr roiPtrInputCropRegion,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle);

// -------------------- vignette --------------------

RppStatus vignette_u8_u8_host_tensor(Rpp8u *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp8u *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32f *vignetteIntensityTensor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

RppStatus vignette_f32_f32_host_tensor(Rpp32f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32f *vignetteIntensityTensor,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams layoutParams,
                                      rpp::Handle& handle);

RppStatus vignette_i8_i8_host_tensor(Rpp8s *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp8s *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32f *vignetteIntensityTensor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

RppStatus vignette_f16_f16_host_tensor(Rpp16f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp16f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32f *vignetteIntensityTensor,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams layoutParams,
                                      rpp::Handle& handle);

// -------------------- erase --------------------

template <typename T>
RppStatus erase_host_tensor(T *srcPtr,
                            RpptDescPtr srcDescPtr,
                            T *dstPtr,
                            RpptDescPtr dstDescPtr,
                            RpptRoiLtrb *anchorBoxInfoTensor,
                            T *colorsTensor,
                            Rpp32u *numBoxesTensor,
                            RpptROIPtr roiTensorPtrSrc,
                            RpptRoiType roiType,
                            RppLayoutParams layoutParams,
                            rpp::Handle& handle);

// -------------------- fog --------------------

RppStatus fog_u8_u8_host_tensor(Rpp8u *srcPtr,
                                RpptDescPtr srcDescPtr,
                                Rpp8u *dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32f *fogAlphaMask,
                                Rpp32f *fogIntensityMask,
                                Rpp32f *intensityFactor,
                                Rpp32f *grayFactor,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                RppLayoutParams layoutParams,
                                rpp::Handle& handle);

RppStatus fog_f16_f16_host_tensor(Rpp16f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp16f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32f* fogAlphaMask,
                                  Rpp32f* fogIntensityMask,
                                  Rpp32f *intensityFactor,
                                  Rpp32f *grayFactor,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle);

RppStatus fog_f32_f32_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32f* fogAlphaMask,
                                  Rpp32f* fogIntensityMask,
                                  Rpp32f *intensityFactor,
                                  Rpp32f *grayFactor,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle);

RppStatus fog_i8_i8_host_tensor(Rpp8s *srcPtr,
                                RpptDescPtr srcDescPtr,
                                Rpp8s *dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32f* fogAlphaMask,
                                Rpp32f* fogIntensityMask,
                                Rpp32f *intensityFactor,
                                Rpp32f *grayFactor,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                RppLayoutParams layoutParams,
                                rpp::Handle& handle);

// -------------------- rain --------------------

RppStatus rain_u8_u8_host_tensor(Rpp8u *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8u *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp32f rainPercentage,
                                 Rpp32u rainWidth,
                                 Rpp32u rainHeight,
                                 Rpp32f slantAngle,
                                 Rpp32f *alphaValues,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams,
                                 rpp::Handle& handle);

RppStatus rain_f32_f32_host_tensor(Rpp32f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp32f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32f rainPercentage,
                                   Rpp32u rainWidth,
                                   Rpp32u rainHeight,
                                   Rpp32f slantAngle,
                                   Rpp32f *alphaValues,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

RppStatus rain_f16_f16_host_tensor(Rpp16f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp16f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32f rainPercentage,
                                   Rpp32u rainWidth,
                                   Rpp32u rainHeight,
                                   Rpp32f slantAngle,
                                   Rpp32f *alphaValues,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

RppStatus rain_i8_i8_host_tensor(Rpp8s *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8s *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp32f rainPercentage,
                                 Rpp32u rainWidth,
                                 Rpp32u rainHeight,
                                 Rpp32f slantAngle,
                                 Rpp32f *alphaValues,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams,
                                 rpp::Handle& handle);

/**************************************** FILTER AUGMENTATIONS ****************************************/

// -------------------- gaussian_filter --------------------

template<typename T>
RppStatus gaussian_filter_host_tensor(T *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      T *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32f *stdDevTensor,
                                      Rpp32u kernelSize,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams layoutParams,
                                      rpp::Handle& handle);

template<typename T>
RppStatus gaussian_filter_generic_host_tensor(T *srcPtr,
                                              RpptDescPtr srcDescPtr,
                                              T *dstPtr,
                                              RpptDescPtr dstDescPtr,
                                              Rpp32f *stdDevTensor,
                                              Rpp32u kernelSize,
                                              RpptROIPtr roiTensorPtrSrc,
                                              RpptRoiType roiType,
                                              RppLayoutParams layoutParams,
                                              rpp::Handle& handle);

// -------------------- box_filter --------------------

template<typename T>
RppStatus box_filter_char_host_tensor(T *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      T *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32u kernelSize,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams layoutParams,
                                      rpp::Handle& handle);

// F32 and F16 bitdepth
template<typename T>
RppStatus box_filter_float_host_tensor(T *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       T *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32u kernelSize,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

/**************************************** GEOMETRIC AUGMENTATIONS ****************************************/

// -------------------- crop --------------------

RppStatus crop_u8_u8_host_tensor(Rpp8u *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8u *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams,
                                 rpp::Handle& handle);

RppStatus crop_f32_f32_host_tensor(Rpp32f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp32f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

RppStatus crop_f16_f16_host_tensor(Rpp16f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp16f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

RppStatus crop_i8_i8_host_tensor(Rpp8s *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8s *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams,
                                 rpp::Handle& handle);

// -------------------- crop_and_patch --------------------

RppStatus crop_and_patch_u8_u8_host_tensor(Rpp8u *srcPtr1,
                                           Rpp8u *srcPtr2,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8u *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           RpptROIPtr roiTensorPtrDst,
                                           RpptROIPtr cropRoiTensor,
                                           RpptROIPtr patchRoiTensor,
                                           RpptRoiType roiType,
                                           RppLayoutParams layoutParams,
                                           rpp::Handle& handle);

RppStatus crop_and_patch_f32_f32_host_tensor(Rpp32f *srcPtr1,
                                             Rpp32f *srcPtr2,
                                             RpptDescPtr srcDescPtr,
                                             Rpp32f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             RpptROIPtr roiTensorPtrDst,
                                             RpptROIPtr cropRoiTensor,
                                             RpptROIPtr patchRoiTensor,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams,
                                             rpp::Handle& handle);

RppStatus crop_and_patch_f16_f16_host_tensor(Rpp16f *srcPtr1,
                                             Rpp16f *srcPtr2,
                                             RpptDescPtr srcDescPtr,
                                             Rpp16f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             RpptROIPtr roiTensorPtrDst,
                                             RpptROIPtr cropRoiTensor,
                                             RpptROIPtr patchRoiTensor,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams,
                                             rpp::Handle& handle);

RppStatus crop_and_patch_i8_i8_host_tensor(Rpp8s *srcPtr1,
                                           Rpp8s *srcPtr2,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8s *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           RpptROIPtr roiTensorPtrDst,
                                           RpptROIPtr cropRoiTensor,
                                           RpptROIPtr patchRoiTensor,
                                           RpptRoiType roiType,
                                           RppLayoutParams layoutParams,
                                           rpp::Handle& handle);

// -------------------- crop_mirror_normalize --------------------

RppStatus crop_mirror_normalize_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp8u *dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  Rpp32f *offsetTensor,
                                                  Rpp32f *multiplierTensor,
                                                  Rpp32u *mirrorTensor,
                                                  RpptROIPtr roiTensorPtrSrc,
                                                  RpptRoiType roiType,
                                                  RppLayoutParams layoutParams,
                                                  rpp::Handle& handle);

RppStatus crop_mirror_normalize_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                    RpptDescPtr srcDescPtr,
                                                    Rpp32f *dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    Rpp32f *offsetTensor,
                                                    Rpp32f *multiplierTensor,
                                                    Rpp32u *mirrorTensor,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptRoiType roiType,
                                                    RppLayoutParams layoutParams,
                                                    rpp::Handle& handle);

RppStatus crop_mirror_normalize_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                    RpptDescPtr srcDescPtr,
                                                    Rpp16f *dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    Rpp32f *offsetTensor,
                                                    Rpp32f *multiplierTensor,
                                                    Rpp32u *mirrorTensor,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptRoiType roiType,
                                                    RppLayoutParams layoutParams,
                                                    rpp::Handle& handle);

RppStatus crop_mirror_normalize_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp8s *dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  Rpp32f *offsetTensor,
                                                  Rpp32f *multiplierTensor,
                                                  Rpp32u *mirrorTensor,
                                                  RpptROIPtr roiTensorPtrSrc,
                                                  RpptRoiType roiType,
                                                  RppLayoutParams layoutParams,
                                                  rpp::Handle& handle);

RppStatus crop_mirror_normalize_u8_f32_host_tensor(Rpp8u *srcPtr,
                                                   RpptDescPtr srcDescPtr,
                                                   Rpp32f *dstPtr,
                                                   RpptDescPtr dstDescPtr,
                                                   Rpp32f *offsetTensor,
                                                   Rpp32f *multiplierTensor,
                                                   Rpp32u *mirrorTensor,
                                                   RpptROIPtr roiTensorPtrSrc,
                                                   RpptRoiType roiType,
                                                   RppLayoutParams layoutParams,
                                                   rpp::Handle& handle);

RppStatus crop_mirror_normalize_u8_f16_host_tensor(Rpp8u *srcPtr,
                                                   RpptDescPtr srcDescPtr,
                                                   Rpp16f *dstPtr,
                                                   RpptDescPtr dstDescPtr,
                                                   Rpp32f *offsetTensor,
                                                   Rpp32f *multiplierTensor,
                                                   Rpp32u *mirrorTensor,
                                                   RpptROIPtr roiTensorPtrSrc,
                                                   RpptRoiType roiType,
                                                   RppLayoutParams layoutParams,
                                                   rpp::Handle& handle);

// -------------------- flip --------------------

RppStatus flip_u8_u8_host_tensor(Rpp8u *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8u *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp32u *horizontalTensor,
                                 Rpp32u *verticalTensor,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams,
                                 rpp::Handle& handle);

RppStatus flip_f32_f32_host_tensor(Rpp32f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp32f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32u *horizontalTensor,
                                   Rpp32u *verticalTensor,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

RppStatus flip_f16_f16_host_tensor(Rpp16f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp16f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32u *horizontalTensor,
                                   Rpp32u *verticalTensor,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

RppStatus flip_i8_i8_host_tensor(Rpp8s *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8s *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp32u *horizontalTensor,
                                 Rpp32u *verticalTensor,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams,
                                 rpp::Handle& handle);

RppStatus flip_voxel_f32_f32_host_tensor(Rpp32f *srcPtr,
                                         RpptGenericDescPtr srcGenericDescPtr,
                                         Rpp32f *dstPtr,
                                         RpptGenericDescPtr dstGenericDescPtr,
                                         Rpp32u *horizontalTensor,
                                         Rpp32u *verticalTensor,
                                         Rpp32u *depthTensor,
                                         RpptROI3DPtr roiGenericPtrSrc,
                                         RpptRoi3DType roiType,
                                         RppLayoutParams layoutParams,
                                         rpp::Handle& handle);

RppStatus flip_voxel_u8_u8_host_tensor(Rpp8u *srcPtr,
                                       RpptGenericDescPtr srcGenericDescPtr,
                                       Rpp8u *dstPtr,
                                       RpptGenericDescPtr dstGenericDescPtr,
                                       Rpp32u *horizontalTensor,
                                       Rpp32u *verticalTensor,
                                       Rpp32u *depthTensor,
                                       RpptROI3DPtr roiGenericPtrSrc,
                                       RpptRoi3DType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

// -------------------- remap --------------------

/************* NEAREST NEIGHBOR INTERPOLATION *************/

RppStatus remap_nn_u8_u8_host_tensor(Rpp8u *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp8u *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32f *rowRemapTable,
                                     Rpp32f *colRemapTable,
                                     RpptDescPtr remapTableDescPtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);

RppStatus remap_nn_f32_f32_host_tensor(Rpp32f *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp32f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *rowRemapTable,
                                       Rpp32f *colRemapTable,
                                       RpptDescPtr remapTableDescPtr,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

RppStatus remap_nn_i8_i8_host_tensor(Rpp8s *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp8s *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32f *rowRemapTable,
                                     Rpp32f *colRemapTable,
                                     RpptDescPtr remapTableDescPtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);

RppStatus remap_nn_f16_f16_host_tensor(Rpp16f *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp16f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *rowRemapTable,
                                       Rpp32f *colRemapTable,
                                       RpptDescPtr remapTableDescPtr,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

/************* BILINEAR INTERPOLATION *************/

RppStatus remap_bilinear_u8_u8_host_tensor(Rpp8u *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8u *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *rowRemapTable,
                                           Rpp32f *colRemapTable,
                                           RpptDescPtr remapTableDescPtr,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams layoutParams,
                                           rpp::Handle& handle);

RppStatus remap_bilinear_f32_f32_host_tensor(Rpp32f *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp32f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *rowRemapTable,
                                             Rpp32f *colRemapTable,
                                             RpptDescPtr remapTableDescPtr,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams,
                                             rpp::Handle& handle);

RppStatus remap_bilinear_i8_i8_host_tensor(Rpp8s *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8s *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *rowRemapTable,
                                           Rpp32f *colRemapTable,
                                           RpptDescPtr remapTableDescPtr,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams ,
                                           rpp::Handle& handle);

RppStatus remap_bilinear_f16_f16_host_tensor(Rpp16f *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp16f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *rowRemapTable,
                                             Rpp32f *colRemapTable,
                                             RpptDescPtr remapTableDescPtr,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams,
                                             rpp::Handle& handle);

// -------------------- resize --------------------

/************* NEAREST NEIGHBOR INTERPOLATION *************/

RppStatus resize_nn_u8_u8_host_tensor(Rpp8u *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8u *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptImagePatchPtr dstImgSize,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams srcLayoutParams,
                                      rpp::Handle& handle);

RppStatus resize_nn_f32_f32_host_tensor(Rpp32f *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp32f *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        RpptImagePatchPtr dstImgSize,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams srcLayoutParams,
                                        rpp::Handle& handle);

RppStatus resize_nn_i8_i8_host_tensor(Rpp8s *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8s *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptImagePatchPtr dstImgSize,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams srcLayoutParams,
                                      rpp::Handle& handle);

RppStatus resize_nn_f16_f16_host_tensor(Rpp16f *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp16f *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        RpptImagePatchPtr dstImgSize,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams srcLayoutParams,
                                        rpp::Handle& handle);

/************* BILINEAR INTERPOLATION *************/

RppStatus resize_bilinear_u8_u8_host_tensor(Rpp8u *srcPtr,
                                            RpptDescPtr srcDescPtr,
                                            Rpp8u *dstPtr,
                                            RpptDescPtr dstDescPtr,
                                            RpptImagePatchPtr dstImgSize,
                                            RpptROIPtr roiTensorPtrSrc,
                                            RpptRoiType roiType,
                                            RppLayoutParams srcLayoutParams,
                                            rpp::Handle& handle);

RppStatus resize_bilinear_f32_f32_host_tensor(Rpp32f *srcPtr,
                                              RpptDescPtr srcDescPtr,
                                              Rpp32f *dstPtr,
                                              RpptDescPtr dstDescPtr,
                                              RpptImagePatchPtr dstImgSize,
                                              RpptROIPtr roiTensorPtrSrc,
                                              RpptRoiType roiType,
                                              RppLayoutParams srcLayoutParams,
                                              rpp::Handle& handle);

RppStatus resize_bilinear_f16_f16_host_tensor(Rpp16f *srcPtr,
                                              RpptDescPtr srcDescPtr,
                                              Rpp16f *dstPtr,
                                              RpptDescPtr dstDescPtr,
                                              RpptImagePatchPtr dstImgSize,
                                              RpptROIPtr roiTensorPtrSrc,
                                              RpptRoiType roiType,
                                              RppLayoutParams srcLayoutParams,
                                              rpp::Handle& handle);

RppStatus resize_bilinear_i8_i8_host_tensor(Rpp8s *srcPtr,
                                            RpptDescPtr srcDescPtr,
                                            Rpp8s *dstPtr,
                                            RpptDescPtr dstDescPtr,
                                            RpptImagePatchPtr dstImgSize,
                                            RpptROIPtr roiTensorPtrSrc,
                                            RpptRoiType roiType,
                                            RppLayoutParams srcLayoutParams,
                                            rpp::Handle& handle);

template <typename T>
RppStatus resize_separable_host_tensor(T *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       T *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f * tempPtr,
                                       RpptDescPtr tempDescPtr,
                                       RpptImagePatchPtr dstImgSize,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams srcLayoutParams,
                                       RpptInterpolationType interpolationType,
                                       rpp::Handle& handle);

// -------------------- resize_crop_mirror --------------------

RppStatus resize_crop_mirror_u8_u8_host_tensor(Rpp8u *srcPtr,
                                               RpptDescPtr srcDescPtr,
                                               Rpp8u *dstPtr,
                                               RpptDescPtr dstDescPtr,
                                               RpptImagePatchPtr dstImgSize,
                                               Rpp32u *mirrorTensor,
                                               RpptROIPtr roiTensorPtrSrc,
                                               RpptRoiType roiType,
                                               RppLayoutParams layoutParams,
                                               rpp::Handle& handle);

RppStatus resize_crop_mirror_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                 RpptDescPtr srcDescPtr,
                                                 Rpp32f *dstPtr,
                                                 RpptDescPtr dstDescPtr,
                                                 RpptImagePatchPtr dstImgSize,
                                                 Rpp32u *mirrorTensor,
                                                 RpptROIPtr roiTensorPtrSrc,
                                                 RpptRoiType roiType,
                                                 RppLayoutParams layoutParams,
                                                 rpp::Handle& handle);

RppStatus resize_crop_mirror_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                 RpptDescPtr srcDescPtr,
                                                 Rpp16f *dstPtr,
                                                 RpptDescPtr dstDescPtr,
                                                 RpptImagePatchPtr dstImgSize,
                                                 Rpp32u *mirrorTensor,
                                                 RpptROIPtr roiTensorPtrSrc,
                                                 RpptRoiType roiType,
                                                 RppLayoutParams layoutParams,
                                                 rpp::Handle& handle);

RppStatus resize_crop_mirror_i8_i8_host_tensor(Rpp8s *srcPtr,
                                               RpptDescPtr srcDescPtr,
                                               Rpp8s *dstPtr,
                                               RpptDescPtr dstDescPtr,
                                               RpptImagePatchPtr dstImgSize,
                                               Rpp32u *mirrorTensor,
                                               RpptROIPtr roiTensorPtrSrc,
                                               RpptRoiType roiType,
                                               RppLayoutParams layoutParams,
                                               rpp::Handle& handle);

// -------------------- resize_mirror_normalize --------------------

RppStatus resize_mirror_normalize_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                    RpptDescPtr srcDescPtr,
                                                    Rpp8u *dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    RpptImagePatchPtr dstImgSize,
                                                    Rpp32f *meanTensor,
                                                    Rpp32f *stdDevTensor,
                                                    Rpp32u *mirrorTensor,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptRoiType roiType,
                                                    RppLayoutParams layoutParams,
                                                    rpp::Handle& handle);

RppStatus resize_mirror_normalize_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                      RpptDescPtr srcDescPtr,
                                                      Rpp32f *dstPtr,
                                                      RpptDescPtr dstDescPtr,
                                                      RpptImagePatchPtr dstImgSize,
                                                      Rpp32f *meanTensor,
                                                      Rpp32f *stdDevTensor,
                                                      Rpp32u *mirrorTensor,
                                                      RpptROIPtr roiTensorPtrSrc,
                                                      RpptRoiType roiType,
                                                      RppLayoutParams layoutParams,
                                                      rpp::Handle& handle);

RppStatus resize_mirror_normalize_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                      RpptDescPtr srcDescPtr,
                                                      Rpp16f *dstPtr,
                                                      RpptDescPtr dstDescPtr,
                                                      RpptImagePatchPtr dstImgSize,
                                                      Rpp32f *meanTensor,
                                                      Rpp32f *stdDevTensor,
                                                      Rpp32u *mirrorTensor,
                                                      RpptROIPtr roiTensorPtrSrc,
                                                      RpptRoiType roiType,
                                                      RppLayoutParams layoutParams,
                                                      rpp::Handle& handle);

RppStatus resize_mirror_normalize_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                    RpptDescPtr srcDescPtr,
                                                    Rpp8s *dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    RpptImagePatchPtr dstImgSize,
                                                    Rpp32f *meanTensor,
                                                    Rpp32f *stdDevTensor,
                                                    Rpp32u *mirrorTensor,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptRoiType roiType,
                                                    RppLayoutParams layoutParams,
                                                    rpp::Handle& handle);

RppStatus resize_mirror_normalize_u8_f32_host_tensor(Rpp8u *srcPtr,
                                                     RpptDescPtr srcDescPtr,
                                                     Rpp32f *dstPtr,
                                                     RpptDescPtr dstDescPtr,
                                                     RpptImagePatchPtr dstImgSize,
                                                     Rpp32f *meanTensor,
                                                     Rpp32f *stdDevTensor,
                                                     Rpp32u *mirrorTensor,
                                                     RpptROIPtr roiTensorPtrSrc,
                                                     RpptRoiType roiType,
                                                     RppLayoutParams layoutParams,
                                                     rpp::Handle& handle);

RppStatus resize_mirror_normalize_u8_f16_host_tensor(Rpp8u *srcPtr,
                                                     RpptDescPtr srcDescPtr,
                                                     Rpp16f *dstPtr,
                                                     RpptDescPtr dstDescPtr,
                                                     RpptImagePatchPtr dstImgSize,
                                                     Rpp32f *meanTensor,
                                                     Rpp32f *stdDevTensor,
                                                     Rpp32u *mirrorTensor,
                                                     RpptROIPtr roiTensorPtrSrc,
                                                     RpptRoiType roiType,
                                                     RppLayoutParams layoutParams,
                                                     rpp::Handle& handle);

// -------------------- lens_correction --------------------

void compute_lens_correction_remap_tables_host_tensor(RpptDescPtr srcDescPtr,
                                                      Rpp32f *rowRemapTable,
                                                      Rpp32f *colRemapTable,
                                                      RpptDescPtr tableDescPtr,
                                                      Rpp32f *cameraMatrixTensor,
                                                      Rpp32f *distortionCoeffsTensor,
                                                      RpptROIPtr roiTensorPtrSrc,
                                                      rpp::Handle& handle);

// -------------------- phase --------------------

RppStatus phase_u8_u8_host_tensor(Rpp8u *srcPtr1,
                                  Rpp8u *srcPtr2,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8u *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle);

RppStatus phase_f32_f32_host_tensor(Rpp32f *srcPtr1,
                                    Rpp32f *srcPtr2,
                                    RpptDescPtr srcDescPtr,
                                    Rpp32f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

RppStatus phase_f16_f16_host_tensor(Rpp16f *srcPtr1,
                                    Rpp16f *srcPtr2,
                                    RpptDescPtr srcDescPtr,
                                    Rpp16f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

RppStatus phase_i8_i8_host_tensor(Rpp8s *srcPtr1,
                                  Rpp8s *srcPtr2,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8s *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle);

// -------------------- slice --------------------

template<typename T>
RppStatus slice_host_tensor(T *srcPtr,
                            RpptGenericDescPtr srcGenericDescPtr,
                            T *dstPtr,
                            RpptGenericDescPtr dstGenericDescPtr,
                            Rpp32s *anchorTensor,
                            Rpp32s *shapeTensor,
                            T* fillValue,
                            bool enablePadding,
                            Rpp32u *roiTensor,
                            RppLayoutParams layoutParams,
                            rpp::Handle& handle);

// -------------------- transpose --------------------

RppStatus transpose_f32_f32_host_tensor(Rpp32f *srcPtr,
                                        RpptGenericDescPtr srcGenericDescPtr,
                                        Rpp32f *dstPtr,
                                        RpptGenericDescPtr dstGenericDescPtr,
                                        Rpp32u *permTensor,
                                        Rpp32u *roiTensor,
                                        rpp::Handle& handle);

template<typename T>
RppStatus transpose_generic_host_tensor(T *srcPtr,
                                        RpptGenericDescPtr srcGenericDescPtr,
                                        T *dstPtr,
                                        RpptGenericDescPtr dstGenericDescPtr,
                                        Rpp32u *permTensor,
                                        Rpp32u *roiTensor,
                                        rpp::Handle& handle);

// -------------------- warp_affine --------------------

/************* NEAREST NEIGHBOUR INTERPOLATION *************/

RppStatus warp_affine_nn_u8_u8_host_tensor(Rpp8u *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8u *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *affineTensor,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams srcLayoutParams,
                                           rpp::Handle& handle);

RppStatus warp_affine_nn_f32_f32_host_tensor(Rpp32f *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp32f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *affineTensor,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams srcLayoutParams,
                                             rpp::Handle& handle);

RppStatus warp_affine_nn_i8_i8_host_tensor(Rpp8s *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8s *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *affineTensor,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams srcLayoutParams,
                                           rpp::Handle& handle);

RppStatus warp_affine_nn_f16_f16_host_tensor(Rpp16f *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp16f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *affineTensor,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams srcLayoutParams,
                                             rpp::Handle& handle);

/************* BILINEAR INTERPOLATION *************/

RppStatus warp_affine_bilinear_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                 RpptDescPtr srcDescPtr,
                                                 Rpp8u *dstPtr,
                                                 RpptDescPtr dstDescPtr,
                                                 Rpp32f *affineTensor,
                                                 RpptROIPtr roiTensorPtrSrc,
                                                 RpptRoiType roiType,
                                                 RppLayoutParams srcLayoutParams,
                                                 rpp::Handle& handle);

RppStatus warp_affine_bilinear_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                   RpptDescPtr srcDescPtr,
                                                   Rpp32f *dstPtr,
                                                   RpptDescPtr dstDescPtr,
                                                   Rpp32f *affineTensor,
                                                   RpptROIPtr roiTensorPtrSrc,
                                                   RpptRoiType roiType,
                                                   RppLayoutParams srcLayoutParams,
                                                   rpp::Handle& handle);

RppStatus warp_affine_bilinear_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                 RpptDescPtr srcDescPtr,
                                                 Rpp8s *dstPtr,
                                                 RpptDescPtr dstDescPtr,
                                                 Rpp32f *affineTensor,
                                                 RpptROIPtr roiTensorPtrSrc,
                                                 RpptRoiType roiType,
                                                 RppLayoutParams srcLayoutParams,
                                                 rpp::Handle& handle);

RppStatus warp_affine_bilinear_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                   RpptDescPtr srcDescPtr,
                                                   Rpp16f *dstPtr,
                                                   RpptDescPtr dstDescPtr,
                                                   Rpp32f *affineTensor,
                                                   RpptROIPtr roiTensorPtrSrc,
                                                   RpptRoiType roiType,
                                                   RppLayoutParams srcLayoutParams,
                                                   rpp::Handle& handle);

// -------------------- warp_perspective --------------------

/************* NEAREST NEIGHBOR INTERPOLATION *************/

RppStatus warp_perspective_nn_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                RpptDescPtr srcDescPtr,
                                                Rpp8u *dstPtr,
                                                RpptDescPtr dstDescPtr,
                                                Rpp32f *perspectiveTensor,
                                                RpptROIPtr roiTensorPtrSrc,
                                                RpptRoiType roiType,
                                                RppLayoutParams srcLayoutParams,
                                                rpp::Handle& handle);

RppStatus warp_perspective_nn_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp32f *dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  Rpp32f *perspectiveTensor,
                                                  RpptROIPtr roiTensorPtrSrc,
                                                  RpptRoiType roiType,
                                                  RppLayoutParams srcLayoutParams,
                                                  rpp::Handle& handle);

RppStatus warp_perspective_nn_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                RpptDescPtr srcDescPtr,
                                                Rpp8s *dstPtr,
                                                RpptDescPtr dstDescPtr,
                                                Rpp32f *perspectiveTensor,
                                                RpptROIPtr roiTensorPtrSrc,
                                                RpptRoiType roiType,
                                                RppLayoutParams srcLayoutParams,
                                                rpp::Handle& handle);

RppStatus warp_perspective_nn_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp16f *dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  Rpp32f *perspectiveTensor,
                                                  RpptROIPtr roiTensorPtrSrc,
                                                  RpptRoiType roiType,
                                                  RppLayoutParams srcLayoutParams,
                                                  rpp::Handle& handle);

/************* BILINEAR INTERPOLATION *************/

RppStatus warp_perspective_bilinear_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                      RpptDescPtr srcDescPtr,
                                                      Rpp8u *dstPtr,
                                                      RpptDescPtr dstDescPtr,
                                                      Rpp32f *perspectiveTensor,
                                                      RpptROIPtr roiTensorPtrSrc,
                                                      RpptRoiType roiType,
                                                      RppLayoutParams srcLayoutParams,
                                                      rpp::Handle& handle);

RppStatus warp_perspective_bilinear_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                        RpptDescPtr srcDescPtr,
                                                        Rpp32f *dstPtr,
                                                        RpptDescPtr dstDescPtr,
                                                        Rpp32f *perspectiveTensor,
                                                        RpptROIPtr roiTensorPtrSrc,
                                                        RpptRoiType roiType,
                                                        RppLayoutParams srcLayoutParams,
                                                        rpp::Handle& handle);

RppStatus warp_perspective_bilinear_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                      RpptDescPtr srcDescPtr,
                                                      Rpp8s *dstPtr,
                                                      RpptDescPtr dstDescPtr,
                                                      Rpp32f *perspectiveTensor,
                                                      RpptROIPtr roiTensorPtrSrc,
                                                      RpptRoiType roiType,
                                                      RppLayoutParams srcLayoutParams,
                                                      rpp::Handle& handle);

RppStatus warp_perspective_bilinear_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                        RpptDescPtr srcDescPtr,
                                                        Rpp16f *dstPtr,
                                                        RpptDescPtr dstDescPtr,
                                                        Rpp32f *perspectiveTensor,
                                                        RpptROIPtr roiTensorPtrSrc,
                                                        RpptRoiType roiType,
                                                        RppLayoutParams srcLayoutParams,
                                                        rpp::Handle& handle);

/**************************************** STATISTICAL OPERATIONS ****************************************/

// -------------------- normalize --------------------

RppStatus normalize_f32_f32_host_tensor(Rpp32f *srcPtr,
                                        RpptGenericDescPtr srcGenericDescPtr,
                                        Rpp32f *dstPtr,
                                        RpptGenericDescPtr dstGenericDescPtr,
                                        Rpp32u axisMask,
                                        Rpp32f *meanTensorPtr,
                                        Rpp32f *stdDevTensorPtr,
                                        Rpp8u computeMeanStddev,
                                        Rpp32f scale,
                                        Rpp32f shift,
                                        Rpp32u *roiTensor,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle);

template<typename T1, typename T2>
RppStatus normalize_generic_host_tensor(T1 *srcPtr,
                                        RpptGenericDescPtr srcGenericDescPtr,
                                        T2 *dstPtr,
                                        RpptGenericDescPtr dstGenericDescPtr,
                                        Rpp32u axisMask,
                                        Rpp32f *meanTensorPtr,
                                        Rpp32f *stdDevTensorPtr,
                                        Rpp8u computeMeanStddev,
                                        Rpp32f scale,
                                        Rpp32f shift,
                                        Rpp32u *roiTensor,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle);


// -------------------- tensor_sum --------------------

RppStatus tensor_sum_u8_u64_host(Rpp8u *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp64u *tensorSumArr,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams);

RppStatus tensor_sum_f32_f32_host(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *tensorSumArr,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams);

RppStatus tensor_sum_f16_f32_host(Rpp16f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *tensorSumArr,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams);

RppStatus tensor_sum_i8_i64_host(Rpp8s *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp64s *tensorSumArr,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams);

// -------------------- tensor_min --------------------

RppStatus tensor_min_u8_u8_host(Rpp8u *srcPtr,
                                RpptDescPtr srcDescPtr,
                                Rpp8u *minArr,
                                Rpp32u minArrLength,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                RppLayoutParams layoutParams);

RppStatus tensor_min_f32_f32_host(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *minArr,
                                  Rpp32u minArrLength,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams);

RppStatus tensor_min_f16_f16_host(Rpp16f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp16f *minArr,
                                  Rpp32u minArrLength,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams);

RppStatus tensor_min_i8_i8_host(Rpp8s *srcPtr,
                                RpptDescPtr srcDescPtr,
                                Rpp8s *minArr,
                                Rpp32u minArrLength,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                RppLayoutParams layoutParams);

// -------------------- tensor_max --------------------

RppStatus tensor_max_u8_u8_host(Rpp8u *srcPtr,
                                RpptDescPtr srcDescPtr,
                                Rpp8u *maxArr,
                                Rpp32u maxArrLength,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                RppLayoutParams layoutParams);

RppStatus tensor_max_f32_f32_host(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *maxArr,
                                  Rpp32u maxArrLength,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams);

RppStatus tensor_max_f16_f16_host(Rpp16f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp16f *maxArr,
                                  Rpp32u maxArrLength,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams);

RppStatus tensor_max_i8_i8_host(Rpp8s *srcPtr,
                                RpptDescPtr srcDescPtr,
                                Rpp8s *maxArr,
                                Rpp32u maxArrLength,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                RppLayoutParams layoutParams);

// -------------------- tensor_mean --------------------

RppStatus tensor_mean_u8_f32_host(Rpp8u *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *tensorMeanArr,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle);

RppStatus tensor_mean_f32_f32_host(Rpp32f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp32f *tensorMeanArr,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

RppStatus tensor_mean_f16_f32_host(Rpp16f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp32f *tensorMeanArr,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

RppStatus tensor_mean_i8_f32_host(Rpp8s *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *tensorMeanArr,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle);

// -------------------- tensor_stddev --------------------

RppStatus tensor_stddev_u8_f32_host(Rpp8u *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp32f *tensorStddevArr,
                                    Rpp32f *meanTensor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

RppStatus tensor_stddev_f32_f32_host(Rpp32f *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp32f *tensorStddevArr,
                                     Rpp32f *meanTensor,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);

RppStatus tensor_stddev_f16_f32_host(Rpp16f *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp32f *tensorStddevArr,
                                     Rpp32f *meanTensor,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);

RppStatus tensor_stddev_i8_f32_host(Rpp8s *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp32f *tensorStddevArr,
                                    Rpp32f *meanTensor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

// -------------------- threshold --------------------

RppStatus threshold_u8_u8_host_tensor(Rpp8u *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8u *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32f *minTensor,
                                      Rpp32f *maxTensor,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams layoutParams,
                                      rpp::Handle& handle);

RppStatus threshold_f32_f32_host_tensor(Rpp32f *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp32f *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        Rpp32f *minTensor,
                                        Rpp32f *maxTensor,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle);

RppStatus threshold_i8_i8_host_tensor(Rpp8s *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8s *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32f *minTensor,
                                      Rpp32f *maxTensor,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams layoutParams,
                                      rpp::Handle& handle);

RppStatus threshold_f16_f16_host_tensor(Rpp16f *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp16f *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        Rpp32f *minTensor,
                                        Rpp32f *maxTensor,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle);

// -------------------- concat --------------------

RppStatus concat_u8_u8_host_tensor(Rpp8u *srcPtr1,
                                   Rpp8u *srcPtr2,
                                   RpptGenericDescPtr srcPtr1GenericDescPtr,
                                   RpptGenericDescPtr srcPtr2GenericDescPtr,
                                   Rpp8u *dstPtr,
                                   RpptGenericDescPtr dstGenericDescPtr,
                                   Rpp32u axisMask,
                                   Rpp32u *srcPtr1roiTensor,
                                   Rpp32u *srcPtr2roiTensor,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

RppStatus concat_f32_f32_host_tensor(Rpp32f *srcPtr1,
                                     Rpp32f *srcPtr2,
                                     RpptGenericDescPtr srcPtr1GenericDescPtr,
                                     RpptGenericDescPtr srcPtr2GenericDescPtr,
                                     Rpp32f *dstPtr,
                                     RpptGenericDescPtr dstGenericDescPtr,
                                     Rpp32u axisMask,
                                     Rpp32u *srcPtr1roiTensor,
                                     Rpp32u *srcPtr2roiTensor,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);

template<typename T1, typename T2>
RppStatus concat_generic_host_tensor(T1 *srcPtr1,
                                     T1 *srcPtr2,
                                     RpptGenericDescPtr srcPtr1GenericDescPtr,
                                     RpptGenericDescPtr srcPtr2GenericDescPtr,
                                     T2 *dstPtr,
                                     RpptGenericDescPtr dstGenericDescPtr,
                                     Rpp32u axisMask,
                                     Rpp32u *srcPtr1roiTensor,
                                     Rpp32u *srcPtr2roiTensor,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);


#endif // HOST_TENSOR_EXECUTORS_HPP
