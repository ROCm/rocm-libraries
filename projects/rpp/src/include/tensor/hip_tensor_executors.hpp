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

#ifndef HIP_TENSOR_EXECUTORS_HPP
#define HIP_TENSOR_EXECUTORS_HPP

#include <hip/hip_runtime.h>

#include "rpp_hip_load_store.hpp"

/**************************************** ARITHMETIC OPERATIONS
 * ****************************************/

// -------------------- add_scalar --------------------

RppStatus hip_exec_add_scalar_tensor(Rpp32f* srcPtr, RpptGenericDescPtr srcGenericDescPtr,
                                     Rpp32f* dstPtr, RpptGenericDescPtr dstGenericDescPtr,
                                     RpptROI3DPtr roiGenericPtrSrc, Rpp32f* addTensor,
                                     rpp::Handle& handle);

// -------------------- fused_multiply_add_scalar --------------------

RppStatus hip_exec_fused_multiply_add_scalar_tensor(
    Rpp32f* srcPtr, RpptGenericDescPtr srcGenericDescPtr, Rpp32f* dstPtr,
    RpptGenericDescPtr dstGenericDescPtr, RpptROI3DPtr roiGenericPtrSrc, Rpp32f* mulTensor,
    Rpp32f* addTensor, rpp::Handle& handle);

// -------------------- log_generic --------------------

template <typename T, typename U>
RppStatus hip_exec_log_generic_tensor(T* srcPtr, RpptGenericDescPtr srcGenericDescPtr, U* dstPtr,
                                      RpptGenericDescPtr dstGenericDescPtr, uint* roiTensor,
                                      rpp::Handle& handle);

// -------------------- log1p --------------------

RppStatus hip_exec_log1p_i16_f32_tensor(Rpp16s* srcPtr, RpptGenericDescPtr srcGenericDescPtr,
                                        Rpp32f* dstPtr, RpptGenericDescPtr dstGenericDescPtr,
                                        uint* roiTensor, rpp::Handle& handle);

// -------------------- magnitude --------------------

template <typename T>
RppStatus hip_exec_magnitude_tensor(T* srcPtr1, T* srcPtr2, RpptDescPtr srcDescPtr, T* dstPtr,
                                    RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType, rpp::Handle& handle);

// -------------------- multiply_scalar --------------------

RppStatus hip_exec_multiply_scalar_tensor(Rpp32f* srcPtr, RpptGenericDescPtr srcGenericDescPtr,
                                          Rpp32f* dstPtr, RpptGenericDescPtr dstGenericDescPtr,
                                          RpptROI3DPtr roiGenericPtrSrc, Rpp32f* mulTensor,
                                          rpp::Handle& handle);

// -------------------- subtract_scalar --------------------

RppStatus hip_exec_subtract_scalar_tensor(Rpp32f* srcPtr, RpptGenericDescPtr srcGenericDescPtr,
                                          Rpp32f* dstPtr, RpptGenericDescPtr dstGenericDescPtr,
                                          RpptROI3DPtr roiGenericPtrSrc, Rpp32f* subtractTensor,
                                          rpp::Handle& handle);

// -------------------- tensor_arithmetic_operations --------------------

template <typename T1, typename T2>
RppStatus tensor_binary_arithmetic_op_dispatch_gpu_tensor(
    T1* srcPtr1, T1* srcPtr2, RpptGenericDescPtr srcPtr1GenericDescPtr,
    RpptGenericDescPtr srcPtr2GenericDescPtr, T2* dstPtr, RpptGenericDescPtr dstGenericDescPtr,
    RpptOp tensorOp, RpptBroadcastMode broadcastMode, Rpp32u* roiTensor1, Rpp32u* roiTensor2,
    rpp::Handle& handle);

/**************************************** AUDIO AUGMENTATIONS
 * ****************************************/

#ifdef AUDIO_SUPPORT

// -------------------- down_mixing --------------------

RppStatus hip_exec_down_mixing_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* dstPtr,
                                      RpptDescPtr dstDescPtr, Rpp32s* srcDimsTensor,
                                      bool normalizeWeights, rpp::Handle& handle);

// -------------------- mel_filter_bank --------------------

RppStatus hip_exec_mel_filter_bank_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* dstPtr,
                                          RpptDescPtr dstDescPtr, Rpp32s* srcDimsTensor,
                                          Rpp32f maxFreqVal, Rpp32f minFreqVal,
                                          RpptMelScaleFormula melFormula, Rpp32s numFilter,
                                          Rpp32f sampleRate, bool normalize, rpp::Handle& handle);

// -------------------- non_silent_region_detection --------------------

RppStatus hip_exec_non_silent_region_detection_tensor(
    Rpp32f* srcPtr, RpptDescPtr srcDescPtr, Rpp32s* srcLengthTensor, Rpp32s* detectedIndexTensor,
    Rpp32s* detectionLengthTensor, Rpp32f cutOffDB, Rpp32s windowLength, Rpp32f referencePower,
    Rpp32s resetInterval, rpp::Handle& handle);

// -------------------- pre_emphasis_filter --------------------

RppStatus hip_exec_pre_emphasis_filter_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr,
                                              Rpp32f* dstPtr, RpptDescPtr dstDescPtr,
                                              Rpp32f* coeffTensor, Rpp32s* srcLengthTensor,
                                              RpptAudioBorderType borderType, rpp::Handle& handle);

// -------------------- resample --------------------

RppStatus hip_exec_resample_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* dstPtr,
                                   RpptDescPtr dstDescPtr, Rpp32f* inRateTensor,
                                   Rpp32f* outRateTensor, Rpp32s* srcDimsTensor,
                                   RpptResamplingWindow& window, rpp::Handle& handle);

// -------------------- spectrogram --------------------

RppStatus hip_exec_spectrogram_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* dstPtr,
                                      RpptDescPtr dstDescPtr, Rpp32s* srcLengthTensor,
                                      bool centerWindows, bool reflectPadding,
                                      Rpp32f* windowFunction, Rpp32s nfft, Rpp32s power,
                                      Rpp32s windowLength, Rpp32s windowStep, rpp::Handle& handle);

// -------------------- to_decibels --------------------

RppStatus hip_exec_to_decibels_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* dstPtr,
                                      RpptDescPtr dstDescPtr, RpptImagePatchPtr srcDims,
                                      Rpp32f cutOffDB, Rpp32f multiplier, Rpp32f referenceMagnitude,
                                      rpp::Handle& handle);

// -------------------- audio_tensor_add_tensor --------------------

RppStatus hip_exec_audio_tensor_add_tensor(Rpp32f* srcPtr1, Rpp32f* srcPtr2, RpptDescPtr srcDescPtr,
                                           Rpp32f* dstPtr, RpptDescPtr dstDescPtr,
                                           Rpp32s* srcLengthTensor, rpp::Handle& handle);

// -------------------- audio_tensor_mul_scalar --------------------

RppStatus hip_exec_audio_tensor_mul_scalar(Rpp32f* srcPtr, Rpp32f scalarValue,
                                           RpptDescPtr srcDescPtr, Rpp32f* dstPtr,
                                           RpptDescPtr dstDescPtr, Rpp32s* srcLengthTensor,
                                           rpp::Handle& handle);

#endif  // AUDIO_SUPPORT

/**************************************** BITWISE OPERATIONS
 * ****************************************/

// -------------------- bitwise_and --------------------

RppStatus hip_exec_bitwise_and_tensor(Rpp8u* srcPtr1, Rpp8u* srcPtr2, RpptDescPtr srcDescPtr,
                                      Rpp8u* dstPtr, RpptDescPtr dstDescPtr,
                                      RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                      rpp::Handle& handle);

// -------------------- bitwise_not --------------------

RppStatus hip_exec_bitwise_not_tensor(Rpp8u* srcPtr, RpptDescPtr srcDescPtr, Rpp8u* dstPtr,
                                      RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType, rpp::Handle& handle);

// -------------------- bitwise_or --------------------

RppStatus hip_exec_bitwise_or_tensor(Rpp8u* srcPtr1, Rpp8u* srcPtr2, RpptDescPtr srcDescPtr,
                                     Rpp8u* dstPtr, RpptDescPtr dstDescPtr,
                                     RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                     rpp::Handle& handle);

// -------------------- bitwise_xor --------------------

RppStatus hip_exec_bitwise_xor_tensor(Rpp8u* srcPtr1, Rpp8u* srcPtr2, RpptDescPtr srcDescPtr,
                                      Rpp8u* dstPtr, RpptDescPtr dstDescPtr,
                                      RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                      rpp::Handle& handle);

// -------------------- tensor_bitwise_operations --------------------

template <typename T>
RppStatus tensor_binary_bitwise_op_dispatch_gpu_tensor(
    T* srcPtr1, T* srcPtr2, RpptGenericDescPtr srcPtr1GenericDescPtr,
    RpptGenericDescPtr srcPtr2GenericDescPtr, T* dstPtr, RpptGenericDescPtr dstGenericDescPtr,
    RpptBitwiseOp tensorOp, RpptBroadcastMode broadcastMode, Rpp32u* roiTensor1, Rpp32u* roiTensor2,
    rpp::Handle& handle);

/**************************************** COLOR AUGMENTATIONS
 * ****************************************/

// -------------------- brightness --------------------

template <typename T>
RppStatus hip_exec_brightness_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32f* alphaTensor,
                                     Rpp32f* betaTensor, RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType, rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_brightness_single_image(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                           RpptDescPtr dstDescPtr, Rpp32f* alphaTensor,
                                           Rpp32f* betaTensor, RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType, rpp::Handle& handle);

// -------------------- blend --------------------

template <typename T>
RppStatus hip_exec_blend_tensor(T* srcPtr1, T* srcPtr2, RpptDescPtr srcDescPtr, T* dstPtr,
                                RpptDescPtr dstDescPtr, Rpp32f* alphaTensor,
                                RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_blend_single_image(T* srcPtr1, T* srcPtr2, RpptDescPtr srcDescPtr, T* dstPtr,
                                      RpptDescPtr dstDescPtr, Rpp32f* alphaTensor,
                                      RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                      rpp::Handle& handle);

// -------------------- color_cast --------------------

template <typename T>
RppStatus hip_exec_color_cast_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                     RpptDescPtr dstDescPtr, RpptRGB* rgbTensor,
                                     Rpp32f* alphaTensor, RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType, rpp::Handle& handle);

// -------------------- color_temperature --------------------

template <typename T>
RppStatus hip_exec_color_temperature_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                            RpptDescPtr dstDescPtr, Rpp32s* adjustmentValueTensor,
                                            RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                            rpp::Handle& handle);

// -------------------- hue --------------------

template <typename T>
RppStatus hip_exec_hue_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr, RpptDescPtr dstDescPtr,
                              Rpp32f* hueTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                              rpp::Handle& handle);

// -------------------- saturation --------------------

template <typename T>
RppStatus hip_exec_saturation_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32f* saturationTensor,
                                     RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                     rpp::Handle& handle);

// -------------------- color_twist --------------------

template <typename T>
RppStatus hip_exec_color_twist_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                      RpptDescPtr dstDescPtr, Rpp32f* brightnessTensor,
                                      Rpp32f* contrastTensor, Rpp32f* hueTensor,
                                      Rpp32f* saturationTensor, RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType, rpp::Handle& handle);

// -------------------- contrast --------------------

template <typename T>
RppStatus hip_exec_contrast_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                   RpptDescPtr dstDescPtr, Rpp32f* contrastFactorTensor,
                                   Rpp32f* contrastCenterTensor, RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType, rpp::Handle& handle);

// -------------------- exposure --------------------

template <typename T>
RppStatus hip_exec_exposure_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                   RpptDescPtr dstDescPtr, Rpp32f* exposureFactorTensor,
                                   RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                   rpp::Handle& handle);

// -------------------- gamma_correction --------------------

template <typename T>
RppStatus hip_exec_gamma_correction_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                           RpptDescPtr dstDescPtr, Rpp32f* gammaTensor,
                                           RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                           rpp::Handle& handle);

// -------------------- lut --------------------

template <typename T, typename U>
RppStatus hip_exec_lut_tensor(T* srcPtr, RpptDescPtr srcDescPtr, U* dstPtr, RpptDescPtr dstDescPtr,
                              U* lutPtr, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                              rpp::Handle& handle);

// -------------------- histogram_equalize --------------------

RppStatus hip_exec_histogram_equalize_tensor(Rpp8u* srcPtr, RpptDescPtr srcDescPtr, Rpp8u* dstPtr,
                                             RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType, rpp::Handle& handle);

/**************************************** DATA EXCHANGE OPERATIONS
 * ****************************************/

// -------------------- color_to_greyscale --------------------

template <typename T>
RppStatus hip_exec_color_to_greyscale_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                             RpptDescPtr dstDescPtr, Rpp32f* channelWeights,
                                             rpp::Handle& handle);

// -------------------- copy --------------------

template <typename T>
RppStatus hip_exec_copy_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr, RpptDescPtr dstDescPtr,
                               rpp::Handle& handle);

// -------------------- channel_permute --------------------

template <typename T>
RppStatus hip_exec_channel_permute_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                          RpptDescPtr dstDescPtr, Rpp32u* permutationTensor,
                                          rpp::Handle& handle);

// -------------------- yuv_to_rgb (NV12 8-bit) --------------------

template <typename T>
RppStatus hip_exec_yuv_to_rgb(T* srcYPtr, Rpp32u src_y_pitch, T* srcUVPtr, Rpp32u src_uv_pitch,
                              T* dstPtr, Rpp32u dst_pitch, Rpp32u width, Rpp32u height,
                              RpptColorStandard col_standard, RpptColorRange color_range,
                              rpp::Handle& handle);

// -------------------- yuv_to_rgb_cubic_v (NV12 8-bit, cubic vertical chroma upsample)
// --------------------

template <typename T>
RppStatus hip_exec_yuv_to_rgb_cubic_v(T* srcYPtr, Rpp32u src_y_pitch, T* srcUVPtr,
                                      Rpp32u src_uv_pitch, T* dstPtr, Rpp32u dst_pitch,
                                      Rpp32u width, Rpp32u height, RpptColorStandard col_standard,
                                      RpptColorRange color_range, rpp::Handle& handle);

// -------------------- yuv_to_rgb_linear_v (NV12 8-bit, linear vertical chroma upsample)
// --------------------

template <typename T>
RppStatus hip_exec_yuv_to_rgb_linear_v(T* srcYPtr, Rpp32u src_y_pitch, T* srcUVPtr,
                                       Rpp32u src_uv_pitch, T* dstPtr, Rpp32u dst_pitch,
                                       Rpp32u width, Rpp32u height, RpptColorStandard col_standard,
                                       RpptColorRange color_range, rpp::Handle& handle);

/**************************************** EFFECTS AUGMENTATIONS
 * ****************************************/

// -------------------- erase --------------------

template <typename T, typename U>
RppStatus hip_exec_erase_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                RpptDescPtr dstDescPtr, RpptRoiLtrb* anchorBoxInfoTensor,
                                U* colorsTensor, Rpp32u* numBoxesTensor, RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType, rpp::Handle& handle);

// -------------------- fog --------------------

template <typename T>
RppStatus hip_exec_fog_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr, RpptDescPtr dstDescPtr,
                              Rpp32f* d_fogAlphaMaskPtr, Rpp32f* d_fogIntensityMaskPtr,
                              Rpp32f* intensityFactor, Rpp32f* greyFactor, Rpp32u* maskLocOffsetX,
                              Rpp32u* maskLocOffsetY, RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType, rpp::Handle& handle);

// -------------------- noise_gaussian --------------------

template <typename T>
RppStatus hip_exec_gaussian_noise_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                         RpptDescPtr dstDescPtr, Rpp32f* meanTensor,
                                         Rpp32f* stdDevTensor,
                                         RpptXorwowStateBoxMuller* xorwowInitialStatePtr,
                                         RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                         rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_gaussian_noise_voxel_tensor(T* srcPtr, RpptGenericDescPtr srcGenericDescPtr,
                                               T* dstPtr, RpptGenericDescPtr dstGenericDescPtr,
                                               RpptXorwowStateBoxMuller* xorwowInitialStatePtr,
                                               Rpp32f* meanTensor, Rpp32f* stdDevTensor,
                                               RpptROI3DPtr roiGenericPtrSrc, rpp::Handle& handle);

// -------------------- glitch --------------------

template <typename T>
RppStatus hip_exec_glitch_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                 RpptDescPtr dstDescPtr, RpptChannelOffsets* rgbOffsets,
                                 RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                 rpp::Handle& handle);

// -------------------- gridmask --------------------

template <typename T>
RppStatus hip_exec_gridmask_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                   RpptDescPtr dstDescPtr, Rpp32u tileWidth, Rpp32f gridRatio,
                                   Rpp32f gridAngle, RpptUintVector2D translateVector,
                                   RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                   rpp::Handle& handle);

// -------------------- jitter --------------------

template <typename T>
RppStatus hip_exec_jitter_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                 RpptDescPtr dstDescPtr, uint* kernelSizeTensor,
                                 RpptXorwowStateBoxMuller* xorwowInitialStatePtr,
                                 RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                 rpp::Handle& handle);

// -------------------- non_linear_blend --------------------

template <typename T>
RppStatus hip_exec_non_linear_blend_tensor(T* srcPtr1, T* srcPtr2, RpptDescPtr srcDescPtr,
                                           T* dstPtr, RpptDescPtr dstDescPtr, Rpp32f* stdDevTensor,
                                           RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                           rpp::Handle& handle);

// -------------------- rain --------------------

template <typename T>
RppStatus hip_exec_rain_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr, RpptDescPtr dstDescPtr,
                               Rpp32f rainPercentage, Rpp32u rainWidth, Rpp32u rainHeight,
                               Rpp32f slantAngle, Rpp32f* alpha, RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType, rpp::Handle& handle);

// -------------------- ricap --------------------

template <typename T>
RppStatus hip_exec_ricap_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                RpptDescPtr dstDescPtr, Rpp32u* permutationTensor,
                                RpptROIPtr roiPtrInputCropRegion, RpptRoiType roiType,
                                rpp::Handle& handle);

// -------------------- noise_salt_and_pepper --------------------

template <typename T>
RppStatus hip_exec_salt_and_pepper_noise_tensor(
    T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr, RpptDescPtr dstDescPtr,
    Rpp32f* noiseProbabilityTensor, Rpp32f* saltProbabilityTensor, Rpp32f* saltValueTensor,
    Rpp32f* pepperValueTensor, RpptXorwowState* xorwowInitialStatePtr, RpptROIPtr roiTensorPtrSrc,
    RpptRoiType roiType, rpp::Handle& handle);

// -------------------- noise_shot --------------------

template <typename T>
RppStatus hip_exec_shot_noise_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32f* shotNoiseFactorTensor,
                                     RpptXorwowStateBoxMuller* xorwowInitialStatePtr,
                                     RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                     rpp::Handle& handle);

// -------------------- spatter --------------------

template <typename T>
RppStatus hip_exec_spatter_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                  RpptDescPtr dstDescPtr, RpptRGB spatterColor,
                                  Rpp32u* maskLocArrHostX, Rpp32u* maskLocArrHostY,
                                  RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                  rpp::Handle& handle);

// -------------------- vignette --------------------

template <typename T>
RppStatus hip_exec_vignette_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                   RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc,
                                   Rpp32f* vignetteIntensityTensor, RpptRoiType roiType,
                                   rpp::Handle& handle);

// -------------------- water --------------------

template <typename T>
RppStatus hip_exec_water_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                RpptDescPtr dstDescPtr, Rpp32f* amplitudeXTensor,
                                Rpp32f* amplitudeYTensor, Rpp32f* frequencyXTensor,
                                Rpp32f* frequencyYTensor, Rpp32f* phaseXTensor,
                                Rpp32f* phaseYTensor, RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType, rpp::Handle& handle);

// -------------------- posterize --------------------

template <typename T>
RppStatus hip_exec_posterize_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                    RpptDescPtr dstDescPtr, Rpp8u* posterizeLevelBits,
                                    RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                    rpp::Handle& handle);

// -------------------- solarize --------------------

template <typename T>
RppStatus hip_exec_solarize_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                   RpptDescPtr dstDescPtr, Rpp32f* thresholdTensor,
                                   RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                   rpp::Handle& handle);

// -------------------- snow --------------------

template <typename T>
RppStatus hip_exec_snow_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr, RpptDescPtr dstDescPtr,
                               Rpp32f* brightnessCoefficient, Rpp32f* snowThreshold,
                               Rpp32s* darkMode, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                               rpp::Handle& handle);

// -------------------- channel_dropout --------------------

template <typename T>
RppStatus hip_exec_channel_dropout_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                          RpptDescPtr dstDescPtr, Rpp8u* dropoutTensor,
                                          RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                          rpp::Handle& handle);

// -------------------- coarse_dropout --------------------

template <typename T>
RppStatus hip_exec_coarse_dropout_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                         RpptDescPtr dstDescPtr, RpptRoiLtrb* anchorBoxInfoTensor,
                                         Rpp32u* numBoxesTensor, Rpp32u maxBoxesPerImage,
                                         RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                         rpp::Handle& handle);

// -------------------- grid_dropout --------------------

template <typename T>
RppStatus hip_exec_grid_dropout_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                       RpptDescPtr dstDescPtr, RpptRoiLtrb* anchorBoxInfoTensor,
                                       Rpp32u boxesInEachImage, Rpp32u maxHoleW, Rpp32u maxHoleH,
                                       RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                       rpp::Handle& handle);

// -------------------- random_erase --------------------

template <typename T>
RppStatus hip_exec_random_erase_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                       RpptDescPtr dstDescPtr, RpptRoiLtrb* anchorBoxInfoTensor,
                                       T* noiseBuffer, RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType, rpp::Handle& handle);

/**************************************** FILTER AUGMENTATIONS
 * ****************************************/

// -------------------- box_filter --------------------

template <typename T>
RppStatus hip_exec_box_filter_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32u kernelSize,
                                     RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                     rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_box_filter_single_image(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                           RpptDescPtr dstDescPtr, Rpp32u kernelSize,
                                           RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                           rpp::Handle& handle);

// -------------------- gaussian_filter --------------------

template <typename T>
RppStatus hip_exec_gaussian_filter_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                          RpptDescPtr dstDescPtr, Rpp32f* stdDevTensor,
                                          Rpp32u kernelSize, RpptROIPtr roiTensorPtrSrc,
                                          RpptRoiType roiType, rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_gaussian_filter_single_image(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                                RpptDescPtr dstDescPtr, Rpp32f* stdDevTensor,
                                                Rpp32u kernelSize, RpptROIPtr roiTensorPtrSrc,
                                                RpptRoiType roiType, rpp::Handle& handle);

// -------------------- sobel_filter --------------------

template <typename T>
RppStatus hip_exec_sobel_filter_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                       RpptDescPtr dstDescPtr, Rpp32u sobelType, Rpp32u kernelSize,
                                       RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                       rpp::Handle& handle);

// -------------------- median_filter --------------------

template <typename T>
RppStatus hip_exec_median_filter_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                        RpptDescPtr dstDescPtr, Rpp32u kernelSize,
                                        RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                        rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_median_filter_single_image(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                              RpptDescPtr dstDescPtr, Rpp32u kernelSize,
                                              RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                              rpp::Handle& handle);

// -------------------- emboss --------------------

template <typename T>
RppStatus hip_exec_emboss_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                 RpptDescPtr dstDescPtr, Rpp32f* strength, Rpp32u kernelSize,
                                 RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                 rpp::Handle& handle);

/**************************************** GEOMETRIC AUGMENTATIONS
 * ****************************************/

// -------------------- crop --------------------

template <typename T>
RppStatus hip_exec_crop_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr, RpptDescPtr dstDescPtr,
                               RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                               rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_crop_single_image(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                     RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType, rpp::Handle& handle);

// -------------------- crop_and_patch --------------------

template <typename T>
RppStatus hip_exec_crop_and_patch_tensor(T* srcPtr1, T* srcPtr2, RpptDescPtr srcDescPtr, T* dstPtr,
                                         RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc,
                                         RpptROIPtr cropTensorPtr, RpptROIPtr patchTensorPtr,
                                         RpptRoiType roiType, rpp::Handle& handle);

// -------------------- crop_mirror_normalize --------------------

template <typename T, typename U>
RppStatus hip_exec_crop_mirror_normalize_tensor(T* srcPtr, RpptDescPtr srcDescPtr, U* dstPtr,
                                                RpptDescPtr dstDescPtr, Rpp32f* offsetTensor,
                                                Rpp32f* multiplierTensor, Rpp32u* mirrorTensor,
                                                RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                                rpp::Handle& handle);

// -------------------- flip --------------------

template <typename T>
RppStatus hip_exec_flip_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr, RpptDescPtr dstDescPtr,
                               Rpp32u* horizontalTensor, Rpp32u* verticalTensor,
                               RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                               rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_flip_single_image(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32u* horizontalTensor,
                                     Rpp32u* verticalTensor, RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType, rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_flip_voxel_tensor(T* srcPtr, RpptGenericDescPtr srcGenericDescPtr, T* dstPtr,
                                     RpptGenericDescPtr dstGenericDescPtr,
                                     RpptROI3DPtr roiGenericPtrSrc, Rpp32u* horizontalTensor,
                                     Rpp32u* verticalTensor, Rpp32u* depthTensor,
                                     RpptRoi3DType roiType, rpp::Handle& handle);

// -------------------- remap --------------------

template <typename T>
RppStatus hip_exec_remap_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                RpptDescPtr dstDescPtr, Rpp32f* rowRemapTable,
                                Rpp32f* colRemapTable, RpptDescPtr remapTableDescPtr,
                                RpptInterpolationType interpolationType, RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType, rpp::Handle& handle);

// -------------------- resize --------------------

template <typename T>
RppStatus hip_exec_resize_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                 RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSize,
                                 RpptInterpolationType interpolationType,
                                 RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                 rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_resize_single_image(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                       RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSize,
                                       RpptInterpolationType interpolationType,
                                       RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                       rpp::Handle& handle);

// -------------------- resize_crop_mirror --------------------

template <typename T>
RppStatus hip_exec_resize_crop_mirror_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                             RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSizes,
                                             Rpp32u* mirrorTensor,
                                             RpptInterpolationType interpolationType,
                                             RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                             rpp::Handle& handle);

// -------------------- resize_mirror_normalize --------------------

template <typename T, typename U>
RppStatus hip_exec_resize_mirror_normalize_tensor(T* srcPtr, RpptDescPtr srcDescPtr, U* dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  RpptImagePatchPtr dstImgSizes,
                                                  RpptInterpolationType interpolationType,
                                                  Rpp32f* meanTensor, Rpp32f* stdDevTensor,
                                                  Rpp32u* mirrorTensor, RpptROIPtr roiTensorPtrSrc,
                                                  RpptRoiType roiType, rpp::Handle& handle);

// -------------------- lens_correction --------------------

RppStatus hip_exec_lens_correction_tensor(RpptDescPtr dstDescPtr, Rpp32f* rowRemapTable,
                                          Rpp32f* colRemapTable, RpptDescPtr remapTableDescPtr,
                                          Rpp32f* cameraMatrix, Rpp32f* distanceCoeffs,
                                          RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                          rpp::Handle& handle);

// -------------------- phase --------------------

template <typename T>
RppStatus hip_exec_phase_tensor(T* srcPtr1, T* srcPtr2, RpptDescPtr srcDescPtr, T* dstPtr,
                                RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType, rpp::Handle& handle);

// -------------------- slice --------------------

template <typename T>
RppStatus hip_exec_slice_tensor(T* srcPtr, RpptGenericDescPtr srcGenericDescPtr, T* dstPtr,
                                RpptGenericDescPtr dstGenericDescPtr, Rpp32s* anchorTensor,
                                Rpp32s* shapeTensor, T* fillValue, bool enablePadding,
                                Rpp32u* roiTensor, rpp::Handle& handle);

// -------------------- transpose --------------------

template <typename T>
RppStatus hip_exec_transpose_tensor(T* srcPtr, RpptGenericDescPtr srcGenericDescPtr, T* dstPtr,
                                    RpptGenericDescPtr dstGenericDescPtr, Rpp32u* permTensor,
                                    Rpp32u* roiTensor, rpp::Handle& handle);

// -------------------- warp_affine --------------------

template <typename T>
RppStatus hip_exec_warp_affine_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                      RpptDescPtr dstDescPtr, Rpp32f* affineTensor,
                                      RpptInterpolationType interpolationType,
                                      RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                      rpp::Handle& handle);

// -------------------- warp_perspective --------------------

template <typename T>
RppStatus hip_exec_warp_perspective_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                           RpptDescPtr dstDescPtr, Rpp32f* perspectiveTensor,
                                           RpptInterpolationType interpolationType,
                                           RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                           rpp::Handle& handle);

// -------------------- fisheye --------------------

template <typename T>
RppStatus hip_exec_fisheye_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                  RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType, rpp::Handle& handle);

/**************************************** MORPHOLOGICAL OPERATIONS
 * ****************************************/

template <typename T>
RppStatus hip_exec_dilate_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                 RpptDescPtr dstDescPtr, Rpp32u kernelSize,
                                 RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                 rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_erode_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                RpptDescPtr dstDescPtr, Rpp32u kernelSize,
                                RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                rpp::Handle& handle);

// -------------------- normalize --------------------

template <typename T>
RppStatus hip_exec_normalize_tensor(T* srcPtr, RpptGenericDescPtr srcGenericDescPtr, T* dstPtr,
                                    RpptGenericDescPtr dstGenericDescPtr, Rpp32u axisMask,
                                    Rpp32f* meanTensor, Rpp32f* stdDevTensor,
                                    Rpp8u computeMeanStddev, Rpp32f scale, Rpp32f shift,
                                    Rpp32u* roiTensor, rpp::Handle& handle);

// -------------------- tensor_max --------------------

template <typename T, typename U>
RppStatus hip_exec_tensor_max(T* srcPtr, RpptDescPtr srcDescPtr, U* maxArr,
                              RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rpp::Handle& handle);

// -------------------- tensor_mean --------------------

template <typename T, typename U>
RppStatus hip_exec_tensor_mean(T* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* tensorMeanArr,
                               RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                               rpp::Handle& handle);

// -------------------- tensor_min --------------------

template <typename T, typename U>
RppStatus hip_exec_tensor_min(T* srcPtr, RpptDescPtr srcDescPtr, U* minArr,
                              RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rpp::Handle& handle);

// -------------------- tensor_stddev --------------------

template <typename T>
RppStatus hip_exec_tensor_stddev(T* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* imageStddevArr,
                                 Rpp32f* meanTensor, RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType, rpp::Handle& handle);

// -------------------- tensor_sum --------------------

template <typename T, typename U>
__global__ void tensor_sum_pln1_hip(T* srcPtr, uint2 srcStridesNH, U* tensorSumArr,
                                    RpptROIPtr roiTensorPtrSrc);

template <>
__global__ void tensor_sum_pln1_hip<Rpp8u, Rpp32u>(Rpp8u* srcPtr, uint2 srcStridesNH,
                                                   Rpp32u* tensorSumArr,
                                                   RpptROIPtr roiTensorPtrSrc);

template <>
__global__ void tensor_sum_pln1_hip<Rpp8s, Rpp32s>(Rpp8s* srcPtr, uint2 srcStridesNH,
                                                   Rpp32s* tensorSumArr,
                                                   RpptROIPtr roiTensorPtrSrc);

// Handle f16/f32
template <typename T, typename U>
__global__ void tensor_sum_pln3_hip(T* srcPtr, uint3 srcStridesNCH, U* tensorSumArr,
                                    RpptROIPtr roiTensorPtrSrc);

template <>
__global__ void tensor_sum_pln3_hip<Rpp8u, Rpp32u>(Rpp8u* srcPtr, uint3 srcStridesNH,
                                                   Rpp32u* tensorSumArr,
                                                   RpptROIPtr roiTensorPtrSrc);

template <>
__global__ void tensor_sum_pln3_hip<Rpp8s, Rpp32s>(Rpp8s* srcPtr, uint3 srcStridesNH,
                                                   Rpp32s* tensorSumArr,
                                                   RpptROIPtr roiTensorPtrSrc);

template <typename T, typename U>
__global__ void tensor_sum_pkd3_hip(T* srcPtr, uint2 srcStridesNH, U* tensorSumArr,
                                    RpptROIPtr roiTensorPtrSrc);

template <>
__global__ void tensor_sum_pkd3_hip<Rpp8u, Rpp32u>(Rpp8u* srcPtr, uint2 srcStridesNH,
                                                   Rpp32u* tensorSumArr,
                                                   RpptROIPtr roiTensorPtrSrc);

template <>
__global__ void tensor_sum_pkd3_hip<Rpp8s, Rpp32s>(Rpp8s* srcPtr, uint2 srcStridesNH,
                                                   Rpp32s* tensorSumArr,
                                                   RpptROIPtr roiTensorPtrSrc);

template <typename T, typename U>
RppStatus hip_exec_tensor_sum(T* srcPtr, RpptDescPtr srcDescPtr, U* tensorSumArr,
                              RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rpp::Handle& handle);

template <>
RppStatus hip_exec_tensor_sum<Rpp8u, Rpp64u>(Rpp8u* srcPtr, RpptDescPtr srcDescPtr,
                                             Rpp64u* tensorSumArr, RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType, rpp::Handle& handle);

template <>
RppStatus hip_exec_tensor_sum<Rpp8s, Rpp64s>(Rpp8s* srcPtr, RpptDescPtr srcDescPtr,
                                             Rpp64s* tensorSumArr, RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType, rpp::Handle& handle);

// -------------------- threshold --------------------

template <typename T>
RppStatus hip_exec_threshold_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                    RpptDescPtr dstDescPtr, Rpp32f* minTensor, Rpp32f* maxTensor,
                                    RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                    rpp::Handle& handle);

// -------------------- concat --------------------

template <typename T>
RppStatus hip_exec_concat_tensor(T* srcPtr1, RpptGenericDescPtr srcPtr1GenericDescPtr, T* srcPtr2,
                                 RpptGenericDescPtr srcPtr2GenericDescPtr, T* dstPtr,
                                 RpptGenericDescPtr dstGenericDescPtr, Rpp32u axis,
                                 Rpp32u* srcPtr1roiTensor, Rpp32u* srcPtr2roiTensor,
                                 rpp::Handle& handle);

// -------------------- jpeg_compression distortion --------------------

template <typename T>
RppStatus hip_exec_jpeg_compression_distortion(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                               RpptDescPtr dstDescPtr, Rpp32s* qualityTensor,
                                               RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                               rpp::Handle& handle);

#endif  // HIP_TENSOR_EXECUTORS_HPP
