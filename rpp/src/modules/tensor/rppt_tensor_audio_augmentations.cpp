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

#ifdef AUDIO_SUPPORT

#include "rppdefs.h"
#include "rppi_validate.hpp"
#include "rppt_tensor_audio_augmentations.h"
#include "host_tensor_executors.hpp"

#ifdef HIP_COMPILE
#include "hip_tensor_executors.hpp"
#endif // HIP_COMPILE

/******************** non_silent_region_detection ********************/

RppStatus rppt_non_silent_region_detection_host(RppPtr_t srcPtr,
                                                RpptDescPtr srcDescPtr,
                                                Rpp32s *srcLengthTensor,
                                                Rpp32s *detectedIndexTensor,
                                                Rpp32s *detectionLengthTensor,
                                                Rpp32f cutOffDB,
                                                Rpp32s windowLength,
                                                Rpp32f referencePower,
                                                Rpp32s resetInterval,
                                                rppHandle_t rppHandle)
{
    // Disabled this check for now.
    // This check will be re-enabled when the numDims based changes are added in MIVisionX */
    // Rpp32u tensorDims = srcDescPtr->numDims - 1; // exclude batchsize from input dims
    // if (tensorDims != 1)
    //     return RPP_ERROR_INVALID_SRC_DIMS;

    if (srcDescPtr->dataType == RpptDataType::F32)
    {
        non_silent_region_detection_host_tensor(static_cast<Rpp32f*>(srcPtr),
                                                srcDescPtr,
                                                srcLengthTensor,
                                                detectedIndexTensor,
                                                detectionLengthTensor,
                                                cutOffDB,
                                                windowLength,
                                                referencePower,
                                                resetInterval,
                                                rpp::deref(rppHandle));

        return RPP_SUCCESS;
    }
    else
    {
        return RPP_ERROR_NOT_IMPLEMENTED;
    }

    return RPP_SUCCESS;
}

/******************** to_decibels ********************/

RppStatus rppt_to_decibels_host(RppPtr_t srcPtr,
                                RpptDescPtr srcDescPtr,
                                RppPtr_t dstPtr,
                                RpptDescPtr dstDescPtr,
                                RpptImagePatchPtr srcDims,
                                Rpp32f cutOffDB,
                                Rpp32f multiplier,
                                Rpp32f referenceMagnitude,
                                rppHandle_t rppHandle)
{
    // Disabled this check for now.
    // This check will be re-enabled when the numDims based changes are added in MIVisionX */
    // Rpp32u tensorDims = srcDescPtr->numDims - 1; // exclude batchsize from input dims
    // if (tensorDims != 1 && tensorDims != 2)
    //     return RPP_ERROR_INVALID_SRC_DIMS;

    if (!multiplier)
        return RPP_ERROR_ZERO_DIVISION;

    if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        to_decibels_host_tensor(static_cast<Rpp32f*>(srcPtr),
                                srcDescPtr,
                                static_cast<Rpp32f*>(dstPtr),
                                dstDescPtr,
                                srcDims,
                                cutOffDB,
                                multiplier,
                                referenceMagnitude,
                                rpp::deref(rppHandle));

        return RPP_SUCCESS;
    }
    else
    {
        return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    }
}

/******************** pre_emphasis_filter ********************/

RppStatus rppt_pre_emphasis_filter_host(RppPtr_t srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        RppPtr_t dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        Rpp32s *srcLengthTensor,
                                        Rpp32f *coeffTensor,
                                        RpptAudioBorderType borderType,
                                        rppHandle_t rppHandle)
{
    // Disabled this check for now.
    // This check will be re-enabled when the numDims based changes are added in MIVisionX */
    // if (srcDescPtr->numDims != 2)
    //     return RPP_ERROR_INVALID_SRC_DIMS;

    if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        pre_emphasis_filter_host_tensor(static_cast<Rpp32f*>(srcPtr),
                                        srcDescPtr,
                                        static_cast<Rpp32f*>(dstPtr),
                                        dstDescPtr,
                                        srcLengthTensor,
                                        coeffTensor,
                                        borderType,
                                        rpp::deref(rppHandle));

        return RPP_SUCCESS;
    }
    else
    {
        return RPP_ERROR_NOT_IMPLEMENTED;
    }
}

/******************** down_mixing ********************/

RppStatus rppt_down_mixing_host(RppPtr_t srcPtr,
                                RpptDescPtr srcDescPtr,
                                RppPtr_t dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32s *srcDimsTensor,
                                bool  normalizeWeights,
                                rppHandle_t rppHandle)
{
    // Disabled this check for now.
    // This check will be re-enabled when the numDims based changes are added in MIVisionX */
    // Rpp32u tensorDims = srcDescPtr->numDims - 1; // exclude batchsize from input dims
    // if (tensorDims != 1 && tensorDims != 2)
    //     return RPP_ERROR_INVALID_SRC_DIMS;

    if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        down_mixing_host_tensor(static_cast<Rpp32f*>(srcPtr),
                                srcDescPtr,
                                static_cast<Rpp32f*>(dstPtr),
                                dstDescPtr,
                                srcDimsTensor,
                                normalizeWeights,
                                rpp::deref(rppHandle));

        return RPP_SUCCESS;
    }
    else
    {
        return RPP_ERROR_NOT_IMPLEMENTED;
    }
}

/******************** spectrogram ********************/

RppStatus rppt_spectrogram_host(RppPtr_t srcPtr,
                                RpptDescPtr srcDescPtr,
                                RppPtr_t dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32s *srcLengthTensor,
                                bool centerWindows,
                                bool reflectPadding,
                                Rpp32f *windowFunction,
                                Rpp32s nfft,
                                Rpp32s power,
                                Rpp32s windowLength,
                                Rpp32s windowStep,
                                rppHandle_t rppHandle)
{
    if ((dstDescPtr->layout != RpptLayout::NFT) && (dstDescPtr->layout != RpptLayout::NTF))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    // Disabled this checks for now.
    // This check will be re-enabled when the numDims based changes are added in MIVisionX */
    // Rpp32u srcTensorDims = srcDescPtr->numDims - 1; // exclude batchsize from input dims
    // Rpp32u dstTensorDims = dstDescPtr->numDims - 1; // exclude batchsize from output dims
    // if (srcTensorDims != 1)
    //     return RPP_ERROR_INVALID_SRC_DIMS;
    // if (dstTensorDims != 2)
    //     return RPP_ERROR_INVALID_DST_DIMS;

    if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        spectrogram_host_tensor(static_cast<Rpp32f*>(srcPtr),
                                srcDescPtr,
                                static_cast<Rpp32f*>(dstPtr),
                                dstDescPtr,
                                srcLengthTensor,
                                centerWindows,
                                reflectPadding,
                                windowFunction,
                                nfft,
                                power,
                                windowLength,
                                windowStep,
                                rpp::deref(rppHandle));

        return RPP_SUCCESS;
    }
    else
    {
        return RPP_ERROR_NOT_IMPLEMENTED;
    }
}

/******************** mel_filter_bank ********************/

RppStatus rppt_mel_filter_bank_host(RppPtr_t srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    RppPtr_t dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32s* srcDimsTensor,
                                    Rpp32f maxFreq,
                                    Rpp32f minFreq,
                                    RpptMelScaleFormula melFormula,
                                    Rpp32s numFilter,
                                    Rpp32f sampleRate,
                                    bool normalize,
                                    rppHandle_t rppHandle)
{
    if (srcDescPtr->layout != RpptLayout::NFT) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if (dstDescPtr->layout != RpptLayout::NFT) return RPP_ERROR_INVALID_DST_LAYOUT;
    // Disabled this check for now.
    // This check will be re-enabled when the numDims based changes are added in MIVisionX */
    // if (maxFreq < 0 || maxFreq > sampleRate / 2)
    //     return RPP_ERROR_INVALID_ARGUMENTS;
    // if (minFreq < 0 || minFreq > sampleRate / 2)
    //     return RPP_ERROR_INVALID_ARGUMENTS;

    if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        mel_filter_bank_host_tensor(static_cast<Rpp32f*>(srcPtr),
                                    srcDescPtr,
                                    static_cast<Rpp32f*>(dstPtr),
                                    dstDescPtr,
                                    srcDimsTensor,
                                    maxFreq,
                                    minFreq,
                                    melFormula,
                                    numFilter,
                                    sampleRate,
                                    normalize,
                                    rpp::deref(rppHandle));
        return RPP_SUCCESS;
    }
    else
    {
        return RPP_ERROR_NOT_IMPLEMENTED;
    }
}

/******************** resample ********************/

RppStatus rppt_resample_host(RppPtr_t srcPtr,
                             RpptDescPtr srcDescPtr,
                             RppPtr_t dstPtr,
                             RpptDescPtr dstDescPtr,
                             Rpp32f *inRateTensor,
                             Rpp32f *outRateTensor,
                             Rpp32s *srcDimsTensor,
                             RpptResamplingWindow &window,
                             rppHandle_t rppHandle)
{
    // Disabled this check for now.
    // This check will be re-enabled when the numDims based changes are added in MIVisionX */
    // Rpp32u tensorDims = srcDescPtr->numDims - 1; // exclude batchsize from input dims
    // if (tensorDims != 1 && tensorDims != 2)
    //     return RPP_ERROR_INVALID_SRC_DIMS;

    if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        resample_host_tensor(static_cast<Rpp32f*>(srcPtr),
                             srcDescPtr,
                             static_cast<Rpp32f*>(dstPtr),
                             dstDescPtr,
                             inRateTensor,
                             outRateTensor,
                             srcDimsTensor,
                             window,
                             rpp::deref(rppHandle));

        return RPP_SUCCESS;
    }
    else
    {
        return RPP_ERROR_NOT_IMPLEMENTED;
    }
}

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** non_silent_region_detection ********************/

RppStatus rppt_non_silent_region_detection_gpu(RppPtr_t srcPtr,
                                               RpptDescPtr srcDescPtr,
                                               Rpp32s *srcLengthTensor,
                                               Rpp32s *detectedIndexTensor,
                                               Rpp32s *detectionLengthTensor,
                                               Rpp32f cutOffDB,
                                               Rpp32s windowLength,
                                               Rpp32f referencePower,
                                               Rpp32s resetInterval,
                                               rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    Rpp32u tensorDims = srcDescPtr->numDims - 1; // exclude batchsize from input dims
    if (tensorDims != 1)
        return RPP_ERROR_INVALID_SRC_DIMS;

    if (srcDescPtr->dataType == RpptDataType::F32)
    {
        return hip_exec_non_silent_region_detection_tensor(static_cast<Rpp32f*>(srcPtr),
                                                           srcDescPtr,
                                                           srcLengthTensor,
                                                           detectedIndexTensor,
                                                           detectionLengthTensor,
                                                           cutOffDB,
                                                           windowLength,
                                                           referencePower,
                                                           resetInterval,
                                                           rpp::deref(rppHandle));
    }
    else
    {
        return RPP_ERROR_NOT_IMPLEMENTED;
    }

#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** to_decibels ********************/

RppStatus rppt_to_decibels_gpu(RppPtr_t srcPtr,
                               RpptDescPtr srcDescPtr,
                               RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr,
                               RpptImagePatchPtr srcDims,
                               Rpp32f cutOffDB,
                               Rpp32f multiplier,
                               Rpp32f referenceMagnitude,
                               rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    Rpp32u tensorDims = srcDescPtr->numDims - 1; // exclude batchsize from input dims
    if (tensorDims != 1 && tensorDims != 2)
        return RPP_ERROR_INVALID_SRC_DIMS;

    if (!multiplier)
        return RPP_ERROR_ZERO_DIVISION;

    if (srcDescPtr->dataType == RpptDataType::F32)
    {
        hip_exec_to_decibels_tensor(static_cast<Rpp32f*>(srcPtr),
                                    srcDescPtr,
                                    static_cast<Rpp32f*>(dstPtr),
                                    dstDescPtr,
                                    srcDims,
                                    cutOffDB,
                                    multiplier,
                                    referenceMagnitude,
                                    rpp::deref(rppHandle));
    }
    else
    {
        return RPP_ERROR_NOT_IMPLEMENTED;
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** down_mixing ********************/

RppStatus rppt_down_mixing_gpu(RppPtr_t srcPtr,
                               RpptDescPtr srcDescPtr,
                               RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr,
                               Rpp32s *srcDimsTensor,
                               bool  normalizeWeights,
                               rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    Rpp32u tensorDims = srcDescPtr->numDims - 1; // exclude batchsize from input dims
    if (tensorDims != 1 && tensorDims != 2)
        return RPP_ERROR_INVALID_SRC_DIMS;

    if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_down_mixing_tensor(static_cast<Rpp32f*>(srcPtr),
                                    srcDescPtr,
                                    static_cast<Rpp32f*>(dstPtr),
                                    dstDescPtr,
                                    srcDimsTensor,
                                    normalizeWeights,
                                    rpp::deref(rppHandle));
    }
    else
    {
        return RPP_ERROR_NOT_IMPLEMENTED;
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** pre_emphasis_filter ********************/

RppStatus rppt_pre_emphasis_filter_gpu(RppPtr_t srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       RppPtr_t dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32s *srcLengthTensor,
                                       Rpp32f *coeffTensor,
                                       RpptAudioBorderType borderType,
                                       rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcDescPtr->numDims != 2)
        return RPP_ERROR_INVALID_SRC_DIMS;

    if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        return hip_exec_pre_emphasis_filter_tensor(static_cast<Rpp32f*>(srcPtr),
                                                   srcDescPtr,
                                                   static_cast<Rpp32f*>(dstPtr),
                                                   dstDescPtr,
                                                   coeffTensor,
                                                   srcLengthTensor,
                                                   borderType,
                                                   rpp::deref(rppHandle));
    }
    else
    {
        return RPP_ERROR_NOT_IMPLEMENTED;
    }

#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** mel_filter_bank ********************/

RppStatus rppt_mel_filter_bank_gpu(RppPtr_t srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   RppPtr_t dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32s* srcDimsTensor,
                                   Rpp32f maxFreq,
                                   Rpp32f minFreq,
                                   RpptMelScaleFormula melFormula,
                                   Rpp32s numFilter,
                                   Rpp32f sampleRate,
                                   bool normalize,
                                   rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    Rpp32u tensorDims = srcDescPtr->numDims - 1; // exclude batchsize from input dims
    if (tensorDims != 2)
        return RPP_ERROR_INVALID_SRC_DIMS;
    if (srcDescPtr->layout != RpptLayout::NFT) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if (dstDescPtr->layout != RpptLayout::NFT) return RPP_ERROR_INVALID_DST_LAYOUT;
    if (maxFreq < 0 || maxFreq > sampleRate / 2)
        return RPP_ERROR_INVALID_ARGUMENTS;
    if (minFreq < 0 || minFreq > sampleRate / 2)
        return RPP_ERROR_INVALID_ARGUMENTS;

    if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        return hip_exec_mel_filter_bank_tensor(static_cast<Rpp32f*>(srcPtr),
                                               srcDescPtr,
                                               static_cast<Rpp32f*>(dstPtr),
                                               dstDescPtr,
                                               srcDimsTensor,
                                               maxFreq,
                                               minFreq,
                                               melFormula,
                                               numFilter,
                                               sampleRate,
                                               normalize,
                                               rpp::deref(rppHandle));
    }
    else
    {
        return RPP_ERROR_NOT_IMPLEMENTED;
    }

#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** resample ********************/

RppStatus rppt_resample_gpu(RppPtr_t srcPtr,
                            RpptDescPtr srcDescPtr,
                            RppPtr_t dstPtr,
                            RpptDescPtr dstDescPtr,
                            Rpp32f *inRateTensor,
                            Rpp32f *outRateTensor,
                            Rpp32s *srcDimsTensor,
                            RpptResamplingWindow &window,
                            rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    Rpp32u tensorDims = srcDescPtr->numDims - 1; // exclude batchsize from input dims
    if (tensorDims != 1 && tensorDims != 2)
        return RPP_ERROR_INVALID_SRC_DIMS;

    if (srcDescPtr->dataType == RpptDataType::F32)
    {
        hip_exec_resample_tensor(static_cast<Rpp32f*>(srcPtr),
                                 srcDescPtr,
                                 static_cast<Rpp32f*>(dstPtr),
                                 dstDescPtr,
                                 inRateTensor,
                                 outRateTensor,
                                 srcDimsTensor,
                                 window,
                                 rpp::deref(rppHandle));
        return RPP_SUCCESS;
    }
    else
    {
        return RPP_ERROR_NOT_IMPLEMENTED;
    }

#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** spectrogram ********************/

RppStatus rppt_spectrogram_gpu(RppPtr_t srcPtr,
                               RpptDescPtr srcDescPtr,
                               RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr,
                               Rpp32s *srcLengthTensor,
                               bool centerWindows,
                               bool reflectPadding,
                               Rpp32f *windowFunction,
                               Rpp32s nfft,
                               Rpp32s power,
                               Rpp32s windowLength,
                               Rpp32s windowStep,
                               rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((dstDescPtr->layout != RpptLayout::NFT) && (dstDescPtr->layout != RpptLayout::NTF))
        return RPP_ERROR_INVALID_DST_LAYOUT;
    Rpp32u srcTensorDims = srcDescPtr->numDims - 1; // exclude batchsize from input dims
    Rpp32u dstTensorDims = dstDescPtr->numDims - 1; // exclude batchsize from output dims
    if (srcTensorDims != 1)
        return RPP_ERROR_INVALID_SRC_DIMS;
    if (dstTensorDims != 2)
        return RPP_ERROR_INVALID_DST_DIMS;

    if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_spectrogram_tensor(static_cast<Rpp32f*>(srcPtr),
                                    srcDescPtr,
                                    static_cast<Rpp32f*>(dstPtr),
                                    dstDescPtr,
                                    srcLengthTensor,
                                    centerWindows,
                                    reflectPadding,
                                    windowFunction,
                                    nfft,
                                    power,
                                    windowLength,
                                    windowStep,
                                    rpp::deref(rppHandle));

        return RPP_SUCCESS;
    }
    else
    {
        return RPP_ERROR_NOT_IMPLEMENTED;
    }

#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

#endif // GPU_SUPPORT
#endif // AUDIO_SUPPORT
