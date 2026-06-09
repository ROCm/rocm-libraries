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
#include "rppt_validate.hpp"
#include "rppt_tensor_audio_augmentations.h"
#include "host_tensor_executors.hpp"

#ifdef GPU_SUPPORT
#include "hip_tensor_executors.hpp"
#endif // GPU_SUPPORT

/******************** non_silent_region_detection ********************/

RppStatus rppt_non_silent_region_detection(RppPtr_t srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp32s *srcLengthTensor,
                                           Rpp32s *detectedIndexTensor,
                                           Rpp32s *detectionLengthTensor,
                                           Rpp32f cutOffDB,
                                           Rpp32s windowLength,
                                           Rpp32f referencePower,
                                           Rpp32s resetInterval,
                                           rppHandle_t rppHandle,
                                           RppBackend executionBackend)
{
    // Disabled this check for now.
    // This check will be re-enabled when the numDims based changes are added in MIVisionX */
    // Rpp32u tensorDims = srcDescPtr->numDims - 1; // exclude batchsize from input dims
    // if (tensorDims != 1)
    //     return RPP_ERROR_INVALID_SRC_DIMS;

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
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
                                                    handle);
        }
        else
        {
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
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
                                                               handle);
        }
        else
        {
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** to_decibels ********************/

RppStatus rppt_to_decibels(RppPtr_t srcPtr,
                           RpptDescPtr srcDescPtr,
                           RppPtr_t dstPtr,
                           RpptDescPtr dstDescPtr,
                           RpptImagePatchPtr srcDims,
                           Rpp32f cutOffDB,
                           Rpp32f multiplier,
                           Rpp32f referenceMagnitude,
                           rppHandle_t rppHandle,
                           RppBackend executionBackend)
{
    if (!multiplier)
        return RPP_ERROR_ZERO_DIVISION;

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        // Disabled this check for now.
        // This check will be re-enabled when the numDims based changes are added in MIVisionX */
        // Rpp32u tensorDims = srcDescPtr->numDims - 1; // exclude batchsize from input dims
        // if (tensorDims != 1 && tensorDims != 2)
        //     return RPP_ERROR_INVALID_SRC_DIMS;

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
                                    handle);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        Rpp32u tensorDims = srcDescPtr->numDims - 1; // exclude batchsize from input dims
        if (tensorDims != 1 && tensorDims != 2)
            return RPP_ERROR_INVALID_SRC_DIMS;

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
                                        handle);
        }
        else
        {
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** pre_emphasis_filter ********************/

RppStatus rppt_pre_emphasis_filter(RppPtr_t srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   RppPtr_t dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32s *srcLengthTensor,
                                   Rpp32f *coeffTensor,
                                   RpptAudioBorderType borderType,
                                   rppHandle_t rppHandle,
                                   RppBackend executionBackend)
{
    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
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
                                            handle);
        }
        else
        {
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
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
                                                       handle);
        }
        else
        {
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** down_mixing ********************/

RppStatus rppt_down_mixing(RppPtr_t srcPtr,
                           RpptDescPtr srcDescPtr,
                           RppPtr_t dstPtr,
                           RpptDescPtr dstDescPtr,
                           Rpp32s *srcDimsTensor,
                           bool normalizeWeights,
                           rppHandle_t rppHandle,
                           RppBackend executionBackend)
{
    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
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
                                    handle);
        }
        else
        {
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
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
                                        handle);
        }
        else
        {
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** spectrogram ********************/

RppStatus rppt_spectrogram(RppPtr_t srcPtr,
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
                           rppHandle_t rppHandle,
                           RppBackend executionBackend)
{
    if ((dstDescPtr->layout != RpptLayout::NFT) && (dstDescPtr->layout != RpptLayout::NTF))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
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
                                    handle);
        }
        else
        {
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
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
                                        handle);
        }
        else
        {
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** mel_filter_bank ********************/

RppStatus rppt_mel_filter_bank(RppPtr_t srcPtr,
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
                               rppHandle_t rppHandle,
                               RppBackend executionBackend)
{
    if (srcDescPtr->layout != RpptLayout::NFT) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if (dstDescPtr->layout != RpptLayout::NFT) return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
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
                                        handle);
        }
        else
        {
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        Rpp32u tensorDims = srcDescPtr->numDims - 1; // exclude batchsize from input dims
        if (tensorDims != 2)
            return RPP_ERROR_INVALID_SRC_DIMS;
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
                                                   handle);
        }
        else
        {
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** resample ********************/

RppStatus rppt_resample(RppPtr_t srcPtr,
                        RpptDescPtr srcDescPtr,
                        RppPtr_t dstPtr,
                        RpptDescPtr dstDescPtr,
                        Rpp32f *inRateTensor,
                        Rpp32f *outRateTensor,
                        Rpp32s *srcDimsTensor,
                        RpptResamplingWindow &window,
                        rppHandle_t rppHandle,
                        RppBackend executionBackend)
{
    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
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
                                 handle);
        }
        else
        {
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
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
                                     handle);
        }
        else
        {
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** audio_tensor_add_tensor ********************/

RppStatus rppt_audio_tensor_add_tensor(RppPtr_t srcPtr1,
                                       RppPtr_t srcPtr2,
                                       RpptDescPtr srcDescPtr,
                                       RppPtr_t dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32s *srcLengthTensor,
                                       rppHandle_t rppHandle,
                                       RppBackend executionBackend)
{
    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            audio_tensor_add_tensor_host(static_cast<Rpp32f*>(srcPtr1),
                                         static_cast<Rpp32f*>(srcPtr2),
                                         srcDescPtr,
                                         static_cast<Rpp32f*>(dstPtr),
                                         dstDescPtr,
                                         srcLengthTensor,
                                         handle);
        }
        else
        {
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            hip_exec_audio_tensor_add_tensor(static_cast<Rpp32f*>(srcPtr1),
                                             static_cast<Rpp32f*>(srcPtr2),
                                             srcDescPtr,
                                             static_cast<Rpp32f*>(dstPtr),
                                             dstDescPtr,
                                             srcLengthTensor,
                                             handle);
        }
        else
        {
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** audio_tensor_mul_scalar ********************/

RppStatus rppt_audio_tensor_mul_scalar(RppPtr_t srcPtr,
                                       Rpp32f scalarValue,
                                       RpptDescPtr srcDescPtr,
                                       RppPtr_t dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32s *srcLengthTensor,
                                       rppHandle_t rppHandle,
                                       RppBackend executionBackend)
{
    rpp::Handle &handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if(executionBackend == RppBackend::RPP_HOST_BACKEND)
    {
        if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            audio_tensor_mul_scalar_host(static_cast<Rpp32f*>(srcPtr),
                                         scalarValue,
                                         srcDescPtr,
                                         static_cast<Rpp32f*>(dstPtr),
                                         dstDescPtr,
                                         srcLengthTensor,
                                         handle);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if((handleBackend == RppBackend::RPP_HIP_BACKEND) && (executionBackend == RppBackend::RPP_HIP_BACKEND))
    {
        if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            hip_exec_audio_tensor_mul_scalar(static_cast<Rpp32f*>(srcPtr),
                                             scalarValue,
                                             srcDescPtr,
                                             static_cast<Rpp32f*>(dstPtr),
                                             dstDescPtr,
                                             srcLengthTensor,
                                             handle);
        }
        else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

#endif // AUDIO_SUPPORT
