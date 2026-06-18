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

#ifndef RPPT_TENSOR_AUDIO_AUGMENTATIONS_H
#define RPPT_TENSOR_AUDIO_AUGMENTATIONS_H

#ifdef AUDIO_SUPPORT

#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \file
 * \brief RPPT Tensor Operations - Audio Augmentations.
 * \defgroup group_rppt_tensor_audio_augmentations RPPT Tensor Operations - Audio Augmentations.
 * \brief RPPT Tensor Operations - Audio Augmentations.
 */

/*! \addtogroup group_rppt_tensor_audio_augmentations
 * @{
 */

/*! \brief Non Silent Region Detection augmentation on HIP/HOST backend
 * \details Non Silent Region Detection augmentation for 1D audio buffer
            \n Finds the starting index and length of non silent region in the audio buffer by
 comparing the calculated short-term power with cutoff value passed
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 2, offsetInBytes >= 0,
 dataType = F32)
 * \param [in] srcLengthTensor source audio buffer length (1D tensor in HIP memory (for HIP backend)
 or HOST memory (for HOST backend), of size batchSize)
 * \param [out] detectedIndexTensor beginning index of non silent region (1D tensor in pinned memory
 (for HIP backend) or HOST memory (for HOST backend), of size batchSize)
 * \param [out] detectionLengthTensor length of non silent region  (1D tensor in pinned memory (for
 HIP backend) or HOST memory (for HOST backend), of size batchSize)
 * \param [in] cutOffDB cutOff in dB below which the signal is considered silent
 * \param [in] windowLength window length used for computing short-term power of the signal
 * \param [in] referencePower reference power that is used to convert the signal to dB
 * \param [in] resetInterval number of samples after which the moving mean average is recalculated
 to avoid precision loss
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_non_silent_region_detection(RppPtr_t srcPtr, RpptDescPtr srcDescPtr,
                                           Rpp32s* srcLengthTensor, Rpp32s* detectedIndexTensor,
                                           Rpp32s* detectionLengthTensor, Rpp32f cutOffDB,
                                           Rpp32s windowLength, Rpp32f referencePower,
                                           Rpp32s resetInterval, rppHandle_t rppHandle,
                                           RppBackend executionBackend);

/*! \brief To Decibels augmentation on HIP/HOST backend
 * \details To Decibels augmentation for 1D/2D audio buffer converts magnitude values to decibel
 * values \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 2 or 3 (for
 * single-channel or multi-channel/2D audio tensor with 1 channel), offsetInBytes >= 0, dataType =
 * F32) \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for
 * HOST backend) \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 2 or
 * 3 (for single-channel or multi-channel/2D audio tensor with 1 channel), offsetInBytes >= 0,
 * dataType = F32) \param [in] srcDims source tensor sizes for each element in batch (2D tensor in
 * pinned memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize * 2) \param
 * [in] cutOffDB  minimum or cut-off ratio in dB \param [in] multiplier factor by which the
 * logarithm is multiplied \param [in] referenceMagnitude Reference magnitude if not provided
 * maximum value of input used as reference \param [in] rppHandle RPP HOST handle created with
 * <tt>\ref rppCreate()</tt> \return A <tt> \ref RppStatus</tt> enumeration. \retval RPP_SUCCESS
 * Successful completion. \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_to_decibels(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                           RpptDescPtr dstDescPtr, RpptImagePatchPtr srcDims, Rpp32f cutOffDB,
                           Rpp32f multiplier, Rpp32f referenceMagnitude, rppHandle_t rppHandle,
                           RppBackend executionBackend);

/*! \brief Pre Emphasis Filter augmentation on HIP/HOST backend
 * \details Pre Emphasis Filter augmentation for audio data
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 3,
 * offsetInBytes >= 0, dataType = F32) \param [out] dstPtr destination tensor in HIP memory (for HIP
 * backend) or HOST memory (for HOST backend) \param [in] dstDescPtr destination tensor descriptor
 * (Restrictions - numDims = 3, offsetInBytes >= 0, dataType = F32) \param [in] srcLengthTensor
 * source audio buffer length (1D tensor in pinned memory (for HIP backend) or HOST memory (for HOST
 * backend), of size batchSize) \param [in] coeffTensor preemphasis coefficient (1D tensor in pinned
 * memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize) \param [in]
 * borderType border value policy \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref
 * rppCreate()</tt> \return A <tt> \ref RppStatus</tt> enumeration. \retval RPP_SUCCESS Successful
 * completion. \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_pre_emphasis_filter(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                                   RpptDescPtr dstDescPtr, Rpp32s* srcLengthTensor,
                                   Rpp32f* coeffTensor, RpptAudioBorderType borderType,
                                   rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Down Mixing augmentation on HIP/HOST backend
 * \details Down Mixing augmentation for audio data
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 2 or 3 (for
 * single-channel or multi-channel audio tensor), offsetInBytes >= 0, dataType = F32) \param [out]
 * dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 2, offsetInBytes
 * >= 0, dataType = F32) \param [in] srcDimsTensor source audio buffer length and number of channels
 * (1D tensor in pinned memory (for HIP backend) or HOST memory (for HOST backend), of size
 * batchSize * 2) \param [in] normalizeWeights bool flag to specify if normalization of weights is
 * needed \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt> \return A
 * <tt> \ref RppStatus</tt> enumeration. \retval RPP_SUCCESS Successful completion. \retval
 * RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_down_mixing(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                           RpptDescPtr dstDescPtr, Rpp32s* srcDimsTensor, bool normalizeWeights,
                           rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Produces a spectrogram from a 1D audio buffer on HIP/HOST backend
 * \details Spectrogram for 1D audio buffer
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 2,
 * offsetInBytes >= 0, dataType = F32) \param [out] dstPtr destination tensor in HIP memory (for HIP
 * backend) or HOST memory (for HOST backend) \param [in] dstDescPtr destination tensor descriptor
 * (Restrictions - numDims = 3, offsetInBytes >= 0, dataType = F32, layout - NFT / NTF) \param [in]
 * srcLengthTensor source audio buffer length (1D tensor in pinned memory (for HIP backend) or HOST
 * memory (for HOST backend), of size batchSize) \param [in] centerWindows indicates whether
 * extracted windows should be padded so that the window function is centered at multiples of
 * window_step \param [in] reflectPadding indicates the padding policy when sampling outside the
 * bounds of the signal \param [in] windowFunction samples of the window function that will be
 * multiplied to each extracted window when calculating the Short Time Fourier Transform (STFT).<br>
 * if windowFunction is a nullptr, then required windowFunction values will be generated inside the
 * kernel \param [in] nfft size of the FFT \param [in] power exponent of the magnitude of the
 * spectrum \param [in] windowLength window size in number of samples \param [in] windowStep step
 * between the STFT windows in number of samples \param [in] rppHandle RPP HIP/HOST handle created
 * with <tt>\ref rppCreate()</tt> \return A <tt> \ref RppStatus</tt> enumeration. \retval
 * RPP_SUCCESS Successful completion. \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_spectrogram(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                           RpptDescPtr dstDescPtr, Rpp32s* srcLengthTensor, bool centerWindows,
                           bool reflectPadding, Rpp32f* windowFunction, Rpp32s nfft, Rpp32s power,
                           Rpp32s windowLength, Rpp32s windowStep, rppHandle_t rppHandle,
                           RppBackend executionBackend);

/*! \brief Mel filter bank augmentation HIP/HOST backend
 * \details Mel filter bank augmentation for audio data
 * \param[in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param[in] srcDescPtr source tensor descriptor (Restrictions - numDims = 3, offsetInBytes >= 0,
 * dataType = F32, layout - NFT) \param[out] dstPtr destination tensor in HIP memory (for HIP
 * backend) or HOST memory (for HOST backend) \param[in] dstDescPtr destination tensor descriptor
 * (Restrictions - numDims = 3, offsetInBytes >= 0, dataType = F32, layout - NFT) \param[in]
 * srcDimsTensor source audio buffer length and number of channels (1D tensor in pinned memory (for
 * HIP backend) or HOST memory (for HOST backend), of size batchSize * 2) \param[in] maxFreq maximum
 * frequency if not provided maxFreq = sampleRate / 2 \param[in] minFreq minimum frequency
 * \param[in] melFormula formula used to convert frequencies from hertz to mel and from mel to hertz
 * (SLANEY / HTK) \param[in] numFilter number of mel filters \param[in] sampleRate sampling rate of
 * the audio \param[in] normalize boolean variable that determine whether to normalize weights / not
 * \param[in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_mel_filter_bank(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr, Rpp32s* srcDims, Rpp32f maxFreq,
                               Rpp32f minFreq, RpptMelScaleFormula melFormula, Rpp32s numFilter,
                               Rpp32f sampleRate, bool normalize, rppHandle_t rppHandle,
                               RppBackend executionBackend);

/*! \brief Resample augmentation on HIP/HOST backend
 * \details Resample augmentation for audio data
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 2 or 3 (for
 * single-channel or multi-channel audio tensor), offsetInBytes >= 0, dataType = F32) \param [out]
 * dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 2 or 3 (for
 * single-channel or multi-channel audio tensor), offsetInBytes >= 0, dataType = F32) \param [in]
 * inRate Input sampling rate (1D tensor in pinned memory (for HIP backend) or HOST memory (for HOST
 * backend), of size batchSize) \param [in] outRate Output sampling rate (1D tensor in pinned memory
 * (for HIP backend) or HOST memory (for HOST backend), of size batchSize) \param [in] srcDimsTensor
 * source audio buffer length and number of channels (1D tensor in pinned memory (for HIP backend)
 * or HOST memory (for HOST backend), of size batchSize * 2) \param [in] window Resampling window
 * (struct of type RpptRpptResamplingWindow) \param [in] rppHandle RPP HOST handle created with
 * <tt>\ref rppCreate()</tt> \return A <tt> \ref RppStatus</tt> enumeration. \retval RPP_SUCCESS
 * Successful completion. \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_resample(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                        RpptDescPtr dstDescPtr, Rpp32f* inRateTensor, Rpp32f* outRateTensor,
                        Rpp32s* srcDimsTensor, RpptResamplingWindow& window, rppHandle_t rppHandle,
                        RppBackend executionBackend);

/*! \brief Audio Tensor Add Tensor augmentation on HIP/HOST backend
 * \details Audio Tensor Add Tensor augmentation for audio data. Performs scalar broadcasting
 * addition where srcPtr2 contains one scalar value per batch (shape: batchSize) that is broadcasted
 * and added to all elements in the corresponding batch of srcPtr1. \param [in] srcPtr1 first source
 * tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend) \param [in] srcPtr2
 * second source tensor containing one scalar per batch in HIP memory (for HIP backend) or HOST
 * memory (for HOST backend), of size batchSize \param [in] srcDescPtr source tensor descriptor
 * (Restrictions - numDims = 2 or 3 (for single-channel or multi-channel audio tensor),
 * offsetInBytes >= 0, dataType = F32) \param [out] dstPtr destination tensor in HIP memory (for HIP
 * backend) or HOST memory (for HOST backend) \param [in] dstDescPtr destination tensor descriptor
 * (Restrictions - numDims = 2 or 3 (for single-channel or multi-channel audio tensor),
 * offsetInBytes >= 0, dataType = F32) \param [in] srcLengthTensor source audio buffer length (1D
 * tensor in pinned memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend RPP execution backend (HOST or HIP)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_audio_tensor_add_tensor(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptDescPtr srcDescPtr,
                                       RppPtr_t dstPtr, RpptDescPtr dstDescPtr,
                                       Rpp32s* srcLengthTensor, rppHandle_t rppHandle,
                                       RppBackend executionBackend);

/*! \brief Audio Tensor Multiply Scalar augmentation on HIP/HOST backend
 * \details Audio Tensor Multiply Scalar augmentation for audio data. Multiplies each element in
 * srcPtr by a single scalar value. The scalar value is broadcasted and multiplied to all elements
 * across the entire tensor. \param [in] srcPtr source tensor in HIP memory (for HIP backend) or
 * HOST memory (for HOST backend) \param [in] scalarValue scalar value to multiply with all elements
 * in the tensor \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 2 or 3
 * (for single-channel or multi-channel audio tensor), offsetInBytes >= 0, dataType = F32) \param
 * [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 2 or 3 (for
 * single-channel or multi-channel audio tensor), offsetInBytes >= 0, dataType = F32) \param [in]
 * srcLengthTensor source audio buffer length (1D tensor in pinned memory (for HIP backend) or HOST
 * memory (for HOST backend), of size batchSize) \param [in] rppHandle RPP HIP/HOST handle created
 * with <tt>\ref rppCreate()</tt> \param [in] executionBackend RPP execution backend (HOST or HIP)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_audio_tensor_mul_scalar(RppPtr_t srcPtr, Rpp32f scalarValue, RpptDescPtr srcDescPtr,
                                       RppPtr_t dstPtr, RpptDescPtr dstDescPtr,
                                       Rpp32s* srcLengthTensor, rppHandle_t rppHandle,
                                       RppBackend executionBackend);

/*! @}
 */

#ifdef __cplusplus
}
#endif

#endif  // AUDIO_SUPPORT

#endif  // RPPT_TENSOR_AUDIO_AUGMENTATIONS_H
