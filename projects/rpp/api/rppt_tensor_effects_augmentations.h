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

#ifndef RPPT_TENSOR_EFFECTS_AUGMENTATIONS_H
#define RPPT_TENSOR_EFFECTS_AUGMENTATIONS_H

#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \file
 * \brief RPPT Tensor Operations - Effects Augmentations.
 * \defgroup group_rppt_tensor_effects_augmentations RPPT Tensor Operations - Effects Augmentations.
 * \brief RPPT Tensor Operations - Effects Augmentations.
 */

/*! \addtogroup group_rppt_tensor_effects_augmentations
 * @{
 */

/*! \brief Gridmask augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The gridmask augmentation runs as per https://arxiv.org/abs/2001.04086 for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_gridmask_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] tileWidth tileWidth value for gridmask calculation = width of black square + width of spacing until next black square on grid (a single Rpp32u number with tileWidth <= min(srcDescPtr->w, srcDescPtr->h) that applies to all images in the batch)
 * \param [in] gridRatio gridRatio value for gridmask calculation = black square width / tileWidth (a single Rpp32f number with 0 <= gridRatio <= 1 that applies to all images in the batch)
 * \param [in] gridAngle gridAngle value for gridmask calculation = grid rotation angle in radians (a single Rpp32f number that applies to all images in the batch)
 * \param [in] translateVector translateVector for gridmask calculation = grid X and Y translation lengths in pixels (a single RpptUintVector2D x,y value pair that applies to all images in the batch)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_gridmask(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u tileWidth, Rpp32f gridRatio, Rpp32f gridAngle, RpptUintVector2D translateVector, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Spatter augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The spatter augmentation adds random spatter of a user-defined color, for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_spatter_typeMud_img150x150.png Sample Output - Mud Spatter
 * \image html effects_augmentations_spatter_typeInk_img150x150.png Sample Output - Ink Spatter
 * \image html effects_augmentations_spatter_typeBlood_img150x150.png Sample Output - Blood Spatter
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] spatterColor RGB values to use for the spatter augmentation (A single set of 3 Rpp8u values as RpptRGB that applies to all images in the batch)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y)) | (Restrictions - roiTensorPtrSrc[i].xywhROI.roiWidth <= 1920 and roiTensorPtrSrc[i].xywhROI.roiHeight <= 1080)
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_spatter(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptRGB spatterColor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Salt and pepper noise augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The salt and pepper noise augmentation adds SnP noise based on user defined noise/salt probabilities, and user defined salt/pepper values for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_salt_and_pepper_noise_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] noiseProbabilityTensor noiseProbability values to decide if a destination pixel is a noise-pixel, or equal to source (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize with 0 <= noiseProbabilityTensor[i] <= 1 for each image in batch)
 * \param [in] saltProbabilityTensor saltProbability values to decide if a given destination noise-pixel is salt or pepper (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize with 0 <= saltProbabilityTensor[i] <= 1 for each image in batch)
 * \param [in] saltValueTensor A user-defined salt noise value (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize with 0 <= saltValueTensor[i] <= 1 for each image in batch)
 * \param [in] pepperValueTensor A user-defined pepper noise value (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize with 0 <= pepperValueTensor[i] <= 1 for each image in batch)
 * \param [in] seed A user-defined seed value (single Rpp32u value)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_salt_and_pepper_noise(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *noiseProbabilityTensor, Rpp32f *saltProbabilityTensor, Rpp32f *saltValueTensor, Rpp32f *pepperValueTensor, Rpp32u seed, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Shot noise augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The shot noise augmentation adds Poisson/shot noise based on a user defined shotNoiseFactor, for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_shot_noise_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] shotNoiseFactorTensor shotNoiseFactor values for each image, which are used to compute the lambda values in a poisson distribution (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize with shotNoiseFactorTensor[i] >= 0 for each image in batch)
 * \param [in] seed A user-defined seed value (single Rpp32u value)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_shot_noise(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *shotNoiseFactorTensor, Rpp32u seed, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Gaussian noise augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The gaussian noise augmentation adds Gaussian noise based on user defined means and standard deviations, for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_gaussian_noise_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] meanTensor mean values for each image, which are used to compute the generalized Box-Mueller transforms in a gaussian distribution (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize with meanTensor[i] >= 0 for each image in batch)
 * \param [in] stdDevTensor stdDev values for each image, which are used to compute the generalized Box-Mueller transforms in a gaussian distribution (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize with stdDevTensor[i] >= 0 for each image in batch)
 * \param [in] seed A user-defined seed value (single Rpp32u value)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_gaussian_noise(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *meanTensor, Rpp32f *stdDevTensor, Rpp32u seed, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Non linear blend augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The non linear blend augmentation adds standard deviation based non-linear alpha-blending, between two sets of batches of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input1
 * \image html img150x150_2.png Sample Input2
 * \image html effects_augmentations_non_linear_blend_img150x150.png Sample Output
 * \param [in] srcPtr1 source1 tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcPtr2 source2 tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] stdDevTensor stdDev values for each image, which are used to compute the generalized Box-Mueller transforms in a gaussian distribution (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize with stdDevTensor[i] >= 0 for each image in batch)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_non_linear_blend(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *stdDevTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Water augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The water augmentation adds a water effect for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_water_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param[in] amplitudeXTensor amplitudeX values for water effect (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize)
 * \param[in] amplitudeYTensor amplitudeY values for water effect (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize)
 * \param[in] frequencyXTensor frequencyX values for water effect (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize)
 * \param[in] frequencyYTensor frequencyY values for water effect (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize)
 * \param[in] phaseXTensor phaseX values for water effect (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize)
 * \param[in] phaseYTensor phaseY values for water effect (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_water(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *amplitudeXTensor, Rpp32f *amplitudeYTensor, Rpp32f *frequencyXTensor, Rpp32f *frequencyYTensor, Rpp32f *phaseXTensor, Rpp32f *phaseYTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief RICAP (Random Image Crop And Patch) augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The RICAP (Random Image Crop And Patch) augmentation runs as per https://arxiv.org/abs/1811.09030 for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * The RICAP augmentation requires dimensions of input images to be the same across entire batch.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input1
 * \image html img150x150_1.png Sample Input2
 * \image html img150x150_2.png Sample Input3
 * \image html effects_augmentations_ricap_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param[in] permutationTensor Array of batchSize permutation sets (2D tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend), of batchSize * 4. Each set of 4 permutations contains Rpp32u image indices for each region in the respective RICAP-output-image in the batch)
 * \param[in] roiPtrInputCropRegion Array of 4 ROIs (2D tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend), of size 4 * 4-elements per ROI, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_ricap(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u *permutationTensor, RpptROIPtr roiPtrInputCropRegion, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Vignette augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The Vignette augmentation adds a vignette effect for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_vignette_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param[in] vignetteIntensityTensor intensity values to quantify vignette effect (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize with 0 < vignetteIntensityTensor[n] for each image in batch)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
// NOTE: Pixel mismatch of 5% is expected between HIP and HOST Tensor variations due to usage of fastexpavx() instead of exp() in HOST Tensor.
RppStatus rppt_vignette(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *vignetteIntensityTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Jitter augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The jitter augmentation adds a jitter effect for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_jitter_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] kernelSizeTensor kernelsize value for jitter calculation (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), kernelSize = 3/5/7 for optimal use)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_jitter(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u *kernelSizeTensor, Rpp32u seed, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief  Gaussian noise augmentation on HIP/HOST backend
 * \details This function adds gaussian noise to a batch of 4D tensors.
 *          Support added for u8 -> u8, f32 -> f32 datatypes.
 * \image html https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenInputs/input150x150x4.gif Sample Input
 * \image html https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenOutputs/effects_augmentations_gaussian_noise_150x150x4.gif Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcGenericDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstGenericDescPtr destination tensor descriptor
 * \param [in] meanTensor mean values for each input, which are used to compute the generalized Box-Mueller transforms in a gaussian distribution (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize with meanTensor[i] >= 0 for each image in batch)
 * \param [in] stdDevTensor stdDev values for each image, which are used to compute the generalized Box-Mueller transforms in a gaussian distribution (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize with stdDevTensor[i] >= 0 for each image in batch)
 * \param [in] seed A user-defined seed value (single Rpp32u value)
 * \param [in] roiGenericPtrSrc ROI data for each image in source tensor (tensor of batchSize RpptRoiGeneric values)
 * \param [in] roiType ROI type used (RpptRoi3DType::XYZWHD or RpptRoi3DType::LTFRBB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_gaussian_noise_voxel(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32f *meanTensor, Rpp32f *stdDevTensor, Rpp32u seed, RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType, rppHandle_t rppHandle,  RppBackend executionBackend);

/*! \brief Erase augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details This function erases one or more user defined regions from an image, for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 *          srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 *          dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_erase_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] anchorBoxInfoTensor anchorBoxInfo values of type RpptRoiLtrb for each erase-region inside each image in the batch (tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)). Restrictions -
            - 0 <= anchorBoxInfo[i] < respective image width/height
            - Erase-region anchor boxes on each image given by the user must not overlap
 * \param [in] colorsTensor RGB values to use for each erase-region inside each image in the batch (tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)). (colors[i] will have range equivalent of srcPtr)
 * \param [in] numBoxesTensor number of erase-regions per image, for each image in the batch (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend)). (numBoxesTensor[n] >= 0)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_erase(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptRoiLtrb *anchorBoxInfoTensor, RppPtr_t colorsTensor, Rpp32u *numBoxesTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Glitch augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The glitch augmentation adds a glitch effect for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_glitch_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] rgbOffsets RGB offset values to use for the glitch augmentation (A single set of 3 Rppi point values that applies to all images in the batch.
 *                        For each point and for each image in the batch: 0 < point.x < width, 0 < point.y < height)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend) for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_glitch(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptChannelOffsets *rgbOffsets, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Rain augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The rain augmentation simulates a rain effect for a batch of RGB (3-channel) / greyscale (1-channel) images with an NHWC/NCHW tensor layout.<br>
 * <b> NOTE: This augmentation gives a more realistic Rain output when all images in a batch are of similar / same sizes </b> <br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be the same depth as srcPtr.
 * \image html img640x480.png Sample Input
 * \image html effects_augmentations_rain_img640x480.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] rainPercentage The percentage of the rain effect to be applied (0 <= rainPercentage <= 100)
 * \param [in] rainWidth Width of the rain drops in pixels. To be tuned by user depending on size of the image.
 * \param [in] rainHeight Height of the rain drops in pixels. To be tuned by user depending on size of the image.
 * \param [in] slantAngle Slant angle of the rain drops (positive value for right slant, negative for left slant). A single Rpp32s/f representing the slant of raindrops in degrees.
 *                        Values range from [-90, 90], where -90 represents extreme left slant, 0 is vertical, and 90 is extreme right slant.
 * \param [in] alpha An array of alpha blending values to be used for blending the rainLayer and the input image for each image in the batch (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), 0 ≤ alpha ≤ 1 for each image in the batch).
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend) for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreateWithBatchSize()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_rain(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f rainPercentage, Rpp32u rainWidth, Rpp32u rainHeight, Rpp32f slantAngle, Rpp32f *alpha, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Pixelate augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The pixelate augmentation performs a pixelate transformation for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_pixelate_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] intermediateScratchBufferPtr intermediate scratch buffer in HIP memory (for HIP backend) or HOST memory (for HOST backend) (Minimum size = srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(Rpp32f))
 * \param [in] pixelationPercentage 'pixelationPercentage' variable controls how much pixelation is applied to images.(pixelationPercentage value ranges from 0 to 100)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_pixelate(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RppPtr_t intermediateScratchBufferPtr, Rpp32f pixelationPercentage, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Fog augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The fog augmentation adds a fog effect for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * <b> NOTE: This augmentation gives a more realistic fog output when all images in a batch are of similar / same sizes </b> <br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img640x480.png Sample Input
 * \image html effects_augmentations_fog_img640x480.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] intensityFactor intensity factor values for fog calculation (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize, with 0 <= intensityFactor <= 0.5 for each image in batch)
 * \param [in] greyFactor gray factor values to introduce grayness in the image for fog calculation (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize, with 0 <= greyFactor <= 1 for each image in batch)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend) for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST/HIP handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_fog(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *intensityFactor, Rpp32f *greyFactor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Posterize augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The posterize augmentation adds a posterize effect for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_posterize_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] posterizeLevelBits number of bits used to represent the image with posterize Operation (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize, with 1 <= posterizeLevelBits <= 8 for each image in batch))
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend) for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_posterize(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp8u *posterizeLevelBits, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Solarize augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The solarize augmentation inverts pixel values above a specified threshold for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_solarize_img150x150.jpg Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] thresholdTensor threshold values for solarize effect (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize, with 0 <= threshold <= 1 for each image in batch)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend) for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_solarize(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f* thresholdTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Snow augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The Snow augmentation adds a snowed-in effect on a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_snow_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] brightnessCoefficient brightness modification parameter for snow calculation (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize with brightnessCoefficient[i] in the range (1, 4] for each image in batch)
 * \param [in] snowThreshold threshold parameter for snow calculation (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize with 0 < snowThreshold[i] <= 1 for each image in batch)
 * \param [in] darkMode darkMode values to set dark mode on/off (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize, with darkMode[i] = 0/1)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_snow(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *brightnessCoefficient, Rpp32f *snowThreshold, Rpp32s *darkMode, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Channel dropout augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The channel dropout augmentation function erases one or more user defined channel from an image, for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_channel_dropout_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] dropoutTensor channel dropout tensor (1D Rpp8u tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize * channels Values must be 0 (Drop) or 1 (Keep) for each channel of each image in batch)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_channel_dropout(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp8u *dropoutTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Cutout dropout augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details Cutout dropout function erases random regions from an image, for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_cutout_dropout_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] anchorBoxInfoTensor precomputed cutout erase regions for the batch in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), stored as a flat array of RpptRoiLtrb of size (batchSize * maxBoxesPerImage), where maxBoxesPerImage = max(numBoxesTensor[n])
 * \param [in] colorsTensor pointer to erase color values for each erase region in HIP memory (for HIP backend) or HOST memory (for HOST backend), laid out identically to anchorBoxInfoTensor, i.e., of size (batchSize * maxBoxesPerImage), with colorsTensor[(n * maxBoxesPerImage) + k]
 * \param [in] numBoxesTensor number of erase regions per image in the batch in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend) (1D tensor of size batchSize, Data Type - Rpp32u*)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_cutout_dropout(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptRoiLtrb *anchorBoxInfoTensor, RppPtr_t colorsTensor, Rpp32u *numBoxesTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Grid dropout augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details Grid dropout function erases grid wise random regions from an image, for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_grid_dropout_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] anchorBoxInfoTensor Precomputed grid erase regions for the batch in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), stored as an array of RpptRoiLtrb of size (batchSize * boxesInEachImage)
 * \param [in] boxesInEachImage Number of grid boxes per image (Data Type - Rpp32u)
 * \param [in] maxHoleW Maximum hole width across all grid boxes (Data Type - Rpp32u)
 * \param [in] maxHoleH Maximum hole height across all grid boxes (Data Type - Rpp32u)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_grid_dropout(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptRoiLtrb *anchorBoxInfoTensor, Rpp32u boxesInEachImage, Rpp32u maxHoleW, Rpp32u maxHoleH, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Random Erase augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details This function erases random regions from an image and fills with random noise, for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 *          srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 *          dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_random_erase_dropout_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] anchorBoxInfoTensor anchorBoxInfo values of type RpptRoiLtrb for each erase-region inside each image in the batch (tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend)). Restrictions -
            - 0 <= anchorBoxInfo[i] < respective image width/height
            - Erase-region anchor boxes on each image given by the user must not overlap
 * \param [in] noiseBuffer pre-allocated buffer containing random noise values in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend) (Buffer size must be 255 * 255 * srcDescPtr->c. Values are accessed spatially (tiled) to fill erased regions)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_random_erase(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptRoiLtrb *anchorBoxInfoTensor, RppPtr_t noiseBuffer, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Coarse dropout augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details This function erases one or more user defined regions from an image, for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 *          srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 *          dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_coarse_dropout_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] anchorBoxInfoTensor anchorBoxInfo values of type RpptRoiLtrb for each erase-region inside each image in the batch (tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend)). Restrictions -
            - 0 <= anchorBoxInfo[i] < respective image width/height
            - Erase-region anchor boxes on each image given by the user must not overlap
 * \param [in] numBoxesTensor number of erase-regions per image, for each image in the batch (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend)). (numBoxesTensor[n] >= 0)
 * \param [in] maxBoxesPerImage Maximum number of erase-regions allocated per image in the batch (stride for anchorBoxInfoTensor/colorsTensor)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_coarse_dropout(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptRoiLtrb *anchorBoxInfoTensor, Rpp32u *numBoxesTensor, Rpp32u maxBoxesPerImage, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! @}
 */

#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_EFFECTS_AUGMENTATIONS_H
