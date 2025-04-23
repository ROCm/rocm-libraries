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

/*! \brief Gridmask augmentation on HOST backend for a NCHW/NHWC layout tensor
 * \details The gridmask augmentation runs as per https://arxiv.org/abs/2001.04086 for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_gridmask_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HOST memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] tileWidth tileWidth value for gridmask calculation = width of black square + width of spacing until next black square on grid (a single Rpp32u number with tileWidth <= min(srcDescPtr->w, srcDescPtr->h) that applies to all images in the batch)
 * \param [in] gridRatio gridRatio value for gridmask calculation = black square width / tileWidth (a single Rpp32f number with 0 <= gridRatio <= 1 that applies to all images in the batch)
 * \param [in] gridAngle gridAngle value for gridmask calculation = grid rotation angle in radians (a single Rpp32f number that applies to all images in the batch)
 * \param [in] translateVector translateVector for gridmask calculation = grid X and Y translation lengths in pixels (a single RpptUintVector2D x,y value pair that applies to all images in the batch)
 * \param [in] roiTensorPtrSrc ROI data in HOST memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_gridmask_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u tileWidth, Rpp32f gridRatio, Rpp32f gridAngle, RpptUintVector2D translateVector, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Gridmask augmentation on HIP backend for a NCHW/NHWC layout tensor
 * \details The gridmask augmentation runs as per https://arxiv.org/abs/2001.04086 for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_gridmask_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] tileWidth tileWidth value for gridmask calculation = width of black square + width of spacing until next black square on grid (a single Rpp32u number with tileWidth <= min(srcDescPtr->w, srcDescPtr->h) that applies to all images in the batch)
 * \param [in] gridRatio gridRatio value for gridmask calculation = black square width / tileWidth (a single Rpp32f number with 0 <= gridRatio <= 1 that applies to all images in the batch)
 * \param [in] gridAngle gridAngle value for gridmask calculation = grid rotation angle in radians (a single Rpp32f number that applies to all images in the batch)
 * \param [in] translateVector translateVector for gridmask calculation = grid X and Y translation lengths in pixels (a single RpptUintVector2D x,y value pair that applies to all images in the batch)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_gridmask_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u tileWidth, Rpp32f gridRatio, Rpp32f gridAngle, RpptUintVector2D translateVector, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Spatter augmentation on HOST backend for a NCHW/NHWC layout tensor
 * \details The spatter augmentation adds random spatter of a user-defined color, for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_spatter_typeMud_img150x150.png Sample Output - Mud Spatter
 * \image html effects_augmentations_spatter_typeInk_img150x150.png Sample Output - Ink Spatter
 * \image html effects_augmentations_spatter_typeBlood_img150x150.png Sample Output - Blood Spatter
 * \param [in] srcPtr source tensor in HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HOST memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] spatterColor RGB values to use for the spatter augmentation (A single set of 3 Rpp8u values as RpptRGB that applies to all images in the batch)
 * \param [in] roiTensorPtrSrc ROI data in HOST memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y)) | (Restrictions - roiTensorPtrSrc[i].xywhROI.roiWidth <= 1920 and roiTensorPtrSrc[i].xywhROI.roiHeight <= 1080)
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_spatter_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptRGB spatterColor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Spatter augmentation on HIP backend for a NCHW/NHWC layout tensor
 * \details The spatter augmentation adds random spatter of a user-defined color, for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_spatter_typeMud_img150x150.png Sample Output - Mud Spatter
 * \image html effects_augmentations_spatter_typeInk_img150x150.png Sample Output - Ink Spatter
 * \image html effects_augmentations_spatter_typeBlood_img150x150.png Sample Output - Blood Spatter
 * \param [in] srcPtr source tensor in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] spatterColor RGB values to use for the spatter augmentation (A single set of 3 Rpp8u values as RpptRGB that applies to all images in the batch)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y)) | (Restrictions - roiTensorPtrSrc[i].xywhROI.roiWidth <= 1920 and roiTensorPtrSrc[i].xywhROI.roiHeight <= 1080)
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_spatter_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptRGB spatterColor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Salt and pepper noise augmentation on HOST backend for a NCHW/NHWC layout tensor
 * \details The salt and pepper noise augmentation adds SnP noise based on user defined noise/salt probabilities, and user defined salt/pepper values for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_salt_and_pepper_noise_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HOST memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] noiseProbailityTensor noiseProbaility values to decide if a destination pixel is a noise-pixel, or equal to source (1D tensor in HOST memory, of size batchSize with 0 <= noiseProbailityTensor[i] <= 1 for each image in batch)
 * \param [in] saltProbailityTensor saltProbaility values to decide if a given destination noise-pixel is salt or pepper (1D tensor in HOST memory, of size batchSize with 0 <= saltProbailityTensor[i] <= 1 for each image in batch)
 * \param [in] saltValueTensor A user-defined salt noise value (1D tensor in HOST memory, of size batchSize with 0 <= saltValueTensor[i] <= 1 for each image in batch)
 * \param [in] pepperValueTensor A user-defined pepper noise value (1D tensor in HOST memory, of size batchSize with 0 <= pepperValueTensor[i] <= 1 for each image in batch)
 * \param [in] seed A user-defined seed value (single Rpp32u value)
 * \param [in] roiTensorPtrSrc ROI data in HOST memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_salt_and_pepper_noise_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *noiseProbabilityTensor, Rpp32f *saltProbabilityTensor, Rpp32f *saltValueTensor, Rpp32f *pepperValueTensor, Rpp32u seed, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Salt and pepper noise augmentation on HIP backend for a NCHW/NHWC layout tensor
 * \details The salt and pepper noise augmentation adds SnP noise based on user defined noise/salt probabilities, and user defined salt/pepper values for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_salt_and_pepper_noise_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] noiseProbailityTensor noiseProbaility values to decide if a destination pixel is a noise-pixel, or equal to source (1D tensor in pinned/HOST memory, of size batchSize with 0 <= noiseProbailityTensor[i] <= 1 for each image in batch)
 * \param [in] saltProbailityTensor saltProbaility values to decide if a given destination noise-pixel is salt or pepper (1D tensor in pinned/HOST memory, of size batchSize with 0 <= saltProbailityTensor[i] <= 1 for each image in batch)
 * \param [in] saltValueTensor A user-defined salt noise value (1D tensor in pinned/HOST memory, of size batchSize with 0 <= saltValueTensor[i] <= 1 for each image in batch)
 * \param [in] pepperValueTensor A user-defined pepper noise value (1D tensor in pinned/HOST memory, of size batchSize with 0 <= pepperValueTensor[i] <= 1 for each image in batch)
 * \param [in] seed A user-defined seed value (single Rpp32u value)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_salt_and_pepper_noise_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *noiseProbabilityTensor, Rpp32f *saltProbabilityTensor, Rpp32f *saltValueTensor, Rpp32f *pepperValueTensor, Rpp32u seed, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Shot noise augmentation on HOST backend for a NCHW/NHWC layout tensor
 * \details The shot noise augmentation adds Poisson/shot noise based on a user defined shotNoiseFactor, for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_shot_noise_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HOST memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] shotNoiseFactorTensor shotNoiseFactor values for each image, which are used to compute the lambda values in a poisson distribution (1D tensor in HOST memory, of size batchSize with shotNoiseFactorTensor[i] >= 0 for each image in batch)
 * \param [in] seed A user-defined seed value (single Rpp32u value)
 * \param [in] roiTensorPtrSrc ROI data in HOST memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_shot_noise_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *shotNoiseFactorTensor, Rpp32u seed, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Shot noise augmentation on HIP backend for a NCHW/NHWC layout tensor
 * \details The shot noise augmentation adds Poisson/shot noise based on a user defined shotNoiseFactor, for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_shot_noise_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] shotNoiseFactorTensor shotNoiseFactor values for each image, which are used to compute the lambda values in a poisson distribution (1D tensor in pinned/HOST memory, of size batchSize with shotNoiseFactorTensor[i] >= 0 for each image in batch)
 * \param [in] seed A user-defined seed value (single Rpp32u value)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_shot_noise_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *shotNoiseFactorTensor, Rpp32u seed, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Gaussian noise augmentation on HOST backend for a NCHW/NHWC layout tensor
 * \details The gaussian noise augmentation adds Gaussian noise based on user defined means and standard deviations, for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_gaussian_noise_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HOST memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] meanTensor mean values for each image, which are used to compute the generalized Box-Mueller transforms in a gaussian distribution (1D tensor in HOST memory, of size batchSize with meanTensor[i] >= 0 for each image in batch)
 * \param [in] stdDevTensor stdDev values for each image, which are used to compute the generalized Box-Mueller transforms in a gaussian distribution (1D tensor in HOST memory, of size batchSize with stdDevTensor[i] >= 0 for each image in batch)
 * \param [in] seed A user-defined seed value (single Rpp32u value)
 * \param [in] roiTensorPtrSrc ROI data in HOST memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_gaussian_noise_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *meanTensor, Rpp32f *stdDevTensor, Rpp32u seed, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Gaussian noise augmentation on HIP backend for a NCHW/NHWC layout tensor
 * \details The gaussian noise augmentation adds Gaussian noise based on user defined means and standard deviations, for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_gaussian_noise_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] meanTensor mean values for each image, which are used to compute the generalized Box-Mueller transforms in a gaussian distribution (1D tensor in pinned/HOST memory, of size batchSize with meanTensor[i] >= 0 for each image in batch)
 * \param [in] stdDevTensor stdDev values for each image, which are used to compute the generalized Box-Mueller transforms in a gaussian distribution (1D tensor in pinned/HOST memory, of size batchSize with stdDevTensor[i] >= 0 for each image in batch)
 * \param [in] seed A user-defined seed value (single Rpp32u value)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_gaussian_noise_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *meanTensor, Rpp32f *stdDevTensor, Rpp32u seed, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Non linear blend augmentation on HOST backend for a NCHW/NHWC layout tensor
 * \details The non linear blend augmentation adds standard deviation based non-linear alpha-blending, between two sets of batches of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input1
 * \image html img150x150_2.png Sample Input2
 * \image html effects_augmentations_non_linear_blend_img150x150.png Sample Output
 * \param [in] srcPtr1 source1 tensor in HOST memory
 * \param [in] srcPtr2 source2 tensor in HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HOST memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] stdDevTensor stdDev values for each image, which are used to compute the generalized Box-Mueller transforms in a gaussian distribution (1D tensor in HOST memory, of size batchSize with stdDevTensor[i] >= 0 for each image in batch)
 * \param [in] roiTensorPtrSrc ROI data in HOST memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_non_linear_blend_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *stdDevTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Non linear blend augmentation on HIP backend for a NCHW/NHWC layout tensor
 * \details The non linear blend augmentation adds standard deviation based non-linear alpha-blending, between two sets of batches of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input1
 * \image html img150x150_2.png Sample Input2
 * \image html effects_augmentations_non_linear_blend_img150x150.png Sample Output
 * \param [in] srcPtr1 source1 tensor in HIP memory
 * \param [in] srcPtr2 source2 tensor in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] stdDevTensor stdDev values for each image, which are used to compute the generalized Box-Mueller transforms in a gaussian distribution (1D tensor in pinned/HOST memory, of size batchSize with stdDevTensor[i] >= 0 for each image in batch)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_non_linear_blend_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *stdDevTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Water augmentation on HOST backend for a NCHW/NHWC layout tensor
 * \details The water augmentation adds a water effect for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_water_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HOST memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param[in] amplitudeXTensor amplitudeX values for water effect (1D tensor in HOST memory, of size batchSize)
 * \param[in] amplitudeYTensor amplitudeY values for water effect (1D tensor in HOST memory, of size batchSize)
 * \param[in] frequencyXTensor frequencyX values for water effect (1D tensor in HOST memory, of size batchSize)
 * \param[in] frequencyYTensor frequencyY values for water effect (1D tensor in HOST memory, of size batchSize)
 * \param[in] phaseXTensor phaseX values for water effect (1D tensor in HOST memory, of size batchSize)
 * \param[in] phaseYTensor phaseY values for water effect (1D tensor in HOST memory, of size batchSize)
 * \param [in] roiTensorPtrSrc ROI data in HOST memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_water_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *amplitudeXTensor, Rpp32f *amplitudeYTensor, Rpp32f *frequencyXTensor, Rpp32f *frequencyYTensor, Rpp32f *phaseXTensor, Rpp32f *phaseYTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Water augmentation on HIP backend for a NCHW/NHWC layout tensor
 * \details The water augmentation adds a water effect for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_water_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param[in] amplitudeXTensor amplitudeX values for water effect (1D tensor in pinned/HOST memory, of size batchSize)
 * \param[in] amplitudeYTensor amplitudeY values for water effect (1D tensor in pinned/HOST memory, of size batchSize)
 * \param[in] frequencyXTensor frequencyX values for water effect (1D tensor in pinned/HOST memory, of size batchSize)
 * \param[in] frequencyXTensor frequencyY values for water effect (1D tensor in pinned/HOST memory, of size batchSize)
 * \param[in] phaseXTensor phaseX values for water effect (1D tensor in pinned/HOST memory, of size batchSize)
 * \param[in] phaseYTensor phaseY values for water effect (1D tensor in pinned/HOST memory, of size batchSize)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_water_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *amplitudeXTensor, Rpp32f *amplitudeYTensor, Rpp32f *frequencyXTensor, Rpp32f *frequencyYTensor, Rpp32f *phaseXTensor, Rpp32f *phaseYTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief RICAP (Random Image Crop And Patch) augmentation on HOST backend for a NCHW/NHWC layout tensor
 * \details The RICAP (Random Image Crop And Patch) augmentation runs as per https://arxiv.org/abs/1811.09030 for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * The RICAP augmentation requires dimensions of input images to be the same across entire batch.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input1
 * \image html img150x150_1.png Sample Input2
 * \image html img150x150_2.png Sample Input3
 * \image html effects_augmentations_ricap_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HOST memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param[in] permutedIndicesTensor Array of batchSize permutation sets (2D tensor in HOST memory, of batchSize * 4. Each set of 4 permutations contains Rpp32u image indices for each region in the respective RICAP-output-image in the batch)
 * \param[in] roiPtrInputCropRegion Array of 4 ROIs (2D tensor in HOST memory, of size 4 * 4-elements per ROI, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_ricap_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u *permutedIndicesTensor, RpptROIPtr roiPtrInputCropRegion, RpptRoiType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief RICAP (Random Image Crop And Patch) augmentation on HIP backend for a NCHW/NHWC layout tensor
 * \details The RICAP (Random Image Crop And Patch) augmentation runs as per https://arxiv.org/abs/1811.09030 for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * The RICAP augmentation requires dimensions of input images to be the same across entire batch.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input1
 * \image html img150x150_1.png Sample Input2
 * \image html img150x150_2.png Sample Input3
 * \image html effects_augmentations_ricap_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param[in] permutedIndicesTensor Array of batchSize permutation sets (2D tensor in pinned/HOST memory, of batchSize * 4. Each set of 4 permutations contains Rpp32u image indices for each region in the respective RICAP-output-image in the batch)
 * \param[in] roiPtrInputCropRegion Array of 4 ROIs (2D tensor in HIP memory, of size 4 * 4-elements per ROI, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_ricap_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u *permutedIndicesTensor, RpptROIPtr roiPtrInputCropRegion, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Vignette augmentation on HOST backend for a NCHW/NHWC layout tensor
 * \details The Vignette augmentation adds a vignette effect for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_vignette_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HOST memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param[in] vignetteIntensityTensor intensity values to quantify vignette effect (1D tensor of size batchSize with 0 < vignetteIntensityTensor[n] for each image in batch)
 * \param [in] roiTensorPtrSrc ROI data in HOST memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
// NOTE: Pixel mismatch of 5% is expected between HIP and HOST Tensor variations due to usage of fastexpavx() instead of exp() in HOST Tensor.
RppStatus rppt_vignette_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *vignetteIntensityTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Vignette augmentation on HIP backend for a NCHW/NHWC layout tensor
 * \details The vignette augmentation adds a vignette effect for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_vignette_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param[in] vignetteIntensityTensor intensity values to quantify vignette effect (1D tensor of size batchSize with 0 < vignetteIntensityTensor[n] for each image in batch)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_vignette_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *vignetteIntensityTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Jitter augmentation on HOST backend for a NCHW/NHWC layout tensor
 * \details The jitter augmentation adds a jitter effect for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_jitter_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HOST memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] kernelSizeTensor kernelsize value for jitter calculation (kernelSize = 3/5/7 for optimal use)
 * \param [in] roiTensorPtrSrc ROI data in HOST memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_jitter_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u *kernelSizeTensor, Rpp32u seed, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Jitter augmentation on HIP backend for a NCHW/NHWC layout tensor
 * \details The jitter augmentation adds a jitter effect for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_jitter_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory
 * \param un[in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in]  kernelSizeTensor kernelsize value for jitter calculation (kernelSize = 3/5/7 for optimal use)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_jitter_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u *kernelSizeTensor, Rpp32u seed, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief  Gaussian noise augmentation on HOST backend
 * \details This function adds gaussian noise to a batch of 4D tensors.
 *          Support added for u8 -> u8, f32 -> f32 datatypes.
 * \image html https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenInputs/input150x150x4.gif Sample Input
 * \image html https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenOutputs/effects_augmentations_gaussian_noise_150x150x4.gif Sample Output
 * \param [in] srcPtr source tensor in HOST memory
 * \param [in] srcGenericDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor in HOST memory
 * \param [in] dstGenericDescPtr destination tensor descriptor
 * \param [in] meanTensor mean values for each input, which are used to compute the generalized Box-Mueller transforms in a gaussian distribution (1D tensor of size batchSize with meanTensor[i] >= 0 for each image in batch)
 * \param [in] stdDevTensor stdDev values for each image, which are used to compute the generalized Box-Mueller transforms in a gaussian distribution (1D tensor of size batchSize with stdDevTensor[i] >= 0 for each image in batch)
 * \param [in] seed A user-defined seed value (single Rpp32u value)
 * \param [in] roiGenericPtrSrc ROI data for each image in source tensor (tensor of batchSize RpptRoiGeneric values)
 * \param [in] roiType ROI type used (RpptRoi3DType::XYZWHD or RpptRoi3DType::LTFRBB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_gaussian_noise_voxel_host(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32f *meanTensor, Rpp32f *stdDevTensor, Rpp32u seed, RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief  Gaussian noise augmentation on HIP backend
 * \details This function adds gaussian noise to a batch of 4D tensors.
 *          Support added for u8 -> u8, f32 -> f32 datatypes.
 * \image html https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenInputs/input150x150x4.gif Sample Input
 * \image html https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenOutputs/effects_augmentations_gaussian_noise_150x150x4.gif Sample Output
 * \param [in] srcPtr source tensor in HIP memory
 * \param [in] srcGenericDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstGenericDescPtr destination tensor descriptor
 * \param [in] meanTensor mean values for each input, which are used to compute the generalized Box-Mueller transforms in a gaussian distribution (1D tensor of size batchSize with meanTensor[i] >= 0 for each image in batch)
 * \param [in] stdDevTensor stdDev values for each input, which are used to compute the generalized Box-Mueller transforms in a gaussian distribution (1D tensor of size batchSize with stdDevTensor[i] >= 0 for each image in batch)
 * \param [in] seed A user-defined seed value (single Rpp32u value)
 * \param [in] roiGenericPtrSrc ROI data for each image in source tensor (tensor of batchSize RpptRoiGeneric values)
 * \param [in] roiType ROI type used (RpptRoi3DType::XYZWHD or RpptRoi3DType::LTFRBB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_gaussian_noise_voxel_gpu(RppPtr_t srcPtr, RpptGenericDescPtr srcDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstDescPtr, Rpp32f *meanTensor, Rpp32f *stdDevTensor, Rpp32u seed, RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Erase augmentation on HOST backend for a NCHW/NHWC layout tensor
 * \details This function erases one or more user defined regions from an image, for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 *          srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 *          dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_erase_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HOST memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] anchorBoxInfoTensor anchorBoxInfo values of type RpptRoiLtrb for each erase-region inside each image in the batch. Restrictions -
            - 0 <= anchorBoxInfo[i] < respective image width/height
            - Erase-region anchor boxes on each image given by the user must not overlap
 * \param [in] colorsTensor RGB values to use for each erase-region inside each image in the batch. (colors[i] will have range equivalent of srcPtr)
 * \param [in] numBoxesTensor number of erase-regions per image, for each image in the batch. (numBoxesTensor[n] >= 0)
 * \param [in] roiTensorPtrSrc ROI data in HOST memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_erase_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptRoiLtrb *anchorBoxInfoTensor, RppPtr_t colorsTensor, Rpp32u *numBoxesTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Erase augmentation on HIP backend for a NCHW/NHWC layout tensor
 * \details This function erases one or more user defined regions from an image, for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 *          srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 *          dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_erase_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] anchorBoxInfoTensor anchorBoxInfo values of type RpptRoiLtrb for each erase-region inside each image in the batch. Restrictions -
            - 0 <= anchorBoxInfo[i] < respective image width/height
            - Erase-region anchor boxes on each image given by the user must not overlap
 * \param [in] colorsTensor RGB values to use for each erase-region inside each image in the batch. (colors[i] will have range equivalent of srcPtr)
 * \param [in] numBoxesTensor number of erase-regions per image, for each image in the batch. (numBoxesTensor[n] >= 0)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_erase_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptRoiLtrb *anchorBoxInfoTensor, RppPtr_t colorsTensor, Rpp32u *numBoxesTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Glitch augmentation on HOST backend for a NCHW/NHWC layout tensor
 * \details The glitch augmentation adds a glitch effect for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_glitch_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HOST memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] rgbOffsets RGB offset values to use for the glitch augmentation (A single set of 3 Rppi point values that applies to all images in the batch.
 *                        For each point and for each image in the batch: 0 < point.x < width, 0 < point.y < height)
 * \param [in] roiTensorPtrSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_glitch_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptChannelOffsets *rgbOffsets, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Glitch augmentation on HIP backend for a NCHW/NHWC layout tensor
 * \details The glitch augmentation adds a glitch effect for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_glitch_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] rgbOffsets RGB offset values to use for the glitch augmentation (A 1D tensor in pinned/HOST memory contains single set of 3 Rppi point values that applies to all images in the batch.
 *                        For each point and for each image in the batch: 0 < point.x < width, 0 < point.y < height)
 * \param [in] roiTensorPtrSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_glitch_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptChannelOffsets *rgbOffsets, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Rain augmentation on HOST backend for a NCHW/NHWC layout tensor
 * \details The rain augmentation simulates a rain effect for a batch of RGB (3-channel) / greyscale (1-channel) images with an NHWC/NCHW tensor layout.<br>
 * <b> NOTE: This augmentation gives a more realistic Rain output when all images in a batch are of similar / same sizes </b> <br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be the same depth as srcPtr.
 * \image html img640x480.png Sample Input
 * \image html effects_augmentations_rain_img640x480.png Sample Output
 * \param [in] srcPtr source tensor in HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HOST memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] rainPercentage The percentage of the rain effect to be applied (0 <= rainPercentage <= 100)
 * \param [in] rainWidth Width of the rain drops in pixels. To be tuned by user depending on size of the image.
 * \param [in] rainHeight Height of the rain drops in pixels. To be tuned by user depending on size of the image.
 * \param [in] slantAngle Slant angle of the rain drops (positive value for right slant, negative for left slant). A single Rpp32s/f representing the slant of raindrops in degrees.
 *                        Values range from [-90, 90], where -90 represents extreme left slant, 0 is vertical, and 90 is extreme right slant.
 * \param [in] alpha An array of alpha blending values to be used for blending the rainLayer and the input image for each image in the batch (0 ≤ alpha ≤ 1 for each image in the batch).
 * \param [in] roiTensorPtrSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreateWithBatchSize()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_rain_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f rainPercentage, Rpp32u rainWidth, Rpp32u rainHeight, Rpp32f slantAngle, Rpp32f *alpha, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Rain augmentation on HIP backend for a NCHW/NHWC layout tensor
 * \details The rain augmentation simulates a rain effect for a batch of RGB (3-channel) / greyscale (1-channel) images with an NHWC/NCHW tensor layout.<br>
 * <b> NOTE: This augmentation gives a more realistic Rain output when all images in a batch are of similar / same sizes </b> <br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be the same depth as srcPtr.
 * \image html img640x480.png Sample Input
 * \image html effects_augmentations_rain_img640x480.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] rainPercentage The percentage of the rain effect to be applied (0 <= rainPercentage <= 100)
 * \param [in] rainWidth Width of the rain drops in pixels. To be tuned by user depending on size of the image.
 * \param [in] rainHeight Height of the rain drops in pixels. To be tuned by user depending on size of the image.
 * \param [in] slantAngle Slant angle of the rain drops (positive value for right slant, negative for left slant). A single Rpp32s/f representing the slant of raindrops in degrees.
 *                        Values range from [-90, 90], where -90 represents extreme left slant, 0 is vertical, and 90 is extreme right slant.
 * \param [in] alpha An array of alpha blending values in pinned / HIP memory is used for blending the rainLayer and the input image for each image in the batch (0 ≤ alpha ≤ 1 for each image in the batch).
 * \param [in] roiTensorPtrSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreateWithStreamAndBatchSize()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_rain_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f rainPercentage, Rpp32u rainWidth, Rpp32u rainHeight, Rpp32f slantAngle, Rpp32f *alpha, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Pixelate augmentation on HOST backend for a NCHW/NHWC layout tensor
 * \details The pixelate augmentation performs a pixelate transformation for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_pixelate_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HOST memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] intermediateScratchBufferPtr intermediate scratch buffer in HOST memory (Minimum size = srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(Rpp32f))
 * \param [in] pixelationPercentage 'pixelationPercentage' variable controls how much pixelation is applied to images.(pixelationPercentage value ranges from 0 to 100)
 * \param [in] roiTensorPtrSrc ROI data in HOST memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_pixelate_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RppPtr_t intermediateScratchBufferPtr, Rpp32f pixelationPercentage, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Pixelate augmentation on HIP backend for a NCHW/NHWC layout tensor
 * \details The pixelate augmentation performs a pixelate transformation for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html effects_augmentations_pixelate_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] intermediateScratchBufferPtr intermediate scratch buffer in HIP memory (Minimum size = srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(Rpp32f))
 * \param [in] pixelationPercentage 'pixelationPercentage' variable controls how much pixelation is applied to images.(pixelationPercentage value ranges from 0 to 100)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_pixelate_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RppPtr_t intermediateScratchBufferPtr, Rpp32f pixelationPercentage, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Fog augmentation on HOST backend for a NCHW/NHWC layout tensor
 * \details The fog augmentation adds a fog effect for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * <b> NOTE: This augmentation gives a more realistic fog output when all images in a batch are of similar / same sizes </b> <br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img640x480.png Sample Input
 * \image html effects_augmentations_fog_img640x480.png Sample Output
 * \param [in] srcPtr source tensor in HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HOST memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] intensityFactor intensity factor values for fog calculation (1D tensor in HOST memory, of size batchSize, with 0 <= intensityFactor <= 0.5 for each image in batch)
 * \param [in] grayFactor gray factor values to introduce grayness in the image for fog calculation (1D tensor in HOST memory, of size batchSize, with 0 <= grayFactor <= 1 for each image in batch)
 * \param [in] roiTensorPtrSrc ROI data for each image in source tensor in HOST memory (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_fog_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *intensityFactor,Rpp32f *grayFactor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Fog augmentation on HIP backend for a NCHW/NHWC layout tensor
 * \details The fog augmentation adds a fog effect for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * <b> NOTE: This augmentation gives a more realistic fog output when all images in a batch are of similar / same sizes </b> <br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img640x480.png Sample Input
 * \image html effects_augmentations_fog_img640x480.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] intensityFactor intensity factor values for fog calculation (1D tensor in pinned/HIP memory, of size batchSize, with 0 <= intensityFactor <= 0.5 for each image in batch)
 * \param [in] greyFactor gray factor values to introduce grayness in the image for fog calculation (1D tensor in pinned/HIP memory, of size batchSize, with 0 <= greyFactor <= 1 for each image in batch)
 * \param [in] roiTensorPtrSrc ROI data for each image in source tensor in pinned memory (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_fog_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *intensityFactor, Rpp32f *greyFactor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT
/*! @}
 */

#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_EFFECTS_AUGMENTATIONS_H
