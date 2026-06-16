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

#ifndef RPPT_TENSOR_GEOMETRIC_AUGMENTATIONS_H
#define RPPT_TENSOR_GEOMETRIC_AUGMENTATIONS_H

#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \file
 * \brief RPPT Tensor Operations - Geometric Augmentations.
 * \defgroup group_rppt_tensor_geometric_augmentations RPPT Tensor Operations - Geometric
 * Augmentations. \brief RPPT Tensor Operations - Geometric Augmentations.
 */

/*! \addtogroup group_rppt_tensor_geometric_augmentations
 * @{
 */

/*! \brief Crop augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The crop augmentation crops each image to a given ROI, for a batch of RGB(3 channel) /
 * greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html geometric_augmentations_crop_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4,
 * offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3) \param [out] dstPtr
 * destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend) \param [in]
 * dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0,
 * dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr) \param [in]
 * roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for
 * each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y,
 * roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y)) \param [in] roiType ROI type used
 * (RpptRoiType::XYWH or RpptRoiType::LTRB) \param [in] rppHandle RPP HIP/HOST handle created with
 * <tt>\ref rppCreate()</tt> \param [in] executionBackend backend for execution
 * (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND) \return A <tt> \ref RppStatus</tt>
 * enumeration. \retval RPP_SUCCESS Successful completion. \retval RPP_ERROR* Unsuccessful
 * completion.
 */
RppStatus rppt_crop(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                    RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                    rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Crop mirror normalize augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The crop mirror normalize augmentation crops each image to a given ROI, does an optional
 * mirror and/or normalize for a batch of RGB(3 channel) / greyscale(1 channel) images with an
 * NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html geometric_augmentations_crop_mirror_normalize_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4,
 * offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3) \param [out] dstPtr
 * destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend) \param [in]
 * dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0,
 * dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr) \param [in]
 * offsetTensor offset values for normalization (1D tensor in pinned / HIP memory (for HIP backend)
 * or HOST memory (for HOST backend), of size batchSize, with offsetTensor[n] <= 0) \param [in]
 * multiplierTensor multiplier values for normalization (1D tensor in pinned / HIP memory (for HIP
 * backend) or HOST memory (for HOST backend), of size batchSize, with multiplierTensor[n] > 0)
 * \param [in] mirrorTensor mirror flag values to set mirroring on/off (1D tensor in pinned / HIP
 * memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize, with
 * mirrorTensor[n] = 0/1) \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or
 * HOST memory (for HOST backend), for each image in source tensor (2D tensor of size batchSize * 4,
 * in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y)) \param
 * [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB) \param [in] rppHandle RPP
 * HIP/HOST handle created with <tt>\ref rppCreate()</tt> \param [in] executionBackend backend for
 * execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND) \return A <tt> \ref
 * RppStatus</tt> enumeration. \retval RPP_SUCCESS Successful completion. \retval RPP_ERROR*
 * Unsuccessful completion.
 */
RppStatus rppt_crop_mirror_normalize(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32f* offsetTensor,
                                     Rpp32f* multiplierTensor, Rpp32u* mirrorTensor,
                                     RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                     rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Warp affine augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The warp affine augmentation performs affine transformations for a batch of RGB(3
 * channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html geometric_augmentations_warp_affine_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP/HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0,
 * dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3) \param [out] dstPtr destination tensor in
 * HIP/HOST memory \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4,
 * offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] affineTensor affine matrix values for transformation calculation (2D tensor in pinned
 * / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize * 6 for each
 * image in batch) \param [in] interpolationType Interpolation type used
 * (RpptInterpolationType::XYWH or RpptRoiType::LTRB) \param [in] roiTensorPtrSrc ROI data in HIP
 * memory (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D
 * tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or
 * LTRB(lt.x, lt.y, rb.x, rb.y)) \param [in] roiType ROI type used (RpptRoiType::XYWH or
 * RpptRoiType::LTRB) \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref
 * rppCreate()</tt> \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND
 * or RppBackend::RPP_HIP_BACKEND) \return A <tt> \ref RppStatus</tt> enumeration. \retval
 * RPP_SUCCESS Successful completion. \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_warp_affine(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                           RpptDescPtr dstDescPtr, Rpp32f* affineTensor,
                           RpptInterpolationType interpolationType, RpptROIPtr roiTensorPtrSrc,
                           RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Flip augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The flip augmentation performs a mask-controlled horizontal/vertical flip for a batch of
 * RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html geometric_augmentations_flip_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP/HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0,
 * dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3) \param [out] dstPtr destination tensor in
 * HIP/HOST memory \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4,
 * offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] horizontalTensor horizontal flag values to set horizontal flip on/off (1D tensor in
 * pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize, with
 * horizontalTensor[i] = 0/1) \param [in] verticalTensor vertical flag values to set vertical flip
 * on/off (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of
 * size batchSize, with verticalTensor[i] = 0/1) \param [in] roiTensorPtrSrc ROI data in HIP memory
 * (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor
 * of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x,
 * lt.y, rb.x, rb.y)) \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or
 * RppBackend::RPP_HIP_BACKEND) \return A <tt> \ref RppStatus</tt> enumeration. \retval RPP_SUCCESS
 * Successful completion. \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_flip(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                    RpptDescPtr dstDescPtr, Rpp32u* horizontalTensor, Rpp32u* verticalTensor,
                    RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle,
                    RppBackend executionBackend);

/*! \brief Resize augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The resize augmentation performs an image resize for a batch of RGB(3 channel) /
 * greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html geometric_augmentations_resize_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4,
 * offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3) \param [out] dstPtr
 * destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend) \param [in]
 * dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0,
 * dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr) \param [in]
 * dstImgSizes destination image sizes (<tt> \ref RpptImagePatchPtr </tt> type pointer to array, in
 * pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize)
 * \param [in] interpolationType Interpolation type used in <tt> \ref RpptInterpolationType </tt>
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format -
 * XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y)) \param [in] roiType ROI
 * type used (RpptRoiType::XYWH or RpptRoiType::LTRB) \param [in] rppHandle RPP HIP/HOST handle
 * created with <tt>\ref rppCreate()</tt> \param [in] executionBackend backend for execution
 * (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND) \return A <tt> \ref RppStatus</tt>
 * enumeration. \retval RPP_SUCCESS Successful completion. \retval RPP_ERROR* Unsuccessful
 * completion.
 */
RppStatus rppt_resize(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                      RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSizes,
                      RpptInterpolationType interpolationType, RpptROIPtr roiTensorPtrSrc,
                      RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Resize mirror normalize augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The resize mirror normalize augmentation performs an image resize, an optional mirror
 * and/or normalize, for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW
 * tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html geometric_augmentations_resize_mirror_normalize_img115x115.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4,
 * offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3) \param [out] dstPtr
 * destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend) \param [in]
 * dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0,
 * dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr) \param [in]
 * dstImgSizes destination image sizes (<tt> \ref RpptImagePatchPtr </tt> type pointer to array, in
 * HOST memory, of size batchSize) \param [in] interpolationType Interpolation type used in <tt>
 * \ref RpptInterpolationType </tt> \param [in] meanTensor mean value for each image in the batch
 * (meanTensor[n] >= 0, 1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST
 * backend), of size = batchSize for greyscale images, size = batchSize * 3 for RGB images)) \param
 * [in] stdDevTensor standard deviation value for each image in the batch (stdDevTensor[n] >= 0, 1D
 * tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size =
 * batchSize for greyscale images, size = batchSize * 3 for RGB images) \param [in] mirrorTensor
 * mirror flag value to set mirroring on/off (1D tensor in pinned / HIP memory (for HIP backend) or
 * HOST memory (for HOST backend), of size batchSize, with mirrorTensor[n] = 0/1) \param [in]
 * roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for
 * each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y,
 * roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y)) \param [in] roiType ROI type used
 * (RpptRoiType::XYWH or RpptRoiType::LTRB) \param [in] rppHandle RPP HIP/HOST handle created with
 * <tt>\ref rppCreate()</tt> \param [in] executionBackend backend for execution
 * (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND) \return A <tt> \ref RppStatus</tt>
 * enumeration. \retval RPP_SUCCESS Successful completion. \retval RPP_ERROR* Unsuccessful
 * completion.
 */
RppStatus rppt_resize_mirror_normalize(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                                       RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSizes,
                                       RpptInterpolationType interpolationType, Rpp32f* meanTensor,
                                       Rpp32f* stdDevTensor, Rpp32u* mirrorTensor,
                                       RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                       rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Resize crop mirror augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The resize crop mirror augmentation performs an image resized crop, with an optional
 * mirror, for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor
 * layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html geometric_augmentations_resize_crop_mirror_img115x115.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4,
 * offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3) \param [out] dstPtr
 * destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend) \param [in]
 * dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0,
 * dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr) \param [in]
 * dstImgSizes destination image sizes (<tt> \ref RpptImagePatchPtr </tt> type pointer to array, in
 * HOST memory, of size batchSize) \param [in] interpolationType Interpolation type used in <tt>
 * \ref RpptInterpolationType </tt> \param [in] mirrorTensor mirror flag value to set mirroring
 * on/off (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of
 * size batchSize, with mirrorTensor[n] = 0/1) \param [in] roiTensorPtrSrc ROI data in HIP memory
 * (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor
 * of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x,
 * lt.y, rb.x, rb.y)) \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or
 * RppBackend::RPP_HIP_BACKEND) \return A <tt> \ref RppStatus</tt> enumeration. \retval RPP_SUCCESS
 * Successful completion. \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_resize_crop_mirror(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                                  RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSizes,
                                  RpptInterpolationType interpolationType, Rpp32u* mirrorTensor,
                                  RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                  rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Rotate augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The rotate augmentation performs a rotate transformations for a batch of RGB(3 channel)
 * / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html geometric_augmentations_rotate_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4,
 * offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3) \param [out] dstPtr
 * destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend) \param [in]
 * dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0,
 * dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr) \param [in] angle
 * image rotation angle in degrees - positive deg-anticlockwise/negative deg-clockwise (1D tensor in
 * pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize)
 * \param [in] interpolationType Interpolation type used (RpptInterpolationType::XYWH or
 * RpptRoiType::LTRB) \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST
 * memory (for HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in
 * either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y)) \param
 * [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB) \param [in] rppHandle RPP
 * HIP/HOST handle created with <tt>\ref rppCreate()</tt> \param [in] executionBackend execution
 * backend to run the augmentation on (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_rotate(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                      RpptDescPtr dstDescPtr, Rpp32f* angle,
                      RpptInterpolationType interpolationType, RpptROIPtr roiTensorPtrSrc,
                      RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Phase augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The phase augmentation computes phase of corresponding pixels for a batch of RGB(3
 * channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input1
 * \image html img150x150_2.png Sample Input2
 * \image html geometric_augmentations_phase_img150x150.png Sample Output
 * \param [in] srcPtr1 source1 tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) \param [in] srcPtr2 source2 tensor in HIP memory (for HIP backend) or HOST memory (for
 * HOST backend) \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4,
 * offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3) \param [out] dstPtr
 * destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend) \param [in]
 * dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0,
 * dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr) \param [in]
 * roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for
 * each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y,
 * roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y)) \param [in] roiType ROI type used
 * (RpptRoiType::XYWH or RpptRoiType::LTRB) \param [in] rppHandle RPP HIP/HOST handle created with
 * <tt>\ref rppCreate()</tt> \param [in] executionBackend execution backend to run the augmentation
 * on (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND) \return A <tt> \ref
 * RppStatus</tt> enumeration. \retval RPP_SUCCESS Successful completion. \retval RPP_ERROR*
 * Unsuccessful completion.
 */
RppStatus rppt_phase(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                     RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                     rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Slice augmentation on HIP/HOST backend
 * \details This function performs slice augmentation on a generic 4D tensor.
 *          Slice augmentation involves selecting a region of interest (ROI) from the source tensor
 *          and copying it to the destination tensor. Support added for f32 -> f32 and u8 -> u8
 * dataypes. \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for
 * HOST backend) \param [in] srcGenericDescPtr source tensor descriptor \param [out] dstPtr
 * destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend) \param [in]
 * dstGenericDescPtr destination tensor descriptor \param [in] anchorTensor starting index of the
 * slice for each dimension in input (1D tensor in pinned / HIP memory (for HIP backend) or HOST
 * memory (for HOST backend), of size = batchSize * numberOfDimensions) \param [in] shapeTensor
 * length of the slice for each dimension in input (1D tensor in pinned / HIP memory (for HIP
 * backend) or HOST memory (for HOST backend), of size = batchSize * numberOfDimensions) \param [in]
 * fillValue fill value that is used to fill output if enablePadding is set to true \param [in]
 * enablePadding boolean flag to specify if padding is enabled or not \param [in] roiTensor roi data
 * in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend) (1D tensor of size =
 * batchSize * numberOfDimensions * 2) \param [in] rppHandle RPP HIP/HOST handle created with
 * <tt>\ref rppCreate()</tt> \param [in] executionBackend execution backend to run the augmentation
 * on (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND) \return A <tt> \ref
 * RppStatus</tt> enumeration. \retval RPP_SUCCESS Successful completion. \retval RPP_ERROR*
 * Unsuccessful completion.
 */
RppStatus rppt_slice(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr,
                     RpptGenericDescPtr dstGenericDescPtr, Rpp32s* anchorTensor,
                     Rpp32s* shapeTensor, RppPtr_t fillValue, bool enablePadding, Rpp32u* roiTensor,
                     rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Crop and Patch augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The crop and patch augmentation crops a ROI from 1st image and patches the cropped
 region in 2nd image as per the patch co-ordinates for a batch of RGB(3 channel) / greyscale(1
 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr1 depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - srcPtr2 depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input1
 * \image html img150x150_2.png Sample Input2
 * \image html geometric_augmentations_crop_and_patch_img150x150.png Sample Output
 * \param [in] srcPtr1 source tensor1 in HIP memory (for HIP backend) or HOST memory (for HOST
 backend)
 * \param [in] srcPtr2 source tensor2 in HIP memory (for HIP backend) or HOST memory (for HOST
 backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0,
 dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes
 >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] roiTensorPtrDst ROI data in HIP memory (for HIP backend) or HOST memory (for HOST
 backend), for each image in destination tensor (2D tensor of size batchSize * 4, in either format -
 XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] cropRoiTensor crop co-ordinates in HIP memory (for HIP backend) or HOST memory (for
 HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format -
 XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] patchRoiTensor patch co-ordinates in HIP memory (for HIP backend) or HOST memory (for
 HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format -
 XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend execution backend to run the augmentation on
 (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_crop_and_patch(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptDescPtr srcDescPtr,
                              RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrDst,
                              RpptROIPtr cropRoiTensor, RpptROIPtr patchRoiTensor,
                              RpptRoiType roiType, rppHandle_t rppHandle,
                              RppBackend executionBackend);

/*! \brief Flip voxel augmentation on HIP/HOST backend
 * \details The flip voxel augmentation performs a mask-controlled horizontal/vertical/depth flip on
 a generic 4D tensor. <br> Support added for f32 -> f32 and u8 -> u8 dataypes.
 * \image html
 https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenInputs/input150x150x4.gif
 Sample Input
 * \image html
 https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenOutputs/geometric_augmentations_flip_150x150x4.gif
 Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 backend)
 * \param [in] srcGenericDescPtr source tensor descriptor (Restrictions - numDims = 5, offsetInBytes
 >= 0, dataType = U8/F32, layout = NCDHW/NDHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 backend)
 * \param [in] dstGenericDescPtr destination tensor descriptor (Restrictions - numDims = 5,
 offsetInBytes >= 0, dataType = U8/F32, layout = NCDHW/NDHWC, c = 1/3)
 * \param [in] horizontalTensor horizontal flag values to set horizontal flip on/off (1D tensor in
 pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize, with
 horizontalTensor[i] = 0/1)
 * \param [in] verticalTensor vertical flag values to set vertical flip on/off (1D tensor in pinned
 / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize, with
 verticalTensor[i] = 0/1)
 * \param [in] depthTensor depth flag values to set depth flip on/off (1D tensor in pinned / HIP
 memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize, with depthTensor[i]
 = 0/1)
 * \param [in] roiGenericPtrSrc ROI data for each image in source tensor (tensor of batchSize
 RpptRoiGeneric values)
 * \param [in] roiType ROI type used (RpptRoi3DType::XYZWHD or RpptRoi3DType::LTFRBB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend execution backend to run the augmentation on
 (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 * \ingroup group_tensor_geometric
 */
RppStatus rppt_flip_voxel(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr,
                          RpptGenericDescPtr dstGenericDescPtr, Rpp32u* horizontalTensor,
                          Rpp32u* verticalTensor, Rpp32u* depthTensor,
                          RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType,
                          rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Remap augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details Performs a remap operation using user specified remap tables for a batch of RGB(3
 * channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout. For each image, the
 * output(x,y) = input(mapx(x, y), mapy(x, y)) for every (x,y) in the destination image.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html geometric_augmentations_remap_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4,
 * offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3) \param [out] dstPtr
 * destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend) \param [in]
 * dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0,
 * dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr) \param [in]
 * rowRemapTable Rpp32f row numbers in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) for every pixel in the input batch of images (Restrictions - rois in the rowRemapTable
 * data for each image in batch must match roiTensorPtrSrc) \param [in] colRemapTable Rpp32f column
 * numbers in HIP memory (for HIP backend) or HOST memory (for HOST backend) for every pixel in the
 * input batch of images (Restrictions - rois in the colRemapTable data for each image in batch must
 * match roiTensorPtrSrc) \param [in] tableDescPtr rowRemapTable and colRemapTable common tensor
 * descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = F32, layout = NHWC, c = 1)
 * \param [in] interpolationType Interpolation type used in <tt> \ref RpptInterpolationType </tt>
 * (Restrictions - Supports only NEAREST_NEIGHBOR and BILINEAR) \param [in] roiTensorPtrSrc ROI data
 * in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in source
 * tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight)
 * or LTRB(lt.x, lt.y, rb.x, rb.y)) \param [in] roiType ROI type used (RpptRoiType::XYWH or
 * RpptRoiType::LTRB) \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref
 * rppCreate()</tt> \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND
 * or RppBackend::RPP_HIP_BACKEND) \return A <tt> \ref RppStatus</tt> enumeration. \retval
 * RPP_SUCCESS Successful completion. \retval RPP_ERROR* Unsuccessful completion. \ingroup
 * group_tensor_geometric
 */
RppStatus rppt_remap(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                     RpptDescPtr dstDescPtr, Rpp32f* rowRemapTable, Rpp32f* colRemapTable,
                     RpptDescPtr tableDescPtr, RpptInterpolationType interpolationType,
                     RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle,
                     RppBackend executionBackend);

/*! \brief Lens correction transformation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details Performs lens correction transforms on an image to compensate barrel lens distortion of
 * RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * Note: Returns a black image if the passed camera matrix has a 0 determinant
 * \image html lens_img640x480.png Sample Input
 * \image html geometric_augmentations_lens_correction_img_640x480.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4,
 * offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3) \param [out] dstPtr
 * destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend) \param [in]
 * dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0,
 * dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr) \param [in]
 * rowRemapTable Rpp32f row numbers in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) for every pixel in the input batch of images (1D tensor of size width * height *
 * batchSize) \param [in] colRemapTable Rpp32f column numbers in HIP memory (for HIP backend) or
 * HOST memory (for HOST backend) for every pixel in the input batch of images (1D tensor of size
 * width * height * batchSize) \param [in] tableDescPtr table tensor descriptor (Restrictions -
 * numDims = 4, offsetInBytes >= 0, dataType = F32, layout = NHWC, c = 1) \param [in]
 * cameraMatrixTensor contains camera intrinsic parameters required to compute lens corrected image.
 * (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend), of size 9
 * * batchSize) \param [in] distortionCoeffsTensor contains distortion coefficients required to
 * compute lens corrected image. (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory
 * (for HOST backend), of size 8 * batchSize) \param [in] roiTensorPtrSrc ROI data in HIP memory
 * (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor
 * of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x,
 * lt.y, rb.x, rb.y)) \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or
 * RppBackend::RPP_HIP_BACKEND) \return A <tt> \ref RppStatus</tt> enumeration. \retval RPP_SUCCESS
 * Successful completion. \retval RPP_ERROR* Unsuccessful completion. \ingroup
 * group_tensor_geometric
 */
RppStatus rppt_lens_correction(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr, Rpp32f* rowRemapTable, Rpp32f* colRemapTable,
                               RpptDescPtr tableDescPtr, Rpp32f* cameraMatrixTensor,
                               Rpp32f* distortionCoeffsTensor, RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType, rppHandle_t rppHandle,
                               RppBackend executionBackend);

/*! \brief Transpose Generic augmentation on HIP/HOST backend
 * \details The transpose augmentation performs an input-permutation based transpose on a generic ND
 * Tensor. \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) \param [in] srcGenericDescPtr source tensor descriptor \param [out] dstPtr destination
 * tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend) \param [in]
 * dstGenericDescPtr destination tensor descriptor \param [in] permTensor permutation tensor for
 * transpose operation in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] roiTensor ROI data in pinned / HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) for each element in source tensor (tensor of batchSize * number of dimensions * 2
 * values) \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt> \param
 * [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or
 * RppBackend::RPP_HIP_BACKEND) \return A <tt> \ref RppStatus</tt> enumeration. \retval RPP_SUCCESS
 * Successful completion. \retval RPP_ERROR* Unsuccessful completion. \ingroup
 * group_tensor_geometric
 */
RppStatus rppt_transpose(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr,
                         RpptGenericDescPtr dstGenericDescPtr, Rpp32u* permTensor,
                         Rpp32u* roiTensor, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Warp perspective augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The warp perspective performs perspective transformations for a batch of RGB(3 channel)
 * / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html geometric_augmentations_warp_perspective_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4,
 * offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3) \param [out] dstPtr
 * destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend) \param [in]
 * dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0,
 * dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr) \param [in]
 * perspectiveTensor perspective matrix values for transformation calculation (2D tensor in pinned /
 * HIP memory (for HIP backend) or HOST memory (for HOST backend), of size batchSize * 9 for each
 * image in batch) \param [in] interpolationType Interpolation type used
 * (RpptInterpolationType::BILINEAR or RpptRoiType::NEAREST_NEIGHBOR) \param [in] roiTensorPtrSrc
 * ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in
 * source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth,
 * roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y)) \param [in] roiType ROI type used (RpptRoiType::XYWH
 * or RpptRoiType::LTRB) \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref
 * rppCreate()</tt> \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND
 * or RppBackend::RPP_HIP_BACKEND) \return A <tt> \ref RppStatus</tt> enumeration. \retval
 * RPP_SUCCESS Successful completion. \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_warp_perspective(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                                RpptDescPtr dstDescPtr, Rpp32f* perspectiveTensor,
                                RpptInterpolationType interpolationType, RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType, rppHandle_t rppHandle,
                                RppBackend executionBackend);

/*! \brief Concat Generic augmentation on HIP/HOST backend
 * \details Concatenates two 2D, 3D or ND tensors in HIP/HOST memory along a specified axis.
 * It is optimized for 2D and 3D tensors, ensuring that all dimensions except the concatenation axis
 * must match. \param [in] srcPtr1 source tensor in HIP memory (for HIP backend) or HOST memory (for
 * HOST backend) \param [in] srcPtr2 source tensor in HIP memory (for HIP backend) or HOST memory
 * (for HOST backend) \param [in] srcPtr1GenericDescPtr source tensor descriptor for the input
 * tensor srcPtr1 \param [in] srcPtr2GenericDescPtr source tensor descriptor for the input tensor
 * srcPtr2 \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory
 * (for HOST backend) \param [in] dstGenericDescPtr destination tensor descriptor \param [in]
 * axisMask axis along which concat needs to be done \param [in] srcPtr1roiTensor values to
 * represent dimensions of input tensor srcPtr1 in pinned / HIP memory (for HIP backend) or HOST
 * memory (for HOST backend) \param [in] srcPtr2roiTensor values to represent dimensions of input
 * tensor srcPtr2 in pinned / HIP memory (for HIP backend) or HOST memory (for HOST backend) \param
 * [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreateWithBatchSize()</tt> \param
 * [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or
 * RppBackend::RPP_HIP_BACKEND) \return A <tt> \ref RppStatus</tt> enumeration. \retval RPP_SUCCESS
 * Successful completion. \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_concat(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptGenericDescPtr srcPtr1GenericDescPtr,
                      RpptGenericDescPtr srcPtr2GenericDescPtr, RppPtr_t dstPtr,
                      RpptGenericDescPtr dstGenericDescPtr, Rpp32u axisMask,
                      Rpp32u* srcPtr1roiTensor, Rpp32u* srcPtr2roiTensor, rppHandle_t rppHandle,
                      RppBackend executionBackend);

/*!
 * \brief JPEG Compression Distortion on HIP/HOST backend
 * \details This function simulates JPEG compression distortion on an image tensor.
 *          It introduces artifacts seen in lossy JPEG compression by converting the image to the
 * frequency domain using the Discrete Cosine Transform (DCT), applying quantization, and then
 * reconstructing the image using the inverse DCT (IDCT). This process introduces
 * compression-related distortions similar to those in JPEG images. \image html img150x150.png
 * Sample Input \image html geometric_augmentations_jpeg_compression_distortion_img_150x150.png
 * Sample Output \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory
 * (for HOST backend) \param [in] srcGenericDescPtr source tensor descriptor \param [out] dstPtr
 * destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend) \param [in]
 * dstGenericDescPtr destination tensor descriptor \param [in] qualityTensor JPEG quality factor
 * controlling the level of compression (1D tensor in pinned / HIP memory (for HIP backend) or HOST
 * memory (for HOST backend)). Valid range is 1 to 99 (inclusive). \param [in] roiTensorPtrSrc ROI
 * data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in source
 * tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight)
 * or LTRB(lt.x, lt.y, rb.x, rb.y)) \param [in] roiType ROI type used (RpptRoiType::XYWH or
 * RpptRoiType::LTRB) \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref
 * rppCreate()</tt> \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND
 * or RppBackend::RPP_HIP_BACKEND) \return A <tt> \ref RppStatus</tt> enumeration. \retval
 * RPP_SUCCESS Successful completion. \retval RPP_ERROR* Unsuccessful completion. \ingroup
 * group_tensor_geometric
 */
RppStatus rppt_jpeg_compression_distortion(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                                           RpptDescPtr dstDescPtr, Rpp32s* qualityTensor,
                                           RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                           rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Fisheye augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The Fisheye augmentation applies fisheye effect for a batch of RGB(3 channel) /
 * greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html geometric_augmentations_fisheye_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4,
 * offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3) \param [out] dstPtr
 * destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend) \param [in]
 * dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0,
 * dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr) \param [in]
 * roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for
 * each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y,
 * roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y)) \param [in] roiType ROI type used
 * (RpptRoiType::XYWH or RpptRoiType::LTRB) \param [in] rppHandle RPP HIP/HOST handle created with
 * <tt>\ref rppCreate()</tt> \param [in] executionBackend backend for execution
 * (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND) \return A <tt> \ref RppStatus</tt>
 * enumeration. \retval RPP_SUCCESS Successful completion. \retval RPP_ERROR* Unsuccessful
 * completion. \ingroup group_tensor_geometric
 */
RppStatus rppt_fisheye(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                       RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                       rppHandle_t rppHandle, RppBackend executionBackend);

/*! @}
 */

#ifdef __cplusplus
}
#endif
#endif  // RPPT_TENSOR_GEOMETRIC_AUGMENTATIONS_H
