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

#ifndef RPPT_TENSOR_STATISTICAL_OPERATIONS_H
#define RPPT_TENSOR_STATISTICAL_OPERATIONS_H

#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \file
 * \brief RPPT Tensor Operations - Statistical Operations.
 * \defgroup group_rppt_tensor_statistical_operations RPPT Tensor Operations - Statistical
 * Operations. \brief RPPT Tensor Operations - Statistical Operations.
 */

/*! \addtogroup group_rppt_tensor_statistical_operations
 * @{
 */

/*! \brief Tensor sum operation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The tensor sum is a reduction operation that finds the channel-wise (R sum / G sum / B
 * sum) and total sum for each image in a batch of RGB(3 channel) / greyscale(1 channel) images with
 * an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4,
 * offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3) \param [out]
 * tensorSumArr destination array in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] tensorSumArrLength length of provided destination array (Restrictions - if
 * srcDescPtr->c == 1 then tensorSumArrLength >= srcDescPtr->n, and if srcDescPtr->c == 3 then
 * tensorSumArrLength >= srcDescPtr->n * 4) \param [in] roiTensorPtrSrc ROI data in HIP memory (for
 * HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor of
 * size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y,
 * rb.x, rb.y)) | (Restrictions - roiTensorPtrSrc[i].xywhROI.roiWidth <= 3840 and
 * roiTensorPtrSrc[i].xywhROI.roiHeight <= 2160) \param [in] roiType ROI type used
 * (RpptRoiType::XYWH or RpptRoiType::LTRB) \param [in] rppHandle RPP HIP/HOST handle created with
 * <tt>\ref rppCreate()</tt> \param [in] executionBackend execution backend to run the operation on
 * (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND) \return A <tt> \ref RppStatus</tt>
 * enumeration. \retval RPP_SUCCESS Successful completion. \retval RPP_ERROR* Unsuccessful
 * completion.
 */
RppStatus rppt_tensor_sum(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t tensorSumArr,
                          Rpp32u tensorSumArrLength, RpptROIPtr roiTensorPtrSrc,
                          RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Tensor min operation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The tensor min is a reduction operation that finds the channel-wise (R min / G min / B
 * min) and overall min for each image in a batch of RGB(3 channel) / greyscale(1 channel) images
 * with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4,
 * offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3) \param [out] minArr
 * destination array in HIP memory (for HIP backend) or HOST memory (for HOST backend) \param [in]
 * minArrLength length of provided destination array (Restrictions - if srcDescPtr->c == 1 then
 * minArrLength >= srcDescPtr->n, and if srcDescPtr->c == 3 then minArrLength >= srcDescPtr->n * 4)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format -
 * XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y)) | (Restrictions -
 * roiTensorPtrSrc[i].xywhROI.roiWidth <= 3840 and roiTensorPtrSrc[i].xywhROI.roiHeight <= 2160)
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend execution backend to run the operation on
 * (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND) \return A <tt> \ref RppStatus</tt>
 * enumeration. \retval RPP_SUCCESS Successful completion. \retval RPP_ERROR* Unsuccessful
 * completion.
 */
RppStatus rppt_tensor_min(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t minArr,
                          Rpp32u minArrLength, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                          rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Tensor max operation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The tensor max is a reduction operation that finds the channel-wise (R max / G max / B
 * max) and overall max for each image in a batch of RGB(3 channel) / greyscale(1 channel) images
 * with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4,
 * offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3) \param [out] maxArr
 * destination array in HIP memory (for HIP backend) or HOST memory (for HOST backend) \param [in]
 * maxArrLength length of provided destination array (Restrictions - if srcDescPtr->c == 1 then
 * maxArrLength >= srcDescPtr->n, and if srcDescPtr->c == 3 then maxArrLength >= srcDescPtr->n * 4)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format -
 * XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y)) | (Restrictions -
 * roiTensorPtrSrc[i].xywhROI.roiWidth <= 3840 and roiTensorPtrSrc[i].xywhROI.roiHeight <= 2160)
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend execution backend to run the operation on
 * (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND) \return A <tt> \ref RppStatus</tt>
 * enumeration. \retval RPP_SUCCESS Successful completion. \retval RPP_ERROR* Unsuccessful
 * completion.
 */
RppStatus rppt_tensor_max(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t maxArr,
                          Rpp32u maxArrLength, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                          rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Normalize Generic augmentation on HIP/HOST backend
 * \details Normalizes the input generic ND buffer by removing the mean and dividing by the standard
 * deviation for a given ND Tensor. Supports u8->f32, i8->f32, f16->f16 and f32->f32 datatypes. Also
 * has toggle variant(NHWC->NCHW) support for 3D. \param [in] srcPtr source tensor in HIP memory
 * (for HIP backend) or HOST memory (for HOST backend) \param [in] srcGenericDescPtr source tensor
 * descriptor \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory
 * (for HOST backend) \param [in] dstGenericDescPtr destination tensor descriptor \param [in]
 * axisMask axis along which normalization needs to be done \param [in] meanTensor values to be
 * subtracted from input (in pinned / HIP memory (for HIP backend) or HOST memory (for HOST
 * backend)) \param [in] stdDevTensor standard deviation values to scale the input (in pinned / HIP
 * memory (for HIP backend) or HOST memory (for HOST backend)) \param [in] computeMeanStddev flag to
 * represent internal computation of mean, stddev (Wherein 0th bit used to represent computeMean and
 * 1st bit for computeStddev, 0- Externally provided) \param [in] scale value to be multiplied with
 * data after subtracting from mean \param [in] shift value to be added finally \param [in]
 * roiTensor values to represent dimensions of input tensor (in pinned / HIP memory (for HIP
 * backend) or HOST memory (for HOST backend)) \param [in] rppHandle RPP HIP/HOST handle created
 * with <tt>\ref rppCreate()</tt> \param [in] executionBackend execution backend to run the
 * operation on (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND) \return A <tt> \ref
 * RppStatus</tt> enumeration. \retval RPP_SUCCESS Successful completion. \retval RPP_ERROR*
 * Unsuccessful completion.
 */
RppStatus rppt_normalize(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr,
                         RpptGenericDescPtr dstGenericDescPtr, Rpp32u axisMask, Rpp32f* meanTensor,
                         Rpp32f* stdDevTensor, Rpp8u computeMeanStddev, Rpp32f scale, Rpp32f shift,
                         Rpp32u* roiTensor, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Tensor mean operation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The tensor mean is a reduction operation that finds the channel-wise (R mean / G mean /
 * B mean) and total mean for each image in a batch of RGB(3 channel) / greyscale(1 channel) images
 * with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4,
 * offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3) \param [out]
 * tensorMeanArr destination array in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] tensorMeanArrLength length of provided destination array (Restrictions - if
 * srcDescPtr->c == 1 then tensorMeanArrLength = srcDescPtr->n, and if srcDescPtr->c == 3 then
 * tensorMeanArrLength = srcDescPtr->n * 4) \param [in] roiTensorPtrSrc ROI data in HIP memory (for
 * HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor of
 * size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y,
 * rb.x, rb.y)) | (Restrictions - roiTensorPtrSrc[i].xywhROI.roiWidth <= 3840 and
 * roiTensorPtrSrc[i].xywhROI.roiHeight <= 2160) \param [in] roiType ROI type used
 * (RpptRoiType::XYWH or RpptRoiType::LTRB) \param [in] rppHandle RPP HIP/HOST handle created with
 * <tt>\ref rppCreate()</tt> \param [in] executionBackend execution backend to run the operation on
 * (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND) \return A <tt> \ref RppStatus</tt>
 * enumeration. \retval RPP_SUCCESS Successful completion. \retval RPP_ERROR* Unsuccessful
 * completion.
 */
RppStatus rppt_tensor_mean(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t tensorMeanArr,
                           Rpp32u tensorMeanArrLength, RpptROIPtr roiTensorPtrSrc,
                           RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Tensor stddev operation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The tensor stddev is a reduction operation that finds the channel-wise (R stddev / G
 * stddev / B stddev) and total standard deviation for each image with respect to meanTensor
 * passed.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4,
 * offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3) \param [out]
 * tensorStddevArr destination array in HIP memory (for HIP backend) or HOST memory (for HOST
 * backend) \param [in] tensorStddevArrLength length of provided destination array (Restrictions -
 * if srcDescPtr->c == 1 then tensorStddevArrLength = srcDescPtr->n, and if srcDescPtr->c == 3 then
 * tensorStddevArrLength = srcDescPtr->n * 4) \param [in] meanTensor mean values for stddev
 * calculation (1D tensor in pinned / HIP memory (for HIP backend) or HOST memory (for HOST
 * backend), of size batchSize * 4 in format (MeanR, MeanG, MeanB, MeanImage) for each image in
 * batch) \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for
 * HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format
 * - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y)) | (Restrictions -
 * roiTensorPtrSrc[i].xywhROI.roiWidth <= 3840 and roiTensorPtrSrc[i].xywhROI.roiHeight <= 2160)
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend execution backend to run the operation on
 * (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND) \return A <tt> \ref RppStatus</tt>
 * enumeration. \retval RPP_SUCCESS Successful completion. \retval RPP_ERROR* Unsuccessful
 * completion.
 */
RppStatus rppt_tensor_stddev(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t tensorStddevArr,
                             Rpp32u tensorStddevArrLength, Rpp32f* meanTensor,
                             RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle,
                             RppBackend executionBackend);

/*! \brief Threshold augmentation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The Threshold augmentation outputs a black/white binary mask image, based on whether or
 not each pixel is within the user-specified pixel-range bounds, for a batch of RGB(3 channel) /
 greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.<br>
 * Note: Returns a black image for below 2 cases:
 *       1. If the minimum cutoff value greater than the maximum cutoff value for the given input in
 a batch.<br>
 *       2. Values provided for minimum cutoff value, maximum cutoff value are beyond the below
 specified min and max values.<br> Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128
 to 127).
 * \image html img150x150.png Sample Input
 * \image html statistical_operations_threshold_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0,
 dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST
 backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes
 >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] minTensor minimum cutoff value (1D tensor in pinned / HIP memory (for HIP backend) or
 HOST memory (for HOST backend), of size batchSize * channels) - minTensor ranges - Rpp8u (0 to
 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * \param [in] maxTensor maximum cutoff value (1D tensor in pinned / HIP memory (for HIP backend) or
 HOST memory (for HOST backend), of size batchSize * channels) - maxTensor ranges - Rpp8u (0 to
 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST
 backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format -
 XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreateWithBatchSize()</tt>
 * \param [in] executionBackend execution backend to run the augmentation on
 (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_threshold(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                         RpptDescPtr dstDescPtr, Rpp32f* minTensor, Rpp32f* maxTensor,
                         RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle,
                         RppBackend executionBackend);

/*! @}
 */

#ifdef __cplusplus
}
#endif
#endif  // RPPT_TENSOR_STATISTICAL_OPERATIONS_H
