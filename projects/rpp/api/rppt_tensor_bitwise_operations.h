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

#ifndef RPPT_TENSOR_BITWISE_OPERATIONS_H
#define RPPT_TENSOR_BITWISE_OPERATIONS_H

#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \file
 * \brief RPPT Tensor Operations - Bitwise Operations.
 * \defgroup group_rppt_tensor_bitwise_operations RPPT Tensor Operations - Bitwise Operations.
 * \brief RPPT Tensor Operations - Bitwise Operations.
 */

/*! \addtogroup group_rppt_tensor_bitwise_operations
 * @{
 */

/*! \brief Bitwise AND computation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details This function computes bitwise AND of corresponding pixels for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 *          srcPtr depth ranges - Rpp8u (0 to 255).
 *          dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input1
 * \image html img150x150_2.png Sample Input2
 * \image html bitwise_operations_bitwise_and_img150x150.png Sample Output
 * \param [in] srcPtr1 source1 tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcPtr2 source2 tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_bitwise_and(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Bitwise XOR computation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details This function computes bitwise XOR of corresponding pixels for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 *          srcPtr depth ranges - Rpp8u (0 to 255).
 *          dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input1
 * \image html img150x150_2.png Sample Input2
 * \image html bitwise_operations_bitwise_xor_img150x150.png Sample Output
 * \param [in] srcPtr1 source1 tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcPtr2 source2 tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreateWithBatchSize()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_bitwise_xor(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Bitwise OR computation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details This function computes bitwise OR of corresponding pixels for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 *          srcPtr depth ranges - Rpp8u (0 to 255).
 *          dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input1
 * \image html img150x150_2.png Sample Input2
 * \image html bitwise_operations_bitwise_or_img150x150.png Sample Output
 * \param [in] srcPtr1 source1 tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcPtr2 source2 tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_bitwise_or(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Bitwise NOT computation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details This function computes bitwise NOT of corresponding pixels for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 *          srcPtr depth ranges - Rpp8u (0 to 255).
 *          dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html bitwise_operations_bitwise_not_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory (for HIP backend) or HOST memory (for HOST backend), for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreateWithBatchSize()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_bitwise_not(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Bitwise AND Generic augmentation on HIP/HOST backend with broadcasting support
 * \details This function computes bitwise AND between two 2D, 3D or ND tensors with broadcasting support
 *          Broadcasting is permitted when, for each axis, the corresponding dimensions of the input tensors are either equal or one of them is 1
 * \param [in] srcPtr1 source tensor memory in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcPtr2 source tensor memory in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcPtr1GenericDescPtr source tensor descriptor for the input tensor srcPtr1
 * \param [in] srcPtr2GenericDescPtr source tensor descriptor for the input tensor srcPtr2
 * \param [out] dstPtr destination tensor memory in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstGenericDescPtr destination tensor descriptor
 * \param [in] broadcastMode enum used to represent if broadcast support is enabled or disabled for the binary operation (can only be disabled if input tensors are of same shape).
 * \param [in] srcPtr1roiTensor values to represent dimensions of input tensor srcPtr1
 * \param [in] srcPtr2roiTensor values to represent dimensions of input tensor srcPtr2
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_tensor_and_tensor(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptGenericDescPtr srcGenericDescPtr1, RpptGenericDescPtr srcGenericDescPtr2, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, RpptBroadcastMode broadcastMode, Rpp32u *roiTensor1, Rpp32u *roiTensor2, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Bitwise OR Generic augmentation on HIP/HOST backend with broadcasting support
 * \details This function computes bitwise OR between two 2D, 3D or ND tensors with broadcasting support
 *          Broadcasting is permitted when, for each axis, the corresponding dimensions of the input tensors are either equal or one of them is 1
 * \param [in] srcPtr1 source tensor memory in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcPtr2 source tensor memory in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcPtr1GenericDescPtr source tensor descriptor for the input tensor srcPtr1
 * \param [in] srcPtr2GenericDescPtr source tensor descriptor for the input tensor srcPtr2
 * \param [out] dstPtr destination tensor memory in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstGenericDescPtr destination tensor descriptor
 * \param [in] broadcastMode enum used to represent if broadcast support is enabled or disabled for the binary operation (can only be disabled if input tensors are of same shape).
 * \param [in] srcPtr1roiTensor values to represent dimensions of input tensor srcPtr1
 * \param [in] srcPtr2roiTensor values to represent dimensions of input tensor srcPtr2
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_tensor_or_tensor(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptGenericDescPtr srcGenericDescPtr1, RpptGenericDescPtr srcGenericDescPtr2, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, RpptBroadcastMode broadcastMode, Rpp32u *roiTensor1, Rpp32u *roiTensor2, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Bitwise XOR Generic augmentation on HIP/HOST backend with broadcasting support
 * \details This function computes bitwise XOR between two 2D, 3D or ND tensors with broadcasting support
 *          Broadcasting is permitted when, for each axis, the corresponding dimensions of the input tensors are either equal or one of them is 1
 * \param [in] srcPtr1 source tensor memory in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcPtr2 source tensor memory in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcPtr1GenericDescPtr source tensor descriptor for the input tensor srcPtr1
 * \param [in] srcPtr2GenericDescPtr source tensor descriptor for the input tensor srcPtr2
 * \param [out] dstPtr destination tensor memory in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstGenericDescPtr destination tensor descriptor
 * \param [in] broadcastMode enum used to represent if broadcast support is enabled or disabled for the binary operation (can only be disabled if input tensors are of same shape).
 * \param [in] srcPtr1roiTensor values to represent dimensions of input tensor srcPtr1
 * \param [in] srcPtr2roiTensor values to represent dimensions of input tensor srcPtr2
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_tensor_xor_tensor(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptGenericDescPtr srcGenericDescPtr1, RpptGenericDescPtr srcGenericDescPtr2, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, RpptBroadcastMode broadcastMode, Rpp32u *roiTensor1, Rpp32u *roiTensor2, rppHandle_t rppHandle, RppBackend executionBackend);
/*! @}
 */

#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_BITWISE_OPERATIONS_H
