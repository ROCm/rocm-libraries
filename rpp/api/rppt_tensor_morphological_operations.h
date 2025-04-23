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

#ifndef RPPT_TENSOR_MORPHOLOGICAL_OPERATIONS_H
#define RPPT_TENSOR_MORPHOLOGICAL_OPERATIONS_H

#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \file
 * \brief RPPT Tensor Operations - Morphological Operations.
 * \defgroup group_rppt_tensor_morphological_operations RPPT Tensor Operations - Morphological Operations.
 * \brief RPPT Tensor Operations - Morphological Operations.
 */

/*! \addtogroup group_rppt_tensor_morphological_operations
 * @{
 */

#ifdef GPU_SUPPORT
/*! \brief Erode augmentation on HIP backend for a NCHW/NHWC layout tensor
 * \details The erode augmentation runs for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html morphological_operations_erode_kSize3_img150x150.png Sample 3x3 Output
 * \image html morphological_operations_erode_kSize5_img150x150.png Sample 5x5 Output
 * \image html morphological_operations_erode_kSize7_img150x150.png Sample 7x7 Output
 * \image html morphological_operations_erode_kSize9_img150x150.png Sample 9x9 Output
 * \param [in] srcPtr source tensor in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] kernelSize kernel size for box filter (a single Rpp32u odd number with kernelSize = 3/5/7/9 that applies to all images in the batch)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_erode_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u kernelSize, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

#ifdef GPU_SUPPORT
/*! \brief Dilate augmentation on HIP backend for a NCHW/NHWC layout tensor
 * \details The dilate augmentation runs for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html morphological_operations_dilate_kSize3_img150x150.png Sample 3x3 Output
 * \image html morphological_operations_dilate_kSize5_img150x150.png Sample 5x5 Output
 * \image html morphological_operations_dilate_kSize7_img150x150.png Sample 7x7 Output
 * \image html morphological_operations_dilate_kSize9_img150x150.png Sample 9x9 Output
 * \param [in] srcPtr source tensor in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] kernelSize kernel size for box filter (a single Rpp32u odd number with kernelSize = 3/5/7/9 that applies to all images in the batch)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_dilate_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u kernelSize, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! @}
 */

#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_MORPHOLOGICAL_OPERATIONS_H
