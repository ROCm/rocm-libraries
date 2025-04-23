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

#ifndef RPPT_TENSOR_FILTER_AUGMENTATIONS_H
#define RPPT_TENSOR_FILTER_AUGMENTATIONS_H

#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \file
 * \brief RPPT Tensor Operations - Filter Augmentations.
 * \defgroup group_rppt_tensor_filter_augmentations RPPT Tensor Operations - Filter Augmentations.
 * \brief RPPT Tensor Operations - Filter Augmentations.
 */

/*! \addtogroup group_rppt_tensor_filter_augmentations
 * @{
 */

/*! \brief Box Filter augmentation on HOST backend for a NCHW/NHWC layout tensor
 * \details The box filter augmentation runs for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html filter_augmentations_box_filter_kSize3_img150x150.png Sample 3x3 Output
 * \image html filter_augmentations_box_filter_kSize5_img150x150.png Sample 5x5 Output
 * \image html filter_augmentations_box_filter_kSize7_img150x150.png Sample 7x7 Output
 * \image html filter_augmentations_box_filter_kSize9_img150x150.png Sample 9x9 Output
 * \param [in] srcPtr source tensor in HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HOST memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] kernelSize kernel size for box filter (a single Rpp32u number with kernelSize > 0 that applies to all images in the batch. kernelSize = 3/5/7/9 are optimized to run faster)
 * \param [in] roiTensorPtrSrc ROI data in HOST memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_box_filter_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u kernelSize, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Box Filter augmentation on HIP backend for a NCHW/NHWC layout tensor
 * \details The box filter augmentation runs for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html filter_augmentations_box_filter_kSize3_img150x150.png Sample 3x3 Output
 * \image html filter_augmentations_box_filter_kSize5_img150x150.png Sample 5x5 Output
 * \image html filter_augmentations_box_filter_kSize7_img150x150.png Sample 7x7 Output
 * \image html filter_augmentations_box_filter_kSize9_img150x150.png Sample 9x9 Output
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
RppStatus rppt_box_filter_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u kernelSize, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Gaussian Filter augmentation on HOST backend for a NCHW/NHWC layout tensor
 * \details The Gaussian filter augmentation runs for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html filter_augmentations_gaussian_filter_kSize3_img150x150.png Sample 3x3 Output
 * \image html filter_augmentations_gaussian_filter_kSize5_img150x150.png Sample 5x5 Output
 * \image html filter_augmentations_gaussian_filter_kSize7_img150x150.png Sample 7x7 Output
 * \image html filter_augmentations_gaussian_filter_kSize9_img150x150.png Sample 9x9 Output
 * \param [in] srcPtr source tensor in HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HOST memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] stdDevTensor stdDev values for gaussian calculation (1D tensor in HOST memory, of size batchSize, for each image in batch)
 * \param [in] kernelSize kernel size for gaussian filter (a single Rpp32u number with kernelSize > 0 that applies to all images in the batch. kernelSize = 3/5/7/9 are optimized to run faster)
 * \param [in] roiTensorPtrSrc ROI data in HOST memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_gaussian_filter_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *stdDevTensor, Rpp32u kernelSize, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
/*! \brief Gaussian Filter augmentation on HIP backend for a NCHW/NHWC layout tensor
 * \details The gaussian filter augmentation runs for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html filter_augmentations_gaussian_filter_kSize3_img150x150.png Sample 3x3 Output
 * \image html filter_augmentations_gaussian_filter_kSize5_img150x150.png Sample 5x5 Output
 * \image html filter_augmentations_gaussian_filter_kSize7_img150x150.png Sample 7x7 Output
 * \image html filter_augmentations_gaussian_filter_kSize9_img150x150.png Sample 9x9 Output
 * \param [in] srcPtr source tensor in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] stdDevTensor stdDev values for gaussian calculation (1D tensor in pinned/HOST memory, of size batchSize, for each image in batch)
 * \param [in] kernelSize kernel size for gaussian filter (a single Rpp32u odd number with kernelSize = 3/5/7/9 that applies to all images in the batch)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_gaussian_filter_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *stdDevTensor, Rpp32u kernelSize, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! @}
 */

#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_FILTER_AUGMENTATIONS_H
