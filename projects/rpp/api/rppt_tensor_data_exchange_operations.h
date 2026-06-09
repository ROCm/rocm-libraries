/*
MIT License

Copyright (c) 2019 - 2026 Advanced Micro Devices, Inc.

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

#ifndef RPPT_DATA_EXCHANGE_OPERATIONS_H
#define RPPT_DATA_EXCHANGE_OPERATIONS_H

#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \file
 * \brief RPPT Tensor Operations - Data Exchange Operations.
 * \defgroup group_rppt_tensor_data_exchange_operations RPPT Tensor Operations - Data Exchange Operations.
 * \brief RPPT Tensor Operations - Data Exchange Operations.
 */

/*! \addtogroup group_rppt_tensor_data_exchange_operations
 * @{
 */

/*! \brief Copy operation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The copy operation runs a buffer copy for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html data_exchange_operations_copy_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_copy(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Channel permute operation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details This function performs one of six possible channel permutations (R-G-B, R-B-G, G-R-B, G-B-R, B-R-G, B-G-R)
 * for an image in a batch of RGB(3 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html data_exchange_operations_channel_permute_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] permutationTensor A tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend) specifying the channel permutation for each image. Size: 3 × srcDescPtr->n. Each value must satisfy: 0 ≤ permutationTensor[i] ≤ 2.
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_channel_permute(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u *permutationTensor, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief Color to greyscale operation on HIP/HOST backend for a NCHW/NHWC layout tensor
 * \details The color to greyscale operation runs for a batch of RGB(3 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.png Sample Input
 * \image html data_exchange_operations_color_to_greyscale_img150x150.png Sample Output
 * \param [in] srcPtr source tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 3)
 * \param [out] dstPtr destination tensor in HIP memory (for HIP backend) or HOST memory (for HOST backend)
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] srcSubpixelLayout A RpptSubpixelLayout type enum to specify source subpixel layout (RGBtype or BGRtype)
 * \param [in] rppHandle RPP HIP/HOST handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend backend for execution (RppBackend::RPP_HOST_BACKEND or RppBackend::RPP_HIP_BACKEND)
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_color_to_greyscale(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptSubpixelLayout srcSubpixelLayout, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief YUV to RGB color conversion on HIP backend (NV12 8-bit only)
 * \details Converts semi-planar NV12 (separate Y and interleaved UV planes) to packed RGB24.<br>
 * - Source: srcYPtr = luma plane, srcUVPtr = interleaved UV
 * - src_y_pitch / src_uv_pitch: row strides in bytes
 * - Destination: packed RGB at dstPtr (RGB24, Rpp8u) with row stride dst_pitch bytes.
 * - Supported execution backend: RppBackend::RPP_HIP_BACKEND only.
 * \param [in] srcYPtr pointer to Y plane in HIP memory
 * \param [in] srcUVPtr pointer to interleaved UV plane in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (dataType must be U8)
 * \param [out] dstPtr pointer to RGB output buffer in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (dataType must be U8)
 * \param [in] src_y_pitch row pitch of Y plane in bytes
 * \param [in] src_uv_pitch row pitch of UV plane in bytes
 * \param [in] dst_pitch row pitch of RGB output in bytes
 * \param [in] width image width in pixels
 * \param [in] height image height in pixels
 * \param [in] col_standard Luma/matrix family: \ref RpptColorStandard (unknown values use BT.709).
 * \param [in] color_range Luma range: \ref RpptColorRange_STUDIO or \ref RpptColorRange_FULL (other values behave like studio).
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend must be RppBackend::RPP_HIP_BACKEND
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion (e.g. RPP_ERROR_INCOMPATIBLE_BACKEND if executionBackend is not HIP).
 */
RppStatus rppt_yuv_to_rgb(RppPtr_t srcYPtr, RppPtr_t srcUVPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u src_y_pitch, Rpp32u src_uv_pitch, Rpp32u dst_pitch, Rpp32u width, Rpp32u height, RpptColorStandard col_standard, RpptColorRange color_range, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief YUV to RGB color conversion with cubic vertical chroma upsampling on HIP backend (NV12 8-bit only)
 * \details Converts semi-planar NV12 (separate Y and interleaved UV planes) to packed RGB24,
 * using Mitchell-Netravali cubic interpolation (B=0, C=0.6) to vertically upsample chroma.
 * Odd luma rows pass through chroma unchanged (identity); even luma rows use a symmetric 4-tap
 * filter. Horizontal chroma upsampling remains nearest-neighbor.
 * - Source: srcYPtr = luma plane, srcUVPtr = interleaved UV
 * - src_y_pitch / src_uv_pitch: row strides in bytes
 * - Destination: packed RGB at dstPtr (RGB24, Rpp8u) with row stride dst_pitch bytes.
 * - Supported execution backend: RppBackend::RPP_HIP_BACKEND only.
 * \param [in] srcYPtr pointer to Y plane in HIP memory
 * \param [in] srcUVPtr pointer to interleaved UV plane in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (dataType must be U8)
 * \param [out] dstPtr pointer to RGB output buffer in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (dataType must be U8)
 * \param [in] src_y_pitch row pitch of Y plane in bytes
 * \param [in] src_uv_pitch row pitch of UV plane in bytes
 * \param [in] dst_pitch row pitch of RGB output in bytes
 * \param [in] width image width in pixels
 * \param [in] height image height in pixels
 * \param [in] col_standard Luma/matrix family: \ref RpptColorStandard (unknown values use BT.709).
 * \param [in] color_range Luma range: \ref RpptColorRange_STUDIO or \ref RpptColorRange_FULL (other values behave like studio).
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend must be RppBackend::RPP_HIP_BACKEND
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion (e.g. RPP_ERROR_INCOMPATIBLE_BACKEND if executionBackend is not HIP).
 */
RppStatus rppt_yuv_to_rgb_cubic_v(RppPtr_t srcYPtr, RppPtr_t srcUVPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u src_y_pitch, Rpp32u src_uv_pitch, Rpp32u dst_pitch, Rpp32u width, Rpp32u height, RpptColorStandard col_standard, RpptColorRange color_range, rppHandle_t rppHandle, RppBackend executionBackend);

/*! \brief YUV to RGB color conversion with linear vertical chroma upsampling on HIP backend (NV12 8-bit only)
 * \details Converts semi-planar NV12 (separate Y and interleaved UV planes) to packed RGB24,
 * using linear interpolation to vertically upsample chroma. Odd luma rows pass through chroma
 * unchanged (identity); even luma rows average the two nearest chroma rows (frac=0.5).
 * Horizontal chroma upsampling remains nearest-neighbor.<br>
 * - Source: srcYPtr = luma plane, srcUVPtr = interleaved UV
 * - src_y_pitch / src_uv_pitch: row strides in bytes
 * - Destination: packed RGB at dstPtr (RGB24, Rpp8u) with row stride dst_pitch bytes.
 * - Supported execution backend: RppBackend::RPP_HIP_BACKEND only.
 * \param [in] srcYPtr pointer to Y plane in HIP memory
 * \param [in] srcUVPtr pointer to interleaved UV plane in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (dataType must be U8)
 * \param [out] dstPtr pointer to RGB output buffer in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (dataType must be U8)
 * \param [in] src_y_pitch row pitch of Y plane in bytes
 * \param [in] src_uv_pitch row pitch of UV plane in bytes
 * \param [in] dst_pitch row pitch of RGB output in bytes
 * \param [in] width image width in pixels
 * \param [in] height image height in pixels
 * \param [in] col_standard Luma/matrix family: \ref RpptColorStandard (unknown values use BT.709).
 * \param [in] color_range Luma range: \ref RpptColorRange_STUDIO or \ref RpptColorRange_FULL (other values behave like studio).
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \param [in] executionBackend must be RppBackend::RPP_HIP_BACKEND
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion (e.g. RPP_ERROR_INCOMPATIBLE_BACKEND if executionBackend is not HIP).
 */
RppStatus rppt_yuv_to_rgb_linear_v(RppPtr_t srcYPtr, RppPtr_t srcUVPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u src_y_pitch, Rpp32u src_uv_pitch, Rpp32u dst_pitch, Rpp32u width, Rpp32u height, RpptColorStandard col_standard, RpptColorRange color_range, rppHandle_t rppHandle, RppBackend executionBackend);

/*! @}
 */

#ifdef __cplusplus
}
#endif
#endif // RPPT_DATA_EXCHANGE_OPERATIONS_H
