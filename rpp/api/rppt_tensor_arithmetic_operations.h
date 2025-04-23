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

#ifndef RPPT_TENSOR_ARITHMETIC_OPERATIONS_H
#define RPPT_TENSOR_ARITHMETIC_OPERATIONS_H

#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \file
 * \brief RPPT Tensor Operations - Arithmetic Operations.
 * \defgroup group_rppt_tensor_arithmetic_operations RPPT Tensor Operations - Arithmetic Operations.
 * \brief RPPT Tensor Operations - Arithmetic Operations.
 */

/*! \addtogroup group_rppt_tensor_arithmetic_operations
 * @{
 */

/*! \brief Fused multiply add scalar augmentation on HOST backend
 * \details This function performs the fmadd operation on a batch of 4D tensors.
 *          It multiplies each element of the source tensor by a corresponding element in the 'mulTensor',
 *          adds a corresponding element from the 'addTensor', and stores the result in the destination tensor.
 *          Support added for f32 -> f32 dataype.
 * \image html https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenInputs/input150x150x4.gif Sample Input
 * \image html https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenOutputs/arithmetic_operations_fused_multiply_add_scalar_150x150x4.gif Sample Output
 * \param [in] srcPtr source tensor in HOST memory
 * \param[in] srcGenericDescPtr source tensor descriptor
 * \param[out] dstPtr destination tensor in HOST memory
 * \param[in] dstGenericDescPtr destination tensor descriptor
 * \param[in] mulTensor mul values for fmadd calculation (1D tensor of batchSize Rpp32f values)
 * \param[in] addTensor add values for fmadd calculation (1D tensor of batchSize Rpp32f values)
 * \param[in] roiGenericPtrSrc ROI data for each image in source tensor (tensor of batchSize RpptRoiGeneric values)
 * \param[in] roiType ROI type used (RpptRoi3DType::XYZWHD or RpptRoi3DType::LTFRBB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_fused_multiply_add_scalar_host(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32f *mulTensor, Rpp32f *addTensor, RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Fused multiply add scalar augmentation on HIP backend
 * \details This function performs the fmadd operation on a batch of 4D tensors.
 *          It multiplies each element of the source tensor by a corresponding element in the 'mulTensor',
 *          adds a corresponding element from the 'addTensor', and stores the result in the destination tensor.
 *          Support added for f32 -> f32 dataype.
 * \image html https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenInputs/input150x150x4.gif Sample Input
 * \image html https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenOutputs/arithmetic_operations_fused_multiply_add_scalar_150x150x4.gif Sample Output
 * \param [in] srcPtr source tensor in HIP memory
 * \param[in] srcGenericDescPtr source tensor descriptor
 * \param[out] dstPtr destination tensor in HIP memory
 * \param[in] dstGenericDescPtr destination tensor descriptor
 * \param[in] mulTensor mul values for fmadd calculation (1D tensor of batchSize Rpp32f values)
 * \param[in] addTensor add values for fmadd calculation (1D tensor of batchSize Rpp32f values)
 * \param[in] roiGenericPtrSrc ROI data for each image in source tensor (tensor of batchSize RpptRoiGeneric values)
 * \param[in] roiType ROI type used (RpptRoi3DType::XYZWHD or RpptRoi3DType::LTFRBB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_fused_multiply_add_scalar_gpu(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32f *mulTensor, Rpp32f *addTensor, RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Add scalar augmentation on HOST backend
 * \details This function performs the addition operation on a batch of 4D tensors.
 *          It adds a corresponding element from the 'addTensor' to source tensor, and stores the result in the destination tensor.
 *          Support added for f32 -> f32 dataype.
 * \image html https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenInputs/input150x150x4.gif Sample Input
 * \image html https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenOutputs/arithmetic_operations_add_scalar_150x150x4.gif Sample Output
 * \param [in] srcPtr source tensor in HOST memory
 * \param[in] srcGenericDescPtr source tensor descriptor
 * \param[out] dstPtr destination tensor in HOST memory
 * \param[in] dstGenericDescPtr destination tensor descriptor
 * \param[in] addTensor add values for used for addition (1D tensor of batchSize Rpp32f values)
 * \param[in] roiGenericPtrSrc ROI data for each image in source tensor (tensor of batchSize RpptRoiGeneric values)
 * \param[in] roiType ROI type used (RpptRoi3DType::XYZWHD or RpptRoi3DType::LTFRBB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_add_scalar_host(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32f *addTensor, RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Add scalar augmentation on HIP backend
 * \details This function performs the addition operation on a batch of 4D tensors.
 *          It adds a corresponding element from the 'addTensor' to source tensor, and stores the result in the destination tensor.
 *          Support added for f32 -> f32 dataype.
 * \image html https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenInputs/input150x150x4.gif Sample Input
 * \image html https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenOutputs/arithmetic_operations_add_scalar_150x150x4.gif Sample Output
 * \param [in] srcPtr source tensor in HIP memory
 * \param[in] srcGenericDescPtr source tensor descriptor
 * \param[out] dstPtr destination tensor in HIP memory
 * \param[in] dstGenericDescPtr destination tensor descriptor
 * \param[in] addTensor add values for used for addition (1D tensor of batchSize Rpp32f values)
 * \param[in] roiGenericPtrSrc ROI data for each image in source tensor (tensor of batchSize RpptRoiGeneric values)
 * \param[in] roiType ROI type used (RpptRoi3DType::XYZWHD or RpptRoi3DType::LTFRBB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_add_scalar_gpu(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32f *addTensor, RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Subtract scalar augmentation on HOST backend
 * \details This function performs the subtraction operation on a batch of 4D tensors.
 *          It takes a corresponding element from 'subtractTensor' and subtracts it from source tensor. Result is stored in the destination tensor.
 *          Support added for f32 -> f32 dataype.
 * \image html https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenInputs/input150x150x4.gif Sample Input
 * \image html https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenOutputs/arithmetic_operations_subtract_scalar_150x150x4.gif Sample Output
 * \param [in] srcPtr source tensor in HOST memory
 * \param[in] srcGenericDescPtr source tensor descriptor
 * \param[out] dstPtr destination tensor in HOST memory
 * \param[in] dstGenericDescPtr destination tensor descriptor
 * \param[in] subtractTensor subtract values for used for subtraction (1D tensor of batchSize Rpp32f values)
 * \param[in] roiGenericPtrSrc ROI data for each image in source tensor (tensor of batchSize RpptRoiGeneric values)
 * \param[in] roiType ROI type used (RpptRoi3DType::XYZWHD or RpptRoi3DType::LTFRBB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_subtract_scalar_host(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32f *subtractTensor, RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Subtract scalar augmentation on HIP backend
 * \details This function performs the subtraction operation on a batch of 4D tensors.
 *          It takes a corresponding element from 'subtractTensor' and subtracts it from source tensor. Result is stored in the destination tensor.
 *          Support added for f32 -> f32 dataype.
 * \image html https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenInputs/input150x150x4.gif Sample Input
 * \image html https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenOutputs/arithmetic_operations_subtract_scalar_150x150x4.gif Sample Output
 * \param [in] srcPtr source tensor in HIP memory
 * \param[in] srcGenericDescPtr source tensor descriptor
 * \param[out] dstPtr destination tensor in HIP memory
 * \param[in] dstGenericDescPtr destination tensor descriptor
 * \param[in] subtractTensor subtract values for used for subtraction (1D tensor of batchSize Rpp32f values)
 * \param[in] roiGenericPtrSrc ROI data for each image in source tensor (tensor of batchSize RpptRoiGeneric values)
 * \param[in] roiType ROI type used (RpptRoi3DType::XYZWHD or RpptRoi3DType::LTFRBB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_subtract_scalar_gpu(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32f *subtractTensor, RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Multiply scalar augmentation on HOST backend
 * \details This function performs the multiplication operation on a batch of 4D tensors.
 *          It takes a corresponding element from 'multiplyTensor' and multiplies it with source tensor. Result is stored in the destination tensor.
 *          Support added for f32 -> f32 dataype.
 * \image html https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenInputs/input150x150x4.gif Sample Input
 * \image html https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenOutputs/arithmetic_operations_multiply_scalar_150x150x4.gif Sample Output
 * \param [in] srcPtr source tensor in HOST memory
 * \param[in] srcGenericDescPtr source tensor descriptor
 * \param[out] dstPtr destination tensor in HOST memory
 * \param[in] dstGenericDescPtr destination tensor descriptor
 * \param[in] mulTensor multiplier values for used for multiplication (1D tensor of batchSize Rpp32f values)
 * \param[in] roiGenericPtrSrc ROI data for each image in source tensor (tensor of batchSize RpptRoiGeneric values)
 * \param[in] roiType ROI type used (RpptRoi3DType::XYZWHD or RpptRoi3DType::LTFRBB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_multiply_scalar_host(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32f *mulTensor, RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Multiply scalar augmentation on HIP backend
 * \details This function performs the multiplication operation on a batch of 4D tensors.
 *          It takes a corresponding element from 'multiplyTensor' and multiplies it with source tensor. Result is stored in the destination tensor.
 *          Support added for f32 -> f32 dataype.
 * \image html https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenInputs/input150x150x4.gif Sample Input
 * \image html https://raw.githubusercontent.com/ROCm/rpp/develop/docs/data/doxygenOutputs/arithmetic_operations_multiply_scalar_150x150x4.gif Sample Output
 * \param [in] srcPtr source tensor in HIP memory
 * \param[in] srcGenericDescPtr source tensor descriptor
 * \param[out] dstPtr destination tensor in HIP memory
 * \param[in] dstGenericDescPtr destination tensor descriptor
 * \param[in] mulTensor multiplier values for used for multiplication (1D tensor of batchSize Rpp32f values)
 * \param[in] roiGenericPtrSrc ROI data for each image in source tensor (tensor of batchSize RpptRoiGeneric values)
 * \param[in] roiType ROI type used (RpptRoi3DType::XYZWHD or RpptRoi3DType::LTFRBB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_multiply_scalar_gpu(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32f *mulTensor, RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Magnitude computation on HOST backend for a NCHW/NHWC layout tensor
 * \details This function computes magnitude of corresponding pixels for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 *          srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 *          dstPtr depth ranges - Will be same depth as srcPtr. <br>
 * \image html img150x150.png Sample Input1
 * \image html img150x150_2.png Sample Input2
 * \image html arithmetic_operations_magnitude_img150x150.png Sample Output
 * \param [in] srcPtr1 source1 tensor in HOST memory
 * \param [in] srcPtr2 source2 tensor in HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HOST memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] roiTensorPtrSrc ROI data in HOST memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_magnitude_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Magnitude computation on HIP backend for a NCHW/NHWC layout tensor
 * \details This function computes magnitude of corresponding pixels for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 *          srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 *          dstPtr depth ranges - Will be same depth as srcPtr. <br>
 * \image html img150x150.png Sample Input1
 * \image html img150x150_2.png Sample Input2
 * \image html arithmetic_operations_magnitude_img150x150.png Sample Output
 * \param [in] srcPtr1 source1 tensor in HIP memory
 * \param [in] srcPtr2 source2 tensor in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] roiTensorPtrSrc ROI data in HIP memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_magnitude_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Logarithm operation on HOST backend
 * \details Computes Log to base e(natural log) of the input for a given ND Tensor.
 *          Supports u8->f32, i8->f32, f16->f16 and f32->f32 datatypes.
 *          Uses Absolute of input for log computation and uses nextafter() if input is 0 to avoid undefined result.
 * \param [in] srcPtr source tensor in HOST memory
 * \param [in] srcGenericDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor in HOST memory
 * \param [in] dstGenericDescPtr destination tensor descriptor
 * \param [in] roiTensor values to represent dimensions of input tensor
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_log_host(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *roiTensor, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Logarithm operation on HIP backend
 * \details Computes Log to base e(natural log) of the input for a given ND Tensor.
 *          Supports u8->f32, i8->f32, f16->f16 and f32->f32 datatypes.
 *          Uses Absolute of input for log computation and uses nextafter() if input is 0 to avoid undefined result.
 * \param [in] srcPtr source tensor in HIP memory
 * \param [in] srcGenericDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstGenericDescPtr destination tensor descriptor
 * \param [in] roiTensor values to represent dimensions of input tensor
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_log_gpu(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *roiTensor, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Log1p operation on HOST backend
 * \details Computes Log1p i.e (log(1 + x)) of the input for a given ND Tensor.
 *          Supports i16->f32 datatype.
 *          Uses Absolute of input for log1p computation to avoid undefined result.
 * \param [in] srcPtr source tensor in HOST memory
 * \param [in] srcGenericDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor in HOST memory
 * \param [in] dstGenericDescPtr destination tensor descriptor
 * \param [in] roiTensor values to represent dimensions of input tensor
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_log1p_host(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *roiTensor, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Log1p operation on HIP backend
 * \details Computes Log1p i.e (log(1 + x)) of the input for a given ND Tensor.
 *          Supports i16->f32 datatype.
 *          Uses Absolute of input for log1p computation to avoid undefined result.
 * \param [in] srcPtr source tensor in HIP memory
 * \param [in] srcGenericDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstGenericDescPtr destination tensor descriptor
 * \param [in] roiTensor values to represent dimensions of input tensor
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreate()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_log1p_gpu(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *roiTensor, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! @}
 */

#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_ARITHMETIC_OPERATIONS_H
