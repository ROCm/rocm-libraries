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

#ifndef RPPI_ARITHMATIC_OPERATIONS_H
#define RPPI_ARITHMATIC_OPERATIONS_H

#include "rppdefs.h"
#include "rpp.h"
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \file
 * \brief RPPI Image Operations - Arithmetic Operations - To be deprecated.
 * \defgroup group_rppi_arithmetic_operations RPPI Image Operations - Arithmetic Operations
 * \brief RPPI Image Operations - Arithmetic Operations - To be deprecated.
 * \deprecated
 */

/*! \addtogroup group_rppi_arithmetic_operations
 * @{
 */

/******************** add ********************/

// Adds corresponding pixels of two batches of images

// *param[in] srcPtr1 Input image1 batch
// *param[in] srcPtr2 Input image2 batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_add_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_add_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_add_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_add_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_add_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_add_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** subtract ********************/

// Subtracts corresponding pixels of two batches of images

// *param[in] srcPtr1 Input image1 batch
// *param[in] srcPtr2 Input image2 batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_subtract_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_subtract_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_subtract_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_subtract_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_subtract_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_subtract_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** multiply ********************/

// Multiplies corresponding pixels of two batches of images

// *param[in] srcPtr1 Input image1 batch
// *param[in] srcPtr2 Input image2 batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_multiply_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_multiply_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_multiply_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_multiply_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_multiply_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_multiply_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** absolute_difference ********************/

// Computes absolute differnece of corresponding pixels of two batches of images

// *param[in] srcPtr1 Input image1 batch
// *param[in] srcPtr2 Input image2 batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_absolute_difference_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_absolute_difference_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_absolute_difference_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_absolute_difference_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_absolute_difference_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_absolute_difference_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** phase ********************/

// Computes phase of corresponding pixels of two batches of images

// *param[in] srcPtr1 Input image1 batch
// *param[in] srcPtr2 Input image2 batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_phase_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_phase_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_phase_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_phase_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_phase_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_phase_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** magnitude ********************/

// Computes magnitude of corresponding pixels of two batches of images

// *param[in] srcPtr1 Input image1 batch
// *param[in] srcPtr2 Input image2 batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_magnitude_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_magnitude_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_magnitude_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_magnitude_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_magnitude_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_magnitude_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** accumulate ********************/

// Accumulates corresponding pixels of two batches of images, into the first batch

// *param[in/out] srcPtr1 Input image1 batch
// *param[in] srcPtr2 Input image2 batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_accumulate_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_accumulate_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** accumulate_weighted ********************/

// Alpha-blends corresponding pixels of two batches of images, into the first batch

// *param[in/out] srcPtr1 Input image1 batch
// *param[in] srcPtr2 Input image2 batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[in] alpha Array of floats containing one alpha value for each image in the batch
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_accumulate_weighted_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32f *alpha, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_weighted_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32f *alpha, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_weighted_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32f *alpha, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_accumulate_weighted_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32f *alpha, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_weighted_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32f *alpha, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_weighted_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32f *alpha, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** accumulate_squared ********************/

// Squares pixel values of a batch of images in-place

// *param[in/out] srcPtr Input image
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_accumulate_squared_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_squared_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_squared_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_accumulate_squared_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_squared_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_squared_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** tensor_add ********************/

// Adds corresponding elements of two tensors

// *param[in] srcPtr1 Input tensor1
// *param[in] srcPtr2 Input tensor2
// *param[out] dstPtr Output tensor
// *param[in] tensorDimension Number of dimensions in the tensor
// *param[in] tensorDimensionValues Array of length - "tensorDimension", containing size of each dimension in the tensor
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_tensor_add_u8_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_tensor_add_u8_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** tensor_subtract ********************/

// Subtracts corresponding elements of two tensors

// *param[in] srcPtr1 Input tensor1
// *param[in] srcPtr2 Input tensor2
// *param[out] dstPtr Output tensor
// *param[in] tensorDimension Number of dimensions in the tensor
// *param[in] tensorDimensionValues Array of length - "tensorDimension", containing size of each dimension in the tensor
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_tensor_subtract_u8_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_tensor_subtract_u8_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** tensor_multiply ********************/

// Multiplies corresponding elements of two tensors

// *param[in] srcPtr1 Input tensor1
// *param[in] srcPtr2 Input tensor2
// *param[out] dstPtr Output tensor
// *param[in] tensorDimension Number of dimensions in the tensor
// *param[in] tensorDimensionValues Array of length - "tensorDimension", containing size of each dimension in the tensor
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_tensor_multiply_u8_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_tensor_multiply_u8_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! @}
 */

#ifdef __cplusplus
}
#endif

#endif
