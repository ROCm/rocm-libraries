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

#ifndef RPPI_MORPHOLOGICAL_TRANSFORMS_H
#define RPPI_MORPHOLOGICAL_TRANSFORMS_H

#include "rppdefs.h"
#include "rpp.h"
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \file
 * \brief RPPI Image Operations - Morphological Transforms - To be deprecated.
 * \defgroup group_rppi_morphological_transforms RPPI Image Operations - Morphological Transforms
 * \brief RPPI Image Operations - Morphological Transforms - To be deprecated.
 * \deprecated
 */

/*! \addtogroup group_rppi_morphological_transforms
 * @{
 */

/******************** erode ********************/

// Performs an erode operation, where the output pixel is computed as the minimum value of the pixels under a [kernelSize X kernelSize] square mask, for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] kernelSize Array containing an Rpp32u kernel size for each image in the batch (kernelSize[n] = 3/5/7 for optimal use)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_erode_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erode_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erode_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_erode_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erode_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erode_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** dilate ********************/

// Performs a dilate operation, where the output pixel is computed as the maximum value of the pixels under a [kernelSize X kernelSize] square mask, for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] kernelSize Array containing an Rpp32u kernel size for each image in the batch (kernelSize[n] = 3/5/7 for optimal use)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_dilate_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_dilate_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_dilate_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_dilate_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_dilate_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_dilate_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! @}
 */

#ifdef __cplusplus
}
#endif
#endif