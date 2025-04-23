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

#ifndef RPPI_GEOMETRY_TRANSFORMS_H
#define RPPI_GEOMETRY_TRANSFORMS_H

#include "rppdefs.h"
#include "rpp.h"
#ifdef __cplusplus
extern "C"
{
#endif // cpusplus

/*!
 * \file
 * \brief RPPI Image Operations - Geometry Transforms - To be deprecated.
 * \defgroup group_rppi_geometry_transforms RPPI Image Operations - Geometry Transforms
 * \brief RPPI Image Operations - Geometry Transforms - To be deprecated.
 * \deprecated
 */

/*! \addtogroup group_rppi_geometry_transforms
 * @{
 */

/******************** flip ********************/

// Performs a flip transformation for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] flipAxis Array containing an Rpp32u flipping axis for each image in the batch (flipAxis[n] = 0/1/2 for flip-horizontal/flip-vertical/both)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_flip_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_flip_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_flip_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_flip_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_flip_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_flip_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** resize ********************/

// Performs a resize transformation for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the input batch
// *param[out] dstPtr Output image batch
// *param[in] dstSize Array containing an RppiSize for each image in the batch
// *param[in] maxDstSize A single RppiSize which is the maxWidth and maxHeight for all images in the output batch
// *param[in] outputFormatToggle An Rpp32u flag to set layout toggling on/off for each image in the batch (outputFormatToggle = 0/1)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_resize_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_u8_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_u8_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_u8_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_u8_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_u8_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_u8_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_u8_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_u8_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_u8_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_resize_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_u8_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_u8_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_u8_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_u8_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_u8_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_u8_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_u8_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_u8_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_u8_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** resize_crop ********************/

// Performs a crop followed by resize transformation for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the input batch
// *param[out] dstPtr Output image batch
// *param[in] dstSize Array containing an RppiSize for each image in the batch
// *param[in] maxDstSize A single RppiSize which is the maxWidth and maxHeight for all images in the output batch
// *param[in] xRoiBegin Array containing an Rpp32u crop start x-position for each image in the batch (0 <= xRoiBegin[n] < srcSize[n].width)
// *param[in] xRoiEnd Array containing an Rpp32u crop end x-position for each image in the batch (0 <= xRoiEnd[n] < srcSize[n].width)
// *param[in] yRoiBegin Array containing an Rpp32u crop start y-position for each image in the batch (0 <= yRoiBegin[n] < srcSize[n].height)
// *param[in] yRoiEnd Array containing an Rpp32u crop end y-position for each image in the batch (0 <= yRoiEnd[n] < srcSize[n].height)
// *param[in] outputFormatToggle An Rpp32u flag to set layout toggling on/off for each image in the batch (outputFormatToggle = 0/1)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_resize_crop_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_resize_crop_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputChannelToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputChnnelToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputChannelToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** rotate ********************/

// Performs a rotate transformation for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the input batch
// *param[out] dstPtr Output image batch
// *param[in] dstSize Array containing an RppiSize for each image in the batch
// *param[in] maxDstSize A single RppiSize which is the maxWidth and maxHeight for all images in the output batch
// *param[in] angleDeg Array containing an Rpp32f rotation-angle in degrees for each image in the batch
// *param[in] outputFormatToggle An Rpp32u flag to set layout toggling on/off for each image in the batch (outputFormatToggle = 0/1)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_rotate_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rotate_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rotate_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rotate_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rotate_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rotate_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rotate_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rotate_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rotate_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rotate_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rotate_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rotate_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_rotate_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rotate_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputForamtToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rotate_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputForamtToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rotate_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputForamtToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rotate_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rotate_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputForamtToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rotate_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputForamtToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rotate_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputForamtToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rotate_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rotate_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputForamtToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rotate_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputForamtToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rotate_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputForamtToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** warp_affine ********************/

// Performs a warp affine transformation for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the input batch
// *param[out] dstPtr Output image batch
// *param[in] dstSize Array containing an RppiSize for each image in the batch
// *param[in] maxDstSize A single RppiSize which is the maxWidth and maxHeight for all images in the output batch
// *param[in] affineMatrix Array containing a set of 6 Rpp32f values for each image in the batch. The 6 values define the affine-transformation matrix for a particular image in the batch.
// *param[in] outputFormatToggle An Rpp32u flag to set layout toggling on/off for each image in the batch (outputFormatToggle = 0/1)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_warp_affine_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_affine_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_affine_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_affine_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_affine_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_affine_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_affine_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_affine_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_affine_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_affine_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_affine_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_affine_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_warp_affine_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_affine_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_affine_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_affine_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_affine_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_affine_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_affine_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_affine_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_affine_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_affine_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_affine_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_affine_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** fisheye ********************/

// Performs a fisheye transformation for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_fisheye_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_fisheye_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_fisheye_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_fisheye_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_fisheye_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_fisheye_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** lens_correction ********************/

// Performs a lens correction transformation to correct barrel distorsions for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] strength Array containing an Rpp32f strength value for each image in the batch. (strength[n] >= 0)
// *param[in] zoom Array containing an Rpp32f zoom value for each image in the batch. (zoom[n] >= 1)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_lens_correction_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_lens_correction_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_lens_correction_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_lens_correction_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_lens_correction_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_lens_correction_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** scale ********************/

// Performs a scale transformation for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] dstSize Array containing an RppiSize for each image in the batch
// *param[in] maxDstSize A single RppiSize which is the maxWidth and maxHeight for all images in the output batch
// *param[in] percentage Array containing an Rpp32f scaling-percentage value for each image in the batch. (percentage[n] >= 0)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_scale_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_scale_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_scale_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_scale_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_scale_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_scale_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** warp_perspective ********************/

// Performs a warp perspective transformation for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] dstSize Array containing an RppiSize for each image in the batch
// *param[in] maxDstSize A single RppiSize which is the maxWidth and maxHeight for all images in the output batch
// *param[in] perspectiveMatrix Array containing a set of 9 Rpp32f values for each image in the batch. The 9 values define the perspective-transformation matrix for a particular image in the batch.
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_warp_perspective_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_perspective_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_perspective_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_warp_perspective_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_perspective_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_warp_perspective_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! @}
 */

#ifdef __cplusplus
}
#endif

#endif
