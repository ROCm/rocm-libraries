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

#ifndef RPPI_IMAGE_AUGMENTATIONS_H
#define RPPI_IMAGE_AUGMENTATIONS_H

#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \file
 * \brief RPPI Image Operations - Image Augmentations - To be deprecated.
 * \defgroup group_rppi_image_augmentations RPPI Image Operations - Image Augmentations
 * \brief RPPI Image Operations - Image Augmentations - To be deprecated.
 * \deprecated
 */

/*! \addtogroup group_rppi_image_augmentations
 * @{
 */

/******************** brightness ********************/

// Adjusts brightness for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] alpha Array containing an Rpp32f alpha for each image in the batch (0 <= alpha <= 20 for brightness calculation)
// *param[in] beta Array containing an Rpp32f beta for each image in the batch (0 <= beta <= 255 for brightness calculation)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_brightness_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_brightness_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_brightness_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_brightness_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_brightness_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_brightness_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** gamma_correction ********************/

// Gamma correction for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] gamma Array containing an Rpp32f gamma for each image in the batch (gamma >= 0 for gamma correction)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_gamma_correction_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *gamma, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_gamma_correction_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *gamma, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_gamma_correction_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *gamma, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_gamma_correction_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *gamma, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_gamma_correction_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *gamma, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_gamma_correction_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *gamma, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** blend ********************/

// Alpha blends corresponding pixels between two batches of images

// *param[in] srcPtr1 Input image1 batch
// *param[in] srcPtr2 Input image2 batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] alpha Array containing an Rpp32f alpha for each image in the batch (transparency factor 0 <= alpha <= 1 for alpha-blending)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_blend_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_blend_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_blend_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_blend_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_blend_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_blend_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** blur ********************/

// Blurs a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] kernelSize Array containing an Rpp32u kernel size for each image in the batch (kernelSize = 3/5/7 for optimal use)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_blur_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_blur_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_blur_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_blur_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_blur_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_blur_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** contrast ********************/

// Adjusts contrast for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] newMin Array containing an Rpp32u new minimum pixel-value for each image in the batch (0 <= newMin <= 255 for contrast calculation)
// *param[in] newMax Array containing an Rpp32u new maximum pixel-value for each image in the batch (0 <= newMax <= 255 for contrast calculation)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_contrast_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *newMin, Rpp32u *newMax, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_contrast_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *newMin, Rpp32u *newMax, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_contrast_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *newMin, Rpp32u *newMax, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_contrast_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *newMin, Rpp32u *newMax, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_contrast_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *newMin, Rpp32u *newMax, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_contrast_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *newMin, Rpp32u *newMax, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** pixelate ********************/

// Pixelates a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_pixelate_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_pixelate_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_pixelate_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_pixelate_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_pixelate_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_pixelate_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** jitter ********************/

// Jitters a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] kernelSize Array containing an Rpp32u kernel size for each image in the batch (kernelSize = 3/5/7 for optimal use)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_jitter_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_jitter_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_jitter_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_jitter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kenelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_jitter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kenelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_jitter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kenelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** snow ********************/

// Adds a snowfall overlay for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] snowValue Array containing an Rpp32f snow-value for each image in the batch (0 <= snowValue <= 1)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_snow_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *snowValue, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_snow_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *snowValue, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_snow_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *snowValue, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_snow_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *snowValue, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_snow_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *snowValue, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_snow_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *snowValue, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** noise ********************/

// Adds a noise overlay for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] noiseProbability Array containing an Rpp32f noise-probability for each image in the batch (0 <= noiseProbability <= 1)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_noise_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *noiseProbability, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_noise_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *noiseProbability, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_noise_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *noiseProbability, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_noise_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *noiseProbability, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_noise_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *noiseProbability, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_noise_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *noiseProbability, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** random_shadow ********************/

// Adds random shadows for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] x1 Array containing an Rpp32u x1 for each image in the batch (x1 specifies Left-Top X coordinate for the subregion in which shadows will be created)
// *param[in] y1 Array containing an Rpp32u y1 for each image in the batch (y1 specifies Left-Top Y coordinate for the subregion in which shadows will be created)
// *param[in] x2 Array containing an Rpp32u x2 for each image in the batch (x1 specifies Right-Bottom X coordinate for the subregion in which shadows will be created)
// *param[in] y2 Array containing an Rpp32u y2 for each image in the batch (x1 specifies Right-Bottom Y coordinate for the subregion in which shadows will be created)
// *param[in] numberOfShadows Array containing an Rpp32u number-of-shadows for each image in the batch (numberOfShadows[n] >= 0)
// *param[in] maxSizeX Array containing an Rpp32u max-shadow-width for each image in the batch (None of the created shadows will exceed the maxSizeX[n] width for each image in the batch)
// *param[in] maxSizeY Array containing an Rpp32u max-shadow-height for each image in the batch (None of the created shadows will exceed the maxSizeY[n] height for each image in the batch)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_random_shadow_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x1, Rpp32u *y1, Rpp32u *x2, Rpp32u *y2, Rpp32u *numberOfShadows, Rpp32u *maxSizeX, Rpp32u *maxSizeY, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_random_shadow_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x1, Rpp32u *y1, Rpp32u *x2, Rpp32u *y2, Rpp32u *numberOfShadows, Rpp32u *maxSizeX, Rpp32u *maxSizeY, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_random_shadow_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x1, Rpp32u *y1, Rpp32u *x2, Rpp32u *y2, Rpp32u *numberOfShadows, Rpp32u *maxSizeX, Rpp32u *maxSizeY, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_random_shadow_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x1, Rpp32u *y1, Rpp32u *x2, Rpp32u *y2, Rpp32u *numberOfShadows, Rpp32u *maxSizeX, Rpp32u *maxSizeY, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_random_shadow_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x1, Rpp32u *y1, Rpp32u *x2, Rpp32u *y2, Rpp32u *numberOfShadows, Rpp32u *maxSizeX, Rpp32u *maxSizeY, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_random_shadow_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x1, Rpp32u *y1, Rpp32u *x2, Rpp32u *y2, Rpp32u *numberOfShadows, Rpp32u *maxSizeX, Rpp32u *maxSizeY, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** fog ********************/

// Adds a fog overlay for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] fogValue Array containing an Rpp32f fog-value for each image in the batch (0 <= fogValue[n] <= 1)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_fog_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *fogValue, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_fog_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *fogValue, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_fog_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *fogValue, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_fog_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *fogValue, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_fog_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *fogValue, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_fog_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *fogValue, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** rain ********************/

// Adds a rainfall overlay for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] rainPercentage Array containing an Rpp32f rain percentage for each image in the batch (0 <= rainPercentage[n] <= 1)
// *param[in] rainWidth Array containing an Rpp32u rain width for each image in the batch (rainWidth[n] >= 0)
// *param[in] rainHeight Array containing an Rpp32u rain height for each image in the batch (rainHeight[n] >= 0)
// *param[in] rainTransparency Array containing an Rpp32f rain transparency for each image in the batch (0 <= rainTransparency[n] <= 1)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_rain_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *rainPercentage, Rpp32u *rainWidth, Rpp32u *rainHeight, Rpp32f *transperancy, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rain_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *rainPercentage, Rpp32u *rainWidth, Rpp32u *rainHeight, Rpp32f *transperancy, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rain_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *rainPercentage, Rpp32u *rainWidth, Rpp32u *rainHeight, Rpp32f *transperancy, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_rain_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *rainPercentage, Rpp32u *rainWidth, Rpp32u *rainHeight, Rpp32f *transperancy, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rain_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *rainPercentage, Rpp32u *rainWidth, Rpp32u *rainHeight, Rpp32f *transperancy, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_rain_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *rainPercentage, Rpp32u *rainWidth, Rpp32u *rainHeight, Rpp32f *transperancy, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** random_crop_letterbox ********************/

// Crops the ROI and adds a letterbox border for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] dstSize Array containing an RppiSize for each image in the batch
// *param[in] maxDstSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[in] xRoiBegin Array containing an x1 (ROI Top-Left) for each image in the batch
// *param[in] xRoiEnd Array containing an x2 (ROI Right-Bottom) for each image in the batch
// *param[in] yRoiBegin Array containing an y1 (ROI Top-Left) for each image in the batch
// *param[in] yRoiEnd Array containing an y2 (ROI Right-Bottom) for each image in the batch
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_random_crop_letterbox_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_random_crop_letterbox_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_random_crop_letterbox_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_random_crop_letterbox_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_random_crop_letterbox_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_random_crop_letterbox_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** exposure ********************/

// Adjusts exposure for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] exposureValue Array containing an Rpp32f exposure factor for each image in the batch (exposureValue[n] >= 0)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_exposure_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *exposureValue, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_exposure_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *exposureValue, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_exposure_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *exposureValue, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_exposure_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *exposureValue, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_exposure_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *exposureValue, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_exposure_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *exposureValue, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** histogram_balance ********************/

// Performs histogram balancee for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_histogram_balance_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_histogram_balance_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_histogram_balance_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_histogram_balance_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_histogram_balance_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_histogram_balance_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! @}
 */

#ifdef __cplusplus
}
#endif
#endif
