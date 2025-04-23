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

#ifndef RPPI_STATISTICAL_OPERATIONS_H
#define RPPI_STATISTICAL_OPERATIONS_H

#include "rppdefs.h"
#include "rpp.h"
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \file
 * \brief RPPI Image Operations - Statistical Operations - To be deprecated.
 * \defgroup group_rppi_statistical_operations RPPI Image Operations - Statistical Operations
 * \brief RPPI Image Operations - Statistical Operations - To be deprecated.
 * \deprecated
 */

/*! \addtogroup group_rppi_statistical_operations
 * @{
 */

/******************** thresholding ********************/

// Thresholds the input element-wise, and produces boolean outputs for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] minThreshold Array containing an Rpp8u minimum threshold for every pixel in each image in the batch (0 <= minThreshold <= 255)
// *param[in] maxThreshold Array containing an Rpp8u maximum threshold for every pixel in each image in the batch (0 <= maxThreshold <= 255)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_thresholding_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *minThreshold, Rpp8u *maxThreshold, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_thresholding_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *minThreshold, Rpp8u *maxThreshold, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_thresholding_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *minThreshold, Rpp8u *maxThreshold, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_thresholding_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *minThreshold, Rpp8u *maxThreshold, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_thresholding_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *minThreshold, Rpp8u *maxThreshold, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_thresholding_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *minThreshold, Rpp8u *maxThreshold, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** min ********************/

// Computes minimum of corresponding pixels of two batches of images

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

RppStatus rppi_min_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_min_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_min_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_min_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_min_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_min_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** max ********************/

// Computes maximum of corresponding pixels of two batches of images

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

RppStatus rppi_max_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_max_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_max_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_max_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_max_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_max_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** min_max_loc ********************/

// Computes minimum, maximum pixel values, and their locations in an image

// *param[in] srcPtr Input image
// *param[in] srcSize An RppiSize containing the width and height of the image
// *param[out] min Returning a Rpp8u min computed for the image
// *param[out] max Returning a Rpp8u max computed for the image
// *param[out] minLoc Returning a Rpp32u index of the min value for the image
// *param[out] maxLoc Returning a Rpp32u index of the max value for the image
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_min_max_loc_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp8u *min, Rpp8u *max, Rpp32u *minLoc, Rpp32u *maxLoc, rppHandle_t rppHandle);
RppStatus rppi_min_max_loc_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp8u *min, Rpp8u *max, Rpp32u *minLoc, Rpp32u *maxLoc, rppHandle_t rppHandle);
RppStatus rppi_min_max_loc_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp8u *min, Rpp8u *max, Rpp32u *minLoc, Rpp32u *maxLoc, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_min_max_loc_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, Rpp8u *min, Rpp8u *max, Rpp32u *minLoc, Rpp32u *maxLoc, rppHandle_t rppHandle);
RppStatus rppi_min_max_loc_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, Rpp8u *min, Rpp8u *max, Rpp32u *minLoc, Rpp32u *maxLoc, rppHandle_t rppHandle);
RppStatus rppi_min_max_loc_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, Rpp8u *min, Rpp8u *max, Rpp32u *minLoc, Rpp32u *maxLoc, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** integral ********************/

// Performs integral-image for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_integral_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_integral_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_integral_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_integral_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_integral_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_integral_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** histogram_equalization ********************/

// Performs histogram equalization of an image for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_histogram_equalization_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_histogram_equalization_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_histogram_equalization_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_histogram_equalization_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_histogram_equalization_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_histogram_equalization_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** mean_stddev ********************/

// Computes the mean and standard deviation of an image

// *param[in] srcPtr Input image
// *param[in] srcSize An RppiSize containing the width and height of the image
// *param[out] mean Returning a Rpp32f mean computed for the image
// *param[out] stdDev Returning a Rpp32f stdDev computed for the image
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_mean_stddev_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32f *mean, Rpp32f *stddev, rppHandle_t rppHandle);
RppStatus rppi_mean_stddev_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32f *mean, Rpp32f *stddev, rppHandle_t rppHandle);
RppStatus rppi_mean_stddev_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32f *mean, Rpp32f *stddev, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_mean_stddev_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, Rpp32f *mean, Rpp32f *stddev, rppHandle_t rppHandle);
RppStatus rppi_mean_stddev_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, Rpp32f *mean, Rpp32f *stddev, rppHandle_t rppHandle);
RppStatus rppi_mean_stddev_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, Rpp32f *mean, Rpp32f *stddev, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** histogram ********************/

// Computes the histogram of an image

// *param[in] srcPtr Input image
// *param[in] srcSize An RppiSize containing the width and height of the image
// *param[out] outputHistogram Array returning the histogram of the image
// *param[in] bins Number of bins for the histogram
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_histogram_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins, rppHandle_t rppHandle);
RppStatus rppi_histogram_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins, rppHandle_t rppHandle);
RppStatus rppi_histogram_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins, rppHandle_t rppHandle);

/*! @}
 */

#ifdef __cplusplus
}
#endif
#endif