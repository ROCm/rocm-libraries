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

#ifndef RPPI_COMPUTER_VISION_H
#define RPPI_COMPUTER_VISION_H

#include "rppdefs.h"
#include "rpp.h"
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \file
 * \brief RPPI Image Operations - Computer Vision - To be deprecated.
 * \defgroup group_rppi_computer_vision RPPI Image Operations - Computer Vision
 * \brief RPPI Image Operations - Computer Vision - To be deprecated.
 * \deprecated
 */

/*! \addtogroup group_rppi_computer_vision
 * @{
 */

/******************** local_binary_pattern ********************/

// Performs the 8 neighbor Local Binary Pattern (LBP), where the LBP for each pixel is defined by comparing the pixel value against its 8 neighbors, for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_local_binary_pattern_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_local_binary_pattern_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_local_binary_pattern_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_local_binary_pattern_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_local_binary_pattern_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_local_binary_pattern_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** data_object_copy ********************/

// Performs a buffer copy for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_data_object_copy_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_data_object_copy_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_data_object_copy_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_data_object_copy_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_data_object_copy_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_data_object_copy_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** gaussian_image_pyramid ********************/

// Performs a gaussian image pyramid computation for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] stdDev Array containing an Rpp32f stdDev for each image in the batch (stdDev[n] >= 0)
// *param[in] kernelSize Array containing an Rpp32u kernel size for each image in the batch (kernelSize[n] = 3/5/7 for optimal use)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_gaussian_image_pyramid_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_gaussian_image_pyramid_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_gaussian_image_pyramid_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_gaussian_image_pyramid_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_gaussian_image_pyramid_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_gaussian_image_pyramid_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** laplacian_image_pyramid ********************/

// Performs a laplacian image pyramid computation for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] stdDev Array containing an Rpp32f stdDev for each image in the batch (stdDev[n] >= 0)
// *param[in] kernelSize Array containing an Rpp32u kernel size for each image in the batch (kernelSize[n] = 3/5/7 for optimal use)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_laplacian_image_pyramid_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_laplacian_image_pyramid_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_laplacian_image_pyramid_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_laplacian_image_pyramid_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_laplacian_image_pyramid_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_laplacian_image_pyramid_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** canny_edge_detector ********************/

// Performs a canny edge detection and outputs edge-images for a batch of images

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

RppStatus rppi_canny_edge_detector_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *minThreshold, Rpp8u *maxThreshold, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_canny_edge_detector_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *minThreshold, Rpp8u *maxThreshold, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_canny_edge_detector_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *minThreshold, Rpp8u *maxThreshold, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_canny_edge_detector_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *minThreshold, Rpp8u *maxThreshold, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_canny_edge_detector_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *minThreshold, Rpp8u *maxThreshold, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_canny_edge_detector_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *minThreshold, Rpp8u *maxThreshold, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** harris_corner_detector ********************/

// Performs a harris corner detection and outputs corner-overlayed-images for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] gaussianKernelSize Array containing an Rpp32u gaussian kernel size for each image in the batch (gaussianKernelSize[n] = 3/5/7 for optimal use)
// *param[in] stdDev Array containing an Rpp32f standard deviation for each image in the batch (stdDev >= 0)
// *param[in] kernelSize Array containing an Rpp32u corner detection kernel size for each image in the batch (kernelSize[n] = 3/5/7 for optimal use)
// *param[in] kValue Array containing an appropriate Rpp32f k value for each image in the batch
// *param[in] threshold Array containing an appropriate Rpp32f threshold for each image in the batch
// *param[in] nonmaxKernelSize Array containing an Rpp32u nonmax suppression kernel size for each image in the batch (nonmaxKernelSize[n] = 3/5/7 for optimal use)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_harris_corner_detector_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *gaussianKernelSize, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32f *kValue, Rpp32f *threshold, Rpp32u *nonmaxKernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_harris_corner_detector_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *gaussianKernelSize, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32f *kValue, Rpp32f *threshold, Rpp32u *nonmaxKernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_harris_corner_detector_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *gaussianKernelSize, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32f *kValue, Rpp32f *threshold, Rpp32u *nonmaxKernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_harris_corner_detector_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *gaussianKernelSize, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32f *kValue, Rpp32f *threshold, Rpp32u *nonmaxKernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_harris_corner_detector_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *gaussianKernelSize, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32f *kValue, Rpp32f *threshold, Rpp32u *nonmaxKernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_harris_corner_detector_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *gaussianKernelSize, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32f *kValue, Rpp32f *threshold, Rpp32u *nonmaxKernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** tensor_convert_bit_depth ********************/

// Converts bit depth of the elements of a tensor

// *param[in] srcPtr Input tensor
// *param[out] dstPtr Output tensor
// *param[in] tensorDimension Number of dimensions in the tensor
// *param[in] tensorDimensionValues Array of length - "tensorDimension", containing size of each dimension in the tensor
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_tensor_convert_bit_depth_u8s8_host(RppPtr_t srcPtr, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues);
RppStatus rppi_tensor_convert_bit_depth_u8u16_host(RppPtr_t srcPtr, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues);
RppStatus rppi_tensor_convert_bit_depth_u8s16_host(RppPtr_t srcPtr, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues);

/******************** fast_corner_detector ********************/

// Performs a fast corner detection and outputs corner-overlayed-images for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] numOfPixels Array containing an Rpp32u minimum number of contiguous pixel to detect a corner, for each image in the batch (numOfPixels[n] >= 0)
// *param[in] threshold Array containing an appropriate Rpp8u intensity-difference threshold for corners for each image in the batch (0 <= threshold[n] <= 255)
// *param[in] nonmaxKernelSize Array containing an Rpp32u nonmax suppression kernel size for each image in the batch (nonmaxKernelSize[n] = 3/5/7/9/11/15 for optimal use)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_fast_corner_detector_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *numOfPixels, Rpp8u *threshold, Rpp32u *nonmaxKernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_fast_corner_detector_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *numOfPixels, Rpp8u *threshold, Rpp32u *nonmaxKernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_fast_corner_detector_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *numOfPixels, Rpp8u *threshold, Rpp32u *nonmaxKernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** reconstruction_laplacian_image_pyramid ********************/

// Performs a reconstruction of original image from its laplacian image pyramid for a batch of images

// *param[in] srcPtr1 Input image1
// *param[in] srcSize1 Array containing an RppiSize for each image in the image1 batch
// *param[in] maxSrcSize1 A single RppiSize which is the maxWidth and maxHeight for all images in the image1 batch
// *param[in] srcPtr2 Input image2
// *param[in] srcSize1 Array containing an RppiSize for each image in the image2 batch
// *param[in] maxSrcSize1 A single RppiSize which is the maxWidth and maxHeight for all images in the image2 batch
// *param[out] dstPtr Output image batch
// *param[in] stdDev Array containing an Rpp32f standard deviation for each image in the batch (stdDev[n] >= 0)
// *param[in] kerenelSize Array containing an Rpp32u kernel size for each image in the batch (kernelSize[n] = 3/5/7 for optimal use)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_reconstruction_laplacian_image_pyramid_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppiSize *srcSize1, RppiSize maxSrcSize1, RppPtr_t srcPtr2, RppiSize *srcSize2, RppiSize maxSrcSize2, RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_reconstruction_laplacian_image_pyramid_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppiSize *srcSize1, RppiSize maxSrcSize1, RppPtr_t srcPtr2, RppiSize *srcSize2, RppiSize maxSrcSize2, RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_reconstruction_laplacian_image_pyramid_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppiSize *srcSize1, RppiSize maxSrcSize1, RppPtr_t srcPtr2, RppiSize *srcSize2, RppiSize maxSrcSize2, RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** control_flow ********************/

RppStatus rpp_bool_control_flow(bool num1, bool num2, bool *output, RppOp operation, rppHandle_t rppHandle);
RppStatus rpp_u8_control_flow(Rpp8u num1, Rpp8u num2, Rpp8u *output, RppOp operation, rppHandle_t rppHandle);

/******************** hough_lines ********************/

// Runs the hough lines algorithm using the progressive probabilistic Hough Transform to find lines in a batch of images (the inputs must be single channel outputs from a canny edge detector)

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] lines Output line coordinate in the form [x1_start, y1_start, x1_end, y1_end, x2_start, y2_start, x2_end, y2_end .....]
// *param[in] rho Array containing an Rpp32f rho for each image in the batch (Distance resolution of the parameter in pixels)
// *param[in] theta Array containing an Rpp32f theta for each image in the batch (Angle resolution of the parameter in radians)
// *param[in] threshold Array containing an Rpp32u threshold for each image in the batch (The minimum number of intersections to detect a line)
// *param[in] minLineLength Array containing an Rpp32u minimum line length for each image in the batch (The minimum number of points that can form a line. Line segments shorter than that are rejected)
// *param[in] maxLineGap Array containing an Rpp32u maximum line gap for each image in the batch (The maximum allowed gap between points on the same line to link them)
// *param[in] linesMax Array containing an Rpp32u maximum number of detected lines for each image in the batch
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_hough_lines_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t lines, Rpp32f* rho, Rpp32f* theta, Rpp32u *threshold, Rpp32u *minLineLength, Rpp32u *maxLineGap, Rpp32u *linesMax, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** hog ********************/

// Finds the histogram of oriented gradients for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] binsTensor Output HOG bins tensor
// *param[in] binsTensorLength Array containing an Rpp32u length value of the HOG bins tensor for each image in the batch
// *param[in] kernelSize Array containing an RppiSize kernel width/height size pair for each image in the batch
// *param[in] windowSize Array containing an RppiSize window width/height size pair for each image in the batch (windowSize[n] must be even muliple of kernelSize[n])
// *param[in] windowStride Array containing an Rpp32u window stride for each image in the batch
// *param[in] numOfBins Array containing an Rpp32u number of HOG bins for each image in the batch
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_hog_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t binsTensor, Rpp32u *binsTensorLength, RppiSize *kernelSize, RppiSize *windowSize, Rpp32u *windowStride, Rpp32u *numOfBins, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** remap ********************/

// Performs a remap operation using user specified remap tables for a batch of images. For each image, the output(x,y) = input(mapx(x, y), mapy(x, y)) for every (x,y) in the destination image

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] rowRemapTable Array of Rpp32u row numbers for every pixel in the input batch of images
// *param[in] colRemapTable Array of Rpp32u column numbers for every pixel in the input batch of images
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_remap_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *rowRemapTable, Rpp32u *colRemapTable, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_remap_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *rowRemapTable, Rpp32u *colRemapTable, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_remap_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *rowRemapTable, Rpp32u *colRemapTable, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_remap_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *rowRemapTable, Rpp32u *colRemapTable, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_remap_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *rowRemapTable, Rpp32u *colRemapTable, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_remap_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *rowRemapTable, Rpp32u *colRemapTable, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** tensor_matrix_multiply ********************/

// Performs matrix multiplication on two tensors

// *param[in] srcPtr1 Input tensor1
// *param[in] srcPtr2 Input tensor2
// *param[out] dstPtr Output tensor
// *param[in] tensorDimensionValues1 Array containing dimensions of tensor1
// *param[in] tensorDimensionValues2 Array containing dimensions of tensor2
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_tensor_matrix_multiply_u8_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, RppPtr_t tensorDimensionValues1, RppPtr_t tensorDimensionValues2, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_tensor_matrix_multiply_u8_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, RppPtr_t tensorDimensionValues1, RppPtr_t tensorDimensionValues2, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** tensor_transpose ********************/

// Performs a transpose of the input tensor based on the current shape and the permutation desired

// *param[in] srcPtr Input tensor
// *param[out] dstPtr Output tensor
// *param[in] shape Current shape of tensor
// *param[in] perm Permutation of dimensions desired
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_tensor_transpose_u8_host(RppPtr_t srcPtr, RppPtr_t dstPtr, Rpp32u *shape, Rpp32u *perm, rppHandle_t rppHandle);
RppStatus rppi_tensor_transpose_f16_host(RppPtr_t srcPtr, RppPtr_t dstPtr, Rpp32u *shape, Rpp32u *perm, rppHandle_t rppHandle);
RppStatus rppi_tensor_transpose_f32_host(RppPtr_t srcPtr, RppPtr_t dstPtr, Rpp32u *shape, Rpp32u *perm, rppHandle_t rppHandle);
RppStatus rppi_tensor_transpose_i8_host(RppPtr_t srcPtr, RppPtr_t dstPtr, Rpp32u *shape, Rpp32u *perm, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_tensor_transpose_u8_gpu(RppPtr_t srcPtr, RppPtr_t dstPtr, RppPtr_t shape, RppPtr_t perm, rppHandle_t rppHandle);
RppStatus rppi_tensor_transpose_f16_gpu(RppPtr_t srcPtr, RppPtr_t dstPtr, RppPtr_t shape, RppPtr_t perm, rppHandle_t rppHandle);
RppStatus rppi_tensor_transpose_f32_gpu(RppPtr_t srcPtr, RppPtr_t dstPtr, RppPtr_t shape, RppPtr_t perm, rppHandle_t rppHandle);
RppStatus rppi_tensor_transpose_i8_gpu(RppPtr_t srcPtr, RppPtr_t dstPtr, RppPtr_t shape, RppPtr_t perm, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** convert_bit_depth ********************/

// Converts bit depth of the elements of a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_convert_bit_depth_u8s8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_convert_bit_depth_u8u16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_convert_bit_depth_u8s16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_convert_bit_depth_u8s8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_convert_bit_depth_u8u16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_convert_bit_depth_u8s16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_convert_bit_depth_u8s8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_convert_bit_depth_u8u16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_convert_bit_depth_u8s16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_convert_bit_depth_u8s8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_convert_bit_depth_u8u16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_convert_bit_depth_u8s16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_convert_bit_depth_u8s8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_convert_bit_depth_u8u16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_convert_bit_depth_u8s16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_convert_bit_depth_u8s8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_convert_bit_depth_u8u16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_convert_bit_depth_u8s16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! @}
 */

#ifdef __cplusplus
}
#endif

#endif